#!/usr/bin/env python3
"""Picker interactivo de counting zone + counting-line para los devices del People Counter.

Captura un frame (en vivo desde las cámaras CSI o desde una imagen estática) y
sirve una UI HTTP chica. El operador hace click-drag para definir la counting
zone rectangular y después hace click adentro para poner la línea virtual de
conteo (snapeada a la orientación configurada). Al save, el server escribe un
snippet YAML que entra directo bajo ``counter:`` en el config del dispositivo.

Uso típico en el dispositivo:
    PYTHONPATH=. python3 scripts/counting_zone_picker.py
    # Abrir http://people-counter.local:8080

Dev/test en laptop (sin picamera2):
    python scripts/counting_zone_picker.py --image sample.jpg --output ./out.yaml

El snippet YAML producido matchea ``counter.counting_zone`` / ``counter.line`` en
``config/config.example.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("counting_zone_picker")


# Shared state populated before the server starts. The handler reads these
# read-only, save_event/save_payload are written by the POST handler.
_frame_jpeg: bytes = b""
_frame_width: int = 0
_frame_height: int = 0
_output_path: Path = Path("./counting_zone_config.yaml")

_save_event = threading.Event()
_save_result: dict = {}
_save_lock = threading.Lock()


def _capture_from_cameras(
    camera: str, resolution: tuple[int, int],
) -> np.ndarray:
    """Toma un frame BGR único desde picamera2.

    Args:
        camera: "left", "right" o "both" (vista side-by-side, dibuja sobre L).
        resolution: (W, H) matcheando la del runtime — fundamental para que
            la counting zone / línea dibujadas tengan el mismo sistema de
            coordenadas que el pipeline.

    Returns:
        Un frame BGR único.

    Raises:
        RuntimeError: si picamera2 no está instalado o alguna cámara no abre.
    """
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "picamera2 no está instalado — usá --image PATH para testing de "
            "desarrollo, o instalá picamera2 en la RPi."
        ) from exc

    want_left = camera in ("left", "both")
    want_right = camera in ("right", "both")

    cam_left = None
    cam_right = None
    try:
        if want_left:
            cam_left = Picamera2(0)
        if want_right:
            cam_right = Picamera2(1)
    except Exception as exc:
        if cam_left is not None:
            try:
                cam_left.close()
            except Exception:
                pass
        if cam_right is not None:
            try:
                cam_right.close()
            except Exception:
                pass
        raise RuntimeError(f"No pude abrir cámara(s): {exc}") from exc

    from src.config.hardware import load_hardware_params
    hw = load_hardware_params()

    frames: dict[str, np.ndarray] = {}
    try:
        for cam, key in [(cam_left, "left"), (cam_right, "right")]:
            if cam is None:
                continue
            # raw mode desde el config del device (sensor.default_res).
            # Anclar el mode evita que picamera2 caiga al Mode 0 cropeado
            # del IMX708 — la counting zone tiene que dibujarse sobre el
            # mismo FOV que vé el runtime.
            cfg = cam.create_still_configuration(
                main={"size": resolution, "format": "BGR888"},
                raw={"size": hw.default_res},
            )
            cam.configure(cfg)
            cam.start()
        # Esperamos que el AE se asiente.
        time.sleep(hw.ae_initial_settle_seconds)
        for cam, key in [(cam_left, "left"), (cam_right, "right")]:
            if cam is None:
                continue
            raw = cam.capture_array("main")
            # picamera2 con "BGR888" entrega RGB en builds actuales de RPi
            # OS; convertimos a BGR para alinear con el resto del codebase.
            frames[key] = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    finally:
        for cam in (cam_left, cam_right):
            if cam is None:
                continue
            try:
                cam.stop()
                cam.close()
            except Exception:
                logger.warning("Error cerrando cámara", exc_info=True)

    if camera == "both":
        if "left" not in frames or "right" not in frames:
            raise RuntimeError("No pude capturar ambas cámaras")
        return np.hstack([frames["left"], frames["right"]])
    if camera == "left":
        return frames["left"]
    return frames["right"]


def _load_image(path: Path) -> np.ndarray:
    """Carga una imagen estática del disco.

    Raises:
        RuntimeError: si la imagen no se puede leer.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"No pude leer la imagen: {path}")
    return img


def _local_ip() -> str:
    """IP local best-effort (para el hint de arranque)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _format_yaml(x_min: int, x_max: int, y_min: int, y_max: int,
                 orientation: str, position: int) -> str:
    """Renderiza el snippet counter.counting_zone / counter.line."""
    return (
        "counter:\n"
        "  counting_zone:\n"
        f"    x_min: {x_min}\n"
        f"    x_max: {x_max}\n"
        f"    y_min: {y_min}\n"
        f"    y_max: {y_max}\n"
        "  line:\n"
        f"    orientation: {orientation}\n"
        f"    position: {position}\n"
    )


def _validate_payload(data: dict, img_w: int, img_h: int) -> tuple[Optional[dict], Optional[str]]:
    """Valida el payload JSON que viene de la UI.

    Devuelve ``(cleaned_dict, None)`` en éxito, ``(None, error_message)``
    en falla. Las coordenadas quedan clampeadas a los bounds de la imagen
    y la línea tiene que caer estrictamente dentro de la counting zone.
    """
    try:
        x_min = int(round(float(data["x_min"])))
        x_max = int(round(float(data["x_max"])))
        y_min = int(round(float(data["y_min"])))
        y_max = int(round(float(data["y_max"])))
        orientation = str(data["orientation"])
        position = int(round(float(data["position"])))
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"Malformed payload: {exc}"

    if orientation not in ("horizontal", "vertical"):
        return None, f"Invalid orientation: {orientation!r}"

    # Normalize min/max ordering.
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    # Clamp to image bounds.
    x_min = max(0, min(img_w - 1, x_min))
    x_max = max(0, min(img_w - 1, x_max))
    y_min = max(0, min(img_h - 1, y_min))
    y_max = max(0, min(img_h - 1, y_max))

    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return None, "Counting zone too small (min 10px each side)"

    if orientation == "horizontal":
        if not (y_min < position < y_max):
            return None, (
                f"Horizontal line y={position} must lie strictly inside "
                f"counting zone y=[{y_min},{y_max}]"
            )
    else:
        if not (x_min < position < x_max):
            return None, (
                f"Vertical line x={position} must lie strictly inside "
                f"counting zone x=[{x_min},{x_max}]"
            )

    return (
        {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "orientation": orientation, "position": position,
        },
        None,
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Counting Zone Picker</title>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: monospace; }
  header { padding: 8px 12px; background: #1e1e1e; display: flex;
           gap: 12px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; }
  button, select { background: #333; color: #eee; border: 1px solid #555;
                   padding: 6px 12px; font-family: inherit; cursor: pointer; }
  button.primary { background: #2d6; color: #000; border-color: #2d6; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  #stage { display: flex; justify-content: center; padding: 10px; }
  #wrap { position: relative; display: inline-block; }
  canvas { display: block; border: 1px solid #444; cursor: crosshair; }
  #info { padding: 6px 12px; color: #bbb; font-size: 13px; min-height: 18px; }
  #banner { padding: 6px 12px; font-size: 13px; min-height: 18px; }
  .err { background: #611; color: #fee; }
  .ok  { background: #161; color: #efe; }
  .step { color: #ff6; }
</style>
</head>
<body>
<header>
  <h1>Counting Zone Picker</h1>
  <label>Line orientation:
    <select id="orient">
      <option value="horizontal" selected>horizontal</option>
      <option value="vertical">vertical</option>
    </select>
  </label>
  <button id="reset">Reset</button>
  <button id="save" class="primary" disabled>Save</button>
  <span class="step" id="step">Step 1: click-drag to draw counting zone</span>
</header>
<div id="banner"></div>
<div id="info"></div>
<div id="stage">
  <div id="wrap">
    <canvas id="cv"></canvas>
  </div>
</div>

<script>
(function() {
  const canvas = document.getElementById('cv');
  const ctx = canvas.getContext('2d');
  const info = document.getElementById('info');
  const banner = document.getElementById('banner');
  const step = document.getElementById('step');
  const saveBtn = document.getElementById('save');
  const resetBtn = document.getElementById('reset');
  const orient = document.getElementById('orient');

  // Image pixel dims (original, authoritative for YAML output).
  let imgW = 0, imgH = 0;
  // Scale factor from image px -> canvas px (<=1 so we fit the screen).
  let scale = 1;

  const img = new Image();
  img.onload = () => {
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    const maxW = Math.min(window.innerWidth - 40, 1600);
    const maxH = window.innerHeight - 160;
    scale = Math.min(1, maxW / imgW, maxH / imgH);
    canvas.width = Math.round(imgW * scale);
    canvas.height = Math.round(imgH * scale);
    redraw();
  };
  img.src = '/frame.jpg?ts=' + Date.now();

  // State. All coords stored in IMAGE pixel space.
  let phase = 'zone';  // 'zone' | 'zone-dragging' | 'line'
  let zone = null;     // {x_min,x_max,y_min,y_max}
  let dragStart = null;
  let dragEnd = null;
  let line = null;    // number (y for horizontal, x for vertical)

  function evToImg(e) {
    const r = canvas.getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    return {
      x: Math.max(0, Math.min(imgW - 1, Math.round(cx / scale))),
      y: Math.max(0, Math.min(imgH - 1, Math.round(cy / scale))),
    };
  }

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw pending drag rectangle.
    let rect = zone;
    if (phase === 'zone-dragging' && dragStart && dragEnd) {
      rect = {
        x_min: Math.min(dragStart.x, dragEnd.x),
        x_max: Math.max(dragStart.x, dragEnd.x),
        y_min: Math.min(dragStart.y, dragEnd.y),
        y_max: Math.max(dragStart.y, dragEnd.y),
      };
    }
    if (rect) {
      const x = rect.x_min * scale;
      const y = rect.y_min * scale;
      const w = (rect.x_max - rect.x_min) * scale;
      const h = (rect.y_max - rect.y_min) * scale;
      ctx.fillStyle = 'rgba(0, 200, 120, 0.20)';
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = '#2d6';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }

    // Draw line (clipped to the counting zone for visual clarity).
    if (zone && line != null) {
      ctx.strokeStyle = '#f60';
      ctx.lineWidth = 3;
      ctx.beginPath();
      if (orient.value === 'horizontal') {
        ctx.moveTo(zone.x_min * scale, line * scale);
        ctx.lineTo(zone.x_max * scale, line * scale);
      } else {
        ctx.moveTo(line * scale, zone.y_min * scale);
        ctx.lineTo(line * scale, zone.y_max * scale);
      }
      ctx.stroke();
    }

    // Info line.
    const parts = [];
    if (zone) parts.push(`zone: x=[${zone.x_min},${zone.x_max}] y=[${zone.y_min},${zone.y_max}]`);
    if (zone && line != null) parts.push(`line ${orient.value} @ ${line}`);
    parts.push(`image ${imgW}x${imgH}`);
    info.textContent = parts.join('  |  ');
  }

  canvas.addEventListener('mousedown', (e) => {
    const p = evToImg(e);
    if (phase === 'zone' || phase === 'zone-dragging' || !zone) {
      phase = 'zone-dragging';
      dragStart = p;
      dragEnd = p;
      redraw();
    } else {
      // Placing the line.
      if (!withinZone(p)) {
        showBanner('Click inside the counting zone to place the line.', false);
        return;
      }
      line = (orient.value === 'horizontal') ? p.y : p.x;
      clearBanner();
      redraw();
      updateSaveState();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (phase !== 'zone-dragging') return;
    dragEnd = evToImg(e);
    redraw();
  });

  canvas.addEventListener('mouseup', (e) => {
    if (phase !== 'zone-dragging') return;
    dragEnd = evToImg(e);
    const w = Math.abs(dragEnd.x - dragStart.x);
    const h = Math.abs(dragEnd.y - dragStart.y);
    if (w < 10 || h < 10) {
      phase = 'zone';
      dragStart = dragEnd = null;
      showBanner('Counting zone too small, try again.', false);
      redraw();
      return;
    }
    zone = {
      x_min: Math.min(dragStart.x, dragEnd.x),
      x_max: Math.max(dragStart.x, dragEnd.x),
      y_min: Math.min(dragStart.y, dragEnd.y),
      y_max: Math.max(dragStart.y, dragEnd.y),
    };
    // Reset line if it is no longer inside the new counting zone.
    if (line != null && !lineInsideZone()) line = null;
    phase = 'line';
    step.textContent = 'Step 2: click inside the counting zone to place the line';
    clearBanner();
    dragStart = dragEnd = null;
    redraw();
    updateSaveState();
  });

  orient.addEventListener('change', () => {
    // Invalidate the prior line because the axis changed.
    line = null;
    clearBanner();
    redraw();
    updateSaveState();
  });

  resetBtn.addEventListener('click', () => {
    zone = null; line = null; phase = 'zone';
    step.textContent = 'Step 1: click-drag to draw counting zone';
    clearBanner();
    redraw();
    updateSaveState();
  });

  saveBtn.addEventListener('click', async () => {
    if (!zone || line == null) return;
    saveBtn.disabled = true;
    try {
      const resp = await fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          x_min: zone.x_min, x_max: zone.x_max,
          y_min: zone.y_min, y_max: zone.y_max,
          orientation: orient.value, position: line,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        showBanner('Save failed: ' + (data.error || resp.statusText), false);
        saveBtn.disabled = false;
        return;
      }
      showBanner('Saved to ' + data.path + '. Server shutting down.', true);
      step.textContent = 'Done.';
    } catch (err) {
      showBanner('Network error: ' + err, false);
      saveBtn.disabled = false;
    }
  });

  function withinZone(p) {
    return zone && p.x >= zone.x_min && p.x <= zone.x_max &&
                   p.y >= zone.y_min && p.y <= zone.y_max;
  }

  function lineInsideZone() {
    if (!zone || line == null) return false;
    if (orient.value === 'horizontal') return line > zone.y_min && line < zone.y_max;
    return line > zone.x_min && line < zone.x_max;
  }

  function updateSaveState() {
    saveBtn.disabled = !(zone && line != null && lineInsideZone());
  }

  function showBanner(msg, ok) {
    banner.textContent = msg;
    banner.className = ok ? 'ok' : 'err';
  }

  function clearBanner() {
    banner.textContent = '';
    banner.className = '';
  }
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    """Serves the picker UI, the background frame, and handles save POSTs."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("http %s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/frame.jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(_frame_jpeg)))
            self.end_headers()
            self.wfile.write(_frame_jpeg)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/save":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._json(400, {"ok": False, "error": f"Bad JSON: {exc}"})
            return

        cleaned, err = _validate_payload(data, _frame_width, _frame_height)
        if err is not None or cleaned is None:
            self._json(400, {"ok": False, "error": err or "validation failed"})
            return

        snippet = _format_yaml(
            cleaned["x_min"], cleaned["x_max"],
            cleaned["y_min"], cleaned["y_max"],
            cleaned["orientation"], cleaned["position"],
        )

        try:
            _output_path.parent.mkdir(parents=True, exist_ok=True)
            _output_path.write_text(snippet, encoding="utf-8")
        except OSError as exc:
            self._json(500, {"ok": False, "error": f"Write failed: {exc}"})
            return

        with _save_lock:
            _save_result["snippet"] = snippet
            _save_result["path"] = str(_output_path)
            _save_result["data"] = cleaned

        self._json(200, {"ok": True, "path": str(_output_path)})
        # Signal the main thread to shut down after flushing this response.
        _save_event.set()

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def main(argv: Optional[list[str]] = None) -> int:
    global _frame_jpeg, _frame_width, _frame_height, _output_path

    parser = argparse.ArgumentParser(
        description="Interactive counting zone + virtual line picker."
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Use a static image instead of a live camera capture.",
    )
    parser.add_argument(
        "--camera", choices=["left", "right", "both"], default="left",
        help="Which camera to use when capturing live (default: left).",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--output", type=Path, default=Path("./counting_zone_config.yaml"),
        help="Where to write the YAML snippet on save.",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=None,
        help="Capture resolution (W H) when --image is not set. Default: "
             "reads vision.resolution from /etc/people-counter/config.yaml "
             "so the counting zone / line use the same coordinate system "
             "as the runtime.",
    )
    args = parser.parse_args(argv)

    _output_path = args.output

    # Acquire the frame.
    try:
        if args.image is not None:
            logger.info("Loading static image %s", args.image)
            frame = _load_image(args.image)
        else:
            if args.resolution is None:
                from src.config.loader import (
                    DEFAULT_DEVICE_CONFIG_PATH,
                    load_device_config,
                )
                try:
                    cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
                except FileNotFoundError:
                    parser.error(
                        f"--resolution no provisto y "
                        f"{DEFAULT_DEVICE_CONFIG_PATH} no existe. Pasá "
                        "--resolution explícito o aprovisioná el config "
                        "per-device."
                    )
                res = cfg.get("vision", {}).get("resolution")
                if not res or not isinstance(res, list) or len(res) != 2:
                    parser.error(
                        f"vision.resolution no definido o malformado en "
                        f"{DEFAULT_DEVICE_CONFIG_PATH}. Esperaba lista "
                        "[W, H]."
                    )
                args.resolution = [int(res[0]), int(res[1])]
            logger.info(
                "Capturing live frame (camera=%s, resolution=%dx%d)",
                args.camera, args.resolution[0], args.resolution[1],
            )
            frame = _capture_from_cameras(
                args.camera,
                (int(args.resolution[0]), int(args.resolution[1])),
            )
    except RuntimeError as exc:
        logger.error("Frame acquisition failed: %s", exc)
        return 2

    h, w = frame.shape[:2]
    _frame_width = int(w)
    _frame_height = int(h)
    try:
        _frame_jpeg = _encode_jpeg(frame)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    # Spin up HTTP server on a background thread so we can wait for save.
    try:
        server = HTTPServer(("0.0.0.0", args.port), Handler)
    except OSError as exc:
        logger.error("Could not bind port %d: %s", args.port, exc)
        return 2

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    ip = _local_ip()
    print()
    print(
        f"Counting zone picker ready — open http://{ip}:{args.port}  "
        f"(or http://localhost:{args.port})"
    )
    print("  1. Click-drag to draw the counting zone rectangle.")
    print("  2. Choose orientation, then click inside the counting zone to place the line.")
    print("  3. Press Save. YAML snippet is written to:")
    print(f"     {_output_path.resolve()}")
    print("Press Ctrl-C to abort without saving.")
    print()

    exit_code = 0
    try:
        while not _save_event.wait(timeout=0.5):
            pass
    except KeyboardInterrupt:
        logger.info("Interrupted — no counting zone saved.")
        exit_code = 130
    finally:
        # Give the last HTTP response time to flush before we tear down.
        time.sleep(0.2)
        server.shutdown()
        server.server_close()

    if _save_event.is_set():
        with _save_lock:
            snippet = _save_result.get("snippet", "")
            path = _save_result.get("path", str(_output_path))
        print()
        print(f"Saved counting zone + line to {path}")
        print("-" * 60)
        print(snippet, end="")
        print("-" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

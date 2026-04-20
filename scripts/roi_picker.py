#!/usr/bin/env python3
"""Interactive ROI + counting-line picker for People Counter devices.

Captures one frame (live from CSI cameras or a static image) and serves a
small HTTP UI. The operator click-drags to define the rectangular ROI and
then clicks inside to place the virtual counting line (snapped to the
configured orientation). On save, the server writes a YAML snippet that
slots directly under ``counter:`` in the device config.

Typical usage on device:
    PYTHONPATH=. python3 scripts/roi_picker.py
    # Open http://people-counter.local:8080

Dev/test on a laptop (no picamera2):
    python scripts/roi_picker.py --image sample.jpg --output ./out.yaml

The YAML snippet produced matches ``counter.roi`` / ``counter.line`` in
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
logger = logging.getLogger("roi_picker")


# Shared state populated before the server starts. The handler reads these
# read-only, save_event/save_payload are written by the POST handler.
_frame_jpeg: bytes = b""
_frame_width: int = 0
_frame_height: int = 0
_output_path: Path = Path("./roi_config.yaml")

_save_event = threading.Event()
_save_result: dict = {}
_save_lock = threading.Lock()


def _capture_from_cameras(camera: str) -> np.ndarray:
    """Grab a single BGR frame from picamera2.

    Args:
        camera: "left", "right" or "both" (side-by-side view, draws on left).

    Returns:
        Single BGR frame.

    Raises:
        RuntimeError: If picamera2 is not installed or a camera fails to open.
    """
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "picamera2 not installed — use --image PATH for dev testing, "
            "or install picamera2 on the RPi."
        ) from exc

    want_left = camera in ("left", "both")
    want_right = camera in ("right", "both")

    cam_left = None
    cam_right = None
    try:
        if want_left:
            cam_left = Picamera2(1)
        if want_right:
            cam_right = Picamera2(0)
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
        raise RuntimeError(f"Failed to open camera(s): {exc}") from exc

    frames: dict[str, np.ndarray] = {}
    try:
        for cam, key in [(cam_left, "left"), (cam_right, "right")]:
            if cam is None:
                continue
            cfg = cam.create_still_configuration(
                main={"size": (1296, 972), "format": "BGR888"},
            )
            cam.configure(cfg)
            cam.start()
        # Let auto-exposure settle.
        time.sleep(1.0)
        for cam, key in [(cam_left, "left"), (cam_right, "right")]:
            if cam is None:
                continue
            raw = cam.capture_array("main")
            # picamera2 with "BGR888" delivers RGB on current RPi OS builds;
            # convert to BGR to match the rest of the codebase.
            frames[key] = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    finally:
        for cam in (cam_left, cam_right):
            if cam is None:
                continue
            try:
                cam.stop()
                cam.close()
            except Exception:
                logger.warning("Error closing camera", exc_info=True)

    if camera == "both":
        if "left" not in frames or "right" not in frames:
            raise RuntimeError("Could not capture both cameras")
        return np.hstack([frames["left"], frames["right"]])
    if camera == "left":
        return frames["left"]
    return frames["right"]


def _load_image(path: Path) -> np.ndarray:
    """Load a static image from disk.

    Raises:
        RuntimeError: If the image cannot be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def _local_ip() -> str:
    """Best-effort local IP (for the startup hint)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _format_yaml(x_min: int, x_max: int, y_min: int, y_max: int,
                 orientation: str, position: int) -> str:
    """Render the counter.roi / counter.line snippet."""
    return (
        "counter:\n"
        "  roi:\n"
        f"    x_min: {x_min}\n"
        f"    x_max: {x_max}\n"
        f"    y_min: {y_min}\n"
        f"    y_max: {y_max}\n"
        "  line:\n"
        f"    orientation: {orientation}\n"
        f"    position: {position}\n"
    )


def _validate_payload(data: dict, img_w: int, img_h: int) -> tuple[Optional[dict], Optional[str]]:
    """Validate the JSON payload from the UI.

    Returns (cleaned_dict, None) on success, (None, error_message) on failure.
    Coordinates are clamped to image bounds and the line must lie strictly
    inside the ROI.
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
        return None, "ROI too small (min 10px each side)"

    if orientation == "horizontal":
        if not (y_min < position < y_max):
            return None, (
                f"Horizontal line y={position} must lie strictly inside "
                f"ROI y=[{y_min},{y_max}]"
            )
    else:
        if not (x_min < position < x_max):
            return None, (
                f"Vertical line x={position} must lie strictly inside "
                f"ROI x=[{x_min},{x_max}]"
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
<title>ROI Picker</title>
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
  <h1>ROI Picker</h1>
  <label>Line orientation:
    <select id="orient">
      <option value="horizontal" selected>horizontal</option>
      <option value="vertical">vertical</option>
    </select>
  </label>
  <button id="reset">Reset</button>
  <button id="save" class="primary" disabled>Save</button>
  <span class="step" id="step">Step 1: click-drag to draw ROI</span>
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
  let phase = 'roi';  // 'roi' | 'roi-dragging' | 'line'
  let roi = null;     // {x_min,x_max,y_min,y_max}
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
    let rect = roi;
    if (phase === 'roi-dragging' && dragStart && dragEnd) {
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

    // Draw line (clipped to ROI for visual clarity).
    if (roi && line != null) {
      ctx.strokeStyle = '#f60';
      ctx.lineWidth = 3;
      ctx.beginPath();
      if (orient.value === 'horizontal') {
        ctx.moveTo(roi.x_min * scale, line * scale);
        ctx.lineTo(roi.x_max * scale, line * scale);
      } else {
        ctx.moveTo(line * scale, roi.y_min * scale);
        ctx.lineTo(line * scale, roi.y_max * scale);
      }
      ctx.stroke();
    }

    // Info line.
    const parts = [];
    if (roi) parts.push(`ROI: x=[${roi.x_min},${roi.x_max}] y=[${roi.y_min},${roi.y_max}]`);
    if (roi && line != null) parts.push(`line ${orient.value} @ ${line}`);
    parts.push(`image ${imgW}x${imgH}`);
    info.textContent = parts.join('  |  ');
  }

  canvas.addEventListener('mousedown', (e) => {
    const p = evToImg(e);
    if (phase === 'roi' || phase === 'roi-dragging' || !roi) {
      phase = 'roi-dragging';
      dragStart = p;
      dragEnd = p;
      redraw();
    } else {
      // Placing the line.
      if (!withinRoi(p)) {
        showBanner('Click inside the ROI to place the line.', false);
        return;
      }
      line = (orient.value === 'horizontal') ? p.y : p.x;
      clearBanner();
      redraw();
      updateSaveState();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (phase !== 'roi-dragging') return;
    dragEnd = evToImg(e);
    redraw();
  });

  canvas.addEventListener('mouseup', (e) => {
    if (phase !== 'roi-dragging') return;
    dragEnd = evToImg(e);
    const w = Math.abs(dragEnd.x - dragStart.x);
    const h = Math.abs(dragEnd.y - dragStart.y);
    if (w < 10 || h < 10) {
      phase = 'roi';
      dragStart = dragEnd = null;
      showBanner('ROI too small, try again.', false);
      redraw();
      return;
    }
    roi = {
      x_min: Math.min(dragStart.x, dragEnd.x),
      x_max: Math.max(dragStart.x, dragEnd.x),
      y_min: Math.min(dragStart.y, dragEnd.y),
      y_max: Math.max(dragStart.y, dragEnd.y),
    };
    // Reset line if it is no longer inside the new ROI.
    if (line != null && !lineInsideRoi()) line = null;
    phase = 'line';
    step.textContent = 'Step 2: click inside the ROI to place the line';
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
    roi = null; line = null; phase = 'roi';
    step.textContent = 'Step 1: click-drag to draw ROI';
    clearBanner();
    redraw();
    updateSaveState();
  });

  saveBtn.addEventListener('click', async () => {
    if (!roi || line == null) return;
    saveBtn.disabled = true;
    try {
      const resp = await fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          x_min: roi.x_min, x_max: roi.x_max,
          y_min: roi.y_min, y_max: roi.y_max,
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

  function withinRoi(p) {
    return roi && p.x >= roi.x_min && p.x <= roi.x_max &&
                  p.y >= roi.y_min && p.y <= roi.y_max;
  }

  function lineInsideRoi() {
    if (!roi || line == null) return false;
    if (orient.value === 'horizontal') return line > roi.y_min && line < roi.y_max;
    return line > roi.x_min && line < roi.x_max;
  }

  function updateSaveState() {
    saveBtn.disabled = !(roi && line != null && lineInsideRoi());
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
        description="Interactive ROI + virtual line picker."
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
        "--output", type=Path, default=Path("./roi_config.yaml"),
        help="Where to write the YAML snippet on save.",
    )
    args = parser.parse_args(argv)

    _output_path = args.output

    # Acquire the frame.
    try:
        if args.image is not None:
            logger.info("Loading static image %s", args.image)
            frame = _load_image(args.image)
        else:
            logger.info("Capturing live frame (camera=%s)", args.camera)
            frame = _capture_from_cameras(args.camera)
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
    print(f"ROI picker ready — open http://{ip}:{args.port}  (or http://localhost:{args.port})")
    print("  1. Click-drag to draw the counting ROI rectangle.")
    print("  2. Choose orientation, then click inside the ROI to place the line.")
    print("  3. Press Save. YAML snippet is written to:")
    print(f"     {_output_path.resolve()}")
    print("Press Ctrl-C to abort without saving.")
    print()

    exit_code = 0
    try:
        while not _save_event.wait(timeout=0.5):
            pass
    except KeyboardInterrupt:
        logger.info("Interrupted — no ROI saved.")
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
        print(f"Saved ROI + line to {path}")
        print("-" * 60)
        print(snippet, end="")
        print("-" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

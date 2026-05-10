#!/usr/bin/env python3
"""Preview en vivo estéreo para apuntar las cámaras.

Preview minimal browser-driven — sin detección, sin audio, sin análisis.
Solamente lo que las cámaras ven, side-by-side, con una grilla de tercios +
crosshair central para ayudar al encuadre. Misma UX que focus_assist /
calibrate (overlay de start, header).

Uso:
    sudo PYTHONPATH=. python3 scripts/preview.py
    # después abrir http://people-counter.local:8080
"""

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()
shutting_down = False
preview_started = False
preview_started_lock = threading.Lock()


HTML_PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Stereo preview</title>
<style>
  :root {
    --bg: #0a0a0a; --fg: #ddd; --muted: #888; --panel: #141414;
    --border: #222; --accent: #2ecc71;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: var(--bg);
               color: var(--fg);
               font-family: -apple-system, system-ui, sans-serif; }
  header { padding: 12px 18px; background: var(--panel);
           border-bottom: 1px solid var(--border);
           display: flex; align-items: center; justify-content: space-between;
           font-size: 14px; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }
  .stage { padding: 18px; }
  .stream { display: block; width: 100%; height: auto; max-width: 100%;
            border-radius: 6px; border: 1px solid var(--border);
            background: #000; }
  .legend { margin-top: 12px; color: var(--muted); font-size: 12px;
            display: flex; gap: 18px; flex-wrap: wrap; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 12px; height: 12px; display: inline-block;
              border-radius: 2px; background: var(--accent); }

  /* Start overlay (idem focus_assist / calibrate) */
  #start-overlay { position: fixed; inset: 0; background: rgba(10,10,10,0.95);
                   display: flex; align-items: center; justify-content: center;
                   z-index: 999; }
  #start-overlay .card { max-width: 460px; padding: 28px 32px;
                         background: var(--panel); border: 1px solid var(--border);
                         border-radius: 10px; text-align: center; }
  #start-overlay h2 { margin: 0 0 12px; font-size: 18px; }
  #start-overlay p { margin: 0 0 20px; color: var(--muted);
                     font-size: 13px; line-height: 1.5; }
  #start-overlay button { padding: 10px 28px; font-size: 14px;
                          border: 0; border-radius: 6px; background: var(--accent);
                          color: #000; cursor: pointer; font-weight: 600; }
  #start-overlay button:hover { background: #27ae60; }
</style>
</head>
<body>

<div id="start-overlay">
  <div class="card">
    <h2>Stereo preview</h2>
    <p>Vista en vivo de ambas cámaras lado a lado. Útil para apuntar el bracket,
       reframear escenas o verificar oclusiones antes de focus / calibración.
       Sin detección, sin métricas — solo lo que ven los lentes.</p>
    <button id="start-btn">Comenzar</button>
  </div>
</div>

<header>
  <h1>Stereo preview</h1>
  <span class="meta">Ctrl+C en la terminal para salir</span>
</header>

<div class="stage">
  <img id="stream" class="stream" />
  <div class="legend">
    <span><i style="background:#3c3"></i> Centro: cruz blanca</span>
    <span><i style="background:#666"></i> Grid de tercios: gris</span>
    <span>Etiquetas IZQ / DER en cada mitad</span>
  </div>
</div>

<script>
  const overlay = document.getElementById("start-overlay");
  const startBtn = document.getElementById("start-btn");
  const stream = document.getElementById("stream");

  startBtn.addEventListener("click", async () => {
    try {
      await fetch("/start", { method: "POST" });
    } catch (_) {}
    stream.src = "/stream";
    overlay.style.display = "none";
  });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()
            try:
                while not shutting_down:
                    with jpeg_lock:
                        jpg = latest_jpeg
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        global preview_started
        if self.path == "/start":
            with preview_started_lock:
                preview_started = True
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def _draw_overlay(vis: np.ndarray, label: str) -> None:
    """Center crosshair + thirds grid + camera label."""
    h, w = vis.shape[:2]
    grid = (60, 60, 60)
    for i in (1, 2):
        x = w * i // 3
        y = h * i // 3
        cv2.line(vis, (x, 0), (x, h), grid, 1)
        cv2.line(vis, (0, y), (w, y), grid, 1)
    cx, cy = w // 2, h // 2
    cross = (200, 200, 200)
    cv2.line(vis, (cx - 14, cy), (cx + 14, cy), cross, 1)
    cv2.line(vis, (cx, cy - 14), (cx, cy + 14), cross, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (4, 4), (4 + tw + 10, 4 + th + 12), (0, 0, 0), -1)
    cv2.putText(
        vis, label, (9, 4 + th + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _draw_counter_overlay(
    vis: np.ndarray,
    counter_cfg: dict,
    src_w: int,
    src_h: int,
) -> None:
    """Draw ROI rectangle + virtual counting line + direction labels on the
    preview, matching the runtime config of the pipeline. Coords in the
    config are at the source frame resolution (e.g. 2304x1296); we scale
    them to the preview resolution.
    """
    ph, pw = vis.shape[:2]
    sx = pw / src_w
    sy = ph / src_h

    roi = counter_cfg.get("roi") or {}
    line = counter_cfg.get("line") or {}
    labels = counter_cfg.get("direction_labels") or {}

    # ROI rectangle (yellow)
    if all(k in roi for k in ("x_min", "x_max", "y_min", "y_max")):
        x1 = int(roi["x_min"] * sx)
        x2 = int(roi["x_max"] * sx)
        y1 = int(roi["y_min"] * sy)
        y2 = int(roi["y_max"] * sy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.putText(
            vis, "ROI", (x1 + 6, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2, cv2.LINE_AA,
        )

    # Counting line (red) + direction labels on each side
    label_a = labels.get("side_a_to_b", "ingress")
    label_b = labels.get("side_b_to_a", "egress")
    if line.get("orientation") == "horizontal" and "position" in line:
        ly = int(line["position"] * sy)
        cv2.line(vis, (0, ly), (pw, ly), (0, 0, 255), 2)
        # Side A above the line, Side B below (per main.py convention)
        cv2.putText(
            vis, f"-> {label_a}", (12, ly - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            vis, f"<- {label_b}", (12, ly + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )
    elif line.get("orientation") == "vertical" and "position" in line:
        lx = int(line["position"] * sx)
        cv2.line(vis, (lx, 0), (lx, ph), (0, 0, 255), 2)
        cv2.putText(
            vis, f"v {label_a}", (lx + 6, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            vis, f"^ {label_b}", (lx + 6, ph - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )


def main() -> None:
    global latest_jpeg, shutting_down

    parser = argparse.ArgumentParser(description="Stereo live preview")
    parser.add_argument("--left", type=int, default=0,
                        help="Left camera index. Default 0.")
    parser.add_argument("--right", type=int, default=1,
                        help="Right camera index. Default 1.")
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        help="Capture resolution. Default: reads "
                             "vision.resolution from /etc/people-counter/"
                             "config.yaml (matches the runtime). Pass "
                             "explicit only for tests / dev workstations.")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port for the preview UI. Default 8080.")
    parser.add_argument("--meter", choices=("matrix", "centre", "spot"),
                        default="matrix",
                        help="AE metering mode. Use 'centre' / 'spot' when "
                             "bright zones in the periphery (windows, lights) "
                             "drag exposure down on the centre.")
    parser.add_argument("--config", default=None,
                        help="Optional path to the runtime config.yaml. When "
                             "provided, the preview matches the pipeline's "
                             "actual setup: takes resolution from config, "
                             "rectifies frames using the calibration .npz, "
                             "and draws the ROI rectangle + counting line + "
                             "direction labels on the L preview. Use this to "
                             "verify the runtime layout before launching "
                             "main.py.")
    args = parser.parse_args()

    # Optional: load runtime config so the preview reflects the pipeline's
    # actual setup (resolution, calibration, ROI / line overlay).
    runtime_cfg = None
    calibration = None
    counter_cfg = None
    if args.config:
        from src.config.loader import load_config
        runtime_cfg = load_config(args.config)
        vision_cfg = runtime_cfg.get("vision", {})
        # Honour the resolution from config so the ROI/line scale match
        # whatever main.py will see.
        cfg_res = vision_cfg.get("resolution")
        if cfg_res:
            if args.resolution is not None and tuple(cfg_res) != tuple(args.resolution):
                print(f"Resolución del config: {cfg_res} (override del CLI "
                      f"{args.resolution})")
            args.resolution = list(cfg_res)
        # Calibration for rectification — without it the ROI overlay still
        # shows but at the un-rectified frame, which won't match the
        # pipeline's coordinate system exactly.
        cal_path = vision_cfg.get("calibration_file")
        if cal_path and Path(cal_path).exists():
            from src.vision.calibration import load_calibration
            calibration = load_calibration(cal_path)
            print(f"Calibración cargada: {cal_path} (frames serán rectificados)")
        elif cal_path:
            print(f"⚠ calibration_file en config ({cal_path}) no existe — "
                  f"frames sin rectificar; ROI/línea pueden no coincidir "
                  f"exacto con lo que ve el pipeline.")
        counter_cfg = runtime_cfg.get("counter")
        if counter_cfg:
            print("ROI + línea de conteo se dibujarán sobre el preview L.")

    if args.resolution is None:
        # No --config y no --resolution: leer del device config strict —
        # fuente única de verdad para que el preview matchee el runtime.
        from src.config.loader import (
            DEFAULT_DEVICE_CONFIG_PATH,
            load_device_config,
        )
        try:
            cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
        except FileNotFoundError:
            parser.error(
                f"--resolution no provisto y {DEFAULT_DEVICE_CONFIG_PATH} "
                "no existe. Pasá --resolution o --config explícito."
            )
        res = cfg.get("vision", {}).get("resolution")
        if not res or not isinstance(res, list) or len(res) != 2:
            parser.error(
                f"vision.resolution no definido o malformado en "
                f"{DEFAULT_DEVICE_CONFIG_PATH}. Esperaba lista [W, H]."
            )
        args.resolution = [int(res[0]), int(res[1])]

    from picamera2 import Picamera2
    from libcamera import controls as _libcam_controls

    cam_l = Picamera2(args.left)
    cam_r = Picamera2(args.right)
    w, h = args.resolution
    for cam in [cam_l, cam_r]:
        config = cam.create_still_configuration(
            main={"size": (w, h), "format": "BGR888"},
            raw={"size": (w, h)},
        )
        cam.configure(config)
        cam.start()

    if args.meter != "matrix":
        meter_map = {
            "centre": _libcam_controls.AeMeteringModeEnum.CentreWeighted,
            "spot": _libcam_controls.AeMeteringModeEnum.Spot,
        }
        for cam in [cam_l, cam_r]:
            cam.set_controls({"AeMeteringMode": meter_map[args.meter]})

    time.sleep(1)

    ThreadingHTTPServer.allow_reuse_address = True
    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    except OSError as e:
        print(f"❌ No se pudo abrir el puerto {args.port}: {e}")
        try:
            cam_l.stop(); cam_l.close()
            cam_r.stop(); cam_r.close()
        except Exception:
            pass
        sys.exit(1)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    host = socket.gethostname()
    print(f"Preview en http://{host}.local:{args.port} (o http://0.0.0.0:{args.port})")
    print("Ctrl+C para salir.")

    half_w, half_h = 960, 540

    # Single try/finally around BOTH the wait-for-start phase and the
    # main capture loop so Ctrl+C in either path goes through cleanup.
    # Antes el wait loop estaba afuera del try y un Ctrl+C ahí dejaba
    # el thread del HTTP server colgando, dejando el puerto en TIME_WAIT
    # cuando arrancaba la siguiente tool.
    try:
        # Esperar a que el operador haga click en "Comenzar". La URL es
        # cargable pero el loop de captura de cámara solo arranca cuando
        # la página está abierta.
        print("Esperando 'Comenzar' en el browser...")
        while not shutting_down:
            with preview_started_lock:
                if preview_started:
                    break
            time.sleep(0.1)

        while not shutting_down:
            frame_l = cv2.cvtColor(
                cam_l.capture_array("main"), cv2.COLOR_RGB2BGR,
            )
            frame_r = cv2.cvtColor(
                cam_r.capture_array("main"), cv2.COLOR_RGB2BGR,
            )

            # Rectificación opcional — cuando --config nos dio una
            # calibración, el preview muestra lo que el pipeline
            # efectivamente ve post-rectify. Sin eso se muestran los
            # frames raw.
            if calibration is not None:
                from src.vision.calibration import rectify_pair
                frame_l, frame_r = rectify_pair(frame_l, frame_r, calibration)

            src_h, src_w = frame_l.shape[:2]
            vis_l = cv2.resize(frame_l, (half_w, half_h))
            vis_r = cv2.resize(frame_r, (half_w, half_h))
            _draw_overlay(vis_l, "IZQ")
            _draw_overlay(vis_r, "DER")

            # ROI + counting line drawn only on L (the preview reference for
            # alignment), in the same coordinate system the pipeline uses.
            if counter_cfg is not None:
                _draw_counter_overlay(vis_l, counter_cfg, src_w, src_h)

            combined = np.hstack([vis_l, vis_r])
            _, jpeg = cv2.imencode(
                ".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 75],
            )

            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nCerrando...")
    finally:
        shutting_down = True
        try:
            cam_l.stop(); cam_l.close()
            cam_r.stop(); cam_r.close()
        except Exception:
            pass
        try:
            server.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

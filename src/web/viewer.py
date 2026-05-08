"""Viewer HTTP en vivo del pipeline runtime.

Streamea un composite de 3 paneles (left annotated | right raw | depth
colormap) como MJPEG sobre HTTP. Hecho para debug on-site: el operador abre
la IP del dispositivo en un celular o laptop, camina bajo las cámaras y ve
el overlay de detección / tracking / conteo en vivo.

Diseñado para ser seguro en el hot path del runtime:

- ``push`` es no-bloqueante. La queue de encode descarta el más viejo si el
  cliente es lento; el pipeline nunca se traba esperando al viewer.
- El encoding JPEG corre en su propio thread, no en el loop del pipeline.
- Las fallas de bind (puerto en uso, CAP_NET_BIND_SERVICE faltante en el
  puerto 80) se loggean pero no matan el pipeline — el counter sigue corriendo.

Puerto default 80 porque el operador on-site no carga una lista de puertos
custom. ``--web-viewer-port 0`` deshabilita el viewer entero.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_HTML = """<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>People Counter \xe2\x80\x94 Live</title>
  <style>
    body { margin: 0; background: #000; color: #eee;
           font-family: ui-monospace, Menlo, monospace; }
    /* Una sola línea compacta de stats así la imagen se queda casi con
       toda la altura del viewport. */
    #stats { padding: 4px 12px; background: #1a1a1a;
             border-bottom: 1px solid #2c2c2c; font-size: 12px;
             line-height: 1.4; white-space: nowrap; overflow-x: auto; }
    #stats span { margin-right: 14px; }
    #stats b { color: #fff; }
    #stats .label { color: #999; }
    img { display: block; max-width: 100%; max-height: calc(100vh - 26px);
          margin: 0 auto; }
  </style>
</head>
<body>
  <div id='stats'>
    <span><span class='label'>IN:</span> <b id='in'>0</b></span>
    <span><span class='label'>OUT:</span> <b id='out'>0</b></span>
    <span><span class='label'>FPS:</span> <b id='fps'>0</b></span>
    <span><span class='label'>Tracks:</span> <b id='tracks'>0</b></span>
    <span><span class='label'>Dets:</span> <b id='dets'>0</b></span>
  </div>
  <img src='/stream' alt='live stream'/>
  <script>
    setInterval(async () => {
      try {
        const r = await fetch('/stats', {cache: 'no-store'});
        const s = await r.json();
        document.getElementById('in').textContent = s.total_in;
        document.getElementById('out').textContent = s.total_out;
        document.getElementById('fps').textContent =
            (typeof s.fps === 'number') ? s.fps.toFixed(1) : '-';
        document.getElementById('tracks').textContent = s.tracks;
        document.getElementById('dets').textContent = s.dets;
      } catch (e) {}
    }, 1000);
  </script>
</body>
</html>
"""


class _ReusableServer(ThreadingHTTPServer):
    # SO_REUSEADDR así un restart rápido no pega contra un socket TIME_WAIT.
    allow_reuse_address = True
    daemon_threads = True


class WebViewer:
    """Viewer MJPEG live sobre HTTP, aislado del hot path del pipeline.

    Lifecycle:
        viewer = WebViewer(port=80)
        if viewer.start():            # False en falla de bind (puerto ocupado / perm)
            ...
            viewer.push(frame_bgr, {"total_in": ..., ...})
            ...
            viewer.stop()
    """

    def __init__(
        self,
        port: int = 80,
        host: str = "0.0.0.0",
        jpeg_quality: int = 70,
        queue_size: int = 4,
    ) -> None:
        self.port = port
        self.host = host
        self.jpeg_quality = int(jpeg_quality)
        # Queue de input al encoder. Semántica drop-oldest enforced en ``push``.
        self._encode_queue: deque[np.ndarray] = deque(maxlen=queue_size)
        self._encode_lock = threading.Lock()
        self._encode_evt = threading.Event()
        # Output JPEG más reciente, servido a todos los clientes suscriptos.
        self._latest_jpeg: Optional[bytes] = None
        self._latest_id = 0
        self._latest_cond = threading.Condition()
        # Stats publicados por /stats.
        self._stats: dict[str, Any] = {
            "total_in": 0, "total_out": 0, "fps": 0.0,
            "tracks": 0, "dets": 0,
        }
        self._stats_lock = threading.Lock()
        self._server: Optional[_ReusableServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._encode_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    # ----------------------------------------------------------------- API
    @property
    def running(self) -> bool:
        return self._server is not None and not self._stop_evt.is_set()

    def start(self) -> bool:
        """Bindea, arranca a servir, devuelve True ante éxito.

        Ante falla de bind (puerto en uso, sin permiso para puerto 80)
        loguea un warning y devuelve False — el caller debería tratar
        al viewer como deshabilitado pero mantener el pipeline corriendo.
        """
        handler_cls = _build_handler(self)
        try:
            self._server = _ReusableServer(
                (self.host, self.port), handler_cls,
            )
        except (OSError, PermissionError) as e:
            logger.warning(
                "WebViewer bind to %s:%d failed (%s) - viewer disabled, "
                "pipeline continues.",
                self.host, self.port, e,
            )
            self._server = None
            return False

        self._stop_evt.clear()
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="web-viewer-http", daemon=True,
        )
        self._server_thread.start()
        self._encode_thread = threading.Thread(
            target=self._encode_loop,
            name="web-viewer-encode", daemon=True,
        )
        self._encode_thread.start()
        logger.info(
            "WebViewer listening on http://%s:%d/", self.host, self.port,
        )
        return True

    def stop(self) -> None:
        if self._server is None:
            return
        self._stop_evt.set()
        # Despertar a los waiters así los handlers de /stream salen limpio.
        with self._latest_cond:
            self._latest_cond.notify_all()
        self._encode_evt.set()
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            logger.exception("WebViewer shutdown failed")
        if self._encode_thread is not None:
            self._encode_thread.join(timeout=2.0)
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        self._server = None

    def push(
        self, frame_bgr: np.ndarray, stats: Optional[dict] = None,
    ) -> None:
        """No bloqueante. Drop-oldest si la queue de encode está llena."""
        if self._server is None:
            return
        with self._encode_lock:
            self._encode_queue.append(frame_bgr)
        self._encode_evt.set()
        if stats is not None:
            with self._stats_lock:
                self._stats.update(stats)

    # ------------------------------------------------------------- internal
    def _encode_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._encode_evt.wait(timeout=0.5)
            self._encode_evt.clear()
            while not self._stop_evt.is_set():
                with self._encode_lock:
                    if not self._encode_queue:
                        break
                    # Cuando el encoder se queda atrás, saltar adelante
                    # al frame más nuevo y descartar el resto.
                    frame = self._encode_queue.pop()
                    self._encode_queue.clear()
                try:
                    ok, buf = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                except Exception:
                    logger.exception("JPEG encode failed")
                    continue
                if not ok:
                    continue
                with self._latest_cond:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_id += 1
                    self._latest_cond.notify_all()

    def _wait_next_jpeg(
        self, last_seen: int, timeout: float = 5.0,
    ) -> tuple[Optional[bytes], int]:
        with self._latest_cond:
            if self._latest_id == last_seen and not self._stop_evt.is_set():
                self._latest_cond.wait(timeout=timeout)
            return self._latest_jpeg, self._latest_id

    def _stats_json(self) -> bytes:
        with self._stats_lock:
            return json.dumps(self._stats).encode()


def _build_handler(viewer: WebViewer):
    class Handler(BaseHTTPRequestHandler):
        # Silenciar el log default per-request a stderr (spamearía el
        # log con cada byte de MJPEG).
        def log_message(self, fmt, *args) -> None:  # noqa: ARG002
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                _html_bytes = _HTML.encode("utf-8")
                self.send_header("Content-Length", str(len(_html_bytes)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(_html_bytes)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path == "/stats":
                payload = viewer._stats_json()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path == "/stream":
                self._stream_mjpeg()
                return
            self.send_error(404)

        def _stream_mjpeg(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            last_id = -1
            try:
                while not viewer._stop_evt.is_set():
                    jpeg, last_id = viewer._wait_next_jpeg(last_id)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # Cliente desconectado mid-stream; esperado.
                pass

    return Handler

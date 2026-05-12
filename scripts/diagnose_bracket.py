#!/usr/bin/env python3
"""Diagnóstico standalone de alineamiento del bracket estéreo — browser driven.

Pensado para QC en línea de ensamblado: la persona arma la unidad
(cámaras + bracket + PoE HAT), posiciona el board ChArUco frontal a
~1m, abre la UI en el browser, hace click en Comenzar, y obtiene
PASS/FAIL en ~15 segundos por unidad. Sin necesidad de calibración
estéreo previa, sin necesidad de bajar al lab.

Cómo funciona
-------------
Captura ``--frames`` pares L/R con AE lockeado tras un click de start.
Por cada par:
  - Detecta esquinas ChArUco en ambas cámaras (sub-pixel).
  - solvePnP en cada cámara con K nominal del IMX708 (focal calculada
    desde NOMINAL_FOCAL_PX escalada a la resolución de captura), sin
    distorsión — para markers cerca del centro óptico la distorsión
    fisheye es despreciable (<2% error en proyección a ±10°).
  - Compone la pose relativa L→R y descompone en pitch / yaw / roll +
    offsets Y / Z usando ``lens_alignment_metrics``. La baseline (offset
    X) NO se mide acá — el bracket 3D-printed la fija con tolerancia de
    fabricación, y además su accuracy depende fuerte del D real del lens
    (que no tenemos pre-calibración).

Las medidas se agregan con mediana (robusta a outliers de detección)
y std (estabilidad cross-frame). El verdict final compara la mediana
contra thresholds operativos.

Thresholds default (matchean los del reporte del wizard)
--------------------------------------------------------
  pitch / roll < 0.5°
  yaw         < 1.0°
  offset Y / Z < 2 mm

Uso
---
  sudo PYTHONPATH=. python3 scripts/diagnose_bracket.py
  abrir http://people-counter.local:8080

Posicionamiento del operador:
  - Board ChArUco frontal a ~1m de las cámaras (entre 0.7m y 1.3m).
  - Centrado en el FOV de ambas cámaras (verificar en la UI antes de
    Comenzar).
  - Iluminación uniforme, sin glare especular sobre el board.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import logging
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.calibration import (
    NOMINAL_FOCAL_PX,
    NOMINAL_FULL_RES,
    PoseTarget,
    create_charuco_board,
    detect_charuco_dual_pass,
    lens_alignment_metrics,
    project_pose,
)
from src.vision.capture import CANONICAL_RAW_SIZE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose_bracket")


THR_PITCH_DEG = 0.5
THR_YAW_DEG = 1.0
THR_ROLL_DEG = 0.5
THR_OFFSET_Y_MM = 2.0
THR_OFFSET_Z_MM = 2.0


# -----------------------------------------------------------------------------
# Math helpers (idénticos a la versión CLI; reusan calibration.py)
# -----------------------------------------------------------------------------


def _build_nominal_k(width: int, height: int) -> np.ndarray:
    """K matrix nominal del IMX708 escalada a la resolución de captura.

    Principal point asumido centrado. Fallback cuando no hay calibration
    disponible — accuracy del diagnose se degrada significativamente sin
    K-B distortion (ver _solve_board_pose).
    """
    fx = NOMINAL_FOCAL_PX * width / NOMINAL_FULL_RES[0]
    fy = NOMINAL_FOCAL_PX * height / NOMINAL_FULL_RES[1]
    return np.array(
        [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def _solve_board_pose(
    corners: np.ndarray, ids: np.ndarray,
    board: cv2.aruco.CharucoBoard, K: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """SolvePnP del board en el frame de la cámara.

    Pinhole + D=0 sobre imagen fisheye introduce bias en los tvecs
    (sobre-estimación de magnitud por la distorsión periférica no
    modelada). Los biases son simétricos entre L y R, y al componer
    R_LR/T_LR los efectos sobre rotaciones y offsets Y/Z se cancelan
    razonablemente bien (±0.5° / ±10mm). La magnitud absoluta (que
    sería la baseline) NO cancela y por eso no se reporta — para
    medir baseline precisa hace falta calibración previa, que es lo
    que viene después de este QC.
    """
    obj_pts = board.getChessboardCorners()[ids.flatten()].astype(np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)
    if len(obj_pts) < 6:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    return rvec, tvec


def _compose_relative_pose(
    rvec_l: np.ndarray, tvec_l: np.ndarray,
    rvec_r: np.ndarray, tvec_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R_L, _ = cv2.Rodrigues(rvec_l)
    R_R, _ = cv2.Rodrigues(rvec_r)
    R_LR = R_R @ R_L.T
    T_LR = tvec_r - R_LR @ tvec_l
    return R_LR, T_LR


# -----------------------------------------------------------------------------
# Estado compartido entre el capture thread y el HTTP server
# -----------------------------------------------------------------------------


class _State:
    """Estado mutable de la sesión, accedido desde el HTTP handler y el
    capture loop. Las escrituras son single-writer (capture thread)
    así no hace falta lock para reads del HTTP server."""

    def __init__(self) -> None:
        self.phase: str = "waiting"          # waiting | collecting | done | error
        self.message: str = "Posicioná el board y apretá Comenzar"
        self.samples: list[dict[str, float]] = []
        self.target_samples: int = 30
        self.skipped: int = 0
        self.latest_jpeg: bytes = b""
        # Latest per-frame metrics (sirven para preview live aunque
        # no estemos coleccionando todavía).
        self.live_metrics: Optional[dict[str, float]] = None
        self.final_verdict: Optional[bool] = None
        self.final_summary: Optional[dict[str, object]] = None
        # Snapshot del par L|R compuesto en el momento del verdict,
        # encodeable a base64 para embeber en el reporte HTML como
        # prueba visual del estado de la unidad.
        self.snapshot_jpeg: bytes = b""
        # Path al reporte HTML guardado, exposeado por /status para
        # que la UI lo linkee.
        self.report_path: Optional[str] = None


_state = _State()
_start_event = threading.Event()
_finish_requested = threading.Event()
_capture_done = threading.Event()


# -----------------------------------------------------------------------------
# HTTP server (idéntico patrón a focus_assist / wizard)
# -----------------------------------------------------------------------------


_HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Diagnóstico de bracket</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a1a; color: #ddd; margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 12px 0; color: #fff; }
  .preview { background: #000; border: 1px solid #333; max-width: 100%; }
  .panel { margin-top: 12px; padding: 10px 14px; border-radius: 4px; background: #2a2a2a; }
  .banner { padding: 14px; margin: 12px 0; text-align: center; font-weight: bold; font-size: 16px; border-radius: 4px; }
  .banner.waiting { background: #555; color: #fff; }
  .banner.collecting { background: #3a3a72; color: #fff; }
  .banner.pass { background: #2ecc71; color: #000; }
  .banner.fail { background: #e74c3c; color: #fff; }
  table { width: 100%; border-collapse: collapse; }
  td { padding: 4px 8px; font-family: monospace; font-size: 13px; }
  td.label { color: #888; width: 90px; }
  td.value { font-weight: bold; }
  td.thr { color: #666; font-size: 11px; text-align: right; }
  td.verdict { text-align: center; width: 60px; font-weight: bold; }
  td.verdict.pass { color: #2ecc71; }
  td.verdict.fail { color: #e74c3c; }
  button { padding: 12px 32px; font-size: 15px; background: #2ecc71; color: #000; border: 0; border-radius: 4px; cursor: pointer; font-weight: bold; }
  button:hover { background: #27ae60; }
  button.secondary { background: #555; color: #fff; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .progress { background: #333; height: 8px; border-radius: 4px; overflow: hidden; margin: 6px 0; }
  .progress-bar { background: #2ecc71; height: 100%; transition: width 0.2s; }
  .footer { color: #666; font-size: 11px; margin-top: 16px; }
</style></head>
<body>
<h1>Diagnóstico de bracket estéreo</h1>
<img id="preview" class="preview" src="/stream" />
<div id="banner" class="banner waiting">Cargando...</div>
<div class="panel">
  <table>
    <tr><td class="label">pitch</td><td class="value" id="m-pitch">—</td><td class="thr" id="t-pitch"></td><td class="verdict" id="v-pitch">—</td></tr>
    <tr><td class="label">yaw</td><td class="value" id="m-yaw">—</td><td class="thr" id="t-yaw"></td><td class="verdict" id="v-yaw">—</td></tr>
    <tr><td class="label">roll</td><td class="value" id="m-roll">—</td><td class="thr" id="t-roll"></td><td class="verdict" id="v-roll">—</td></tr>
    <tr><td class="label">offset Y</td><td class="value" id="m-offy">—</td><td class="thr" id="t-offy"></td><td class="verdict" id="v-offy">—</td></tr>
    <tr><td class="label">offset Z</td><td class="value" id="m-offz">—</td><td class="thr" id="t-offz"></td><td class="verdict" id="v-offz">—</td></tr>
  </table>
  <div class="progress" id="progress-wrap" style="display:none"><div class="progress-bar" id="progress" style="width:0%"></div></div>
</div>
<div style="margin-top:12px; display:flex; gap:8px; flex-wrap: wrap;">
  <button id="start-btn" onclick="onStart()">Comenzar</button>
  <button id="finish-btn" onclick="onFinish()" style="display:none">Abrir reporte y finalizar</button>
  <button class="secondary" id="cancel-btn" onclick="onCancel()">Cancelar</button>
</div>
<div class="footer">Posicioná el board ChArUco frontal a ~1m alineado con el ghost de la mitad izquierda. Iluminación uniforme. La cámara captura {target} frames y agrega con mediana. El reporte HTML queda en disco como prueba de PASS por unidad.</div>

<script>
function fmt(val, unit, digits) {
  if (val === null || val === undefined) return '—';
  return (val >= 0 ? '+' : '') + val.toFixed(digits) + unit;
}
function setVerdict(id, ok) {
  const el = document.getElementById(id);
  if (ok === null || ok === undefined) { el.textContent = '—'; el.className = 'verdict'; return; }
  el.textContent = ok ? 'OK' : 'FAIL';
  el.className = 'verdict ' + (ok ? 'pass' : 'fail');
}
function onStart() {
  fetch('/start', {method: 'POST'});
}
function onFinish() {
  // Abrir reporte en nueva pestaña ANTES de pedir shutdown — el
  // contenido lo tenemos que pedir al server mientras siga vivo. El
  // browser permite window.open porque viene de un click del operador.
  const w = window.open('/report', '_blank');
  if (!w) {
    alert('El navegador bloqueó la apertura — permití popups y volvé a apretar.');
    return;
  }
  // Esperá un poquito para que el GET /report termine de cargar antes
  // de tirar el server.
  setTimeout(() => {
    fetch('/finish', {method: 'POST'});
    document.body.innerHTML = '<h2 style="padding:32px;color:#888;font-family:sans-serif">Proceso terminado. Reporte abierto en otra pestaña.</h2>';
  }, 400);
}
function onCancel() {
  if (!confirm('Cancelar sin guardar reporte?')) return;
  fetch('/finish', {method: 'POST'});
  document.body.innerHTML = '<h2 style="padding:32px;color:#888;font-family:sans-serif">Cancelado.</h2>';
}
async function poll() {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    const banner = document.getElementById('banner');
    banner.textContent = s.message;
    banner.className = 'banner ' + s.phase_class;
    document.getElementById('start-btn').style.display = (s.phase === 'waiting' ? 'inline-block' : 'none');
    document.getElementById('finish-btn').style.display = (s.phase === 'done' && s.report_path ? 'inline-block' : 'none');
    document.getElementById('progress-wrap').style.display = (s.phase === 'collecting' ? 'block' : 'none');
    document.getElementById('progress').style.width = (100 * s.samples / s.target_samples) + '%';
    const m = s.metrics || {};
    document.getElementById('m-pitch').textContent = fmt(m.pitch, '°', 3);
    document.getElementById('m-yaw').textContent = fmt(m.yaw, '°', 3);
    document.getElementById('m-roll').textContent = fmt(m.roll, '°', 3);
    document.getElementById('m-offy').textContent = fmt(m.offset_y, 'mm', 2);
    document.getElementById('m-offz').textContent = fmt(m.offset_z, 'mm', 2);
    const v = s.checks || {};
    setVerdict('v-pitch', v.pitch);
    setVerdict('v-yaw', v.yaw);
    setVerdict('v-roll', v.roll);
    setVerdict('v-offy', v.offset_y);
    setVerdict('v-offz', v.offset_z);
    const t = s.thresholds || {};
    document.getElementById('t-pitch').textContent = '±' + t.pitch + '°';
    document.getElementById('t-yaw').textContent = '±' + t.yaw + '°';
    document.getElementById('t-roll').textContent = '±' + t.roll + '°';
    document.getElementById('t-offy').textContent = '±' + t.offset_y + 'mm';
    document.getElementById('t-offz').textContent = '±' + t.offset_z + 'mm';
  } catch (e) {}
  setTimeout(poll, 200);
}
poll();
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silenciar access logs
        pass

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            html = _HTML.replace("{target}", str(_state.target_samples))
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _finish_requested.is_set():
                    jpg = _state.latest_jpeg
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                        self.wfile.write(str(len(jpg)).encode())
                        self.wfile.write(b"\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/status":
            payload = _build_status_json().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/report":
            self._serve_report()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/start":
            _start_event.set()
            self.send_response(204)
            self.end_headers()
        elif self.path == "/finish":
            _finish_requested.set()
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def _serve_report(self) -> None:
        if not _state.report_path:
            self.send_error(404)
            return
        try:
            payload = Path(_state.report_path).read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# -----------------------------------------------------------------------------
# Agregación + thresholds (compartido con CLI legacy)
# -----------------------------------------------------------------------------


def _aggregate(samples: list[dict[str, float]]) -> dict[str, object]:
    def med(name: str) -> tuple[float, float]:
        vals = np.array([s[name] for s in samples], dtype=np.float64)
        return float(np.median(vals)), float(vals.std())

    pitch_med, pitch_std = med("rotation_x_deg")
    yaw_med, yaw_std = med("rotation_y_deg")
    roll_med, roll_std = med("rotation_z_deg")
    off_y_med, off_y_std = med("offset_y_mm")
    off_z_med, off_z_std = med("offset_z_mm")

    checks = {
        "pitch":    abs(pitch_med) <= THR_PITCH_DEG,
        "yaw":      abs(yaw_med) <= THR_YAW_DEG,
        "roll":     abs(roll_med) <= THR_ROLL_DEG,
        "offset_y": abs(off_y_med) <= THR_OFFSET_Y_MM,
        "offset_z": abs(off_z_med) <= THR_OFFSET_Z_MM,
    }
    return {
        "samples": len(samples),
        "metrics": {
            "pitch": pitch_med, "pitch_std": pitch_std,
            "yaw": yaw_med, "yaw_std": yaw_std,
            "roll": roll_med, "roll_std": roll_std,
            "offset_y": off_y_med, "offset_y_std": off_y_std,
            "offset_z": off_z_med, "offset_z_std": off_z_std,
        },
        "checks": checks,
        "verdict": all(checks.values()),
    }


def _build_status_json() -> str:
    phase = _state.phase
    phase_class = {
        "waiting": "waiting",
        "collecting": "collecting",
        "done": "pass" if _state.final_verdict else "fail",
        "error": "fail",
    }.get(phase, "waiting")

    metrics_payload: dict[str, Optional[float]] = {}
    checks_payload: dict[str, Optional[bool]] = {}

    if _state.final_summary is not None:
        m = _state.final_summary["metrics"]
        metrics_payload = {
            "pitch": m["pitch"], "yaw": m["yaw"], "roll": m["roll"],
            "offset_y": m["offset_y"], "offset_z": m["offset_z"],
        }
        checks_payload = {k: bool(v) for k, v in _state.final_summary["checks"].items()}
    elif _state.live_metrics is not None:
        lm = _state.live_metrics
        metrics_payload = {
            "pitch": lm["rotation_x_deg"], "yaw": lm["rotation_y_deg"],
            "roll": lm["rotation_z_deg"],
            "offset_y": lm["offset_y_mm"], "offset_z": lm["offset_z_mm"],
        }
        checks_payload = {k: None for k in
                          ("pitch", "yaw", "roll", "offset_y", "offset_z")}

    data = {
        "phase": phase,
        "phase_class": phase_class,
        "message": _state.message,
        "samples": len(_state.samples),
        "target_samples": _state.target_samples,
        "metrics": metrics_payload,
        "checks": checks_payload,
        "thresholds": {
            "pitch": THR_PITCH_DEG, "yaw": THR_YAW_DEG, "roll": THR_ROLL_DEG,
            "offset_y": THR_OFFSET_Y_MM, "offset_z": THR_OFFSET_Z_MM,
        },
        "report_path": _state.report_path,
    }
    return json.dumps(data)


# -----------------------------------------------------------------------------
# Reporte HTML — proof-of-PASS por unidad para el flujo de ensamblado
# -----------------------------------------------------------------------------


def _pill(ok: bool) -> str:
    color = "#2ecc71" if ok else "#e74c3c"
    label = "PASS" if ok else "FAIL"
    return f'<span style="background:{color};color:#000;padding:2px 8px;border-radius:3px;font-weight:bold;font-size:11px">{label}</span>'


def _save_report(
    summary: dict[str, object],
    snapshot_jpeg: bytes,
    device_id: str,
    output_dir: Path,
) -> Path:
    """Genera y persiste un HTML self-contained con el verdict de
    una unidad bajo prueba. Pensado para audit trail de fabricación —
    cada unidad ensamblada queda con su reporte timestamped en disco.
    """
    ts = _dt.datetime.now()
    m: dict[str, float] = summary["metrics"]  # type: ignore
    checks: dict[str, bool] = summary["checks"]  # type: ignore
    verdict: bool = bool(summary["verdict"])

    snapshot_b64 = base64.b64encode(snapshot_jpeg).decode("ascii") if snapshot_jpeg else ""

    rows = []
    rows.append((
        "pitch (rotación X)", f"{m['pitch']:+.3f}°", f"{m['pitch_std']:.3f}°",
        f"±{THR_PITCH_DEG:.2f}°", checks["pitch"],
    ))
    rows.append((
        "yaw (rotación Y)", f"{m['yaw']:+.3f}°", f"{m['yaw_std']:.3f}°",
        f"±{THR_YAW_DEG:.2f}°", checks["yaw"],
    ))
    rows.append((
        "roll (rotación Z)", f"{m['roll']:+.3f}°", f"{m['roll_std']:.3f}°",
        f"±{THR_ROLL_DEG:.2f}°", checks["roll"],
    ))
    rows.append((
        "offset Y", f"{m['offset_y']:+.2f} mm", f"{m['offset_y_std']:.3f} mm",
        f"±{THR_OFFSET_Y_MM:.1f} mm", checks["offset_y"],
    ))
    rows.append((
        "offset Z", f"{m['offset_z']:+.2f} mm", f"{m['offset_z_std']:.3f} mm",
        f"±{THR_OFFSET_Z_MM:.1f} mm", checks["offset_z"],
    ))

    table_rows = "\n".join(
        f"<tr><td>{label}</td><td><b>{val}</b></td><td>{std}</td>"
        f"<td>{thr}</td><td>{_pill(ok)}</td></tr>"
        for (label, val, std, thr, ok) in rows
    )

    banner_color = "#2ecc71" if verdict else "#e74c3c"
    banner_text = "PASS — bracket dentro de tolerancia" if verdict else "FAIL — re-ajustar bracket y volver a medir"

    snapshot_html = (
        f'<h2>Preview en el momento del verdict</h2>'
        f'<img src="data:image/jpeg;base64,{snapshot_b64}" '
        f'style="max-width:100%;border:1px solid #333" />'
    ) if snapshot_b64 else ""

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Diagnóstico de bracket — {device_id}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; color: #222; }}
h1 {{ margin: 0; font-size: 22px; }}
h2 {{ margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
.banner {{ padding: 16px; margin: 16px 0; background: {banner_color}; color: #000; font-weight: bold; text-align: center; border-radius: 4px; font-size: 16px; }}
table {{ width: 100%; border-collapse: collapse; }}
td, th {{ padding: 6px 10px; text-align: left; font-family: monospace; font-size: 13px; border-bottom: 1px solid #eee; }}
th {{ background: #f4f4f4; }}
.meta {{ color: #666; font-size: 13px; margin-top: 12px; }}
</style></head><body>
<h1>Diagnóstico de bracket estéreo</h1>
<div class="meta">Unidad: <b>{device_id}</b> · Timestamp: <b>{ts:%Y-%m-%d %H:%M:%S}</b> · Muestras: {summary['samples']}</div>

<div class="banner">{banner_text}</div>

<h2>Métricas</h2>
<table>
<tr><th>Métrica</th><th>Mediana</th><th>Std</th><th>Threshold</th><th>Veredicto</th></tr>
{table_rows}
</table>

<p style="color:#666;font-size:12px;margin-top:8px">
Mediana: robusta a outliers de detección frame por frame. Std: variabilidad
cross-frame, idealmente &lt;0.1° / &lt;0.5mm cuando el board está estable y bien
iluminado. Thresholds matchean los del reporte del wizard de calibración —
bracket "bien armado" para QA de fabricación.
</p>

{snapshot_html}

<p class="meta">Generado por <b>diagnose_bracket.py</b> — people-counter edge system.</p>
</body></html>
"""
    safe_device = "".join(c if c.isalnum() or c in "-_" else "_" for c in device_id)
    fname = f"bracket_{safe_device}_{ts:%Y%m%d_%H%M%S}_{'PASS' if verdict else 'FAIL'}.html"
    path = output_dir / fname
    path.write_text(html, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Loop principal de captura — corre en el thread principal tras el start
# -----------------------------------------------------------------------------


def _capture_loop(
    args: argparse.Namespace,
    board: cv2.aruco.CharucoBoard,
    K: np.ndarray,
    output_dir: Path,
) -> None:
    from picamera2 import Picamera2

    w, h = int(args.resolution[0]), int(args.resolution[1])

    # Preview respetando aspect ratio del capture (sino el preview sale
    # estirado vertical cuando captura 16:9 → preview 4:3).
    preview_w = 640
    preview_h = int(round(preview_w * h / w))
    preview_size = (preview_w, preview_h)

    # Ghost del board target a board_distance_mm, frontal, centrado.
    # Precomputado una vez; el operador alinea el board físico al
    # outline naranja antes de Comenzar.
    ghost_pose = PoseTarget(
        id="frontal",
        label=f"Frontal a {args.board_distance_mm/1000:.2f}m",
        tvec_mm=(0.0, 0.0, float(args.board_distance_mm)),
    )
    ghost = project_pose(
        ghost_pose,
        board_size=(args.board_cols, args.board_rows),
        square_length=args.square_mm,
        preview_size=preview_size,
        focal_full_px=NOMINAL_FOCAL_PX * w / NOMINAL_FULL_RES[0],
        full_res=(w, h),
    )

    cam_l = Picamera2(args.left)
    cam_r = Picamera2(args.right)
    # Cap de exposure idéntico al pipeline runtime (vision.max_exposure_us
    # = 16ms en config.example.yaml). Sin este cap, en luz baja AE empuja
    # el shutter al máximo del framerate (~66ms) → cualquier vibración
    # micro (operador caminando cerca, piso/techo) se integra en 66ms y
    # produce ghosting que rompe el decoder ArUco asimétricamente entre
    # L y R (las dos cámaras vibran levemente distinto por estar en
    # sockets CSI separados). Con 16ms el shutter freezea esa vibración.
    # AE compensa con AnalogueGain más alto — más ruido por pixel pero
    # cero blur. Mismo trade-off que la pipeline runtime hace para
    # detector accuracy.
    max_exposure_us = int(args.max_exposure_us)
    initial_controls = {
        "FrameDurationLimits": (max_exposure_us, max_exposure_us),
    }
    for cam in [cam_l, cam_r]:
        cfg = cam.create_still_configuration(
            main={"size": (w, h), "format": "BGR888"},
            raw={"size": CANONICAL_RAW_SIZE},
            controls=initial_controls,
        )
        cam.configure(cfg)
        cam.start()
    # Lock inicial provisional con 2s de settle. La escena puede no
    # tener el board todavía (operador acaba de lanzar el script),
    # pero un lock estable evita la oscilación de AE auto durante
    # waiting. Cuando el operador apreta Comenzar, re-habilitamos AE
    # con el board ya en posición, esperamos otro settle, y re-
    # lockeamos para la captura — así los valores lockeados reflejan
    # la escena REAL de medición (recomendación: settle con board en
    # posición, después lock).
    time.sleep(2.0)
    for cam in [cam_l, cam_r]:
        m = cam.capture_metadata()
        cam.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": m.get("ExposureTime", 30000),
            "AnalogueGain": m.get("AnalogueGain", 1.0),
            "ColourGains": m.get("ColourGains", (1.0, 1.0)),
        })
    time.sleep(0.3)
    ae_locked = True
    logger.info(
        "Lock provisional tras 2s settle. L: exp=%dus gain=%.2f | R: exp=%dus gain=%.2f",
        cam_l.capture_metadata().get("ExposureTime", 0),
        cam_l.capture_metadata().get("AnalogueGain", 0),
        cam_r.capture_metadata().get("ExposureTime", 0),
        cam_r.capture_metadata().get("AnalogueGain", 0),
    )

    _state.message = "Posicioná el board y apretá Comenzar"

    frame_counter = 0
    try:
        while not _finish_requested.is_set():
            frame_counter += 1
            arr_l = cam_l.capture_array("main")
            arr_r = cam_r.capture_array("main")
            frame_l = cv2.cvtColor(arr_l, cv2.COLOR_RGB2BGR)
            frame_r = cv2.cvtColor(arr_r, cv2.COLOR_RGB2BGR)

            corners_l, ids_l = detect_charuco_dual_pass(frame_l, board)
            corners_r, ids_r = detect_charuco_dual_pass(frame_r, board)

            # Diagnóstico del flicker.
            nl = 0 if ids_l is None else len(ids_l)
            nr = 0 if ids_r is None else len(ids_r)
            # Log per-frame durante collecting: timestamp, exposures,
            # brillo, corner counts. Una línea por frame, fácil de leer.
            if _state.phase == "collecting":
                meta_l = cam_l.capture_metadata()
                meta_r = cam_r.capture_metadata()
                mean_l = float(np.mean(frame_l))
                mean_r = float(np.mean(frame_r))
                bad = (nl < 8 or nr < 8)
                tag = "BAD " if bad else "ok  "
                logger.info(
                    "[%s] f=%3d  L: corners=%2d brillo=%5.1f exp=%5dus gain=%4.2f | "
                    "R: corners=%2d brillo=%5.1f exp=%5dus gain=%4.2f",
                    tag, frame_counter,
                    nl, mean_l, meta_l.get("ExposureTime", 0),
                    meta_l.get("AnalogueGain", 0),
                    nr, mean_r, meta_r.get("ExposureTime", 0),
                    meta_r.get("AnalogueGain", 0),
                )
                # Guardá el frame de la 1ra falla a disco para inspección.
                if bad and not hasattr(_capture_loop, "_saved_bad"):
                    cv2.imwrite("/tmp/bracket_bad_l.jpg", frame_l)
                    cv2.imwrite("/tmp/bracket_bad_r.jpg", frame_r)
                    _capture_loop._saved_bad = True
                    logger.warning("Frame de falla guardado en /tmp/bracket_bad_{l,r}.jpg")

            pose_pair = None
            if corners_l is not None and corners_r is not None:
                # CRÍTICO: fitear solvePnP solo a los corners detectados
                # en AMBAS cámaras. Si L detecta 36 corners y R solo 21
                # (asimetría común por glare/distorsión periférica), los
                # dos PnP independientes fitean sets físicos distintos y
                # la composición R_LR, T_LR sale mal escalada (vimos
                # baseline 225mm cuando el real era 146mm). Con corners
                # comunes, los biases por NOMINAL K se cancelan en la
                # resta T_R - R·T_L.
                common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
                if len(common_ids) >= 8:
                    mask_l = np.isin(ids_l.flatten(), common_ids)
                    mask_r = np.isin(ids_r.flatten(), common_ids)
                    order_l = np.argsort(ids_l.flatten()[mask_l])
                    order_r = np.argsort(ids_r.flatten()[mask_r])
                    c_l_common = corners_l[mask_l][order_l]
                    c_r_common = corners_r[mask_r][order_r]
                    common_ids_sorted = np.sort(common_ids).reshape(-1, 1)
                    pose_l = _solve_board_pose(
                        c_l_common, common_ids_sorted, board, K,
                    )
                    pose_r = _solve_board_pose(
                        c_r_common, common_ids_sorted, board, K,
                    )
                    if pose_l is not None and pose_r is not None:
                        R_LR, T_LR = _compose_relative_pose(*pose_l, *pose_r)
                        _state.live_metrics = lens_alignment_metrics(R_LR, T_LR)
                        pose_pair = (pose_l, pose_r)

            # Dibujar preview: L | R con corners + ghost del board target.
            preview = _render_preview(
                frame_l, frame_r, corners_l, ids_l, corners_r, ids_r,
                ghost=ghost, preview_size=preview_size,
            )
            ok, jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                _state.latest_jpeg = jpg.tobytes()

            # Si estamos en fase de captura, acumular
            if _state.phase == "collecting" and pose_pair is not None:
                _state.samples.append(_state.live_metrics)
                if len(_state.samples) >= _state.target_samples:
                    summary = _aggregate(_state.samples)
                    _state.final_summary = summary
                    _state.final_verdict = bool(summary["verdict"])
                    _state.phase = "done"
                    _state.snapshot_jpeg = jpg.tobytes() if ok else b""
                    _state.message = (
                        f"VEREDICTO: {'PASS' if summary['verdict'] else 'FAIL'} "
                        f"({len(_state.samples)} muestras, "
                        f"{_state.skipped} skipped)"
                    )
                    logger.info("Veredicto: %s", _state.message)
                    # Generar y persistir reporte HTML — proof-of-PASS
                    # por unidad para el flujo de ensamblado.
                    try:
                        report_path = _save_report(
                            summary=summary,
                            snapshot_jpeg=_state.snapshot_jpeg,
                            device_id=args.device_id,
                            output_dir=output_dir,
                        )
                        _state.report_path = str(report_path)
                        logger.info("Reporte guardado: %s", report_path)
                    except Exception:
                        logger.exception("No pude generar el reporte HTML")
            elif _state.phase == "collecting" and pose_pair is None:
                _state.skipped += 1
                _state.message = (
                    f"Coleccionando... {len(_state.samples)}/"
                    f"{_state.target_samples} (board no detectado en uno o "
                    f"ambos lados)"
                )
            elif _state.phase == "collecting":
                _state.message = (
                    f"Coleccionando... {len(_state.samples)}/"
                    f"{_state.target_samples}"
                )

            # Transición waiting → collecting cuando el usuario clickea start.
            # Re-settle AE con el board ya posicionado, después re-lock —
            # así los valores lockeados reflejan la escena REAL de medición,
            # no la del startup (que pudo ser distinta).
            if _start_event.is_set() and _state.phase == "waiting":
                _state.message = "Re-settle AE con board en escena..."
                for cam in [cam_l, cam_r]:
                    cam.set_controls({"AeEnable": True, "AwbEnable": True})
                time.sleep(1.5)
                for cam in [cam_l, cam_r]:
                    m = cam.capture_metadata()
                    cam.set_controls({
                        "AeEnable": False, "AwbEnable": False,
                        "ExposureTime": m.get("ExposureTime", 30000),
                        "AnalogueGain": m.get("AnalogueGain", 1.0),
                        "ColourGains": m.get("ColourGains", (1.0, 1.0)),
                    })
                time.sleep(0.3)
                logger.info(
                    "Re-lock con board en escena. L: exp=%dus gain=%.2f | "
                    "R: exp=%dus gain=%.2f",
                    cam_l.capture_metadata().get("ExposureTime", 0),
                    cam_l.capture_metadata().get("AnalogueGain", 0),
                    cam_r.capture_metadata().get("ExposureTime", 0),
                    cam_r.capture_metadata().get("AnalogueGain", 0),
                )
                _state.phase = "collecting"
                _state.samples = []
                _state.skipped = 0
                _state.message = f"Coleccionando... 0/{_state.target_samples}"
                _start_event.clear()
                logger.info("Captura iniciada")
    finally:
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        _capture_done.set()


def _render_preview(
    frame_l: np.ndarray, frame_r: np.ndarray,
    corners_l: Optional[np.ndarray], ids_l: Optional[np.ndarray],
    corners_r: Optional[np.ndarray], ids_r: Optional[np.ndarray],
    ghost: Optional[dict[str, np.ndarray]] = None,
    preview_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Compone preview L|R con annotations de detección + ghost del board.

    preview_size: (W, H) por mitad, debe respetar el aspect ratio del
    frame de captura para no estirar/aplastar la imagen.
    """
    if preview_size is None:
        # Default 16:9 (matchea captura 1152×648). Si la captura es
        # otra resolución, el caller debería pasarse el preview_size
        # apropiado.
        preview_size = (640, 360)
    PREVIEW_W, PREVIEW_H = preview_size

    vis_l = cv2.resize(frame_l, (PREVIEW_W, PREVIEW_H))
    vis_r = cv2.resize(frame_r, (PREVIEW_W, PREVIEW_H))
    scale_x = PREVIEW_W / frame_l.shape[1]
    scale_y = PREVIEW_H / frame_l.shape[0]

    def _draw_corners(vis, corners, ids):
        if corners is None or ids is None:
            return
        for pt in corners.reshape(-1, 2):
            x = int(pt[0] * scale_x)
            y = int(pt[1] * scale_y)
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

    _draw_corners(vis_l, corners_l, ids_l)
    _draw_corners(vis_r, corners_r, ids_r)

    # Ghost del board target — SOLO en L (cámara de referencia). En
    # un par estéreo, la misma posición 3D proyecta a pixels distintos
    # en L vs R por la baseline (~40px a 1m con baseline 140mm), así
    # que dibujar el mismo ghost en ambas mitades sería ambiguo para
    # el operador. La regla: alinear el board físico al ghost de L;
    # en R va a aparecer levemente desplazado hacia la izquierda por
    # la disparidad — eso es lo esperado en un bracket bien armado.
    if ghost is not None:
        outer = ghost["outer_corners"].astype(np.int32)
        cv2.polylines(
            vis_l, [outer.reshape(-1, 1, 2)], isClosed=True,
            color=(0, 180, 255), thickness=2,
        )
        cx = int(np.mean(outer[:, 0]))
        cy = int(np.mean(outer[:, 1]))
        cv2.drawMarker(
            vis_l, (cx, cy), (0, 180, 255),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
        )
        cv2.putText(
            vis_l, "alinear board aqui", (cx - 70, cy - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(vis_l, "L", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_r, "R", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    nl = 0 if ids_l is None else len(ids_l)
    nr = 0 if ids_r is None else len(ids_r)
    cv2.putText(vis_l, f"{nl} corners", (8, PREVIEW_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(vis_r, f"{nr} corners", (8, PREVIEW_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    return np.hstack([vis_l, vis_r])


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnóstico de bracket estéreo (browser-driven). "
                    "Mide pitch/yaw/roll + offsets Y/Z entre cámaras sin "
                    "necesidad de calibración previa. Para verificar el "
                    "ensamblado físico del bracket antes de la calibración "
                    "completa.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        help="Resolución de captura. Default: lee "
                             "vision.resolution de /etc/people-counter/"
                             "config.yaml.")
    parser.add_argument("--frames", type=int, default=30,
                        help="Frames a agregar tras Comenzar. Default 30.")
    parser.add_argument("--board-cols", type=int, default=9)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=45.0)
    parser.add_argument("--marker-mm", type=float, default=33.0)
    parser.add_argument("--dict", dest="aruco_dict", default="DICT_4X4_100")
    parser.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--board-distance-mm", type=float, default=1000.0,
                        help="Distancia objetivo del board para el ghost del "
                             "preview (mm). Default 1000 (1m). Subir/bajar "
                             "si el setup de QC tiene espacio distinto.")
    parser.add_argument("--device-id", default="unknown",
                        help="ID de la unidad bajo prueba, va al header del "
                             "reporte HTML. Default 'unknown' — pasalo por "
                             "barcode scanner o serial en la línea.")
    parser.add_argument("--output-dir", default="./bracket_reports",
                        help="Directorio donde se guardan los reportes HTML. "
                             "Default ./bracket_reports (relativo al CWD).")
    parser.add_argument("--max-exposure-us", type=int, default=16000,
                        help="Cap del shutter time en microsegundos (default "
                             "16000 = 16ms, matchea vision.max_exposure_us "
                             "del runtime). Sin cap, AE en luz baja sube "
                             "shutter a ~66ms y vibraciones micro hacen "
                             "ghosting asimétrico L/R que rompe el decoder "
                             "ArUco. 16ms freezea cualquier vibración a "
                             "costa de más gain (más noisy pero crisp).")
    args = parser.parse_args()

    if args.resolution is None:
        try:
            from src.config.loader import (
                DEFAULT_DEVICE_CONFIG_PATH, load_device_config,
            )
            cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
            res = cfg.get("vision", {}).get("resolution")
            if res and isinstance(res, list) and len(res) == 2:
                args.resolution = [int(res[0]), int(res[1])]
        except Exception:
            pass
        if args.resolution is None:
            parser.error("--resolution no provisto y no se pudo leer del config")

    dict_attr = (
        args.aruco_dict if args.aruco_dict.startswith("DICT_")
        else f"DICT_{args.aruco_dict}"
    )
    dict_id = getattr(cv2.aruco, dict_attr)
    board = create_charuco_board(
        board_size=(args.board_cols, args.board_rows),
        square_length=args.square_mm,
        marker_length=args.marker_mm,
        dict_id=dict_id,
        legacy_pattern=args.legacy_pattern,
    )

    w, h = int(args.resolution[0]), int(args.resolution[1])

    # K nominal del IMX708 (focal escalada desde NOMINAL_FOCAL_PX al
    # res de captura, principal point centrado). D=0 implícito en
    # _solve_board_pose — sobre fisheye introduce bias en magnitudes
    # absolutas (por eso no reportamos baseline) pero los biases son
    # razonablemente simétricos entre L y R y cancelan en la
    # composición R_LR/T_LR para rotaciones y offsets Y/Z. Accuracy
    # esperada: ±0.5° rotaciones, ±10mm offsets — suficiente para
    # detectar brackets mal armados.
    K = _build_nominal_k(w, h)
    _state.target_samples = int(args.frames)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info("Diagnóstico de bracket — http://people-counter.local:%d", args.port)
    logger.info("Posicioná el board ChArUco a ~1m, centrado en ambos lados, y apretá Comenzar.")

    try:
        _capture_loop(args, board, K, output_dir)
    except KeyboardInterrupt:
        logger.info("Interrumpido (Ctrl-C)")
    finally:
        _finish_requested.set()
        server.shutdown()

    # Exit code: 0 si PASS, 1 si FAIL, 2 si nunca llegó a coleccionar.
    if _state.phase == "done":
        sys.exit(0 if _state.final_verdict else 1)
    sys.exit(2)


if __name__ == "__main__":
    main()

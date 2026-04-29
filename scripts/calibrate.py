#!/usr/bin/env python3
"""CLI tool for stereo camera calibration using ChArUco patterns.

Usage:
    # Step 1: Capture image pairs (interactive, with HTTP preview)
    python scripts/calibrate.py capture --count 30

    # Step 2: Run calibration from captured pairs
    python scripts/calibrate.py calibrate --input-dir ./calibration/captures --output calibration.npz

    # Step 3: Verify calibration (draw epipolar lines on rectified pair)
    python scripts/calibrate.py verify --calibration calibration.npz --input-dir ./calibration/captures
"""

import argparse
import datetime as _dt
import json
import logging
import math
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.vision.calibration as _calib_mod
from src.vision.calibration import (
    ALIGN_CENTER_TOL_PX,
    ALIGN_MEAN_ERR_TOL_PX_LOOSE,
    ALIGN_MEAN_ERR_TOL_PX_TIGHT,
    ALIGN_MATCHED_MIN_LOOSE,
    ALIGN_MATCHED_MIN_TIGHT,
    NOMINAL_FOCAL_PX,
    DEFAULT_BOARD_SIZE,
    DEFAULT_DIST_FAR_MM,
    DEFAULT_DIST_MID_MM,
    DEFAULT_DIST_NEAR_MM,
    DEFAULT_MARKER_LENGTH,
    DEFAULT_SQUARE_LENGTH,
    StabilityTracker,
    alignment_hint_by_corners,
    analyze_pose_coverage,
    assess_frame_quality,
    calibrate_stereo,
    compute_alignment_by_corners,
    compute_per_pair_residuals,
    count_common_corners,
    create_charuco_board,
    default_pose_sequence,
    detect_charuco_corners,
    fit_single_camera_intrinsics,
    generate_board_image,
    is_aligned_by_corners,
    live_lighting_warnings,
    load_calibration,
    project_pose,
    rectify_pair,
    save_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate")


def _resolve_aruco_dict(name: str) -> int:
    """Resolve a user-provided ArUco dict name to the cv2 constant."""
    dict_attr = name if name.startswith("DICT_") else f"DICT_{name}"
    if not hasattr(cv2.aruco, dict_attr):
        raise SystemExit(f"ArUco dict desconocido: {name}")
    return getattr(cv2.aruco, dict_attr)


_TOLERANCE_PRESETS = {
    "loose": (50.0, 25.0),
    "normal": (ALIGN_MEAN_ERR_TOL_PX_LOOSE, ALIGN_MEAN_ERR_TOL_PX_TIGHT),
    "strict": (15.0, 8.0),
}


def _apply_low_light_overrides() -> None:
    """Relax frame-quality gates for PoC runs in low-light / small-room scenes.

    Mutates the constants that ``assess_frame_quality`` reads. NOT for
    production calibration — pairs accepted under these thresholds will have
    poor SNR and the resulting .npz won't survive ground-truth verification.
    """
    _calib_mod.QUALITY_MIN_CORNERS = 8
    _calib_mod.QUALITY_MIN_BLUR = 3.0
    _calib_mod.QUALITY_MIN_EXPOSURE = 5.0
    _calib_mod.QUALITY_MIN_CORNER_SHARPNESS = 5.0
    _calib_mod.QUALITY_MAX_LR_BRIGHTNESS_PCT = 70.0
    logger.warning(
        "Low-light mode enabled — quality gates relaxed (exposure/blur/"
        "corner-sharp/LR-balance). PoC only, do NOT trust the resulting "
        "calibration for depth.",
    )


def _resolve_tolerance(args: argparse.Namespace) -> tuple[float, float]:
    """Return (loose_px, tight_px). Per-flag overrides beat the preset."""
    preset = getattr(args, "tolerance", "normal")
    base_loose, base_tight = _TOLERANCE_PRESETS.get(
        preset, _TOLERANCE_PRESETS["normal"],
    )
    override_loose = getattr(args, "align_tol_loose_px", None)
    override_tight = getattr(args, "align_tol_tight_px", None)
    loose = override_loose if override_loose is not None else base_loose
    tight = override_tight if override_tight is not None else base_tight
    return loose, tight


def _ask_operator_ui(prompt_type: str, message: str, options: dict = None) -> str:
    """Render an interactive prompt in the browser and block until /wizard-input
    receives the operator's answer. Returns the raw string the operator sent.

    prompt_type drives what controls the JS renders:
      * "diversity" → two buttons: Continuar / Cancelar
      * "ground_truth" → input-text + Validar / Saltear buttons
    """
    global _post_capture_html, _post_capture_active, _wizard_input_value
    _post_capture_active = True
    _wizard_input_event.clear()
    _wizard_input_value = ""

    if prompt_type == "diversity":
        html = (
            f'<div data-phase="prompt" data-prompt-type="diversity">'
            f'<div style="color:#f1c40f;font-size:20px;font-weight:700;'
            f'margin-bottom:10px">Confirmación requerida</div>'
            f'<div style="color:#eee;font-size:14px;line-height:1.6;'
            f'white-space:pre-line">{message}</div>'
            f'</div>'
        )
    elif prompt_type == "ground_truth":
        html = (
            f'<div data-phase="prompt" data-prompt-type="ground_truth">'
            f'<div style="color:#3fb6f0;font-size:20px;font-weight:700;'
            f'margin-bottom:10px">Validación ground-truth (opcional)</div>'
            f'<div style="color:#eee;font-size:14px;line-height:1.6;'
            f'white-space:pre-line">{message}</div>'
            f'</div>'
        )
    else:
        html = f'<div data-phase="prompt">{message}</div>'

    with _post_capture_lock:
        _post_capture_html = html

    _wizard_input_event.wait()
    return _wizard_input_value


def _set_post_capture_phase(
    phase: str, message: str, progress_pct: int | None = None,
    verdict: str | None = None, report_available: bool = False,
) -> None:
    """Push a progress message to the browser panel while the wizard runs
    through post-capture phases. Embedded data-phase attribute lets the JS
    detect completion and auto-open the report.
    """
    global _post_capture_html, _post_capture_active
    _post_capture_active = True
    if phase == "complete":
        verdict_color = "#2ecc71" if verdict == "PASS" else "#e74c3c"
        report_btn = ""
        if report_available:
            report_btn = (
                '<a href="/report" target="_blank" '
                'style="display:inline-block;margin-top:14px;padding:10px 18px;'
                'background:#2980b9;color:#fff;text-decoration:none;'
                'border-radius:6px;font-weight:600;font-size:14px">'
                'Abrir reporte en nueva pestaña</a>'
            )
        has_report_attr = '1' if report_available else '0'
        html = (
            f'<div data-phase="complete" data-has-report="{has_report_attr}">'
            f'<div style="color:{verdict_color};font-size:22px;font-weight:700;'
            f'margin-bottom:12px">Calibración finalizada'
            + (f' — {verdict}' if verdict else '')
            + f'</div>'
            f'<div style="color:#aaa;font-size:13px;line-height:1.6">'
            f'{message}</div>'
            f'{report_btn}'
            f'</div>'
        )
    else:
        progress_html = ""
        if progress_pct is not None:
            progress_html = (
                f'<div style="height:6px;background:#222;border-radius:3px;'
                f'overflow:hidden;margin:10px 0 6px">'
                f'<div style="height:100%;background:#3fb6f0;width:{progress_pct}%;'
                f'transition:width .3s"></div></div>'
                f'<div style="color:#888;font-size:12px">{progress_pct}%</div>'
            )
        html = (
            f'<div data-phase="{phase}">'
            f'<div style="color:#f1c40f;font-size:18px;font-weight:600;'
            f'margin-bottom:6px">Procesando calibración</div>'
            f'<div style="color:#eee;font-size:15px;line-height:1.6">'
            f'{message}</div>'
            f'{progress_html}'
            f'</div>'
        )
    with _post_capture_lock:
        _post_capture_html = html

# ---------------------------------------------------------------------------
# HTTP preview for capture
# ---------------------------------------------------------------------------

_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()
_shutting_down = False
_trigger_armed = False
_manual_enabled = False


class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        global _trigger_armed
        if self.path == "/capture" and _manual_enabled:
            _trigger_armed = True
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            btn = (
                b'<button id="cap" onclick="fetch(\'/capture\',{method:\'POST\'})"'
                b' style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);'
                b'padding:16px 32px;font-size:20px;background:#28a745;color:white;'
                b'border:none;border-radius:8px;cursor:pointer">CAPTURE</button>'
                if _manual_enabled else b""
            )
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Calibration Capture</title>
<style>body{background:#111;margin:0;display:flex;justify-content:center;
align-items:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/stream">""" + btn + b"""</body></html>""")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _shutting_down:
                    with _jpeg_lock:
                        frame = _latest_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.15)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def _update_preview(jpeg_bytes: bytes) -> None:
    global _latest_jpeg
    with _jpeg_lock:
        _latest_jpeg = jpeg_bytes


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------


GRID_RECTANGULAR = np.ones((4, 5), dtype=np.int32)

GRID_CIRCULAR = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
], dtype=np.int32)


def _compute_coverage_center(
    corners: np.ndarray, w: int, h: int, grid_mask: np.ndarray,
) -> tuple[int, int]:
    """Get grid cell (row, col) for the center of detected corners."""
    cx = np.mean(corners[:, 0, 0])
    cy = np.mean(corners[:, 0, 1])
    grid_rows, grid_cols = grid_mask.shape
    col = int(cx / w * grid_cols)
    row = int(cy / h * grid_rows)
    col = max(0, min(col, grid_cols - 1))
    row = max(0, min(row, grid_rows - 1))
    return row, col


def _draw_coverage(
    frame: np.ndarray, coverage: np.ndarray, grid_mask: np.ndarray,
    cell_target: np.ndarray | None = None,
) -> None:
    """Draw coverage grid overlay on frame."""
    h, w = frame.shape[:2]
    grid_rows, grid_cols = coverage.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            if grid_mask[r, c] == 0:
                # Inactive cell — dim overlay
                overlay = frame[y1:y2, x1:x2].copy()
                dark = np.full_like(overlay, (0, 0, 0))
                cv2.addWeighted(dark, 0.5, overlay, 0.5, 0, frame[y1:y2, x1:x2])
                continue
            target = cell_target[r, c] if cell_target is not None else 0
            is_full = target > 0 and coverage[r, c] >= target
            if coverage[r, c] > 0:
                overlay = frame[y1:y2, x1:x2].copy()
                tint = (0, 80, 0) if is_full else (0, 60, 60)
                cv2.addWeighted(np.full_like(overlay, tint), 0.3, overlay, 0.7, 0, frame[y1:y2, x1:x2])
            label = f"{int(coverage[r, c])}/{target}" if target > 0 else str(int(coverage[r, c]))
            color = (0, 255, 0) if is_full else (0, 200, 255)
            cv2.putText(frame, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Guided capture: ghost-based aiming with auto-capture
# ---------------------------------------------------------------------------

GUIDED_PREVIEW = (1296, 486)  # combined preview: 2x (648, 486)
GUIDED_HALF = (648, 486)  # each camera's preview half
STABILITY_HOLD_SEC = 1.5
SKIP_POSE_TIMEOUT_SEC = 180.0

# Bootstrap: first N captures use loose tolerance + nominal intrinsics; after
# that we fit per-sensor intrinsics from the captured left frames and use them
# for ghost projection with tight tolerance.
BOOTSTRAP_COUNT = 6
LR_SYNC_MAX_DELTA_NS = 250_000_000  # 250 ms. Software sync via two picamera2
# instances drifts further than the original 80ms target on real hardware
# (60-120 ms typical, occasional 200ms spikes when CPU is busy). Hardware-
# triggered sync would be needed for tight bounds. Board is static during
# a calibration capture so the temporal delta doesn't affect the result —
# a generous ceiling here just keeps captures from being wrongly rejected.
LR_MIN_COMMON_CORNERS = 15

# Ambient drift: compare median brightness every ~30 frames against baseline.
DRIFT_BASELINE_FRAMES = 10
DRIFT_CHECK_EVERY_FRAMES = 30
DRIFT_WARN_PCT = 25.0


class _GuidedState:
    """Shared mutable state between capture loop and HTTP handler."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_pose_idx = 0
        self.pose_status: list[str] = []  # "pending" / "captured" / "skipped"
        self.captured_pairs: list[tuple[Path, Path, str]] = []  # (l, r, pose_id)
        self.rms_text: str = "—"
        self.rms_color: str = "#888"
        self.undo_requested = False
        self.skip_requested = False
        self.finish_requested = False
        self.status_html: str = ""
        self.banner_text: str = ""
        self.banner_color: str = "#444"
        self.hold_progress: float = 0.0  # 0..1 during stability countdown
        self.bootstrap_done = False
        self.fitted_K: Optional[np.ndarray] = None
        self.drift_warning: Optional[str] = None
        # A short event key that the audio layer in the browser will speak
        # when it changes. Updated each tick — browser throttles actual speech.
        self.audio_event: str = ""
        self.audio_event_seq: int = 0
        # Last hint we actually spoke (digit-stripped key). Separate from
        # banner_text so we don't "remember" saying hints that were
        # suppressed during the pose-announcement lockout.
        self.last_spoken_key: str = ""
        # Full text of the last spoken hint. Used to override the live
        # banner so what the operator reads matches what they just heard
        # (otherwise 1-cm fluctuations between emit and display sneak in).
        self.locked_alignment_text: str = ""


_guided_state: Optional[_GuidedState] = None

# Set to True when the operator clicks "Comenzar" in the browser. Blocks
# the main capture loop until that happens so (a) the operator has a
# chance to position themselves and (b) the click unlocks the browser
# audio context for TTS / beeps.
_capture_started = False
_capture_started_lock = threading.Lock()

# After guided capture finishes, the same HTTP server stays alive and shows
# wizard-phase progress / finalised screen. Updated from the main wizard
# thread as phases complete.
_post_capture_html: str = ""
_post_capture_lock = threading.Lock()
_post_capture_active = False
_report_path_for_http: Optional[Path] = None
_guided_server: Optional[ThreadingHTTPServer] = None

# Operator input shuttle: the wizard can block on _ask_operator_ui() which
# renders a prompt in the browser and waits for /wizard-input POST.
_wizard_input_event = threading.Event()
_wizard_input_value: str = ""

# True while the browser is narrating a pose announcement. Set by the
# _pose_announce emission path; cleared by /announce-done which the JS
# hits on SpeechSynthesisUtterance.onend (or immediately if audio is off).
# Captures and other audio emissions gate on this so the three-piece
# "number / label / distance" block is never interrupted.
_announce_pending = False


class _GuidedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if _guided_state is None:
            self.send_response(503); self.end_headers(); return
        if self.path == "/undo":
            with _guided_state.lock:
                _guided_state.undo_requested = True
            self.send_response(204); self.end_headers()
        elif self.path == "/skip":
            with _guided_state.lock:
                _guided_state.skip_requested = True
            self.send_response(204); self.end_headers()
        elif self.path == "/finish":
            with _guided_state.lock:
                _guided_state.finish_requested = True
            self.send_response(204); self.end_headers()
        elif self.path == "/start":
            global _capture_started
            with _capture_started_lock:
                _capture_started = True
            self.send_response(204); self.end_headers()
        elif self.path == "/wizard-input":
            global _wizard_input_value
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
            _wizard_input_value = raw
            _wizard_input_event.set()
            self.send_response(204); self.end_headers()
        elif self.path == "/announce-done":
            global _announce_pending
            _announce_pending = False
            self.send_response(204); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_guided_html().encode("utf-8"))
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _shutting_down:
                    with _jpeg_lock:
                        frame = _latest_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if _post_capture_active:
                with _post_capture_lock:
                    self.wfile.write(_post_capture_html.encode("utf-8"))
            elif _guided_state is not None:
                with _guided_state.lock:
                    self.wfile.write(_guided_state.status_html.encode("utf-8"))
        elif self.path == "/report":
            if _report_path_for_http is None or not _report_path_for_http.exists():
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write("Reporte no disponible aún".encode("utf-8"))
                return
            try:
                body = _report_path_for_http.read_bytes()
            except OSError as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error leyendo reporte: {e}".encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def _guided_html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Asistente de calibración</title>
<style>
  *{box-sizing:border-box}
  body{background:#0b0b0d;margin:0;color:#eee;
       font-family:-apple-system,Segoe UI,Roboto,sans-serif;
       display:flex;flex-direction:column;min-height:100vh}
  header{display:flex;align-items:center;justify-content:space-between;
         padding:10px 20px;background:#141418;border-bottom:1px solid #26262c}
  header h1{margin:0;font-size:16px;font-weight:600;letter-spacing:.3px}
  header .sub{color:#888;font-size:13px}
  #stream{flex:0 0 auto;max-width:100vw;max-height:58vh;object-fit:contain;
          display:block;margin:0 auto;background:#000;
          border-radius:6px;box-shadow:0 2px 16px rgba(0,0,0,0.6)}
  #stage{padding:12px;background:#000}
  #panel{flex:1;padding:14px 20px;overflow-y:auto;background:#141418;
         border-top:1px solid #26262c}
  #banner{padding:12px;border-radius:8px;text-align:center;font-size:20px;
          font-weight:600;margin-bottom:12px}
  #progress-bar{height:10px;background:#222;border-radius:5px;
                overflow:hidden;margin:6px 0 14px}
  #progress-fill{height:100%;background:#2ecc71;
                 transition:width .15s linear;width:0%}
  .row{display:flex;gap:10px;margin:8px 0;flex-wrap:wrap}
  .btn{padding:10px 18px;font-size:15px;border:none;border-radius:8px;
       cursor:pointer;font-weight:600;transition:background .15s;color:#fff}
  .btn-undo{background:#e67e22}
  .btn-undo:hover{background:#c76e1b}
  .btn-skip{background:#555}
  .btn-skip:hover{background:#444}
  .btn-finish{background:#c0392b}
  .btn-finish:hover{background:#a83224}
  .btn-audio{background:#555}
  .btn-audio:hover{background:#444}
  .btn-audio.on{background:#27ae60}
  .btn-audio.on:hover{background:#1f8c4e}
  .stat{font-size:14px;line-height:1.6}
  .stat b{color:#7fdbff}
  .warn{color:#f1c40f;font-weight:600}
  .rms{font-size:13px;display:inline-block;padding:3px 10px;
       border-radius:12px;color:#000}
  .pill-phase{font-size:12px;padding:2px 10px;border-radius:10px;
              background:#34495e;color:#ecf0f1;margin-left:8px}
  @keyframes spin { to { transform: rotate(360deg); } }
</style></head>
<body>
  <header>
    <h1>Asistente de calibración</h1>
    <span class="sub">Captura guiada ChArUco — stereo IMX708</span>
  </header>
  <div id="start-overlay" style="position:fixed;inset:0;background:rgba(11,11,13,0.95);
       z-index:9999;display:flex;align-items:center;justify-content:center;
       flex-direction:column;gap:18px">
    <div style="color:#eee;font-size:26px;font-weight:700">
      Asistente de calibración listo
    </div>
    <div style="color:#aaa;font-size:15px;max-width:480px;text-align:center;line-height:1.6">
      Posicioná el board ChArUco frente a las cámaras. Cuando esté listo,
      presioná <b>Comenzar</b>. Esto también activa el audio del navegador
      para las indicaciones de voz.
    </div>
    <button id="btn-start" style="padding:14px 36px;font-size:18px;
         background:#27ae60;color:#fff;border:none;border-radius:10px;
         cursor:pointer;font-weight:700" onclick="startCapture()">
      Comenzar
    </button>
  </div>
  <div id="stage"><img id="stream" src="/stream"/></div>
  <div id="panel">
    <div id="banner" style="background:#333">Esperando...</div>
    <div id="progress-bar"><div id="progress-fill"></div></div>
    <div id="status" class="stat">Conectando...</div>
    <div class="row">
      <button id="btn-audio" class="btn btn-audio" onclick="toggleAudio()">Audio OFF</button>
      <button class="btn btn-undo" onclick="flushSpeech();post('/undo')">Deshacer última</button>
      <button class="btn btn-skip" onclick="flushSpeech();post('/skip')">Saltear pose</button>
      <button class="btn btn-finish" onclick="if(confirm('Finalizar captura?')){flushSpeech();post('/finish')}">Finalizar</button>
    </div>
  </div>
<script>
function post(p){fetch(p,{method:'POST'})}
// Set after the operator submits a prompt response. While true, the
// /status poll is NOT allowed to re-inject the prompt controls — we
// already replaced them with a spinner and any phase change will
// clear the flag.
let promptSubmitted = false;
function showSpinner(msg){
  const status = document.getElementById('status');
  if(!status) return;
  status.innerHTML =
      '<div style="color:#3fb6f0;font-size:18px;font-weight:600;'
      + 'margin-bottom:10px">'
      + '<span class="spinner" style="display:inline-block;width:16px;'
      + 'height:16px;border:2px solid #3fb6f0;border-top-color:transparent;'
      + 'border-radius:50%;vertical-align:middle;margin-right:10px;'
      + 'animation:spin 0.9s linear infinite"></span>'
      + msg + '</div>';
}
function sendInput(val){
  promptSubmitted = true;
  showSpinner('Procesando...');
  fetch('/wizard-input', {method:'POST', body: val, headers: {'Content-Type':'text/plain'}});
}
function submitGt(){
  const el = document.getElementById('gt-dist');
  const val = el ? el.value : '';
  promptSubmitted = true;
  showSpinner(val
      ? 'Capturando y analizando profundidad... (puede tardar hasta 15 segundos)'
      : 'Salteando validación...');
  fetch('/wizard-input', {method:'POST', body: val, headers: {'Content-Type':'text/plain'}});
}
function flushSpeech(){
  // Clear any queued-up pose announcements so they don't trail into the
  // next phase (e.g. "pose 9" still pending when capture is already done).
  if (window.speechSynthesis) {
    try { window.speechSynthesis.cancel(); } catch(e){}
  }
  // Also tell the backend the announce is "done" — if we cancelled mid-
  // utterance, its onend won't fire and the server would stay locked.
  try { fetch('/announce-done', {method:'POST'}); } catch(e){}
}
function startCapture(){
  // User gesture — unlock the AudioContext for beeps + TTS.
  ensureAudioCtx();
  if (audioOn && window.speechSynthesis) {
    try {
      const u = new SpeechSynthesisUtterance('Comenzando calibración');
      u.lang = 'es-ES';
      window.speechSynthesis.speak(u);
    } catch(e){}
  }
  const overlay = document.getElementById('start-overlay');
  if (overlay) overlay.style.display = 'none';
  fetch('/start',{method:'POST'});
}
// Default ON; operator can turn it off and we remember that preference.
let audioOn = localStorage.getItem('guided.audio') !== '0';
let lastSpokenSeq = -1;
let lastSpokenAt = 0;
// Beep state machine: track previous hold-progress bucket + last capture count
let lastBeepBucket = -1;  // 0..3 = tick buckets at 0/33/66% of hold
let lastCapturedN = 0;
let audioCtx = null;
function ensureAudioCtx(){
  if (!audioCtx && window.AudioContext) {
    try { audioCtx = new AudioContext(); } catch(e){}
  }
  return audioCtx;
}
function beep(freq, durMs, gain){
  const ctx = ensureAudioCtx();
  if (!ctx) return;
  try {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = 'sine'; osc.frequency.value = freq;
    g.gain.value = gain || 0.15;
    osc.connect(g); g.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + durMs/1000);
  } catch(e){}
}
function updateAudioBtn(){
  const b = document.getElementById('btn-audio');
  b.textContent = audioOn ? 'Audio ON' : 'Audio OFF';
  b.classList.toggle('on', audioOn);
}
function toggleAudio(){
  audioOn = !audioOn;
  localStorage.setItem('guided.audio', audioOn ? '1' : '0');
  updateAudioBtn();
  if (audioOn) { ensureAudioCtx(); speak('Audio activado'); beep(800, 80); }
  else {
    // Cancel in-flight speech + queue so the OFF is immediate. Also unblocks
    // the backend if a pose announce is in progress — flushSpeech POSTs
    // /announce-done since onend won't fire after cancel().
    flushSpeech();
  }
}
function speak(text){
  // Queue utterances instead of cancelling — operator hears one then the
  // next. Our 3s dedup on identical hints keeps the queue short.
  if (!audioOn || !window.speechSynthesis) return;
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES'; u.rate = 1.05;
    window.speechSynthesis.speak(u);
  } catch(e){}
}
updateAudioBtn();
let postCaptureMode = false;
let finalizedSeen = false;
let lastRenderedPhase = '';
let lastRenderedPromptType = '';
async function refresh(){
  try {
    const r = await fetch('/status');
    const html = await r.text();
    // Post-capture modes (processing / finalised) identified via data-phase.
    const phaseMatch = html.match(/data-phase="([^"]+)"/);
    if (phaseMatch) {
      postCaptureMode = true;
      const thisPhase = phaseMatch[1];
      const thisPromptType = (html.match(/data-prompt-type="([^"]+)"/) || [,''])[1];
      // Re-render only when the phase OR prompt type changes. Otherwise
      // we'd wipe the ground-truth <input> every poll and steal focus.
      const shouldRender = (
        thisPhase !== lastRenderedPhase ||
        thisPromptType !== lastRenderedPromptType
      );
      if (shouldRender) {
        document.getElementById('status').innerHTML = html;
        lastRenderedPhase = thisPhase;
        lastRenderedPromptType = thisPromptType;
        // Phase actually changed — clear any pending prompt state so
        // subsequent prompts can render cleanly.
        promptSubmitted = false;
        // Entering any post-capture phase drops queued pose announcements.
        flushSpeech();
      }
      // Hide the live stream once capture is done.
      const stage = document.getElementById('stage');
      if (stage) stage.style.display = 'none';
      // Hide the action row (undo/skip/finish), replace banner with dynamic.
      const btns = document.querySelectorAll('#panel .row button');
      btns.forEach(b => b.style.display = 'none');
      const banner = document.getElementById('banner');
      if (banner) banner.style.display = 'none';
      const bar = document.getElementById('progress-bar');
      if (bar) bar.style.display = 'none';
      // Interactive prompt from the wizard: show controls that POST back
      // to /wizard-input. Only inject on first sight AND while the operator
      // hasn't submitted yet — after submitting we keep the spinner.
      if (phaseMatch[1] === 'prompt'
          && !document.getElementById('prompt-ctrls')
          && !promptSubmitted) {
        const typeMatch = html.match(/data-prompt-type="([^"]+)"/);
        const type = typeMatch ? typeMatch[1] : '';
        const ctrls = document.createElement('div');
        ctrls.id = 'prompt-ctrls';
        ctrls.style.cssText = 'display:flex;gap:10px;margin-top:14px;flex-wrap:wrap';
        const mkBtn = (text, style, onClick) => {
          const b = document.createElement('button');
          b.className = 'btn';
          b.style.cssText = style;
          b.textContent = text;
          b.addEventListener('click', onClick);
          return b;
        };
        if (type === 'diversity') {
          ctrls.appendChild(mkBtn(
              'Cancelar y recapturar',
              'width:auto;flex:1;background:#c0392b',
              () => sendInput('no')
          ));
          ctrls.appendChild(mkBtn(
              'Continuar igual',
              'width:auto;flex:1;background:#27ae60',
              () => sendInput('si')
          ));
        } else if (type === 'ground_truth') {
          const input = document.createElement('input');
          input.id = 'gt-dist';
          // type=text + inputmode=numeric avoids the native spinner UI
          // while still bringing up the numeric keyboard on phones.
          input.type = 'text';
          input.inputMode = 'numeric';
          input.pattern = '[0-9]*';
          input.placeholder = 'Distancia en mm (ej 2000)';
          input.style.cssText = 'flex:1 1 160px;padding:10px;font-size:15px;'
              + 'border-radius:6px;border:1px solid #555;background:#1a1a1e;color:#eee';
          ctrls.appendChild(input);
          ctrls.appendChild(mkBtn(
              'Validar',
              'width:auto;flex:0 0 auto;background:#3fb6f0',
              submitGt
          ));
          ctrls.appendChild(mkBtn(
              'Saltear',
              'width:auto;flex:0 0 auto;background:#555',
              () => sendInput('')
          ));
        }
        document.getElementById('status').appendChild(ctrls);
      }
      if (phaseMatch[1] === 'complete' && !finalizedSeen) {
        finalizedSeen = true;
        // Only auto-open the report when one actually exists (success path).
        const hasReport = /data-has-report="1"/.test(html);
        if (hasReport) {
          const w = window.open('/report', '_blank');
          if (!w) {
            const note = document.createElement('div');
            note.style.cssText =
                'color:#f1c40f;font-size:12px;margin-top:8px';
            note.textContent =
                'El navegador bloqueó el popup — usá el botón "Abrir reporte".';
            document.getElementById('status').appendChild(note);
          }
        }
      }
      return;
    }
    document.getElementById('status').innerHTML = html;
    // Detect pose change and flush the speech queue so the new pose
    // announcement starts cleanly — without any residue from the
    // previous pose's leftover utterances draining first.
    const poseIdxMatch = html.match(/data-pose-idx="([^"]+)"/);
    if (poseIdxMatch) {
      const idx = poseIdxMatch[1];
      if (window._lastPoseIdx !== undefined && window._lastPoseIdx !== idx) {
        flushSpeech();
      }
      window._lastPoseIdx = idx;
    }
    const m = html.match(/data-banner="([^"]*)" data-color="([^"]*)" data-progress="([^"]*)" data-audioseq="([^"]*)" data-audiotext="([^"]*)" data-captured="([^"]*)"/);
    if (m) {
      document.getElementById('banner').textContent = m[1];
      document.getElementById('banner').style.background = m[2];
      const progress = parseFloat(m[3]);
      document.getElementById('progress-fill').style.width = (progress*100)+'%';
      const seq = parseInt(m[4], 10);
      const audioText = m[5];
      const capturedN = parseInt(m[6], 10) || 0;
      const now = Date.now();
      // TTS: speak every new seq. The backend already dedupes by semantic
      // key (digits stripped), so a seq change means a genuinely new hint
      // worth speaking. We rely on the queue instead of a time-based
      // throttle — so nothing gets silently dropped during fast captures.
      if (seq !== lastSpokenSeq) {
        lastSpokenSeq = seq;
        lastSpokenAt = now;
        // Pose announcements get special handling: they're the "atomic
        // block" (number/label/distance) and the backend waits on the
        // onend signal before unlocking capture or other hints.
        const isPoseAnnounce = audioText && audioText.startsWith('Pose ');
        if (audioOn && audioText) {
          if (isPoseAnnounce && window.speechSynthesis) {
            try {
              const u = new SpeechSynthesisUtterance(audioText);
              u.lang = 'es-ES'; u.rate = 1.05;
              u.onend = () => fetch('/announce-done', {method:'POST'});
              u.onerror = () => fetch('/announce-done', {method:'POST'});
              window.speechSynthesis.speak(u);
            } catch(e) {
              fetch('/announce-done', {method:'POST'});
            }
          } else {
            speak(audioText);
          }
        } else if (isPoseAnnounce) {
          // Audio off — unblock the backend right away so the operator
          // isn't stuck in a lockout they can't hear.
          fetch('/announce-done', {method:'POST'});
        }
      }
      // Beep: tick at 0/33/66% of hold-progress (buckets 0/1/2). Reset on progress=0.
      if (audioOn) {
        if (progress <= 0.01) {
          lastBeepBucket = -1;
        } else {
          const bucket = Math.floor(progress * 3);  // 0,1,2
          if (bucket !== lastBeepBucket && bucket < 3) {
            lastBeepBucket = bucket;
            beep(1000, 70, 0.12);  // short high tick
          }
        }
        // Capture confirmation: capturedN jumped
        if (capturedN > lastCapturedN) {
          lastCapturedN = capturedN;
          beep(660, 220, 0.18);  // long warm confirmation
        } else if (capturedN < lastCapturedN) {
          // UNDO happened
          lastCapturedN = capturedN;
          beep(330, 250, 0.15);
        }
      } else {
        lastCapturedN = capturedN;
      }
    }
  } catch(e){}
}
setInterval(refresh, 150);
</script>
</body></html>"""


def _draw_ghost(vis: np.ndarray, outer_corners: np.ndarray,
                color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a ghost quadrilateral on a preview frame."""
    pts = outer_corners.reshape(-1, 1, 2).astype(np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    # Corner markers
    for p in pts.reshape(-1, 2):
        cv2.circle(vis, tuple(p), 6, color, -1, lineType=cv2.LINE_AA)


def _draw_direction_arrow(vis: np.ndarray, err: dict[str, float],
                          ghost_center: np.ndarray) -> None:
    """Draw an arrow from ghost center indicating where to move the board."""
    # compute_alignment_by_corners returns centroid_offset_px; the legacy
    # compute_alignment_error returns center_px. Accept either.
    offset = err.get("centroid_offset_px", err.get("center_px", 0.0))
    if offset <= ALIGN_CENTER_TOL_PX:
        return
    # Arrow points FROM where the board IS to where it SHOULD BE (ghost center).
    # offset_x is (det - ghost), so flip sign for direction hint
    dx, dy = -err["offset_x"], -err["offset_y"]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    # Clamp magnitude so arrow stays on screen
    max_len = 80
    scale = min(1.0, max_len / length)
    dx *= scale; dy *= scale
    gx, gy = int(ghost_center[0]), int(ghost_center[1])
    # Draw from ghost center OUTWARD in the direction the board should move
    # (but the board is currently OFFSET the opposite way, so arrow shows move-direction)
    tip_x = int(gx - dx)
    tip_y = int(gy - dy)
    cv2.arrowedLine(vis, (gx, gy), (tip_x, tip_y), (50, 220, 255), 4,
                    tipLength=0.35, line_type=cv2.LINE_AA)


def _rms_color_for(rms: float) -> str:
    if rms < 1.5:
        return "#2ecc71"
    if rms < 3.0:
        return "#f1c40f"
    return "#e74c3c"


def _background_rms_worker(state: _GuidedState, board, board_size, sq_len, mk_len,
                           captures_dir: Path, stop_evt: threading.Event,
                           legacy_pattern: bool = True) -> None:
    """Every 3 new captures past the 8th, attempt an incremental calibration."""
    last_attempt_count = 0
    while not stop_evt.is_set():
        time.sleep(2.0)
        with state.lock:
            snapshot = list(state.captured_pairs)
        n = len(snapshot)
        if n < 8 or (n - last_attempt_count) < 3:
            continue
        last_attempt_count = n
        pairs = []
        for lp, rp, _pid in snapshot:
            il = cv2.imread(str(lp))
            ir = cv2.imread(str(rp))
            if il is not None and ir is not None:
                pairs.append((il, ir))
        try:
            result = calibrate_stereo(
                pairs, board_size=board_size, square_length=sq_len, marker_length=mk_len,
                legacy_pattern=legacy_pattern,
            )
            rms_l = _residual_estimate(pairs, board, result)
            with state.lock:
                state.rms_text = f"RMS≈{rms_l:.2f}px ({n} pares)"
                state.rms_color = _rms_color_for(rms_l)
        except Exception as e:
            logger.debug("Incremental calibration skipped: %s", e)


def _residual_estimate(pairs, board, result) -> float:
    """Quick residual estimate for a calibration — reuse left camera RMS.

    Projects each pair's observed corners through the fitted fisheye model and
    returns the mean per-pair RMS. Cheaper than re-running fisheye.calibrate and
    robust to the few degenerate poses that occasionally appear mid-capture.
    """
    from src.vision.calibration import compute_per_pair_residuals
    per_pair = compute_per_pair_residuals(pairs, board, result)
    rms_vals = [p["rms_l"] for p in per_pair
                if p["rms_l"] == p["rms_l"]]  # NaN filter
    if not rms_vals:
        return 99.0
    return float(sum(rms_vals) / len(rms_vals))


SESSION_SIDECAR = "session.json"
SESSION_VERSION = 1


def _session_path(output_dir: Path) -> Path:
    return output_dir / SESSION_SIDECAR


def _session_params(args: argparse.Namespace) -> dict:
    return {
        "board_size": [args.columns, args.rows],
        "square_length": args.square_length,
        "marker_length": args.marker_length,
        "dist_near_mm": getattr(args, "dist_near_mm", DEFAULT_DIST_NEAR_MM),
        "dist_mid_mm": getattr(args, "dist_mid_mm", DEFAULT_DIST_MID_MM),
        "dist_far_mm": getattr(args, "dist_far_mm", DEFAULT_DIST_FAR_MM),
        # Resolution must match across resume — calibration needs all frames
        # at the same size or the math breaks. 2026-04-28 session would have
        # silently corrupted if the operator had retried at a different res.
        "resolution": list(args.resolution),
    }


def _save_session(
    output_dir: Path, state: "_GuidedState",
    poses: list, args: argparse.Namespace,
) -> None:
    """Write sidecar session.json atomically after each state change."""
    data = {
        "version": SESSION_VERSION,
        "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "params": _session_params(args),
        "pose_sequence_ids": [p.id for p in poses],
        "pose_status": {poses[i].id: state.pose_status[i] for i in range(len(poses))},
        "captures": [
            {"pose_id": pid, "left": Path(lp).name, "right": Path(rp).name}
            for lp, rp, pid in state.captured_pairs
        ],
    }
    path = _session_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        logger.warning("No pude escribir session.json: %s", e)


def _load_session(output_dir: Path, args: argparse.Namespace) -> Optional[dict]:
    """Load session.json and validate params match current args.

    Returns the parsed dict, or None if file is missing/invalid/incompatible.
    """
    path = _session_path(output_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("session.json ilegible: %s", e)
        return None

    expected = _session_params(args)
    actual = data.get("params", {})
    mismatches = [
        k for k in expected
        if expected[k] != actual.get(k)
    ]
    if mismatches:
        logger.error(
            "session.json incompatible — parámetros distintos: %s. "
            "Usá las mismas flags que en la sesión original o borrá session.json.",
            mismatches,
        )
        return None
    return data


def _run_guided_capture(args: argparse.Namespace) -> None:
    """Guided capture loop: ghost + corner-ID matching + auto-capture with
    stability, L/R sync gate, quality gate, bootstrap intrinsics handoff,
    ambient drift warnings, and optional web-UI audio.

    Supports --resume: restores captured_pairs + pose_status from session.json,
    re-fits bootstrap K from existing captures if count >= BOOTSTRAP_COUNT.
    """
    global _shutting_down, _guided_state, _latest_jpeg

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.vision.capture import StereoCapture

    cap = StereoCapture(
        cam_left_id=args.left, cam_right_id=args.right,
        resolution=tuple(args.resolution), fps=args.fps,
    )
    cap.open()

    board_size = (args.columns, args.rows)
    dict_id = _resolve_aruco_dict(getattr(args, "aruco_dict", "DICT_4X4_100"))
    board = create_charuco_board(
        board_size=board_size,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_id=dict_id,
        legacy_pattern=args.legacy_pattern,
    )

    poses = default_pose_sequence(
        near_mm=getattr(args, "dist_near_mm", DEFAULT_DIST_NEAR_MM),
        mid_mm=getattr(args, "dist_mid_mm", DEFAULT_DIST_MID_MM),
        far_mm=getattr(args, "dist_far_mm", DEFAULT_DIST_FAR_MM),
    )
    state = _GuidedState()
    state.pose_status = ["pending"] * len(poses)
    _guided_state = state

    # Resume handling. Without --resume we wipe any previous session in the
    # output dir so the operator doesn't have to clean up manually between
    # runs. With --resume we validate the sidecar and continue from it.
    resume_flag = getattr(args, "resume", False)
    existing_session = _load_session(output_dir, args)
    if existing_session is not None and not resume_flag:
        logger.info(
            "Sesión previa encontrada en %s — se descarta (no se pasó --resume).",
            output_dir,
        )
        for old_png in output_dir.glob("left_*.png"):
            try:
                old_png.unlink()
            except OSError:
                pass
        for old_png in output_dir.glob("right_*.png"):
            try:
                old_png.unlink()
            except OSError:
                pass
        sidecar = _session_path(output_dir)
        try:
            sidecar.unlink()
        except OSError:
            pass
        existing_session = None

    count = 0
    if resume_flag and existing_session is not None:
        pose_idx_by_id = {p.id: i for i, p in enumerate(poses)}
        # Restore captures
        for cap_rec in existing_session.get("captures", []):
            pid = cap_rec.get("pose_id")
            if pid not in pose_idx_by_id:
                continue
            lp = output_dir / cap_rec["left"]
            rp = output_dir / cap_rec["right"]
            if not (lp.exists() and rp.exists()):
                logger.warning("Captura ausente en disco, se descarta: %s", pid)
                continue
            state.captured_pairs.append((lp, rp, pid))
            state.pose_status[pose_idx_by_id[pid]] = "captured"
        # Restore skipped
        for pid, status in existing_session.get("pose_status", {}).items():
            if status == "skipped" and pid in pose_idx_by_id:
                if state.pose_status[pose_idx_by_id[pid]] == "pending":
                    state.pose_status[pose_idx_by_id[pid]] = "skipped"
        count = len(state.captured_pairs)
        # Advance pose pointer to first pending
        for i, status in enumerate(state.pose_status):
            if status == "pending":
                state.current_pose_idx = i
                break
        else:
            logger.info("Sesión ya completa (%d capturas). Nada que resumir.", count)
        # Re-fit bootstrap K if enough captures
        if count >= BOOTSTRAP_COUNT:
            lefts = []
            for lp, _rp, _pid in state.captured_pairs:
                img = cv2.imread(str(lp))
                if img is not None:
                    lefts.append(img)
            if len(lefts) >= 4:
                fitted = fit_single_camera_intrinsics(
                    lefts,
                    create_charuco_board(
                        board_size=board_size,
                        square_length=args.square_length,
                        marker_length=args.marker_length,
                        dict_id=dict_id,
                        legacy_pattern=args.legacy_pattern,
                    ),
                )
                if fitted is not None:
                    state.fitted_K = fitted
                    state.bootstrap_done = True
                    logger.info(
                        "Resume: %d capturas restauradas, bootstrap K re-ajustado "
                        "(fx=%.0f fy=%.0f)",
                        count, fitted[0, 0], fitted[1, 1],
                    )
                else:
                    logger.warning("Resume: bootstrap fit falló, seguimos con K nominal")
        else:
            logger.info(
                "Resume: %d capturas restauradas (< bootstrap threshold %d, "
                "seguimos con K nominal)",
                count, BOOTSTRAP_COUNT,
            )

    # SO_REUSEADDR so a Ctrl-C'd previous instance doesn't leave the port
    # in TIME_WAIT for the next run.
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _GuidedHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    rms_stop = threading.Event()
    rms_thread = threading.Thread(
        target=_background_rms_worker,
        args=(state, board, board_size, args.square_length,
              args.marker_length, output_dir, rms_stop,
              getattr(args, "legacy_pattern", True)),
        daemon=True,
    )
    rms_thread.start()

    logger.info("Calibración guiada — preview: http://people-counter.local:%d", args.port)
    logger.info("Poses objetivo: %d (bootstrap con las primeras %d)", len(poses), BOOTSTRAP_COUNT)
    logger.info("Seguí la silueta fantasma, mantené quieto %.1fs para capturar.", STABILITY_HOLD_SEC)
    logger.info("Esperando que el operador haga click en Comenzar...")

    # Block until the operator clicks "Comenzar" in the browser. This gives
    # them time to position the board + activates the browser audio context.
    try:
        while not _capture_started:
            if state.finish_requested:
                logger.info("Cancelado antes de comenzar.")
                cap.close()
                try:
                    server.shutdown()
                except Exception:
                    pass
                return
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrumpido antes de comenzar (Ctrl-C).")
        try:
            cap.close()
        except Exception:
            pass
        try:
            server.shutdown()
        except Exception:
            pass
        sys.exit(0)
    logger.info("Ctrl+C para cancelar, o usá el botón Finalizar en la UI.\n")

    stability = StabilityTracker()
    hold_started_at: Optional[float] = None
    pose_started_at = time.time()
    count = 0

    # Ambient drift baseline (from first N frames)
    drift_baseline: Optional[float] = None
    drift_samples: list[float] = []
    frame_counter = 0
    # Sensor temperature drift tracking
    temp_baseline: Optional[float] = None
    temp_samples: list[float] = []
    last_temp: Optional[float] = None
    # Asymmetric-detection streak (for the "one camera silently failing" hint).
    # If one camera keeps detecting the board but the other doesn't, we want
    # to surface that in the UI — operator was getting "captura rechazada"
    # without knowing which camera was failing.
    only_l_detect_streak = 0
    only_r_detect_streak = 0
    ASYMMETRIC_DETECT_WARN_FRAMES = 20

    def _emit_audio(text: str) -> None:
        state.audio_event = text
        state.audio_event_seq += 1

    def _pose_announce(idx: int) -> str:
        """Spoken prompt for a pose: label + target distance in cm."""
        p = poses[idx]
        z_cm = p.tvec_mm[2] / 10.0
        return f"Pose {idx + 1}. {p.label}. A {z_cm:.0f} centímetros de la cámara"

    def _emit_pose_announce(idx: int) -> None:
        """Emit the pose announcement and lock out everything else until the
        browser signals the utterance finished (POST /announce-done)."""
        global _announce_pending
        _announce_pending = True
        _emit_audio(_pose_announce(idx))

    # Announce the first pending pose (= poses[0] on fresh start, or the
    # next pending after resume restore)
    _emit_pose_announce(state.current_pose_idx)

    # Persist initial/resumed state so a crash before any new capture still
    # leaves a valid session.json on disk.
    _save_session(output_dir, state, poses, args)

    # Debounce the L/R desync warning: sync delta varies frame-to-frame and
    # spikes briefly after events like skip/undo (main thread busy => buffers
    # drift). We only show the warning when we see >= this many consecutive
    # bad frames, hiding single-frame noise spikes.
    SYNC_WARN_CONSECUTIVE_FRAMES = 4
    sync_bad_streak = 0

    prev_pose_idx = state.current_pose_idx

    try:
        while count < len(poses):
            # Detect any pose change and reset "what we last spoke" so the
            # first movement hint for the new pose always plays, even if
            # the previous pose's last hint was the same phrase.
            if state.current_pose_idx != prev_pose_idx:
                state.last_spoken_key = ""
                state.locked_alignment_text = ""
                sync_bad_streak = 0  # clean slate so skip doesn't replay the warning
                prev_pose_idx = state.current_pose_idx

            pose = poses[state.current_pose_idx]
            if state.pose_status[state.current_pose_idx] != "pending":
                state.current_pose_idx = (state.current_pose_idx + 1) % len(poses)
                if all(s != "pending" for s in state.pose_status):
                    break
                continue

            # Capture with timestamps + sensor temperature for drift tracking
            try:
                frame_l, frame_r, ts_l, ts_r, temp_l, temp_r = cap.read_with_metadata()
                lr_sync_ok = abs(ts_l - ts_r) <= LR_SYNC_MAX_DELTA_NS
                lr_delta_ms = abs(ts_l - ts_r) / 1e6
                temps = [t for t in (temp_l, temp_r) if t is not None]
                last_temp = sum(temps) / len(temps) if temps else None
            except Exception:
                frame_l, frame_r = cap.read()
                lr_sync_ok = True
                lr_delta_ms = 0.0
                last_temp = None

            if lr_sync_ok:
                sync_bad_streak = 0
            else:
                sync_bad_streak += 1
            # Suppress the warning for 2s after each pose transition — the
            # picamera2 pipeline routinely spikes right after skip/capture
            # while the main thread was busy, and that drift settles within
            # a second or two without operator action.
            pose_settling = (time.time() - pose_started_at) < 2.0
            sync_warn_active = (
                sync_bad_streak >= SYNC_WARN_CONSECUTIVE_FRAMES
                and not pose_settling
            )

            frame_counter += 1

            # Temperature baseline lock (first N valid samples)
            if last_temp is not None:
                if temp_baseline is None:
                    temp_samples.append(last_temp)
                    if len(temp_samples) >= DRIFT_BASELINE_FRAMES:
                        temp_baseline = float(np.median(temp_samples))
                        logger.info("Sensor temperature baseline: %.1f°C", temp_baseline)
                elif frame_counter % DRIFT_CHECK_EVERY_FRAMES == 0:
                    delta_c = last_temp - temp_baseline
                    if abs(delta_c) > 10.0:
                        state.drift_warning = (
                            f"Sensor +{delta_c:.1f}°C desde el inicio "
                            f"({temp_baseline:.1f}→{last_temp:.1f}°C) "
                            f"— los intrínsecos pueden derivar"
                        )

            # Ambient drift tracking
            gray_small = cv2.cvtColor(
                cv2.resize(frame_l, (320, 240)), cv2.COLOR_BGR2GRAY,
            )
            brightness = float(np.median(gray_small))
            if drift_baseline is None:
                drift_samples.append(brightness)
                if len(drift_samples) >= DRIFT_BASELINE_FRAMES:
                    drift_baseline = float(np.median(drift_samples))
                    logger.info("Drift baseline locked at median brightness %.1f", drift_baseline)
            elif frame_counter % DRIFT_CHECK_EVERY_FRAMES == 0:
                drift_pct = abs(brightness - drift_baseline) / max(drift_baseline, 1) * 100
                if drift_pct > DRIFT_WARN_PCT:
                    direction = "aumentó" if brightness > drift_baseline else "bajó"
                    state.drift_warning = (
                        f"Luz ambiente {direction} {drift_pct:.0f}% desde el inicio "
                        f"— considerá reiniciar para relockear exposición"
                    )
                else:
                    state.drift_warning = None

            # ChArUco detection on full-res. lenient=True so 33mm markers at
            # the 3m far poses (~22 px wide → 0.5% of 4608) still get picked up
            # — strict default rejects markers under 3% of the image dimension.
            # min_corners=4 matches the alignment-gate threshold below; with the
            # default of 8, partial detections at the fisheye edges (corner
            # poses) bounce between "8 corners" and "7 corners" frame to frame
            # and the wizard flips between Alineado and "board no visible",
            # never holding stable long enough to capture.
            corners_l, ids_l = detect_charuco_corners(
                frame_l, board, lenient=True, min_corners=4,
            )
            corners_r, ids_r = detect_charuco_corners(
                frame_r, board, lenient=True, min_corners=4,
            )

            # Track asymmetric detection — one camera consistently failing
            # while the other detects fine is the silent killer the 2026-04-28
            # session ran into (R returned 0 corners on C1/C2 while L saw 40,
            # operator had no signal in the UI).
            l_detected = corners_l is not None and ids_l is not None and len(ids_l) >= 4
            r_detected = corners_r is not None and ids_r is not None and len(ids_r) >= 4
            if l_detected and not r_detected:
                only_l_detect_streak += 1
                only_r_detect_streak = 0
            elif r_detected and not l_detected:
                only_r_detect_streak += 1
                only_l_detect_streak = 0
            else:
                only_l_detect_streak = 0
                only_r_detect_streak = 0

            vis_l = cv2.resize(frame_l, GUIDED_HALF)
            vis_r = cv2.resize(frame_r, GUIDED_HALF)
            scale_x = GUIDED_HALF[0] / frame_l.shape[1]
            scale_y = GUIDED_HALF[1] / frame_l.shape[0]

            # Project ghost — uses fitted_K after bootstrap
            ghost = project_pose(
                pose, board_size, args.square_length, GUIDED_HALF,
                fitted_K=state.fitted_K,
            )

            # Corner-ID matching vs left camera detection
            scaled_corners_l: Optional[np.ndarray] = None
            err: Optional[dict] = None
            aligned = False
            if corners_l is not None and ids_l is not None and len(ids_l) >= 4:
                scaled = corners_l.reshape(-1, 2).copy()
                scaled[:, 0] *= scale_x
                scaled[:, 1] *= scale_y
                scaled_corners_l = scaled
                err = compute_alignment_by_corners(
                    scaled, ids_l, ghost["inner_corners"],
                )
                tol_loose, tol_tight = _resolve_tolerance(args)
                if state.bootstrap_done:
                    aligned = is_aligned_by_corners(
                        err,
                        mean_err_tol_px=tol_tight,
                        min_matched=ALIGN_MATCHED_MIN_TIGHT,
                    )
                else:
                    aligned = is_aligned_by_corners(
                        err,
                        mean_err_tol_px=tol_loose,
                        min_matched=ALIGN_MATCHED_MIN_LOOSE,
                    )

            # Stability
            stability.push(scaled_corners_l if aligned else None)
            stable = stability.is_stable() if aligned else False

            now = time.time()
            # Gate everything on the precise "announcement finished"
            # signal from the browser (POST /announce-done on
            # SpeechSynthesisUtterance.onend). This way the three-piece
            # "number / label / distance" block is guaranteed to play
            # end-to-end regardless of voice speed — captures and other
            # audio hints stay blocked until the operator has actually
            # heard the whole thing. Only manual Skip can interrupt.
            announce_settling = _announce_pending
            announce_audio_lockout = _announce_pending
            if aligned and stable and not announce_settling:
                if hold_started_at is None:
                    hold_started_at = now
                    # Only voice "Mantené quieto" once the pose name has
                    # finished playing. Capture can still run during that
                    # window — it just won't announce.
                    if not announce_audio_lockout:
                        _emit_audio("Mantené quieto")
                hold_progress = min(1.0, (now - hold_started_at) / STABILITY_HOLD_SEC)
            else:
                hold_started_at = None
                hold_progress = 0.0

            should_capture = (
                aligned and stable and not announce_settling
                and hold_progress >= 1.0
            )

            warnings = live_lighting_warnings(frame_l, frame_r)

            # Handle user actions
            with state.lock:
                if state.finish_requested:
                    state.finish_requested = False
                    break
                if state.undo_requested:
                    state.undo_requested = False
                    if state.captured_pairs:
                        lp, rp, pid = state.captured_pairs.pop()
                        try: lp.unlink()
                        except Exception: pass
                        try: rp.unlink()
                        except Exception: pass
                        count = max(0, count - 1)
                        for i, p in enumerate(poses):
                            if p.id == pid:
                                state.pose_status[i] = "pending"
                                state.current_pose_idx = i
                                break
                        pose_started_at = time.time()
                        hold_started_at = None
                        stability.reset()
                        logger.info("UNDO — removida captura de pose %s", pid)
                        _emit_audio("Deshecha última captura")
                        # If we dropped below bootstrap, reset fitted_K too
                        if count < BOOTSTRAP_COUNT:
                            state.bootstrap_done = False
                            state.fitted_K = None
                        _save_session(output_dir, state, poses, args)
                if state.skip_requested:
                    state.skip_requested = False
                    state.pose_status[state.current_pose_idx] = "skipped"
                    logger.info("Pose %s saltada por usuario", pose.id)
                    state.current_pose_idx = (state.current_pose_idx + 1) % len(poses)
                    pose_started_at = time.time()
                    hold_started_at = None
                    stability.reset()
                    # Announce the NEW pose so the operator knows where to
                    # move next. Skip itself isn't announced (they clicked
                    # the button, they know).
                    if any(s == "pending" for s in state.pose_status):
                        _emit_pose_announce(state.current_pose_idx)
                    _save_session(output_dir, state, poses, args)
                    continue

            # Auto-skip timeout (configurable via --pose-timeout-sec)
            pose_timeout = getattr(args, "pose_timeout_sec", SKIP_POSE_TIMEOUT_SEC)
            if now - pose_started_at > pose_timeout:
                state.pose_status[state.current_pose_idx] = "skipped"
                logger.info("Pose %s auto-saltada por timeout", pose.id)
                _emit_audio("Pose saltada por timeout")
                state.current_pose_idx = (state.current_pose_idx + 1) % len(poses)
                pose_started_at = time.time()
                hold_started_at = None
                stability.reset()
                _save_session(output_dir, state, poses, args)
                if any(s == "pending" for s in state.pose_status):
                    _emit_pose_announce(state.current_pose_idx)
                continue

            # Detected corners first, then ghost on top — keeps the ghost
            # outline visible above the busy charuco overlay (lines + IDs)
            # so the operator can actually see the alignment target. Ghost
            # is only drawn on L: alignment is computed against L only, and
            # showing it on R was misleading because parallax (14 cm baseline)
            # makes the same physical board land 14 cm offset in R — operators
            # were trying to fit both ghosts simultaneously, which is impossible.
            if corners_l is not None and ids_l is not None:
                sc = corners_l.copy()
                sc[:, 0, 0] *= scale_x
                sc[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_l, sc, ids_l, (0, 255, 0))
            if corners_r is not None and ids_r is not None:
                sc = corners_r.copy()
                sc[:, 0, 0] *= scale_x
                sc[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_r, sc, ids_r, (0, 255, 0))

            ghost_color = (80, 220, 80) if aligned else (80, 180, 255)
            _draw_ghost(vis_l, ghost["outer_corners"], ghost_color)

            # Direction arrow on LEFT
            if err is not None and not aligned and err["matched"] >= 4:
                _draw_direction_arrow(vis_l, err, ghost["center"])

            # Hold progress bar on LEFT preview top
            if hold_progress > 0:
                bar_w = int(GUIDED_HALF[0] * hold_progress)
                cv2.rectangle(vis_l, (0, 0), (GUIDED_HALF[0], 8), (30, 30, 30), -1)
                cv2.rectangle(vis_l, (0, 0), (bar_w, 8), (0, 255, 0), -1)

            cv2.putText(vis_l, "L", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_r, "R", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            combined = np.hstack([vis_l, vis_r])

            # Capture gate: quality + L/R sync + common corners
            if should_capture:
                n_corners_l = len(corners_l) if corners_l is not None else 0
                common_n = count_common_corners(ids_l, ids_r)
                quality = assess_frame_quality(
                    frame_l, frame_r, n_corners_l,
                    corners_l=corners_l, corners_r=corners_r,
                )
                reject_reasons: list[str] = []
                if not quality["all_pass"]:
                    reject_reasons.extend(quality["reasons"])
                if common_n < LR_MIN_COMMON_CORNERS:
                    reject_reasons.append(
                        f"pocas esquinas en común L∩R ({common_n}<{LR_MIN_COMMON_CORNERS})"
                    )
                if not lr_sync_ok:
                    reject_reasons.append(
                        f"L/R desincronizados ({lr_delta_ms:.1f}ms>"
                        f"{LR_SYNC_MAX_DELTA_NS/1e6:.0f}ms)"
                    )

                if not reject_reasons:
                    # Stable filename keyed by pose_id — makes --resume idempotent
                    # and keeps existing left_NNN.png pattern so downstream tools
                    # (calibrate subcommand glob) still work.
                    ordinal = len(state.captured_pairs)
                    left_path = output_dir / f"left_{ordinal:03d}_{pose.id}.png"
                    right_path = output_dir / f"right_{ordinal:03d}_{pose.id}.png"
                    cv2.imwrite(str(left_path), frame_l)
                    cv2.imwrite(str(right_path), frame_r)
                    with state.lock:
                        state.captured_pairs.append((left_path, right_path, pose.id))
                        state.pose_status[state.current_pose_idx] = "captured"
                    _save_session(output_dir, state, poses, args)
                    count += 1
                    cv2.rectangle(combined, (0, 0),
                                  (combined.shape[1] - 1, combined.shape[0] - 1),
                                  (0, 255, 0), 8)
                    logger.info(
                        "[%d/%d] Pose %s capturada — %s (common=%d, sync=%.1fms)",
                        count, len(poses), pose.id, pose.label, common_n, lr_delta_ms,
                    )
                    # Don't TTS "Capturada" — the capture beep already
                    # confirms it and a spoken confirmation just delays the
                    # next pose announcement from being heard.

                    # Bootstrap handoff: after N captures fit intrinsics
                    if not state.bootstrap_done and count >= BOOTSTRAP_COUNT:
                        logger.info("Fitting bootstrap intrinsics from %d captures...", count)
                        lefts = []
                        for lp, _rp, _pid in state.captured_pairs:
                            img = cv2.imread(str(lp))
                            if img is not None:
                                lefts.append(img)
                        fitted = fit_single_camera_intrinsics(lefts, board)
                        if fitted is not None:
                            state.fitted_K = fitted
                            state.bootstrap_done = True
                            logger.info(
                                "Bootstrap done: f_x=%.0f, f_y=%.0f, cx=%.0f, cy=%.0f — "
                                "switching to tight tolerance",
                                fitted[0, 0], fitted[1, 1], fitted[0, 2], fitted[1, 2],
                            )
                            _emit_audio("Intrínsecos bootstrap ajustados, tolerancia ahora estricta")
                        else:
                            logger.warning("Bootstrap fit failed; staying with nominal K")

                    # Advance to next pending pose
                    nxt = state.current_pose_idx
                    for _ in range(len(poses)):
                        nxt = (nxt + 1) % len(poses)
                        if state.pose_status[nxt] == "pending":
                            break
                    state.current_pose_idx = nxt
                    pose_started_at = time.time()
                    hold_started_at = None
                    stability.reset()
                    if any(s == "pending" for s in state.pose_status):
                        _emit_pose_announce(nxt)
                else:
                    logger.info("Captura rechazada: %s", "; ".join(reject_reasons))
                    stability.reset()
                    hold_started_at = None

            # Encode preview JPEG
            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with _jpeg_lock:
                _latest_jpeg = jpeg.tobytes()

            # Status composition
            captured_n = sum(1 for s in state.pose_status if s == "captured")
            skipped_n = sum(1 for s in state.pose_status if s == "skipped")

            if aligned and stable:
                banner = f"Capturando... {int(hold_progress * 100)}%"
                banner_color = "#f1c40f"
                audio_text = ""  # don't interrupt hold
            elif aligned:
                banner = "Alineado — mantené quieto"
                banner_color = "#3498db"
                audio_text = ""
            elif err is not None and err["matched"] >= 4:
                # Convert pixel offsets to cm for the voice hint — much more
                # intuitive for the operator than "movelo 47 píxeles".
                if state.fitted_K is not None:
                    f_px_for_hint = float(state.fitted_K[0, 0])
                else:
                    f_px_for_hint = NOMINAL_FOCAL_PX
                # err offsets are in PREVIEW pixel space (GUIDED_HALF, ~648
                # wide after the scale_x multiplication earlier). The fitted_K
                # / nominal f_px are at full-res 4608. Scale f down so the
                # mm_per_px ratio is consistent with the offset units.
                f_px_preview = f_px_for_hint * scale_x
                # PoseTarget stores position as tvec_mm = (x, y, z); z is depth.
                mm_per_px_here = pose.tvec_mm[2] / f_px_preview
                banner = alignment_hint_by_corners(err, mm_per_px=mm_per_px_here)
                banner_color = "#e67e22"
                audio_text = banner
            else:
                banner = "Board no visible — movelo al centro"
                banner_color = "#e74c3c"
                audio_text = ""

            # Refresh audio event only when banner changes SEMANTICALLY —
            # strip digits so "movelo izquierda 4cm" vs "movelo izquierda
            # 3cm" count as the same hint and don't re-trigger TTS.
            # Also skip during the post-pose-announcement grace window so
            # the pose-name TTS isn't immediately overwritten by a movement
            # hint (state.audio_event is a single slot — last write wins).
            import re as _re
            def _key(s: str) -> str:
                return _re.sub(r"[\d.,]+", "", s or "").strip()
            # `last_spoken_key` tracks WHAT WAS SPOKEN, separate from the
            # visible banner. If we suppressed an emit during the lockout
            # we must not pretend we already said it — otherwise when the
            # lockout ends and the hint is still current, _key comparison
            # thinks we delivered it already and stays silent forever.
            spoke_audio = False
            if (audio_text
                    and not announce_audio_lockout
                    and _key(audio_text) != state.last_spoken_key):
                _emit_audio(audio_text)
                state.last_spoken_key = _key(audio_text)
                state.locked_alignment_text = audio_text
                spoke_audio = True

            # When the alignment hint has the same semantic key as what the
            # operator just heard, keep the on-screen banner frozen on the
            # exact text we said. Prevents 1-cm flicker between live
            # measurement and stale audio (operator hears "15cm", sees
            # "14cm" and thinks they don't match).
            if (audio_text
                    and state.locked_alignment_text
                    and _key(audio_text) == state.last_spoken_key):
                banner = state.locked_alignment_text

            phase_label = "bootstrap" if not state.bootstrap_done else "estricto"
            warnings_html = ""
            if warnings:
                warnings_html = '<div class="warn">⚠ ' + " · ".join(warnings) + "</div>"
            if state.drift_warning:
                warnings_html += f'<div class="warn">⚠ {state.drift_warning}</div>'

            rms_html = (f'<span class="rms" style="background:{state.rms_color}">'
                        f'{state.rms_text}</span>')

            sync_note = ""
            if sync_warn_active:
                sync_note = f" · <span class=\"warn\">L/R desync {lr_delta_ms:.1f}ms</span>"

            # Escape audio text for the HTML data-* attribute
            audio_escaped = (state.audio_event or "").replace('"', "&quot;")
            banner_escaped = banner.replace('"', "&quot;")

            n_l = len(corners_l) if corners_l is not None else 0
            n_r = len(corners_r) if corners_r is not None else 0

            asymmetric_warn_html = ""
            if only_l_detect_streak >= ASYMMETRIC_DETECT_WARN_FRAMES:
                asymmetric_warn_html = (
                    '<div class="warn" style="font-weight:600">'
                    '⚠ Cámara R no detecta el board hace varios segundos '
                    '(L sí). Limpiá el lens, chequeá foco, o asegurate '
                    'que el board entre en su FOV.</div>'
                )
            elif only_r_detect_streak >= ASYMMETRIC_DETECT_WARN_FRAMES:
                asymmetric_warn_html = (
                    '<div class="warn" style="font-weight:600">'
                    '⚠ Cámara L no detecta el board hace varios segundos '
                    '(R sí). Limpiá el lens, chequeá foco, o asegurate '
                    'que el board entre en su FOV.</div>'
                )

            # Color L/R counts: highlight in amber when one is 0 and the other > 0
            def _detect_pill(label: str, n: int, asymmetric_zero: bool) -> str:
                if asymmetric_zero:
                    color = "#e67e22"
                elif n >= 8:
                    color = "#2ecc71"
                elif n >= 4:
                    color = "#f1c40f"
                else:
                    color = "#888"
                return f'<span style="color:{color};font-weight:600">{label}:{n}</span>'

            l_zero_asym = (n_l == 0 and n_r > 0)
            r_zero_asym = (n_r == 0 and n_l > 0)
            detect_pills = (
                f'{_detect_pill("L", n_l, l_zero_asym)} · '
                f'{_detect_pill("R", n_r, r_zero_asym)}'
            )

            status = f"""
<div data-banner="{banner_escaped}" data-color="{banner_color}" data-progress="{hold_progress:.2f}" data-audioseq="{state.audio_event_seq}" data-audiotext="{audio_escaped}" data-captured="{captured_n}" data-pose-idx="{state.current_pose_idx}"></div>
<div>Pose <b>{state.current_pose_idx + 1}/{len(poses)}</b> — {pose.label} · <b>{pose.tvec_mm[2] / 10.0:.0f}cm</b> <span class="pill-phase">{phase_label}</span></div>
<div>Capturadas: <b>{captured_n}</b> · Skipped: <b>{skipped_n}</b> · Restantes: <b>{len(poses) - captured_n - skipped_n}</b></div>
<div>{rms_html}</div>
{warnings_html}
{asymmetric_warn_html}
<div style="color:#888;font-size:12px;margin-top:8px">Detección: {detect_pills} esquinas · matched L={err['matched'] if err else 0}{sync_note}</div>
"""
            with state.lock:
                state.status_html = status
                state.hold_progress = hold_progress
                state.banner_text = banner

            print(
                f"\r  Pose {state.current_pose_idx+1}/{len(poses)} [{pose.id}] "
                f"cap={captured_n} skip={skipped_n} "
                f"{'STABLE' if stable else 'ALIGN' if aligned else 'AIM'}  ",
                end="", flush=True,
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")

    _shutting_down = True
    rms_stop.set()
    cap.close()
    captured_n = sum(1 for s in state.pose_status if s == "captured")
    print(f"\n\nCapturas: {captured_n} pares guardados en {output_dir}")
    return


def cmd_generate_board(args: argparse.Namespace) -> None:
    """Generate a printable ChArUco board image."""
    dict_id = _resolve_aruco_dict(args.aruco_dict)
    board = create_charuco_board(
        board_size=(args.columns, args.rows),
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_id=dict_id,
        legacy_pattern=args.legacy_pattern,
    )
    img = generate_board_image(board, (args.width, args.height))
    cv2.imwrite(args.output, img)
    logger.info("Board saved to %s (%dx%d)", args.output, args.width, args.height)


def cmd_capture(args: argparse.Namespace) -> None:
    """Interactive capture with HTTP preview and coverage tracking."""
    global _shutting_down

    if args.guided:
        _run_guided_capture(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.vision.capture import StereoCapture

    cap = StereoCapture(
        cam_left_id=args.left,
        cam_right_id=args.right,
        resolution=tuple(args.resolution),
        fps=args.fps,
    )
    cap.open()

    dict_id = _resolve_aruco_dict(getattr(args, "aruco_dict", "DICT_4X4_100"))
    board = create_charuco_board(
        board_size=(args.columns, args.rows),
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_id=dict_id,
        legacy_pattern=getattr(args, "legacy_pattern", True),
    )

    # Start HTTP preview
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Coverage grid
    if args.grid == "circular":
        grid_mask = GRID_CIRCULAR
    else:
        grid_mask = GRID_RECTANGULAR
    grid_rows, grid_cols = grid_mask.shape
    coverage = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    # Per-cell targets: for binary grids use --per-cell if set, else no per-cell limit.
    if args.per_cell > 0:
        cell_target = grid_mask * args.per_cell
    else:
        cell_target = None  # no per-cell limit

    count = 0
    last_capture_time = 0.0
    valid_cells = int(np.count_nonzero(grid_mask))
    total_target = int(cell_target.sum()) if cell_target is not None else args.count

    # Distance recommendation
    cols, rows = args.columns, args.rows
    board_w_mm = cols * args.square_length
    board_h_mm = rows * args.square_length
    board_diag = f"{board_w_mm:.0f}x{board_h_mm:.0f}mm"
    dist_range = (f"{args.dist_near_mm/1000:.1f}/{args.dist_mid_mm/1000:.1f}/"
                  f"{args.dist_far_mm/1000:.1f}m (lab protocol spans fleet mount 3-4.5m)")

    # Manual trigger: Enter on stdin OR POST /capture via web UI
    global _trigger_armed, _manual_enabled
    _manual_enabled = args.manual
    if args.manual:
        def _stdin_listener():
            global _trigger_armed
            while not _shutting_down:
                try:
                    sys.stdin.readline()
                    _trigger_armed = True
                except Exception:
                    break
        threading.Thread(target=_stdin_listener, daemon=True).start()

    logger.info("Calibration capture — preview: http://people-counter.local:%d", args.port)
    logger.info("Board: %s (%s). Recommended distance: %s", board_diag, f"{cols}x{rows}", dist_range)
    logger.info("Grid: %s (%dx%d, %d active cells, %d total captures)", args.grid, grid_rows, grid_cols, valid_cells, total_target)
    logger.info("IMPORTANT: Tilt the board 20-30 deg in every capture. Never hold it flat/frontal.")
    logger.info("Move the ChArUco to cover all grid cells. Vary angles (pitch/yaw/roll) in each cell.")
    if args.manual:
        logger.info("MANUAL mode: press ENTER here OR click CAPTURE in the web UI to trigger.")
    else:
        logger.info("Auto-captures when board detected in both cameras. %.1fs cooldown between captures.", args.cooldown)
    logger.info("Ctrl+C to stop.\n")
    try:
        while True:
            # Stop condition
            if cell_target is not None:
                if np.all(coverage[grid_mask > 0] >= cell_target[grid_mask > 0]):
                    break
            elif count >= args.count and np.count_nonzero(coverage * grid_mask) >= valid_cells:
                break
            frame_l, frame_r = cap.read()

            # Detect corners (lenient: see wizard loop comment)
            corners_l, ids_l = detect_charuco_corners(frame_l, board, lenient=True)
            corners_r, ids_r = detect_charuco_corners(frame_r, board, lenient=True)

            # Build preview (resize for HTTP)
            vis_l = cv2.resize(frame_l, (648, 486))
            vis_r = cv2.resize(frame_r, (648, 486))
            scale_x = 648 / frame_l.shape[1]
            scale_y = 486 / frame_l.shape[0]

            detected = False
            n_common = 0

            if corners_l is not None and ids_l is not None:
                # Draw corners on left preview
                scaled_corners_l = corners_l.copy()
                scaled_corners_l[:, 0, 0] *= scale_x
                scaled_corners_l[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_l, scaled_corners_l, ids_l, (0, 255, 0))

            if corners_r is not None and ids_r is not None:
                scaled_corners_r = corners_r.copy()
                scaled_corners_r[:, 0, 0] *= scale_x
                scaled_corners_r[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_r, scaled_corners_r, ids_r, (0, 255, 0))

            if corners_l is not None and corners_r is not None and ids_l is not None and ids_r is not None:
                n_common = len(np.intersect1d(ids_l.flatten(), ids_r.flatten()))
                detected = n_common >= 8

            # Draw coverage grid on left preview
            _draw_coverage(vis_l, coverage, grid_mask, cell_target)

            # Status text
            color = (0, 255, 0) if detected else (0, 0, 255)
            status = f"Pair {count}/{total_target} | Common: {n_common}"
            cv2.putText(vis_l, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            covered_cells = int(np.count_nonzero(coverage * grid_mask))
            coverage_pct = int(covered_cells / valid_cells * 100)
            cov_color = (0, 255, 0) if covered_cells == valid_cells else (0, 200, 255)
            cv2.putText(vis_r, f"Coverage: {covered_cells}/{valid_cells} ({coverage_pct}%)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cov_color, 2)

            if detected:
                cv2.putText(vis_r, "BOARD DETECTED", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            combined = np.hstack([vis_l, vis_r])
            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _update_preview(jpeg.tobytes())

            # Trigger: manual (Enter pressed) or auto (detected + cooldown)
            now = time.time()
            if args.manual:
                should_capture = detected and _trigger_armed
                if _trigger_armed and not detected:
                    logger.warning("Trigger ignored — board not detected in both cameras.")
                    _trigger_armed = False
            else:
                should_capture = detected and (now - last_capture_time) >= args.cooldown
            if should_capture:
                # Check if this cell is already full
                row, col = _compute_coverage_center(
                    corners_l, frame_l.shape[1], frame_l.shape[0], grid_mask,
                )
                cell_max = cell_target[row, col] if cell_target is not None else 0
                if cell_max > 0 and coverage[row, col] >= cell_max:
                    # Cell full — skip but show feedback
                    cv2.putText(vis_r, f"Cell ({row},{col}) full ({cell_max}/{cell_max})", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    left_path = output_dir / f"left_{count:03d}.png"
                    right_path = output_dir / f"right_{count:03d}.png"
                    cv2.imwrite(str(left_path), frame_l)
                    cv2.imwrite(str(right_path), frame_r)

                    coverage[row, col] += 1
                    count += 1
                    last_capture_time = now
                    _trigger_armed = False
                    cell_limit = cell_target[row, col] if cell_target is not None else "inf"
                    logger.info(
                        "Pair %d/%d saved — %d common corners, coverage %d%%, cell (%d,%d): %d/%s",
                        count, total_target, n_common, coverage_pct,
                        row, col, coverage[row, col], cell_limit,
                    )

            remaining = f" | Missing {valid_cells - covered_cells} cells!" if count >= total_target and covered_cells < valid_cells else ""
            print(
                f"\r  Pairs: {count}/{total_target} | Common: {n_common:2d} | Coverage: {covered_cells}/{valid_cells}{remaining}   ",
                end="", flush=True,
            )
            time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    _shutting_down = True
    cap.close()
    print(f"\n\nCaptured {count} pairs in {output_dir}")
    print(f"Coverage: {int(np.count_nonzero(coverage * grid_mask))}/{valid_cells} cells")
    print("\nCoverage grid:")
    print(coverage)
    import os
    os._exit(0)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run stereo calibration from captured image pairs."""
    input_dir = Path(args.input_dir)

    # Find all left_*.png files and match with right_*.png
    left_files = sorted(input_dir.glob("left_*.png"))
    if not left_files:
        logger.error("No left_*.png files found in %s", input_dir)
        sys.exit(1)

    pairs = []
    for lf in left_files:
        rf = lf.parent / lf.name.replace("left_", "right_")
        if rf.exists():
            img_l = cv2.imread(str(lf))
            img_r = cv2.imread(str(rf))
            if img_l is not None and img_r is not None:
                pairs.append((img_l, img_r))
            else:
                logger.warning("Failed to read pair: %s", lf.stem)
        else:
            logger.warning("Missing right image for %s", lf.name)

    logger.info("Loaded %d image pairs from %s", len(pairs), input_dir)

    dict_id = _resolve_aruco_dict(getattr(args, "aruco_dict", "DICT_4X4_100"))
    try:
        result = calibrate_stereo(
            pairs,
            board_size=(args.columns, args.rows),
            square_length=args.square_length,
            marker_length=args.marker_length,
            dict_id=dict_id,
            legacy_pattern=getattr(args, "legacy_pattern", True),
        )
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        sys.exit(1)

    save_calibration(result, args.output)
    logger.info("Calibration saved to %s", args.output)

    # Print summary
    fx = result["camera_matrix_l"][0, 0]
    fy = result["camera_matrix_l"][1, 1]
    tx = result["T"][0, 0]
    logger.info("Left focal: fx=%.1f fy=%.1f px", fx, fy)
    logger.info("Baseline (T_x): %.1f mm", abs(tx))


def _wizard_preflight(args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Quick pre-flight before starting the wizard. Returns (ok_to_continue, messages).

    Hard failures block (port busy, output dir unwritable). Soft failures warn
    but don't abort (low disk space, calibration.npz already exists).
    """
    import shutil
    import socket
    import stat

    messages: list[str] = []
    hard_fail = False

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".preflight_probe"
        test_file.write_bytes(b"")
        test_file.unlink()
    except Exception as e:
        messages.append(f"❌ output_dir no escribible: {output_dir} ({e})")
        hard_fail = True
    else:
        messages.append(f"✓ output_dir escribible: {output_dir}")

    calib_out = Path(args.output)
    try:
        calib_out.parent.mkdir(parents=True, exist_ok=True)
        probe = calib_out.parent / ".preflight_calib_probe"
        probe.write_bytes(b"")
        probe.unlink()
    except Exception as e:
        messages.append(f"❌ calibration output dir no escribible: {calib_out.parent} ({e})")
        hard_fail = True
    else:
        messages.append(f"✓ calibration output dir escribible: {calib_out.parent}")

    # Backup existing calibration.npz if present
    if calib_out.exists():
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = calib_out.with_suffix(f".npz.bak.{ts}")
        try:
            import shutil as _sh
            _sh.copy2(calib_out, backup)
            messages.append(f"✓ Backup de calibration previa: {backup.name}")
        except Exception as e:
            messages.append(f"⚠ No pude hacer backup de {calib_out}: {e}")

    # Disk space
    try:
        free_mb = shutil.disk_usage(output_dir).free // (1024 * 1024)
        if free_mb < 500:
            messages.append(f"⚠ Poco espacio libre en disco: {free_mb}MB (<500MB)")
        else:
            messages.append(f"✓ Espacio libre: {free_mb}MB")
    except Exception:
        pass

    # Port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.bind(("0.0.0.0", args.port))
        messages.append(f"✓ Puerto {args.port} libre")
    except OSError:
        # Try to find which PID is holding the port so the operator gets
        # an actionable error instead of "go look it up".
        culprit = _find_port_culprit(args.port)
        if culprit:
            pid, cmd = culprit
            messages.append(
                f"❌ Puerto {args.port} ocupado — PID {pid} ({cmd}). "
                f"Liberalo con: kill {pid}  (o kill -9 {pid} si no muere)"
            )
        else:
            messages.append(
                f"❌ Puerto {args.port} ocupado pero no pude identificar el PID. "
                f"Probá con: sudo lsof -i :{args.port}  o usá --port {args.port + 10}"
            )
        hard_fail = True
    finally:
        sock.close()

    return (not hard_fail), messages


def _find_port_culprit(port: int) -> Optional[tuple[int, str]]:
    """Best-effort lookup of the PID + command holding a TCP port.

    Returns (pid, command) on Linux; None on other platforms or if we can't
    figure it out (e.g. process owned by another UID and we're not root).
    """
    import os as _os
    if not _os.path.exists("/proc"):
        return None
    # Find the inode of the listening socket on this port
    try:
        with open("/proc/net/tcp") as f:
            lines = f.readlines()
    except OSError:
        return None
    target_hex = f"{port:04X}"
    target_inodes: set[str] = set()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 10:
            continue
        local = parts[1]
        state = parts[3]
        # state 0A = LISTEN
        if state != "0A":
            continue
        if not local.endswith(":" + target_hex):
            continue
        target_inodes.add(parts[9])
    if not target_inodes:
        return None
    # Walk /proc/*/fd to find which PID holds that socket inode
    for pid_str in _os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        fd_dir = f"/proc/{pid_str}/fd"
        try:
            for fd in _os.listdir(fd_dir):
                try:
                    target = _os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in target_inodes:
                    try:
                        with open(f"/proc/{pid_str}/cmdline") as f:
                            cmdline = f.read().replace("\x00", " ").strip()
                    except OSError:
                        cmdline = ""
                    short = cmdline.split()[-1] if cmdline else "?"
                    return int(pid_str), short
        except OSError:
            continue
    return None


def _run_ground_truth_phase(
    args: argparse.Namespace, calibration: dict,
) -> Optional[dict]:
    """Ask the operator for a known distance, capture one frame, run SGBM +
    5-zone depth analysis with the fresh calibration, return zones dict for
    the HTML report. Skip cleanly on empty input or failure.
    """
    prompt_msg = (
        "Poné una superficie plana (pared, cartón) a 2 m de las cámaras.\n"
        "Medí con cinta la distancia exacta y escribila en mm (ej: 2000).\n"
        "Si no querés hacer esta validación, tocá Saltear."
    )
    try:
        raw = _ask_operator_ui("ground_truth", prompt_msg).strip()
    except KeyboardInterrupt:
        print()
        return None

    if not raw:
        logger.info("Ground-truth check salteado.")
        return None

    try:
        distance_mm = float(raw)
    except ValueError:
        logger.warning("Distancia inválida (%r), salteando check.", raw)
        return None

    if distance_mm <= 0:
        logger.warning("Distancia debe ser positiva.")
        return None

    logger.info("Capturando para ground-truth a %.0f mm...", distance_mm)

    try:
        from src.vision.capture import StereoCapture
        from src.vision.depth import compute_disparity, create_sgbm
    except ImportError as e:
        logger.warning("No pude importar módulos de depth: %s", e)
        return None

    cap = StereoCapture(
        cam_left_id=args.left, cam_right_id=args.right,
        resolution=tuple(args.resolution), fps=args.fps,
    )
    try:
        cap.open()
        # Countdown
        for i in range(3, 0, -1):
            print(f"  Capturando en {i}...", end="\r", flush=True)
            time.sleep(1)
        print()
        frame_l, frame_r = cap.read()
    except Exception as e:
        logger.warning("Captura ground-truth falló: %s", e)
        try: cap.close()
        except Exception: pass
        return None
    finally:
        try: cap.close()
        except Exception: pass

    try:
        rect_l, rect_r = rectify_pair(frame_l, frame_r, calibration)
    except Exception as e:
        logger.warning("Rectificación ground-truth falló: %s", e)
        return None

    sgbm = create_sgbm()
    disparity = compute_disparity(rect_l, rect_r, sgbm=sgbm, use_wls_filter=False)

    fx = float(calibration["P1"][0, 0])
    baseline_mm = float(np.linalg.norm(calibration["T"]))

    h, w = disparity.shape
    short = min(h, w)
    half = int(short * 0.15 / 2)
    margin = int(short * 0.10)

    zones_coords = {
        "center":       (h // 2, w // 2),
        "top-left":     (margin + half, margin + half),
        "top-right":    (margin + half, w - margin - half),
        "bottom-left":  (h - margin - half, margin + half),
        "bottom-right": (h - margin - half, w - margin - half),
    }

    def _zone_stats(cy: int, cx: int) -> Optional[tuple[float, float, float, float]]:
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        disp_roi = disparity[y1:y2, x1:x2]
        valid = disp_roi[disp_roi > 0.1]
        if len(valid) == 0:
            return None
        depths = fx * baseline_mm / valid
        fill_pct = 100.0 * len(valid) / disp_roi.size
        err_pct = (float(np.median(depths)) - distance_mm) / distance_mm * 100
        return (float(np.median(depths)), float(np.std(depths)), err_pct, fill_pct)

    zones: dict = {}
    for name, (cy, cx) in zones_coords.items():
        zones[name] = _zone_stats(cy, cx)

    # Thresholds: <5% @ 2m, <10% @ 3m, <2× edge/center ratio
    d_m = distance_mm / 1000
    if d_m <= 2.0:
        center_threshold = 5.0
    elif d_m >= 3.0:
        center_threshold = 10.0
    else:
        center_threshold = 5.0 + (d_m - 2.0) * 5.0

    center = zones.get("center")
    edges = [v for k, v in zones.items() if k != "center" and v is not None]
    edge_ratio = float("nan")
    center_err_abs = float("nan")
    overall_pass = False
    if center is not None:
        center_err_abs = abs(center[2])
        if edges:
            edge_errs = [abs(v[2]) for v in edges]
            edge_ratio = max(edge_errs) / max(center_err_abs, 0.1)
        checks_pass = center_err_abs <= center_threshold and edge_ratio <= 2.0
        overall_pass = bool(checks_pass)

    zones["_pass"] = overall_pass
    zones["_distance_mm"] = distance_mm
    zones["_center_err"] = center_err_abs
    zones["_edge_ratio"] = edge_ratio

    # Depth heatmap for the report — scene image with disparity-colored overlay.
    try:
        valid_mask = disparity > 0.1
        norm = np.zeros_like(disparity, dtype=np.uint8)
        if valid_mask.any():
            vmin = float(disparity[valid_mask].min())
            vmax = float(disparity[valid_mask].max())
            if vmax > vmin:
                norm[valid_mask] = np.clip(
                    (disparity[valid_mask] - vmin) / (vmax - vmin) * 255, 0, 255,
                ).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        heat[~valid_mask] = (40, 40, 40)
        # Side by side: rectified L + heatmap
        side = np.hstack([rect_l, heat])
        # Mark each zone center with a box + label
        for name, (cy, cx) in zones_coords.items():
            px_x = cx + rect_l.shape[1]  # shift into heatmap side
            cv2.rectangle(
                side, (px_x - half, cy - half), (px_x + half, cy + half),
                (255, 255, 255), 2,
            )
            cv2.putText(
                side, name, (px_x - half, cy - half - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        gt_viz_path = Path(args.output).parent / "ground_truth_depth.png"
        cv2.imwrite(str(gt_viz_path), side)
        zones["_image_path"] = str(gt_viz_path)
    except Exception as e:
        logger.warning("No pude guardar viz de ground-truth: %s", e)

    logger.info(
        "Ground-truth: centro err=%.2f%% (umbral %.1f%%), borde/centro=%.2f× (umbral 2×) → %s",
        center_err_abs, center_threshold, edge_ratio,
        "PASS" if overall_pass else "FAIL",
    )
    return zones


def cmd_wizard(args: argparse.Namespace) -> None:
    """One-shot calibration wizard: preflight → guided capture → calibrate →
    verify → ground-truth check → report."""
    from src.vision.report import generate_html_report, save_report

    if getattr(args, "low_light", False):
        _apply_low_light_overrides()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0 — pre-flight
    logger.info("=" * 60)
    logger.info("WIZARD PHASE 0/4 — Pre-flight checks")
    logger.info("=" * 60)
    ok, messages = _wizard_preflight(args)
    for m in messages:
        logger.info(m)
    if not ok:
        logger.error("Pre-flight falló. Resolvé los items marcados con ❌ y reintentá.")
        sys.exit(1)

    # Phase 1 — guided capture
    args.guided = True
    args.grid = "rectangular"
    args.manual = False
    args.per_cell = 0
    args.cooldown = 1.5
    args.count = 20
    logger.info("=" * 60)
    logger.info("WIZARD PHASE 1/4 — Captura guiada")
    logger.info("=" * 60)
    _run_guided_capture(args)
    _set_post_capture_phase(
        "processing",
        "Procesando capturas, analizando cobertura y detectando esquinas "
        "ChArUco en cada par...",
        progress_pct=10,
    )

    # Collect captured pairs (those that survived UNDOs). Pose IDs are tracked
    # in _guided_state.captured_pairs — fall back to ordinal if state missing.
    left_files = sorted(output_dir.glob("left_*.png"))
    pose_id_by_path: dict[str, str] = {}
    if _guided_state is not None:
        for lp, _rp, pid in _guided_state.captured_pairs:
            pose_id_by_path[str(lp)] = pid

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pair_meta: list[tuple[Path, Path, str, int, int]] = []
    captured_pose_ids: list[str] = []
    dict_id = _resolve_aruco_dict(getattr(args, "aruco_dict", "DICT_4X4_100"))
    board_tmp = create_charuco_board(
        (args.columns, args.rows), args.square_length, args.marker_length, dict_id,
        legacy_pattern=getattr(args, "legacy_pattern", True),
    )
    for lf in left_files:
        rf = lf.parent / lf.name.replace("left_", "right_")
        if not rf.exists():
            continue
        il = cv2.imread(str(lf))
        ir = cv2.imread(str(rf))
        if il is None or ir is None:
            continue
        pairs.append((il, ir))
        pose_id = pose_id_by_path.get(str(lf), lf.stem.replace("left_", "pose-"))
        captured_pose_ids.append(pose_id)
        # Re-detect with the SAME lenient mode the calibration step will use
        # so this sanity count matches what calibrate_stereo will actually see.
        corners_l, _ = detect_charuco_corners(il, board_tmp, lenient=True)
        corners_r, _ = detect_charuco_corners(ir, board_tmp, lenient=True)
        n_l = len(corners_l) if corners_l is not None else 0
        n_r = len(corners_r) if corners_r is not None else 0
        pair_meta.append((lf, rf, pose_id, n_l, n_r))

    min_captures = getattr(args, "min_captures", 15)
    if len(pairs) < min_captures:
        msg = (
            f"Capturas insuficientes ({len(pairs)}). Necesitás al menos "
            f"{min_captures} para calibrar."
        )
        logger.error(msg)
        _set_post_capture_phase(
            "complete", msg, verdict="FAIL", report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    # Pre-calibration sanity: count pairs where BOTH cameras produced a
    # usable detection (≥ 8 common corners is what calibrate_stereo's
    # _detect_all_pairs gates on). If too many pairs fail this re-detect,
    # the live capture loop accepted frames that won't actually feed the
    # calibration math — and we'd silently calibrate on a tiny subset
    # (2026-04-28 session: 7 of 15 valid → degenerate fit).
    valid_both = 0
    invalid_lines: list[str] = []
    for lf, rf, pid, n_l, n_r in pair_meta:
        if n_l >= 8 and n_r >= 8:
            valid_both += 1
        else:
            invalid_lines.append(
                f"{pid}: L={n_l} corners, R={n_r} corners (need ≥8 en ambas)"
            )
    detect_rate = valid_both / len(pairs) if pairs else 0.0
    min_rate = getattr(args, "min_detect_rate", 0.7)
    logger.info(
        "Pre-calibration sanity: %d/%d pares con detección válida en ambas "
        "cámaras (%.0f%%, umbral %.0f%%)",
        valid_both, len(pairs), detect_rate * 100, min_rate * 100,
    )
    if detect_rate < min_rate:
        offending = "\n".join(f"• {l}" for l in invalid_lines)
        msg = (
            f"Solo {valid_both} de {len(pairs)} pares ({detect_rate*100:.0f}%) "
            f"sobrevivieron la re-detección. Umbral mínimo: {min_rate*100:.0f}%. "
            f"Calibrar con esta data va a producir un fit degenerado.\n\n"
            f"Pares con detección incompleta:\n{offending}\n\n"
            f"Recapturá las poses con problemas (lens limpio, foco, exposición) "
            f"o bajá --min-detect-rate si entendés el riesgo."
        )
        logger.error("❌ Pre-calibration sanity falló")
        for line in invalid_lines:
            logger.error("    %s", line)
        _set_post_capture_phase(
            "complete", msg, verdict="FAIL", report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    # Diversity check — warn if the captured set is degenerate
    coverage = analyze_pose_coverage(
        captured_pose_ids,
        all_poses=default_pose_sequence(
            near_mm=args.dist_near_mm, mid_mm=args.dist_mid_mm, far_mm=args.dist_far_mm,
        ),
    )
    logger.info(
        "Cobertura: distancia %s · tilts %s",
        coverage["by_distance"], coverage["by_tilt_axis"],
    )
    # Critical gaps are hard-blockers: missing entire pose groups or
    # distance bands produces degenerate calibrations (low RMS but
    # geometrically wrong) — exactly the failure mode of the 2026-04-28
    # session. Operator can override with --force-degenerate-coverage.
    critical_gaps = coverage.get("critical", [])
    force_flag = getattr(args, "force_degenerate_coverage", False)
    if critical_gaps and not force_flag:
        logger.error("❌ Coverage crítico insuficiente — calibración bloqueada:")
        for c in critical_gaps:
            logger.error("    - %s", c)
        msg = (
            "Coverage crítico insuficiente. La calibración va a producir un fit "
            "degenerado (RMS bajo pero geométricamente incorrecto) que falla "
            "ground-truth. Recapturá las poses faltantes:\n\n"
            + "\n".join(f"• {c}" for c in critical_gaps)
            + "\n\nSi entendés los riesgos y querés forzarla igual, "
            "agregá --force-degenerate-coverage al wizard."
        )
        _set_post_capture_phase(
            "complete", msg, verdict="FAIL", report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    soft_warnings = coverage.get("warnings", [])
    if soft_warnings:
        logger.warning("⚠ Diversidad limitada en el set de capturas:")
        for w in soft_warnings:
            logger.warning("    - %s", w)
        warnings_text = "\n".join(f"• {w}" for w in soft_warnings)
        prompt_msg = (
            "Las capturas no tienen suficiente diversidad de pose:\n\n"
            f"{warnings_text}\n\n"
            "Podés continuar igual (la calibración va a converger con más "
            "error) o cancelar para recapturar."
        )
        answer = _ask_operator_ui("diversity", prompt_msg).strip().lower()
        if answer not in ("y", "yes", "s", "si", "sí"):
            logger.info("Calibración cancelada — podés recapturar y volver a correr el wizard.")
            _set_post_capture_phase(
                "complete",
                "Calibración cancelada por el operador (diversidad limitada).",
                verdict="FAIL", report_available=False,
            )
            time.sleep(10)
            sys.exit(0)

    # Phase 2 — calibrate
    logger.info("=" * 60)
    logger.info("WIZARD PHASE 2/4 — Calibración estéreo con %d pares", len(pairs))
    logger.info("=" * 60)
    _set_post_capture_phase(
        "calibrating",
        f"Calculando parámetros intrínsecos y extrínsecos con {len(pairs)} "
        f"pares estéreo (puede tardar 20-60 segundos)...",
        progress_pct=40,
    )
    try:
        result = calibrate_stereo(
            pairs,
            board_size=(args.columns, args.rows),
            square_length=args.square_length,
            marker_length=args.marker_length,
            dict_id=dict_id,
            min_pairs=min_captures,
            legacy_pattern=getattr(args, "legacy_pattern", True),
        )
    except ValueError as e:
        msg = f"Calibración falló: {e}"
        logger.error(msg)
        _set_post_capture_phase(
            "complete", msg, verdict="FAIL", report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    calib_out = Path(args.output)
    save_calibration(result, str(calib_out))

    fx_l = float(result["camera_matrix_l"][0, 0])
    baseline_mm = float(abs(result["T"][0, 0]))
    logger.info("fx=%.1f px   baseline=%.2f mm (diseño 140mm)", fx_l, baseline_mm)

    board_for_residuals = create_charuco_board(
        (args.columns, args.rows), args.square_length, args.marker_length, dict_id,
        legacy_pattern=getattr(args, "legacy_pattern", True),
    )

    # Phase 3 — verify (epipolar) + per-pair residuals
    logger.info("=" * 60)
    logger.info("WIZARD PHASE 3/4 — Verificación + residuales por par")
    logger.info("=" * 60)
    _set_post_capture_phase(
        "verifying",
        "Verificando alineación epipolar y calculando residuales por par...",
        progress_pct=70,
    )

    epi_path: Optional[Path] = None
    if pairs:
        rect_l, rect_r = rectify_pair(pairs[0][0], pairs[0][1], result)
        combined = np.hstack([rect_l, rect_r])
        for y in range(0, combined.shape[0], 30):
            color = (0, 255, 0) if (y // 30) % 2 == 0 else (0, 200, 255)
            cv2.line(combined, (0, y), (combined.shape[1], y), color, 1)
        epi_path = calib_out.parent / "verify_epipolar.png"
        cv2.imwrite(str(epi_path), combined)
        logger.info("Verificación epipolar: %s", epi_path)

    rms_est = None
    try:
        rms_est = _residual_estimate(pairs, board_for_residuals, result)
    except Exception:
        pass

    try:
        per_pair = compute_per_pair_residuals(pairs, board_for_residuals, result)
        valid = [r["rms"] for r in per_pair if r["rms"] == r["rms"]]
        if valid:
            median = sorted(valid)[len(valid) // 2]
            n_outliers = sum(1 for v in valid if v > 2 * max(median, 1.0))
            logger.info(
                "Per-pair residuals: mediana=%.2fpx · %d outliers (>2× mediana)",
                median, n_outliers,
            )
    except Exception as e:
        logger.warning("Per-pair residuals failed: %s", e)
        per_pair = None

    # Phase 4 — ground-truth depth check
    logger.info("=" * 60)
    logger.info("WIZARD PHASE 4/4 — Ground-truth depth check (opcional)")
    _set_post_capture_phase(
        "ground_truth",
        "Fase opcional de validación con distancia conocida — mirá la "
        "terminal para continuar.",
        progress_pct=85,
    )
    logger.info("=" * 60)
    diagnose_zones = _run_ground_truth_phase(args, result)

    # Final report
    ts = _dt.datetime.now()
    gt_image = None
    if diagnose_zones and diagnose_zones.get("_image_path"):
        gt_image = Path(diagnose_zones["_image_path"])
    html = generate_html_report(
        calibration=result,
        diagnose_zones=diagnose_zones,
        capture_pairs=pair_meta,
        device_id=args.device_id,
        rms_stereo=rms_est,
        timestamp=ts,
        per_pair_residuals=per_pair,
        epipolar_image=epi_path,
        ground_truth_image=gt_image,
    )
    report_name = f"calibration_report_{args.device_id}_{ts:%Y%m%d_%H%M%S}.html"
    report_path = save_report(html, calib_out.parent / report_name)

    logger.info("=" * 60)
    logger.info("WIZARD COMPLETO")
    logger.info("  Calibración: %s", calib_out)
    logger.info("  Reporte HTML: %s", report_path)
    logger.info("  Baseline: %.2f mm (diseño 140mm, Δ %+.2f mm)",
                baseline_mm, baseline_mm - 140.0)
    if rms_est is not None:
        logger.info("  RMS estimado: %.3f px", rms_est)
    if diagnose_zones and diagnose_zones.get("_pass") is not None:
        logger.info("  Ground-truth depth: %s",
                    "PASS" if diagnose_zones["_pass"] else "FAIL")
    logger.info("=" * 60)

    if abs(baseline_mm - 140.0) > 5.0:
        logger.warning(
            "⚠ Baseline estimada fuera de tolerancia (%.2fmm vs diseño 140mm, Δ %+.2fmm). "
            "La baseline se estima del set de capturas — si son pocas o poco "
            "diversas el solver converge con error.",
            baseline_mm, baseline_mm - 140.0,
        )

    # Point the capture-UI /report endpoint at the saved report so the browser
    # can auto-open it via the existing HTTP server. We also keep serving it
    # via QR (different port) for distribution to a phone.
    global _report_path_for_http
    _report_path_for_http = report_path

    overall_pass = True
    if diagnose_zones and diagnose_zones.get("_pass") is False:
        overall_pass = False
    if abs(baseline_mm - 140.0) > 5.0:
        overall_pass = False
    verdict = "PASS" if overall_pass else "FAIL"

    completion_msg = (
        f"Reporte generado ({report_path.name}). "
        f"Baseline: {baseline_mm:.2f}mm (diseño 140mm). "
    )
    if rms_est is not None:
        completion_msg += f"RMS estimado: {rms_est:.2f}px. "
    completion_msg += "Se abre el reporte en una pestaña nueva."
    _set_post_capture_phase(
        "complete",
        completion_msg,
        verdict=verdict,
        report_available=True,
    )

    # Grace period so the browser picks up the "complete" status and opens
    # the report in a new tab before the daemon HTTP server dies with main.
    time.sleep(10)


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify calibration by showing rectified pairs with epipolar lines."""
    cal = load_calibration(args.calibration)
    input_dir = Path(args.input_dir)

    left_files = sorted(input_dir.glob("left_*.png"))
    if not left_files:
        logger.error("No images found in %s", input_dir)
        sys.exit(1)

    # Use first pair for verification
    lf = left_files[0]
    rf = lf.parent / lf.name.replace("left_", "right_")
    img_l = cv2.imread(str(lf))
    img_r = cv2.imread(str(rf))

    rect_l, rect_r = rectify_pair(img_l, img_r, cal)

    # Draw horizontal epipolar lines on the rectified pair
    combined = np.hstack([rect_l, rect_r])
    h = combined.shape[0]
    for y in range(0, h, 30):
        color = (0, 255, 0) if (y // 30) % 2 == 0 else (0, 200, 255)
        cv2.line(combined, (0, y), (combined.shape[1], y), color, 1)

    output_path = str(Path(args.calibration).parent / "verify_epipolar.png")
    cv2.imwrite(output_path, combined)
    logger.info("Verification image saved to %s", output_path)
    logger.info(
        "Check that corresponding features lie on the same horizontal line."
    )


def cmd_reset(args: argparse.Namespace) -> None:
    """Wipe captures + session.json + calibration.npz so the next wizard
    run starts from a clean slate. Useful when a session went sideways
    (degenerate captures, mismatched resolution, corrupted state) and you
    don't want to debug — you just want to start over.
    """
    output_dir = Path(args.output_dir)
    calib_out = Path(args.output)
    items_to_remove: list[tuple[str, Path]] = []

    if output_dir.exists():
        captures = (
            list(output_dir.glob("left_*.png"))
            + list(output_dir.glob("right_*.png"))
        )
        for p in captures:
            items_to_remove.append(("capture", p))
        sidecar = output_dir / "session.json"
        if sidecar.exists():
            items_to_remove.append(("session", sidecar))

    if calib_out.exists():
        items_to_remove.append(("calibration", calib_out))

    if not items_to_remove:
        logger.info("No hay nada que limpiar — directorios ya vacíos.")
        return

    by_kind: dict[str, int] = {}
    for kind, _ in items_to_remove:
        by_kind[kind] = by_kind.get(kind, 0) + 1
    summary = ", ".join(f"{n} {k}{'s' if n != 1 else ''}" for k, n in by_kind.items())

    if not args.yes:
        logger.info(
            "Voy a borrar: %s. Pasá --yes para confirmar.",
            summary,
        )
        sys.exit(1)

    logger.info("Borrando: %s", summary)
    for kind, p in items_to_remove:
        try:
            p.unlink()
        except OSError as e:
            logger.warning("    No pude borrar %s: %s", p, e)
    logger.info("Reset completo.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stereo calibration tool for People Counter"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate-board ---
    p_board = sub.add_parser("generate-board", help="Generate printable ChArUco board")
    p_board.add_argument("--output", default="calibration/charuco_board.png")
    p_board.add_argument("--columns", type=int, default=DEFAULT_BOARD_SIZE[0], help=f"Board columns (default {DEFAULT_BOARD_SIZE[0]})")
    p_board.add_argument("--rows", type=int, default=DEFAULT_BOARD_SIZE[1], help=f"Board rows (default {DEFAULT_BOARD_SIZE[1]})")
    p_board.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH, help=f"Square side in mm (default {DEFAULT_SQUARE_LENGTH})")
    p_board.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH, help=f"Marker side in mm (default {DEFAULT_MARKER_LENGTH})")
    p_board.add_argument("--dict", dest="aruco_dict", default="DICT_4X4_100",
                        help="ArUco dictionary name (e.g. DICT_4X4_100, DICT_5X5_100). Default DICT_4X4_100")
    p_board.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use pre-4.6 ChArUco marker enumeration. Default True matches calib.io.")
    p_board.add_argument("--width", type=int, default=4961, help="Image width (px) — default A3 landscape @ 300 DPI")
    p_board.add_argument("--height", type=int, default=3508, help="Image height (px) — default A3 landscape @ 300 DPI")
    p_board.set_defaults(func=cmd_generate_board)

    # --- capture ---
    p_cap = sub.add_parser("capture", help="Interactive capture with HTTP preview")
    p_cap.add_argument("--left", type=int, default=0, help="Left camera index (lente izquierda mirando desde la cámara hacia la escena). Default 0 matches the fleet wiring.")
    p_cap.add_argument("--right", type=int, default=1, help="Right camera index. Default 1.")
    p_cap.add_argument("--resolution", type=int, nargs=2, default=[2304, 1296],
                        help="Capture resolution. Default 2304x1296 (IMX708 binned, "
                             "full FOV, ~4x faster ChArUco detection than full-res). "
                             "Runtime config must match this — see config.example.yaml.")
    p_cap.add_argument("--fps", type=int, default=5)
    p_cap.add_argument("--output-dir", default="./calibration/captures")
    p_cap.add_argument("--count", type=int, default=30, help="Minimum number of pairs")
    p_cap.add_argument("--per-cell", type=int, default=0,
                        help="Max captures per grid cell (0=unlimited). Stops cell when reached.")
    p_cap.add_argument("--cooldown", type=float, default=1.5,
                        help="Seconds to wait after each capture before next one")
    p_cap.add_argument("--pose-timeout-sec", type=float,
                        default=SKIP_POSE_TIMEOUT_SEC,
                        help=f"Seconds before an uncaptured pose is auto-"
                             f"skipped (default {SKIP_POSE_TIMEOUT_SEC:.0f}).")
    p_cap.add_argument("--port", type=int, default=8080, help="HTTP preview port")
    p_cap.add_argument("--columns", type=int, default=DEFAULT_BOARD_SIZE[0], help=f"Board columns (default {DEFAULT_BOARD_SIZE[0]})")
    p_cap.add_argument("--rows", type=int, default=DEFAULT_BOARD_SIZE[1], help=f"Board rows (default {DEFAULT_BOARD_SIZE[1]})")
    p_cap.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH, help=f"Square side in mm (default {DEFAULT_SQUARE_LENGTH})")
    p_cap.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH, help=f"Marker side in mm (default {DEFAULT_MARKER_LENGTH})")
    p_cap.add_argument("--dict", dest="aruco_dict", default="DICT_4X4_100",
                        help="ArUco dictionary name. Default DICT_4X4_100 (the final board)")
    p_cap.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use pre-4.6 ChArUco marker enumeration. Default True matches calib.io.")
    p_cap.add_argument("--grid", choices=["rectangular", "circular"], default="rectangular",
                        help="Legacy coverage grid (ignored when --guided is on)")
    p_cap.add_argument("--manual", action="store_true",
                        help="Manual trigger: press Enter in terminal to capture (board must be detected). "
                             "Use with a rigid support for the board to eliminate motion between L/R frames.")
    p_cap.add_argument("--guided", action="store_true",
                        help="Guided capture: shows ghost silhouettes of target poses, "
                             "auto-captures when aligned and stable. Recommended default.")
    p_cap.add_argument("--dist-near-mm", type=float, default=DEFAULT_DIST_NEAR_MM,
                        help=f"Near distance for pose sequence (default {DEFAULT_DIST_NEAR_MM:.0f}mm)")
    p_cap.add_argument("--dist-mid-mm", type=float, default=DEFAULT_DIST_MID_MM,
                        help=f"Mid distance for pose sequence (default {DEFAULT_DIST_MID_MM:.0f}mm)")
    p_cap.add_argument("--dist-far-mm", type=float, default=DEFAULT_DIST_FAR_MM,
                        help=f"Far distance for pose sequence (default {DEFAULT_DIST_FAR_MM:.0f}mm)")
    p_cap.add_argument("--resume", action="store_true",
                        help="Continue a previous guided session from its session.json")
    p_cap.set_defaults(func=cmd_capture)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Run stereo calibration")
    p_cal.add_argument("--input-dir", required=True, help="Dir with left_/right_ images")
    p_cal.add_argument("--output", default="calibration.npz")
    p_cal.add_argument("--columns", type=int, default=DEFAULT_BOARD_SIZE[0], help=f"Board columns (default {DEFAULT_BOARD_SIZE[0]})")
    p_cal.add_argument("--rows", type=int, default=DEFAULT_BOARD_SIZE[1], help=f"Board rows (default {DEFAULT_BOARD_SIZE[1]})")
    p_cal.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH, help=f"Square side in mm (default {DEFAULT_SQUARE_LENGTH})")
    p_cal.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH, help=f"Marker side in mm (default {DEFAULT_MARKER_LENGTH})")
    p_cal.add_argument("--dict", dest="aruco_dict", default="DICT_4X4_100",
                        help="ArUco dictionary name. Default DICT_4X4_100")
    p_cal.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use pre-4.6 ChArUco marker enumeration. Default True matches calib.io.")
    p_cal.set_defaults(func=cmd_calibrate)

    # --- verify ---
    p_ver = sub.add_parser("verify", help="Verify calibration visually")
    p_ver.add_argument("--calibration", required=True, help="Path to calibration.npz")
    p_ver.add_argument("--input-dir", required=True, help="Dir with image pairs")
    p_ver.set_defaults(func=cmd_verify)

    # --- wizard (one-shot: guided capture + calibrate + verify + report) ---
    p_wiz = sub.add_parser("wizard",
        help="One-shot guided calibration: capture -> calibrate -> verify -> report")
    p_wiz.add_argument("--device-id", default="unknown", help="Device identifier for the report")
    p_wiz.add_argument("--output", default="calibration.npz", help="Output calibration file")
    p_wiz.add_argument("--output-dir", default="./calibration/captures", help="Dir for captured pairs")
    p_wiz.add_argument("--left", type=int, default=0)
    p_wiz.add_argument("--right", type=int, default=1)
    p_wiz.add_argument("--resolution", type=int, nargs=2, default=[2304, 1296],
                        help="Capture resolution. Default 2304x1296 (binned, full FOV). "
                             "Full-res 4608x2592 makes per-frame detection 4x slower, "
                             "which on Pi 5 dropped FPS to <2 and caused intermittent "
                             "marker detection at the FOV edges (B/C/D corner poses) — "
                             "the calibration came out degenerate (low RMS but failing "
                             "ground-truth) on 2026-04-28 lab session because of that. "
                             "Runtime config must match this resolution.")
    p_wiz.add_argument("--fps", type=int, default=5)
    p_wiz.add_argument("--port", type=int, default=8080)
    p_wiz.add_argument("--columns", type=int, default=DEFAULT_BOARD_SIZE[0])
    p_wiz.add_argument("--rows", type=int, default=DEFAULT_BOARD_SIZE[1])
    p_wiz.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH)
    p_wiz.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH)
    p_wiz.add_argument("--dict", dest="aruco_dict", default="DICT_4X4_100",
                        help="ArUco dictionary name. Default DICT_4X4_100 (the final board)")
    p_wiz.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use pre-4.6 ChArUco marker enumeration. Default True matches calib.io.")
    p_wiz.add_argument("--min-captures", type=int, default=15,
                        help="Minimum pair count needed to run calibration. "
                             "Default 15 (statistically robust). Lower for "
                             "dry-runs / debugging only.")
    p_wiz.add_argument("--pose-timeout-sec", type=float,
                        default=SKIP_POSE_TIMEOUT_SEC,
                        help=f"Seconds before an uncaptured pose is auto-"
                             f"skipped (default {SKIP_POSE_TIMEOUT_SEC:.0f}). "
                             f"Bump to 180+ when the board is tripod-mounted "
                             f"and needs time to reposition between poses.")
    p_wiz.add_argument("--tolerance", choices=["loose", "normal", "strict"],
                        default="normal",
                        help="Capture strictness preset. "
                             "loose = permissive (PoC / small board / tricky "
                             "lighting, bootstrap 50px / tight 25px). "
                             "normal = defaults tuned for the A3 final board "
                             "(bootstrap 25px / tight 12px). "
                             "strict = production-grade (bootstrap 15px / "
                             "tight 8px, minimises outliers).")
    p_wiz.add_argument("--align-tol-loose-px", type=float, default=None,
                        help="Fine-grained override of the bootstrap alignment "
                             "tolerance (takes precedence over --tolerance).")
    p_wiz.add_argument("--align-tol-tight-px", type=float, default=None,
                        help="Fine-grained override of the post-bootstrap "
                             "alignment tolerance.")
    p_wiz.add_argument("--dist-near-mm", type=float, default=DEFAULT_DIST_NEAR_MM,
                        help=f"Near distance for pose sequence (default {DEFAULT_DIST_NEAR_MM:.0f}mm)")
    p_wiz.add_argument("--dist-mid-mm", type=float, default=DEFAULT_DIST_MID_MM,
                        help=f"Mid distance for pose sequence (default {DEFAULT_DIST_MID_MM:.0f}mm)")
    p_wiz.add_argument("--dist-far-mm", type=float, default=DEFAULT_DIST_FAR_MM,
                        help=f"Far distance for pose sequence (default {DEFAULT_DIST_FAR_MM:.0f}mm)")
    # --- reset ---
    p_reset = sub.add_parser("reset",
        help="Wipe captures + session.json + calibration.npz for a clean restart")
    p_reset.add_argument("--output", default="calibration.npz",
                         help="Calibration file to remove (default: calibration.npz)")
    p_reset.add_argument("--output-dir", default="./calibration/captures",
                         help="Captures dir to clean (default: ./calibration/captures)")
    p_reset.add_argument("--yes", action="store_true",
                         help="Confirm deletion (without this the command lists what would be removed)")
    p_reset.set_defaults(func=cmd_reset)

    p_wiz.add_argument("--resume", action="store_true",
                        help="Continue a previous wizard session — skips already-captured poses")
    p_wiz.add_argument("--force-degenerate-coverage", action="store_true",
                        help="Bypass the critical-coverage block. By default the "
                             "wizard refuses to calibrate when entire pose groups "
                             "(A/B/C/D) or distance bands (near/mid/far) are "
                             "missing, because that produces a low-RMS but "
                             "geometrically wrong fit (2026-04-28 session). Use "
                             "this flag only if you understand the trade-off.")
    p_wiz.add_argument("--min-detect-rate", type=float, default=0.7,
                        help="Pre-calibration sanity threshold: if fewer than this "
                             "fraction of captured pairs survive a strict "
                             "re-detection pass, the wizard aborts before running "
                             "fisheye.calibrate. Default 0.7 (70%%).")
    p_wiz.add_argument("--low-light", action="store_true",
                        help="PoC mode for low-light / small-room runs. Relaxes "
                             "frame-quality gates (exposure, blur, corner sharpness, "
                             "L/R brightness balance) so captures aren't rejected for "
                             "scene conditions. The resulting calibration will NOT "
                             "be valid for production depth — use only to validate "
                             "the wizard end-to-end.")
    p_wiz.set_defaults(func=cmd_wizard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

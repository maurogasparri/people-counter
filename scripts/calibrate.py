#!/usr/bin/env python3
"""Tool CLI para calibración de cámaras estéreo usando patrones ChArUco.

Uso:
    # Paso 1: Capturar pares de imágenes (interactivo, con preview HTTP)
    python scripts/calibrate.py capture --count 30

    # Paso 2: Correr calibración a partir de los pares capturados
    python scripts/calibrate.py calibrate --input-dir ./calibration/captures --output calibration.npz

    # Paso 3: Verificar la calibración (dibuja líneas epipolares sobre el par rectificado)
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

# Agregar la raíz del proyecto al path para los imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.hardware import (
    FLEET_DEFAULTS as _FALLBACK_HW,
    HardwareParams,
    load_hardware_params,
)
import src.vision.calibration as _calib_mod
from src.vision.calibration import (
    ALIGN_CENTER_TOL_PX,
    ALIGN_MEAN_ERR_TOL_PX_LOOSE,
    ALIGN_MEAN_ERR_TOL_PX_TIGHT,
    ALIGN_MATCHED_MIN_LOOSE,
    ALIGN_MATCHED_MIN_TIGHT,
    DEFAULT_DIST_FAR_MM,
    DEFAULT_DIST_MID_MM,
    DEFAULT_DIST_NEAR_MM,
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
    detect_charuco_dual_pass,
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
    """Resuelve un nombre de dict ArUco provisto por el usuario a la constante cv2."""
    dict_attr = name if name.startswith("DICT_") else f"DICT_{name}"
    if not hasattr(cv2.aruco, dict_attr):
        raise SystemExit(f"ArUco dict desconocido: {name}")
    return getattr(cv2.aruco, dict_attr)


# Hardware params canónicos del device — leídos al main(). Inicializado
# con fleet defaults para que las funciones top-level + argparse defaults
# tengan valores razonables durante import.
HW: HardwareParams = _FALLBACK_HW


_TOLERANCE_PRESETS = {
    "loose": (50.0, 25.0),
    "normal": (ALIGN_MEAN_ERR_TOL_PX_LOOSE, ALIGN_MEAN_ERR_TOL_PX_TIGHT),
    "strict": (15.0, 8.0),
}


def _captured_have_distance_diversity(
    captured_pairs: list,
    poses: list,
    near_mm: float,
    mid_mm: float,
    far_mm: float,
) -> bool:
    """True sii las capturas cubren las 3 bandas de distancia (near/mid/far).

    El bootstrap fittea los intrínsecos a partir de las capturas hasta
    el momento. Si se hace solo con capturas de la banda near, converge
    a un focal que fittea esas profundidades pero extrapola mal — y ese
    focal incorrecto después maneja el render del ghost para poses far,
    diciéndole al operador que ponga el board a una distancia que no
    coincide con la marca en el piso. Diferir el bootstrap hasta tener
    evidencia de cada banda.
    """
    if not captured_pairs:
        return False
    pose_distance = {p.id: p.tvec_mm[2] for p in poses}
    mid_lo = (near_mm + mid_mm) / 2
    far_lo = (mid_mm + far_mm) / 2
    has_near = has_mid = has_far = False
    for entry in captured_pairs:
        # los items de captured_pairs son tuplas (left_path, right_path, pose_id)
        pose_id = entry[2] if len(entry) >= 3 else None
        z = pose_distance.get(pose_id)
        if z is None:
            continue
        if z < mid_lo:
            has_near = True
        elif z < far_lo:
            has_mid = True
        else:
            has_far = True
    return has_near and has_mid and has_far


def _apply_low_light_overrides() -> None:
    """Afloja los gates de calidad de frame para corridas PoC en escenas
    de luz baja / cuartos chicos.

    Muta las constantes que lee ``assess_frame_quality``. NO usar para
    calibración productiva — los pares aceptados bajo estos thresholds
    tendrán SNR pobre y el .npz resultante no va a sobrevivir la
    verificación ground-truth.
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
    """Devuelve (loose_px, tight_px). Los overrides por flag ganan al preset."""
    preset = getattr(args, "tolerance", "normal")
    base_loose, base_tight = _TOLERANCE_PRESETS.get(
        preset,
        _TOLERANCE_PRESETS["normal"],
    )
    override_loose = getattr(args, "align_tol_loose_px", None)
    override_tight = getattr(args, "align_tol_tight_px", None)
    loose = override_loose if override_loose is not None else base_loose
    tight = override_tight if override_tight is not None else base_tight
    return loose, tight


def _ask_operator_ui(prompt_type: str, message: str, options: dict = None) -> str:
    """Renderiza un prompt interactivo en el browser y bloquea hasta
    que /wizard-input recibe la respuesta del operador. Devuelve el
    string raw que el operador envió.

    prompt_type maneja qué controles renderiza el JS:
      * "diversity" → dos botones: Continuar / Cancelar
      * "ground_truth" → input-text + botones Validar / Saltear
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
            f"</div>"
        )
    elif prompt_type == "ground_truth":
        html = (
            f'<div data-phase="prompt" data-prompt-type="ground_truth">'
            f'<div style="color:#3fb6f0;font-size:20px;font-weight:700;'
            f'margin-bottom:10px">Validación ground-truth (opcional)</div>'
            f'<div style="color:#eee;font-size:14px;line-height:1.6;'
            f'white-space:pre-line">{message}</div>'
            f"</div>"
        )
    else:
        html = f'<div data-phase="prompt">{message}</div>'

    with _post_capture_lock:
        _post_capture_html = html

    _wizard_input_event.wait()
    return _wizard_input_value


def _set_post_capture_phase(
    phase: str,
    message: str,
    progress_pct: int | None = None,
    verdict: str | None = None,
    report_available: bool = False,
) -> None:
    """Pushea un mensaje de progreso al panel del browser mientras el
    wizard corre las fases post-captura. El atributo embedded
    data-phase permite que el JS detecte completion y auto-abra el
    reporte.
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
                "background:#2980b9;color:#fff;text-decoration:none;"
                'border-radius:6px;font-weight:600;font-size:14px">'
                "Abrir reporte en nueva pestaña</a>"
            )
        has_report_attr = "1" if report_available else "0"
        html = (
            f'<div data-phase="complete" data-has-report="{has_report_attr}">'
            f'<div style="color:{verdict_color};font-size:22px;font-weight:700;'
            f'margin-bottom:12px">Calibración finalizada'
            + (f" — {verdict}" if verdict else "")
            + f"</div>"
            f'<div style="color:#aaa;font-size:13px;line-height:1.6">'
            f"{message}</div>"
            f"{report_btn}"
            f"</div>"
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
            f"{message}</div>"
            f"{progress_html}"
            f"</div>"
        )
    with _post_capture_lock:
        _post_capture_html = html


# ---------------------------------------------------------------------------
# Preview HTTP para la captura
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
                b"<button id=\"cap\" onclick=\"fetch('/capture',{method:'POST'})\""
                b' style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);'
                b"padding:16px 32px;font-size:20px;background:#28a745;color:white;"
                b'border:none;border-radius:8px;cursor:pointer">CAPTURE</button>'
                if _manual_enabled
                else b""
            )
            self.wfile.write(
                b"""<!DOCTYPE html>
<html><head><title>Calibration Capture</title>
<style>body{background:#111;margin:0;display:flex;justify-content:center;
align-items:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/stream">"""
                + btn
                + b"""</body></html>"""
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
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
# Tracking de cobertura
# ---------------------------------------------------------------------------


GRID_RECTANGULAR = np.ones((4, 5), dtype=np.int32)

GRID_CIRCULAR = np.array(
    [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    dtype=np.int32,
)


def _compute_coverage_center(
    corners: np.ndarray,
    w: int,
    h: int,
    grid_mask: np.ndarray,
) -> tuple[int, int]:
    """Obtiene la celda del grid (row, col) para el centro de los corners detectados."""
    cx = np.mean(corners[:, 0, 0])
    cy = np.mean(corners[:, 0, 1])
    grid_rows, grid_cols = grid_mask.shape
    col = int(cx / w * grid_cols)
    row = int(cy / h * grid_rows)
    col = max(0, min(col, grid_cols - 1))
    row = max(0, min(row, grid_rows - 1))
    return row, col


def _draw_coverage(
    frame: np.ndarray,
    coverage: np.ndarray,
    grid_mask: np.ndarray,
    cell_target: np.ndarray | None = None,
) -> None:
    """Dibuja el overlay del grid de cobertura sobre el frame."""
    h, w = frame.shape[:2]
    grid_rows, grid_cols = coverage.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            if grid_mask[r, c] == 0:
                # Celda inactiva — overlay tenue
                overlay = frame[y1:y2, x1:x2].copy()
                dark = np.full_like(overlay, (0, 0, 0))
                cv2.addWeighted(dark, 0.5, overlay, 0.5, 0, frame[y1:y2, x1:x2])
                continue
            target = cell_target[r, c] if cell_target is not None else 0
            is_full = target > 0 and coverage[r, c] >= target
            if coverage[r, c] > 0:
                overlay = frame[y1:y2, x1:x2].copy()
                tint = (0, 80, 0) if is_full else (0, 60, 60)
                cv2.addWeighted(
                    np.full_like(overlay, tint),
                    0.3,
                    overlay,
                    0.7,
                    0,
                    frame[y1:y2, x1:x2],
                )
            label = (
                f"{int(coverage[r, c])}/{target}"
                if target > 0
                else str(int(coverage[r, c]))
            )
            color = (0, 255, 0) if is_full else (0, 200, 255)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Captura guiada: aiming basado en ghost con auto-captura
# ---------------------------------------------------------------------------

GUIDED_PREVIEW = (1296, 486)  # preview combinado: 2x (648, 486)
GUIDED_HALF = (648, 486)  # cada mitad del preview de una cámara
STABILITY_HOLD_SEC = 1.5
SKIP_POSE_TIMEOUT_SEC = 180.0

# Bootstrap: las primeras N capturas usan tolerance loose + intrínsecos
# nominales; después de eso fitteamos intrínsecos per-sensor a partir de
# los left frames capturados y los usamos para proyectar el ghost con
# tolerance tight.
BOOTSTRAP_COUNT = 6
LR_SYNC_MAX_DELTA_NS = 250_000_000  # 250 ms. El sync por software vía dos
# instancias de picamera2 driftea más allá del target original de 80ms en
# hardware real (60-120 ms típico, picos ocasionales de 200ms cuando la
# CPU está ocupada). Habría que tener sync por hardware para bounds más
# tight. El board está estático durante la captura de calibración así que
# el delta temporal no afecta el resultado — un techo generoso acá solo
# evita que las capturas sean rechazadas por error.
LR_MIN_COMMON_CORNERS = 15

# Drift de ambiente: compara la median brightness cada ~30 frames contra el baseline.
DRIFT_BASELINE_FRAMES = 10
DRIFT_CHECK_EVERY_FRAMES = 30
DRIFT_WARN_PCT = 25.0


class _GuidedState:
    """Estado mutable compartido entre el loop de captura y el HTTP handler."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_pose_idx = 0
        self.pose_status: list[str] = []  # "pending" / "captured" / "skipped"
        self.captured_pairs: list[tuple[Path, Path, str]] = []  # (l, r, pose_id)
        self.rms_text: str = "—"
        self.rms_color: str = "#888"
        # Lo settea el worker de RMS en background cuando full coverage
        # + el gate de RMS pasan los dos — se surface como un hint en el
        # panel de status diciéndole al operador que puede finalizar
        # temprano. No bloquea; el botón existente Finalizar es lo que
        # apreta.
        self.early_stop_ready: bool = False
        self.early_stop_msg: str = ""
        self.undo_requested = False
        self.skip_requested = False
        self.finish_requested = False
        self.capture_requested = False  # manual: set por POST /capture
        self.status_html: str = ""
        self.banner_text: str = ""
        self.banner_color: str = "#444"
        self.hold_progress: float = 0.0  # 0..1 durante el countdown de estabilidad
        self.bootstrap_done = False
        self.fitted_K: Optional[np.ndarray] = None
        self.drift_warning: Optional[str] = None
        # Texto del evento de audio que el browser usa para detectar
        # pose-announcements (los que arrancan con "Pose "). Los demás
        # hints quedan solo en el banner visual.
        self.audio_event: str = ""
        self.audio_event_seq: int = 0
        # Último hint que efectivamente promovimos al banner (key sin
        # dígitos). Separada de banner_text así no "recordamos" haber
        # mostrado hints que quedaron suprimidos durante el lockout de
        # la pose-announcement.
        self.last_spoken_key: str = ""
        # Texto completo del último hint promovido al banner. Se usa
        # para mantener estable lo que lee el operador frente a
        # fluctuaciones numéricas (ej. "movelo 4cm" vs "movelo 3cm").
        self.locked_alignment_text: str = ""


_guided_state: Optional[_GuidedState] = None

# Se setea True cuando el operador apreta "Comenzar" en el browser.
# Bloquea el loop principal de captura hasta que eso pase para que (a)
# el operador tenga oportunidad de posicionarse y (b) el click unlockee
# el AudioContext del browser para los beeps de notificación.
_capture_started = False
_capture_started_lock = threading.Lock()

# Después de que termina la captura guiada, el mismo HTTP server queda
# vivo y muestra progreso de las fases del wizard / pantalla finalizada.
# Lo actualiza el thread principal del wizard a medida que las fases
# completan.
_post_capture_html: str = ""
_post_capture_lock = threading.Lock()
_post_capture_active = False
_report_path_for_http: Optional[Path] = None
_guided_server: Optional[ThreadingHTTPServer] = None

# Shuttle de input del operador: el wizard puede bloquear en
# _ask_operator_ui() que renderiza un prompt en el browser y espera el
# POST a /wizard-input.
_wizard_input_event = threading.Event()
_wizard_input_value: str = ""

# True mientras el browser está reproduciendo el beep de pose-announce.
# Lo setea el path de emisión _pose_announce; lo limpia /announce-done
# que pega el JS al terminar el patrón de beep (o inmediatamente si el
# audio está apagado). Las capturas y otras emisiones gatean en esto
# así el operador siempre tiene oportunidad de leer el banner del
# bloque "número / label / distancia" antes de que cambie.
_announce_pending = False
# Modo manual del wizard: cuando True, la captura NO es automática por
# estabilidad — el operador apreta "Capturar" y no hay auto-skip por timeout.
_guided_manual_enabled = False
# Sweep (barrido libre): cuando True, do_GET sirve la página de barrido en vez
# de la guiada por-pose.
_sweep_mode = False


class _GuidedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if _guided_state is None:
            self.send_response(503)
            self.end_headers()
            return
        if self.path == "/undo":
            with _guided_state.lock:
                _guided_state.undo_requested = True
            self.send_response(204)
            self.end_headers()
        elif self.path == "/skip":
            with _guided_state.lock:
                _guided_state.skip_requested = True
            self.send_response(204)
            self.end_headers()
        elif self.path == "/capture":
            with _guided_state.lock:
                _guided_state.capture_requested = True
            self.send_response(204)
            self.end_headers()
        elif self.path == "/finish":
            with _guided_state.lock:
                _guided_state.finish_requested = True
            self.send_response(204)
            self.end_headers()
        elif self.path == "/start":
            global _capture_started
            with _capture_started_lock:
                _capture_started = True
            self.send_response(204)
            self.end_headers()
        elif self.path == "/wizard-input":
            global _wizard_input_value
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = (
                self.rfile.read(length).decode("utf-8", errors="replace")
                if length
                else ""
            )
            _wizard_input_value = raw
            _wizard_input_event.set()
            self.send_response(204)
            self.end_headers()
        elif self.path == "/announce-done":
            global _announce_pending
            _announce_pending = False
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            page = _sweep_html() if _sweep_mode else _guided_html()
            self.wfile.write(page.encode("utf-8"))
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
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
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def _guided_html() -> str:
    _capture_btn = (
        '<button class="btn btn-capture" '
        "onclick=\"flushAnnounce();post('/capture')\">Capturar</button>"
        if _guided_manual_enabled
        else ""
    )
    return (
        """<!DOCTYPE html>
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
  .btn-capture{background:#27ae60}
  .btn-capture:hover{background:#1f9952}
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
      para las notificaciones sonoras.
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
      __CAPTURE_BTN__
      <button id="btn-audio" class="btn btn-audio" onclick="toggleAudio()">Audio OFF</button>
      <button class="btn btn-undo" onclick="flushAnnounce();post('/undo')">Deshacer última</button>
      <button class="btn btn-skip" onclick="flushAnnounce();post('/skip')">Saltear pose</button>
      <button class="btn btn-finish" onclick="if(confirm('Finalizar captura?')){flushAnnounce();post('/finish')}">Finalizar</button>
    </div>
  </div>
<script>
function post(p){fetch(p,{method:'POST'})}
// Se setea después de que el operador envía la respuesta al prompt.
// Mientras esté true, el poll de /status NO puede re-inyectar los
// controles del prompt — ya los reemplazamos con un spinner y
// cualquier cambio de fase va a limpiar el flag.
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
function flushAnnounce(){
  // Si hay un pose-announce pending, desbloqueamos el backend ya
  // mismo. Sin esto el server queda lockeado esperando el
  // /announce-done que el setTimeout de beepPose tenía pendiente.
  try { fetch('/announce-done', {method:'POST'}); } catch(e){}
}
function startCapture(){
  // Gesto del usuario — unlockea el AudioContext para los beeps.
  ensureAudioCtx();
  if (audioOn) beepStart();
  const overlay = document.getElementById('start-overlay');
  if (overlay) overlay.style.display = 'none';
  fetch('/start',{method:'POST'});
}
// Default ON; el operador lo puede apagar y recordamos esa preferencia.
let audioOn = localStorage.getItem('guided.audio') !== '0';
let lastBeepedSeq = -1;
// State machine de beep: trackea el bucket previo de hold-progress + el last capture count
let lastBeepBucket = -1;  // 0..3 = buckets de tick en 0/33/66% del hold
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
// Patrones diferenciados para los eventos clave del wizard.
function beepStart(){       // par ascendente — arranque de sesión
  beep(600, 80); setTimeout(() => beep(900, 80), 100);
}
function beepPose(){        // doble tap agudo — anuncio de pose nueva
  beep(1100, 80, 0.18); setTimeout(() => beep(1100, 80, 0.18), 130);
}
function beepFinish(){      // triple descendente — fin de sesión
  beep(800, 100);
  setTimeout(() => beep(600, 100), 130);
  setTimeout(() => beep(400, 150), 260);
}
function beepActivated(){   // tap simple — toggle de audio ON
  beep(900, 60);
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
  if (audioOn) { ensureAudioCtx(); beepActivated(); }
  else {
    // Desbloquea el backend si hay un pose announce en curso — el
    // setTimeout del beep que iba a postear /announce-done puede
    // quedar pending cuando se apaga el audio.
    flushAnnounce();
  }
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
    // Modos post-captura (processing / finalised) identificados vía data-phase.
    const phaseMatch = html.match(/data-phase="([^"]+)"/);
    if (phaseMatch) {
      postCaptureMode = true;
      const thisPhase = phaseMatch[1];
      const thisPromptType = (html.match(/data-prompt-type="([^"]+)"/) || [,''])[1];
      // Re-renderizar solo cuando la fase O el tipo de prompt cambian.
      // Si no, borraríamos el <input> ground-truth en cada poll y le
      // robaríamos el focus.
      const shouldRender = (
        thisPhase !== lastRenderedPhase ||
        thisPromptType !== lastRenderedPromptType
      );
      if (shouldRender) {
        document.getElementById('status').innerHTML = html;
        lastRenderedPhase = thisPhase;
        lastRenderedPromptType = thisPromptType;
        // La fase realmente cambió — limpiar cualquier estado de prompt
        // pending así los prompts siguientes pueden renderizarse limpios.
        promptSubmitted = false;
        // Entrar a cualquier fase post-captura desbloquea el backend si
        // había un pose announce pending.
        flushAnnounce();
      }
      // Ocultar el stream en vivo una vez que la captura terminó.
      const stage = document.getElementById('stage');
      if (stage) stage.style.display = 'none';
      // Ocultar la fila de acciones (undo/skip/finish), reemplazar banner con dinámico.
      const btns = document.querySelectorAll('#panel .row button');
      btns.forEach(b => b.style.display = 'none');
      const banner = document.getElementById('banner');
      if (banner) banner.style.display = 'none';
      const bar = document.getElementById('progress-bar');
      if (bar) bar.style.display = 'none';
      // Prompt interactivo del wizard: muestra los controles que postean
      // de vuelta a /wizard-input. Solo inyectar al primer sight Y
      // mientras el operador no haya enviado todavía — después del
      // submit dejamos el spinner.
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
          // type=text + inputmode=numeric evita la UI de spinner nativa
          // mientras igual levanta el teclado numérico en celulares.
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
        if (audioOn) beepFinish();
        // Solo auto-abrir el reporte cuando realmente existe (path de éxito).
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
    // Detectar cambio de pose y desbloquear el backend si quedaba un
    // pose announce pending de la pose anterior.
    const poseIdxMatch = html.match(/data-pose-idx="([^"]+)"/);
    if (poseIdxMatch) {
      const idx = poseIdxMatch[1];
      if (window._lastPoseIdx !== undefined && window._lastPoseIdx !== idx) {
        flushAnnounce();
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
      // Reaccionamos a cada nuevo seq solo para detectar pose-announce
      // (el bloque atómico "número/label/distancia"); los movement
      // hints van solo al banner visual. Si es pose-announce, el beep
      // dispara el unlock del backend cuando termina su patrón.
      if (seq !== lastBeepedSeq) {
        lastBeepedSeq = seq;
        const isPoseAnnounce = audioText && audioText.startsWith('Pose ');
        if (isPoseAnnounce) {
          if (audioOn) {
            beepPose();
            // beepPose dura ~210ms (80 + 50 gap + 80). 250ms le da margen.
            setTimeout(() => fetch('/announce-done', {method:'POST'}), 250);
          } else {
            // Audio off — desbloqueamos el backend ya mismo.
            fetch('/announce-done', {method:'POST'});
          }
        }
      }
      // Beep: tick en 0/33/66% del hold-progress (buckets 0/1/2). Reset en progress=0.
      if (audioOn) {
        if (progress <= 0.01) {
          lastBeepBucket = -1;
        } else {
          const bucket = Math.floor(progress * 3);  // 0,1,2
          if (bucket !== lastBeepBucket && bucket < 3) {
            lastBeepBucket = bucket;
            beep(1000, 70, 0.12);  // tick alto y corto
          }
        }
        // Confirmación de captura: capturedN saltó
        if (capturedN > lastCapturedN) {
          lastCapturedN = capturedN;
          beep(660, 220, 0.18);  // confirmación cálida y larga
        } else if (capturedN < lastCapturedN) {
          // Pasó un UNDO
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
    ).replace("__CAPTURE_BTN__", _capture_btn)


def _draw_ghost(
    vis: np.ndarray,
    outer_corners: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Dibuja un cuadrilátero ghost sobre un frame de preview."""
    pts = outer_corners.reshape(-1, 1, 2).astype(np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.polylines(
        vis,
        [pts],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    # Markers de esquina
    for p in pts.reshape(-1, 2):
        cv2.circle(vis, tuple(p), 6, color, -1, lineType=cv2.LINE_AA)


def _draw_direction_arrow(
    vis: np.ndarray, err: dict[str, float], ghost_center: np.ndarray
) -> None:
    """Dibuja una flecha desde el centro del ghost indicando hacia
    dónde mover el board."""
    # compute_alignment_by_corners devuelve centroid_offset_px; el
    # legacy compute_alignment_error devuelve center_px. Aceptar
    # cualquiera.
    offset = err.get("centroid_offset_px", err.get("center_px", 0.0))
    if offset <= ALIGN_CENTER_TOL_PX:
        return
    # La flecha apunta DESDE donde ESTÁ el board hacia donde DEBERÍA
    # estar (centro del ghost). offset_x es (det - ghost), así que
    # flipeamos el signo para el hint de dirección
    dx, dy = -err["offset_x"], -err["offset_y"]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    # Clampear magnitud así la flecha queda dentro de la pantalla
    max_len = 80
    scale = min(1.0, max_len / length)
    dx *= scale
    dy *= scale
    gx, gy = int(ghost_center[0]), int(ghost_center[1])
    # Dibujar desde el centro del ghost HACIA AFUERA en la dirección
    # en la que el board debería moverse (pero el board está
    # actualmente OFFSET hacia el lado opuesto, así que la flecha
    # muestra la move-direction)
    tip_x = int(gx - dx)
    tip_y = int(gy - dy)
    cv2.arrowedLine(
        vis,
        (gx, gy),
        (tip_x, tip_y),
        (50, 220, 255),
        4,
        tipLength=0.35,
        line_type=cv2.LINE_AA,
    )


def _rms_color_for(rms: float) -> str:
    if rms < 1.5:
        return "#2ecc71"
    if rms < 3.0:
        return "#f1c40f"
    return "#e74c3c"


def _background_rms_worker(
    state: _GuidedState,
    board,
    board_size,
    sq_len,
    mk_len,
    captures_dir: Path,
    stop_evt: threading.Event,
    legacy_pattern: bool = True,
    all_poses: Optional[list] = None,
    early_stop_enabled: bool = True,
    near_mm: float = DEFAULT_DIST_NEAR_MM,
    mid_mm: float = DEFAULT_DIST_MID_MM,
    far_mm: float = DEFAULT_DIST_FAR_MM,
) -> None:
    """Cada 3 capturas nuevas pasada la 8va, intenta una calibración incremental.

    Más allá del hint de RMS en vivo, también evalúa
    ``is_calibration_ready_for_early_stop`` y flipea
    ``state.early_stop_ready`` cuando la calibración ya pasaría los
    gates lab-grade — permite al operador finalizar sin agotar las
    20 poses canónicas en una sesión smooth. La decisión es
    informativa; el botón Finalizar existente es lo que el operador
    clickea.
    """
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
                pairs,
                board_size=board_size,
                square_length=sq_len,
                marker_length=mk_len,
                legacy_pattern=legacy_pattern,
            )
            rms_l = _residual_estimate(pairs, board, result)
            with state.lock:
                state.rms_text = f"RMS≈{rms_l:.2f}px ({n} pares)"
                state.rms_color = _rms_color_for(rms_l)

            # Readiness de early-stop — solo si está habilitado y
            # tenemos la lista de poses contra la cual comparar (el
            # worker se crea con all_poses pasadas por el wizard).
            if early_stop_enabled and all_poses is not None:
                from src.vision.calibration import (
                    analyze_pose_coverage,
                    is_calibration_ready_for_early_stop,
                )

                pose_ids = [pid for _, _, pid in snapshot]
                coverage = analyze_pose_coverage(
                    pose_ids,
                    all_poses=all_poses,
                    near_mm=near_mm,
                    mid_mm=mid_mm,
                    far_mm=far_mm,
                )
                ready, reason = is_calibration_ready_for_early_stop(
                    coverage,
                    per_pair_rms_px=rms_l,
                    captured_count=n,
                )
                with state.lock:
                    state.early_stop_ready = ready
                    state.early_stop_msg = (
                        f"✓ Calibración lista — RMS={rms_l:.2f}px, "
                        f"{n} capturas con cobertura completa. "
                        f"Podés finalizar ahora si querés."
                        if ready
                        else ""
                    )
        except Exception as e:
            logger.debug("Calibración incremental salteada: %s", e)


def _residual_estimate(pairs, board, result) -> float:
    """Estimación rápida de residual para una calibración — reusa el RMS de la cámara izquierda.

    Proyecta los corners observados de cada par a través del modelo
    fisheye fitteado y devuelve el RMS promedio per-pair. Más barato
    que re-correr fisheye.calibrate y robusto a las pocas poses
    degeneradas que aparecen ocasionalmente a mitad de captura.
    """
    from src.vision.calibration import compute_per_pair_residuals

    per_pair = compute_per_pair_residuals(pairs, board, result)
    rms_vals = [
        p["rms_l"] for p in per_pair if p["rms_l"] == p["rms_l"]
    ]  # filtro de NaN
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
        # La resolución tiene que matchear cross-resume — la calibración
        # necesita todos los frames del mismo size o la matemática se
        # rompe. Mezclar resoluciones distintas corrompe los intrínsecos
        # silencioso (K distinto por pose), así que la pineamos.
        "resolution": list(getattr(args, "resolution", [2304, 1296])),
    }


def _save_session(
    output_dir: Path,
    state: "_GuidedState",
    poses: list,
    args: argparse.Namespace,
) -> None:
    """Escribe el sidecar session.json atómicamente después de cada cambio de estado."""
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
    """Carga session.json y valida que los params matcheen los args actuales.

    Devuelve el dict parseado, o None si el archivo está ausente / es
    inválido / es incompatible.
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
    mismatches = [k for k in expected if expected[k] != actual.get(k)]
    if mismatches:
        logger.error(
            "session.json incompatible — parámetros distintos: %s. "
            "Usá las mismas flags que en la sesión original o borrá session.json.",
            mismatches,
        )
        return None
    return data


def _run_guided_capture(args: argparse.Namespace) -> None:
    """Loop de captura guiada: ghost + matching por corner-ID + auto-captura
    con estabilidad, gate de L/R sync, gate de calidad, handoff de bootstrap
    intrínsecos, warnings de drift de ambiente y audio opcional vía web-UI.

    Soporta --resume: restaura captured_pairs + pose_status desde
    session.json, re-fittea el K de bootstrap a partir de las capturas
    existentes si count >= BOOTSTRAP_COUNT.
    """
    global _shutting_down, _guided_state, _latest_jpeg, _guided_manual_enabled
    _guided_manual_enabled = bool(getattr(args, "manual", False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.vision.capture import StereoCapture

    max_exp = getattr(args, "max_exposure_us", 0)
    cap = StereoCapture(
        cam_left_id=args.left,
        cam_right_id=args.right,
        resolution=tuple(args.resolution),
        fps=args.fps,
        meter_mode=getattr(args, "meter", "matrix"),
        lock_ae=getattr(args, "lock_ae", False),
        max_exposure_us=max_exp if max_exp and max_exp > 0 else None,
        sensor_raw_size=HW.default_res,
        initial_settle_seconds=HW.ae_initial_settle_seconds,
        resettle_seconds=HW.ae_resettle_seconds,
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

    # Handling de resume. Sin --resume borramos cualquier sesión previa
    # en el output dir así el operador no tiene que limpiar a mano entre
    # corridas. Con --resume validamos el sidecar y seguimos desde ahí.
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
        # Restaurar capturas
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
        # Restaurar las skipped
        for pid, status in existing_session.get("pose_status", {}).items():
            if status == "skipped" and pid in pose_idx_by_id:
                if state.pose_status[pose_idx_by_id[pid]] == "pending":
                    state.pose_status[pose_idx_by_id[pid]] = "skipped"
        count = len(state.captured_pairs)
        # Avanzar el pose pointer a la primera pending
        for i, status in enumerate(state.pose_status):
            if status == "pending":
                state.current_pose_idx = i
                break
        else:
            logger.info("Sesión ya completa (%d capturas). Nada que resumir.", count)
        # Re-fitear el K de bootstrap si hay suficientes capturas Y diversity de distancia
        if count >= BOOTSTRAP_COUNT:
            diverse = _captured_have_distance_diversity(
                state.captured_pairs,
                poses,
                getattr(args, "dist_near_mm", DEFAULT_DIST_NEAR_MM),
                getattr(args, "dist_mid_mm", DEFAULT_DIST_MID_MM),
                getattr(args, "dist_far_mm", DEFAULT_DIST_FAR_MM),
            )
            if not diverse:
                logger.info(
                    "Resume: %d capturas restauradas pero falta cobertura "
                    "de alguna banda (near/mid/far) — seguimos con K nominal "
                    "hasta completar diversity",
                    count,
                )
            else:
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
                            count,
                            fitted[0, 0],
                            fitted[1, 1],
                        )
                    else:
                        logger.warning(
                            "Resume: bootstrap fit falló, seguimos con K nominal"
                        )
        else:
            logger.info(
                "Resume: %d capturas restauradas (< bootstrap threshold %d, "
                "seguimos con K nominal)",
                count,
                BOOTSTRAP_COUNT,
            )

    # SO_REUSEADDR así una instancia previa Ctrl-C'eada no deja el
    # puerto en TIME_WAIT para la próxima corrida.
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _GuidedHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    rms_stop = threading.Event()
    rms_thread = threading.Thread(
        target=_background_rms_worker,
        args=(
            state,
            board,
            board_size,
            args.square_length,
            args.marker_length,
            output_dir,
            rms_stop,
            getattr(args, "legacy_pattern", True),
        ),
        kwargs={
            "all_poses": poses,
            "early_stop_enabled": not getattr(args, "no_early_stop", False),
            "near_mm": getattr(args, "dist_near_mm", DEFAULT_DIST_NEAR_MM),
            "mid_mm": getattr(args, "dist_mid_mm", DEFAULT_DIST_MID_MM),
            "far_mm": getattr(args, "dist_far_mm", DEFAULT_DIST_FAR_MM),
        },
        daemon=True,
    )
    rms_thread.start()

    logger.info(
        "Calibración guiada — preview: http://people-counter.local:%d", args.port
    )
    logger.info(
        "Poses objetivo: %d (bootstrap con las primeras %d)",
        len(poses),
        BOOTSTRAP_COUNT,
    )
    logger.info(
        "Seguí la silueta fantasma, mantené quieto %.1fs para capturar.",
        STABILITY_HOLD_SEC,
    )
    logger.info("Esperando que el operador haga click en Comenzar...")

    # Bloquear hasta que el operador apreta "Comenzar" en el browser.
    # Eso le da tiempo para posicionar el board + activa el audio
    # context del browser.
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

    # Re-settle AE con el board ya posicionado (no-op si lock_ae=False).
    # El lock provisional del open() ocurrió cuando el operador acababa de
    # lanzar el script; ahora con todo armado, refresca el lock para que
    # los valores reflejen la escena real de medición. Mismo patrón que
    # focus_assist y diagnose_calibration.
    try:
        cap.resettle_and_lock()
    except Exception as e:
        logger.warning("resettle_and_lock no aplicado: %s", e)

    logger.info("Ctrl+C para cancelar, o usá el botón Finalizar en la UI.\n")

    stability = StabilityTracker()
    hold_started_at: Optional[float] = None
    pose_started_at = time.time()
    count = 0

    # Baseline de drift de ambiente (a partir de los primeros N frames)
    drift_baseline: Optional[float] = None
    drift_samples: list[float] = []
    frame_counter = 0
    # Tracking de drift de temperatura del sensor
    temp_baseline: Optional[float] = None
    temp_samples: list[float] = []
    last_temp: Optional[float] = None
    # Streak de detección asimétrica (para el hint "una cámara
    # silenciosamente fallando"). Si una cámara sigue detectando el
    # board pero la otra no, queremos mostrar eso en la UI — el
    # operador recibía "captura rechazada" sin saber qué cámara
    # estaba fallando.
    only_l_detect_streak = 0
    only_r_detect_streak = 0
    ASYMMETRIC_DETECT_WARN_FRAMES = 20

    def _emit_audio(text: str) -> None:
        state.audio_event = text
        state.audio_event_seq += 1

    def _pose_announce(idx: int) -> str:
        """Texto del anuncio de pose: label + distancia target en cm.
        Se muestra en el banner; el browser dispara el beep de pose
        en paralelo cuando lo detecta (texto que arranca con "Pose ")."""
        p = poses[idx]
        z_cm = p.tvec_mm[2] / 10.0
        return f"Pose {idx + 1}. {p.label}. A {z_cm:.0f} centímetros de la cámara"

    def _emit_pose_announce(idx: int) -> None:
        """Emite el anuncio de pose y lockea todo lo demás hasta que el
        browser señalice que el beep de pose terminó (POST /announce-done)."""
        global _announce_pending
        _announce_pending = True
        _emit_audio(_pose_announce(idx))

    # Anunciar la primera pose pending (= poses[0] al arrancar fresco,
    # o la próxima pending después del restore con resume). Saltear
    # si no hay nada que capturar — pasa con --resume en una sesión
    # que ya está completa: el wizard va directo a processing, y un
    # beep de pose suelto en esa transición es ruido sin contexto.
    has_pending = any(s == "pending" for s in state.pose_status)
    if has_pending:
        _emit_pose_announce(state.current_pose_idx)

    # Persistir el estado initial/resumed así un crash antes de
    # cualquier captura nueva igual deja un session.json válido en
    # disco.
    _save_session(output_dir, state, poses, args)

    # Debounce del warning de L/R desync: el sync delta varía
    # frame-a-frame y spikea brevemente después de eventos como
    # skip/undo (main thread ocupado => los buffers driftean). Solo
    # mostramos el warning cuando vemos >= esta cantidad de frames
    # malos consecutivos, escondiendo spikes de ruido single-frame.
    SYNC_WARN_CONSECUTIVE_FRAMES = 4
    sync_bad_streak = 0

    prev_pose_idx = state.current_pose_idx

    try:
        while count < len(poses):
            # Detectar cualquier cambio de pose y resetear "qué dijimos
            # por última vez" así el primer hint de movimiento de la
            # nueva pose siempre suena, aunque el último hint de la
            # pose anterior fuera la misma frase.
            if state.current_pose_idx != prev_pose_idx:
                state.last_spoken_key = ""
                state.locked_alignment_text = ""
                sync_bad_streak = 0  # slate limpia así skip no replayea el warning
                prev_pose_idx = state.current_pose_idx

            pose = poses[state.current_pose_idx]
            if state.pose_status[state.current_pose_idx] != "pending":
                state.current_pose_idx = (state.current_pose_idx + 1) % len(poses)
                if all(s != "pending" for s in state.pose_status):
                    break
                continue

            # Capturar con timestamps + temperatura del sensor para
            # tracking de drift
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
            # Suprimir el warning por 2s después de cada transición
            # de pose — el pipeline de picamera2 rutinariamente
            # spikea justo después de skip/capture mientras el main
            # thread estaba ocupado, y ese drift se asienta en un
            # segundo o dos sin acción del operador.
            pose_settling = (time.time() - pose_started_at) < 2.0
            sync_warn_active = (
                sync_bad_streak >= SYNC_WARN_CONSECUTIVE_FRAMES and not pose_settling
            )

            frame_counter += 1

            # Lock del baseline de temperatura (primeros N samples válidos)
            if last_temp is not None:
                if temp_baseline is None:
                    temp_samples.append(last_temp)
                    if len(temp_samples) >= DRIFT_BASELINE_FRAMES:
                        temp_baseline = float(np.median(temp_samples))
                        logger.info(
                            "Baseline de temperatura del sensor: %.1f°C", temp_baseline
                        )
                elif frame_counter % DRIFT_CHECK_EVERY_FRAMES == 0:
                    delta_c = last_temp - temp_baseline
                    if abs(delta_c) > 10.0:
                        state.drift_warning = (
                            f"Sensor +{delta_c:.1f}°C desde el inicio "
                            f"({temp_baseline:.1f}→{last_temp:.1f}°C) "
                            f"— los intrínsecos pueden derivar"
                        )

            # Tracking de drift de ambiente
            gray_small = cv2.cvtColor(
                cv2.resize(frame_l, (320, 240)),
                cv2.COLOR_BGR2GRAY,
            )
            brightness = float(np.median(gray_small))
            if drift_baseline is None:
                drift_samples.append(brightness)
                if len(drift_samples) >= DRIFT_BASELINE_FRAMES:
                    drift_baseline = float(np.median(drift_samples))
                    logger.info(
                        "Baseline de drift lockeado en median brightness %.1f",
                        drift_baseline,
                    )
            elif frame_counter % DRIFT_CHECK_EVERY_FRAMES == 0:
                drift_pct = (
                    abs(brightness - drift_baseline) / max(drift_baseline, 1) * 100
                )
                if drift_pct > DRIFT_WARN_PCT:
                    direction = "aumentó" if brightness > drift_baseline else "bajó"
                    state.drift_warning = (
                        f"Luz ambiente {direction} {drift_pct:.0f}% desde el inicio "
                        f"— considerá reiniciar para relockear exposición"
                    )
                else:
                    state.drift_warning = None

            # Detección de ChArUco en full-res. lenient=True para que los
            # markers de 33mm en las poses far de 3m (~22 px de ancho →
            # 0.5% de 4608) igual sean detectados — el default strict
            # rechaza markers bajo el 3% de la dimensión de la imagen.
            # min_corners=4 matchea el threshold del alignment-gate de
            # abajo; con el default de 8, las detecciones parciales en
            # los bordes fisheye (poses de esquina) rebotan entre "8
            # esquinas" y "7 esquinas" frame a frame y el wizard flipea
            # entre Alineado y "board no visible", sin holdear estable
            # el tiempo suficiente para capturar.
            # Dual-pass: si el frame original detecta <8 corners,
            # reintenta con sharpen. Recupera markers cuando un copy
            # tiene foco marginal a esa distancia, sin cambiar el
            # comportamiento del calibrate_stereo downstream (que sigue
            # leyendo single-pass via _detect_all_pairs).
            corners_l, ids_l = detect_charuco_dual_pass(
                frame_l,
                board,
                min_corners=4,
            )
            corners_r, ids_r = detect_charuco_dual_pass(
                frame_r,
                board,
                min_corners=4,
            )

            # Trackear detección asimétrica — una cámara fallando
            # consistentemente mientras la otra detecta bien es un
            # silent killer (el operador ve "captura rechazada" pero
            # no sabe qué cámara es la causa). Lo exponemos
            # explícitamente vía una alerta en el panel.
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

            # Proyectar ghost — usa fitted_K después del bootstrap.
            # Pasamos la resolución de captura REAL así project_pose escala
            # fitted_K (que vive a esa res, no a HW.full_res 4608x2592)
            # correctamente al preview. focal_full_px también escalado a la
            # res de captura para el path de bootstrap (fitted_K=None) — los
            # nominales escalan linealmente con la resolución de captura
            # binned (Mode 1 IMX708 = 2304×1296 = full_res / 2).
            capture_res = (frame_l.shape[1], frame_l.shape[0])
            focal_capture = HW.nominal_focal_full_px * capture_res[0] / HW.full_res[0]
            ghost = project_pose(
                pose,
                board_size,
                args.square_length,
                GUIDED_HALF,
                focal_full_px=focal_capture,
                full_res=capture_res,
                fitted_K=state.fitted_K,
            )

            # Matching por corner-ID contra la detección de la cámara izquierda
            scaled_corners_l: Optional[np.ndarray] = None
            err: Optional[dict] = None
            aligned = False
            if corners_l is not None and ids_l is not None and len(ids_l) >= 4:
                scaled = corners_l.reshape(-1, 2).copy()
                scaled[:, 0] *= scale_x
                scaled[:, 1] *= scale_y
                scaled_corners_l = scaled
                err = compute_alignment_by_corners(
                    scaled,
                    ids_l,
                    ghost["inner_corners"],
                )
                tol_loose, tol_tight = _resolve_tolerance(args)
                # Permitir override del corner-gate via --align-min-corners.
                # Default None mantiene el comportamiento canónico (tight=15,
                # loose=12). Valores menores son útiles SOLO para setups con
                # constraint físico que no permite cubrir bien el FOV (mount
                # vertical en cuarto chico, FOV >120° con rectificación
                # agresiva, etc) — el resultado es matemáticamente más
                # subdeterminado y vale solo cuando la alternativa es no
                # poder calibrar nada.
                cli_min = getattr(args, "align_min_corners", None)
                tight_min = (
                    int(cli_min) if cli_min is not None else ALIGN_MATCHED_MIN_TIGHT
                )
                loose_min = (
                    max(1, int(cli_min) - 3)
                    if cli_min is not None
                    else ALIGN_MATCHED_MIN_LOOSE
                )
                # --align-loose-px override de la tolerancia del centroid
                # offset contra el ghost target. Default None usa los
                # thresholds canónicos (tight=12px, loose=25px). Valores más
                # altos relajan el requisito de "matchear la posición del
                # ghost" — útil cuando los ghosts del wizard fueron generados
                # para una geometría distinta a la del operativo (setup
                # vertical en cuarto chico) y matchear-al-pixel es imposible.
                cli_mean_err = getattr(args, "align_loose_px", None)
                eff_tol_tight = (
                    float(cli_mean_err) if cli_mean_err is not None else tol_tight
                )
                eff_tol_loose = (
                    float(cli_mean_err) if cli_mean_err is not None else tol_loose
                )
                if state.bootstrap_done:
                    aligned = is_aligned_by_corners(
                        err,
                        mean_err_tol_px=eff_tol_tight,
                        min_matched=tight_min,
                    )
                else:
                    aligned = is_aligned_by_corners(
                        err,
                        mean_err_tol_px=eff_tol_loose,
                        min_matched=loose_min,
                    )

            # Estabilidad — pasamos los IDs así el tracker tolera
            # fluctuaciones del count de detección (23↔35 esquinas con
            # iluminación marginal) sin resetear el buffer.
            stability.push(
                scaled_corners_l if aligned else None,
                ids=ids_l if aligned else None,
            )
            stable = stability.is_stable() if aligned else False

            now = time.time()
            # Gatear todo contra la señal "announcement finished" del
            # browser (POST /announce-done que dispara el setTimeout
            # del beep de pose-announce). De esta forma el bloque
            # "número / label / distancia" del banner tiene garantizado
            # estar visible un mínimo de tiempo antes de que cambie —
            # las capturas y otros hints quedan bloqueados hasta que el
            # operador haya tenido oportunidad de leerlo. Solo el Skip
            # manual puede interrumpir.
            announce_settling = _announce_pending
            announce_audio_lockout = _announce_pending
            if args.manual:
                # Captura manual: el operador apreta "Capturar". Sin
                # countdown de estabilidad ni gate de alineación al ghost
                # (el gate de calidad de abajo igual rechaza un board no
                # detectado / L-R desincronizado). El ghost sigue de guía.
                hold_started_at = None
                hold_progress = 0.0
                with state.lock:
                    should_capture = state.capture_requested
                    state.capture_requested = False
            else:
                if aligned and stable and not announce_settling:
                    if hold_started_at is None:
                        hold_started_at = now
                        # Solo emitir el hint "Mantené quieto" una vez que
                        # el beep de pose-announce terminó. La captura
                        # igual puede correr durante esa ventana — solo no
                        # se promueve al banner.
                        if not announce_audio_lockout:
                            _emit_audio("Mantené quieto")
                    hold_progress = min(
                        1.0, (now - hold_started_at) / STABILITY_HOLD_SEC
                    )
                else:
                    hold_started_at = None
                    hold_progress = 0.0

                should_capture = (
                    aligned
                    and stable
                    and not announce_settling
                    and hold_progress >= 1.0
                )

            warnings = live_lighting_warnings(frame_l, frame_r)

            # Manejo de acciones del usuario
            with state.lock:
                if state.finish_requested:
                    state.finish_requested = False
                    break
                if state.undo_requested:
                    state.undo_requested = False
                    if state.captured_pairs:
                        lp, rp, pid = state.captured_pairs.pop()
                        try:
                            lp.unlink()
                        except Exception:
                            pass
                        try:
                            rp.unlink()
                        except Exception:
                            pass
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
                        # Si quedamos abajo del bootstrap, resetear también fitted_K
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
                    # Anunciar la NUEVA pose así el operador sabe a
                    # dónde moverse después. El Skip en sí no se anuncia
                    # (clickearon el botón, ya saben).
                    if any(s == "pending" for s in state.pose_status):
                        _emit_pose_announce(state.current_pose_idx)
                    _save_session(output_dir, state, poses, args)
                    continue

            # Timeout de auto-skip (configurable vía --pose-timeout-sec).
            # En modo manual NO hay auto-skip — el operador skipea con el botón.
            pose_timeout = getattr(args, "pose_timeout_sec", SKIP_POSE_TIMEOUT_SEC)
            if not args.manual and now - pose_started_at > pose_timeout:
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

            # Preview L: SIN overlay charuco — los puntos per-corner
            # + IDs llenaban el área del ghost y hacían difícil ver los
            # bordes reales del board contra el outline del ghost.
            # Mostramos solo un badge con el count de esquinas así el
            # operador igual sabe que la detección funciona. R mantiene
            # el overlay completo como diagnóstico de la cámara derecha
            # (no hay ghost compitiendo ahí).
            n_l_detected = len(corners_l) if corners_l is not None else 0
            if corners_r is not None and ids_r is not None:
                sc = corners_r.copy()
                sc[:, 0, 0] *= scale_x
                sc[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_r, sc, ids_r, (0, 255, 0))

            ghost_color = (80, 220, 80) if aligned else (80, 180, 255)
            _draw_ghost(vis_l, ghost["outer_corners"], ghost_color)

            badge_color = (80, 220, 80) if n_l_detected >= 8 else (80, 180, 255)
            cv2.putText(
                vis_l,
                f"{n_l_detected} esquinas",
                (8, GUIDED_HALF[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                badge_color,
                2,
            )

            # Flecha de dirección sobre LEFT
            if err is not None and not aligned and err["matched"] >= 4:
                _draw_direction_arrow(vis_l, err, ghost["center"])

            # Barra de progreso de hold arriba del preview LEFT
            if hold_progress > 0:
                bar_w = int(GUIDED_HALF[0] * hold_progress)
                cv2.rectangle(vis_l, (0, 0), (GUIDED_HALF[0], 8), (30, 30, 30), -1)
                cv2.rectangle(vis_l, (0, 0), (bar_w, 8), (0, 255, 0), -1)

            cv2.putText(
                vis_l, "L", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                vis_r, "R", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            combined = np.hstack([vis_l, vis_r])

            # Gate de captura: calidad + L/R sync + esquinas comunes
            if should_capture:
                n_corners_l = len(corners_l) if corners_l is not None else 0
                common_n = count_common_corners(ids_l, ids_r)
                quality = assess_frame_quality(
                    frame_l,
                    frame_r,
                    n_corners_l,
                    corners_l=corners_l,
                    corners_r=corners_r,
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
                    # Filename estable keyeado por pose_id — hace que
                    # --resume sea idempotente y mantiene el pattern
                    # existente left_NNN.png así las tools downstream
                    # (el glob del subcomando calibrate) siguen
                    # funcionando.
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
                    cv2.rectangle(
                        combined,
                        (0, 0),
                        (combined.shape[1] - 1, combined.shape[0] - 1),
                        (0, 255, 0),
                        8,
                    )
                    logger.info(
                        "[%d/%d] Pose %s capturada — %s (common=%d, sync=%.1fms)",
                        count,
                        len(poses),
                        pose.id,
                        pose.label,
                        common_n,
                        lr_delta_ms,
                    )
                    # El beep de captura cálido (660 Hz × 220ms) ya
                    # confirma el evento — no emitimos ningún audio
                    # adicional para no demorar el beep de la próxima
                    # pose announce.

                    # Handoff de bootstrap: después de N capturas Y
                    # diversity de distancia entre las 3 bandas,
                    # fittear intrínsecos. Sin diversity, el focal
                    # fitteado puede quedar muy mal y corromper el
                    # render del ghost para poses fuera de la banda
                    # capturada.
                    if not state.bootstrap_done and count >= BOOTSTRAP_COUNT:
                        diverse = _captured_have_distance_diversity(
                            state.captured_pairs,
                            poses,
                            getattr(args, "dist_near_mm", DEFAULT_DIST_NEAR_MM),
                            getattr(args, "dist_mid_mm", DEFAULT_DIST_MID_MM),
                            getattr(args, "dist_far_mm", DEFAULT_DIST_FAR_MM),
                        )
                        if not diverse:
                            logger.info(
                                "Bootstrap deferido: %d capturas pero falta cobertura "
                                "de alguna banda (near/mid/far). Seguimos con K nominal "
                                "hasta tener al menos una captura por banda.",
                                count,
                            )
                        else:
                            logger.info(
                                "Fitting bootstrap intrínsecos desde %d capturas...",
                                count,
                            )
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
                                    "Bootstrap completo: f_x=%.0f, f_y=%.0f, cx=%.0f, cy=%.0f — "
                                    "cambiando a tolerancia estricta",
                                    fitted[0, 0],
                                    fitted[1, 1],
                                    fitted[0, 2],
                                    fitted[1, 2],
                                )
                                _emit_audio(
                                    "Intrínsecos bootstrap ajustados, tolerancia ahora estricta"
                                )
                            else:
                                logger.warning(
                                    "Bootstrap fit falló; seguimos con K nominal"
                                )

                    # Avanzar a la próxima pose pending
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

            # Encodear preview JPEG
            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with _jpeg_lock:
                _latest_jpeg = jpeg.tobytes()

            # Composición del status
            captured_n = sum(1 for s in state.pose_status if s == "captured")
            skipped_n = sum(1 for s in state.pose_status if s == "skipped")

            if aligned and stable:
                banner = f"Capturando... {int(hold_progress * 100)}%"
                banner_color = "#f1c40f"
                audio_text = ""  # no interrumpir el hold
            elif aligned:
                banner = "Alineado — mantené quieto"
                banner_color = "#3498db"
                audio_text = ""
            elif err is not None and err["matched"] >= 4:
                # Convertir offsets de píxeles a cm para el hint del banner —
                # mucho más intuitivo para el operador que "movelo 47 píxeles".
                if state.fitted_K is not None:
                    f_px_for_hint = float(state.fitted_K[0, 0])
                else:
                    f_px_for_hint = HW.nominal_focal_full_px
                # Los offsets en err están en el píxel space del PREVIEW
                # (GUIDED_HALF, ~648 de ancho después de la
                # multiplicación scale_x de antes). El fitted_K / f_px
                # nominal están a full-res 4608. Bajar f de escala así
                # el ratio mm_per_px es consistente con las unidades
                # del offset.
                f_px_preview = f_px_for_hint * scale_x
                # PoseTarget guarda la position como tvec_mm = (x, y, z); z es depth.
                mm_per_px_here = pose.tvec_mm[2] / f_px_preview
                banner = alignment_hint_by_corners(err, mm_per_px=mm_per_px_here)
                banner_color = "#e67e22"
                audio_text = banner
            else:
                banner = "Board no visible — movelo al centro"
                banner_color = "#e74c3c"
                audio_text = ""

            # Refrescar el audio event solo cuando el banner cambia
            # SEMÁNTICAMENTE — strippear los dígitos así "movelo
            # izquierda 4cm" vs "movelo izquierda 3cm" cuentan como el
            # mismo hint y no flaggean otra emisión. También saltearlo
            # durante la grace window post-pose-announcement así el
            # texto de la pose en el banner no se sobrescribe inmediatamente
            # por un movement hint (state.audio_event es un slot único
            # — gana el último write).
            import re as _re

            def _key(s: str) -> str:
                return _re.sub(r"[\d.,]+", "", s or "").strip()

            # `last_spoken_key` trackea LO QUE FUE HABLADO, separado del
            # banner visible. Si suprimimos un emit durante el lockout
            # no podemos pretender que ya lo dijimos — si no, cuando el
            # lockout termine y el hint todavía esté vigente, la
            # comparación con _key piensa que ya lo entregamos y queda
            # en silencio para siempre.
            if (
                audio_text
                and not announce_audio_lockout
                and _key(audio_text) != state.last_spoken_key
            ):
                _emit_audio(audio_text)
                state.last_spoken_key = _key(audio_text)
                state.locked_alignment_text = audio_text

            # Cuando el hint de alineación tiene la misma key semántica
            # que lo que el operador acaba de escuchar, mantener el
            # banner on-screen freezeado en el texto exacto que dijimos.
            # Evita el flicker de 1cm entre la medición en vivo y el
            # audio stale (el operador escucha "15cm", ve "14cm" y cree
            # que no matchean).
            if (
                audio_text
                and state.locked_alignment_text
                and _key(audio_text) == state.last_spoken_key
            ):
                banner = state.locked_alignment_text

            phase_label = "bootstrap" if not state.bootstrap_done else "estricto"
            warnings_html = ""
            if warnings:
                warnings_html = '<div class="warn">⚠ ' + " · ".join(warnings) + "</div>"
            if state.drift_warning:
                warnings_html += f'<div class="warn">⚠ {state.drift_warning}</div>'

            rms_html = (
                f'<span class="rms" style="background:{state.rms_color}">'
                f"{state.rms_text}</span>"
            )

            sync_note = ""
            if sync_warn_active:
                sync_note = (
                    f' · <span class="warn">L/R desync {lr_delta_ms:.1f}ms</span>'
                )

            # Escapar el audio text para el atributo HTML data-*
            audio_escaped = (state.audio_event or "").replace('"', "&quot;")
            banner_escaped = banner.replace('"', "&quot;")

            n_l = len(corners_l) if corners_l is not None else 0
            n_r = len(corners_r) if corners_r is not None else 0

            asymmetric_warn_html = ""
            if only_l_detect_streak >= ASYMMETRIC_DETECT_WARN_FRAMES:
                asymmetric_warn_html = (
                    '<div class="warn" style="font-weight:600">'
                    "⚠ Cámara R no detecta el board hace varios segundos "
                    "(L sí). Limpiá el lens, chequeá foco, o asegurate "
                    "que el board entre en su FOV.</div>"
                )
            elif only_r_detect_streak >= ASYMMETRIC_DETECT_WARN_FRAMES:
                asymmetric_warn_html = (
                    '<div class="warn" style="font-weight:600">'
                    "⚠ Cámara L no detecta el board hace varios segundos "
                    "(R sí). Limpiá el lens, chequeá foco, o asegurate "
                    "que el board entre en su FOV.</div>"
                )

            # Colorear los counts L/R: highlightear en ámbar cuando uno es 0 y el otro > 0
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

            l_zero_asym = n_l == 0 and n_r > 0
            r_zero_asym = n_r == 0 and n_l > 0
            detect_pills = (
                f'{_detect_pill("L", n_l, l_zero_asym)} · '
                f'{_detect_pill("R", n_r, r_zero_asym)}'
            )

            early_stop_html = ""
            if state.early_stop_ready and state.early_stop_msg:
                early_stop_html = (
                    f'<div style="margin-top:10px;padding:10px 14px;'
                    f"background:#1e3b2c;border-left:4px solid #2ecc71;"
                    f"border-radius:6px;color:#a8e6c5;font-size:14px;"
                    f'font-weight:500">{state.early_stop_msg}</div>'
                )

            status = f"""
<div data-banner="{banner_escaped}" data-color="{banner_color}" data-progress="{hold_progress:.2f}" data-audioseq="{state.audio_event_seq}" data-audiotext="{audio_escaped}" data-captured="{captured_n}" data-pose-idx="{state.current_pose_idx}"></div>
<div>Pose <b>{state.current_pose_idx + 1}/{len(poses)}</b> — {pose.label} · <b>{pose.tvec_mm[2] / 10.0:.0f}cm</b> <span class="pill-phase">{phase_label}</span></div>
<div>Capturadas: <b>{captured_n}</b> · Skipped: <b>{skipped_n}</b> · Restantes: <b>{len(poses) - captured_n - skipped_n}</b></div>
<div>{rms_html}</div>
{early_stop_html}
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
                end="",
                flush=True,
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
        # En Ctrl+C queremos un hard exit, no un return — si no, el
        # caller del wizard avanzaría a la fase de calibración con las
        # capturas que tengamos, y el HTTP server quedaría vivo
        # holdeando el puerto hasta el eventual exit. (La completion
        # normal mantiene el server vivo a propósito: las fases
        # post-captura — processing, ground-truth, reporte —
        # comparten el mismo HTTP server.)
        _shutting_down = True
        rms_stop.set()
        try:
            cap.close()
        except Exception:
            pass
        try:
            server.shutdown()
        except Exception:
            pass
        sys.exit(0)

    _shutting_down = True
    rms_stop.set()
    cap.close()
    captured_n = sum(1 for s in state.pose_status if s == "captured")
    print(f"\n\nCapturas: {captured_n} pares guardados en {output_dir}")
    return


def cmd_generate_board(args: argparse.Namespace) -> None:
    """Genera una imagen imprimible de un board ChArUco."""
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
    logger.info("Board guardado en %s (%dx%d)", args.output, args.width, args.height)


def cmd_capture(args: argparse.Namespace) -> None:
    """Captura interactiva con preview HTTP y tracking de cobertura."""
    global _shutting_down

    if args.guided:
        _run_guided_capture(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.vision.capture import StereoCapture

    max_exp = getattr(args, "max_exposure_us", 0)
    cap = StereoCapture(
        cam_left_id=args.left,
        cam_right_id=args.right,
        resolution=tuple(args.resolution),
        fps=args.fps,
        max_exposure_us=max_exp if max_exp and max_exp > 0 else None,
        sensor_raw_size=HW.default_res,
        initial_settle_seconds=HW.ae_initial_settle_seconds,
        resettle_seconds=HW.ae_resettle_seconds,
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

    # Arrancar preview HTTP
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Grid de cobertura
    if args.grid == "circular":
        grid_mask = GRID_CIRCULAR
    else:
        grid_mask = GRID_RECTANGULAR
    grid_rows, grid_cols = grid_mask.shape
    coverage = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    # Targets per-cell: para grids binarios usar --per-cell si se
    # setea, si no, sin límite per-cell.
    if args.per_cell > 0:
        cell_target = grid_mask * args.per_cell
    else:
        cell_target = None  # sin límite per-cell

    count = 0
    last_capture_time = 0.0
    valid_cells = int(np.count_nonzero(grid_mask))
    total_target = int(cell_target.sum()) if cell_target is not None else args.count

    # Recomendación de distancia
    cols, rows = args.columns, args.rows
    board_w_mm = cols * args.square_length
    board_h_mm = rows * args.square_length
    board_diag = f"{board_w_mm:.0f}x{board_h_mm:.0f}mm"
    dist_range = (
        f"{args.dist_near_mm/1000:.1f}/{args.dist_mid_mm/1000:.1f}/"
        f"{args.dist_far_mm/1000:.1f}m (protocolo universal — fleet mount 2.0-3.5m)"
    )

    # Trigger manual: Enter en stdin O POST /capture via web UI
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

    logger.info(
        "Captura de calibración — preview: http://people-counter.local:%d", args.port
    )
    logger.info(
        "Board: %s (%s). Distancia recomendada: %s",
        board_diag,
        f"{cols}x{rows}",
        dist_range,
    )
    logger.info(
        "Grid: %s (%dx%d, %d celdas activas, %d capturas totales)",
        args.grid,
        grid_rows,
        grid_cols,
        valid_cells,
        total_target,
    )
    logger.info(
        "IMPORTANTE: Inclinar el board 20-30 grados en cada captura. Nunca sostenerlo plano/frontal."
    )
    logger.info(
        "Mover el ChArUco para cubrir todas las celdas del grid. Variar ángulos (pitch/yaw/roll) en cada celda."
    )
    if args.manual:
        logger.info(
            "Modo MANUAL: apretá ENTER acá O clickeá CAPTURE en la web UI para triggerear."
        )
    else:
        logger.info(
            "Auto-captura cuando el board se detecta en ambas cámaras. %.1fs de cooldown entre capturas.",
            args.cooldown,
        )
    logger.info("Ctrl+C para parar.\n")
    try:
        while True:
            # Condición de stop
            if cell_target is not None:
                if np.all(coverage[grid_mask > 0] >= cell_target[grid_mask > 0]):
                    break
            elif (
                count >= args.count
                and np.count_nonzero(coverage * grid_mask) >= valid_cells
            ):
                break
            frame_l, frame_r = cap.read()

            # Detectar corners con dual-pass (sharpen fallback recupera
            # detecciones marginales para el feedback visual; el fit
            # downstream sigue siendo single-pass).
            corners_l, ids_l = detect_charuco_dual_pass(frame_l, board)
            corners_r, ids_r = detect_charuco_dual_pass(frame_r, board)

            # Armar preview (resize para HTTP)
            vis_l = cv2.resize(frame_l, (648, 486))
            vis_r = cv2.resize(frame_r, (648, 486))
            scale_x = 648 / frame_l.shape[1]
            scale_y = 486 / frame_l.shape[0]

            detected = False
            n_common = 0

            if corners_l is not None and ids_l is not None:
                # Dibujar corners sobre el preview izquierdo
                scaled_corners_l = corners_l.copy()
                scaled_corners_l[:, 0, 0] *= scale_x
                scaled_corners_l[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(
                    vis_l, scaled_corners_l, ids_l, (0, 255, 0)
                )

            if corners_r is not None and ids_r is not None:
                scaled_corners_r = corners_r.copy()
                scaled_corners_r[:, 0, 0] *= scale_x
                scaled_corners_r[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(
                    vis_r, scaled_corners_r, ids_r, (0, 255, 0)
                )

            if (
                corners_l is not None
                and corners_r is not None
                and ids_l is not None
                and ids_r is not None
            ):
                n_common = len(np.intersect1d(ids_l.flatten(), ids_r.flatten()))
                detected = n_common >= 8

            # Dibujar el grid de cobertura sobre el preview izquierdo
            _draw_coverage(vis_l, coverage, grid_mask, cell_target)

            # Texto de status
            color = (0, 255, 0) if detected else (0, 0, 255)
            status = f"Pair {count}/{total_target} | Common: {n_common}"
            cv2.putText(
                vis_l, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            covered_cells = int(np.count_nonzero(coverage * grid_mask))
            coverage_pct = int(covered_cells / valid_cells * 100)
            cov_color = (0, 255, 0) if covered_cells == valid_cells else (0, 200, 255)
            cv2.putText(
                vis_r,
                f"Coverage: {covered_cells}/{valid_cells} ({coverage_pct}%)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cov_color,
                2,
            )

            if detected:
                cv2.putText(
                    vis_r,
                    "BOARD DETECTED",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            combined = np.hstack([vis_l, vis_r])
            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _update_preview(jpeg.tobytes())

            # Trigger: manual (Enter apretado) o auto (detectado + cooldown)
            now = time.time()
            if args.manual:
                should_capture = detected and _trigger_armed
                if _trigger_armed and not detected:
                    logger.warning(
                        "Trigger ignorado — board no detectado en ambas cámaras."
                    )
                    _trigger_armed = False
            else:
                should_capture = detected and (now - last_capture_time) >= args.cooldown
            if should_capture:
                # Chequear si esta celda ya está llena
                row, col = _compute_coverage_center(
                    corners_l,
                    frame_l.shape[1],
                    frame_l.shape[0],
                    grid_mask,
                )
                cell_max = cell_target[row, col] if cell_target is not None else 0
                if cell_max > 0 and coverage[row, col] >= cell_max:
                    # Celda llena — saltear pero mostrar feedback
                    cv2.putText(
                        vis_r,
                        f"Cell ({row},{col}) full ({cell_max}/{cell_max})",
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    left_path = output_dir / f"left_{count:03d}.png"
                    right_path = output_dir / f"right_{count:03d}.png"
                    cv2.imwrite(str(left_path), frame_l)
                    cv2.imwrite(str(right_path), frame_r)

                    coverage[row, col] += 1
                    count += 1
                    last_capture_time = now
                    _trigger_armed = False
                    cell_limit = (
                        cell_target[row, col] if cell_target is not None else "inf"
                    )
                    logger.info(
                        "Par %d/%d guardado — %d esquinas comunes, cobertura %d%%, celda (%d,%d): %d/%s",
                        count,
                        total_target,
                        n_common,
                        coverage_pct,
                        row,
                        col,
                        coverage[row, col],
                        cell_limit,
                    )

            remaining = (
                f" | Missing {valid_cells - covered_cells} cells!"
                if count >= total_target and covered_cells < valid_cells
                else ""
            )
            print(
                f"\r  Pairs: {count}/{total_target} | Common: {n_common:2d} | Coverage: {covered_cells}/{valid_cells}{remaining}   ",
                end="",
                flush=True,
            )
            time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    _shutting_down = True
    cap.close()
    print(f"\n\nCapturados {count} pares en {output_dir}")
    print(
        f"Cobertura: {int(np.count_nonzero(coverage * grid_mask))}/{valid_cells} celdas"
    )
    print("\nGrid de cobertura:")
    print(coverage)
    import os

    os._exit(0)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Corre la calibración estéreo a partir de los pares de imágenes capturados."""
    input_dir = Path(args.input_dir)

    # Buscar todos los archivos left_*.png y matchearlos con right_*.png
    left_files = sorted(input_dir.glob("left_*.png"))
    if not left_files:
        logger.error("No se encontraron archivos left_*.png en %s", input_dir)
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
                logger.warning("Falló la lectura del par: %s", lf.stem)
        else:
            logger.warning("Falta la imagen right para %s", lf.name)

    logger.info("Cargados %d pares de imágenes desde %s", len(pairs), input_dir)

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
        logger.error("Calibración falló: %s", e)
        sys.exit(1)

    save_calibration(result, args.output)
    logger.info("Calibración guardada en %s", args.output)

    # Imprimir resumen
    fx = result["camera_matrix_l"][0, 0]
    fy = result["camera_matrix_l"][1, 1]
    tx = result["T"][0, 0]
    logger.info("Focal izquierdo: fx=%.1f fy=%.1f px", fx, fy)
    logger.info("Baseline (T_x): %.1f mm", abs(tx))


def _wizard_preflight(args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Pre-flight rápido antes de arrancar el wizard. Devuelve (ok_to_continue, messages).

    Los hard failures bloquean (port busy, output dir no escribible).
    Los soft failures avisan pero no abortan (poco espacio en disco,
    calibration.npz ya existe).
    """
    import shutil
    import socket

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
        messages.append(
            f"❌ calibration output dir no escribible: {calib_out.parent} ({e})"
        )
        hard_fail = True
    else:
        messages.append(f"✓ calibration output dir escribible: {calib_out.parent}")

    # Backup de la calibration.npz existente si está
    if calib_out.exists():
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = calib_out.with_suffix(f".npz.bak.{ts}")
        try:
            import shutil as _sh

            _sh.copy2(calib_out, backup)
            messages.append(f"✓ Backup de calibration previa: {backup.name}")
        except Exception as e:
            messages.append(f"⚠ No pude hacer backup de {calib_out}: {e}")

    # Espacio en disco
    try:
        free_mb = shutil.disk_usage(output_dir).free // (1024 * 1024)
        if free_mb < 500:
            messages.append(f"⚠ Poco espacio libre en disco: {free_mb}MB (<500MB)")
        else:
            messages.append(f"✓ Espacio libre: {free_mb}MB")
    except Exception:
        pass

    # Puerto — usar SO_REUSEADDR así un puerto en TIME_WAIT (típico
    # justo después de un Ctrl+C de una corrida previa) no triggerea
    # un falso "puerto ocupado". El HTTP server real también bindea
    # con allow_reuse_address=True, así que el preflight tiene que
    # matchear — si no, el wizard se rehúsa a arrancar cuando en
    # realidad el bind sería exitoso.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    try:
        sock.bind(("0.0.0.0", args.port))
        messages.append(f"✓ Puerto {args.port} libre")
    except OSError:
        # Tratar de encontrar qué PID tiene el puerto así el operador
        # obtiene un error accionable en lugar de "andá a buscarlo".
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
    """Best-effort lookup del PID + comando que tiene tomado un puerto TCP.

    Devuelve (pid, command) en Linux; None en otras plataformas o si
    no podemos identificarlo (ej. el proceso es de otro UID y no
    somos root).
    """
    import os as _os

    if not _os.path.exists("/proc"):
        return None
    # Buscar el inode del listening socket en este puerto
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
    # Recorrer /proc/*/fd para encontrar qué PID tiene ese inode de socket
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
    args: argparse.Namespace,
    calibration: dict,
) -> Optional[dict]:
    """Le pide al operador una distancia conocida, captura un frame,
    corre análisis SGBM + depth de 5 zonas con la calibración
    fresca, devuelve un dict de zones para el reporte HTML. Saltea
    limpio ante input vacío o falla.
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

    max_exp = getattr(args, "max_exposure_us", 0)
    cap = StereoCapture(
        cam_left_id=args.left,
        cam_right_id=args.right,
        resolution=tuple(args.resolution),
        fps=args.fps,
        max_exposure_us=max_exp if max_exp and max_exp > 0 else None,
        sensor_raw_size=HW.default_res,
        initial_settle_seconds=HW.ae_initial_settle_seconds,
        resettle_seconds=HW.ae_resettle_seconds,
    )
    try:
        cap.open()
        # Cuenta regresiva
        for i in range(3, 0, -1):
            print(f"  Capturando en {i}...", end="\r", flush=True)
            time.sleep(1)
        print()
        frame_l, frame_r = cap.read()
    except Exception as e:
        logger.warning("Captura ground-truth falló: %s", e)
        try:
            cap.close()
        except Exception:
            pass
        return None
    finally:
        try:
            cap.close()
        except Exception:
            pass

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
        "center": (h // 2, w // 2),
        "top-left": (margin + half, margin + half),
        "top-right": (margin + half, w - margin - half),
        "bottom-left": (h - margin - half, margin + half),
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

    # Thresholds: <5% @ 2m, <10% @ 3m, ratio borde/centro <2×
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
    # El verdict depende SOLO de la zona del centro — es la única zona
    # cuya distancia el operador midió (con cinta/láser). Las zonas
    # de borde ven lo que sea que esté en la periferia (otros objetos,
    # paredes a distintas profundidades, el piso) y raramente
    # matchean la distancia target en escenas reales. El ratio
    # borde/centro igual se computa y se muestra en el reporte como
    # informativo, pero no gate PASS.
    if center is not None:
        center_err_abs = abs(center[2])
        if edges:
            edge_errs = [abs(v[2]) for v in edges]
            edge_ratio = max(edge_errs) / max(center_err_abs, 0.1)
        overall_pass = center_err_abs <= center_threshold

    zones["_pass"] = overall_pass
    zones["_distance_mm"] = distance_mm
    zones["_center_err"] = center_err_abs
    zones["_edge_ratio"] = edge_ratio
    zones["_center_threshold"] = center_threshold

    # Heatmap de profundidad para el reporte — imagen de escena con overlay coloreado por disparity.
    try:
        valid_mask = disparity > 0.1
        norm = np.zeros_like(disparity, dtype=np.uint8)
        if valid_mask.any():
            vmin = float(disparity[valid_mask].min())
            vmax = float(disparity[valid_mask].max())
            if vmax > vmin:
                norm[valid_mask] = np.clip(
                    (disparity[valid_mask] - vmin) / (vmax - vmin) * 255,
                    0,
                    255,
                ).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        heat[~valid_mask] = (40, 40, 40)
        # Lado a lado: L rectificado + heatmap
        side = np.hstack([rect_l, heat])
        # Marcar el centro de cada zona con una caja + label
        for name, (cy, cx) in zones_coords.items():
            px_x = cx + rect_l.shape[1]  # shifteado al lado del heatmap
            cv2.rectangle(
                side,
                (px_x - half, cy - half),
                (px_x + half, cy + half),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                side,
                name,
                (px_x - half, cy - half - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        gt_viz_path = Path(args.output).parent / "ground_truth_depth.png"
        cv2.imwrite(str(gt_viz_path), side)
        zones["_image_path"] = str(gt_viz_path)
    except Exception as e:
        logger.warning("No pude guardar viz de ground-truth: %s", e)

    logger.info(
        "Ground-truth: centro err=%.2f%% (umbral %.1f%%), borde/centro=%.2f× (informativo) → %s",
        center_err_abs,
        center_threshold,
        edge_ratio,
        "PASS" if overall_pass else "FAIL",
    )
    return zones


# ---------------------------------------------------------------------------
# Modo barrido libre (sweep): captura continua con auto-selección por novedad
# ---------------------------------------------------------------------------

# Tuning del sweep (se afina en hardware; defaults conservadores).
SWEEP_GRID = 3  # grilla NxN de la posición del board en el cuadro
SWEEP_NOVELTY_MIN = 0.12  # distancia mínima en el espacio de firma para aceptar
SWEEP_DIST_BANDS = 3  # bandas de distancia por tamaño aparente del board
SWEEP_MIN_TILTED = 4  # frames con tilt >= SWEEP_TILT_DEG_MIN requeridos
SWEEP_TILT_DEG_MIN = 12.0  # grados para considerar una pose "inclinada"
SWEEP_TARGET_CAPTURES = 18  # capturas para considerar la cobertura completa
# ~23px a 1152 — tolera el temblor natural de sostener el board a mano.
SWEEP_STILL_FRAC = (
    0.02  # desplazamiento máx del centroide (fracción del ancho) p/ "quieto"
)
SWEEP_STILL_FRAMES = 2  # frames consecutivos casi-quietos antes de aceptar una captura


def _sweep_signature(
    corners: np.ndarray,
    image_shape: tuple,
    rvec: Optional[np.ndarray] = None,
) -> dict:
    """Firma de pose normalizada de una detección ChArUco, para los gates de
    novedad y cobertura del sweep.

    Devuelve dict con ``cx``/``cy`` (centroide normalizado en [0,1]), ``size``
    (fracción de área del bbox del board sobre el frame — proxy de distancia) y
    ``tilt`` (grados de inclinación desde ``rvec``; 0 si no se pasa rvec).
    """
    h, w = image_shape[:2]
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    cx = float(pts[:, 0].mean()) / max(w, 1)
    cy = float(pts[:, 1].mean()) / max(h, 1)
    bw = float(pts[:, 0].max() - pts[:, 0].min())
    bh = float(pts[:, 1].max() - pts[:, 1].min())
    size = (bw * bh) / float(max(w * h, 1))
    tilt = 0.0
    if rvec is not None:
        rv = np.asarray(rvec, dtype=np.float64).reshape(-1)
        tilt = float(np.degrees(np.hypot(float(rv[0]), float(rv[1]))))
    return {"cx": cx, "cy": cy, "size": size, "tilt": tilt}


def _sweep_novelty_distance(sig: dict, accepted: list) -> float:
    """Distancia mínima (L2 ponderada) de ``sig`` a las firmas ya aceptadas.
    ``inf`` si no hay ninguna aceptada. Pondera fuerte la posición (cx,cy),
    medio el tamaño y el tilt (normalizado a 60°)."""
    if not accepted:
        return float("inf")

    def _d(a: dict, b: dict) -> float:
        ta = min(a["tilt"], 60.0) / 60.0
        tb = min(b["tilt"], 60.0) / 60.0
        return (
            (a["cx"] - b["cx"]) ** 2
            + (a["cy"] - b["cy"]) ** 2
            + 0.5 * (a["size"] - b["size"]) ** 2
            + 0.5 * (ta - tb) ** 2
        ) ** 0.5

    return min(_d(sig, s) for s in accepted)


def _sweep_is_still(
    recent_centroids: list,
    max_disp_px: float,
    min_frames: int,
) -> bool:
    """True si el board estuvo casi quieto en los últimos frames: hay al menos
    ``min_frames`` centroides recientes y el desplazamiento entre cada par
    consecutivo es < ``max_disp_px``. Evita capturar una pose en pleno
    movimiento (blur + skew L/R en tránsito)."""
    if len(recent_centroids) < min_frames:
        return False
    recent = recent_centroids[-min_frames:]
    for (x0, y0), (x1, y1) in zip(recent, recent[1:]):
        if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 >= max_disp_px:
            return False
    return True


def _sweep_coverage(
    accepted: list,
    grid: int = SWEEP_GRID,
    dist_bands: int = SWEEP_DIST_BANDS,
    min_tilted: int = SWEEP_MIN_TILTED,
    tilt_deg_min: float = SWEEP_TILT_DEG_MIN,
    target: int = SWEEP_TARGET_CAPTURES,
) -> dict:
    """Resumen de cobertura del set aceptado: frames por celda de posición
    (grilla NxN), bandas de distancia ocupadas, frames inclinados y total.
    ``complete`` = listo para calibrar (todas las celdas + >=2 bandas +
    suficientes inclinados + total >= target). ``missing`` = hints de lo que
    falta."""
    cells = [[0] * grid for _ in range(grid)]
    bands = [0] * dist_bands
    n_tilted = 0
    for s in accepted:
        gx = min(grid - 1, max(0, int(s["cx"] * grid)))
        gy = min(grid - 1, max(0, int(s["cy"] * grid)))
        cells[gy][gx] += 1
        b = min(dist_bands - 1, max(0, int((max(0.0, s["size"]) ** 0.5) * dist_bands)))
        bands[b] += 1
        if s["tilt"] >= tilt_deg_min:
            n_tilted += 1
    n = len(accepted)
    cells_covered = sum(1 for row in cells for c in row if c > 0)
    bands_covered = sum(1 for c in bands if c > 0)
    complete = (
        n >= target
        and cells_covered >= grid * grid
        and bands_covered >= 2
        and n_tilted >= min_tilted
    )
    missing: list[str] = []
    if cells_covered < grid * grid:
        missing.append(f"{grid * grid - cells_covered} zonas del cuadro")
    if bands_covered < 2:
        missing.append("otra distancia (acercá/alejá el board)")
    if n_tilted < min_tilted:
        missing.append(f"{min_tilted - n_tilted} poses inclinadas")
    if n < target:
        missing.append(f"{max(0, target - n)} capturas más")
    return {
        "cells": cells,
        "bands": bands,
        "n_tilted": n_tilted,
        "n": n,
        "cells_covered": cells_covered,
        "bands_covered": bands_covered,
        "complete": complete,
        "missing": missing,
    }


def _sweep_coverage_html(cov: dict) -> str:
    """Renderiza el panel de estado del sweep: grilla de cobertura + contadores."""
    rows_html = ""
    for row in cov["cells"]:
        spans = "".join(
            '<span style="display:inline-block;width:34px;height:34px;margin:2px;'
            "border-radius:6px;text-align:center;line-height:34px;font-size:13px;"
            f'color:#fff;background:{"#27ae60" if c > 0 else "#3a3a42"}">'
            f'{c if c else ""}</span>'
            for c in row
        )
        rows_html += f"<div>{spans}</div>"
    if cov["complete"]:
        head = (
            '<div style="color:#2ecc71;font-size:18px;font-weight:700">'
            "COBERTURA COMPLETA — apretá Finalizar para calibrar</div>"
        )
    else:
        falta = " · ".join(cov["missing"]) if cov["missing"] else "—"
        head = (
            '<div style="color:#f1c40f;font-size:16px;font-weight:600">'
            f"Barré el board · falta: {falta}</div>"
        )
    return (
        head
        + f'<div style="margin:10px 0">{rows_html}</div>'
        + '<div style="color:#888;font-size:13px">'
        + f'Capturas: {cov["n"]} · zonas {cov["cells_covered"]}/'
        + f'{SWEEP_GRID * SWEEP_GRID} · distancias {cov["bands_covered"]}/'
        + f'{SWEEP_DIST_BANDS} · inclinadas {cov["n_tilted"]}/{SWEEP_MIN_TILTED}</div>'
    )


def _sweep_html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Calibración — barrido libre</title>
<style>
  *{box-sizing:border-box}
  body{background:#0b0b0d;margin:0;color:#eee;
       font-family:-apple-system,Segoe UI,Roboto,sans-serif;
       display:flex;flex-direction:column;min-height:100vh}
  header{padding:10px 20px;background:#141418;border-bottom:1px solid #26262c}
  header h1{margin:0;font-size:16px;font-weight:600}
  #stream{max-width:100vw;max-height:55vh;object-fit:contain;display:block;
          margin:0 auto;background:#000}
  #panel{flex:1;padding:14px 20px;background:#141418}
  .btn{padding:12px 22px;font-size:15px;border:none;border-radius:8px;
       cursor:pointer;font-weight:600;color:#fff;margin-right:10px}
  .btn-finish{background:#27ae60}
  .btn-undo{background:#e67e22}
  .btn-report{background:#3498db}
  #ov{position:fixed;inset:0;background:rgba(11,11,13,0.95);z-index:99;
      display:flex;align-items:center;justify-content:center;
      flex-direction:column;gap:18px;padding:20px;text-align:center}
</style></head>
<body>
  <header><h1>Calibración — barrido libre</h1></header>
  <div id="ov">
    <div style="color:#eee;font-size:24px;font-weight:700">Barrido libre</div>
    <div style="color:#aaa;max-width:480px;line-height:1.6">
      Apretá <b>Comenzar</b> y movés el board ChArUco lento por todo el cuadro:
      acercándolo y alejándolo, a cada esquina, e inclinándolo. La herramienta
      agarra sola los frames diversos que necesita. Terminás cuando la
      cobertura esté completa (o cuando quieras).
    </div>
    <button class="btn btn-finish" style="font-size:18px;padding:14px 36px"
       onclick="fetch('/start',{method:'POST'});this.parentElement.style.display='none'">
      Comenzar</button>
  </div>
  <img id="stream" src="/stream"/>
  <div id="panel">
    <div id="status">Conectando...</div>
    <div style="margin-top:16px">
      <button class="btn btn-undo" onclick="fetch('/undo',{method:'POST'})">Deshacer última</button>
      <button class="btn btn-finish" onclick="if(confirm('Finalizar y calibrar?')){fetch('/finish',{method:'POST'})}">Finalizar</button>
      <button class="btn btn-report" onclick="window.open('/report','_blank')">Abrir reporte</button>
    </div>
  </div>
<script>
function refresh(){
  fetch('/status').then(function(r){return r.text()}).then(function(t){
    document.getElementById('status').innerHTML=t;
  }).catch(function(e){});
}
setInterval(refresh,250);
</script>
</body></html>"""


def _run_sweep_capture(args: argparse.Namespace) -> None:
    """Captura por barrido libre: el operador mueve el board y la herramienta
    auto-selecciona frames diversos (gate de novedad + el MISMO gate de calidad
    que el guiado) hasta cubrir la grilla de posición + distancias +
    inclinaciones. Reemplaza la captura por-pose; el resto del wizard
    (procesar/calibrar/reporte) sigue igual — globea ``left_*.png`` del output
    dir. No soporta --resume (limpia las capturas previas al arrancar)."""
    global _shutting_down, _guided_state, _latest_jpeg, _sweep_mode, _capture_started
    _sweep_mode = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.vision.capture import StereoCapture

    max_exp = getattr(args, "max_exposure_us", 0)
    cap = StereoCapture(
        cam_left_id=args.left,
        cam_right_id=args.right,
        resolution=tuple(args.resolution),
        fps=args.fps,
        meter_mode=getattr(args, "meter", "matrix"),
        lock_ae=getattr(args, "lock_ae", False),
        max_exposure_us=max_exp if max_exp and max_exp > 0 else None,
        sensor_raw_size=HW.default_res,
        initial_settle_seconds=HW.ae_initial_settle_seconds,
        resettle_seconds=HW.ae_resettle_seconds,
    )
    cap.open()

    dict_id = _resolve_aruco_dict(getattr(args, "aruco_dict", "DICT_4X4_100"))
    board = create_charuco_board(
        board_size=(args.columns, args.rows),
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_id=dict_id,
        legacy_pattern=args.legacy_pattern,
    )
    obj_all = board.getChessboardCorners()

    # K nominal escalada a la resolución de captura — SOLO para el proxy de
    # tilt (solvePnP grueso sobre frames aceptados); NO entra a la calibración.
    w_cap, h_cap = int(args.resolution[0]), int(args.resolution[1])
    scale = w_cap / float(HW.full_res[0])
    f_nom = HW.nominal_focal_full_px * scale
    K_nom = np.array(
        [[f_nom, 0, w_cap / 2.0], [0, f_nom, h_cap / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )

    # Sweep no soporta resume: limpiar capturas previas del output dir.
    for old in list(output_dir.glob("left_*.png")) + list(
        output_dir.glob("right_*.png")
    ):
        try:
            old.unlink()
        except OSError:
            pass

    state = _GuidedState()
    state.pose_status = []
    _guided_state = state
    _capture_started = False

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _GuidedHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    logger.info("Barrido libre — preview: http://people-counter.local:%d", args.port)
    logger.info("Esperando que el operador haga click en Comenzar...")
    with state.lock:
        state.status_html = (
            '<div style="color:#aaa">Apretá Comenzar y barré el board por '
            "todo el cuadro.</div>"
        )

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

    try:
        cap.resettle_and_lock()
    except Exception as e:
        logger.warning("resettle_and_lock no aplicado: %s", e)

    logger.info("Barrido en curso — Finalizar en la UI cuando la cobertura esté lista.")
    accepted_sigs: list = []
    count = 0
    novelty_min = float(getattr(args, "sweep_novelty", SWEEP_NOVELTY_MIN))
    last_status_t = 0.0
    recent_centroids: list = []
    still_px = SWEEP_STILL_FRAC * w_cap
    last_reject_log_t = 0.0

    while not _shutting_down:
        with state.lock:
            if state.finish_requested:
                state.finish_requested = False
                break
            undo = state.undo_requested
            state.undo_requested = False
        if undo and state.captured_pairs:
            lp, rp, _pid = state.captured_pairs.pop()
            for p in (lp, rp):
                try:
                    p.unlink()
                except OSError:
                    pass
            if accepted_sigs:
                accepted_sigs.pop()
            count = max(0, count - 1)
            logger.info("UNDO — removida última captura del sweep (quedan %d)", count)

        try:
            frame_l, frame_r, ts_l, ts_r, _tl, _tr = cap.read_with_metadata()
            lr_sync_ok = abs(ts_l - ts_r) <= LR_SYNC_MAX_DELTA_NS
        except Exception:
            frame_l, frame_r = cap.read()
            lr_sync_ok = True

        corners_l, ids_l = detect_charuco_dual_pass(frame_l, board, min_corners=4)
        corners_r, ids_r = detect_charuco_dual_pass(frame_r, board, min_corners=4)

        # Quietud: trackear el centroide del board entre frames. Solo se captura
        # cuando está casi quieto (no en tránsito) — evita blur + skew L/R de una
        # pose en movimiento, igual que el hold del modo guiado pero liviano.
        if corners_l is not None:
            _cp = corners_l.reshape(-1, 2)
            recent_centroids.append((float(_cp[:, 0].mean()), float(_cp[:, 1].mean())))
            recent_centroids[:] = recent_centroids[-SWEEP_STILL_FRAMES:]
        # NO se limpia ante un None: un parpadeo de detección no debe resetear el
        # streak de quietud (si el board no se movió en el gap, sigue quieto).
        still = _sweep_is_still(recent_centroids, still_px, SWEEP_STILL_FRAMES)

        vis_l = frame_l.copy()
        if corners_l is not None and ids_l is not None:
            try:
                cv2.aruco.drawDetectedCornersCharuco(vis_l, corners_l, ids_l)
            except cv2.error:
                pass
        combined = np.hstack([vis_l, frame_r])

        accepted_this = False
        n_corners_l = len(corners_l) if corners_l is not None else 0
        if corners_l is not None and ids_l is not None and n_corners_l >= 4:
            rvec = None
            try:
                obj = obj_all[ids_l.flatten()].astype(np.float32)
                img = corners_l.reshape(-1, 2).astype(np.float32)
                ok, rvec, _tv = cv2.solvePnP(
                    obj, img, K_nom, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    rvec = None
            except cv2.error:
                rvec = None
            sig = _sweep_signature(corners_l, frame_l.shape, rvec)
            if still and _sweep_novelty_distance(sig, accepted_sigs) >= novelty_min:
                common_n = count_common_corners(ids_l, ids_r)
                quality = assess_frame_quality(
                    frame_l,
                    frame_r,
                    n_corners_l,
                    corners_l=corners_l,
                    corners_r=corners_r,
                )
                reject: list[str] = []
                if not quality["all_pass"]:
                    reject.extend(quality.get("reasons", ["calidad"]))
                if common_n < LR_MIN_COMMON_CORNERS:
                    reject.append(
                        f"esquinas comunes {common_n}<{LR_MIN_COMMON_CORNERS}"
                    )
                if not lr_sync_ok:
                    reject.append("L/R desincronizadas")
                if not reject:
                    ordinal = count
                    lp = output_dir / f"left_{ordinal:03d}_sweep.png"
                    rp = output_dir / f"right_{ordinal:03d}_sweep.png"
                    cv2.imwrite(str(lp), frame_l)
                    cv2.imwrite(str(rp), frame_r)
                    with state.lock:
                        state.captured_pairs.append((lp, rp, f"sweep{ordinal:03d}"))
                    accepted_sigs.append(sig)
                    count += 1
                    accepted_this = True
                    logger.info(
                        "[sweep %d] capturada cx=%.2f cy=%.2f size=%.3f "
                        "tilt=%.0f common=%d",
                        count,
                        sig["cx"],
                        sig["cy"],
                        sig["size"],
                        sig["tilt"],
                        common_n,
                    )
                elif time.time() - last_reject_log_t > 2.0:
                    logger.info(
                        "sweep: frame estable rechazado — %s", "; ".join(reject)
                    )
                    last_reject_log_t = time.time()

        if accepted_this:
            cv2.rectangle(
                combined,
                (0, 0),
                (combined.shape[1] - 1, combined.shape[0] - 1),
                (0, 255, 0),
                8,
            )

        cov = _sweep_coverage(accepted_sigs)
        now = time.time()
        if accepted_this or now - last_status_t > 0.25:
            panel = _sweep_coverage_html(cov)
            if not cov["complete"] and corners_l is not None and not still:
                panel = (
                    '<div style="color:#e67e22;font-size:14px;margin-bottom:6px">'
                    "Pará el board un instante para capturar…</div>"
                ) + panel
            with state.lock:
                state.status_html = panel
            last_status_t = now

        _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with _jpeg_lock:
            _latest_jpeg = jpeg.tobytes()

    # Liberar cámaras; el server queda vivo (daemon) para que el wizard sirva
    # la fase de procesamiento/reporte vía /status, igual que el guiado.
    cap.close()
    logger.info("Barrido finalizado: %d capturas en %s", count, output_dir)


def cmd_wizard(args: argparse.Namespace) -> None:
    """Wizard de calibración one-shot: preflight → captura guiada → calibrar →
    verificar → ground-truth check → reporte."""
    from src.vision.report import generate_html_report, save_report

    if getattr(args, "low_light", False):
        _apply_low_light_overrides()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fase 0 — pre-flight
    logger.info("=" * 60)
    logger.info("WIZARD FASE 0/4 — Pre-flight checks")
    logger.info("=" * 60)
    ok, messages = _wizard_preflight(args)
    for m in messages:
        logger.info(m)
    if not ok:
        logger.error("Pre-flight falló. Resolvé los items marcados con ❌ y reintentá.")
        sys.exit(1)

    # Fase 1 — captura
    logger.info("=" * 60)
    logger.info("WIZARD FASE 1/4 — Captura")
    logger.info("=" * 60)
    # Barrido libre es el modo DEFAULT; --guided usa el modo por poses-silueta
    # (más preciso, requiere más espacio + paciencia).
    use_sweep = not getattr(args, "guided", False)
    if use_sweep:
        _run_sweep_capture(args)
    else:
        args.grid = "rectangular"
        args.manual = getattr(args, "manual", False)
        args.per_cell = 0
        args.cooldown = 1.5
        args.count = 20
        _run_guided_capture(args)
    _set_post_capture_phase(
        "processing",
        "Procesando capturas, analizando cobertura y detectando esquinas "
        "ChArUco en cada par...",
        progress_pct=10,
    )

    # Juntar los pares capturados (los que sobrevivieron a los UNDOs).
    # Los pose IDs se trackean en _guided_state.captured_pairs — caer
    # al ordinal si falta el state.
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
        (args.columns, args.rows),
        args.square_length,
        args.marker_length,
        dict_id,
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
        # Re-detectar con el MISMO modo lenient que va a usar la
        # fase de calibración así este sanity count matchea lo que
        # calibrate_stereo realmente va a ver.
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
            "complete",
            msg,
            verdict="FAIL",
            report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    # Sanity pre-calibración: contar pares donde AMBAS cámaras
    # produjeron una detección utilizable (≥ 8 esquinas comunes es
    # contra lo que gatea _detect_all_pairs de calibrate_stereo). Si
    # demasiados pares fallan esta re-detección, el loop de captura
    # en vivo aceptó frames que en realidad no van a alimentar la
    # matemática de calibración — y calibraríamos silencioso sobre
    # un subset chico, produciendo un fit degenerado.
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
        valid_both,
        len(pairs),
        detect_rate * 100,
        min_rate * 100,
    )
    if detect_rate < min_rate:
        offending = "\n".join(f"• {ln}" for ln in invalid_lines)
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
            "complete",
            msg,
            verdict="FAIL",
            report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    # Check de diversity — avisar si el set capturado es degenerado
    coverage = analyze_pose_coverage(
        captured_pose_ids,
        all_poses=default_pose_sequence(
            near_mm=args.dist_near_mm,
            mid_mm=args.dist_mid_mm,
            far_mm=args.dist_far_mm,
        ),
        near_mm=args.dist_near_mm,
        mid_mm=args.dist_mid_mm,
        far_mm=args.dist_far_mm,
    )
    logger.info(
        "Cobertura: distancia %s · tilts %s",
        coverage["by_distance"],
        coverage["by_tilt_axis"],
    )
    # Los critical gaps son hard-blockers: la falta de grupos enteros
    # de poses o bandas de distancia produce calibraciones
    # degeneradas (RMS bajo pero geométricamente incorrectas). El
    # operador puede overridear con --force-degenerate-coverage
    # cuando sabe lo que está haciendo.
    critical_gaps = coverage.get("critical", [])
    force_flag = getattr(args, "force_degenerate_coverage", False)
    # En modo barrido las capturas no siguen la taxonomía de poses (grupos
    # A/B/C/D, bandas near/mid/far), así que analyze_pose_coverage las ve
    # "vacías" y este gate por-pose siempre fallaría. El barrido tiene su propia
    # cobertura (grilla de posición × distancia × tilt) durante la captura, así
    # que se omite acá — la calidad la validan el RMS + ground-truth.
    if use_sweep and critical_gaps:
        logger.info(
            "Coverage por-pose omitido en modo barrido (usa su propia cobertura)."
        )
    if critical_gaps and not force_flag and not use_sweep:
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
            "complete",
            msg,
            verdict="FAIL",
            report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    soft_warnings = coverage.get("warnings", [])
    if soft_warnings and not use_sweep:
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
            logger.info(
                "Calibración cancelada — podés recapturar y volver a correr el wizard."
            )
            _set_post_capture_phase(
                "complete",
                "Calibración cancelada por el operador (diversidad limitada).",
                verdict="FAIL",
                report_available=False,
            )
            time.sleep(10)
            sys.exit(0)

    # Fase 2 — calibrar
    logger.info("=" * 60)
    logger.info("WIZARD FASE 2/4 — Calibración estéreo con %d pares", len(pairs))
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
            "complete",
            msg,
            verdict="FAIL",
            report_available=False,
        )
        time.sleep(10)
        sys.exit(1)

    calib_out = Path(args.output)
    save_calibration(result, str(calib_out))

    fx_l = float(result["camera_matrix_l"][0, 0])
    baseline_mm = float(abs(result["T"][0, 0]))
    logger.info("fx=%.1f px   baseline=%.2f mm (diseño 140mm)", fx_l, baseline_mm)

    board_for_residuals = create_charuco_board(
        (args.columns, args.rows),
        args.square_length,
        args.marker_length,
        dict_id,
        legacy_pattern=getattr(args, "legacy_pattern", True),
    )

    # Fase 3 — verificar (epipolar) + residuales per-pair
    logger.info("=" * 60)
    logger.info("WIZARD FASE 3/4 — Verificación + residuales por par")
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
                median,
                n_outliers,
            )
    except Exception as e:
        logger.warning("Residuales per-pair fallaron: %s", e)
        per_pair = None

    # Fase 4 — ground-truth depth check
    logger.info("=" * 60)
    logger.info("WIZARD FASE 4/4 — Ground-truth depth check (opcional)")
    _set_post_capture_phase(
        "ground_truth",
        "Fase opcional de validación con distancia conocida — mirá la "
        "terminal para continuar.",
        progress_pct=85,
    )
    logger.info("=" * 60)
    diagnose_zones = _run_ground_truth_phase(args, result)

    # Reporte final
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
        baseline_tol_mm=getattr(args, "baseline_tol_mm", 5.0),
    )
    report_name = f"calibration_report_{args.device_id}_{ts:%Y%m%d_%H%M%S}.html"
    report_path = save_report(html, calib_out.parent / report_name)

    # Emitir un JSON sibling con las mismas métricas en una forma
    # parseable. Permite que las comparaciones de QA / cross-fleet
    # grepen scores numéricos sin parsear el HTML. La shape mirrorea
    # diagnose_depth.py --json así el mismo tooling downstream
    # consume cualquiera de los dos outputs.
    try:
        json_payload: dict[str, object] = {
            "device_id": args.device_id,
            "timestamp": ts.isoformat(timespec="seconds"),
            "rms_stereo_px": (None if rms_est is None else float(rms_est)),
            "baseline_mm": float(baseline_mm),
            "baseline_design_mm": 140.0,
            "baseline_delta_mm": float(baseline_mm - 140.0),
        }
        try:
            from src.vision.calibration import lens_alignment_metrics

            json_payload["alignment"] = lens_alignment_metrics(
                result["R"],
                result["T"],
            )
        except Exception:
            pass
        if diagnose_zones:
            zones_out: dict[str, object] = {}
            for k, v in diagnose_zones.items():
                if k.startswith("_"):
                    continue
                if v is None:
                    zones_out[k] = None
                else:
                    depth, std, err_pct, fill = v
                    zones_out[k] = {
                        "depth_mm": float(depth),
                        "std_mm": float(std),
                        "err_pct": float(err_pct),
                        "fill_pct": float(fill),
                    }
            json_payload["depth_validation"] = {
                "distance_mm": float(diagnose_zones.get("_distance_mm", 0)),
                "center_threshold_pct": float(
                    diagnose_zones.get("_center_threshold", 0),
                ),
                "center_err_pct": float(
                    diagnose_zones.get("_center_err", 0),
                ),
                "edge_ratio": float(
                    diagnose_zones.get("_edge_ratio", float("nan")),
                ),
                "verdict": ("PASS" if diagnose_zones.get("_pass") else "FAIL"),
                "zones": zones_out,
            }
        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(json_payload, indent=2),
            encoding="utf-8",
        )
        logger.info("  Reporte JSON: %s", json_path)
    except Exception:
        logger.exception("Falló el sidecar JSON de calibración (no fatal)")

    logger.info("=" * 60)
    logger.info("WIZARD COMPLETO")
    logger.info("  Calibración: %s", calib_out)
    logger.info("  Reporte HTML: %s", report_path)
    logger.info(
        "  Baseline: %.2f mm (diseño 140mm, Δ %+.2f mm)",
        baseline_mm,
        baseline_mm - 140.0,
    )
    if rms_est is not None:
        logger.info("  RMS estimado: %.3f px", rms_est)
    if diagnose_zones and diagnose_zones.get("_pass") is not None:
        logger.info(
            "  Ground-truth depth: %s", "PASS" if diagnose_zones["_pass"] else "FAIL"
        )
    logger.info("=" * 60)

    if abs(baseline_mm - 140.0) > 5.0:
        logger.warning(
            "⚠ Baseline estimada fuera de tolerancia (%.2fmm vs diseño 140mm, Δ %+.2fmm). "
            "La baseline se estima del set de capturas — si son pocas o poco "
            "diversas el solver converge con error.",
            baseline_mm,
            baseline_mm - 140.0,
        )

    # Apuntar el endpoint /report del capture-UI al reporte guardado
    # así el browser puede auto-abrirlo vía el HTTP server existente.
    # También lo seguimos sirviendo vía QR (puerto distinto) para
    # distribución a un teléfono.
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

    # Grace period así el browser levanta el status "complete" y abre
    # el reporte en una pestaña nueva antes de que el daemon del HTTP
    # server muera con main.
    time.sleep(10)


def cmd_verify(args: argparse.Namespace) -> None:
    """Verifica la calibración mostrando pares rectificados con líneas epipolares."""
    cal = load_calibration(args.calibration)
    input_dir = Path(args.input_dir)

    left_files = sorted(input_dir.glob("left_*.png"))
    if not left_files:
        logger.error("No se encontraron imágenes en %s", input_dir)
        sys.exit(1)

    # Usar el primer par para verificación
    lf = left_files[0]
    rf = lf.parent / lf.name.replace("left_", "right_")
    img_l = cv2.imread(str(lf))
    img_r = cv2.imread(str(rf))

    rect_l, rect_r = rectify_pair(img_l, img_r, cal)

    # Dibujar líneas epipolares horizontales sobre el par rectificado
    combined = np.hstack([rect_l, rect_r])
    h = combined.shape[0]
    for y in range(0, h, 30):
        color = (0, 255, 0) if (y // 30) % 2 == 0 else (0, 200, 255)
        cv2.line(combined, (0, y), (combined.shape[1], y), color, 1)

    output_path = str(Path(args.calibration).parent / "verify_epipolar.png")
    cv2.imwrite(output_path, combined)
    logger.info("Imagen de verificación guardada en %s", output_path)
    logger.info(
        "Chequeá que los features correspondientes caigan sobre la misma línea horizontal."
    )


def cmd_reset(args: argparse.Namespace) -> None:
    """Borra captures + session.json + calibration.npz así la próxima
    corrida del wizard arranca desde slate limpio. Útil cuando una
    sesión salió mal (capturas degeneradas, resolución mismatcheada,
    estado corrupto) y no querés debuggear — querés arrancar de
    cero.
    """
    output_dir = Path(args.output_dir)
    calib_out = Path(args.output)
    items_to_remove: list[tuple[str, Path]] = []

    if output_dir.exists():
        captures = list(output_dir.glob("left_*.png")) + list(
            output_dir.glob("right_*.png")
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
    global HW
    HW = load_hardware_params()

    parser = argparse.ArgumentParser(
        description="Tool de calibración estéreo para People Counter"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate-board ---
    p_board = sub.add_parser(
        "generate-board", help="Genera el board ChArUco imprimible"
    )
    p_board.add_argument("--output", default="calibration/charuco_board.png")
    p_board.add_argument(
        "--board-cols",
        "--columns",
        type=int,
        dest="columns",
        default=HW.board_cols,
        help="Columnas del board. Default desde vision.charuco.board_cols.",
    )
    p_board.add_argument(
        "--board-rows",
        "--rows",
        type=int,
        dest="rows",
        default=HW.board_rows,
        help="Filas del board. Default desde vision.charuco.board_rows.",
    )
    p_board.add_argument(
        "--square-mm",
        "--square-length",
        type=float,
        dest="square_length",
        default=HW.square_mm,
        help="Lado del cuadrado en mm. Default desde vision.charuco.square_mm.",
    )
    p_board.add_argument(
        "--marker-mm",
        "--marker-length",
        type=float,
        dest="marker_length",
        default=HW.marker_mm,
        help="Lado del marker en mm. Default desde vision.charuco.marker_mm.",
    )
    p_board.add_argument(
        "--dict",
        dest="aruco_dict",
        default="DICT_4X4_100",
        help="Nombre del dict ArUco (ej. DICT_4X4_100, DICT_5X5_100). Default DICT_4X4_100",
    )
    p_board.add_argument(
        "--legacy-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar enumeración de markers ChArUco pre-4.6. Default True matchea calib.io.",
    )
    p_board.add_argument(
        "--width",
        type=int,
        default=4961,
        help="Ancho de la imagen (px) — default A3 landscape @ 300 DPI",
    )
    p_board.add_argument(
        "--height",
        type=int,
        default=3508,
        help="Alto de la imagen (px) — default A3 landscape @ 300 DPI",
    )
    p_board.set_defaults(func=cmd_generate_board)

    # --- capture ---
    p_cap = sub.add_parser("capture", help="Captura interactiva con preview HTTP")
    p_cap.add_argument(
        "--left",
        type=int,
        default=0,
        help="Índice de la cámara izquierda (lente izquierda mirando desde la cámara hacia la escena). Default 0 matchea el wiring de la flota.",
    )
    p_cap.add_argument(
        "--right", type=int, default=1, help="Índice de la cámara derecha. Default 1."
    )
    p_cap.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        help="Resolución de captura. Default: lee vision.resolution "
        "de /etc/people-counter/config.yaml (fuente única de verdad — "
        "garantiza match con el runtime). Pasá explícito solo para "
        "tests / dev workstation donde no hay config per-device.",
    )
    p_cap.add_argument("--fps", type=int, default=5)
    p_cap.add_argument("--output-dir", default="./calibration/captures")
    p_cap.add_argument("--count", type=int, default=30, help="Número mínimo de pares")
    p_cap.add_argument(
        "--per-cell",
        type=int,
        default=0,
        help="Máximo de capturas por celda del grid (0=ilimitado). Para la celda al alcanzarlo.",
    )
    p_cap.add_argument(
        "--cooldown-seconds",
        "--cooldown",
        type=float,
        dest="cooldown",
        default=1.5,
        help="Segundos a esperar después de cada captura antes de la próxima",
    )
    p_cap.add_argument(
        "--pose-timeout-seconds",
        "--pose-timeout-sec",
        type=float,
        dest="pose_timeout_sec",
        default=SKIP_POSE_TIMEOUT_SEC,
        help=f"Segundos antes de que una pose no capturada sea "
        f"auto-skippeada (default {SKIP_POSE_TIMEOUT_SEC:.0f}).",
    )
    p_cap.add_argument("--port", type=int, default=8080, help="Puerto del preview HTTP")
    p_cap.add_argument(
        "--board-cols",
        "--columns",
        type=int,
        dest="columns",
        default=HW.board_cols,
        help="Columnas del board. Default desde vision.charuco.board_cols.",
    )
    p_cap.add_argument(
        "--board-rows",
        "--rows",
        type=int,
        dest="rows",
        default=HW.board_rows,
        help="Filas del board. Default desde vision.charuco.board_rows.",
    )
    p_cap.add_argument(
        "--square-mm",
        "--square-length",
        type=float,
        dest="square_length",
        default=HW.square_mm,
        help="Lado del cuadrado en mm. Default desde vision.charuco.square_mm.",
    )
    p_cap.add_argument(
        "--marker-mm",
        "--marker-length",
        type=float,
        dest="marker_length",
        default=HW.marker_mm,
        help="Lado del marker en mm. Default desde vision.charuco.marker_mm.",
    )
    p_cap.add_argument(
        "--dict",
        dest="aruco_dict",
        default="DICT_4X4_100",
        help="Nombre del dict ArUco. Default DICT_4X4_100 (el board final)",
    )
    p_cap.add_argument(
        "--legacy-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar enumeración de markers ChArUco pre-4.6. Default True matchea calib.io.",
    )
    p_cap.add_argument(
        "--grid",
        choices=["rectangular", "circular"],
        default="rectangular",
        help="Grid legacy de cobertura (ignorado cuando --guided está activo)",
    )
    p_cap.add_argument(
        "--manual",
        action="store_true",
        help="Trigger manual: apretar Enter en la terminal para capturar (el board tiene que estar detectado). "
        "Usar con un soporte rígido para el board para eliminar motion entre los frames L/R.",
    )
    p_cap.add_argument(
        "--guided",
        action="store_true",
        help="Captura guiada: muestra siluetas ghost de las poses target, "
        "auto-captura cuando está alineado y estable. Default recomendado.",
    )
    p_cap.add_argument(
        "--dist-near-mm",
        type=float,
        default=DEFAULT_DIST_NEAR_MM,
        help=f"Distancia near para la secuencia de poses (default {DEFAULT_DIST_NEAR_MM:.0f}mm)",
    )
    p_cap.add_argument(
        "--dist-mid-mm",
        type=float,
        default=DEFAULT_DIST_MID_MM,
        help=f"Distancia mid para la secuencia de poses (default {DEFAULT_DIST_MID_MM:.0f}mm)",
    )
    p_cap.add_argument(
        "--dist-far-mm",
        type=float,
        default=DEFAULT_DIST_FAR_MM,
        help=f"Distancia far para la secuencia de poses (default {DEFAULT_DIST_FAR_MM:.0f}mm)",
    )
    p_cap.add_argument(
        "--resume",
        action="store_true",
        help="Continuar una sesión guiada previa desde su session.json",
    )
    p_cap.set_defaults(func=cmd_capture)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Corre la calibración estéreo")
    p_cal.add_argument(
        "--input-dir", required=True, help="Directorio con imágenes left_/right_"
    )
    p_cal.add_argument("--output", default="calibration.npz")
    p_cal.add_argument(
        "--board-cols",
        "--columns",
        type=int,
        dest="columns",
        default=HW.board_cols,
        help="Columnas del board. Default desde vision.charuco.board_cols.",
    )
    p_cal.add_argument(
        "--board-rows",
        "--rows",
        type=int,
        dest="rows",
        default=HW.board_rows,
        help="Filas del board. Default desde vision.charuco.board_rows.",
    )
    p_cal.add_argument(
        "--square-mm",
        "--square-length",
        type=float,
        dest="square_length",
        default=HW.square_mm,
        help="Lado del cuadrado en mm. Default desde vision.charuco.square_mm.",
    )
    p_cal.add_argument(
        "--marker-mm",
        "--marker-length",
        type=float,
        dest="marker_length",
        default=HW.marker_mm,
        help="Lado del marker en mm. Default desde vision.charuco.marker_mm.",
    )
    p_cal.add_argument(
        "--dict",
        dest="aruco_dict",
        default="DICT_4X4_100",
        help="Nombre del dict ArUco. Default DICT_4X4_100",
    )
    p_cal.add_argument(
        "--legacy-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar enumeración de markers ChArUco pre-4.6. Default True matchea calib.io.",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    # --- verify ---
    p_ver = sub.add_parser("verify", help="Verifica la calibración visualmente")
    p_ver.add_argument("--calibration", required=True, help="Path al calibration.npz")
    p_ver.add_argument(
        "--input-dir", required=True, help="Directorio con pares de imágenes"
    )
    p_ver.set_defaults(func=cmd_verify)

    # --- wizard (one-shot: captura guiada + calibrar + verificar + reporte) ---
    p_wiz = sub.add_parser(
        "wizard",
        help="Calibración guiada one-shot: captura -> calibrar -> verificar -> reporte",
    )
    p_wiz.add_argument(
        "--device-id",
        default="unknown",
        help="Identificador de dispositivo para el reporte",
    )
    p_wiz.add_argument(
        "--output", default="calibration.npz", help="Archivo de calibración de output"
    )
    p_wiz.add_argument(
        "--output-dir",
        default="./calibration/captures",
        help="Directorio para los pares capturados",
    )
    p_wiz.add_argument("--left", type=int, default=0)
    p_wiz.add_argument("--right", type=int, default=1)
    p_wiz.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        help="Resolución de captura. Default: lee vision.resolution "
        "de /etc/people-counter/config.yaml (fuente única de verdad — "
        "garantiza match con el runtime). Pasá explícito solo para "
        "tests / dev workstation donde no hay config per-device.",
    )
    p_wiz.add_argument("--fps", type=int, default=5)
    p_wiz.add_argument("--port", type=int, default=8080)
    p_wiz.add_argument(
        "--board-cols", "--columns", type=int, dest="columns", default=HW.board_cols
    )
    p_wiz.add_argument(
        "--board-rows", "--rows", type=int, dest="rows", default=HW.board_rows
    )
    p_wiz.add_argument(
        "--square-mm",
        "--square-length",
        type=float,
        dest="square_length",
        default=HW.square_mm,
    )
    p_wiz.add_argument(
        "--marker-mm",
        "--marker-length",
        type=float,
        dest="marker_length",
        default=HW.marker_mm,
    )
    p_wiz.add_argument(
        "--dict",
        dest="aruco_dict",
        default="DICT_4X4_100",
        help="Nombre del dict ArUco. Default DICT_4X4_100 (el board final)",
    )
    p_wiz.add_argument(
        "--legacy-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar enumeración de markers ChArUco pre-4.6. Default True matchea calib.io.",
    )
    p_wiz.add_argument(
        "--min-captures",
        type=int,
        default=15,
        help="Cantidad mínima de pares necesaria para correr "
        "la calibración. Default 15 (estadísticamente "
        "robusto). Bajar solo para dry-runs / debugging.",
    )
    p_wiz.add_argument(
        "--align-min-corners",
        type=int,
        default=None,
        help="Override del corner-gate de captura. Default "
        "None usa los thresholds canónicos (tight=15 "
        "post-bootstrap, loose=12 durante bootstrap). "
        "Valores menores aceptan capturas con menos "
        "esquinas decodificadas — usar SOLO cuando un "
        "setup físico (mount vertical en cuarto chico, "
        "rectificación fisheye agresiva en periferia) "
        "impide alcanzar los thresholds default. El "
        "resultado es una calibración más subdeterminada "
        "matemáticamente, pero preferible a no calibrar.",
    )
    p_wiz.add_argument(
        "--align-loose-px",
        type=float,
        default=None,
        help="Override de la tolerancia del centroid-offset "
        "contra el ghost target (default 12px tight / "
        "25px loose). Valores mayores (ej. 200) "
        "efectivamente desactivan el match-al-ghost — el "
        "wizard captura cuando el board está estable, "
        "donde esté en el frame. Útil cuando los ghosts "
        "fueron generados para una geometría distinta "
        "(setup vertical, FOV asimétrico) y matchear-al-"
        "pixel es físicamente imposible. La captura "
        "queda atribuida a la pose más cercana del ghost "
        "secuencial, pero el solver no se afecta.",
    )
    p_wiz.add_argument(
        "--pose-timeout-seconds",
        "--pose-timeout-sec",
        type=float,
        dest="pose_timeout_sec",
        default=SKIP_POSE_TIMEOUT_SEC,
        help=f"Segundos antes de que una pose no capturada sea "
        f"auto-skippeada (default {SKIP_POSE_TIMEOUT_SEC:.0f}). "
        f"Subir a 180+ cuando el board está montado en "
        f"trípode y necesita tiempo para reposicionarse "
        f"entre poses.",
    )
    p_wiz.add_argument(
        "--tolerance",
        choices=["loose", "normal", "strict"],
        default="normal",
        help="Preset de strictness de captura. "
        "loose = permisivo (PoC / board chico / "
        "iluminación complicada, bootstrap 50px / tight "
        "25px). normal = defaults tuneados para el A3 "
        "final (bootstrap 25px / tight 12px). "
        "strict = production-grade (bootstrap 15px / "
        "tight 8px, minimiza outliers).",
    )
    p_wiz.add_argument(
        "--align-tol-loose-px",
        type=float,
        default=None,
        help="Override fine de la tolerance de alineación de "
        "bootstrap (tiene precedencia sobre --tolerance).",
    )
    p_wiz.add_argument(
        "--align-tol-tight-px",
        type=float,
        default=None,
        help="Override fine de la tolerance de alineación " "post-bootstrap.",
    )
    p_wiz.add_argument(
        "--dist-near-mm",
        type=float,
        default=DEFAULT_DIST_NEAR_MM,
        help=f"Distancia near para la secuencia de poses (default {DEFAULT_DIST_NEAR_MM:.0f}mm)",
    )
    p_wiz.add_argument(
        "--dist-mid-mm",
        type=float,
        default=DEFAULT_DIST_MID_MM,
        help=f"Distancia mid para la secuencia de poses (default {DEFAULT_DIST_MID_MM:.0f}mm)",
    )
    p_wiz.add_argument(
        "--dist-far-mm",
        type=float,
        default=DEFAULT_DIST_FAR_MM,
        help=f"Distancia far para la secuencia de poses (default {DEFAULT_DIST_FAR_MM:.0f}mm)",
    )
    # --- reset ---
    p_reset = sub.add_parser(
        "reset",
        help="Borra captures + session.json + calibration.npz para un restart limpio",
    )
    p_reset.add_argument(
        "--output",
        default="calibration.npz",
        help="Archivo de calibración a borrar (default: calibration.npz)",
    )
    p_reset.add_argument(
        "--output-dir",
        default="./calibration/captures",
        help="Directorio de captures a limpiar (default: ./calibration/captures)",
    )
    p_reset.add_argument(
        "--yes",
        action="store_true",
        help="Confirma el borrado (sin esto el comando lista lo que se borraría)",
    )
    p_reset.set_defaults(func=cmd_reset)

    p_wiz.add_argument(
        "--resume",
        action="store_true",
        help="Continúa una sesión previa del wizard — saltea las poses ya capturadas",
    )
    p_wiz.add_argument(
        "--manual",
        action="store_true",
        help="(solo con --guided) Captura 100%% manual: el operador apreta "
        "'Capturar' en la UI por cada pose (sin auto-captura por estabilidad) "
        "y sin auto-skip por timeout. El barrido (default) ya es manual por "
        "naturaleza — capturás moviendo el board y pausando.",
    )
    p_wiz.add_argument(
        "--guided",
        action="store_true",
        help="Usa el modo guiado por poses-silueta (el DEFAULT es barrido "
        "libre). Más preciso pero requiere más espacio + paciencia: matcheás "
        "~20 siluetas a 1/2/3m. Usar para máxima calidad cuando tenés buen "
        "espacio y luz. El barrido (default) es mucho más fácil de operar en "
        "espacios chicos / luz difícil.",
    )
    p_wiz.add_argument(
        "--sweep-novelty",
        type=float,
        default=SWEEP_NOVELTY_MIN,
        dest="sweep_novelty",
        help=f"(modo barrido) Umbral de novedad para aceptar un frame "
        f"(distancia en el espacio de firma normalizado). Default "
        f"{SWEEP_NOVELTY_MIN}. Subir = menos frames más diversos.",
    )
    p_wiz.add_argument(
        "--baseline-tol-mm",
        type=float,
        default=5.0,
        help="Tolerancia ± en mm para el check de baseline vs "
        "diseño (140mm) en el reporte. Default 5mm "
        "(matchea el QA target de fabricación del "
        "bracket). Subir cuando un device conocido tiene "
        "drift mayor y querés que el reporte salga PASS — "
        "ojo, la rectificación absorbe el drift, no es "
        "un gate de calidad real (el depth check sí).",
    )
    p_wiz.add_argument(
        "--force-degenerate-coverage",
        action="store_true",
        help="Bypassea el block de coverage crítico. Por "
        "default el wizard se rehúsa a calibrar cuando "
        "faltan grupos enteros de poses (A/B/C/D) o "
        "bandas de distancia (near/mid/far), porque eso "
        "produce un fit con RMS bajo pero geométricamente "
        "incorrecto. Usar este flag solo si entendés el "
        "trade-off.",
    )
    p_wiz.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Deshabilita el hint de early-stop. Por default, "
        "cuando las capturas pasan el coverage completo y "
        "el RMS cae bajo el threshold de lab, el wizard "
        "muestra un hint verde sugiriendo finalizar sin "
        "agotar las 20 poses. El hint es puramente "
        "advisory; el botón Finalizar existente es lo que "
        "realmente termina la sesión. Pasar este flag "
        "para suprimir el hint entero (disciplina completa "
        "de 20 poses).",
    )
    p_wiz.add_argument(
        "--min-detect-rate",
        type=float,
        default=0.7,
        help="Threshold de sanity pre-calibración: si menos "
        "que esta fracción de los pares capturados "
        "sobrevive una pasada estricta de re-detección, "
        "el wizard aborta antes de correr "
        "fisheye.calibrate. Default 0.7 (70%%).",
    )
    p_wiz.add_argument(
        "--low-light",
        action="store_true",
        help="Modo PoC para corridas en luz baja / cuartos "
        "chicos. Afloja los gates de calidad de frame "
        "(exposición, blur, corner sharpness, balance de "
        "brightness L/R) así las capturas no son "
        "rechazadas por las condiciones de la escena. La "
        "calibración resultante NO va a ser válida para "
        "depth de producción — usar solo para validar el "
        "wizard end-to-end.",
    )
    p_wiz.add_argument(
        "--meter",
        choices=("matrix", "centre", "spot"),
        default="matrix",
        help="Modo de AE metering. 'matrix' (default) "
        "pondera todo el frame. 'centre'/'spot' "
        "exponen para el centro del frame — usar estos "
        "en luz baja cuando hay periferias brillantes "
        "(ventanas, paredes) que arrastran la exposición "
        "abajo en el board.",
    )
    p_wiz.add_argument(
        "--lock-ae",
        action="store_true",
        help="Lockea exposición/ganancia/white-balance "
        "después de un settle de AE de 1s. Útil cuando "
        "la escena tiene luz variable (luz natural, "
        "puertas/ventanas que dejan fluctuar la luz) — "
        "el lock evita que el AE driftee independiente "
        "entre L/R a mitad de sesión. Default off: el AE "
        "ajusta durante toda la sesión, más simple y la "
        "imagen del reporte ground-truth matchea lo que "
        "ven las cámaras.",
    )
    p_wiz.add_argument(
        "--max-exposure-us",
        type=int,
        default=16000,
        help="Cap de exposure time en microsegundos vía "
        "FrameDurationLimits. Default 16000us (16ms), "
        "mismo que el runtime. Freezea micro-vibración "
        "que rompe el decoder ArUco asimétricamente "
        "entre L/R en luz baja. AE compensa con "
        "AnalogueGain. Pasar 0 para deshabilitar el cap.",
    )
    p_wiz.set_defaults(func=cmd_wizard)

    args = parser.parse_args()
    _resolve_resolution_from_device_config(args, parser)
    args.func(args)


def _resolve_resolution_from_device_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Si --resolution no fue provisto, lo levanta de /etc/people-counter/config.yaml.

    Los setup tools deben ser consistentes con el runtime — leer la misma
    fuente. Si el config per-device no existe o le falta vision.resolution,
    abortamos con error claro en vez de caer a un default hardcoded.
    Solo aplica a subcomandos que aceptan --resolution (capture, wizard).
    """
    if not hasattr(args, "resolution") or args.resolution is not None:
        return
    from src.config.loader import (
        DEFAULT_DEVICE_CONFIG_PATH,
        load_device_config,
    )

    try:
        cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
    except FileNotFoundError:
        parser.error(
            f"--resolution no provisto y {DEFAULT_DEVICE_CONFIG_PATH} no existe. "
            "Pasá --resolution explícito o aprovisioná el config per-device."
        )
    res = cfg.get("vision", {}).get("resolution")
    if not res or not isinstance(res, list) or len(res) != 2:
        parser.error(
            f"--resolution no provisto y vision.resolution falta/está mal en "
            f"{DEFAULT_DEVICE_CONFIG_PATH} — pasá --resolution explícito."
        )
    args.resolution = [int(res[0]), int(res[1])]


if __name__ == "__main__":
    main()

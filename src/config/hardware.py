"""Parámetros de hardware + lab que definen un device.

Estos valores describen el sensor, el lens, el bracket físico, el board
de calibración y los timings AE asociados. Son la base hardware-agnostic
del pipeline: cualquier setup distinto (otro modelo de cámara, otro
bracket con baseline distinto, otro board ChArUco, otra escena con
necesidad de settle AE más largo) se acomoda editando estas keys del
``/etc/people-counter/config.yaml``, sin tocar código.

Los setup tools (focus_assist, calibrate, preview, roi_picker,
diagnose_bracket) leen ``HardwareParams`` al startup; cuando el config
per-device omite una key, cae al valor de ``FLEET_DEFAULTS`` (la flota
canónica que shippeamos: RPi5 + IMX708 B0310 + bracket 140mm + ChArUco
A3 9×6/45mm/33mm/DICT_4X4_100).

Importante: este módulo NO carga el runtime config completo — para eso
está ``src.config.loader.load_config()``. ``load_hardware_params()`` es
lenient (no exige que el config sea válido para el runtime), pensado
para tools que pueden correr antes de que el device esté completamente
aprovisionado.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HardwareParams:
    """Snapshot inmutable de los parámetros físicos + lab de un device."""

    # --- Sensor ---
    full_res: tuple[int, int]
    """Resolución nativa del sensor (W, H)."""

    default_res: tuple[int, int]
    """Sensor mode canónico para foco/calibración/runtime — el ``raw`` size
    pasado a picamera2. En IMX708 = (2304, 1296) = Mode 1 full-FOV binned
    2×2. Sin este override picamera2 elige Mode 0 cropeado.
    """

    nominal_focal_full_px: float
    """Focal pinhole-equivalente en pixels a la resolución ``full_res``,
    región central. Para fisheye Kannala-Brandt es solo el equivalente
    central — la K real se obtiene de la calibración. Se usa para
    bootstraps + tools que necesitan una K antes de tener .npz.
    """

    # --- Lens ---
    hfov_deg: float
    """Field of view horizontal del lens, en grados."""

    lens_type: str
    """Identificador del lens (display only)."""

    # --- Bracket físico ---
    baseline_mm: float
    """Distancia mecánica entre centros ópticos L↔R, en mm. Propiedad del
    bracket fabricado. La calibración mide un valor cercano pero el
    diseño físico es el ground truth."""

    camera_left_csi: int
    """Índice CSI del lens IZQUIERDO mirando desde la cámara hacia la escena."""

    camera_right_csi: int
    """Índice CSI del lens DERECHO."""

    # --- ChArUco board ---
    board_cols: int
    board_rows: int
    square_mm: float
    marker_mm: float
    aruco_dict: str
    legacy_pattern: bool
    """Enumeración de markers ChArUco. True matchea calib.io (pre-OpenCV-4.6)."""

    dual_pass_accept_threshold: int
    """Cuando ``detect_charuco_dual_pass`` detecta < este número de corners
    en el frame original, retry con sharpen 3×3 y se queda con el mejor."""

    # --- AE lock timings ---
    ae_initial_settle_seconds: float
    """Segundos de settle antes del lock provisional (al abrir captura)."""

    ae_resettle_seconds: float
    """Segundos de re-settle entre la habilitación de AE y el re-lock final
    (después de que el operador apreta Comenzar con la escena armada)."""


FLEET_DEFAULTS = HardwareParams(
    # Sensor: IMX708 (Arducam B0310)
    full_res=(4608, 2592),
    default_res=(2304, 1296),
    nominal_focal_full_px=2050.0,
    # Lens: M12 fisheye Arducam B0310
    hfov_deg=120.0,
    lens_type="m12_120deg",
    # Bracket: diseño propio, baseline 140mm
    baseline_mm=140.0,
    camera_left_csi=0,
    camera_right_csi=1,
    # Board: ChArUco A3 calib.io canónico
    board_cols=9,
    board_rows=6,
    square_mm=45.0,
    marker_mm=33.0,
    aruco_dict="DICT_4X4_100",
    legacy_pattern=True,
    dual_pass_accept_threshold=8,
    # AE timings
    ae_initial_settle_seconds=2.0,
    ae_resettle_seconds=1.5,
)


DEFAULT_DEVICE_CONFIG_PATH = "/etc/people-counter/config.yaml"


def _as_tuple_2(v: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    """Coerce list/tuple de 2 ints a tuple[int, int]; fallback si no calza."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return (int(v[0]), int(v[1]))
        except (TypeError, ValueError):
            pass
    return fallback


def load_hardware_params(
    config_path: str | Path = DEFAULT_DEVICE_CONFIG_PATH,
) -> HardwareParams:
    """Carga ``HardwareParams`` del config per-device, fallback a fleet
    defaults para cualquier key ausente.

    No raisea si ``config_path`` no existe — devuelve ``FLEET_DEFAULTS``
    directo. Pensado para tools que pueden correr antes del provisioning
    (factory QC, dev workstation).
    """
    path = Path(config_path)
    if not path.exists():
        return FLEET_DEFAULTS
    try:
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return FLEET_DEFAULTS

    sensor = cfg.get("sensor") or {}
    lens = cfg.get("lens") or {}
    bracket = cfg.get("bracket") or {}
    vision = cfg.get("vision") or {}
    charuco = (vision.get("charuco") or {}) if isinstance(vision, dict) else {}
    ae_lock = (vision.get("ae_lock") or {}) if isinstance(vision, dict) else {}

    return HardwareParams(
        full_res=_as_tuple_2(
            sensor.get("full_res"), FLEET_DEFAULTS.full_res
        ),
        default_res=_as_tuple_2(
            sensor.get("default_res"), FLEET_DEFAULTS.default_res
        ),
        nominal_focal_full_px=float(
            sensor.get("nominal_focal_full_px")
            or FLEET_DEFAULTS.nominal_focal_full_px
        ),
        hfov_deg=float(lens.get("hfov_deg") or FLEET_DEFAULTS.hfov_deg),
        lens_type=str(lens.get("type") or FLEET_DEFAULTS.lens_type),
        baseline_mm=float(
            bracket.get("baseline_mm") or FLEET_DEFAULTS.baseline_mm
        ),
        camera_left_csi=int(
            bracket.get("camera_left_csi", FLEET_DEFAULTS.camera_left_csi)
        ),
        camera_right_csi=int(
            bracket.get("camera_right_csi", FLEET_DEFAULTS.camera_right_csi)
        ),
        board_cols=int(charuco.get("board_cols", FLEET_DEFAULTS.board_cols)),
        board_rows=int(charuco.get("board_rows", FLEET_DEFAULTS.board_rows)),
        square_mm=float(charuco.get("square_mm", FLEET_DEFAULTS.square_mm)),
        marker_mm=float(charuco.get("marker_mm", FLEET_DEFAULTS.marker_mm)),
        aruco_dict=str(charuco.get("dict") or FLEET_DEFAULTS.aruco_dict),
        legacy_pattern=bool(
            charuco.get("legacy_pattern", FLEET_DEFAULTS.legacy_pattern)
        ),
        dual_pass_accept_threshold=int(
            charuco.get(
                "dual_pass_accept_threshold",
                FLEET_DEFAULTS.dual_pass_accept_threshold,
            )
        ),
        ae_initial_settle_seconds=float(
            ae_lock.get(
                "initial_settle_seconds",
                FLEET_DEFAULTS.ae_initial_settle_seconds,
            )
        ),
        ae_resettle_seconds=float(
            ae_lock.get(
                "resettle_seconds", FLEET_DEFAULTS.ae_resettle_seconds
            )
        ),
    )

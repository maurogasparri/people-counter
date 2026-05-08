"""Hardware-design constants loader.

Reads ``config/hardware.yaml`` (shipped with the codebase) into a typed
dict. These values are bracket / sensor invariants — every device built
to spec has the same numbers. Per-site / per-device settings live in
``config.yaml`` (see ``src.config.loader``).

Cached after first read because nothing in this file changes at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default location: <repo-root>/config/hardware.yaml. Tests override via
# load_hardware_config(path=...).
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "hardware.yaml"

_cache: dict[str, Any] | None = None


# Defaults applied when the optional ``best_frame`` block is missing entirely.
# Mirrors the schema documented in ``config/hardware.yaml`` — keep both in sync.
# ``enabled: False`` is the GDPR/LPDP-safe default: zero storage, zero PII,
# zero side effects when the operator hasn't explicitly opted in.
BEST_FRAME_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "output_dir": "/var/lib/people-counter/best_frames",
    "retention_days": 7,
    "buffer_size": 20,
    "jpeg_quality": 85,
    "scoring": {
        "confidence_weight": 0.4,
        "bbox_area_weight": 0.2,
        "centrality_weight": 0.2,
        "sharpness_weight": 0.2,
    },
}


def load_hardware_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the hardware-constants YAML.

    Cached after first successful read. Pass an explicit ``path`` to
    bypass the cache (used by tests).
    """
    global _cache
    if path is None:
        if _cache is not None:
            return _cache
        path = _DEFAULT_PATH

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Hardware config not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _validate(data)
    _normalise_best_frame(data)

    if path == _DEFAULT_PATH:
        _cache = data
    return data


def _normalise_best_frame(data: dict[str, Any]) -> None:
    """Fill in ``best_frame`` defaults + validate when the operator opted in.

    The block is *optional* — a hardware.yaml without it gets the GDPR-safe
    defaults inlined (``enabled: False``, no side effects). When the block is
    present we validate types and ranges so a typo in the YAML doesn't slip
    through to runtime.
    """
    raw = data.get("best_frame")
    if raw is None:
        # Inline a fresh copy so callers can mutate without polluting the
        # module-level default dict.
        data["best_frame"] = {
            **BEST_FRAME_DEFAULTS,
            "scoring": dict(BEST_FRAME_DEFAULTS["scoring"]),
        }
        return

    if not isinstance(raw, dict):
        raise ValueError("hardware.yaml: best_frame must be a mapping")

    enabled = raw.get("enabled", BEST_FRAME_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise ValueError("hardware.yaml: best_frame.enabled must be a bool")

    retention_days = raw.get("retention_days", BEST_FRAME_DEFAULTS["retention_days"])
    if not isinstance(retention_days, int) or retention_days <= 0:
        raise ValueError(
            "hardware.yaml: best_frame.retention_days must be a positive int"
        )

    buffer_size = raw.get("buffer_size", BEST_FRAME_DEFAULTS["buffer_size"])
    if not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ValueError("hardware.yaml: best_frame.buffer_size must be a positive int")

    jpeg_quality = raw.get("jpeg_quality", BEST_FRAME_DEFAULTS["jpeg_quality"])
    if not isinstance(jpeg_quality, int) or not (1 <= jpeg_quality <= 100):
        raise ValueError(
            "hardware.yaml: best_frame.jpeg_quality must be an int in [1, 100]"
        )

    output_dir = raw.get("output_dir", BEST_FRAME_DEFAULTS["output_dir"])
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError(
            "hardware.yaml: best_frame.output_dir must be a non-empty string"
        )

    scoring_default = BEST_FRAME_DEFAULTS["scoring"]
    scoring_in = raw.get("scoring") or {}
    if not isinstance(scoring_in, dict):
        raise ValueError("hardware.yaml: best_frame.scoring must be a mapping")
    scoring: dict[str, float] = {}
    for key in scoring_default:
        val = scoring_in.get(key, scoring_default[key])
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(
                f"hardware.yaml: best_frame.scoring.{key} must be a "
                "non-negative number"
            )
        scoring[key] = float(val)

    # Soft-warn (not raise) when weights drift far from 1.0 — the scoring
    # function still works with arbitrary positive weights, but a wildly
    # off-sum is almost always an operator typo.
    weight_sum = sum(scoring.values())
    if weight_sum > 0 and not (0.9 <= weight_sum <= 1.1):
        logger.warning(
            "best_frame.scoring weights sum to %.3f — expected ~1.0. "
            "Component contributions will be non-uniform; verify YAML is "
            "intentional.",
            weight_sum,
        )

    data["best_frame"] = {
        "enabled": enabled,
        "output_dir": output_dir,
        "retention_days": retention_days,
        "buffer_size": buffer_size,
        "jpeg_quality": jpeg_quality,
        "scoring": scoring,
    }


def _validate(data: dict[str, Any]) -> None:
    required = {
        "bracket": ("baseline_mm", "camera_left_csi", "camera_right_csi"),
        "sensor": (
            "model",
            "full_res",
            "default_res",
            "default_fps",
            "nominal_focal_full_px",
        ),
        "lens": ("type", "hfov_deg"),
        "vision_runtime": ("sgbm", "resolution", "fps", "calibration_file"),
        "detection": (
            "architecture",
            "model_path",
            "confidence_threshold",
            "nms_threshold",
            "cluster_distance_px",
        ),
        "tracking": ("max_disappeared", "max_distance", "state_machine"),
        "wifi_ble": (
            "wifi_interface",
            "probe_interval_seconds",
            "cross_protocol_window_seconds",
            "cross_protocol_rssi_delta",
        ),
        "mqtt": ("port", "topics"),
        "buffer": ("db_path", "max_age_hours"),
        "logging": ("format", "file"),
    }
    for section, keys in required.items():
        if section not in data:
            raise ValueError(f"hardware.yaml missing section: {section}")
        for key in keys:
            if key not in data[section]:
                raise ValueError(f"hardware.yaml missing key: {section}.{key}")

    bracket = data["bracket"]
    if not isinstance(bracket["camera_left_csi"], int):
        raise ValueError("bracket.camera_left_csi must be int")
    if not isinstance(bracket["camera_right_csi"], int):
        raise ValueError("bracket.camera_right_csi must be int")
    if bracket["camera_left_csi"] == bracket["camera_right_csi"]:
        raise ValueError("bracket.camera_left_csi and camera_right_csi must differ")

    # Nested sub-sections.
    sgbm = data["vision_runtime"]["sgbm"]
    for k in ("num_disparities", "block_size", "downscale"):
        if k not in sgbm:
            raise ValueError(f"hardware.yaml missing key: vision_runtime.sgbm.{k}")

    sm = data["tracking"]["state_machine"]
    for k in ("confirm_frames", "pending_max_frames", "reid_gate_px", "depth_gate_m"):
        if k not in sm:
            raise ValueError(f"hardware.yaml missing key: tracking.state_machine.{k}")
    # Kalman block is OPTIONAL — tracker has sane defaults if absent.
    # When present, every leaf must be a number; missing leaves still
    # fall back to the tracker default rather than crashing.
    kalman = sm.get("kalman")
    if kalman is not None:
        if not isinstance(kalman, dict):
            raise ValueError(
                "hardware.yaml tracking.state_machine.kalman must be a mapping"
            )
        for k in ("process_noise", "measurement_noise", "initial_velocity_uncertainty"):
            if k in kalman and not isinstance(kalman[k], (int, float)):
                raise ValueError(
                    "hardware.yaml tracking.state_machine.kalman."
                    f"{k} must be a number"
                )

    topics = data["mqtt"]["topics"]
    for k in ("counting", "wifi_ble", "telemetry", "shadow"):
        if k not in topics:
            raise ValueError(f"hardware.yaml missing key: mqtt.topics.{k}")

    # Optional two-stage matching threshold. Must be:
    #   - missing or null  → feature disabled
    #   - 0 < value < confidence_threshold → feature enabled, two-stage
    #     matching pipes detections in [low, high) only into re-association
    # Values >= confidence_threshold are silently coerced to "off" + warned
    # so a misconfiguration never silently *expands* the spawn pool.
    detection = data["detection"]
    low = detection.get("low_confidence_threshold")
    if low is not None:
        if not isinstance(low, (int, float)) or isinstance(low, bool):
            raise ValueError(
                "hardware.yaml: detection.low_confidence_threshold "
                "must be null or a number"
            )
        low_f = float(low)
        if low_f <= 0.0:
            raise ValueError(
                "hardware.yaml: detection.low_confidence_threshold "
                "must be > 0 (or null to disable)"
            )
        high_f = float(detection["confidence_threshold"])
        if low_f >= high_f:
            logger.warning(
                "detection.low_confidence_threshold (%.3f) >= "
                "confidence_threshold (%.3f) — disabling two-stage "
                "matching (treated as null)",
                low_f,
                high_f,
            )
            detection["low_confidence_threshold"] = None


def reset_cache() -> None:
    """Drop the cached hardware config. Used by tests."""
    global _cache
    _cache = None

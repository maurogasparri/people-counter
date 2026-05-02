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

    with open(p) as f:
        data = yaml.safe_load(f) or {}

    _validate(data)

    if path == _DEFAULT_PATH:
        _cache = data
    return data


def _validate(data: dict[str, Any]) -> None:
    required = {
        "bracket": ("baseline_mm", "camera_left_csi", "camera_right_csi"),
        "sensor": ("model", "full_res", "default_res", "default_fps",
                   "nominal_focal_full_px"),
        "lens": ("type", "hfov_deg"),
    }
    for section, keys in required.items():
        if section not in data:
            raise ValueError(f"hardware.yaml missing section: {section}")
        for key in keys:
            if key not in data[section]:
                raise ValueError(
                    f"hardware.yaml missing key: {section}.{key}"
                )

    bracket = data["bracket"]
    if not isinstance(bracket["camera_left_csi"], int):
        raise ValueError("bracket.camera_left_csi must be int")
    if not isinstance(bracket["camera_right_csi"], int):
        raise ValueError("bracket.camera_right_csi must be int")
    if bracket["camera_left_csi"] == bracket["camera_right_csi"]:
        raise ValueError(
            "bracket.camera_left_csi and camera_right_csi must differ"
        )


def reset_cache() -> None:
    """Drop the cached hardware config. Used by tests."""
    global _cache
    _cache = None

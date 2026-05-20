"""Fixtures compartidas para tests del loader del config.

Después del refactor strict (sin merge con ``config.example.yaml`` en
runtime), los tests tienen que pasarle a ``load_config`` un YAML que
contenga TODAS las keys requeridas. Antes esto se lograba pasando una
porción minimal y dejando que el deep_merge la completara con los
defaults bundled — ese path ya no existe.

Las fixtures acá producen configs completos: ``complete_config`` devuelve
un dict in-memory; ``complete_config_yaml`` lo escribe a tmp_path y
devuelve el string del path. Los tests overridean solo lo que les
importa y siguen el patrón ``cfg = complete_config | {...}``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

# Config completo canónico — base para tests del loader. Mantenelo en sync
# con la lista de secciones requeridas en ``_validate`` (loader.py).
_COMPLETE_CONFIG: dict = {
    "device": {"id": "test-device", "store_id": "test-store"},
    "bracket": {
        "baseline_mm": 140,
        "camera_left_csi": 0,
        "camera_right_csi": 1,
    },
    "sensor": {
        "model": "imx708",
        "full_res": [4608, 2592],
        "default_res": [2304, 1296],
        "default_fps": 15,
        "nominal_focal_full_px": 2050,
    },
    "lens": {"type": "m12_120deg", "hfov_deg": 120},
    "vision": {
        "mounting_height_m": 3.0,
        "resolution": [1152, 648],
        "fps": 30,
        "calibration_file": "/etc/people-counter/calibration.npz",
        "sgbm": {"num_disparities": "auto", "block_size": 9, "downscale": 4},
    },
    "detection": {
        "architecture": "yolov8",
        "model_path": "/usr/src/people-counter/models/people-counter-detector.hef",
        "confidence_threshold": 0.30,
        "nms_threshold": 0.45,
        "cluster_distance_px": 200,
    },
    "tracking": {
        "max_disappeared_frames": 30,
        "max_distance_px": 50,
        "state_machine": {
            "confirm_frames": 1,
            "pending_max_frames": 20,
            "reid_gate_px": 300,
            "depth_gate_m": 0.5,
        },
    },
    "counter": {},
    "wifi_ble": {
        "enabled": True,
        "wifi_interface": "wlan0",
        "probe_interval_seconds": 900,
        "cross_protocol_window_seconds": 2,
        "cross_protocol_rssi_delta": 5,
        "rssi_passerby_threshold": -75,
        "rssi_shopper_threshold": -55,
    },
    "mqtt": {
        "endpoint": "test.iot.amazonaws.com",
        "port": 8883,
        "cert_path": "/x/cert.pem",
        "key_path": "/x/key.pem",
        "ca_path": "/x/ca.pem",
        "topics": {
            "counting": "store/{store_id}/counting",
            "wifi_ble": "store/{store_id}/wifi_ble",
            "telemetry": "store/{store_id}/telemetry",
            "shadow": "$aws/things/{device_id}/shadow",
        },
    },
    "buffer": {
        "db_path": "/var/lib/people-counter/buffer.db",
        "max_age_hours": 72,
    },
    "logging": {"format": "json", "file": "/var/log/people-counter/app.log"},
    "cloud_defaults": {
        # Cada día explícito así no hay leak de "abierto" donde el test
        # espera "cerrado".
        "operating_hours": {
            "monday": "10:00-22:00",
            "tuesday": "10:00-22:00",
            "wednesday": None,
            "thursday": None,
            "friday": None,
            "saturday": None,
            "sunday": "11:00-20:00",
        },
        "on_invalid_schedule": "fail_open",
        "footfall_scaling_factor": 1.0,
        "counting_enabled": True,
        "wifi_ble_enabled": True,
    },
}


@pytest.fixture
def complete_config() -> dict:
    """Devuelve una deep-copy del config completo canónico (in-memory).

    Los tests overridean lo que necesitan con ``cfg = complete_config; ...``.
    """
    return copy.deepcopy(_COMPLETE_CONFIG)


@pytest.fixture
def complete_config_yaml(tmp_path, complete_config) -> str:
    """Escribe el config completo a tmp_path y devuelve el path string."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(complete_config), encoding="utf-8")
    return str(p)


def write_config(tmp_path: Path, data: dict) -> str:
    """Helper plain (no fixture) — escribe ``data`` a tmp_path/config.yaml."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)

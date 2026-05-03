"""Tests for the hardware-config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.hardware import (
    load_hardware_config,
    reset_cache,
)


VALID_HW = {
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
    "lens": {
        "type": "m12_120deg",
        "hfov_deg": 120,
    },
    "vision_runtime": {
        "sgbm": {"num_disparities": "auto", "block_size": 9, "downscale": 2},
    },
    "detection": {"confidence_threshold": 0.5, "nms_threshold": 0.45},
    "tracking": {
        "max_disappeared": 30,
        "max_distance": 50,
        "state_machine": {
            "confirm_frames": 3,
            "pending_max_frames": 5,
            "reid_gate_px": 60,
            "depth_gate_m": 0.5,
        },
    },
    "wifi_ble": {
        "wifi_interface": "wlan0",
        "probe_interval_seconds": 900,
        "cross_protocol_window_seconds": 2,
        "cross_protocol_rssi_delta": 5,
    },
    "mqtt": {
        "port": 8883,
        "topics": {
            "counting": "store/{store_id}/counting",
            "wifi_ble": "store/{store_id}/wifi_ble",
            "telemetry": "store/{store_id}/telemetry",
            "shadow": "$aws/things/{device_id}/shadow",
        },
    },
    "buffer": {"db_path": "/var/lib/people-counter/buffer.db",
               "max_age_hours": 72},
    "logging": {"format": "json", "file": "/var/log/people-counter/app.log"},
}


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "hardware.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


class TestLoadHardware:
    def test_load_valid(self, tmp_path):
        p = _write_yaml(tmp_path, VALID_HW)
        cfg = load_hardware_config(p)
        assert cfg["bracket"]["baseline_mm"] == 140
        assert cfg["bracket"]["camera_left_csi"] == 0
        assert cfg["bracket"]["camera_right_csi"] == 1
        assert cfg["sensor"]["model"] == "imx708"

    def test_load_default_path(self):
        cfg = load_hardware_config()
        assert "bracket" in cfg
        assert "sensor" in cfg
        assert "lens" in cfg

    def test_default_path_is_cached(self):
        cfg1 = load_hardware_config()
        cfg2 = load_hardware_config()
        assert cfg1 is cfg2

    def test_explicit_path_bypasses_cache(self, tmp_path):
        custom = dict(VALID_HW)
        custom["bracket"] = dict(VALID_HW["bracket"])
        custom["bracket"]["baseline_mm"] = 200
        p = _write_yaml(tmp_path, custom)
        cfg = load_hardware_config(p)
        assert cfg["bracket"]["baseline_mm"] == 200

    def test_missing_file_raises(self, tmp_path):
        bogus = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError):
            load_hardware_config(bogus)

    def test_missing_section_raises(self, tmp_path):
        bad = {"bracket": VALID_HW["bracket"], "sensor": VALID_HW["sensor"]}
        # missing 'lens'
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="lens"):
            load_hardware_config(p)

    def test_missing_key_raises(self, tmp_path):
        bad = {
            "bracket": {"baseline_mm": 140, "camera_left_csi": 0},
            "sensor": VALID_HW["sensor"],
            "lens": VALID_HW["lens"],
        }
        # missing bracket.camera_right_csi
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="camera_right_csi"):
            load_hardware_config(p)

    def test_csi_indices_must_differ(self, tmp_path):
        bad = dict(VALID_HW)
        bad["bracket"] = {
            "baseline_mm": 140,
            "camera_left_csi": 0,
            "camera_right_csi": 0,  # Same as left — invalid
        }
        p = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="must differ"):
            load_hardware_config(p)


class TestShippedHardware:
    """Sanity check on the actual config/hardware.yaml in the repo."""

    def test_shipped_file_loads(self):
        cfg = load_hardware_config()
        assert cfg["bracket"]["baseline_mm"] == 140
        assert cfg["bracket"]["camera_left_csi"] == 0
        assert cfg["bracket"]["camera_right_csi"] == 1
        assert cfg["sensor"]["model"] == "imx708"
        assert tuple(cfg["sensor"]["default_res"]) == (2304, 1296)
        assert cfg["lens"]["type"] == "m12_120deg"

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
        "resolution": [1152, 648],
        "fps": 30,
        "calibration_file": "/etc/people-counter/calibration.npz",
        "sgbm": {"num_disparities": "auto", "block_size": 9, "downscale": 2},
    },
    "detection": {
        "architecture": "yolov8",
        "model_path": "/usr/src/people-counter/models/yolov8n.hef",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.45,
        "cluster_distance_px": 50,
    },
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
    "buffer": {"db_path": "/var/lib/people-counter/buffer.db", "max_age_hours": 72},
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

    def test_shipped_low_confidence_threshold_active(self):
        """Two-stage matching ships activated at the FFC reference value
        (0.10). Strict < confidence_threshold so the validator accepts it."""
        cfg = load_hardware_config()
        low = cfg["detection"].get("low_confidence_threshold")
        high = cfg["detection"]["confidence_threshold"]
        assert low is not None, "low_confidence_threshold must ship activated"
        assert 0.0 < low < high, (
            f"low_confidence_threshold {low} must be in (0, {high})"
        )

    def test_shipped_new_track_threshold_active(self):
        """Spawn floor ships activated above confidence_threshold so weak
        detections re-associate but never spawn ghost tracks."""
        cfg = load_hardware_config()
        nt = cfg["detection"].get("new_track_threshold")
        high = cfg["detection"]["confidence_threshold"]
        assert nt is not None, "new_track_threshold must ship activated"
        assert nt >= high, (
            f"new_track_threshold {nt} must be >= confidence_threshold {high}"
        )

    def test_shipped_best_frame_default_off(self):
        """Hardware shipped to the fleet must have best_frame OFF —
        this is the GDPR/LPDP-safe baseline. A repo PR that flips the
        default to True without explicit DPIA should fail this test."""
        cfg = load_hardware_config()
        bf = cfg.get("best_frame", {})
        assert bf.get("enabled") is False, (
            "best_frame.enabled must default to False in shipped config — "
            "see docs/privacy.md"
        )


class TestLowConfidenceThreshold:
    def test_explicit_null_is_accepted(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["low_confidence_threshold"] = None
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"]["low_confidence_threshold"] is None

    def test_missing_is_accepted(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"].pop("low_confidence_threshold", None)
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        # Missing -> dict get returns None.
        assert loaded["detection"].get("low_confidence_threshold") is None

    def test_valid_low_threshold_accepted(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["confidence_threshold"] = 0.30
        cfg["detection"]["low_confidence_threshold"] = 0.10
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"]["low_confidence_threshold"] == 0.10

    def test_zero_or_negative_rejected(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["low_confidence_threshold"] = 0.0
        p = _write_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match="must be > 0"):
            load_hardware_config(p)

    def test_non_numeric_rejected(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["low_confidence_threshold"] = "low"
        p = _write_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match="must be null or a number"):
            load_hardware_config(p)

    def test_above_high_threshold_disables_with_warning(self, tmp_path, caplog):
        """If low >= high, the validator coerces to None (feature off) and
        logs a warning — never silently expands the spawn pool.
        """
        import logging

        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["confidence_threshold"] = 0.30
        cfg["detection"]["low_confidence_threshold"] = 0.50  # >= high
        p = _write_yaml(tmp_path, cfg)
        with caplog.at_level(logging.WARNING):
            loaded = load_hardware_config(p)
        assert loaded["detection"]["low_confidence_threshold"] is None
        assert any(
            "low_confidence_threshold" in rec.message and "disabling" in rec.message
            for rec in caplog.records
        )

    def test_equal_to_high_threshold_disables(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["confidence_threshold"] = 0.30
        cfg["detection"]["low_confidence_threshold"] = 0.30
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"]["low_confidence_threshold"] is None


class TestNewTrackThreshold:
    """Optional spawn floor: feature off when null, validates >= conf when set."""

    def test_explicit_null_is_accepted(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["new_track_threshold"] = None
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"]["new_track_threshold"] is None

    def test_missing_is_accepted(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"].pop("new_track_threshold", None)
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"].get("new_track_threshold") is None

    def test_valid_above_high_threshold_accepted(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["confidence_threshold"] = 0.30
        cfg["detection"]["new_track_threshold"] = 0.50
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"]["new_track_threshold"] == 0.50

    def test_zero_or_negative_rejected(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["new_track_threshold"] = 0.0
        p = _write_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match="must be in"):
            load_hardware_config(p)

    def test_above_one_rejected(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["new_track_threshold"] = 1.5
        p = _write_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match="must be in"):
            load_hardware_config(p)

    def test_non_numeric_rejected(self, tmp_path):
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["new_track_threshold"] = "high"
        p = _write_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match="must be null or a number"):
            load_hardware_config(p)

    def test_below_high_threshold_disables_with_warning(self, tmp_path, caplog):
        """If new_track < confidence_threshold, the validator coerces to
        None (feature off) and logs a warning — never silently lowers
        the spawn floor below confidence_threshold."""
        import logging

        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["confidence_threshold"] = 0.30
        cfg["detection"]["new_track_threshold"] = 0.20
        p = _write_yaml(tmp_path, cfg)
        with caplog.at_level(logging.WARNING):
            loaded = load_hardware_config(p)
        assert loaded["detection"]["new_track_threshold"] is None
        assert any(
            "new_track_threshold" in rec.message and "disabling" in rec.message
            for rec in caplog.records
        )

    def test_equal_to_high_threshold_accepted(self, tmp_path):
        """When new_track_threshold == confidence_threshold, spawn floor
        is unchanged (no-op feature). Accepted, not coerced."""
        cfg = dict(VALID_HW)
        cfg["detection"] = dict(VALID_HW["detection"])
        cfg["detection"]["confidence_threshold"] = 0.30
        cfg["detection"]["new_track_threshold"] = 0.30
        p = _write_yaml(tmp_path, cfg)
        loaded = load_hardware_config(p)
        assert loaded["detection"]["new_track_threshold"] == 0.30


class TestBestFrameValidation:
    """Optional best_frame block: defaults when missing, validation when set."""

    def test_best_frame_block_missing_uses_defaults(self, tmp_path):
        # VALID_HW has no best_frame key; the loader inlines defaults.
        p = _write_yaml(tmp_path, VALID_HW)
        cfg = load_hardware_config(p)
        bf = cfg["best_frame"]
        assert bf["enabled"] is False
        assert bf["retention_days"] == 7
        assert bf["buffer_size"] == 20
        assert bf["jpeg_quality"] == 85
        assert "scoring" in bf
        # scoring weights default to ~1.0 sum
        assert sum(bf["scoring"].values()) == pytest.approx(1.0)

    def test_best_frame_partial_block_fills_defaults(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = {"enabled": True, "retention_days": 14}
        p = _write_yaml(tmp_path, data)
        cfg = load_hardware_config(p)
        bf = cfg["best_frame"]
        assert bf["enabled"] is True
        assert bf["retention_days"] == 14
        # Other fields fall back to defaults.
        assert bf["buffer_size"] == 20
        assert bf["jpeg_quality"] == 85

    def test_best_frame_enabled_must_be_bool(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = {"enabled": "yes"}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="enabled"):
            load_hardware_config(p)

    def test_best_frame_retention_must_be_positive(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = {"retention_days": 0}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="retention_days"):
            load_hardware_config(p)

    def test_best_frame_buffer_size_must_be_positive(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = {"buffer_size": -1}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="buffer_size"):
            load_hardware_config(p)

    def test_best_frame_jpeg_quality_range(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = {"jpeg_quality": 150}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="jpeg_quality"):
            load_hardware_config(p)

    def test_best_frame_scoring_negative_weight_raises(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = {
            "scoring": {"confidence_weight": -0.5},
        }
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="confidence_weight"):
            load_hardware_config(p)

    def test_best_frame_must_be_mapping(self, tmp_path):
        data = dict(VALID_HW)
        data["best_frame"] = "not a dict"
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="best_frame must be a mapping"):
            load_hardware_config(p)

    def test_best_frame_weights_off_by_factor_warns(self, tmp_path, caplog):
        """Weights summing far from 1.0 produce a WARN log — soft guard
        for operator typos, not a hard failure."""
        import logging

        data = dict(VALID_HW)
        data["best_frame"] = {
            "scoring": {
                "confidence_weight": 0.5,
                "bbox_area_weight": 0.5,
                "centrality_weight": 0.5,
                "sharpness_weight": 0.5,
            },
        }
        p = _write_yaml(tmp_path, data)
        with caplog.at_level(logging.WARNING):
            load_hardware_config(p)
        msgs = [r.message for r in caplog.records]
        assert any("weights sum" in m for m in msgs)

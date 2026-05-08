"""Tests para los validators del config unificado.

Estos ejercen los validators dentro de ``src/config/loader.py``: low/
new_track threshold gates, best_frame normalisation, counter toggle,
required-section presence, etc.

Each test builds a complete valid config dict, optionally mutates one
field, writes it to a YAML, and loads it WITHOUT the bundled defaults
(``defaults_path=empty``) so the assertion targets the validator
directly, not the defaults fill-in path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config, load_defaults


VALID_CONFIG: dict = {
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
        "architecture": "rapid",
        "model_path": "/usr/src/people-counter/models/rapid_mwhb1024.hef",
        "confidence_threshold": 0.30,
        "nms_threshold": 0.45,
        "cluster_distance_px": 200,
    },
    "tracking": {
        "max_disappeared": 30,
        "max_distance": 50,
        "state_machine": {
            "confirm_frames": 1,
            "pending_max_frames": 20,
            "reid_gate_px": 300,
            "depth_gate_m": 0.5,
        },
    },
    "counter": {"foot_projection_enabled": False},
    "wifi_ble": {
        "wifi_interface": "wlan0",
        "probe_interval_seconds": 900,
        "cross_protocol_window_seconds": 2,
        "cross_protocol_rssi_delta": 5,
    },
    "mqtt": {
        "endpoint": "test.iot.amazonaws.com",
        "port": 8883,
        "cert_path": "/x",
        "key_path": "/x",
        "ca_path": "/x",
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


@pytest.fixture
def empty_defaults(tmp_path):
    p = tmp_path / "empty_defaults.yaml"
    p.write_text("", encoding="utf-8")
    return p


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _load(tmp_path: Path, data: dict, empty_defaults: Path) -> dict:
    return load_config(_write_yaml(tmp_path, data), defaults_path=empty_defaults)


# ---------------------------------------------------------------------------
# Shipped config.example.yaml — sanity checks on the canonical defaults
# ---------------------------------------------------------------------------


class TestShippedDefaults:
    """The bundled defaults file must load and validate as a complete config.

    The shipped ``config.example.yaml`` has every required key so that
    when the per-device YAML overrides only a subset, the merged result
    is always complete. These tests guard against PRs that delete a key
    from the example without realising it's load-bearing.
    """

    def test_defaults_load_clean(self):
        defaults = load_defaults()
        assert isinstance(defaults, dict)
        assert "bracket" in defaults
        assert "vision" in defaults

    def test_shipped_baseline_is_design(self):
        defaults = load_defaults()
        assert defaults["bracket"]["baseline_mm"] == 140

    def test_shipped_low_confidence_threshold_active(self):
        """Two-stage matching ships activated at the FFC reference value
        (0.10). Strict < confidence_threshold so the validator accepts it."""
        defaults = load_defaults()
        low = defaults["detection"].get("low_confidence_threshold")
        high = defaults["detection"]["confidence_threshold"]
        assert low is not None, "low_confidence_threshold must ship activated"
        assert 0.0 < low < high

    def test_shipped_new_track_threshold_active(self):
        defaults = load_defaults()
        nt = defaults["detection"].get("new_track_threshold")
        high = defaults["detection"]["confidence_threshold"]
        assert nt is not None, "new_track_threshold must ship activated"
        assert nt >= high

    def test_shipped_best_frame_default_off(self):
        """GDPR/LPDP-safe baseline: best_frame OFF unless DPIA is in place."""
        defaults = load_defaults()
        bf = defaults.get("best_frame", {})
        assert bf.get("enabled") is False

    def test_shipped_foot_projection_default_off(self):
        """Default off — el feature requiere depth válido (calibración
        que pase diagnose_depth.py con error <5% al centro a 2m). Con
        depth incorrecto el head_height_mm queda mal y la proyección
        comprime la trayectoria del foot-point dentro del ROI hasta
        que los exits no disparan. Activar per-site recién después
        de validar la calibración."""
        defaults = load_defaults()
        assert defaults["counter"]["foot_projection_enabled"] is False


# ---------------------------------------------------------------------------
# low_confidence_threshold (two-stage matching)
# ---------------------------------------------------------------------------


class TestLowConfidenceThreshold:
    def test_explicit_null_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "low_confidence_threshold": None}
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["low_confidence_threshold"] is None

    def test_missing_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {k: v for k, v in VALID_CONFIG["detection"].items()
                             if k != "low_confidence_threshold"}
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"].get("low_confidence_threshold") is None

    def test_valid_below_high_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {
            **VALID_CONFIG["detection"],
            "confidence_threshold": 0.30,
            "low_confidence_threshold": 0.10,
        }
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["low_confidence_threshold"] == 0.10

    def test_zero_or_negative_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "low_confidence_threshold": 0.0}
        with pytest.raises(ValueError, match="must be > 0"):
            _load(tmp_path, cfg, empty_defaults)

    def test_non_numeric_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "low_confidence_threshold": "low"}
        with pytest.raises(ValueError, match="must be null or a number"):
            _load(tmp_path, cfg, empty_defaults)

    def test_above_high_disables_with_warning(self, tmp_path, empty_defaults, caplog):
        """If low >= confidence_threshold, the validator coerces to None
        (feature off) and logs a warning — never silently broadens spawn."""
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {
            **VALID_CONFIG["detection"],
            "confidence_threshold": 0.30,
            "low_confidence_threshold": 0.40,
        }
        with caplog.at_level(logging.WARNING):
            loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["low_confidence_threshold"] is None
        assert any(
            "low_confidence_threshold" in rec.message and "disabling" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# new_track_threshold (spawn floor)
# ---------------------------------------------------------------------------


class TestNewTrackThreshold:
    def test_explicit_null_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "new_track_threshold": None}
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["new_track_threshold"] is None

    def test_missing_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {k: v for k, v in VALID_CONFIG["detection"].items()
                             if k != "new_track_threshold"}
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"].get("new_track_threshold") is None

    def test_valid_above_high_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {
            **VALID_CONFIG["detection"],
            "confidence_threshold": 0.30,
            "new_track_threshold": 0.50,
        }
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["new_track_threshold"] == 0.50

    def test_zero_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "new_track_threshold": 0.0}
        with pytest.raises(ValueError, match="must be in"):
            _load(tmp_path, cfg, empty_defaults)

    def test_above_one_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "new_track_threshold": 1.5}
        with pytest.raises(ValueError, match="must be in"):
            _load(tmp_path, cfg, empty_defaults)

    def test_non_numeric_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {**VALID_CONFIG["detection"], "new_track_threshold": "high"}
        with pytest.raises(ValueError, match="must be null or a number"):
            _load(tmp_path, cfg, empty_defaults)

    def test_below_high_disables_with_warning(self, tmp_path, empty_defaults, caplog):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {
            **VALID_CONFIG["detection"],
            "confidence_threshold": 0.30,
            "new_track_threshold": 0.20,
        }
        with caplog.at_level(logging.WARNING):
            loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["new_track_threshold"] is None
        assert any(
            "new_track_threshold" in rec.message and "disabling" in rec.message
            for rec in caplog.records
        )

    def test_equal_to_high_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["detection"] = {
            **VALID_CONFIG["detection"],
            "confidence_threshold": 0.30,
            "new_track_threshold": 0.30,
        }
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["detection"]["new_track_threshold"] == 0.30


# ---------------------------------------------------------------------------
# best_frame block (optional, GDPR-safe defaults when missing)
# ---------------------------------------------------------------------------


class TestBestFrameValidation:
    def test_block_missing_uses_defaults(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg.pop("best_frame", None)
        loaded = _load(tmp_path, cfg, empty_defaults)
        bf = loaded["best_frame"]
        assert bf["enabled"] is False
        assert bf["retention_days"] == 7
        assert bf["buffer_size"] == 20
        assert bf["jpeg_quality"] == 85
        assert "scoring" in bf
        assert sum(bf["scoring"].values()) == pytest.approx(1.0)

    def test_partial_block_fills_defaults(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {"enabled": True, "retention_days": 14}
        loaded = _load(tmp_path, cfg, empty_defaults)
        bf = loaded["best_frame"]
        assert bf["enabled"] is True
        assert bf["retention_days"] == 14
        assert bf["buffer_size"] == 20
        assert bf["jpeg_quality"] == 85

    def test_enabled_must_be_bool(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {"enabled": "yes"}
        with pytest.raises(ValueError, match="enabled"):
            _load(tmp_path, cfg, empty_defaults)

    def test_retention_must_be_positive(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {"retention_days": 0}
        with pytest.raises(ValueError, match="retention_days"):
            _load(tmp_path, cfg, empty_defaults)

    def test_buffer_size_must_be_positive(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {"buffer_size": -1}
        with pytest.raises(ValueError, match="buffer_size"):
            _load(tmp_path, cfg, empty_defaults)

    def test_jpeg_quality_range(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {"jpeg_quality": 150}
        with pytest.raises(ValueError, match="jpeg_quality"):
            _load(tmp_path, cfg, empty_defaults)

    def test_scoring_negative_weight_raises(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {"scoring": {"confidence_weight": -0.5}}
        with pytest.raises(ValueError, match="confidence_weight"):
            _load(tmp_path, cfg, empty_defaults)

    def test_must_be_mapping(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = "not a dict"
        with pytest.raises(ValueError, match="best_frame must be a mapping"):
            _load(tmp_path, cfg, empty_defaults)

    def test_weights_off_by_factor_warns(self, tmp_path, empty_defaults, caplog):
        cfg = dict(VALID_CONFIG)
        cfg["best_frame"] = {
            "scoring": {
                "confidence_weight": 0.5,
                "bbox_area_weight": 0.5,
                "centrality_weight": 0.5,
                "sharpness_weight": 0.5,
            },
        }
        with caplog.at_level(logging.WARNING):
            _load(tmp_path, cfg, empty_defaults)
        msgs = [r.message for r in caplog.records]
        assert any("weights sum" in m for m in msgs)


# ---------------------------------------------------------------------------
# counter.foot_projection_enabled toggle
# ---------------------------------------------------------------------------


class TestCounterToggle:
    def test_true_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["counter"] = {"foot_projection_enabled": True}
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["counter"]["foot_projection_enabled"] is True

    def test_false_accepted(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["counter"] = {"foot_projection_enabled": False}
        loaded = _load(tmp_path, cfg, empty_defaults)
        assert loaded["counter"]["foot_projection_enabled"] is False

    def test_missing_block_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg.pop("counter", None)
        with pytest.raises(ValueError, match="missing section: counter"):
            _load(tmp_path, cfg, empty_defaults)

    def test_missing_key_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["counter"] = {}
        with pytest.raises(ValueError, match="counter.foot_projection_enabled"):
            _load(tmp_path, cfg, empty_defaults)

    def test_non_bool_rejected(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["counter"] = {"foot_projection_enabled": "yes"}
        with pytest.raises(ValueError, match="must be a bool"):
            _load(tmp_path, cfg, empty_defaults)


# ---------------------------------------------------------------------------
# Hardware section presence (bracket / sensor / lens)
# ---------------------------------------------------------------------------


class TestHardwareSections:
    def test_missing_bracket_section_raises(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg.pop("bracket", None)
        with pytest.raises(ValueError, match="missing section: bracket"):
            _load(tmp_path, cfg, empty_defaults)

    def test_missing_camera_csi_raises(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["bracket"] = {"baseline_mm": 140, "camera_left_csi": 0}
        with pytest.raises(ValueError, match="camera_right_csi"):
            _load(tmp_path, cfg, empty_defaults)

    def test_camera_csi_must_differ(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["bracket"] = {
            "baseline_mm": 140,
            "camera_left_csi": 0,
            "camera_right_csi": 0,  # same as left
        }
        with pytest.raises(ValueError, match="must differ"):
            _load(tmp_path, cfg, empty_defaults)

    def test_kalman_block_optional(self, tmp_path, empty_defaults):
        """Kalman block is optional — tracker has sane defaults if absent."""
        cfg = dict(VALID_CONFIG)
        # Default VALID_CONFIG has no kalman; loading should succeed.
        loaded = _load(tmp_path, cfg, empty_defaults)
        sm = loaded["tracking"]["state_machine"]
        assert sm.get("kalman") is None or isinstance(sm.get("kalman"), dict)

    def test_kalman_block_when_present_validated(self, tmp_path, empty_defaults):
        cfg = dict(VALID_CONFIG)
        cfg["tracking"] = {
            **VALID_CONFIG["tracking"],
            "state_machine": {
                **VALID_CONFIG["tracking"]["state_machine"],
                "kalman": {"process_noise": "not_a_number"},
            },
        }
        with pytest.raises(ValueError, match="kalman"):
            _load(tmp_path, cfg, empty_defaults)

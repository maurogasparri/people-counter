"""Tests para los validators del config strict.

Estos ejercen los validators dentro de ``src/config/loader.py``: low/
new_track threshold gates, best_frame normalisation, counter toggle,
required-section presence, etc.

Después del refactor strict (load_config no mergea con config.example.yaml),
cada test arma un config completo via la fixture ``complete_config`` del
conftest, opcionalmente muta un campo, lo escribe a YAML y lo carga.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config

from .conftest import write_config


# ---------------------------------------------------------------------------
# Shipped config.example.yaml — sanity checks on the canonical template
# ---------------------------------------------------------------------------


class TestShippedTemplate:
    """El template bundled ``config.example.yaml`` tiene que cargar y validar
    como un config completo.

    Aunque ya no se mergea en runtime, sigue siendo la base de aprovisionamiento
    para devices nuevos — si pierde una key requerida, cada device aprovisionado
    desde el template no va a arrancar.
    """

    @staticmethod
    def _load_shipped() -> dict:
        path = Path(__file__).resolve().parents[2] / "config" / "config.example.yaml"
        return load_config(str(path))

    def test_shipped_validates_clean(self):
        cfg = self._load_shipped()
        assert isinstance(cfg, dict)
        assert "bracket" in cfg
        assert "vision" in cfg

    def test_shipped_baseline_is_design(self):
        cfg = self._load_shipped()
        assert cfg["bracket"]["baseline_mm"] == 140

    def test_shipped_low_confidence_threshold_active(self):
        """El matching two-stage embarca activado con un floor de 0.10.
        Estricto < confidence_threshold para que el validator lo acepte."""
        cfg = self._load_shipped()
        low = cfg["detection"].get("low_confidence_threshold")
        high = cfg["detection"]["confidence_threshold"]
        assert low is not None, "low_confidence_threshold must ship activated"
        assert 0.0 < low < high

    def test_shipped_new_track_threshold_active(self):
        cfg = self._load_shipped()
        nt = cfg["detection"].get("new_track_threshold")
        high = cfg["detection"]["confidence_threshold"]
        assert nt is not None, "new_track_threshold must ship activated"
        assert nt >= high

    def test_shipped_best_frame_default_off(self):
        """GDPR/LPDP-safe baseline: best_frame OFF unless DPIA is in place."""
        cfg = self._load_shipped()
        bf = cfg.get("best_frame", {})
        assert bf.get("enabled") is False

    def test_shipped_foot_projection_default_off(self):
        """Default off — el feature requiere depth válido (calibración
        que pase diagnose_depth.py con error <5% al centro a 2m). Con
        depth incorrecto el head_height_mm queda mal y la proyección
        comprime la trayectoria del foot-point dentro del ROI hasta
        que los exits no disparan. Activar per-site recién después
        de validar la calibración."""
        cfg = self._load_shipped()
        assert cfg["counter"]["foot_projection_enabled"] is False


# ---------------------------------------------------------------------------
# low_confidence_threshold (two-stage matching)
# ---------------------------------------------------------------------------


class TestLowConfidenceThreshold:
    def test_explicit_null_accepted(self, tmp_path, complete_config):
        complete_config["detection"]["low_confidence_threshold"] = None
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["low_confidence_threshold"] is None

    def test_missing_accepted(self, tmp_path, complete_config):
        complete_config["detection"].pop("low_confidence_threshold", None)
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"].get("low_confidence_threshold") is None

    def test_valid_below_high_accepted(self, tmp_path, complete_config):
        complete_config["detection"]["confidence_threshold"] = 0.30
        complete_config["detection"]["low_confidence_threshold"] = 0.10
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["low_confidence_threshold"] == 0.10

    def test_zero_or_negative_rejected(self, tmp_path, complete_config):
        complete_config["detection"]["low_confidence_threshold"] = 0.0
        with pytest.raises(ValueError, match="must be > 0"):
            load_config(write_config(tmp_path, complete_config))

    def test_non_numeric_rejected(self, tmp_path, complete_config):
        complete_config["detection"]["low_confidence_threshold"] = "low"
        with pytest.raises(ValueError, match="must be null or a number"):
            load_config(write_config(tmp_path, complete_config))

    def test_above_high_disables_with_warning(self, tmp_path, complete_config, caplog):
        """If low >= confidence_threshold, the validator coerces to None
        (feature off) and logs a warning — never silently broadens spawn."""
        complete_config["detection"]["confidence_threshold"] = 0.30
        complete_config["detection"]["low_confidence_threshold"] = 0.40
        with caplog.at_level(logging.WARNING):
            loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["low_confidence_threshold"] is None
        assert any(
            "low_confidence_threshold" in rec.message and "disabling" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# new_track_threshold (spawn floor)
# ---------------------------------------------------------------------------


class TestNewTrackThreshold:
    def test_explicit_null_accepted(self, tmp_path, complete_config):
        complete_config["detection"]["new_track_threshold"] = None
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["new_track_threshold"] is None

    def test_missing_accepted(self, tmp_path, complete_config):
        complete_config["detection"].pop("new_track_threshold", None)
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"].get("new_track_threshold") is None

    def test_valid_above_high_accepted(self, tmp_path, complete_config):
        complete_config["detection"]["confidence_threshold"] = 0.30
        complete_config["detection"]["new_track_threshold"] = 0.50
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["new_track_threshold"] == 0.50

    def test_zero_rejected(self, tmp_path, complete_config):
        complete_config["detection"]["new_track_threshold"] = 0.0
        with pytest.raises(ValueError, match="must be in"):
            load_config(write_config(tmp_path, complete_config))

    def test_above_one_rejected(self, tmp_path, complete_config):
        complete_config["detection"]["new_track_threshold"] = 1.5
        with pytest.raises(ValueError, match="must be in"):
            load_config(write_config(tmp_path, complete_config))

    def test_non_numeric_rejected(self, tmp_path, complete_config):
        complete_config["detection"]["new_track_threshold"] = "high"
        with pytest.raises(ValueError, match="must be null or a number"):
            load_config(write_config(tmp_path, complete_config))

    def test_below_high_disables_with_warning(self, tmp_path, complete_config, caplog):
        complete_config["detection"]["confidence_threshold"] = 0.30
        complete_config["detection"]["new_track_threshold"] = 0.20
        with caplog.at_level(logging.WARNING):
            loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["new_track_threshold"] is None
        assert any(
            "new_track_threshold" in rec.message and "disabling" in rec.message
            for rec in caplog.records
        )

    def test_equal_to_high_accepted(self, tmp_path, complete_config):
        complete_config["detection"]["confidence_threshold"] = 0.30
        complete_config["detection"]["new_track_threshold"] = 0.30
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["detection"]["new_track_threshold"] == 0.30


# ---------------------------------------------------------------------------
# best_frame block (optional, GDPR-safe defaults when missing)
# ---------------------------------------------------------------------------


class TestBestFrameValidation:
    def test_block_missing_uses_defaults(self, tmp_path, complete_config):
        complete_config.pop("best_frame", None)
        loaded = load_config(write_config(tmp_path, complete_config))
        bf = loaded["best_frame"]
        assert bf["enabled"] is False
        assert bf["retention_days"] == 7
        assert bf["buffer_size"] == 20
        assert bf["jpeg_quality"] == 85
        assert "scoring" in bf
        assert sum(bf["scoring"].values()) == pytest.approx(1.0)

    def test_partial_block_fills_defaults(self, tmp_path, complete_config):
        complete_config["best_frame"] = {"enabled": True, "retention_days": 14}
        loaded = load_config(write_config(tmp_path, complete_config))
        bf = loaded["best_frame"]
        assert bf["enabled"] is True
        assert bf["retention_days"] == 14
        assert bf["buffer_size"] == 20
        assert bf["jpeg_quality"] == 85

    def test_enabled_must_be_bool(self, tmp_path, complete_config):
        complete_config["best_frame"] = {"enabled": "yes"}
        with pytest.raises(ValueError, match="enabled"):
            load_config(write_config(tmp_path, complete_config))

    def test_retention_must_be_positive(self, tmp_path, complete_config):
        complete_config["best_frame"] = {"retention_days": 0}
        with pytest.raises(ValueError, match="retention_days"):
            load_config(write_config(tmp_path, complete_config))

    def test_buffer_size_must_be_positive(self, tmp_path, complete_config):
        complete_config["best_frame"] = {"buffer_size": -1}
        with pytest.raises(ValueError, match="buffer_size"):
            load_config(write_config(tmp_path, complete_config))

    def test_jpeg_quality_range(self, tmp_path, complete_config):
        complete_config["best_frame"] = {"jpeg_quality": 150}
        with pytest.raises(ValueError, match="jpeg_quality"):
            load_config(write_config(tmp_path, complete_config))

    def test_scoring_negative_weight_raises(self, tmp_path, complete_config):
        complete_config["best_frame"] = {"scoring": {"confidence_weight": -0.5}}
        with pytest.raises(ValueError, match="confidence_weight"):
            load_config(write_config(tmp_path, complete_config))

    def test_must_be_mapping(self, tmp_path, complete_config):
        complete_config["best_frame"] = "not a dict"
        with pytest.raises(ValueError, match="best_frame must be a mapping"):
            load_config(write_config(tmp_path, complete_config))

    def test_weights_off_by_factor_warns(self, tmp_path, complete_config, caplog):
        complete_config["best_frame"] = {
            "scoring": {
                "confidence_weight": 0.5,
                "bbox_area_weight": 0.5,
                "centrality_weight": 0.5,
                "sharpness_weight": 0.5,
            },
        }
        with caplog.at_level(logging.WARNING):
            load_config(write_config(tmp_path, complete_config))
        msgs = [r.message for r in caplog.records]
        assert any("weights sum" in m for m in msgs)


# ---------------------------------------------------------------------------
# counter.foot_projection_enabled toggle
# ---------------------------------------------------------------------------


class TestCounterToggle:
    def test_true_accepted(self, tmp_path, complete_config):
        complete_config["counter"] = {"foot_projection_enabled": True}
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["counter"]["foot_projection_enabled"] is True

    def test_false_accepted(self, tmp_path, complete_config):
        complete_config["counter"] = {"foot_projection_enabled": False}
        loaded = load_config(write_config(tmp_path, complete_config))
        assert loaded["counter"]["foot_projection_enabled"] is False

    def test_missing_block_rejected(self, tmp_path, complete_config):
        complete_config.pop("counter", None)
        with pytest.raises(ValueError, match="missing section: counter"):
            load_config(write_config(tmp_path, complete_config))

    def test_missing_key_rejected(self, tmp_path, complete_config):
        complete_config["counter"] = {}
        with pytest.raises(ValueError, match="counter.foot_projection_enabled"):
            load_config(write_config(tmp_path, complete_config))

    def test_non_bool_rejected(self, tmp_path, complete_config):
        complete_config["counter"] = {"foot_projection_enabled": "yes"}
        with pytest.raises(ValueError, match="must be a bool"):
            load_config(write_config(tmp_path, complete_config))


# ---------------------------------------------------------------------------
# Hardware section presence (bracket / sensor / lens)
# ---------------------------------------------------------------------------


class TestHardwareSections:
    def test_missing_bracket_section_raises(self, tmp_path, complete_config):
        complete_config.pop("bracket", None)
        with pytest.raises(ValueError, match="missing section: bracket"):
            load_config(write_config(tmp_path, complete_config))

    def test_missing_camera_csi_raises(self, tmp_path, complete_config):
        complete_config["bracket"] = {"baseline_mm": 140, "camera_left_csi": 0}
        with pytest.raises(ValueError, match="camera_right_csi"):
            load_config(write_config(tmp_path, complete_config))

    def test_camera_csi_must_differ(self, tmp_path, complete_config):
        complete_config["bracket"] = {
            "baseline_mm": 140,
            "camera_left_csi": 0,
            "camera_right_csi": 0,  # same as left
        }
        with pytest.raises(ValueError, match="must differ"):
            load_config(write_config(tmp_path, complete_config))

    def test_kalman_block_optional(self, tmp_path, complete_config):
        """Kalman block is optional — tracker has sane defaults if absent."""
        # complete_config base has no kalman; loading should succeed.
        loaded = load_config(write_config(tmp_path, complete_config))
        sm = loaded["tracking"]["state_machine"]
        assert sm.get("kalman") is None or isinstance(sm.get("kalman"), dict)

    def test_kalman_block_when_present_validated(self, tmp_path, complete_config):
        complete_config["tracking"]["state_machine"]["kalman"] = {
            "process_noise": "not_a_number",
        }
        with pytest.raises(ValueError, match="kalman"):
            load_config(write_config(tmp_path, complete_config))

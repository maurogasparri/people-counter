"""Tests para el loader del config strict + el merge del cloud shadow."""

import json
from pathlib import Path

import pytest
import yaml

from src.config.loader import (
    apply_shadow_delta,
    get_effective_value,
    get_invalid_schedule_mode,
    get_scaling_factor,
    has_schedule_error,
    is_counting_enabled,
    is_wifi_ble_enabled,
    is_within_operating_hours,
    load_config,
    merge_cloud_config,
    validate_operating_hours,
)

from .conftest import write_config


# --- load_config ---


class TestLoadConfig:
    def test_load_valid_config(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert cfg["device"]["id"] == "test-device"
        assert cfg["device"]["store_id"] == "test-store"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_missing_device_id_raises(self, tmp_path, complete_config):
        complete_config["device"] = {"store_id": "s1"}
        with pytest.raises(ValueError, match="device.id"):
            load_config(write_config(tmp_path, complete_config))

    def test_missing_section_raises(self, tmp_path, complete_config):
        complete_config.pop("mqtt", None)
        with pytest.raises(ValueError, match="missing section"):
            load_config(write_config(tmp_path, complete_config))

    def test_rssi_threshold_validation(self, tmp_path, complete_config):
        """Shopper threshold must be greater (less negative) than passerby."""
        complete_config["wifi_ble"]["rssi_passerby_threshold"] = -55
        complete_config["wifi_ble"]["rssi_shopper_threshold"] = -75
        with pytest.raises(ValueError, match="rssi_shopper_threshold"):
            load_config(write_config(tmp_path, complete_config))


# --- merge_cloud_config ---


class TestMergeCloudConfig:
    def test_no_shadow_returns_local(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        merged = merge_cloud_config(cfg, {})
        assert merged["cloud_defaults"]["footfall_scaling_factor"] == 1.0

    def test_shadow_overrides_scaling_factor(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        shadow = {"footfall_scaling_factor": 1.15}
        merged = merge_cloud_config(cfg, shadow)
        assert merged["cloud_defaults"]["footfall_scaling_factor"] == 1.15

    def test_shadow_overrides_operating_hours(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        new_hours = {"monday": "09:00-21:00", "sunday": None}
        shadow = {"operating_hours": new_hours}
        merged = merge_cloud_config(cfg, shadow)
        assert merged["cloud_defaults"]["operating_hours"]["monday"] == "09:00-21:00"
        assert merged["cloud_defaults"]["operating_hours"]["sunday"] is None

    def test_shadow_does_not_override_non_allowed_keys(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        shadow = {"device_id": "hacked", "model_path": "/evil"}
        merged = merge_cloud_config(cfg, shadow)
        assert merged["device"]["id"] == "test-device"
        assert "device_id" not in merged["cloud_defaults"]

    def test_shadow_disables_counting(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        shadow = {"counting_enabled": False}
        merged = merge_cloud_config(cfg, shadow)
        assert not is_counting_enabled(merged)

    def test_original_config_not_mutated(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        original_factor = cfg["cloud_defaults"]["footfall_scaling_factor"]
        merge_cloud_config(cfg, {"footfall_scaling_factor": 2.0})
        assert cfg["cloud_defaults"]["footfall_scaling_factor"] == original_factor


# --- Operating hours ---


class TestOperatingHours:
    def test_within_hours(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert is_within_operating_hours(cfg, "monday", 12, 0)

    def test_before_opening(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert not is_within_operating_hours(cfg, "monday", 9, 30)

    def test_after_closing(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert not is_within_operating_hours(cfg, "monday", 22, 0)

    def test_at_exact_opening(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert is_within_operating_hours(cfg, "monday", 10, 0)

    def test_at_exact_closing_is_outside(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert not is_within_operating_hours(cfg, "monday", 22, 0)

    def test_sunday_different_hours(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert is_within_operating_hours(cfg, "sunday", 15, 0)
        assert not is_within_operating_hours(cfg, "sunday", 20, 30)

    def test_missing_day_returns_false(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert not is_within_operating_hours(cfg, "wednesday", 12, 0)

    def test_null_schedule_returns_false(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        cfg["cloud_defaults"]["operating_hours"]["monday"] = None
        assert not is_within_operating_hours(cfg, "monday", 12, 0)


# --- Feature toggles ---


class TestFeatureToggles:
    def test_counting_enabled_default(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert is_counting_enabled(cfg)

    def test_wifi_ble_enabled_both_true(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert is_wifi_ble_enabled(cfg)

    def test_wifi_ble_disabled_locally(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        cfg["wifi_ble"]["enabled"] = False
        assert not is_wifi_ble_enabled(cfg)

    def test_wifi_ble_disabled_from_cloud(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        merged = merge_cloud_config(cfg, {"wifi_ble_enabled": False})
        assert not is_wifi_ble_enabled(merged)

    def test_scaling_factor_default(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert get_scaling_factor(cfg) == 1.0

    def test_scaling_factor_from_cloud(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        merged = merge_cloud_config(cfg, {"footfall_scaling_factor": 1.1})
        assert get_scaling_factor(merged) == pytest.approx(1.1)

    def test_get_effective_value_with_fallback(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert get_effective_value(cfg, "nonexistent_key", 42) == 42


# --- Operating hours validation + fail mode ---


class TestValidateOperatingHours:
    def test_valid_schedule(self):
        hours = {"monday": "10:00-22:00", "tuesday": None}
        assert validate_operating_hours(hours) is None

    def test_missing_hours(self):
        assert "missing" in validate_operating_hours(None).lower()

    def test_not_a_mapping(self):
        err = validate_operating_hours("10:00-22:00")
        assert err is not None
        assert "mapping" in err

    def test_bad_hour_value(self):
        err = validate_operating_hours({"monday": "25:00-26:00"})
        assert err is not None
        assert "monday" in err
        assert "25:00" in err

    def test_bad_minute_value(self):
        err = validate_operating_hours({"monday": "10:60-22:00"})
        assert err is not None
        assert "10:60" in err

    def test_missing_separator(self):
        err = validate_operating_hours({"monday": "10:00 22:00"})
        assert err is not None
        assert "-" in err or "HH:MM" in err

    def test_non_string_schedule(self):
        err = validate_operating_hours({"monday": 12})
        assert err is not None
        assert "string" in err

    def test_end_before_start(self):
        err = validate_operating_hours({"monday": "22:00-10:00"})
        assert err is not None
        assert "after start" in err

    def test_end_equals_start(self):
        err = validate_operating_hours({"monday": "10:00-10:00"})
        assert err is not None
        assert "after start" in err

    def test_malformed_time(self):
        err = validate_operating_hours({"monday": "ab:cd-22:00"})
        assert err is not None
        assert "ab:cd" in err


class TestLoadConfigSoftScheduleValidation:
    def test_valid_schedule_no_error_flag(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert not has_schedule_error(cfg)
        assert "_schedule_error" not in cfg

    def test_invalid_schedule_does_not_raise(self, tmp_path, complete_config):
        complete_config["cloud_defaults"]["operating_hours"] = {
            "monday": "25:00-26:00",
        }
        cfg = load_config(write_config(tmp_path, complete_config))  # must not raise
        assert has_schedule_error(cfg)
        assert "25:00" in cfg["_schedule_error"]

    def test_invalid_schedule_logs_warning(self, tmp_path, complete_config, caplog):
        complete_config["cloud_defaults"]["operating_hours"] = {
            "monday": "22:00-10:00",
        }
        with caplog.at_level("WARNING", logger="src.config.loader"):
            cfg = load_config(write_config(tmp_path, complete_config))
        assert has_schedule_error(cfg)
        assert any(
            rec.message == "invalid_operating_hours"
            and getattr(rec, "reason", None)
            for rec in caplog.records
        )

    def test_missing_operating_hours_is_error(self, tmp_path, complete_config):
        complete_config["cloud_defaults"] = {}
        cfg = load_config(write_config(tmp_path, complete_config))
        assert has_schedule_error(cfg)


class TestApplyShadowDelta:
    def test_empty_delta_no_applied_keys(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        new_cfg, applied = apply_shadow_delta(cfg, {})
        assert applied == []
        assert new_cfg == cfg

    def test_safe_key_applied_operating_hours(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        new_hours = {"monday": "08:00-20:00", "tuesday": "08:00-20:00"}
        delta = {"cloud_defaults": {"operating_hours": new_hours}}
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "cloud_defaults.operating_hours" in applied
        assert new_cfg["cloud_defaults"]["operating_hours"] == new_hours
        # Original config untouched (deep copy returned).
        assert cfg["cloud_defaults"]["operating_hours"] != new_hours

    def test_unsafe_key_ignored(self, complete_config_yaml, caplog):
        cfg = load_config(complete_config_yaml)
        delta = {"mqtt": {"endpoint": "attacker.example.com"}}
        with caplog.at_level("WARNING", logger="src.config.loader"):
            new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert applied == []
        assert new_cfg["mqtt"]["endpoint"] == cfg["mqtt"]["endpoint"]
        assert any(
            r.message == "shadow_delta_requires_restart"
            for r in caplog.records
        )

    def test_unknown_key_ignored(self, complete_config_yaml, caplog):
        cfg = load_config(complete_config_yaml)
        delta = {"totally_unknown_thing": 42}
        with caplog.at_level("WARNING", logger="src.config.loader"):
            new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert applied == []
        assert "totally_unknown_thing" not in new_cfg
        assert any(
            r.message == "shadow_delta_requires_restart"
            for r in caplog.records
        )

    def test_nested_counter_tracker_prefix_applied(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        delta = {
            "counter": {
                "tracker": {"confirm_frames": 5, "reid_gate_px": 100.0}
            }
        }
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "counter.tracker.confirm_frames" in applied
        assert "counter.tracker.reid_gate_px" in applied
        assert new_cfg["counter"]["tracker"]["confirm_frames"] == 5
        assert new_cfg["counter"]["tracker"]["reid_gate_px"] == 100.0

    def test_mixed_safe_and_unsafe(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        delta = {
            "telemetry": {"interval_seconds": 60},
            "device_id": "hacker",  # identity — must be rejected
            "vision": {"num_disparities": 128, "baseline_cm": 99},
        }
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "telemetry.interval_seconds" in applied
        assert "vision.num_disparities" in applied
        assert "vision.baseline_cm" not in applied
        assert "device_id" not in applied
        assert new_cfg["telemetry"]["interval_seconds"] == 60
        assert new_cfg["vision"]["num_disparities"] == 128
        assert new_cfg["device"]["id"] == cfg["device"]["id"]

    def test_persists_shadow_cache(self, complete_config_yaml, tmp_path):
        cfg = load_config(complete_config_yaml)
        delta = {"cloud_defaults": {"on_invalid_schedule": "fail_closed"}}
        new_cfg, applied = apply_shadow_delta(
            cfg, delta, config_path=complete_config_yaml
        )
        assert applied == ["cloud_defaults.on_invalid_schedule"]

        shadow_file = Path(complete_config_yaml).with_suffix(".shadow.json")
        assert shadow_file.exists()
        persisted = json.loads(shadow_file.read_text())
        assert (
            persisted["state"]["desired"]["cloud_defaults"]["on_invalid_schedule"]
            == "fail_closed"
        )

    def test_no_persist_when_no_keys_applied(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        delta = {"mqtt": {"endpoint": "blocked"}}
        apply_shadow_delta(cfg, delta, config_path=complete_config_yaml)

        shadow_file = Path(complete_config_yaml).with_suffix(".shadow.json")
        assert not shadow_file.exists()

    def test_operational_prefix_applied(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        delta = {"operational": {"max_queue_depth": 500, "debug_trace": True}}
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "operational.max_queue_depth" in applied
        assert "operational.debug_trace" in applied
        assert new_cfg["operational"]["max_queue_depth"] == 500


class TestInvalidScheduleMode:
    def test_default_is_fail_open(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert get_invalid_schedule_mode(cfg) == "fail_open"

    def test_fail_closed_from_cloud_defaults(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        cfg["cloud_defaults"]["on_invalid_schedule"] = "fail_closed"
        assert get_invalid_schedule_mode(cfg) == "fail_closed"

    def test_unknown_value_falls_back_to_fail_open(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        cfg["cloud_defaults"]["on_invalid_schedule"] = "nonsense"
        assert get_invalid_schedule_mode(cfg) == "fail_open"

    def test_shadow_can_override_mode(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        merged = merge_cloud_config(cfg, {"on_invalid_schedule": "fail_closed"})
        assert get_invalid_schedule_mode(merged) == "fail_closed"


class TestBuildReportedState:
    """build_reported_state curates shadow-reportable config + metadata."""

    def _cfg(self) -> dict:
        return {
            "device": {
                "id": "dev-001",
                "store_id": "store-01",
                "firmware_version": "0.9.3-test",
            },
            "mqtt": {
                "endpoint": "x.iot.amazonaws.com",
                "cert_path": "/etc/certs/dev.pem",  # should NOT be reported
            },
            "vision": {
                "num_disparities": 192,
                "block_size": 9,
                "calibration_file": "/etc/people-counter/calibration.npz",
            },
            "cloud_defaults": {
                "operating_hours": {"monday": "09:00-22:00"},
                "on_invalid_schedule": "fail_open",
            },
            "telemetry": {"interval_seconds": 300},
            "counter": {
                "lines": [
                    {
                        "from": [0, 240], "to": [640, 240],
                        "labels": {"top_to_bottom": "ingress"},
                    },
                ],
                "tracker": {"confirm_frames": 3, "depth_gate_m": 0.5},
            },
            "operational": {"paused": False},
        }

    def test_includes_whitelisted_keys_only(self):
        from src.config.loader import build_reported_state

        reported = build_reported_state(self._cfg(), calibration=None)

        # Whitelisted scalars present
        assert reported["vision"]["num_disparities"] == 192
        assert reported["vision"]["block_size"] == 9
        assert reported["telemetry"]["interval_seconds"] == 300
        assert reported["cloud_defaults"]["operating_hours"] == {
            "monday": "09:00-22:00"
        }
        assert reported["cloud_defaults"]["on_invalid_schedule"] == "fail_open"
        # Prefix-based whitelists present
        assert reported["counter"]["tracker"]["confirm_frames"] == 3
        assert reported["counter"]["lines"] == [
            {
                "from": [0, 240], "to": [640, 240],
                "labels": {"top_to_bottom": "ingress"},
            },
        ]
        assert reported["operational"] == {"paused": False}

        # Secrets / endpoints must NOT leak into reported
        assert "mqtt" not in reported
        assert "cert_path" not in reported.get("vision", {})

    def test_metadata_included(self):
        from src.config.loader import build_reported_state

        reported = build_reported_state(self._cfg(), calibration=None)
        assert reported["firmware_version"] == "0.9.3-test"
        assert isinstance(reported["boot_ts"], int)
        assert reported["boot_ts"] > 0
        assert reported["calibration_file_path"] == (
            "/etc/people-counter/calibration.npz"
        )

    def test_firmware_version_defaults_to_unknown(self):
        from src.config.loader import build_reported_state

        cfg = self._cfg()
        cfg["device"].pop("firmware_version")
        reported = build_reported_state(cfg, calibration=None)
        assert reported["firmware_version"] == "unknown"

    def test_none_calibration_yields_none_baseline(self):
        from src.config.loader import build_reported_state

        reported = build_reported_state(self._cfg(), calibration=None)
        assert reported["effective_baseline_mm"] is None

    def test_calibration_baseline_derived_from_T(self):
        import numpy as np

        from src.config.loader import build_reported_state

        calibration = {"T": np.array([[-142.3], [0.0], [0.0]], dtype=np.float64)}
        reported = build_reported_state(self._cfg(), calibration=calibration)
        assert reported["effective_baseline_mm"] == pytest.approx(142.3, rel=1e-6)

    def test_missing_config_sections_gracefully_skipped(self):
        from src.config.loader import build_reported_state

        # Only device section present — should not raise.
        cfg = {"device": {"id": "x", "store_id": "y"}}
        reported = build_reported_state(cfg, calibration=None)
        assert reported["firmware_version"] == "unknown"
        # Either None or the configured path — both valid; assert no crash.
        assert "calibration_file_path" in reported
        assert reported["effective_baseline_mm"] is None
        # None of the whitelisted keys should appear (they're absent from cfg)
        assert "counter" not in reported
        assert "telemetry" not in reported
        assert "cloud_defaults" not in reported


class TestBackwardCompatRenames:
    """El loader acepta los nombres viejos de keys que cambiaron, así un
    config per-device aprovisionado antes del rename sigue cargando sin
    intervención manual del operador. Tests cubren las dos categorías:
    renames (vieja → nueva) y removals (obsoleta).
    """

    def test_max_disappeared_legacy_name_renamed(self, complete_config, tmp_path):
        cfg = complete_config
        cfg["tracking"]["max_disappeared"] = cfg["tracking"].pop("max_disappeared_frames")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(str(cfg_path))
        assert loaded["tracking"]["max_disappeared_frames"] == 30
        assert "max_disappeared" not in loaded["tracking"]

    def test_max_distance_legacy_name_renamed(self, complete_config, tmp_path):
        cfg = complete_config
        cfg["tracking"]["max_distance"] = cfg["tracking"].pop("max_distance_px")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(str(cfg_path))
        assert loaded["tracking"]["max_distance_px"] == 50
        assert "max_distance" not in loaded["tracking"]

    def test_ae_lock_legacy_names_renamed(self, complete_config, tmp_path):
        cfg = complete_config
        cfg["vision"]["ae_lock"] = {
            "initial_settle_s": 3.0,
            "resettle_s": 2.0,
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(str(cfg_path))
        ae_lock = loaded["vision"]["ae_lock"]
        assert ae_lock["initial_settle_seconds"] == 3.0
        assert ae_lock["resettle_seconds"] == 2.0
        assert "initial_settle_s" not in ae_lock
        assert "resettle_s" not in ae_lock

    def test_obsolete_sensor_raw_size_removed(self, complete_config, tmp_path):
        """vision.sensor_raw_size es obsoleta — ahora se usa
        sensor.default_res como fuente única. El loader la dropea
        silenciosamente (con warning), no rompe el load."""
        cfg = complete_config
        cfg["vision"]["sensor_raw_size"] = [2304, 1296]
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(str(cfg_path))
        assert "sensor_raw_size" not in loaded["vision"]

    def test_obsolete_approx_fps_removed(self, complete_config, tmp_path):
        """detection.static_suppressor.approx_fps es obsoleta — el
        supresor mide la ventana con timestamps reales."""
        cfg = complete_config
        cfg["detection"]["static_suppressor"] = {
            "enabled": True,
            "cell_size_px": 30,
            "window_seconds": 3.0,
            "hit_rate_threshold": 0.7,
            "approx_fps": 17,  # legacy
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(str(cfg_path))
        suppressor = loaded["detection"]["static_suppressor"]
        assert "approx_fps" not in suppressor
        # Y las keys actuales siguen intactas.
        assert suppressor["window_seconds"] == 3.0
        assert suppressor["hit_rate_threshold"] == 0.7

    def test_both_old_and_new_present_keeps_new(self, complete_config, tmp_path):
        """Si por accidente el config tiene AMBOS nombres, gana el nuevo
        y el viejo se dropea (con warning)."""
        cfg = complete_config
        # max_disappeared_frames=30 ya está; metemos también el viejo
        # con un valor distinto.
        cfg["tracking"]["max_disappeared"] = 999
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(str(cfg_path))
        assert loaded["tracking"]["max_disappeared_frames"] == 30
        assert "max_disappeared" not in loaded["tracking"]

    def test_config_already_canonical_unchanged(
        self, complete_config_yaml,
    ):
        """Si el config ya está en los nombres canónicos, el load es
        idempotente — no hay renames espurios."""
        loaded = load_config(complete_config_yaml)
        assert loaded["tracking"]["max_disappeared_frames"] == 30
        assert loaded["tracking"]["max_distance_px"] == 50
        # Verificar que no aparecieron las keys legacy por accidente.
        assert "max_disappeared" not in loaded["tracking"]
        assert "max_distance" not in loaded["tracking"]

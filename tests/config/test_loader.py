"""Tests para el loader del config strict + el merge del cloud shadow."""

from pathlib import Path

import pytest
import yaml

from src.config.loader import (
    apply_shadow_delta,
    get_effective_value,
    is_counting_enabled,
    is_external_traffic_enabled,
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
        # Sin shadow, los valores del config.yaml local se preservan.
        assert merged["cloud_defaults"]["counting_enabled"] is True

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
        """``merge_cloud_config`` devuelve una copia — el cfg original no se
        muta aunque el shadow contenga keys overridables."""
        cfg = load_config(complete_config_yaml)
        original_hours = dict(cfg["cloud_defaults"]["operating_hours"])
        merge_cloud_config(cfg, {"operating_hours": {"monday": "00:00-23:59"}})
        assert cfg["cloud_defaults"]["operating_hours"] == original_hours


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

    def test_counting_disabled_by_shadow(self, complete_config_yaml):
        """`counting_enabled` SÍ es overridable por shadow (feature
        toggle whitelisted en CLOUD_OVERRIDABLE)."""
        cfg = load_config(complete_config_yaml)
        merged = merge_cloud_config(cfg, {"counting_enabled": False})
        assert not is_counting_enabled(merged)

    def test_external_traffic_default_true(self, complete_config_yaml):
        """Default: wifi_ble.enabled local + cloud_defaults.external_traffic_enabled
        ambos true → habilitado."""
        cfg = load_config(complete_config_yaml)
        assert is_external_traffic_enabled(cfg)

    def test_external_traffic_disabled_locally(self, complete_config_yaml):
        """Cualquiera de los 2 toggles en false → deshabilitado.
        Local toma efecto sin shadow."""
        cfg = load_config(complete_config_yaml)
        cfg["wifi_ble"]["enabled"] = False
        assert not is_external_traffic_enabled(cfg)

    def test_external_traffic_disabled_by_shadow(self, complete_config_yaml):
        """external_traffic_enabled SÍ es overridable por shadow (toggle
        de privacidad por sucursal, agregado al CLOUD_OVERRIDABLE)."""
        cfg = load_config(complete_config_yaml)
        merged = merge_cloud_config(cfg, {"external_traffic_enabled": False})
        assert not is_external_traffic_enabled(merged)

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


class TestLoadConfigScheduleValidation:
    """operating_hours se valida fail-fast en load_config — sin
    `_schedule_error` flag / fail_open ya no aplica. Para los deltas del
    shadow, el equivalente es el rechazo pre-apply (ver
    TestShadowDeltaValidation)."""

    def test_valid_schedule_loads(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        assert cfg["cloud_defaults"]["operating_hours"]

    def test_invalid_schedule_raises(self, tmp_path, complete_config):
        complete_config["cloud_defaults"]["operating_hours"] = {
            "monday": "25:00-26:00",
        }
        with pytest.raises(ValueError, match="Invalid operating_hours"):
            load_config(write_config(tmp_path, complete_config))

    def test_end_before_start_raises(self, tmp_path, complete_config):
        complete_config["cloud_defaults"]["operating_hours"] = {
            "monday": "22:00-10:00",
        }
        with pytest.raises(ValueError, match="Invalid operating_hours"):
            load_config(write_config(tmp_path, complete_config))

    def test_missing_operating_hours_raises(self, tmp_path, complete_config):
        complete_config["cloud_defaults"] = {}
        with pytest.raises(ValueError, match="Invalid operating_hours"):
            load_config(write_config(tmp_path, complete_config))


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

    def test_bare_top_level_key_mapped_to_cloud_defaults(
        self, complete_config_yaml,
    ):
        """AWS IoT publica state.desired al ras del root (sin namespacing
        operator-friendly): {counting_enabled: false}. apply_shadow_delta
        debe mapearla a cloud_defaults.counting_enabled antes de validar,
        coherente con merge_cloud_config (startup) y con el operator
        guide."""
        cfg = load_config(complete_config_yaml)
        # Delta bare como lo manda AWS desde un push estilo
        # `aws iot-data update-thing-shadow --payload
        #  '{"state":{"desired":{"counting_enabled":false}}}'`.
        delta = {"counting_enabled": False}
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "cloud_defaults.counting_enabled" in applied
        assert new_cfg["cloud_defaults"]["counting_enabled"] is False

    def test_bare_external_traffic_mapped(self, complete_config_yaml):
        """Mismo mapeo bare→cloud_defaults para external_traffic_enabled."""
        cfg = load_config(complete_config_yaml)
        delta = {"external_traffic_enabled": False}
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "cloud_defaults.external_traffic_enabled" in applied
        assert new_cfg["cloud_defaults"]["external_traffic_enabled"] is False

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

    def test_counter_tracker_no_longer_pushable(self, complete_config_yaml):
        """counter.tracker.* dejó de ser pusheable — tuning algorítmico
        requiere SSH + restart, no end user via shadow."""
        cfg = load_config(complete_config_yaml)
        delta = {
            "counter": {
                "tracker": {"confirm_frames": 5, "reid_gate_px": 100.0}
            }
        }
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert applied == []
        # El config no tiene counter.tracker por default; el push no debe
        # crearlo silenciosamente.
        assert "tracker" not in new_cfg.get("counter", {})

    def test_mixed_safe_and_unsafe(self, complete_config_yaml):
        """Solo cloud_defaults.{operating_hours, counting_enabled} son
        runtime-safe. El resto (incluso si parecía técnico-razonable como
        telemetry.interval_seconds) se rechaza."""
        cfg = load_config(complete_config_yaml)
        delta = {
            "cloud_defaults": {"counting_enabled": False},
            "telemetry": {"interval_seconds": 60},
            "device_id": "hacker",
            "vision": {"num_disparities": 128, "baseline_cm": 99},
        }
        new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert "cloud_defaults.counting_enabled" in applied
        assert "telemetry.interval_seconds" not in applied
        assert "vision.num_disparities" not in applied
        assert "device_id" not in applied
        assert new_cfg["cloud_defaults"]["counting_enabled"] is False
        # Las keys técnicas no se inyectaron silenciosamente.
        assert "telemetry" not in new_cfg or new_cfg.get("telemetry") == cfg.get(
            "telemetry"
        )
        assert "vision" not in new_cfg or new_cfg["vision"] == cfg.get("vision")
        assert new_cfg["device"]["id"] == cfg["device"]["id"]

    def test_persists_to_config_yaml(self, complete_config_yaml):
        """Cambio de 2026-05-23: el shadow apply persiste al MISMO
        config.yaml (no a un sibling .shadow.json). Single source of
        truth — el operator SSH ve lo que el device tiene corriendo."""
        cfg = load_config(complete_config_yaml)
        delta = {"counting_enabled": False}
        new_cfg, applied = apply_shadow_delta(
            cfg, delta, config_path=complete_config_yaml
        )
        assert applied == ["cloud_defaults.counting_enabled"]

        # El config.yaml se reescribió con el nuevo valor.
        reloaded = yaml.safe_load(Path(complete_config_yaml).read_text())
        assert reloaded["cloud_defaults"]["counting_enabled"] is False

        # NO se creó sibling .shadow.json (legacy eliminado).
        assert not Path(complete_config_yaml).with_suffix(".shadow.json").exists()

    def test_no_persist_when_no_keys_applied(self, complete_config_yaml):
        cfg = load_config(complete_config_yaml)
        delta = {"mqtt": {"endpoint": "blocked"}}
        # Capturar el contenido original del YAML para detectar mutación.
        original_yaml = Path(complete_config_yaml).read_text()
        apply_shadow_delta(cfg, delta, config_path=complete_config_yaml)
        # Sin keys aplicadas, el config.yaml no se reescribe.
        assert Path(complete_config_yaml).read_text() == original_yaml

    def test_operational_prefix_no_longer_pushable(self, complete_config_yaml):
        """operational.* dejó de ser pusheable — runtime tuning queda en
        config.yaml + SSH."""
        cfg = load_config(complete_config_yaml)
        delta = {"operational": {"max_queue_depth": 500, "debug_trace": True}}
        _new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert applied == []


class TestShadowDeltaValidation:
    """Los deltas que fallan validación se rechazan ANTES de aplicar +
    persistir al config.yaml (no se silencian con fail_open/fail_closed)."""

    def test_invalid_operating_hours_rejected(
        self, complete_config_yaml, caplog,
    ):
        cfg = load_config(complete_config_yaml)
        original = cfg["cloud_defaults"]["operating_hours"].copy()
        # Schedule con end <= start es inválido.
        delta = {"operating_hours": {"monday": "22:00-10:00"}}
        with caplog.at_level("WARNING", logger="src.config.loader"):
            new_cfg, applied = apply_shadow_delta(cfg, delta)
        # No se aplica.
        assert applied == []
        assert new_cfg["cloud_defaults"]["operating_hours"] == original
        # Log del rechazo.
        assert any(
            r.message == "shadow_delta_rejected_invalid" for r in caplog.records
        )

    def test_invalid_counting_enabled_rejected(self, complete_config_yaml, caplog):
        """Un toggle bool con un valor no-bool (ej. string ``"false"``) se
        rechaza ANTES de persistir. Sin esto, ``bool("false")`` es truthy y el
        conteo quedaría encendido cuando el operador quiso apagarlo."""
        cfg = load_config(complete_config_yaml)
        original = cfg["cloud_defaults"].get("counting_enabled")
        delta = {"counting_enabled": "false"}  # string, no bool
        with caplog.at_level("WARNING", logger="src.config.loader"):
            new_cfg, applied = apply_shadow_delta(cfg, delta)
        assert applied == []
        assert new_cfg["cloud_defaults"].get("counting_enabled") == original
        assert any(
            r.message == "shadow_delta_rejected_invalid" for r in caplog.records
        )

    def test_invalid_external_traffic_enabled_rejected(self, complete_config_yaml):
        """Un entero (0/1) tampoco pasa el contrato estricto bool."""
        cfg = load_config(complete_config_yaml)
        new_cfg, applied = apply_shadow_delta(cfg, {"external_traffic_enabled": 1})
        assert applied == []

    def test_valid_counting_enabled_toggle_applies(self, complete_config_yaml):
        """Un bool legítimo sí se aplica y persiste."""
        cfg = load_config(complete_config_yaml)
        new_cfg, applied = apply_shadow_delta(cfg, {"counting_enabled": False})
        assert applied == ["cloud_defaults.counting_enabled"]
        assert new_cfg["cloud_defaults"]["counting_enabled"] is False


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

        cfg = self._cfg()
        # Aseguramos que counting_enabled esté presente para verificar el report.
        cfg["cloud_defaults"]["counting_enabled"] = True
        reported = build_reported_state(cfg, calibration=None)

        # Sólo las 2 keys pusheables están en el reported state.
        assert reported["cloud_defaults"]["operating_hours"] == {
            "monday": "09:00-22:00"
        }
        assert reported["cloud_defaults"]["counting_enabled"] is True

        # Las keys técnicas (no pusheables) no reportan al shadow.
        assert "vision" not in reported
        assert "telemetry" not in reported
        assert "operational" not in reported
        assert reported.get("cloud_defaults", {}).get("on_invalid_cloud_config") is None

        # Secrets / endpoints nunca filtran al reported.
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

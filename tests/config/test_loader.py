"""Tests for configuration loader with local/cloud merge."""

import pytest
import yaml

from src.config.loader import (
    get_effective_value,
    get_scaling_factor,
    is_counting_enabled,
    is_wifi_ble_enabled,
    is_within_operating_hours,
    load_config,
    merge_cloud_config,
)


@pytest.fixture
def minimal_yaml(tmp_path):
    """Write a minimal valid config YAML and return its path."""
    config = {
        "device": {"id": "test-device-01", "store_id": "store-001"},
        "vision": {
            "camera_left": 0,
            "camera_right": 1,
            "resolution": [640, 480],
            "calibration_file": "/tmp/cal.npz",
            "mounting_height_m": 3.0,
            "counting_line_y": 0.5,
        },
        "detection": {
            "model_path": "/tmp/model.hef",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.45,
        },
        "mqtt": {
            "endpoint": "test.iot.amazonaws.com",
            "port": 8883,
            "cert_path": "/tmp/cert.pem",
            "key_path": "/tmp/key.pem",
            "ca_path": "/tmp/ca.pem",
            "topics": {"counting": "store/{store_id}/counting"},
        },
        "buffer": {"db_path": "/tmp/buffer.db", "max_age_hours": 24},
        "wifi_ble": {
            "enabled": True,
            "wifi_interface": "wlan0",
            "rssi_passerby_threshold": -75,
            "rssi_shopper_threshold": -55,
        },
        "cloud_defaults": {
            "operating_hours": {
                "monday": "10:00-22:00",
                "tuesday": "10:00-22:00",
                "sunday": "11:00-20:00",
            },
            "footfall_scaling_factor": 1.0,
            "counting_enabled": True,
            "wifi_ble_enabled": True,
            "telemetry_interval_seconds": 300,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


# --- load_config ---


class TestLoadConfig:
    def test_load_valid_config(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert cfg["device"]["id"] == "test-device-01"
        assert cfg["device"]["store_id"] == "store-001"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_missing_device_id_raises(self, tmp_path):
        bad = {"device": {"store_id": "s1"}, "vision": {}, "detection": {}, "mqtt": {}, "buffer": {}}
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="device.id"):
            load_config(str(path))

    def test_missing_section_raises(self, tmp_path):
        bad = {"device": {"id": "d1", "store_id": "s1"}}
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="Missing required"):
            load_config(str(path))

    def test_rssi_threshold_validation(self, tmp_path):
        """Shopper threshold must be greater (less negative) than passerby."""
        config = {
            "device": {"id": "d1", "store_id": "s1"},
            "vision": {},
            "detection": {},
            "mqtt": {},
            "buffer": {},
            "wifi_ble": {
                "enabled": True,
                "rssi_passerby_threshold": -55,
                "rssi_shopper_threshold": -75,  # WRONG: shopper weaker than passerby
            },
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(config))
        with pytest.raises(ValueError, match="rssi_shopper_threshold"):
            load_config(str(path))


# --- merge_cloud_config ---


class TestMergeCloudConfig:
    def test_no_shadow_returns_local(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        merged = merge_cloud_config(cfg, {})
        assert merged["cloud_defaults"]["footfall_scaling_factor"] == 1.0

    def test_shadow_overrides_scaling_factor(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        shadow = {"footfall_scaling_factor": 1.15}
        merged = merge_cloud_config(cfg, shadow)
        assert merged["cloud_defaults"]["footfall_scaling_factor"] == 1.15

    def test_shadow_overrides_operating_hours(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        new_hours = {"monday": "09:00-21:00", "sunday": None}
        shadow = {"operating_hours": new_hours}
        merged = merge_cloud_config(cfg, shadow)
        assert merged["cloud_defaults"]["operating_hours"]["monday"] == "09:00-21:00"
        assert merged["cloud_defaults"]["operating_hours"]["sunday"] is None

    def test_shadow_does_not_override_non_allowed_keys(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        shadow = {"device_id": "hacked", "model_path": "/evil"}
        merged = merge_cloud_config(cfg, shadow)
        assert merged["device"]["id"] == "test-device-01"
        assert "device_id" not in merged["cloud_defaults"]

    def test_shadow_disables_counting(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        shadow = {"counting_enabled": False}
        merged = merge_cloud_config(cfg, shadow)
        assert not is_counting_enabled(merged)

    def test_original_config_not_mutated(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        original_factor = cfg["cloud_defaults"]["footfall_scaling_factor"]
        merge_cloud_config(cfg, {"footfall_scaling_factor": 2.0})
        assert cfg["cloud_defaults"]["footfall_scaling_factor"] == original_factor


# --- Operating hours ---


class TestOperatingHours:
    def test_within_hours(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert is_within_operating_hours(cfg, "monday", 12, 0)

    def test_before_opening(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert not is_within_operating_hours(cfg, "monday", 9, 30)

    def test_after_closing(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert not is_within_operating_hours(cfg, "monday", 22, 0)

    def test_at_exact_opening(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert is_within_operating_hours(cfg, "monday", 10, 0)

    def test_at_exact_closing_is_outside(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert not is_within_operating_hours(cfg, "monday", 22, 0)

    def test_sunday_different_hours(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert is_within_operating_hours(cfg, "sunday", 15, 0)
        assert not is_within_operating_hours(cfg, "sunday", 20, 30)

    def test_missing_day_returns_false(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert not is_within_operating_hours(cfg, "wednesday", 12, 0)

    def test_null_schedule_returns_false(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        cfg["cloud_defaults"]["operating_hours"]["monday"] = None
        assert not is_within_operating_hours(cfg, "monday", 12, 0)


# --- Feature toggles ---


class TestFeatureToggles:
    def test_counting_enabled_default(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert is_counting_enabled(cfg)

    def test_wifi_ble_enabled_both_true(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert is_wifi_ble_enabled(cfg)

    def test_wifi_ble_disabled_locally(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        cfg["wifi_ble"]["enabled"] = False
        assert not is_wifi_ble_enabled(cfg)

    def test_wifi_ble_disabled_from_cloud(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        merged = merge_cloud_config(cfg, {"wifi_ble_enabled": False})
        assert not is_wifi_ble_enabled(merged)

    def test_scaling_factor_default(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert get_scaling_factor(cfg) == 1.0

    def test_scaling_factor_from_cloud(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        merged = merge_cloud_config(cfg, {"footfall_scaling_factor": 1.1})
        assert get_scaling_factor(merged) == pytest.approx(1.1)

    def test_get_effective_value_with_fallback(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert get_effective_value(cfg, "nonexistent_key", 42) == 42

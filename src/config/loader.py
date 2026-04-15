"""Configuration loader with local/cloud merge support.

Strategy:
    - LOCAL config: hardware-intrinsic settings from YAML file.
    - CLOUD config: business-driven settings from AWS IoT Device Shadow.
    - Cloud values override local defaults in the 'cloud_defaults' section.

The device boots with local YAML, then fetches its IoT Shadow and merges
cloud-pushed overrides. This allows operations teams to change operating
hours, enable/disable features, or adjust scaling factors without SSH.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Keys that may be overridden by cloud shadow
CLOUD_OVERRIDABLE = {
    "operating_hours",
    "footfall_scaling_factor",
    "counting_enabled",
    "wifi_ble_enabled",
    "telemetry_interval_seconds",
}


def load_config(path: str) -> dict[str, Any]:
    """Load and validate device configuration from YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated config dict with 'cloud_defaults' as effective cloud config.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config


def merge_cloud_config(config: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    """Merge AWS IoT Device Shadow overrides into config.

    The shadow 'desired' state may contain keys matching CLOUD_OVERRIDABLE.
    These override the corresponding values in config['cloud_defaults'].

    This function does NOT mutate the input config; it returns a new dict.

    Args:
        config: Local config loaded from YAML.
        shadow: IoT Shadow document (the 'state.desired' portion).

    Returns:
        New config dict with cloud overrides applied.
    """
    merged = copy.deepcopy(config)

    if not shadow:
        logger.debug("No shadow data provided; using local defaults")
        return merged

    cloud = merged.setdefault("cloud_defaults", {})
    applied = []

    for key in CLOUD_OVERRIDABLE:
        if key in shadow:
            old_val = cloud.get(key)
            cloud[key] = shadow[key]
            applied.append(f"{key}: {old_val!r} → {shadow[key]!r}")

    if applied:
        logger.info("Cloud config overrides applied: %s", "; ".join(applied))
    else:
        logger.debug("Shadow contained no overridable keys")

    return merged


def get_effective_value(config: dict[str, Any], key: str, fallback: Any = None) -> Any:
    """Get the effective value for a cloud-overridable config key.

    Looks up the key in cloud_defaults first (which may have been overridden
    by shadow merge), then falls back to the provided default.

    Args:
        config: Config dict (after merge_cloud_config).
        key: Key name from CLOUD_OVERRIDABLE.
        fallback: Default if key not found anywhere.

    Returns:
        Effective value.
    """
    cloud = config.get("cloud_defaults", {})
    return cloud.get(key, fallback)


def is_within_operating_hours(config: dict[str, Any], day_name: str, hour: int, minute: int) -> bool:
    """Check if the current time falls within the operating hours for a given day.

    Args:
        config: Config dict with cloud_defaults.operating_hours.
        day_name: Lowercase day name (e.g. "monday").
        hour: Current hour (0-23).
        minute: Current minute (0-59).

    Returns:
        True if within operating hours, False otherwise.
    """
    hours = get_effective_value(config, "operating_hours", {})
    schedule = hours.get(day_name)

    if not schedule:
        return False

    try:
        open_str, close_str = schedule.split("-")
        open_h, open_m = map(int, open_str.strip().split(":"))
        close_h, close_m = map(int, close_str.strip().split(":"))
    except (ValueError, AttributeError):
        logger.warning("Invalid operating hours format for %s: %r", day_name, schedule)
        return True  # Fail open — count if format is broken

    current_minutes = hour * 60 + minute
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m

    return open_minutes <= current_minutes < close_minutes


def is_counting_enabled(config: dict[str, Any]) -> bool:
    """Check if counting is enabled (can be toggled from cloud)."""
    return bool(get_effective_value(config, "counting_enabled", True))


def is_wifi_ble_enabled(config: dict[str, Any]) -> bool:
    """Check if WiFi/BLE probing is enabled (can be toggled from cloud)."""
    local_enabled = config.get("wifi_ble", {}).get("enabled", False)
    cloud_enabled = get_effective_value(config, "wifi_ble_enabled", True)
    return local_enabled and cloud_enabled


def get_scaling_factor(config: dict[str, Any]) -> float:
    """Get the footfall scaling factor (cloud-overridable)."""
    return float(get_effective_value(config, "footfall_scaling_factor", 1.0))


def _validate(config: dict[str, Any]) -> None:
    """Validate required config keys are present."""
    required = ["device", "vision", "detection", "mqtt", "buffer"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    if "id" not in config["device"]:
        raise ValueError("device.id is required")
    if "store_id" not in config["device"]:
        raise ValueError("device.store_id is required")

    # Validate RSSI thresholds if wifi_ble is configured
    wifi_cfg = config.get("wifi_ble", {})
    if wifi_cfg.get("enabled"):
        passerby = wifi_cfg.get("rssi_passerby_threshold", -75)
        shopper = wifi_cfg.get("rssi_shopper_threshold", -55)
        if shopper <= passerby:
            raise ValueError(
                f"rssi_shopper_threshold ({shopper}) must be greater than "
                f"rssi_passerby_threshold ({passerby}) — "
                "shoppers are closer so their signal is stronger (less negative)"
            )

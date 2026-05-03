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
    "on_invalid_schedule",
    "footfall_scaling_factor",
    "counting_enabled",
    "wifi_ble_enabled",
    "telemetry_interval_seconds",
}

VALID_DAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

VALID_INVALID_SCHEDULE_MODES = {"fail_open", "fail_closed"}

# Dotted-path whitelist of config keys that can be applied at runtime from
# an IoT Shadow delta without restarting the service.  Anything outside
# this set (and unknown keys) is logged as "requires restart" and skipped.
RUNTIME_SAFE_KEYS = frozenset(
    {
        "cloud_defaults.operating_hours",
        "cloud_defaults.on_invalid_schedule",
        "cloud_defaults.rssi_passerby",
        "cloud_defaults.rssi_shopper",
        "telemetry.interval_seconds",
        "counter.roi",
        "counter.line",
        "counter.direction_labels",
        # counter.tracker.* handled via prefix below
        # vision.num_disparities / block_size explicit entries
        "vision.num_disparities",
        "vision.block_size",
        # mounting_height_m is runtime-safe: main.py reads it live for the
        # height classifier and for auto num_disparities on SGBM rebuild.
        "vision.mounting_height_m",
        # operational.* handled via prefix below
    }
)

# Prefixes under which any child key is runtime-safe.
RUNTIME_SAFE_PREFIXES = (
    "counter.tracker.",
    "counter.height_classifier.",
    "operational.",
)

SHADOW_CACHE_SUFFIX = ".shadow.json"


def load_config(path: str) -> dict[str, Any]:
    """Load and validate device configuration from YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated config dict with 'cloud_defaults' as effective cloud config.
        If operating_hours fails soft validation, the dict contains a
        '_schedule_error' key describing the problem so runtime can honor
        on_invalid_schedule.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _validate(config)

    # Soft-validate the schedule: errors are surfaced via _schedule_error
    # rather than raising, so the runtime can honor on_invalid_schedule.
    schedule_error = validate_operating_hours(
        get_effective_value(config, "operating_hours", None)
    )
    if schedule_error is not None:
        logger.warning(
            "invalid_operating_hours",
            extra={"reason": schedule_error},
        )
        config["_schedule_error"] = schedule_error

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
        logger.info(
            "cloud_config_overrides_applied",
            extra={"overrides": applied, "count": len(applied)},
        )
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


def validate_operating_hours(hours: Any) -> str | None:
    """Validate an operating_hours structure.

    Returns None if valid, otherwise a human-readable error string describing
    the first problem found. Unknown days are tolerated (extensions), but a
    day present must be either null (closed) or a "HH:MM-HH:MM" string with
    end strictly after start.

    Args:
        hours: The operating_hours value (expected: dict[str, str|None]).

    Returns:
        None if valid, else error message.
    """
    if hours is None:
        return "operating_hours is missing"
    if not isinstance(hours, dict):
        return f"operating_hours must be a mapping, got {type(hours).__name__}"

    for day, schedule in hours.items():
        if schedule is None:
            continue  # closed
        if not isinstance(schedule, str):
            return f"{day}: schedule must be a string or null, got {type(schedule).__name__}"
        if "-" not in schedule:
            return f"{day}: schedule {schedule!r} missing '-' separator"

        parts = schedule.split("-")
        if len(parts) != 2:
            return f"{day}: schedule {schedule!r} must be 'HH:MM-HH:MM'"
        open_str, close_str = parts[0].strip(), parts[1].strip()

        open_parsed = _parse_hhmm(open_str)
        if open_parsed is None:
            return f"{day}: invalid start time {open_str!r}"
        close_parsed = _parse_hhmm(close_str)
        if close_parsed is None:
            return f"{day}: invalid end time {close_str!r}"

        open_minutes = open_parsed[0] * 60 + open_parsed[1]
        close_minutes = close_parsed[0] * 60 + close_parsed[1]
        if close_minutes <= open_minutes:
            return (
                f"{day}: end {close_str!r} must be after start {open_str!r}"
            )

    return None


def _parse_hhmm(value: str) -> tuple[int, int] | None:
    """Parse a HH:MM string. Returns (hour, minute) or None if invalid."""
    if ":" not in value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def get_invalid_schedule_mode(config: dict[str, Any]) -> str:
    """Return the configured on_invalid_schedule mode.

    Falls back to 'fail_open' for unknown or missing values (back-compat).
    """
    mode = get_effective_value(config, "on_invalid_schedule", "fail_open")
    if mode not in VALID_INVALID_SCHEDULE_MODES:
        logger.warning(
            "unknown_on_invalid_schedule_mode",
            extra={"mode": mode, "default": "fail_open"},
        )
        return "fail_open"
    return mode


def has_schedule_error(config: dict[str, Any]) -> bool:
    """Return True if the loaded config flagged an invalid schedule."""
    return bool(config.get("_schedule_error"))


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
        logger.warning(
            "invalid_operating_hours_format",
            extra={"day": day_name, "schedule": schedule},
        )
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


def _flatten_delta(
    delta: dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Flatten a nested delta into ``[(dotted_path, value), ...]``.

    A sub-dict whose dotted path matches a RUNTIME_SAFE_KEYS entry (or a
    safe prefix) is treated as a leaf so the whole sub-tree is applied
    atomically (e.g. ``cloud_defaults.operating_hours`` keeps its shape).
    """
    items: list[tuple[str, Any]] = []
    for key, value in delta.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and not _is_runtime_safe(path):
            # Only recurse if no ancestor path is a whitelisted leaf.
            items.extend(_flatten_delta(value, prefix=f"{path}."))
        else:
            items.append((path, value))
    return items


def _is_runtime_safe(path: str) -> bool:
    """Return True if a dotted-path delta key is runtime-applicable."""
    if path in RUNTIME_SAFE_KEYS:
        return True
    return any(path.startswith(p) for p in RUNTIME_SAFE_PREFIXES)


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    """Assign ``value`` at dotted ``path`` within ``config`` (mutates)."""
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        nxt = target.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            target[part] = nxt
        target = nxt
    target[parts[-1]] = value


def _shadow_cache_path(config_path: str | Path) -> Path:
    """Return the sibling ``.shadow.json`` path for a YAML config file."""
    p = Path(config_path)
    return p.with_suffix(SHADOW_CACHE_SUFFIX)


def apply_shadow_delta(
    current_config: dict[str, Any],
    delta: dict[str, Any],
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply a shadow delta to the running config.

    Only whitelisted (runtime-safe) keys are applied; other keys are
    logged as requiring a restart and skipped.  The returned config is a
    deep copy — the input is not mutated.

    Args:
        current_config: Currently active config dict.
        delta: The ``state`` sub-document from a shadow delta message
            (i.e. the keys whose desired value differs from reported).
        config_path: Optional path to the main YAML config; when provided
            the merged config is persisted to ``<stem>.shadow.json`` so
            the device picks up the latest shadow on next boot.

    Returns:
        Tuple ``(new_config, applied_keys)``.
    """
    new_config = copy.deepcopy(current_config)
    applied: list[str] = []
    ignored: list[str] = []

    if not delta:
        logger.debug("apply_shadow_delta called with empty delta")
        return new_config, applied

    for path, value in _flatten_delta(delta):
        if _is_runtime_safe(path):
            _set_dotted(new_config, path, value)
            applied.append(path)
        else:
            ignored.append(path)

    if applied:
        logger.info(
            "shadow_delta_applied",
            extra={"keys": sorted(applied), "count": len(applied)},
        )
    if ignored:
        logger.warning(
            "shadow_delta_requires_restart",
            extra={"keys": sorted(ignored), "count": len(ignored)},
        )

    if applied and config_path is not None:
        try:
            cache_path = _shadow_cache_path(config_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {"state": {"desired": delta}},
                    indent=2,
                    sort_keys=True,
                )
            )
            logger.debug(
                "shadow_cache_persisted", extra={"path": str(cache_path)}
            )
        except OSError:
            logger.exception("Failed to persist shadow cache")

    return new_config, applied


def _get_dotted(config: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Look up a dotted ``path`` in ``config``.

    Returns ``(found, value)``.  Missing intermediate keys or non-dict
    parents short-circuit to ``(False, None)``.
    """
    parts = path.split(".")
    cur: Any = config
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def build_reported_state(
    config: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract the runtime-visible config subset for shadow reporting.

    The returned dict is suitable to publish as the ``reported`` state of
    an AWS IoT Device Shadow. It contains:

    * The whitelisted runtime-safe config keys (same set that shadow
      delta can modify — see RUNTIME_SAFE_KEYS / RUNTIME_SAFE_PREFIXES).
    * A small metadata block: ``firmware_version`` (from config), a
      ``boot_ts`` snapshot, and calibration provenance
      (``calibration_file_path`` + ``effective_baseline_mm``).

    Keys absent from ``config`` are silently skipped so the shadow doesn't
    carry ``null``-filled noise. Prefix-based keys are walked and every
    descendant leaf is copied into the reported state, preserving the
    original nested structure (e.g. ``counter.tracker.confirm_frames``).

    Args:
        config: Effective config dict (post-shadow-delta merge).
        calibration: Loaded calibration dict (as returned by
            ``load_calibration``) or None if calibration is not loaded.

    Returns:
        New dict ready for ``{"state": {"reported": ...}}``.
    """
    import time as _time

    reported: dict[str, Any] = {}

    # --- Whitelisted scalar / sub-tree keys --------------------------------
    for path in RUNTIME_SAFE_KEYS:
        found, value = _get_dotted(config, path)
        if found:
            _set_dotted(reported, path, value)

    # --- Whitelisted prefixes: copy the whole sub-tree if present ----------
    for prefix in RUNTIME_SAFE_PREFIXES:
        # prefix looks like "counter.tracker." — strip trailing dot
        prefix_path = prefix.rstrip(".")
        found, value = _get_dotted(config, prefix_path)
        if found and isinstance(value, dict):
            _set_dotted(reported, prefix_path, copy.deepcopy(value))

    # --- Metadata ----------------------------------------------------------
    device_cfg = config.get("device", {}) if isinstance(config.get("device"), dict) else {}
    reported["firmware_version"] = device_cfg.get("firmware_version", "unknown")
    reported["boot_ts"] = int(_time.time())

    cal_path: str | None = None
    vision_cfg = config.get("vision", {})
    if isinstance(vision_cfg, dict):
        cal_path = vision_cfg.get("calibration_file")
    reported["calibration_file_path"] = cal_path

    baseline_mm: float | None = None
    if calibration is not None:
        try:
            t_vec = calibration.get("T") if isinstance(calibration, dict) else None
            if t_vec is not None:
                import numpy as _np

                baseline_mm = float(_np.linalg.norm(_np.asarray(t_vec)))
        except Exception:
            logger.exception("Failed to derive baseline from calibration")
            baseline_mm = None
    reported["effective_baseline_mm"] = baseline_mm

    return reported


def _validate(config: dict[str, Any]) -> None:
    """Validate required config keys are present.

    `buffer` is no longer required at the config layer — its defaults live in
    hardware.yaml (db_path/max_age_hours are install conventions). Same for
    most of `tracking`, `mqtt.{port,topics}`, `detection.{thresholds}`.
    `detection` stays required because `model_path` is per-device install.
    """
    required = ["device", "vision", "detection", "mqtt"]
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

"""Tests for src/telemetry.py — device + runtime observability."""
from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from src.telemetry import collect_telemetry


# ---------------------------------------------------------------------------
# Schema — every documented key is always present
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "uptime_s",
    "cpu_temp_c",
    "disk_free_mb",
    "mem_available_mb",
    "hailo_temp_c",
    "frame_latency_p50_ms",
    "frame_latency_p95_ms",
    "detection_rate_per_min",
    "tracker_confirmed_count",
    "tracker_pending_count",
    "mqtt_disconnect_count",
    "seconds_since_last_reconnect",
    "buffer_backlog_messages",
}


def test_collect_telemetry_empty_state_has_all_keys():
    """With no state, all state-dependent fields are present as None."""
    telem = collect_telemetry(None)
    assert EXPECTED_KEYS.issubset(telem.keys())
    # State-derived fields must be None when state is missing.
    assert telem["frame_latency_p50_ms"] is None
    assert telem["frame_latency_p95_ms"] is None
    assert telem["detection_rate_per_min"] is None
    assert telem["tracker_confirmed_count"] is None
    assert telem["tracker_pending_count"] is None
    assert telem["mqtt_disconnect_count"] is None
    assert telem["seconds_since_last_reconnect"] is None
    assert telem["buffer_backlog_messages"] is None


def test_collect_telemetry_empty_dict_state():
    telem = collect_telemetry({})
    assert EXPECTED_KEYS.issubset(telem.keys())


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentile_p50_median_of_known_values():
    state = {"frame_latencies_ms": [10.0, 20.0, 30.0, 40.0, 50.0]}
    telem = collect_telemetry(state)
    # median of 10,20,30,40,50 -> 30
    assert telem["frame_latency_p50_ms"] == 30.0


def test_percentile_p95_nearest_rank():
    # 20 values: p95 rank = ceil(0.95 * 20) = 19 -> value at index 18
    values = [float(i) for i in range(1, 21)]  # 1..20
    state = {"frame_latencies_ms": values}
    telem = collect_telemetry(state)
    # p95 nearest-rank on 1..20 -> 19 (index 18)
    assert telem["frame_latency_p95_ms"] == 19.0


def test_percentile_single_value():
    state = {"frame_latencies_ms": [42.0]}
    telem = collect_telemetry(state)
    assert telem["frame_latency_p50_ms"] == 42.0
    assert telem["frame_latency_p95_ms"] == 42.0


def test_percentile_empty_list_is_none():
    state = {"frame_latencies_ms": []}
    telem = collect_telemetry(state)
    assert telem["frame_latency_p50_ms"] is None
    assert telem["frame_latency_p95_ms"] is None


# ---------------------------------------------------------------------------
# Detection rate
# ---------------------------------------------------------------------------


def test_detection_rate_from_window_start_ts():
    """Rate = sum(counts)/elapsed * 60, using detection_window_start_ts."""
    # Pretend the window started exactly 30 seconds ago.
    start = time.time() - 30.0
    state = {
        "detection_counts": [1, 2, 3, 0, 4],  # sum=10
        "detection_window_start_ts": start,
    }
    telem = collect_telemetry(state)
    # 10 persons / 30s * 60 = 20 persons/min
    assert telem["detection_rate_per_min"] is not None
    assert 19.5 < telem["detection_rate_per_min"] < 20.5


def test_detection_rate_fallback_to_fps():
    """Without start_ts, falls back to len(counts)/fps."""
    state = {
        "detection_counts": [2, 2, 2, 2, 2],  # sum=10, 5 frames
        "fps": 5,  # so elapsed=1s, rate=600/min
    }
    telem = collect_telemetry(state)
    assert telem["detection_rate_per_min"] == pytest.approx(600.0)


def test_detection_rate_empty():
    telem = collect_telemetry({"detection_counts": []})
    assert telem["detection_rate_per_min"] is None


# ---------------------------------------------------------------------------
# Tracker + MQTT + buffer pass-through fields
# ---------------------------------------------------------------------------


def test_state_passthrough_fields():
    now = time.time()
    state = {
        "tracker_confirmed": 3,
        "tracker_pending": 1,
        "mqtt_disconnect_count": 7,
        "mqtt_reconnect_ts": now - 15.0,
        "buffer_backlog": 42,
    }
    telem = collect_telemetry(state)
    assert telem["tracker_confirmed_count"] == 3
    assert telem["tracker_pending_count"] == 1
    assert telem["mqtt_disconnect_count"] == 7
    assert telem["buffer_backlog_messages"] == 42
    assert telem["seconds_since_last_reconnect"] is not None
    assert 14.0 <= telem["seconds_since_last_reconnect"] <= 16.0


def test_reconnect_ts_none_yields_none():
    telem = collect_telemetry({"mqtt_reconnect_ts": None})
    assert telem["seconds_since_last_reconnect"] is None


def test_reconnect_ts_invalid_yields_none():
    telem = collect_telemetry({"mqtt_reconnect_ts": "not-a-number"})
    assert telem["seconds_since_last_reconnect"] is None


# ---------------------------------------------------------------------------
# Hailo probe — mocked subprocess
# ---------------------------------------------------------------------------


def test_hailo_probe_parses_die_temperature():
    fake_output = (
        "HailoRT Information\n"
        "Firmware version: 4.23.0\n"
        "Die Temperature: 45.3 C\n"
        "Board serial: 123456\n"
    )
    completed = MagicMock()
    completed.stdout = fake_output
    completed.stderr = ""
    completed.returncode = 0
    with patch("src.telemetry.subprocess.run", return_value=completed):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] == pytest.approx(45.3)


def test_hailo_probe_parses_integer_temperature():
    completed = MagicMock()
    completed.stdout = "Chip temperature: 52 C"
    completed.stderr = ""
    completed.returncode = 0
    with patch("src.telemetry.subprocess.run", return_value=completed):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] == pytest.approx(52.0)


def test_hailo_probe_not_installed_yields_none():
    with patch(
        "src.telemetry.subprocess.run",
        side_effect=FileNotFoundError("hailortcli"),
    ):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None
    # Key MUST still be present.
    assert "hailo_temp_c" in telem


def test_hailo_probe_timeout_yields_none():
    with patch(
        "src.telemetry.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="hailortcli", timeout=5),
    ):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None


def test_hailo_probe_unexpected_exception_yields_none():
    with patch(
        "src.telemetry.subprocess.run",
        side_effect=RuntimeError("boom"),
    ):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None


def test_hailo_probe_no_match_in_output_yields_none():
    completed = MagicMock()
    completed.stdout = "Firmware: ok\nBoard serial: 123"
    completed.stderr = ""
    completed.returncode = 0
    with patch("src.telemetry.subprocess.run", return_value=completed):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None


# ---------------------------------------------------------------------------
# Robustness — malformed state doesn't crash
# ---------------------------------------------------------------------------


def test_non_iterable_frame_latencies_does_not_raise():
    telem = collect_telemetry({"frame_latencies_ms": 123})
    assert telem["frame_latency_p50_ms"] is None
    assert telem["frame_latency_p95_ms"] is None


def test_function_never_raises_on_broken_state():
    # All values deliberately wrong type.
    state = {
        "frame_latencies_ms": object(),
        "detection_counts": object(),
        "detection_window_start_ts": object(),
        "tracker_confirmed": object(),
        "mqtt_reconnect_ts": object(),
    }
    telem = collect_telemetry(state)
    assert EXPECTED_KEYS.issubset(telem.keys())

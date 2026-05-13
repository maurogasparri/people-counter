"""Tests para src/telemetry.py — observability del device + runtime."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.telemetry import collect_telemetry


def _fake_hailo_module(
    ts0_temp: float, ts1_temp: float, *, raise_on: str | None = None
):
    """Construye un MagicMock con shape del módulo ``hailo_platform``
    para inyectar en ``sys.modules``. ``raise_on``:
        ``None`` -> happy path, devuelve ts0/ts1 del TemperatureInfo.
        ``"device"`` -> ``hp.Device()`` raisea RuntimeError.
        ``"temp"``  -> ``device.control.get_chip_temperature()`` raisea.
    """
    mod = MagicMock()
    if raise_on == "device":
        mod.Device.side_effect = RuntimeError("device init failed")
        return mod
    device = MagicMock()
    if raise_on == "temp":
        device.control.get_chip_temperature.side_effect = RuntimeError(
            "temp read failed"
        )
    else:
        info = MagicMock()
        info.ts0_temperature = ts0_temp
        info.ts1_temperature = ts1_temp
        device.control.get_chip_temperature.return_value = info
    mod.Device.return_value = device
    return mod


# ---------------------------------------------------------------------------
# Schema — toda key documentada siempre está presente
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
# Hailo probe — mocked vía hailo_platform SDK (post-b2e4112)
# ---------------------------------------------------------------------------
#
# El probe usa ``hailo_platform.Device().control.get_chip_temperature()``
# que devuelve un ``TemperatureInfo`` con ``ts0_temperature`` /
# ``ts1_temperature`` (los dos sensores del die). Tomamos el max como
# métrica conservadora (eso es lo que dispara el throttling primero).
#
# Inyectamos el módulo fake en ``sys.modules`` así el ``import
# hailo_platform`` interno del telemetry lo levanta. ``patch.dict`` cierra
# bien post-test para no contaminar otros suites.


def test_hailo_probe_returns_max_of_two_sensors():
    """ts1 > ts0 → reporta ts1 (el más caliente)."""
    fake = _fake_hailo_module(45.3, 50.1)
    with patch.dict("sys.modules", {"hailo_platform": fake}):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] == pytest.approx(50.1)


def test_hailo_probe_max_picks_ts0_when_hotter():
    """Caso inverso: ts0 > ts1 → reporta ts0. Asegura que la lógica
    no está hardcoded a un sensor."""
    fake = _fake_hailo_module(58.7, 42.0)
    with patch.dict("sys.modules", {"hailo_platform": fake}):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] == pytest.approx(58.7)


def test_hailo_probe_handles_integer_temperatures():
    """Los sensores pueden reportar ints o floats — el ``float()`` cast
    interno tiene que normalizar a float."""
    fake = _fake_hailo_module(52, 48)
    with patch.dict("sys.modules", {"hailo_platform": fake}):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] == pytest.approx(52.0)
    assert isinstance(telem["hailo_temp_c"], float)


def test_hailo_probe_not_installed_yields_none():
    """``hailo_platform`` no instalado (Windows dev, devices sin Hailo)
    → import falla → None. Inyectamos ``None`` en ``sys.modules`` que
    fuerza ImportError en el próximo ``import hailo_platform``."""
    with patch.dict("sys.modules", {"hailo_platform": None}):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None
    # Key MUST still be present.
    assert "hailo_temp_c" in telem


def test_hailo_probe_device_construction_fails_yields_none():
    """``hp.Device()`` raisea (Hailo no detectado, permisos, etc.) →
    catch interno → None."""
    fake = _fake_hailo_module(0.0, 0.0, raise_on="device")
    with patch.dict("sys.modules", {"hailo_platform": fake}):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None


def test_hailo_probe_get_temperature_raises_yields_none():
    """``get_chip_temperature()`` raisea (firmware viejo, chip wedged) →
    None, sin afectar el resto del payload de telemetría."""
    fake = _fake_hailo_module(0.0, 0.0, raise_on="temp")
    with patch.dict("sys.modules", {"hailo_platform": fake}):
        telem = collect_telemetry({})
    assert telem["hailo_temp_c"] is None
    # Otras keys de telemetría siguen presentes.
    assert EXPECTED_KEYS.issubset(telem.keys())


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

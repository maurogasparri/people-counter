"""Tests for WiFi probe request capture module."""
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.wifi_ble.wifi_probe import (
    CHANNELS_24GHZ,
    CHANNELS_5GHZ,
    DEFAULT_HOP_INTERVAL,
    PROBE_REQUEST_SUBTYPE,
    ProbeEvent,
    WiFiProbeCapture,
)


# ---------------------------------------------------------------------------
# ProbeEvent
# ---------------------------------------------------------------------------


def test_probe_event_fields():
    event = ProbeEvent(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-65.0,
        ssid="TestNetwork",
        channel=6,
        timestamp=1000.0,
    )
    assert event.mac == "AA:BB:CC:DD:EE:FF"
    assert event.rssi == -65.0
    assert event.ssid == "TestNetwork"
    assert event.channel == 6


# ---------------------------------------------------------------------------
# WiFiProbeCapture construction
# ---------------------------------------------------------------------------


def test_default_channels():
    cap = WiFiProbeCapture(interface="wlan0")
    assert cap.channels == CHANNELS_24GHZ + CHANNELS_5GHZ
    assert cap.hop_interval == DEFAULT_HOP_INTERVAL


def test_custom_channels():
    cap = WiFiProbeCapture(
        channels_24=[1, 6, 11],
        channels_5=[36, 40],
    )
    assert cap.channels == [1, 6, 11, 36, 40]


def test_initial_state():
    cap = WiFiProbeCapture()
    assert cap.probe_count == 0
    assert cap._capture_thread is None
    assert cap._hop_thread is None
    assert cap.interface == "wlan0"
    assert cap.mon_interface == "wlan0mon"


# ---------------------------------------------------------------------------
# Monitor mode setup (airmon-ng)
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_calls_airmon(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    cap = WiFiProbeCapture(interface="wlan0")
    cap.setup_monitor_mode()

    calls = mock_run.call_args_list
    # Should call: airmon-ng check kill, airmon-ng start wlan0, iw dev wlan0mon info
    assert any("airmon-ng" in str(c) and "kill" in str(c) for c in calls)
    assert any("airmon-ng" in str(c) and "start" in str(c) for c in calls)
    assert any("iw" in str(c) and "wlan0mon" in str(c) for c in calls)


@patch("src.wifi_ble.wifi_probe.subprocess.run", side_effect=FileNotFoundError("airmon-ng"))
def test_setup_monitor_mode_missing_tool(mock_run):
    cap = WiFiProbeCapture()
    with pytest.raises(RuntimeError, match="Required tool not found"):
        cap.setup_monitor_mode()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_teardown_monitor_mode(mock_run):
    cap = WiFiProbeCapture(interface="wlan0")
    cap.teardown_monitor_mode()

    calls = mock_run.call_args_list
    assert any("airmon-ng" in str(c) and "stop" in str(c) for c in calls)


@patch("src.wifi_ble.wifi_probe.subprocess.run", side_effect=Exception("fail"))
def test_teardown_graceful_on_error(mock_run):
    """Teardown should not raise even if commands fail."""
    cap = WiFiProbeCapture()
    cap.teardown_monitor_mode()  # should not raise


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_start_creates_threads(mock_run):
    cap = WiFiProbeCapture()

    with patch.dict("sys.modules", {"scapy": MagicMock(), "scapy.all": MagicMock()}):
        cap.start()
        assert cap._capture_thread is not None
        assert cap._hop_thread is not None
        cap.stop()
        assert cap._capture_thread is None
        assert cap._hop_thread is None


def test_start_twice_warns(caplog):
    cap = WiFiProbeCapture()
    cap._capture_thread = threading.Thread()

    import logging

    with caplog.at_level(logging.WARNING):
        cap.start()

    assert "already running" in caplog.text


# ---------------------------------------------------------------------------
# Channel hopping
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_channel_hop_uses_mon_interface(mock_run):
    """Channel hop should use wlan0mon, not wlan0."""
    cap = WiFiProbeCapture(
        interface="wlan0", channels_24=[1, 6], channels_5=[], hop_interval=0.01
    )
    cap._stop_event.clear()

    hop_thread = threading.Thread(target=cap._channel_hop_loop, daemon=True)
    hop_thread.start()
    time.sleep(0.1)
    cap._stop_event.set()
    hop_thread.join(timeout=1.0)

    assert mock_run.call_count > 0
    # All channel set calls should use wlan0mon
    for c in mock_run.call_args_list:
        args = c[0][0]
        if "channel" in args:
            assert "wlan0mon" in args


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def test_on_probe_callback():
    events = []
    cap = WiFiProbeCapture(on_probe=lambda e: events.append(e))

    event = ProbeEvent(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-60.0,
        ssid="Test",
        channel=6,
        timestamp=time.time(),
    )
    cap.on_probe(event)
    assert len(events) == 1
    assert events[0].mac == "AA:BB:CC:DD:EE:FF"


def test_probe_count_starts_zero():
    cap = WiFiProbeCapture()
    assert cap.probe_count == 0

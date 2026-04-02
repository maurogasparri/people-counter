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


# ---------------------------------------------------------------------------
# Monitor mode setup
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_calls_ip_iw(mock_run):
    cap = WiFiProbeCapture(interface="wlan0")
    cap.setup_monitor_mode()

    assert mock_run.call_count == 3
    # Down, set monitor, up
    calls = mock_run.call_args_list
    assert calls[0][0][0] == ["ip", "link", "set", "wlan0", "down"]
    assert calls[1][0][0] == ["iw", "wlan0", "set", "monitor", "none"]
    assert calls[2][0][0] == ["ip", "link", "set", "wlan0", "up"]


@patch("src.wifi_ble.wifi_probe.subprocess.run", side_effect=FileNotFoundError("iw"))
def test_setup_monitor_mode_missing_tool(mock_run):
    cap = WiFiProbeCapture()
    with pytest.raises(RuntimeError, match="Required tool not found"):
        cap.setup_monitor_mode()


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_command_failure(mock_run):
    import subprocess

    mock_run.side_effect = subprocess.CalledProcessError(
        1, "iw", stderr=b"Operation not permitted"
    )
    cap = WiFiProbeCapture()
    with pytest.raises(RuntimeError, match="Failed to set monitor mode"):
        cap.setup_monitor_mode()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_teardown_monitor_mode(mock_run):
    cap = WiFiProbeCapture(interface="wlan0")
    cap.teardown_monitor_mode()

    assert mock_run.call_count == 3
    calls = mock_run.call_args_list
    assert calls[1][0][0] == ["iw", "wlan0", "set", "type", "managed"]


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

    # Mock scapy import to prevent actual sniffing
    with patch.dict("sys.modules", {"scapy": MagicMock(), "scapy.all": MagicMock()}):
        cap.start()
        assert cap._capture_thread is not None
        assert cap._hop_thread is not None
        cap.stop()
        assert cap._capture_thread is None
        assert cap._hop_thread is None


def test_start_twice_warns(caplog):
    """Starting twice should warn, not create duplicate threads."""
    cap = WiFiProbeCapture()
    cap._capture_thread = threading.Thread()  # fake existing thread
    cap._stop_event.clear()

    import logging

    with caplog.at_level(logging.WARNING):
        cap.start()

    assert "already running" in caplog.text


# ---------------------------------------------------------------------------
# Channel hopping
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_channel_hop_cycles(mock_run):
    """Channel hop should cycle through all channels."""
    cap = WiFiProbeCapture(
        channels_24=[1, 6], channels_5=[], hop_interval=0.01
    )
    cap._stop_event.clear()

    # Run hop loop briefly
    hop_thread = threading.Thread(target=cap._channel_hop_loop, daemon=True)
    hop_thread.start()
    time.sleep(0.1)
    cap._stop_event.set()
    hop_thread.join(timeout=1.0)

    # Should have called iw set channel multiple times
    assert mock_run.call_count > 0
    # Check that channel 1 and 6 were both used
    channel_args = [
        c[0][0][-1] for c in mock_run.call_args_list
    ]
    assert "1" in channel_args
    assert "6" in channel_args


# ---------------------------------------------------------------------------
# Callback invocation
# ---------------------------------------------------------------------------


def test_on_probe_callback():
    """on_probe callback should be called for each probe event."""
    events = []
    cap = WiFiProbeCapture(on_probe=lambda e: events.append(e))

    # Simulate a probe being processed
    event = ProbeEvent(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-60.0,
        ssid="Test",
        channel=6,
        timestamp=time.time(),
    )
    cap._probe_count += 1
    cap.on_probe(event)

    assert len(events) == 1
    assert events[0].mac == "AA:BB:CC:DD:EE:FF"


def test_probe_count_starts_zero():
    cap = WiFiProbeCapture()
    assert cap.probe_count == 0

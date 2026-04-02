"""Tests for BLE advertising capture module."""
import struct
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.wifi_ble.ble_scan import (
    ADV_IND,
    ADV_NONCONN_IND,
    BLEAdvertisement,
    BLEScanner,
)


# ---------------------------------------------------------------------------
# BLEAdvertisement
# ---------------------------------------------------------------------------


def test_ble_advertisement_fields():
    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-70.0,
        adv_type=ADV_IND,
        name="MyDevice",
        timestamp=1000.0,
    )
    assert advert.mac == "AA:BB:CC:DD:EE:FF"
    assert advert.rssi == -70.0
    assert advert.adv_type == ADV_IND
    assert advert.name == "MyDevice"


def test_ble_advertisement_no_name():
    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-70.0,
        adv_type=ADV_NONCONN_IND,
        name=None,
        timestamp=1000.0,
    )
    assert advert.name is None


# ---------------------------------------------------------------------------
# BLEScanner construction
# ---------------------------------------------------------------------------


def test_scanner_defaults():
    scanner = BLEScanner()
    assert scanner.hci_device == "hci0"
    assert scanner.scan_window == 10.0
    assert scanner.advert_count == 0


def test_scanner_custom_params():
    scanner = BLEScanner(hci_device="hci1", scan_window_seconds=5.0)
    assert scanner.hci_device == "hci1"
    assert scanner.scan_window == 5.0


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.ble_scan.subprocess.run")
def test_start_enables_adapter(mock_run):
    scanner = BLEScanner()

    # Patch _scan_loop to not actually scan
    with patch.object(scanner, "_scan_loop"):
        scanner.start()
        # First call should enable the HCI device
        mock_run.assert_called_with(
            ["hciconfig", "hci0", "up"],
            check=True,
            capture_output=True,
        )
        scanner.stop()


@patch("src.wifi_ble.ble_scan.subprocess.run", side_effect=FileNotFoundError("hciconfig"))
def test_start_missing_hciconfig(mock_run, caplog):
    """Should log error and return without starting if hciconfig is missing."""
    import logging

    scanner = BLEScanner()
    with caplog.at_level(logging.ERROR):
        scanner.start()

    assert "hciconfig not found" in caplog.text
    assert scanner._scan_thread is None


def test_start_twice_warns(caplog):
    import logging
    import threading

    scanner = BLEScanner()
    scanner._scan_thread = threading.Thread()  # fake

    with caplog.at_level(logging.WARNING):
        scanner.start()

    assert "already running" in caplog.text


@patch("src.wifi_ble.ble_scan.subprocess.run")
def test_stop_disables_scan(mock_run):
    scanner = BLEScanner()
    scanner._stop_event.clear()
    scanner.stop()
    assert scanner._stop_event.is_set()


# ---------------------------------------------------------------------------
# HCI event parsing
# ---------------------------------------------------------------------------


def _build_hci_le_adv_report(
    mac: str = "AA:BB:CC:DD:EE:FF",
    rssi: int = -65,
    adv_type: int = 0x00,
    adv_data: bytes = b"",
) -> bytes:
    """Build a fake HCI LE Advertising Report event."""
    # HCI event header
    event_type = 0x04
    event_code = 0x3E  # LE Meta Event
    subevent = 0x02  # Advertising Report
    num_reports = 1
    addr_type = 0x00  # public

    # Parse MAC
    mac_bytes = bytes(reversed([int(b, 16) for b in mac.split(":")]))

    # Build report
    report = bytes([adv_type, addr_type]) + mac_bytes
    report += bytes([len(adv_data)]) + adv_data
    report += struct.pack("b", rssi)

    # Full event
    param_data = bytes([subevent, num_reports]) + report
    event = bytes([event_type, event_code, len(param_data)]) + param_data
    return event


def test_parse_hci_event_basic():
    events = []
    scanner = BLEScanner(on_advert=lambda e: events.append(e))

    data = _build_hci_le_adv_report(
        mac="AA:BB:CC:DD:EE:FF", rssi=-65, adv_type=ADV_IND
    )
    scanner._parse_hci_event(data)

    assert len(events) == 1
    assert events[0].mac == "AA:BB:CC:DD:EE:FF"
    assert events[0].rssi == -65.0
    assert events[0].adv_type == ADV_IND
    assert scanner.advert_count == 1


def test_parse_hci_event_with_name():
    events = []
    scanner = BLEScanner(on_advert=lambda e: events.append(e))

    # Build advertising data with Complete Local Name (type 0x09)
    name_bytes = b"TestDev"
    adv_data = bytes([len(name_bytes) + 1, 0x09]) + name_bytes

    data = _build_hci_le_adv_report(
        mac="11:22:33:44:55:66", rssi=-72, adv_data=adv_data
    )
    scanner._parse_hci_event(data)

    assert len(events) == 1
    assert events[0].name == "TestDev"


def test_parse_hci_event_negative_rssi():
    events = []
    scanner = BLEScanner(on_advert=lambda e: events.append(e))

    data = _build_hci_le_adv_report(rssi=-90)
    scanner._parse_hci_event(data)

    assert events[0].rssi == -90.0


def test_parse_hci_event_too_short():
    """Short data should be silently ignored."""
    scanner = BLEScanner()
    scanner._parse_hci_event(b"\x04\x3e")  # too short
    assert scanner.advert_count == 0


def test_parse_hci_event_wrong_event_code():
    """Non-LE Meta events should be ignored."""
    scanner = BLEScanner()
    scanner._parse_hci_event(b"\x04\x0e\x04\x00\x00\x00\x00")
    assert scanner.advert_count == 0


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------


def test_extract_name_complete():
    scanner = BLEScanner()
    # Type 0x09 = Complete Local Name
    adv_data = bytes([6, 0x09]) + b"Hello"
    assert scanner._extract_name(adv_data) == "Hello"


def test_extract_name_shortened():
    scanner = BLEScanner()
    # Type 0x08 = Shortened Local Name
    adv_data = bytes([4, 0x08]) + b"Hi!"
    assert scanner._extract_name(adv_data) == "Hi!"


def test_extract_name_none():
    scanner = BLEScanner()
    # Type 0x01 = Flags, no name
    adv_data = bytes([2, 0x01, 0x06])
    assert scanner._extract_name(adv_data) is None


def test_extract_name_empty():
    scanner = BLEScanner()
    assert scanner._extract_name(b"") is None


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def test_on_advert_callback():
    events = []
    scanner = BLEScanner(on_advert=lambda e: events.append(e))

    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-55.0,
        adv_type=ADV_IND,
        name=None,
        timestamp=time.time(),
    )
    scanner.on_advert(advert)
    assert len(events) == 1


def test_on_advert_callback_error_handled():
    """Errors in callback should be logged, not propagated."""
    def bad_callback(e):
        raise ValueError("boom")

    scanner = BLEScanner(on_advert=bad_callback)

    data = _build_hci_le_adv_report()
    # Should not raise
    scanner._parse_hci_event(data)
    assert scanner.advert_count == 1

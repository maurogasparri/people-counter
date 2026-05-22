"""Tests del fingerprinting WiFi/BLE (estable a la rotación de identificador)."""

from src.wifi_ble.fingerprint import ble_fingerprint, wifi_fingerprint


# --- WiFi ---

def test_wifi_fingerprint_stable_across_ssid_and_order_value():
    # Mismos IEs (rates + HT cap), distinto SSID -> MISMO fingerprint (el valor
    # del SSID se excluye porque cambia por red).
    base = [(0, b"homewifi"), (1, b"\x02\x04\x0b\x16"), (45, b"\xad\x01\x02")]
    other_ssid = [(0, b"cafe-guest"), (1, b"\x02\x04\x0b\x16"), (45, b"\xad\x01\x02")]
    assert wifi_fingerprint(base) == wifi_fingerprint(other_ssid)
    assert wifi_fingerprint(base) != ""


def test_wifi_fingerprint_differs_on_capabilities():
    a = [(0, b"x"), (1, b"\x02\x04"), (45, b"\xad\x01")]
    b = [(0, b"x"), (1, b"\x02\x04"), (45, b"\xff\x00")]  # HT cap distinto
    assert wifi_fingerprint(a) != wifi_fingerprint(b)


def test_wifi_fingerprint_differs_on_ie_order():
    a = [(1, b"\x02"), (45, b"\xad"), (191, b"\x00")]
    b = [(45, b"\xad"), (1, b"\x02"), (191, b"\x00")]  # otro orden de tags
    assert wifi_fingerprint(a) != wifi_fingerprint(b)


def test_wifi_fingerprint_empty():
    assert wifi_fingerprint([]) == ""


# --- BLE ---

def test_ble_fingerprint_apple_continuity_subtypes():
    # Apple (0x4C): mismos subtipos Continuity con payload distinto -> MISMO fp
    # (se extraen los TLV *type*, no los bytes volátiles).
    md1 = {0x004C: bytes([0x10, 0x02, 0xAA, 0xBB])}  # type 0x10, len 2
    md2 = {0x004C: bytes([0x10, 0x02, 0xCC, 0xDD])}  # mismo type, payload distinto
    assert ble_fingerprint(md1, [], None) == ble_fingerprint(md2, [], None)
    # type distinto -> fp distinto
    md3 = {0x004C: bytes([0x0C, 0x02, 0xAA, 0xBB])}
    assert ble_fingerprint(md1, [], None) != ble_fingerprint(md3, [], None)


def test_ble_fingerprint_uses_company_and_uuids_and_tx():
    fp1 = ble_fingerprint({0x0075: b"\x01\x02"}, ["180f"], -59)
    fp2 = ble_fingerprint({0x0075: b"\x01\x02"}, ["180f"], -59)
    assert fp1 == fp2 != ""
    # distinto company / uuid / tx -> distinto fp
    assert fp1 != ble_fingerprint({0x0006: b"\x01\x02"}, ["180f"], -59)
    assert fp1 != ble_fingerprint({0x0075: b"\x01\x02"}, ["180a"], -59)
    assert fp1 != ble_fingerprint({0x0075: b"\x01\x02"}, ["180f"], -40)


def test_ble_fingerprint_empty():
    assert ble_fingerprint(None, None, None) == ""
    assert ble_fingerprint({}, [], None) == ""

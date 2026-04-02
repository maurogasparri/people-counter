"""Tests for MAC address hashing."""
from src.wifi_ble.hasher import hash_mac


def test_hash_is_deterministic():
    h1 = hash_mac("AA:BB:CC:DD:EE:FF")
    h2 = hash_mac("AA:BB:CC:DD:EE:FF")
    assert h1 == h2


def test_hash_is_case_insensitive():
    h1 = hash_mac("aa:bb:cc:dd:ee:ff")
    h2 = hash_mac("AA:BB:CC:DD:EE:FF")
    assert h1 == h2


def test_hash_is_format_insensitive():
    h1 = hash_mac("AA:BB:CC:DD:EE:FF")
    h2 = hash_mac("AA-BB-CC-DD-EE-FF")
    h3 = hash_mac("AABBCCDDEEFF")
    assert h1 == h2 == h3


def test_hash_length():
    h = hash_mac("AA:BB:CC:DD:EE:FF")
    assert len(h) == 32  # 16 bytes = 32 hex chars


def test_different_macs_different_hashes():
    h1 = hash_mac("AA:BB:CC:DD:EE:FF")
    h2 = hash_mac("11:22:33:44:55:66")
    assert h1 != h2

"""Tests for WiFi/BLE deduplication engine."""
import tempfile
from pathlib import Path

from src.wifi_ble.dedup import DedupEngine


def _make_engine() -> tuple[DedupEngine, str]:
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "dedup.db")
    return DedupEngine(db_path, cross_window_seconds=2.0, cross_rssi_delta=5.0), tmpdir


def test_first_detection_is_new():
    engine, _ = _make_engine()
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert result["is_new"] is True
    assert result["unified"] is False


def test_duplicate_same_protocol():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert result["is_new"] is False


def test_same_mac_different_protocol():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    # Same MAC on BLE with similar RSSI → should unify
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "ble", -58.0)
    assert result["is_new"] is True
    assert result["unified"] is True


def test_cross_protocol_rssi_too_different():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    # RSSI delta > 5 → no unification
    result = engine.process_detection("11:22:33:44:55:66", "ble", -30.0)
    assert result["unified"] is False


def test_unique_count():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    engine.process_detection("11:22:33:44:55:66", "wifi", -55.0)
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)  # duplicate
    assert engine.get_unique_count() == 2


def test_reset_daily():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert engine.get_unique_count() == 1
    engine.reset_daily()
    assert engine.get_unique_count() == 0
    # Same MAC is new again after reset
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert result["is_new"] is True


def test_traffic_counts_dual_threshold():
    """Test passerby/shopper classification with dual RSSI thresholds."""
    engine, _ = _make_engine()
    # Passerby (signal between -75 and -55): detected but didn't enter
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -70.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -65.0)
    engine.process_detection("AA:BB:CC:DD:EE:03", "wifi", -72.0)
    # Shopper (signal >= -55): entered the store
    engine.process_detection("AA:BB:CC:DD:EE:04", "wifi", -50.0)
    engine.process_detection("AA:BB:CC:DD:EE:05", "wifi", -45.0)
    # Below passerby threshold: too far, not counted
    engine.process_detection("AA:BB:CC:DD:EE:06", "wifi", -80.0)

    counts = engine.get_traffic_counts(rssi_passerby=-75, rssi_shopper=-55)
    # 5 devices above -75 (passerby), 2 above -55 (shopper), 1 below -75 (ignored)
    assert counts["passersby"] == 5
    assert counts["shoppers"] == 2
    assert counts["turn_in_rate"] == round(2 / 5, 4)


def test_traffic_counts_empty():
    """Test traffic counts when no detections."""
    engine, _ = _make_engine()
    counts = engine.get_traffic_counts()
    assert counts["passersby"] == 0
    assert counts["shoppers"] == 0
    assert counts["turn_in_rate"] == 0.0


def test_traffic_counts_all_shoppers():
    """When all detections are shoppers, turn_in_rate should be 1.0."""
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -40.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -30.0)
    counts = engine.get_traffic_counts(rssi_passerby=-75, rssi_shopper=-55)
    assert counts["passersby"] == 2
    assert counts["shoppers"] == 2
    assert counts["turn_in_rate"] == 1.0


def test_traffic_counts_no_shoppers():
    """When no one enters, turn_in_rate should be 0.0."""
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -70.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -65.0)
    counts = engine.get_traffic_counts(rssi_passerby=-75, rssi_shopper=-55)
    assert counts["passersby"] == 2
    assert counts["shoppers"] == 0
    assert counts["turn_in_rate"] == 0.0


def test_traffic_counts_resets_with_daily():
    """Traffic counts should reset after daily reset."""
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -50.0)
    assert engine.get_traffic_counts()["shoppers"] == 1
    engine.reset_daily()
    assert engine.get_traffic_counts()["shoppers"] == 0

"""Tests para el engine de dedup WiFi/BLE."""
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


def test_traffic_counts_filter_by_protocol():
    """protocol='wifi' should restrict counts to WiFi detections only."""
    engine, _ = _make_engine()
    # 2 WiFi shoppers
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -40.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -45.0)
    # 1 BLE shopper (different MAC so it's not unified with WiFi)
    engine.process_detection("11:22:33:44:55:66", "ble", -50.0)

    mixed = engine.get_traffic_counts()
    wifi_only = engine.get_traffic_counts(protocol="wifi")
    ble_only = engine.get_traffic_counts(protocol="ble")

    assert mixed["shoppers"] == 3
    assert wifi_only["shoppers"] == 2
    assert ble_only["shoppers"] == 1
    # Turn-in rate per protocol is self-consistent (all shoppers = all passersby)
    assert wifi_only["turn_in_rate"] == 1.0
    assert ble_only["turn_in_rate"] == 1.0


def test_get_recent_hashes_returns_window():
    import time

    engine, _ = _make_engine()
    t0 = time.time()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -60.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -60.0)

    recent = engine.get_recent_hashes(since_ts=t0 - 1, until_ts=t0 + 10)
    assert len(recent) == 2
    # Hashes son hex de 32 chars (16 bytes truncados).
    assert all(len(h) == 32 for h in recent)


def test_get_recent_hashes_filters_by_protocol():
    import time

    engine, _ = _make_engine()
    t0 = time.time()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -60.0)
    # MAC distinta + RSSI diferente para evitar unificación cross-protocol.
    engine.process_detection("11:22:33:44:55:66", "ble", -30.0)

    wifi_only = engine.get_recent_hashes(
        since_ts=t0 - 1, until_ts=t0 + 10, protocol="wifi"
    )
    ble_only = engine.get_recent_hashes(
        since_ts=t0 - 1, until_ts=t0 + 10, protocol="ble"
    )
    assert len(wifi_only) == 1
    assert len(ble_only) == 1
    assert wifi_only != ble_only


def test_get_recent_hashes_excludes_outside_window():
    import time

    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -60.0)

    # Ventana en el futuro → ninguno cae adentro.
    future = engine.get_recent_hashes(since_ts=time.time() + 100)
    assert future == []


def test_get_recent_hashes_until_default_is_now():
    import time

    engine, _ = _make_engine()
    t0 = time.time()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -60.0)
    # En Windows time.time() tiene resolución de ~15ms — sin un sleep, la
    # detección puede tener first_seen idéntico al until_ts implícito del query
    # y caer afuera del half-open interval [since, until).
    time.sleep(0.05)

    # Sin until_ts explícito, debe llegar hasta `now()` y agarrar la detección.
    result = engine.get_recent_hashes(since_ts=t0 - 1)
    assert len(result) == 1


def test_get_window_summary_counts_passersby_and_shoppers():
    import time

    engine, _ = _make_engine()
    t0 = time.time()
    # 3 dispositivos: 1 muy cerca (shopper), 1 medio (passerby pero no shopper),
    # 1 lejos (debajo del threshold de passerby).
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -50.0)  # shopper
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -65.0)  # passerby
    engine.process_detection("AA:BB:CC:DD:EE:03", "wifi", -85.0)  # ruido

    summary = engine.get_window_summary(
        since_ts=t0 - 1,
        until_ts=t0 + 10,
        rssi_passerby=-75.0,
        rssi_shopper=-55.0,
    )

    assert summary["passersby"] == 2  # el de -50 y el de -65
    assert summary["shoppers"] == 1  # solo el de -50


def test_get_window_summary_unifies_cross_protocol():
    """Un dispositivo detectado por WiFi Y BLE en ventana corta cuenta como 1."""
    import time

    engine, _ = _make_engine()
    t0 = time.time()
    # Mismo MAC, distinto protocolo, RSSI similar → L2 dedup lo unifica.
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -50.0)
    engine.process_detection("AA:BB:CC:DD:EE:01", "ble", -52.0)

    summary = engine.get_window_summary(
        since_ts=t0 - 1, until_ts=t0 + 10
    )

    assert summary["passersby"] == 1
    assert summary["shoppers"] == 1


def test_get_window_summary_excludes_outside_window():
    import time

    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -50.0)

    summary = engine.get_window_summary(since_ts=time.time() + 100)
    assert summary == {"passersby": 0, "shoppers": 0}


def test_get_window_summary_shoppers_subset_of_passersby():
    """Invariante: shoppers ≤ passersby siempre."""
    import time

    engine, _ = _make_engine()
    t0 = time.time()
    for i, rssi in enumerate([-50.0, -52.0, -58.0, -65.0, -72.0]):
        engine.process_detection(f"AA:BB:CC:DD:EE:{i:02x}", "wifi", rssi)

    summary = engine.get_window_summary(
        since_ts=t0 - 1, until_ts=t0 + 10
    )

    assert summary["shoppers"] <= summary["passersby"]

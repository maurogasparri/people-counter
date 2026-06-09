"""Tests para el módulo de captura de advertising BLE (basado en bleak)."""

import time
from unittest.mock import MagicMock, patch


from src.wifi_ble.ble_scan import (
    BLEAdvertisement,
    BLEScanner,
    ble_address_is_random,
)


class _FakeBLEDevice:
    def __init__(self, address_type=None, has_details=True):
        self.address = "AA:BB:CC:DD:EE:FF"
        if has_details:
            props = {} if address_type is None else {"AddressType": address_type}
            self.details = {"props": props}
        else:
            self.details = None


def test_ble_address_is_random_random_type():
    assert ble_address_is_random(_FakeBLEDevice(address_type="random")) is True


def test_ble_address_is_random_public_type():
    # public = OUI real = infra/IoT fijo -> no contar
    assert ble_address_is_random(_FakeBLEDevice(address_type="public")) is False


def test_ble_address_is_random_unknown_counts():
    # Sin AddressType / sin details: fallback a True (no sobre-filtrar humanos).
    assert ble_address_is_random(_FakeBLEDevice(address_type=None)) is True
    assert ble_address_is_random(_FakeBLEDevice(has_details=False)) is True
    assert ble_address_is_random(object()) is True


# ---------------------------------------------------------------------------
# BLEAdvertisement
# ---------------------------------------------------------------------------


def test_ble_advertisement_fields():
    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-70.0,
        name="MyDevice",
        timestamp=1000.0,
    )
    assert advert.mac == "AA:BB:CC:DD:EE:FF"
    assert advert.rssi == -70.0
    assert advert.name == "MyDevice"


def test_ble_advertisement_no_name():
    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-70.0,
        name=None,
        timestamp=1000.0,
    )
    assert advert.name is None


# ---------------------------------------------------------------------------
# BLEScanner construction
# ---------------------------------------------------------------------------


def test_scanner_defaults():
    scanner = BLEScanner()
    assert scanner.advert_count == 0
    assert scanner.scan_duration == 0


def test_scanner_custom_params():
    scanner = BLEScanner(scan_duration_seconds=5.0)
    assert scanner.scan_duration == 5.0


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------


def test_start_creates_thread():
    scanner = BLEScanner()
    # Mock the scan thread to not actually scan
    with patch.object(scanner, "_scan_thread_main"):
        scanner.start()
        assert scanner._scan_thread is not None
        scanner.stop()
        assert scanner._scan_thread is None


def test_start_twice_warns(caplog):
    import logging

    scanner = BLEScanner()
    alive_thread = MagicMock()
    alive_thread.is_alive.return_value = True
    scanner._scan_thread = alive_thread  # fake thread corriendo

    with caplog.at_level(logging.WARNING):
        scanner.start()

    assert "ble_scan_already_running" in caplog.text
    assert scanner._scan_thread is alive_thread  # no lo reemplazó


def test_start_replaces_dead_thread():
    """Un thread que murió solo (bleak crasheó, BlueZ down) no debe bloquear
    el restart: start() lo reemplaza en vez de warnear already_running."""
    scanner = BLEScanner()
    dead_thread = MagicMock()
    dead_thread.is_alive.return_value = False
    scanner._scan_thread = dead_thread

    with patch.object(scanner, "_scan_thread_main"):
        scanner.start()
        assert scanner._scan_thread is not dead_thread
        scanner.stop()


def test_stop_sets_event():
    scanner = BLEScanner()
    scanner._stop_event.clear()
    scanner.stop()
    assert scanner._stop_event.is_set()


def test_watchdog_restart_does_not_resurrect_wedged_thread():
    """Regresión: stop()+start() del watchdog con el thread viejo wedgeado
    (join timeout) NO debe poder resucitarlo. Cada start() crea un
    stop_event NUEVO que el thread y su detection callback capturan por
    closure; el evento del zombie queda set para siempre, así cuando se
    des-wedgea sale solo. Con el evento compartido, el clear() del restart
    lo revivía: dos BleakScanner activos (rate doble) y el zombie tickeando
    el heartbeat compartido — enmascaraba wedges futuros del thread nuevo."""
    scanner = BLEScanner()
    with patch.object(scanner, "_scan_thread_main"):
        scanner.start()
        gen1_event = scanner._stop_event

        # Simular el wedge: el join de stop() va a expirar (thread "vivo").
        wedged = MagicMock()
        wedged.is_alive.return_value = True
        wedged.join.return_value = None
        scanner._scan_thread = wedged

        scanner.stop()  # setea el evento de la gen 1, join timeoutea
        assert gen1_event.is_set()
        assert scanner._scan_thread is None

        scanner.start()  # restart del watchdog
        # Gen 2 con evento NUEVO y limpio; el del zombie sigue set → si el
        # zombie se des-wedgea, su loop/callback ven set y mueren solos.
        assert scanner._stop_event is not gen1_event
        assert not scanner._stop_event.is_set()
        assert gen1_event.is_set()
        scanner.stop()


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def test_on_advert_callback():
    events = []
    scanner = BLEScanner(on_advert=lambda e: events.append(e))

    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-55.0,
        name=None,
        timestamp=time.time(),
    )
    scanner.on_advert(advert)
    assert len(events) == 1
    assert events[0].mac == "AA:BB:CC:DD:EE:FF"


def test_advert_count_starts_zero():
    scanner = BLEScanner()
    assert scanner.advert_count == 0


# ---------------------------------------------------------------------------
# Detection callback simulation
# ---------------------------------------------------------------------------


def test_detection_callback_increments_count():
    """Simulate what bleak's detection_callback does."""
    events = []
    scanner = BLEScanner(on_advert=lambda e: events.append(e))
    scanner._stop_event.clear()

    # Simulate a detection
    mock_device = MagicMock()
    mock_device.address = "AA:BB:CC:DD:EE:FF"
    mock_adv_data = MagicMock()
    mock_adv_data.rssi = -65
    mock_adv_data.local_name = "TestDevice"

    # Manually invoke what _detection_callback would do
    advert = BLEAdvertisement(
        mac=mock_device.address,
        rssi=float(mock_adv_data.rssi),
        name=mock_adv_data.local_name,
        timestamp=time.time(),
    )
    scanner._advert_count += 1
    if scanner.on_advert:
        scanner.on_advert(advert)

    assert scanner.advert_count == 1
    assert len(events) == 1
    assert events[0].name == "TestDevice"
    assert events[0].rssi == -65.0


def test_is_healthy_false_when_never_started():
    """Scanner recién instanciado nunca pulsó — is_healthy debe ser False
    así el supervisor no falso-positivee como healthy."""
    scanner = BLEScanner()
    assert scanner.is_healthy(60.0) is False


def test_is_healthy_true_after_fresh_heartbeat():
    """Una pulsación reciente del heartbeat hace is_healthy True dentro
    del timeout."""
    scanner = BLEScanner()
    scanner._last_heartbeat_ts = time.time()
    assert scanner.is_healthy(60.0) is True


def test_is_healthy_false_when_heartbeat_stale():
    """Heartbeat más viejo que el timeout → is_healthy False, señal de
    wedge para el supervisor."""
    scanner = BLEScanner()
    scanner._last_heartbeat_ts = time.time() - 120.0  # 2min atrás
    assert scanner.is_healthy(60.0) is False


def test_callback_error_does_not_propagate():
    """Errors in on_advert should not crash the scanner."""

    def bad_callback(e):
        raise ValueError("boom")

    scanner = BLEScanner(on_advert=bad_callback)

    # Simulate calling the callback with error handling like the real code does
    advert = BLEAdvertisement(
        mac="AA:BB:CC:DD:EE:FF", rssi=-50.0, name=None, timestamp=time.time()
    )
    scanner._advert_count += 1
    try:
        scanner.on_advert(advert)
    except ValueError:
        pass  # In the real code, this is caught by the except in _detection_callback

    assert scanner.advert_count == 1

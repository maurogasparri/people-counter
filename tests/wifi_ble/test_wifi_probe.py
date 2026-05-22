"""Tests para el módulo de captura de probe requests WiFi."""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.wifi_ble.wifi_probe import (
    CHANNELS_24GHZ,
    CHANNELS_5GHZ,
    DEFAULT_HOP_INTERVAL,
    ProbeEvent,
    WiFiProbeCapture,
)


def test_setup_and_start_async_retries_until_radio_ready():
    """El setup async no bloquea y reintenta hasta que el radio levanta:
    setup_monitor_mode falla 2 veces (radio aún inicializando en el boot) y al
    3er intento funciona → arranca la captura. Modela el ~1min de init del
    brcmfmac tras el boot sin frenar el pipeline."""
    cap = WiFiProbeCapture(interface="wlan0")
    cap._ASYNC_SETUP_INTERVAL_S = 0.01  # sin esperas largas en el test
    state = {"setup_calls": 0}

    def _setup():
        state["setup_calls"] += 1
        if state["setup_calls"] < 3:
            raise RuntimeError("Operation not possible due to RF-kill")

    cap.setup_monitor_mode = _setup  # type: ignore[method-assign]
    cap.start = MagicMock()  # type: ignore[method-assign]

    cap.setup_and_start_async()  # no bloquea
    cap._setup_thread.join(timeout=5.0)

    assert state["setup_calls"] == 3
    cap.start.assert_called_once()


def test_setup_and_start_async_gives_up_after_deadline():
    """Si el radio nunca levanta, el setup async se rinde al deadline sin
    bloquear ni crashear (degrade: pipeline sigue sin WiFi)."""
    cap = WiFiProbeCapture(interface="wlan0")
    cap._ASYNC_SETUP_DEADLINE_S = 0.0  # se rinde tras el 1er intento fallido
    cap._ASYNC_SETUP_INTERVAL_S = 0.01
    cap.setup_monitor_mode = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("nexmon firmware crashed")
    )
    cap.start = MagicMock()  # type: ignore[method-assign]

    cap.setup_and_start_async()
    cap._setup_thread.join(timeout=5.0)

    cap.start.assert_not_called()


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
    # En RPi5 + CYW43455 + nexmon el chip soporta una vif simultánea; el
    # mon_interface = interface (la propia ``wlan0`` se convierte a monitor
    # mode via ``iw set type``, no hay vif paralela ``wlan0mon`` estilo
    # airmon-ng).
    assert cap.mon_interface == "wlan0"


# ---------------------------------------------------------------------------
# Monitor mode setup (rfkill + airmon-ng kill + iw set type monitor)
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_calls_expected_commands(mock_run):
    """El setup ejecuta el flujo nexmon: rfkill unblock + airmon-ng check
    kill (mata NM/wpa) + ip link down + iw set type monitor + ip link up
    + iw info verify."""
    # El último call es ``iw info`` para verify — tiene que devolver
    # stdout con ``type monitor`` o setup_monitor_mode raisea.
    def _run_side_effect(*args, **kwargs):
        cmd = args[0]
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "info" in cmd:
            result.stdout = "Interface wlan0\n\ttype monitor\n"
        return result

    mock_run.side_effect = _run_side_effect
    cap = WiFiProbeCapture(interface="wlan0")
    cap.setup_monitor_mode()

    calls = mock_run.call_args_list
    # rfkill unblock wifi (best-effort)
    assert any("rfkill" in str(c) for c in calls)
    # airmon-ng check kill (mata NM/wpa)
    assert any("airmon-ng" in str(c) and "kill" in str(c) for c in calls)
    # iw dev wlan0 set type monitor (no wlan0mon — same interface)
    assert any(
        "iw" in str(c) and "wlan0" in str(c) and "monitor" in str(c)
        for c in calls
    )
    # ip link up/down sobre wlan0
    assert any("ip" in str(c) and "link" in str(c) for c in calls)


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_raises_if_verify_fails(mock_run):
    """Si ``iw info`` post-setup no muestra ``type monitor``, raisea —
    el driver probablemente rechazó el cambio de modo."""
    mock_run.return_value = MagicMock(
        returncode=0, stdout="type managed", stderr=""
    )
    cap = WiFiProbeCapture(interface="wlan0")
    cap._MONITOR_SETUP_ATTEMPTS = 1  # sin retry: el verify falla determinístico
    with pytest.raises(RuntimeError, match="no quedó en monitor mode"):
        cap.setup_monitor_mode()


@patch(
    "src.wifi_ble.wifi_probe.subprocess.run",
    side_effect=FileNotFoundError("iw"),
)
def test_setup_monitor_mode_missing_tool(mock_run):
    """Sin ``iw`` instalado el setup raisea con guidance de install."""
    cap = WiFiProbeCapture()
    cap._MONITOR_SETUP_ATTEMPTS = 1  # sin retry: tool faltante no se arregla
    with pytest.raises(RuntimeError, match="Required tool not found"):
        cap.setup_monitor_mode()


@patch("src.wifi_ble.wifi_probe.time.sleep")  # sin delay real en el retry
@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_retries_on_rfkill_race(mock_run, _mock_sleep):
    """Si el primer intento falla con RF-kill (NM re-bloquea wlan0 en el boot
    antes de asentarse), el setup reintenta y termina logrando monitor mode."""
    state = {"attempt": 0}

    def _run_side_effect(*args, **kwargs):
        cmd = args[0]
        # ``ip link set wlan0 up`` falla con RF-kill SOLO en el 1er intento.
        if "ip" in cmd and "up" in cmd:
            state["attempt"] += 1
            if state["attempt"] == 1:
                return MagicMock(
                    returncode=2,
                    stdout="",
                    stderr="RTNETLINK answers: Operation not possible due to RF-kill",
                )
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "info" in cmd:
            result.stdout = "Interface wlan0\n\ttype monitor\n"
        return result

    mock_run.side_effect = _run_side_effect
    cap = WiFiProbeCapture(interface="wlan0")
    cap.setup_monitor_mode()  # no debe raisear: reintenta y logra
    # El 'ip link up' se intentó 2 veces (falló, reintentó OK).
    assert state["attempt"] == 2


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_teardown_monitor_mode(mock_run):
    """Teardown revierte: ip link down + iw set type managed + ip link up
    + nmcli managed yes (best-effort). NO usa ``airmon-ng stop`` porque
    nunca creamos una vif paralela."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    cap = WiFiProbeCapture(interface="wlan0")
    cap.teardown_monitor_mode()

    calls = mock_run.call_args_list
    # ip link down/up + iw set type managed.
    assert any("ip" in str(c) and "link" in str(c) for c in calls)
    assert any(
        "iw" in str(c) and "managed" in str(c) for c in calls
    )
    # nmcli para devolverle la interface a NetworkManager (best-effort).
    assert any("nmcli" in str(c) for c in calls)


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

    assert "wifi_capture_already_running" in caplog.text


# ---------------------------------------------------------------------------
# Channel hopping
# ---------------------------------------------------------------------------


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_channel_hop_uses_mon_interface(mock_run):
    """Channel hop debe usar ``cap.mon_interface`` (= ``wlan0`` en el
    flujo nexmon actual). El test es agnóstico al nombre concreto — usa
    el atributo del cap, así sigue siendo válido si el día de mañana
    arquitecturas con dual-vif overridean mon_interface a ``wlan0mon``."""
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
    # Todos los channel set calls deben usar mon_interface del cap.
    for c in mock_run.call_args_list:
        args = c[0][0]
        if "channel" in args:
            assert cap.mon_interface in args


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

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
    _build_hop_sequence,
    _is_probe_req_radiotap,
    is_randomized_mac,
)


def _radiotap_frame(fc0: int, rt_len: int = 8) -> bytes:
    """Frame radiotap mínimo construido a mano: header de ``rt_len`` bytes
    (versión 0 + len LE + present vacío) seguido del Frame Control 802.11."""
    header = bytes([0, 0, rt_len & 0xFF, rt_len >> 8]) + b"\x00" * (rt_len - 4)
    return header + bytes([fc0, 0x00]) + b"\x00" * 22  # FC + resto del MAC hdr


def test_is_probe_req_radiotap_prefilter():
    """Pre-filtro de bytes crudos que evita la disección completa de scapy
    por frame (el BPF kernel-level no compila en el DLT de nexmon —
    verificado en hardware). True = probe-req, False = otro frame 802.11,
    None = no parece radiotap (el caller cae al parse completo)."""
    # Probe request: type 0 subtype 4 → FC0 = 0x40.
    assert _is_probe_req_radiotap(_radiotap_frame(0x40)) is True
    # Beacon (type 0 subtype 8 → 0x80) y data frame (type 2 → 0x08): False.
    assert _is_probe_req_radiotap(_radiotap_frame(0x80)) is False
    assert _is_probe_req_radiotap(_radiotap_frame(0x08)) is False
    # Bits de protocol version ignorados (máscara 0xFC).
    assert _is_probe_req_radiotap(_radiotap_frame(0x41)) is True
    # Header radiotap más largo (con campos present): el offset del FC sigue.
    assert _is_probe_req_radiotap(_radiotap_frame(0x40, rt_len=18)) is True
    # No parece radiotap (versión != 0) → None (parse completo decide).
    assert _is_probe_req_radiotap(b"\x01\x00\x08\x00" + b"\x00" * 30) is None
    # Truncados / degenerados → False o None, nunca crash.
    assert _is_probe_req_radiotap(b"") is None
    assert _is_probe_req_radiotap(b"\x00\x00") is None
    assert _is_probe_req_radiotap(b"\x00\x00\xff\x7f" + b"\x00" * 10) is False
    assert _is_probe_req_radiotap(b"\x00\x00\x02\x00" + b"\x00" * 10) is False


def test_is_randomized_mac():
    # LA bit (0x02) seteado en el primer octeto = randomizada (humano).
    assert is_randomized_mac("DE:AD:BE:EF:00:01") is True  # 0xDE & 0x02
    assert is_randomized_mac("F2:11:22:33:44:55") is True  # 0xF2 & 0x02
    # MAC global (OUI real, LA bit en 0) = infra/IoT.
    assert is_randomized_mac("B8:27:EB:00:00:01") is False  # OUI RPi
    assert is_randomized_mac("3C:84:6A:90:C0:BE") is False  # 0x3C & 0x02 = 0
    # Malformada / vacía -> no contar.
    assert is_randomized_mac("") is False
    assert is_randomized_mac("ZZ:ZZ") is False


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
    # Secuencia ponderada: 1/6/11 visitados una vez por cada canal de 5 GHz.
    assert cap.channels == _build_hop_sequence(CHANNELS_24GHZ, CHANNELS_5GHZ)
    assert cap.hop_interval == DEFAULT_HOP_INTERVAL
    # 2.4 GHz solo 1/6/11 (no solapados); 5 GHz sin DFS (52-144).
    assert CHANNELS_24GHZ == [1, 6, 11]
    assert all(not (52 <= c <= 144) for c in CHANNELS_5GHZ)


def test_custom_channels():
    cap = WiFiProbeCapture(
        channels_24=[1, 6, 11],
        channels_5=[36, 40],
    )
    # Interleave: [1,6,11, 36, 1,6,11, 40]
    assert cap.channels == [1, 6, 11, 36, 1, 6, 11, 40]


def test_build_hop_sequence_weights_24ghz():
    seq = _build_hop_sequence([1, 6, 11], [36, 40, 44])
    # 1/6/11 aparecen una vez por cada canal de 5 GHz (3 veces); cada 5 GHz una.
    assert seq.count(1) == 3 and seq.count(6) == 3 and seq.count(11) == 3
    assert seq.count(36) == 1 and seq.count(40) == 1 and seq.count(44) == 1
    # Sin canales de 5 GHz, solo el barrido de 2.4.
    assert _build_hop_sequence([1, 6, 11], []) == [1, 6, 11]


def test_probe_rate_per_min():
    cap = WiFiProbeCapture(interface="wlan0")
    # Primera llamada: sin baseline → None (no rate falso al arrancar).
    assert cap.probe_rate_per_min() is None
    # Baseline hace 60s con 0 probes; ahora 60 → ~60 probes/min.
    cap._rate_last_count = 0
    cap._rate_last_ts = time.time() - 60.0
    cap._probe_count = 60
    rate = cap.probe_rate_per_min()
    assert rate is not None
    assert 55.0 < rate < 65.0


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
        "iw" in str(c) and "wlan0" in str(c) and "monitor" in str(c) for c in calls
    )
    # ip link up/down sobre wlan0
    assert any("ip" in str(c) and "link" in str(c) for c in calls)
    # nexutil -m2 (monitor + radiotap en el firmware nexmon)
    assert any("nexutil" in str(c) and "-m2" in str(c) for c in calls)


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_raises_if_nexutil_fails(mock_run):
    """Si nexutil -m2 falla, el setup raisea — sin monitor+radiotap del
    firmware el interface no entrega frames 802.11 y no hay captura."""

    def _run_side_effect(*args, **kwargs):
        cmd = args[0]
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "info" in cmd:
            result.stdout = "Interface wlan0\n\ttype monitor\n"
        if "nexutil" in cmd:
            result.returncode = 1
            result.stderr = "ioctl error"
        return result

    mock_run.side_effect = _run_side_effect
    cap = WiFiProbeCapture(interface="wlan0")
    cap._MONITOR_SETUP_ATTEMPTS = 1
    with pytest.raises(RuntimeError, match="nexutil -m2"):
        cap.setup_monitor_mode()


@patch("src.wifi_ble.wifi_probe.subprocess.run")
def test_setup_monitor_mode_raises_if_verify_fails(mock_run):
    """Si ``iw info`` post-setup no muestra ``type monitor``, raisea —
    el driver probablemente rechazó el cambio de modo."""
    mock_run.return_value = MagicMock(returncode=0, stdout="type managed", stderr="")
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
    assert any("iw" in str(c) and "managed" in str(c) for c in calls)
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

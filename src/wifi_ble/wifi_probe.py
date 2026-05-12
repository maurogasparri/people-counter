"""Captura de probe requests WiFi vía monitor mode.

Usa el CYW43455 (onboard de la RPi5) en monitor mode mediante los parches de
firmware nexmon y airmon-ng (de aircrack-ng). Captura frames probe request 802.11.

WiFi es EXCLUSIVO para probing — la conectividad de red es solo por Ethernet.

Prerrequisitos:
    - firmware nexmon: paquetes firmware-nexmon + brcmfmac-nexmon-dkms
    - aircrack-ng: provee airmon-ng para administrar monitor mode
    - scapy: para parseo de paquetes (pip install scapy)

Setup (una sola vez):
    sudo apt install -y aircrack-ng
    # paquetes nexmon — ver docs/setup-guide.md
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROBE_REQUEST_SUBTYPE = 4

# Secuencia de channel hopping: 2.4 GHz (1-13) + canales 5 GHz comunes
CHANNELS_24GHZ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
CHANNELS_5GHZ = [36, 40, 44, 48, 52, 56, 60, 64, 149, 153, 157, 161, 165]
DEFAULT_HOP_INTERVAL = 0.3  # segundos por canal


@dataclass
class ProbeEvent:
    """Probe request WiFi capturado."""

    mac: str
    rssi: float
    ssid: str
    channel: int
    timestamp: float


class WiFiProbeCapture:
    """Captura probe requests WiFi en monitor mode.

    Usa airmon-ng para crear una interfaz monitor (wlan0mon) y captura los
    probe requests vía scapy con channel hopping.

    Lifecycle:
        1. setup_monitor_mode() — corre airmon-ng start, crea wlan0mon
        2. start() — arranca captura async + channel hopping
        3. stop() — detiene la captura
        4. teardown_monitor_mode() — corre airmon-ng stop

    Cada probe capturado se pasa al callback on_probe, que debería
    alimentarlo al DedupEngine vía hash_mac + process_detection.
    """

    def __init__(
        self,
        interface: str = "wlan0",
        on_probe: Optional[Callable[[ProbeEvent], None]] = None,
        hop_interval: float = DEFAULT_HOP_INTERVAL,
        channels_24: Optional[list[int]] = None,
        channels_5: Optional[list[int]] = None,
    ) -> None:
        self.interface = interface
        self.mon_interface = f"{interface}mon"
        self.on_probe = on_probe
        self.hop_interval = hop_interval
        self.channels = (channels_24 or CHANNELS_24GHZ) + (channels_5 or CHANNELS_5GHZ)
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._hop_thread: Optional[threading.Thread] = None
        self._current_channel = 0
        self._probe_count = 0

    @property
    def probe_count(self) -> int:
        return self._probe_count

    @property
    def is_running(self) -> bool:
        """True si el thread de captura está vivo y no fue señalizado para parar.

        Usado por la telemetría para reportar la health del subsistema WiFi —
        si el firmware nexmon murió o airmon-ng falló silenciosamente, el thread
        sale y este flag pasa a False sin matar el pipeline de visión.
        """
        thread = self._capture_thread
        return (
            thread is not None
            and thread.is_alive()
            and not self._stop_event.is_set()
        )

    def setup_monitor_mode(self) -> None:
        """Crea la interfaz monitor vía airmon-ng.

        Requiere privilegios root y firmware con parches nexmon.
        Crea wlan0mon a partir de wlan0.

        Raises:
            RuntimeError: Si airmon-ng falla.
        """
        try:
            # Mata procesos que interfieren
            subprocess.run(
                ["airmon-ng", "check", "kill"],
                capture_output=True,
            )

            # Arranca monitor mode — crea wlan0mon
            result = subprocess.run(
                ["airmon-ng", "start", self.interface],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(
                "monitor_mode_enabled",
                extra={
                    "interface": self.interface,
                    "mon_interface": self.mon_interface,
                },
            )

            # Verifica que la interfaz monitor exista
            verify = subprocess.run(
                ["iw", "dev", self.mon_interface, "info"],
                capture_output=True,
                text=True,
            )
            if verify.returncode != 0:
                raise RuntimeError(
                    f"Monitor interface {self.mon_interface} not created. "
                    f"airmon-ng output: {result.stdout}"
                )

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Required tool not found: {e}. "
                "Install with: sudo apt install aircrack-ng"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to start monitor mode: {e.stderr}"
            ) from e

    def teardown_monitor_mode(self) -> None:
        """Detiene monitor mode y restaura la interfaz como managed."""
        try:
            subprocess.run(
                ["airmon-ng", "stop", self.mon_interface],
                capture_output=True,
            )
            # Restaura el manejo por NetworkManager
            subprocess.run(
                ["nmcli", "dev", "set", self.interface, "managed", "yes"],
                capture_output=True,
            )
            logger.info("monitor_mode_stopped")
        except Exception:
            logger.exception("Failed to restore managed mode")

    def start(self) -> None:
        """Arranca captura asíncrona de probes y channel hopping."""
        if self._capture_thread is not None:
            logger.warning("wifi_capture_already_running")
            return

        self._stop_event.clear()
        self._probe_count = 0

        self._hop_thread = threading.Thread(
            target=self._channel_hop_loop, daemon=True, name="wifi-hop"
        )
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="wifi-capture"
        )

        self._hop_thread.start()
        self._capture_thread.start()
        logger.info(
            "wifi_capture_started",
            extra={"mon_interface": self.mon_interface},
        )

    def stop(self) -> None:
        """Detiene la captura y el channel hopping."""
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        if self._hop_thread is not None:
            self._hop_thread.join(timeout=5.0)
            self._hop_thread = None
        logger.info(
            "wifi_capture_stopped",
            extra={"count": self._probe_count},
        )

    def _channel_hop_loop(self) -> None:
        """Cicla por los canales con frecuencia hop_interval."""
        idx = 0
        while not self._stop_event.is_set():
            channel = self.channels[idx % len(self.channels)]
            try:
                subprocess.run(
                    ["iw", "dev", self.mon_interface, "set", "channel", str(channel)],
                    check=True,
                    capture_output=True,
                )
                self._current_channel = channel
            except subprocess.CalledProcessError:
                logger.debug(
                    "channel_set_failed", extra={"channel": channel}
                )
            except FileNotFoundError:
                logger.error(
                    "iw_not_found",
                    extra={"reason": "channel hopping disabled"},
                )
                return

            idx += 1
            self._stop_event.wait(self.hop_interval)

    def _capture_loop(self) -> None:
        """Captura probe requests con scapy sobre la interfaz monitor."""
        try:
            from scapy.all import Dot11, Dot11ProbeReq, RadioTap, sniff
        except ImportError:
            logger.error(
                "scapy_not_installed",
                extra={"reason": "WiFi probe capture disabled"},
            )
            return

        def _handle_packet(pkt: Any) -> None:
            if not pkt.haslayer(Dot11):
                return

            dot11 = pkt.getlayer(Dot11)

            # Filtra probe requests (type=0 management, subtype=4)
            if dot11.type != 0 or dot11.subtype != PROBE_REQUEST_SUBTYPE:
                return

            mac = dot11.addr2
            if mac is None:
                return

            # Extrae el RSSI del header RadioTap
            rssi = -100.0
            if pkt.haslayer(RadioTap):
                try:
                    rssi = float(pkt[RadioTap].dBm_AntSignal)
                except (AttributeError, TypeError):
                    pass

            # Extrae el SSID
            ssid = ""
            if pkt.haslayer(Dot11ProbeReq):
                try:
                    raw_ssid = pkt[Dot11ProbeReq].info
                    if raw_ssid:
                        ssid = raw_ssid.decode("utf-8", errors="ignore")
                except (AttributeError, UnicodeDecodeError):
                    pass

            event = ProbeEvent(
                mac=mac,
                rssi=rssi,
                ssid=ssid,
                channel=self._current_channel,
                timestamp=time.time(),
            )

            self._probe_count += 1

            if self.on_probe:
                try:
                    self.on_probe(event)
                except Exception:
                    logger.exception("Error in on_probe callback")

        logger.info(
            "scapy_sniff_starting",
            extra={"mon_interface": self.mon_interface},
        )
        try:
            sniff(
                iface=self.mon_interface,
                prn=_handle_packet,
                store=False,
                stop_filter=lambda _: self._stop_event.is_set(),
            )
        except OSError as e:
            logger.error(
                "wifi_capture_error",
                extra={"mon_interface": self.mon_interface, "error": str(e)},
            )
        except Exception:
            logger.exception("Unexpected capture error")

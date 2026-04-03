"""WiFi probe request capture via monitor mode.

Uses the CYW43455 (RPi5 onboard) in monitor mode via nexmon firmware patches
and airmon-ng (from aircrack-ng). Captures 802.11 probe request frames.

WiFi is EXCLUSIVE for probing — network connectivity is Ethernet only.

Prerequisites:
    - nexmon firmware: firmware-nexmon + brcmfmac-nexmon-dkms packages
    - aircrack-ng: provides airmon-ng for monitor mode management
    - scapy: for packet parsing (pip install scapy)

Setup (one-time):
    sudo apt install -y aircrack-ng
    # nexmon packages — see docs/setup-guide.md
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROBE_REQUEST_SUBTYPE = 4

# Channel hop sequence: 2.4 GHz (1-13) + 5 GHz common channels
CHANNELS_24GHZ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
CHANNELS_5GHZ = [36, 40, 44, 48, 52, 56, 60, 64, 149, 153, 157, 161, 165]
DEFAULT_HOP_INTERVAL = 0.3  # seconds per channel


@dataclass
class ProbeEvent:
    """A captured WiFi probe request."""

    mac: str
    rssi: float
    ssid: str
    channel: int
    timestamp: float


class WiFiProbeCapture:
    """Captures WiFi probe requests in monitor mode.

    Uses airmon-ng to create a monitor interface (wlan0mon), then captures
    probe requests via scapy with channel hopping.

    Lifecycle:
        1. setup_monitor_mode() — runs airmon-ng start, creates wlan0mon
        2. start() — begins async capture + channel hopping
        3. stop() — stops capture
        4. teardown_monitor_mode() — runs airmon-ng stop

    Each captured probe is passed to the on_probe callback, which should
    feed it into the DedupEngine via hash_mac + process_detection.
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

    def setup_monitor_mode(self) -> None:
        """Create monitor interface via airmon-ng.

        Requires root privileges and nexmon-patched firmware.
        Creates wlan0mon from wlan0.

        Raises:
            RuntimeError: If airmon-ng fails.
        """
        try:
            # Kill interfering processes
            subprocess.run(
                ["airmon-ng", "check", "kill"],
                capture_output=True,
            )

            # Start monitor mode — creates wlan0mon
            result = subprocess.run(
                ["airmon-ng", "start", self.interface],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Monitor mode enabled: %s → %s", self.interface, self.mon_interface)

            # Verify the monitor interface exists
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
        """Stop monitor mode and restore managed interface."""
        try:
            subprocess.run(
                ["airmon-ng", "stop", self.mon_interface],
                capture_output=True,
            )
            # Restore NetworkManager management
            subprocess.run(
                ["nmcli", "dev", "set", self.interface, "managed", "yes"],
                capture_output=True,
            )
            logger.info("Monitor mode stopped, managed mode restored")
        except Exception:
            logger.exception("Failed to restore managed mode")

    def start(self) -> None:
        """Start asynchronous probe capture and channel hopping."""
        if self._capture_thread is not None:
            logger.warning("Capture already running")
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
        logger.info("WiFi probe capture started on %s", self.mon_interface)

    def stop(self) -> None:
        """Stop capture and channel hopping."""
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        if self._hop_thread is not None:
            self._hop_thread.join(timeout=5.0)
            self._hop_thread = None
        logger.info(
            "WiFi probe capture stopped. Total probes: %d", self._probe_count
        )

    def _channel_hop_loop(self) -> None:
        """Cycle through channels at hop_interval."""
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
                logger.debug("Failed to set channel %d — skipping", channel)
            except FileNotFoundError:
                logger.error("iw not found — channel hopping disabled")
                return

            idx += 1
            self._stop_event.wait(self.hop_interval)

    def _capture_loop(self) -> None:
        """Capture probe requests using scapy on the monitor interface."""
        try:
            from scapy.all import Dot11, Dot11ProbeReq, RadioTap, sniff
        except ImportError:
            logger.error(
                "scapy not installed. Install with: pip install scapy. "
                "WiFi probe capture disabled."
            )
            return

        def _handle_packet(pkt: Any) -> None:
            if not pkt.haslayer(Dot11):
                return

            dot11 = pkt.getlayer(Dot11)

            # Filter for probe requests (type=0 management, subtype=4)
            if dot11.type != 0 or dot11.subtype != PROBE_REQUEST_SUBTYPE:
                return

            mac = dot11.addr2
            if mac is None:
                return

            # Extract RSSI from RadioTap header
            rssi = -100.0
            if pkt.haslayer(RadioTap):
                try:
                    rssi = float(pkt[RadioTap].dBm_AntSignal)
                except (AttributeError, TypeError):
                    pass

            # Extract SSID
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

        logger.info("Starting scapy sniff on %s", self.mon_interface)
        try:
            sniff(
                iface=self.mon_interface,
                prn=_handle_packet,
                store=False,
                stop_filter=lambda _: self._stop_event.is_set(),
            )
        except OSError as e:
            logger.error("Capture error on %s: %s", self.mon_interface, e)
        except Exception:
            logger.exception("Unexpected capture error")

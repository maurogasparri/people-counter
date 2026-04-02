"""WiFi probe request capture via monitor mode.

Uses the CYW43455 (RPi5 onboard) in monitor mode via nexmon patches.
Captures 802.11 probe request frames on 2.4 GHz and 5 GHz bands.

WiFi is EXCLUSIVE for probing — network connectivity is Ethernet only.
The interface must be set to monitor mode before starting capture.

Requires: scapy (for packet parsing), iw/ip tools (for interface setup).
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Probe request subtype in 802.11 management frames
PROBE_REQUEST_SUBTYPE = 4

# Channel hop sequence: 2.4 GHz (1-13) + 5 GHz common channels
CHANNELS_24GHZ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
CHANNELS_5GHZ = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 149, 153, 157, 161, 165]
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

    Lifecycle:
        1. setup_monitor_mode() — puts interface in monitor mode
        2. start() — begins async capture + channel hopping
        3. stop() — stops capture
        4. teardown_monitor_mode() — restores interface

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
        """Put the WiFi interface into monitor mode.

        Requires root privileges and nexmon-patched firmware.

        Raises:
            RuntimeError: If interface setup fails.
        """
        try:
            subprocess.run(
                ["ip", "link", "set", self.interface, "down"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["iw", self.interface, "set", "monitor", "none"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["ip", "link", "set", self.interface, "up"],
                check=True,
                capture_output=True,
            )
            logger.info("Monitor mode enabled on %s", self.interface)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Required tool not found: {e}. "
                "Ensure iw and ip are installed."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to set monitor mode on {self.interface}: "
                f"{e.stderr.decode().strip()}"
            ) from e

    def teardown_monitor_mode(self) -> None:
        """Restore the WiFi interface to managed mode."""
        try:
            subprocess.run(
                ["ip", "link", "set", self.interface, "down"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["iw", self.interface, "set", "type", "managed"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["ip", "link", "set", self.interface, "up"],
                check=True,
                capture_output=True,
            )
            logger.info("Managed mode restored on %s", self.interface)
        except Exception:
            logger.exception("Failed to restore managed mode on %s", self.interface)

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
        logger.info("WiFi probe capture started on %s", self.interface)

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
                    ["iw", "dev", self.interface, "set", "channel", str(channel)],
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
        """Capture probe requests using scapy."""
        try:
            from scapy.all import RadioTap, Dot11, Dot11ProbeReq, sniff
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

            # Filter for probe requests (type=0 management, subtype=4 probe req)
            if dot11.type != 0 or dot11.subtype != PROBE_REQUEST_SUBTYPE:
                return

            mac = dot11.addr2
            if mac is None:
                return

            # Extract RSSI from RadioTap header
            rssi = -100.0  # default if not available
            if pkt.haslayer(RadioTap):
                try:
                    rssi = float(pkt[RadioTap].dBm_AntSignal)
                except (AttributeError, TypeError):
                    pass

            # Extract SSID from probe request
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

        logger.info("Starting scapy sniff on %s", self.interface)
        try:
            sniff(
                iface=self.interface,
                prn=_handle_packet,
                store=False,
                stop_filter=lambda _: self._stop_event.is_set(),
            )
        except OSError as e:
            logger.error("Capture error on %s: %s", self.interface, e)
        except Exception:
            logger.exception("Unexpected capture error")

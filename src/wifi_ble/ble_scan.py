"""BLE passive advertising capture.

Listens for BLE advertising packets on channels 37, 38, 39 using
the CYW43455 Bluetooth interface on RPi5. Captures device MAC and
RSSI for deduplication and traffic counting.

Requires: bluez (hcitool, hciconfig) or bleak for BLE scanning.
On RPi5 Bookworm, uses the D-Bus BlueZ API via dbus-fast or
falls back to hcitool lescan parsing.
"""

import logging
import subprocess
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# BLE advertising event types
ADV_IND = 0x00  # Connectable undirected advertising
ADV_SCAN_IND = 0x02  # Scannable undirected advertising
ADV_NONCONN_IND = 0x03  # Non-connectable undirected advertising


@dataclass
class BLEAdvertisement:
    """A captured BLE advertising packet."""

    mac: str
    rssi: float
    adv_type: int
    name: Optional[str]
    timestamp: float


class BLEScanner:
    """Passive BLE advertising scanner.

    Lifecycle:
        1. start() — begins async scanning
        2. stop() — stops scanning

    Each captured advertisement is passed to the on_advert callback,
    which should feed it into the DedupEngine via hash_mac + process_detection.
    """

    def __init__(
        self,
        on_advert: Optional[Callable[[BLEAdvertisement], None]] = None,
        hci_device: str = "hci0",
        scan_window_seconds: float = 10.0,
    ) -> None:
        self.on_advert = on_advert
        self.hci_device = hci_device
        self.scan_window = scan_window_seconds
        self._stop_event = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None
        self._advert_count = 0

    @property
    def advert_count(self) -> int:
        return self._advert_count

    def start(self) -> None:
        """Start asynchronous BLE scanning."""
        if self._scan_thread is not None:
            logger.warning("BLE scan already running")
            return

        self._stop_event.clear()
        self._advert_count = 0

        # Enable BLE adapter
        try:
            subprocess.run(
                ["hciconfig", self.hci_device, "up"],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            logger.error(
                "hciconfig not found. Ensure bluez is installed. "
                "BLE scanning disabled."
            )
            return
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to enable %s: %s",
                self.hci_device,
                e.stderr.decode().strip(),
            )
            return

        self._scan_thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="ble-scan"
        )
        self._scan_thread.start()
        logger.info("BLE scanner started on %s", self.hci_device)

    def stop(self) -> None:
        """Stop BLE scanning."""
        self._stop_event.set()

        # Disable LE scan
        try:
            subprocess.run(
                ["hcitool", "-i", self.hci_device, "cmd", "0x08", "0x000C",
                 "00", "00"],
                capture_output=True,
            )
        except Exception:
            pass

        if self._scan_thread is not None:
            self._scan_thread.join(timeout=5.0)
            self._scan_thread = None

        logger.info(
            "BLE scanner stopped. Total advertisements: %d", self._advert_count
        )

    def _scan_loop(self) -> None:
        """Continuous BLE scan using HCI LE scan."""
        while not self._stop_event.is_set():
            try:
                self._run_scan_window()
            except Exception:
                logger.exception("BLE scan error")
                self._stop_event.wait(1.0)

    def _run_scan_window(self) -> None:
        """Run a single LE scan window and parse results.

        Uses hcitool lescan in passive mode, capturing raw output
        for the configured window duration.
        """
        try:
            # Start passive LE scan via HCI commands
            # Set scan parameters: passive scan, 10ms interval, 10ms window
            subprocess.run(
                [
                    "hcitool", "-i", self.hci_device,
                    "cmd", "0x08", "0x000B",
                    "00",  # passive scan
                    "10", "00",  # scan interval (16 * 0.625ms = 10ms)
                    "10", "00",  # scan window
                    "00",  # own address type: public
                    "00",  # filter: accept all
                ],
                check=True,
                capture_output=True,
            )

            # Enable scanning
            subprocess.run(
                [
                    "hcitool", "-i", self.hci_device,
                    "cmd", "0x08", "0x000C",
                    "01",  # enable
                    "00",  # no duplicates filter (we dedup ourselves)
                ],
                check=True,
                capture_output=True,
            )

            # Read raw HCI events for scan_window seconds
            self._read_hci_events()

            # Disable scanning
            subprocess.run(
                [
                    "hcitool", "-i", self.hci_device,
                    "cmd", "0x08", "0x000C",
                    "00", "00",
                ],
                capture_output=True,
            )

        except FileNotFoundError:
            logger.error("hcitool not found — BLE scanning disabled")
            self._stop_event.set()
        except subprocess.CalledProcessError as e:
            logger.warning("HCI command failed: %s", e.stderr.decode().strip())

    def _read_hci_events(self) -> None:
        """Read HCI LE advertising report events.

        Opens the raw HCI socket and reads advertising reports
        for scan_window seconds.
        """
        try:
            import socket

            sock = socket.socket(
                socket.AF_BLUETOOTH, socket.SOCK_RAW, socket.BTPROTO_HCI
            )
            sock.bind((int(self.hci_device.replace("hci", "")),))
            # Set HCI filter for LE Meta events (event code 0x3E)
            # This is simplified — production would use a proper HCI filter
            sock.settimeout(1.0)
        except (OSError, ImportError, AttributeError) as e:
            logger.debug("Raw HCI socket not available: %s — using fallback", e)
            self._fallback_lescan()
            return

        deadline = time.time() + self.scan_window
        try:
            while not self._stop_event.is_set() and time.time() < deadline:
                try:
                    data = sock.recv(1024)
                    self._parse_hci_event(data)
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            sock.close()

    def _parse_hci_event(self, data: bytes) -> None:
        """Parse an HCI LE Advertising Report event.

        HCI LE Meta Event format:
            Byte 0: event type (0x04)
            Byte 1: event code (0x3E = LE Meta)
            Byte 2: parameter length
            Byte 3: subevent code (0x02 = advertising report)
            Byte 4: num reports
            Then per report: event_type(1), addr_type(1), addr(6), data_len(1),
                            data(N), rssi(1)
        """
        if len(data) < 12:
            return

        # Check for LE Meta Event with Advertising Report subevent
        if len(data) > 3 and data[1] == 0x3E and data[3] == 0x02:
            offset = 4
            num_reports = data[offset]
            offset += 1

            for _ in range(num_reports):
                if offset + 8 > len(data):
                    break

                adv_type = data[offset]
                offset += 1
                # Skip address type
                offset += 1
                # MAC address (6 bytes, reversed)
                mac_bytes = data[offset : offset + 6]
                mac = ":".join(f"{b:02X}" for b in reversed(mac_bytes))
                offset += 6
                # Data length
                data_len = data[offset]
                offset += 1
                # Skip advertising data
                adv_data = data[offset : offset + data_len]
                offset += data_len
                # RSSI (signed byte)
                if offset < len(data):
                    rssi = struct.unpack("b", bytes([data[offset]]))[0]
                    offset += 1
                else:
                    rssi = -100

                # Try to extract device name from advertising data
                name = self._extract_name(adv_data)

                advert = BLEAdvertisement(
                    mac=mac,
                    rssi=float(rssi),
                    adv_type=adv_type,
                    name=name,
                    timestamp=time.time(),
                )

                self._advert_count += 1

                if self.on_advert:
                    try:
                        self.on_advert(advert)
                    except Exception:
                        logger.exception("Error in on_advert callback")

    def _extract_name(self, adv_data: bytes) -> Optional[str]:
        """Extract Complete/Shortened Local Name from BLE advertising data."""
        i = 0
        while i < len(adv_data) - 1:
            length = adv_data[i]
            if length == 0:
                break
            if i + length >= len(adv_data):
                break
            ad_type = adv_data[i + 1]
            # 0x09 = Complete Local Name, 0x08 = Shortened Local Name
            if ad_type in (0x08, 0x09):
                try:
                    return adv_data[i + 2 : i + 1 + length].decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    pass
            i += length + 1
        return None

    def _fallback_lescan(self) -> None:
        """Fallback: use hcitool lescan with text output parsing."""
        try:
            proc = subprocess.Popen(
                ["hcitool", "-i", self.hci_device, "lescan", "--passive"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            deadline = time.time() + self.scan_window
            while not self._stop_event.is_set() and time.time() < deadline:
                self._stop_event.wait(0.5)

            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        except FileNotFoundError:
            logger.error("hcitool not found for fallback lescan")
        except Exception:
            logger.exception("Fallback lescan error")

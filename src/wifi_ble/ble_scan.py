"""BLE passive advertising capture using bleak.

Listens for BLE advertising packets using the BlueZ D-Bus API via bleak.
Captures device MAC and RSSI for deduplication and traffic counting.

Requires: bleak (pip install bleak), BlueZ 5.x (pre-installed on RPi OS).
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BLEAdvertisement:
    """A captured BLE advertising packet."""

    mac: str
    rssi: float
    name: Optional[str]
    timestamp: float


class BLEScanner:
    """Passive BLE advertising scanner using bleak.

    Lifecycle:
        1. start() — begins async scanning in a background thread
        2. stop() — stops scanning

    Each captured advertisement is passed to the on_advert callback,
    which should feed it into the DedupEngine via hash_mac + process_detection.
    """

    def __init__(
        self,
        on_advert: Optional[Callable[[BLEAdvertisement], None]] = None,
        scan_duration_seconds: float = 0,
    ) -> None:
        """Initialize BLE scanner.

        Args:
            on_advert: Callback for each detected advertisement.
            scan_duration_seconds: How long to scan. 0 = scan until stop() is called.
        """
        self.on_advert = on_advert
        self.scan_duration = scan_duration_seconds
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

        self._scan_thread = threading.Thread(
            target=self._scan_thread_main, daemon=True, name="ble-scan"
        )
        self._scan_thread.start()
        logger.info("BLE scanner started")

    def stop(self) -> None:
        """Stop BLE scanning."""
        self._stop_event.set()
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=10.0)
            self._scan_thread = None
        logger.info("BLE scanner stopped. Total advertisements: %d", self._advert_count)

    def _scan_thread_main(self) -> None:
        """Run the async scan loop in a dedicated thread."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._scan_async())
        except Exception:
            logger.exception("BLE scan error")
        finally:
            loop.close()

    async def _scan_async(self) -> None:
        """Async BLE scanning using bleak."""
        try:
            from bleak import BleakScanner
        except ImportError:
            logger.error(
                "bleak not installed. Install with: pip install bleak. "
                "BLE scanning disabled."
            )
            return

        def _detection_callback(device, advertisement_data) -> None:
            if self._stop_event.is_set():
                return

            mac = device.address
            rssi = advertisement_data.rssi if advertisement_data.rssi else -100
            name = advertisement_data.local_name

            advert = BLEAdvertisement(
                mac=mac,
                rssi=float(rssi),
                name=name,
                timestamp=time.time(),
            )

            self._advert_count += 1

            if self.on_advert:
                try:
                    self.on_advert(advert)
                except Exception:
                    logger.exception("Error in on_advert callback")

        scanner = BleakScanner(detection_callback=_detection_callback)

        await scanner.start()
        logger.info("Bleak BLE scanner active")

        # Wait until stop is requested
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)
            if self.scan_duration > 0:
                # For fixed-duration scans, check elapsed time
                break

        await scanner.stop()

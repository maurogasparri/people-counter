"""Captura pasiva de advertising BLE usando bleak.

Escucha paquetes de advertising BLE vía la API D-Bus de BlueZ a través de bleak.
Captura MAC del dispositivo y RSSI para dedup y conteo de tráfico.

Requiere: bleak (pip install bleak), BlueZ 5.x (preinstalado en RPi OS).
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
    """Paquete de advertising BLE capturado."""

    mac: str
    rssi: float
    name: Optional[str]
    timestamp: float


class BLEScanner:
    """Scanner pasivo de advertising BLE usando bleak.

    Lifecycle:
        1. start() — arranca el scanning async en un thread background
        2. stop() — detiene el scanning

    Cada advertisement capturado se pasa al callback on_advert, que debería
    alimentarlo al DedupEngine vía hash_mac + process_detection.
    """

    def __init__(
        self,
        on_advert: Optional[Callable[[BLEAdvertisement], None]] = None,
        scan_duration_seconds: float = 0,
    ) -> None:
        """Inicializa el scanner BLE.

        Args:
            on_advert: Callback para cada advertisement detectado.
            scan_duration_seconds: Cuánto escanear. 0 = escanea hasta que se llame stop().
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
        """Arranca el scanning BLE asíncrono."""
        if self._scan_thread is not None:
            logger.warning("ble_scan_already_running")
            return

        self._stop_event.clear()
        self._advert_count = 0

        self._scan_thread = threading.Thread(
            target=self._scan_thread_main, daemon=True, name="ble-scan"
        )
        self._scan_thread.start()
        logger.info("ble_scan_started")

    def stop(self) -> None:
        """Detiene el scanning BLE."""
        self._stop_event.set()
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=10.0)
            self._scan_thread = None
        logger.info(
            "ble_scan_stopped",
            extra={"count": self._advert_count},
        )

    def _scan_thread_main(self) -> None:
        """Corre el loop async de escaneo en un thread dedicado."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._scan_async())
        except Exception:
            logger.exception("BLE scan error")
        finally:
            loop.close()

    async def _scan_async(self) -> None:
        """Scanning BLE async usando bleak."""
        try:
            from bleak import BleakScanner
        except ImportError:
            logger.error(
                "bleak_not_installed",
                extra={"reason": "BLE scanning disabled"},
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
        logger.info("bleak_scanner_active")

        start_time = time.monotonic()
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)
            if self.scan_duration > 0 and (time.monotonic() - start_time) >= self.scan_duration:
                break

        await scanner.stop()

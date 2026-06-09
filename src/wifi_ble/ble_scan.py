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
from typing import Any, Callable, Optional

from src.wifi_ble.fingerprint import ble_fingerprint

logger = logging.getLogger(__name__)


def ble_address_is_random(device: Any) -> bool:
    """True si el address type BLE es ``random`` (RPA resolvable de iOS /
    aleatoria de Android = teléfono humano), False si ``public`` (OUI real
    del fabricante = infra/IoT fijo: beacons, smart-home, parlantes).

    Lee el tipo del backend BlueZ vía ``device.details['props']['AddressType']``.
    Si no se puede determinar (otro backend, estructura distinta), devuelve
    True para NO sobre-filtrar humanos — preferimos contar de más a tirar un
    teléfono real."""
    try:
        addr_type = device.details["props"]["AddressType"]
    except (AttributeError, TypeError, KeyError):
        return True
    return addr_type == "random"


@dataclass
class BLEAdvertisement:
    """Paquete de advertising BLE capturado."""

    mac: str
    rssi: float
    name: Optional[str]
    timestamp: float
    # Fingerprint estable (company ID + subtipos Continuity de Apple + service
    # UUIDs + TX power). Sobrevive la rotación de RPA → el dedup re-une las
    # rotaciones de un mismo teléfono. "" si no hay nada fingerprinteable.
    fingerprint: str = ""


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
        randomized_only: bool = True,
    ) -> None:
        """Inicializa el scanner BLE.

        Args:
            on_advert: Callback para cada advertisement detectado.
            scan_duration_seconds: Cuánto escanear. 0 = escanea hasta que se llame stop().
            randomized_only: Si True (default), solo cuenta dispositivos con
                address type *random* (RPA de iOS, randomizadas de Android =
                teléfonos humanos) y descarta los *public* (OUI real = infra/
                IoT fijo: beacons, smart-home, parlantes). Apagable por config.
        """
        self.on_advert = on_advert
        self.scan_duration = scan_duration_seconds
        self.randomized_only = randomized_only
        self._stop_event = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None
        self._advert_count = 0
        # Heartbeat para watchdog: el thread async actualiza este ts cada
        # ~0.5s mientras está vivo + funcional. Un supervisor externo
        # (ej. main loop o monitor de health) puede chequear
        # ``is_healthy(timeout)`` y, si False, llamar stop()+start() para
        # recover de un wedge de bleak/D-Bus sin matar el resto del
        # pipeline. 0.0 = nunca pulsó (no started yet).
        self._last_heartbeat_ts: float = 0.0

    @property
    def advert_count(self) -> int:
        return self._advert_count

    @property
    def is_running(self) -> bool:
        """True si el thread de scan está vivo y no fue señalizado para parar.

        Usado por la telemetría para reportar la health del subsistema BLE —
        si BlueZ murió o bleak crasheó, el thread sale y este flag pasa a False
        sin matar el pipeline de visión.
        """
        thread = self._scan_thread
        return (
            thread is not None and thread.is_alive() and not self._stop_event.is_set()
        )

    def is_healthy(self, timeout_seconds: float = 60.0) -> bool:
        """True si el scanner pulsó dentro de ``timeout_seconds``.

        Discrimina el wedge case del thread-vivo-pero-stuck (D-Bus
        wedged, bleak hangueado en una llamada interna): ``is_running``
        seguiría True pero el async loop no avanza. El heartbeat tickea
        cada ~0.5s mientras el loop async está vivo; si pasa más que
        ``timeout_seconds`` sin tick, el thread está wedgeado y un
        supervisor debe restartearlo.

        Devuelve False si el scanner nunca arrancó (``_last_heartbeat_ts == 0``).
        """
        if self._last_heartbeat_ts == 0.0:
            return False
        return (time.time() - self._last_heartbeat_ts) < timeout_seconds

    def start(self) -> None:
        """Arranca el scanning BLE asíncrono."""
        if self._scan_thread is not None and self._scan_thread.is_alive():
            logger.warning("ble_scan_already_running")
            return

        # Stop-event NUEVO por generación de thread. El thread y su detection
        # callback capturan ESTE evento (no releen self._stop_event): si un
        # stop() previo timeouteó el join (thread wedgeado en bleak/D-Bus),
        # el zombie quedó con SU evento ya set — cuando se des-wedgee sale
        # solo. Con un evento compartido, el clear() de este restart lo
        # resucitaba: dos BleakScanner activos (rate doble) y el zombie
        # tickeando el heartbeat compartido, enmascarando wedges futuros
        # del thread nuevo (anulaba el watchdog).
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._advert_count = 0
        # Heartbeat de arranque: evita el falso "wedged" en la ventana entre que
        # el thread arranca y el loop async toma el control (``await
        # scanner.start()`` puede tardar 1-3 s). El loop async lo sigue
        # actualizando cada ~0.5 s; si nunca llega a venir, se vuelve stale a los
        # 60 s y el watchdog restartea (comportamiento correcto).
        self._last_heartbeat_ts = time.time()

        self._scan_thread = threading.Thread(
            target=self._scan_thread_main,
            args=(stop_event,),
            daemon=True,
            name="ble-scan",
        )
        self._scan_thread.start()
        logger.info("ble_scan_started")

    def stop(self) -> None:
        """Detiene el scanning BLE."""
        self._stop_event.set()
        thread = self._scan_thread
        if thread is not None:
            thread.join(timeout=10.0)
            if thread.is_alive():
                # Wedge real: el join expiró con el thread vivo. Su stop_event
                # quedó set → al des-wedgearse sale solo, y como cada start()
                # crea un evento nuevo, un restart no puede resucitarlo.
                # daemon=True → tampoco bloquea el shutdown del proceso.
                logger.warning("ble_scan_thread_join_timeout_wedged")
            self._scan_thread = None
        logger.info(
            "ble_scan_stopped",
            extra={"count": self._advert_count},
        )

    def _scan_thread_main(self, stop_event: threading.Event) -> None:
        """Corre el loop async de escaneo en un thread dedicado."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._scan_async(stop_event))
        except Exception:
            logger.exception("BLE scan error")
        finally:
            loop.close()

    async def _scan_async(self, stop_event: threading.Event) -> None:
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
            # Evento de ESTA generación (closure) — un callback zombie de un
            # thread wedgeado ve su evento set y no procesa nada.
            if stop_event.is_set():
                return

            # Solo contar dispositivos "humanos": address type random (RPA/
            # aleatoria = teléfono). Los public (OUI real) son infra/IoT fijo.
            if self.randomized_only and not ble_address_is_random(device):
                return

            mac = device.address
            rssi = advertisement_data.rssi if advertisement_data.rssi else -100
            name = advertisement_data.local_name

            # Fingerprint estable de la advertising data (no del addr que rota).
            fingerprint = ble_fingerprint(
                getattr(advertisement_data, "manufacturer_data", None),
                getattr(advertisement_data, "service_uuids", None),
                getattr(advertisement_data, "tx_power", None),
            )

            advert = BLEAdvertisement(
                mac=mac,
                rssi=float(rssi),
                name=name,
                timestamp=time.time(),
                fingerprint=fingerprint,
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
        self._last_heartbeat_ts = time.time()
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
            # Heartbeat tick: el supervisor externo lee esto para detectar
            # wedges del thread sin afectar al pipeline.
            self._last_heartbeat_ts = time.time()
            if (
                self.scan_duration > 0
                and (time.monotonic() - start_time) >= self.scan_duration
            ):
                break

        await scanner.stop()

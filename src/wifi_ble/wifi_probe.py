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
    """Probe request WiFi capturado.

    ``seqnum`` (12 bits, top 12 del campo 802.11 Sequence Control) lo usa el
    DedupEngine para hacer stitching de MACs randomizadas del mismo chip — el
    seqnum es un contador del chip que tipicamente es continuo cross-MAC-change
    (Android y la mayoria de Apple H1 pre-iPhone 12). ``None`` si scapy no
    pudo parsear el campo (frame malformado o sin Dot11 header).
    """

    mac: str
    rssi: float
    ssid: str
    channel: int
    timestamp: float
    seqnum: int | None = None


def is_randomized_mac(mac: str) -> bool:
    """True si la MAC tiene el bit *locally administered* seteado (bit 1 del
    primer octeto, 0x02). iOS y Android randomizan la MAC de probe seteando
    ese bit → una MAC randomizada es casi siempre un teléfono humano. Las MAC
    *globales* (LA bit en 0) son la OUI quemada del fabricante → suelen ser
    infraestructura/IoT fijo (impresoras, smart-TVs, APs como clientes, etc.).
    Filtrar a randomizadas es el heurístico estándar para contar humanos.

    MAC malformada / vacía → False (no contar lo que no se puede parsear).
    """
    try:
        first_octet = int(mac.split(":")[0], 16)
    except (ValueError, AttributeError, IndexError):
        return False
    return bool(first_octet & 0x02)


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
        randomized_only: bool = True,
    ) -> None:
        self.interface = interface
        # Contar solo probes de MACs randomizadas (locally-administered) =
        # teléfonos humanos; descartar MACs globales = infra/IoT fijo. Ver
        # is_randomized_mac. Apagable por config para sites que necesiten todo.
        self.randomized_only = randomized_only
        # En RPi5 con CYW43455 + nexmon, el chip solo soporta una vif a
        # la vez — airmon-ng falla al crear ``wlan0mon`` como vif paralelo
        # con "Operation not supported (-95)". El approach correcto es
        # convertir la propia ``wlan0`` a monitor mode (mon_interface ==
        # interface). En arquitecturas con soporte dual-vif (Pi 3/4 con
        # algunos drivers más viejos), se puede sobreescribir manualmente.
        self.mon_interface = interface
        self.on_probe = on_probe
        self.hop_interval = hop_interval
        self.channels = (channels_24 or CHANNELS_24GHZ) + (channels_5 or CHANNELS_5GHZ)
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._hop_thread: Optional[threading.Thread] = None
        self._setup_thread: Optional[threading.Thread] = None
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

    # Timeout para los subprocess de setup. Si nexmon/brcmfmac crashearon,
    # airmon-ng e iw se cuelgan esperando respuesta del driver. Sin el
    # timeout, el pipeline entero queda bloqueado en build_wifi_ble antes
    # de llegar a mqtt_client.connect(). Con el timeout, levantamos
    # TimeoutExpired → RuntimeError → build_wifi_ble lo cachea en su
    # try/except y degrada (pipeline sigue sin WiFi probing).
    _SUBPROCESS_TIMEOUT_S = 15.0

    # Retry del setup de monitor mode. En el arranque del device, el
    # people-counter levanta ~8s después del boot, cuando NetworkManager /
    # wpa_supplicant todavía se están inicializando: el ``rfkill unblock`` toma,
    # pero NM RE-bloquea wlan0 segundos después y el ``ip link up`` falla con
    # "Operation not possible due to RF-kill". Una vez que NM se asienta el
    # unblock pega y se queda. Reintentar (re-unblock incluido) tolera ese race
    # sin depender del timing del boot — más robusto que ordenar por systemd.
    _MONITOR_SETUP_ATTEMPTS = 5
    _MONITOR_SETUP_RETRY_DELAY_S = 2.0

    def setup_monitor_mode(self) -> None:
        """Pone wlan0 en monitor mode, con retry para el race del boot.

        Reintenta ``_setup_monitor_once`` hasta ``_MONITOR_SETUP_ATTEMPTS``
        veces porque NetworkManager puede re-bloquear wlan0 vía rfkill justo
        después del unblock mientras se inicializa en el arranque. Cada intento
        re-hace el unblock + kill, así eventualmente le gana al race.

        Raises:
            RuntimeError: Si tras todos los reintentos sigue fallando (tool no
                instalado, firmware colgado, o el bloqueo no se libera).
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self._MONITOR_SETUP_ATTEMPTS + 1):
            try:
                self._setup_monitor_once()
                return
            except RuntimeError as e:
                last_err = e
                if attempt < self._MONITOR_SETUP_ATTEMPTS:
                    logger.warning(
                        "monitor_mode_setup_retry",
                        extra={
                            "attempt": attempt,
                            "max_attempts": self._MONITOR_SETUP_ATTEMPTS,
                            "error": str(e),
                        },
                    )
                    time.sleep(self._MONITOR_SETUP_RETRY_DELAY_S)
        assert last_err is not None
        raise last_err

    def _setup_monitor_once(self) -> None:
        """Un intento de poner wlan0 en monitor mode.

        Approach específico para RPi5 + CYW43455 + nexmon: el chip solo
        soporta una vif simultánea, así que en vez de crear un wlan0mon
        paralelo con airmon-ng (que tira EOPNOTSUPP -95), convertimos la
        propia wlan0 a monitor mode con ``iw set type``. El ``airmon-ng
        check kill`` previo sigue siendo necesario para matar NM/wpa que
        bloquean el cambio de type con EBUSY -16.

        Raises:
            RuntimeError: Si rfkill/iw falla, no está instalado, o se
                cuelga (driver/firmware no responde dentro del timeout).
        """
        try:
            # rfkill soft-block: systemd-rfkill restaura el estado anterior
            # del WiFi al boot. Si el device booteó con WiFi deshabilitado
            # (raspi-config nonint do_boot_behaviour B1 sin asociar a una
            # red), phy0 queda soft-blocked y los comandos iw se cuelgan
            # esperando respuesta del driver. Best-effort: si rfkill no
            # está instalado o el comando falla, seguimos.
            try:
                # ``all`` (no ``wifi``): en el arranque del device el bloqueo
                # puede estar en un switch rfkill que el alias ``wifi`` no
                # cubre; ``all`` los destraba todos (BT incluido — también lo
                # queremos para BLE). Verificado empíricamente que ``all``
                # libera cuando ``wifi`` no alcanza.
                subprocess.run(
                    ["rfkill", "unblock", "all"],
                    capture_output=True,
                    timeout=self._SUBPROCESS_TIMEOUT_S,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("rfkill unblock falló — continuando")

            # Mata NetworkManager + wpa_supplicant. Sin esto, ``iw set
            # type monitor`` falla con EBUSY (-16) porque NM tiene la
            # interface tomada para el flow managed.
            subprocess.run(
                ["airmon-ng", "check", "kill"],
                capture_output=True,
                timeout=self._SUBPROCESS_TIMEOUT_S,
            )

            # Bajar la interface, cambiar type, subirla. ``iw set type``
            # requiere la interface DOWN para no chocar con el station
            # mode actual.
            for cmd in (
                ["ip", "link", "set", self.interface, "down"],
                ["iw", "dev", self.interface, "set", "type", "monitor"],
                ["ip", "link", "set", self.interface, "up"],
            ):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._SUBPROCESS_TIMEOUT_S,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Comando falló: {' '.join(cmd)} — "
                        f"exit={result.returncode} "
                        f"stderr={result.stderr.strip()!r}"
                    )

            # nexmon: activar monitor mode CON radiotap en el FIRMWARE. Sin
            # esto, ``iw set type monitor`` deja el interface en type=monitor
            # pero el firmware sigue entregando frames Ethernet (el netdev
            # reporta DLT EN10MB, no IEEE802_11_RADIO) → scapy parsea Ether y
            # no ve ningún Dot11/probe. ``-m2`` = monitor + radiotap header.
            # (El capture loop además re-parsea los bytes crudos como RadioTap
            # porque el netdev igual reporta ARPHRD_ETHER — ver _capture_loop.)
            nx = subprocess.run(
                ["nexutil", "-m2"],
                capture_output=True,
                text=True,
                timeout=self._SUBPROCESS_TIMEOUT_S,
            )
            if nx.returncode != 0:
                raise RuntimeError(
                    f"nexutil -m2 falló (exit={nx.returncode} "
                    f"stderr={nx.stderr.strip()!r}). Sin monitor+radiotap del "
                    "firmware no hay captura WiFi."
                )

            # Verifica que el type sea monitor
            verify = subprocess.run(
                ["iw", "dev", self.interface, "info"],
                capture_output=True,
                text=True,
                timeout=self._SUBPROCESS_TIMEOUT_S,
            )
            if verify.returncode != 0 or "type monitor" not in verify.stdout:
                raise RuntimeError(
                    f"Interface {self.interface} no quedó en monitor mode. "
                    f"iw info: {verify.stdout!r}"
                )
            logger.info(
                "monitor_mode_enabled",
                extra={"interface": self.interface},
            )

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Required tool not found: {e}. "
                "Install with: sudo apt install aircrack-ng iw rfkill; "
                "nexutil se compila aparte (ver scripts/setup_device.sh)."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Monitor mode setup timed out after {self._SUBPROCESS_TIMEOUT_S}s "
                f"(driver/firmware not responding). Last command: {e.cmd}. "
                "Likely cause: nexmon firmware crashed — reboot to recover."
            ) from e

    def teardown_monitor_mode(self) -> None:
        """Detiene monitor mode y restaura la interfaz como managed."""
        try:
            # Revertir el flow del setup: down → set type managed → up.
            for cmd in (
                ["ip", "link", "set", self.interface, "down"],
                ["iw", "dev", self.interface, "set", "type", "managed"],
                ["ip", "link", "set", self.interface, "up"],
            ):
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self._SUBPROCESS_TIMEOUT_S,
                )
            # Restaura el manejo por NetworkManager (puede no estar
            # instalado en headless — best-effort).
            subprocess.run(
                ["nmcli", "dev", "set", self.interface, "managed", "yes"],
                capture_output=True,
                timeout=self._SUBPROCESS_TIMEOUT_S,
            )
            logger.info("monitor_mode_stopped")
        except Exception:
            logger.exception("Failed to restore managed mode")

    # Presupuesto del setup async. En el arranque del device, el radio
    # brcmfmac/CYW43455 sigue inicializando durante el primer ~minuto: el
    # rfkill queda soft-blocked y el unblock no toma efecto hasta que el
    # driver termina de levantar (independiente de NM/systemd-rfkill). En vez
    # de bloquear el pipeline (cámara/MQTT) esperando, hacemos el setup en un
    # thread y reintentamos pacientemente hasta este deadline. WiFi probing es
    # no-crítico: si no logra, el pipeline degrada (telemetría wifi=None).
    # Deadline generoso: en la Pi piloto el radio CYW43455 tardó ~5min reales
    # tras el boot en aceptar el unblock (el rfkill queda soft-blocked hasta que
    # el firmware termina de levantar). Como el setup es async (no bloquea el
    # pipeline), un deadline largo no cuesta nada y cubre boots lentos.
    _ASYNC_SETUP_DEADLINE_S = 420.0
    _ASYNC_SETUP_INTERVAL_S = 8.0

    def setup_and_start_async(self) -> None:
        """Arranca el setup de monitor mode + captura en un thread background.

        No bloquea al caller: el pipeline de visión sigue arrancando mientras
        el radio WiFi termina de inicializar. Reintenta hasta
        ``_ASYNC_SETUP_DEADLINE_S`` y, al lograr monitor mode, arranca la
        captura. Si se agota el deadline, deja WiFi probing off (degrade)."""
        if self._setup_thread is not None and self._setup_thread.is_alive():
            logger.warning("wifi_setup_already_running")
            return
        self._stop_event.clear()
        self._setup_thread = threading.Thread(
            target=self._setup_and_start_loop, daemon=True, name="wifi-setup"
        )
        self._setup_thread.start()

    def _setup_and_start_loop(self) -> None:
        deadline = time.monotonic() + self._ASYNC_SETUP_DEADLINE_S
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            try:
                self.setup_monitor_mode()
                self.start()
                logger.info(
                    "wifi_probe_capture_started_async",
                    extra={"interface": self.interface, "attempts": attempt},
                )
                return
            except RuntimeError as e:
                if time.monotonic() >= deadline:
                    logger.error(
                        "wifi_probe_setup_gave_up — sin WiFi probing tras %.0fs (%d intentos): %s",
                        self._ASYNC_SETUP_DEADLINE_S,
                        attempt,
                        e,
                    )
                    return
                logger.warning(
                    "wifi_setup_async_retry",
                    extra={"attempt": attempt, "error": str(e)},
                )
                # Espera interrumpible: si el pipeline para, salimos ya.
                if self._stop_event.wait(self._ASYNC_SETUP_INTERVAL_S):
                    return

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
        if self._setup_thread is not None:
            self._setup_thread.join(timeout=5.0)
            self._setup_thread = None
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
                    timeout=self._SUBPROCESS_TIMEOUT_S,
                )
                self._current_channel = channel
            except subprocess.CalledProcessError:
                logger.debug(
                    "channel_set_failed", extra={"channel": channel}
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "channel_set_timeout — driver no responde",
                    extra={"channel": channel},
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

        # Scapy loguea un WARNING ruidoso ("Socket ... failed with [Errno 100]
        # Network is down") cuando el teardown_monitor_mode baja wlan0 mientras
        # sniff() está bloqueado esperando un packet. Es esperable durante el
        # shutdown — el socket muere porque CAMBIAMOS el modo de la interface.
        # El warning sale del módulo scapy.sendrecv/scapy.arch, no del runtime,
        # así que silenciamos el logger padre "scapy" que cascadea a todos los
        # children — escribirlos uno por uno es frágil (puede cambiar entre
        # versiones de scapy).
        logging.getLogger("scapy").setLevel(logging.ERROR)

        def _handle_packet(pkt: Any) -> None:
            # El driver brcmfmac-nexmon entrega frames con header radiotap pero
            # reporta el netdev como ARPHRD_ETHER (DLT EN10MB), así que scapy
            # los parsea como Ethernet y no encuentra la capa Dot11. Re-
            # interpretamos los bytes crudos como RadioTap para recuperar el
            # 802.11. Si el setup ya entregara radiotap nativo (otro driver),
            # el primer haslayer(Dot11) corta sin re-parsear.
            if not pkt.haslayer(Dot11):
                try:
                    pkt = RadioTap(bytes(pkt))
                except Exception:
                    return
            if not pkt.haslayer(Dot11):
                return

            dot11 = pkt.getlayer(Dot11)

            # Filtra probe requests (type=0 management, subtype=4)
            if dot11.type != 0 or dot11.subtype != PROBE_REQUEST_SUBTYPE:
                return

            mac = dot11.addr2
            if mac is None:
                return

            # Solo contar dispositivos "humanos": MACs randomizadas (LA bit).
            # Las MACs globales son infra/IoT fijo (ver is_randomized_mac).
            if self.randomized_only and not is_randomized_mac(mac):
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

            # Sequence number: campo Sequence Control de 16 bits — top 12 son
            # seqnum, bottom 4 son fragnum. Lo usa DedupEngine para hacer
            # stitching de MACs randomizadas (el seqnum es del chip, no de la
            # MAC, asi que tipicamente es continuo cross-rotation).
            seqnum: int | None = None
            try:
                sc = dot11.SC
                if sc is not None:
                    seqnum = int(sc) >> 4
            except (AttributeError, TypeError):
                pass

            event = ProbeEvent(
                mac=mac,
                rssi=rssi,
                ssid=ssid,
                channel=self._current_channel,
                timestamp=time.time(),
                seqnum=seqnum,
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

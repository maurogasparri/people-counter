"""Cliente MQTT para AWS IoT Core con reconexión automática y replay del buffer.

Usa paho-mqtt 2.0+ con TLS mutual authentication (certs cliente X.509).
Se integra con MessageBuffer para resiliencia ante pérdida de conectividad.
"""

import json
import logging
import random
import ssl
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.mqtt.buffer import MessageBuffer

logger = logging.getLogger(__name__)

# Parámetros de reconexión. paho 2.x maneja el backoff internamente con
# jitter (random.uniform en _reconnect_wait) entre min y max — efectivamente
# exponencial con spread, ~suficiente para que una flota no haga DDoS al
# broker tras un outage del backend. Max 60s para que el recovery se sienta
# rápido cuando el outage termina, balanceado con el costo del retry.
RECONNECT_MIN_DELAY = 1  # segundos
RECONNECT_MAX_DELAY = 60  # segundos

# Replay del outbox. El batch pagina el backlog (un outage multi-día acumula
# miles de filas — cargarlas todas de una infla memoria y la cola interna de
# paho); entre batches se espera a que los PUBACKs pendientes bajen del
# low-water antes de seguir, así el drain avanza al ritmo del broker. Si el
# broker deja de ACKear por más del timeout, el replay corta y el próximo
# reconnect retoma donde quedó (las filas siguen sent=0).
REPLAY_BATCH_SIZE = 200
REPLAY_INFLIGHT_LOW_WATER = 50
REPLAY_ACK_DRAIN_TIMEOUT_S = 30.0


class MQTTClient:
    """Cliente MQTT para AWS IoT Core con buffering local.

    Features:
        - TLS mutual auth con certificados X.509.
        - QoS 1 para delivery garantizado.
        - Buffer SQLite local para resiliencia offline.
        - Reconexión automática con backoff exponencial.
        - Replay del buffer al reconectar.
    """

    def __init__(
        self,
        device_id: str,
        endpoint: str,
        port: int,
        cert_path: str,
        key_path: str,
        ca_path: str,
        buffer: MessageBuffer,
        topics: Optional[dict[str, str]] = None,
    ) -> None:
        """Inicializa el cliente MQTT.

        Args:
            device_id: Identificador único del dispositivo (se usa como client_id).
            endpoint: Endpoint de AWS IoT Core (xxxxx.iot.region.amazonaws.com).
            port: Puerto MQTT (8883 para TLS).
            cert_path: Ruta al certificado del dispositivo (.pem.crt).
            key_path: Ruta a la clave privada del dispositivo (.pem.key).
            ca_path: Ruta al certificado Amazon Root CA.
            buffer: Instancia de MessageBuffer para persistencia local.
            topics: Dict que mapea nombres lógicos a strings de topic MQTT.
        """
        self.device_id = device_id
        self.endpoint = endpoint
        self.port = port
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.buffer = buffer
        self.topics = topics or {}

        self._connected = False
        self._stop_event = threading.Event()
        self._replay_lock = threading.Lock()
        self._pending_acks: dict[int, int] = {}  # mqtt_mid -> buffer_msg_id
        self._pending_lock = threading.Lock()
        # Callbacks de shadow-delta registrados por topic MQTT para poder
        # dispatchear mensajes entrantes sin depender del wiring per-sub de paho.
        self._shadow_callbacks: dict[str, Callable[[dict], None]] = {}
        self._shadow_lock = threading.Lock()
        # Hook opcional que se dispara en cada (re)conexión exitosa. Corre en el
        # thread de red de paho, así que los callers tienen que mantener el callback
        # rápido + thread-safe (típicamente: enqueue un job para que lo drene el
        # main loop).
        self.on_connected: Optional[Callable[[], None]] = None
        # Telemetría de conectividad: se incrementa en cada callback de disconnect,
        # reconnect_ts se stampa en cada conexión exitosa. Read-only desde el
        # main thread vía las properties públicas.
        self._disconnect_count = 0
        self._reconnect_ts: float | None = None
        self._conn_lock = threading.Lock()

        # Valida que los archivos de certificado existan
        for name, path in [
            ("cert", cert_path),
            ("key", key_path),
            ("ca", ca_path),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"MQTT {name} file not found: {path}")

        # Crea el cliente paho (API v2)
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=device_id,
            protocol=mqtt.MQTTv311,
        )

        # Configura TLS
        self._client.tls_set(
            ca_certs=ca_path,
            certfile=cert_path,
            keyfile=key_path,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

        # Delega reconnect/backoff a la lógica built-in de paho (activa mientras
        # loop_start() esté corriendo). Evita acumular threads custom de reconexión.
        self._client.reconnect_delay_set(
            min_delay=RECONNECT_MIN_DELAY, max_delay=RECONNECT_MAX_DELAY
        )

        # Setea callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish
        self._client.on_message = self._on_message

        logger.info(
            "MQTT client initialized",
            extra={"device_id": device_id, "endpoint": endpoint},
        )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def disconnect_count(self) -> int:
        """Cuenta acumulada de disconnects inesperados desde la creación del cliente."""
        with self._conn_lock:
            return self._disconnect_count

    @property
    def reconnect_ts(self) -> float | None:
        """Epoch seconds del último (re)conexión exitosa, o None."""
        with self._conn_lock:
            return self._reconnect_ts

    def connect(self, startup_jitter_seconds: float = 0.0) -> None:
        """Conecta al broker MQTT.

        No-bloqueante: el connect lo establece el network loop de paho en
        background vía connect_async() + loop_start(). connect_async NO resuelve
        DNS sincrónicamente ni levanta excepción si la red/DNS no está lista
        todavía (típico apenas bootea el device) — encola la conexión y el loop
        la establece y la reintenta con el backoff de reconnect_delay_set.

        Esto evita el modo de falla observado en la validación en las instalaciones (2026-05-31): una race de
        DNS al arrancar hacía que el connect() sync tirara getaddrinfo y matara el
        thread de conexión, dejando MQTT caído sin reconexión hasta un restart
        manual mientras el outbox se llenaba (el auto-reconnect de paho sólo actúa
        DESPUÉS de un connect inicial exitoso con el loop corriendo).

        Args:
            startup_jitter_seconds: Si >0, esperá entre 0 y este valor de
                segundos (uniform random) antes de iniciar el connect.
                Anti thundering-herd cuando la flota entera bootea después
                de un outage (regional incident, mass restart del retail
                chain) — staggerea los initial TLS handshakes contra IoT
                Core para no DDoSearlo. La espera corre en un thread
                background, así connect() retorna inmediatamente y el
                pipeline puede arrancar sin bloqueo (los eventos se
                bufferean local hasta que la conexión finalice).
        """
        if startup_jitter_seconds > 0:
            delay = random.uniform(0.0, float(startup_jitter_seconds))

            def _delayed_connect() -> None:
                time.sleep(delay)
                self._start_async_connect(jitter=delay)

            threading.Thread(
                target=_delayed_connect, daemon=True, name="mqtt-connect"
            ).start()
            logger.info(
                "MQTT connect scheduled with %.1fs startup jitter "
                "(anti thundering-herd)",
                delay,
            )
            return

        self._start_async_connect()

    def _start_async_connect(self, jitter: float = 0.0) -> None:
        """Arranca el connect vía connect_async() + loop_start().

        Delega el connect inicial Y los reintentos al network loop de paho (con
        el backoff de reconnect_delay_set). A diferencia del connect() sync, no
        levanta por fallos de red/DNS transitorios: el loop reintenta solo. No
        re-levanta — el pipeline corre y bufferea local hasta que conecte.
        """
        try:
            self._client.connect_async(self.endpoint, self.port, keepalive=60)
            self._client.loop_start()
        except Exception:
            # connect_async sólo debería tirar por errores de programación (args
            # inválidos), no por red. Logueamos sin re-levantar para no voltear el
            # pipeline; el buffer local cubre la indisponibilidad.
            logger.exception("MQTT connect_async failed")
            return
        if jitter > 0:
            logger.info(
                "MQTT connecting (async) to %s:%d (after %.1fs startup jitter)",
                self.endpoint,
                self.port,
                jitter,
            )
        else:
            logger.info("MQTT connecting (async) to %s:%d", self.endpoint, self.port)

    def disconnect(self) -> None:
        """Desconecta de manera limpia."""
        self._stop_event.set()
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False
        logger.info("MQTT disconnected")

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        qos: int = 1,
    ) -> Optional[int]:
        """Publica un mensaje, buffereándolo localmente primero.

        El mensaje siempre se escribe al buffer local antes que nada.
        Si está conectado, también se envía inmediatamente vía MQTT.
        La entrada del buffer se marca como enviada solo después del PUBACK.

        Args:
            topic: String del topic MQTT.
            payload: Dict a serializar como JSON.
            qos: Nivel QoS de MQTT (default 1).

        Returns:
            ID del mensaje en el buffer, o None si falla el buffer.
        """
        # Siempre bufferear primero
        try:
            msg_id = self.buffer.enqueue(topic, payload)
        except Exception:
            logger.exception("Falló bufferear el mensaje")
            return None

        if self._connected:
            self._send_buffered_message(msg_id, topic, payload, qos)

        return msg_id

    def publish_event(
        self,
        event_type: str,
        data: dict[str, Any],
        qos: int = 1,
    ) -> Optional[int]:
        """Publica usando un nombre lógico de topic del config.

        Método de conveniencia que busca el topic en self.topics.

        Args:
            event_type: Nombre lógico que matchea una key en self.topics
                (ej: "counting", "wifi_ble", "telemetry").
            data: Dict del payload.
            qos: Nivel QoS de MQTT.

        Returns:
            ID del mensaje en el buffer.
        """
        topic = self.topics.get(event_type)
        if not topic:
            logger.error("Tipo de evento desconocido: %s", event_type)
            return None

        # Agrega metadata estándar
        payload = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        }

        return self.publish(topic, payload, qos)

    def subscribe_shadow_delta(
        self,
        thing_name: str,
        callback: Callable[[dict], None],
    ) -> None:
        """Suscribe al topic delta del shadow del dispositivo.

        Cuando AWS IoT publica un delta (desired != reported), se invoca el
        callback con el dict ``state`` parseado. El callback corre en el thread
        de red de paho; tiene que ser rápido y thread-safe.

        Args:
            thing_name: Nombre del thing en AWS IoT (típicamente el device_id).
            callback: Se invoca con el dict del delta ``state`` parseado.
        """
        topic = f"$aws/things/{thing_name}/shadow/update/delta"
        with self._shadow_lock:
            self._shadow_callbacks[topic] = callback
        try:
            result, _mid = self._client.subscribe(topic, qos=1)
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(
                    "Shadow delta subscribe returned rc=%d for %s",
                    result,
                    topic,
                )
            else:
                logger.info("Subscribed to shadow delta: %s", topic)
        except Exception:
            logger.exception("Shadow delta subscribe failed: %s", topic)

    def publish_shadow_reported(
        self,
        thing_name: str,
        state: dict[str, Any],
    ) -> None:
        """Publica un update al estado ``reported`` del shadow del dispositivo.

        Envuelve ``state`` en el envelope requerido ``{"state": {"reported": ...}}``
        y publica a ``$aws/things/{thing}/shadow/update``.

        Args:
            thing_name: Nombre del thing en AWS IoT (típicamente el device_id).
            state: Dict de valores a reportar.
        """
        topic = f"$aws/things/{thing_name}/shadow/update"
        envelope = {"state": {"reported": state}}
        try:
            result = self._client.publish(
                topic, json.dumps(envelope), qos=1, retain=False
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning("Shadow reported publish rc=%d for %s", result.rc, topic)
            else:
                logger.debug("Shadow reported published to %s", topic)
        except Exception:
            logger.exception("Shadow reported publish failed: %s", topic)

    def replay_buffer(self) -> int:
        """Reenvía TODOS los mensajes pendientes del buffer, en batches.

        Se llama automáticamente al reconectar. También puede llamarse manual.
        Pagina por id con ``after_id`` (las filas recién publicadas siguen
        ``sent=0`` hasta su PUBACK — re-leer solo por ``sent`` haría loop
        sobre el mismo batch) y entre batches espera a que los ACKs drenen
        (ver constantes REPLAY_*). Un corte de conexión o un broker que no
        ACKea interrumpen el drain; el próximo reconnect retoma desde donde
        quedó porque las filas no enviadas siguen ``sent=0``.

        Returns:
            Cantidad de mensajes reenviados.
        """
        with self._replay_lock:
            total = 0
            after_id = 0
            while not self._stop_event.is_set() and self._connected:
                pending = self.buffer.get_pending(
                    limit=REPLAY_BATCH_SIZE, after_id=after_id
                )
                if not pending:
                    break

                # Saltear los msg_ids que ya están in-flight (publicados,
                # esperando PUBACK). En un reconnect rápido un mensaje recién
                # publicado sigue con sent=0 hasta que llega su PUBACK; sin
                # este filtro se re-publica (duplicado — QoS 1 lo tolera pero
                # ensucia el tráfico y el mapeo de _pending_acks). El PUBACK
                # pendiente lo marcará sent.
                with self._pending_lock:
                    in_flight = set(self._pending_acks.values())

                for msg_id, topic, payload in pending:
                    if self._stop_event.is_set() or not self._connected:
                        break
                    if msg_id in in_flight:
                        continue
                    self._send_buffered_message(msg_id, topic, payload, qos=1)
                    total += 1

                after_id = pending[-1][0]
                if len(pending) < REPLAY_BATCH_SIZE:
                    break  # último batch — no quedan filas que paginar
                if not self._wait_inflight_drain():
                    logger.warning(
                        "Buffer replay interrumpido: el broker no drena los "
                        "ACKs in-flight (%d enviados hasta acá; el próximo "
                        "reconnect retoma)",
                        total,
                    )
                    break

            logger.info("Buffer replay: %d messages sent", total)
            return total

    def _wait_inflight_drain(self) -> bool:
        """Espera a que los PUBACKs pendientes bajen del low-water mark.

        Returns:
            True si drenó (o el cliente se paró/desconectó — el loop del
            caller re-chequea esas condiciones); False si expiró el timeout
            con el broker sin ACKear.
        """
        deadline = time.time() + REPLAY_ACK_DRAIN_TIMEOUT_S
        while time.time() < deadline:
            if self._stop_event.is_set() or not self._connected:
                return True
            with self._pending_lock:
                if len(self._pending_acks) <= REPLAY_INFLIGHT_LOW_WATER:
                    return True
            time.sleep(0.1)
        return False

    # --- Métodos internos ---

    def _send_buffered_message(
        self,
        msg_id: int,
        topic: str,
        payload: dict[str, Any],
        qos: int,
    ) -> None:
        """Envía un único mensaje buffereado. Se marca enviado solo al recibir PUBACK."""
        try:
            result = self._client.publish(topic, json.dumps(payload), qos=qos)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                # Trackea mid -> buffer_msg_id; mark_sent ocurre en _on_publish
                with self._pending_lock:
                    self._pending_acks[result.mid] = msg_id
            else:
                logger.warning(
                    "MQTT publish failed: rc=%d, msg_id=%d",
                    result.rc,
                    msg_id,
                )
        except Exception:
            logger.exception("MQTT send error for msg_id=%d", msg_id)

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Se llama cuando se establece la conexión."""
        if rc == 0:
            self._connected = True
            with self._conn_lock:
                self._reconnect_ts = time.time()
            logger.info("MQTT conectado a %s", self.endpoint)

            # Reenvía los mensajes buffereados
            threading.Thread(target=self.replay_buffer, daemon=True).start()

            # Dispara el hook externo on_connected (ej: reconciliación de shadow).
            # Las excepciones acá no deben romper el loop de paho.
            hook = self.on_connected
            if hook is not None:
                try:
                    hook()
                except Exception:
                    logger.exception("on_connected hook raised")
        else:
            logger.error("MQTT connect failed: rc=%s", rc)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        rc: int = 0,
        properties: Any = None,
    ) -> None:
        """Se llama al desconectarse. paho maneja el reconnect mientras loop_start esté activo."""
        self._connected = False
        if self._stop_event.is_set():
            return
        with self._conn_lock:
            self._disconnect_count += 1
        logger.warning("MQTT disconnected: rc=%s (paho will reconnect)", rc)

    def _on_publish(
        self,
        client: mqtt.Client,
        userdata: Any,
        mid: int,
        rc: int = 0,
        properties: Any = None,
    ) -> None:
        """Se llama tras un publish exitoso (PUBACK para QoS 1)."""
        with self._pending_lock:
            buf_id = self._pending_acks.pop(mid, None)
        if buf_id is None:
            # Race publish→registración: el mid se registra DESPUÉS de que
            # publish() retorna, y este callback corre en el thread de red de
            # paho — un PUBACK ultrarrápido puede llegar antes del registro.
            # Un único retry tras 1ms cubre esa ventana (microsegundos). No
            # se puede holdear _pending_lock a través del publish(): paho
            # invoca este callback con sus locks internos tomados y publish()
            # los necesita → deadlock por inversión de orden. Si el retry
            # también falla, la fila queda sent=0 y se re-publica en el
            # próximo replay — duplicado inocuo (ON CONFLICT server-side).
            time.sleep(0.001)
            with self._pending_lock:
                buf_id = self._pending_acks.pop(mid, None)
        if buf_id is not None:
            self.buffer.mark_sent(buf_id)
            logger.debug("MQTT PUBACK received: mid=%d, buffer_id=%d", mid, buf_id)
        else:
            logger.debug("MQTT PUBACK received: mid=%d (no pending buffer entry)", mid)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        """Dispatchea mensajes entrantes a los callbacks de shadow registrados.

        Corre en el thread de red de paho. Los errores de decode JSON y las
        excepciones del callback se loggean pero nunca se propagan, así el
        loop de MQTT sigue sano.
        """
        topic = message.topic
        with self._shadow_lock:
            callback = self._shadow_callbacks.get(topic)
        if callback is None:
            logger.debug("Mensaje MQTT en topic no manejado: %s", topic)
            return

        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning("JSON inválido en topic shadow %s: %s", topic, e)
            return

        state = payload.get("state", payload)
        try:
            callback(state if isinstance(state, dict) else {})
        except Exception:
            logger.exception("Shadow delta callback raised on %s", topic)

    def __enter__(self) -> "MQTTClient":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()

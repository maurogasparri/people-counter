"""Cliente MQTT para AWS IoT Core con reconexión automática y replay del buffer.

Usa paho-mqtt 2.0+ con TLS mutual authentication (certs cliente X.509).
Se integra con MessageBuffer para resiliencia ante pérdida de conectividad.
"""

import json
import logging
import ssl
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.mqtt.buffer import MessageBuffer

logger = logging.getLogger(__name__)

# Parámetros de reconexión (se pasan al reconnect_delay_set built-in de paho)
RECONNECT_MIN_DELAY = 1  # segundos
RECONNECT_MAX_DELAY = 120  # segundos


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

    def connect(self) -> None:
        """Conecta al broker MQTT.

        No-bloqueante: arranca el network loop en un thread background.
        """
        try:
            self._client.connect(self.endpoint, self.port, keepalive=60)
            self._client.loop_start()
            logger.info("MQTT connecting to %s:%d", self.endpoint, self.port)
        except Exception:
            logger.exception("MQTT connect failed")
            raise

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
                logger.warning(
                    "Shadow reported publish rc=%d for %s", result.rc, topic
                )
            else:
                logger.debug("Shadow reported published to %s", topic)
        except Exception:
            logger.exception("Shadow reported publish failed: %s", topic)

    def replay_buffer(self) -> int:
        """Reenvía todos los mensajes pendientes del buffer.

        Se llama automáticamente al reconectar. También puede llamarse manual.

        Returns:
            Cantidad de mensajes reenviados.
        """
        with self._replay_lock:
            pending = self.buffer.get_pending(limit=200)
            if not pending:
                return 0

            count = 0
            for msg_id, topic, payload in pending:
                if self._stop_event.is_set() or not self._connected:
                    break
                self._send_buffered_message(msg_id, topic, payload, qos=1)
                count += 1

            logger.info("Buffer replay: %d messages sent", count)
            return count

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
            result = self._client.publish(
                topic, json.dumps(payload), qos=qos
            )
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
            threading.Thread(
                target=self.replay_buffer, daemon=True
            ).start()

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
            logger.warning(
                "JSON inválido en topic shadow %s: %s", topic, e
            )
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

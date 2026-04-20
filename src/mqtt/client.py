"""AWS IoT Core MQTT client with automatic reconnection and buffer replay.

Uses paho-mqtt 2.0+ with TLS mutual authentication (X.509 client certs).
Integrates with MessageBuffer for resilience against connectivity loss.
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

# Reconnect parameters (passed to paho's built-in reconnect_delay_set)
RECONNECT_MIN_DELAY = 1  # seconds
RECONNECT_MAX_DELAY = 120  # seconds


class MQTTClient:
    """MQTT client for AWS IoT Core with local buffering.

    Features:
        - TLS mutual auth with X.509 certificates.
        - QoS 1 for guaranteed delivery.
        - Local SQLite buffer for offline resilience.
        - Automatic reconnection with exponential backoff.
        - Buffer replay on reconnect.
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
        """Initialize MQTT client.

        Args:
            device_id: Unique device identifier (used as client_id).
            endpoint: AWS IoT Core endpoint (xxxxx.iot.region.amazonaws.com).
            port: MQTT port (8883 for TLS).
            cert_path: Path to device certificate (.pem.crt).
            key_path: Path to device private key (.pem.key).
            ca_path: Path to Amazon Root CA certificate.
            buffer: MessageBuffer instance for local persistence.
            topics: Dict mapping logical names to MQTT topic strings.
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
        # Registered shadow-delta callbacks keyed by MQTT topic so we can
        # dispatch incoming messages without relying on paho's per-sub wiring.
        self._shadow_callbacks: dict[str, Callable[[dict], None]] = {}
        self._shadow_lock = threading.Lock()
        # Optional hook fired on every successful (re)connect. Runs in the paho
        # network thread, so callers must keep the callback fast + thread-safe
        # (typically: enqueue a job for the main loop to drain).
        self.on_connected: Optional[Callable[[], None]] = None
        # Connectivity telemetry: incremented on every disconnect callback,
        # reconnect_ts stamped on each successful connect. Read-only from the
        # main thread via the public properties.
        self._disconnect_count = 0
        self._reconnect_ts: float | None = None
        self._conn_lock = threading.Lock()

        # Validate certificate files exist
        for name, path in [
            ("cert", cert_path),
            ("key", key_path),
            ("ca", ca_path),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"MQTT {name} file not found: {path}")

        # Create paho client (v2 API)
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=device_id,
            protocol=mqtt.MQTTv311,
        )

        # Configure TLS
        self._client.tls_set(
            ca_certs=ca_path,
            certfile=cert_path,
            keyfile=key_path,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

        # Delegate reconnect/backoff to paho's built-in logic (active while
        # loop_start() is running). Avoids stacking custom reconnect threads.
        self._client.reconnect_delay_set(
            min_delay=RECONNECT_MIN_DELAY, max_delay=RECONNECT_MAX_DELAY
        )

        # Set callbacks
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
        """Cumulative count of unexpected disconnects since client creation."""
        with self._conn_lock:
            return self._disconnect_count

    @property
    def reconnect_ts(self) -> float | None:
        """Epoch seconds of the last successful (re)connect, or None."""
        with self._conn_lock:
            return self._reconnect_ts

    def connect(self) -> None:
        """Connect to the MQTT broker.

        Non-blocking: starts the network loop in a background thread.
        """
        try:
            self._client.connect(self.endpoint, self.port, keepalive=60)
            self._client.loop_start()
            logger.info("MQTT connecting to %s:%d", self.endpoint, self.port)
        except Exception:
            logger.exception("MQTT connect failed")
            raise

    def disconnect(self) -> None:
        """Gracefully disconnect."""
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
        """Publish a message, buffering locally first.

        The message is always written to the local buffer first.
        If connected, it's also sent immediately via MQTT.
        The buffer entry is marked as sent only after PUBACK.

        Args:
            topic: MQTT topic string.
            payload: Dict to serialize as JSON.
            qos: MQTT QoS level (default 1).

        Returns:
            Buffer message ID, or None on buffer failure.
        """
        # Always buffer first
        try:
            msg_id = self.buffer.enqueue(topic, payload)
        except Exception:
            logger.exception("Failed to buffer message")
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
        """Publish using a logical topic name from config.

        Convenience method that looks up the topic from self.topics.

        Args:
            event_type: Logical name matching a key in self.topics
                (e.g. "counting", "wifi_ble", "telemetry").
            data: Payload dict.
            qos: MQTT QoS level.

        Returns:
            Buffer message ID.
        """
        topic = self.topics.get(event_type)
        if not topic:
            logger.error("Unknown event type: %s", event_type)
            return None

        # Add standard metadata
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
        """Subscribe to the device shadow delta topic.

        When AWS IoT publishes a delta (desired != reported), the callback
        is invoked with the parsed ``state`` dict.  The callback runs in
        the paho network thread; it should be fast and thread-safe.

        Args:
            thing_name: AWS IoT thing name (typically the device_id).
            callback: Invoked with the parsed ``state`` delta dict.
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
        """Publish an update to the device shadow ``reported`` state.

        Wraps ``state`` in the required ``{"state": {"reported": ...}}``
        envelope and publishes to ``$aws/things/{thing}/shadow/update``.

        Args:
            thing_name: AWS IoT thing name (typically the device_id).
            state: Dict of values to report.
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
        """Replay all pending messages from the buffer.

        Called automatically on reconnect. Can also be called manually.

        Returns:
            Number of messages replayed.
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

    # --- Internal methods ---

    def _send_buffered_message(
        self,
        msg_id: int,
        topic: str,
        payload: dict[str, Any],
        qos: int,
    ) -> None:
        """Send a single buffered message. Marked as sent only on PUBACK."""
        try:
            result = self._client.publish(
                topic, json.dumps(payload), qos=qos
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                # Track mid -> buffer_msg_id; mark_sent happens in _on_publish
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
        """Called when connection is established."""
        if rc == 0:
            self._connected = True
            with self._conn_lock:
                self._reconnect_ts = time.time()
            logger.info("MQTT connected to %s", self.endpoint)

            # Replay buffered messages
            threading.Thread(
                target=self.replay_buffer, daemon=True
            ).start()

            # Fire the external on_connected hook (e.g. shadow reconciliation).
            # Exceptions here must not break the paho loop.
            hook = self.on_connected
            if hook is not None:
                try:
                    hook()
                except Exception:
                    logger.exception("on_connected hook raised")
        else:
            logger.error("MQTT connect failed: rc=%d", rc)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        rc: int = 0,
        properties: Any = None,
    ) -> None:
        """Called when disconnected. paho handles reconnect while loop_start is active."""
        self._connected = False
        if self._stop_event.is_set():
            return
        with self._conn_lock:
            self._disconnect_count += 1
        logger.warning("MQTT disconnected: rc=%d (paho will reconnect)", rc)

    def _on_publish(
        self,
        client: mqtt.Client,
        userdata: Any,
        mid: int,
        rc: int = 0,
        properties: Any = None,
    ) -> None:
        """Called on successful publish (PUBACK for QoS 1)."""
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
        """Dispatch incoming messages to registered shadow callbacks.

        Runs in the paho network thread.  JSON decode errors and
        callback exceptions are logged but never propagated so the
        MQTT loop stays healthy.
        """
        topic = message.topic
        with self._shadow_lock:
            callback = self._shadow_callbacks.get(topic)
        if callback is None:
            logger.debug("MQTT message on unhandled topic: %s", topic)
            return

        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(
                "Invalid JSON on shadow topic %s: %s", topic, e
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

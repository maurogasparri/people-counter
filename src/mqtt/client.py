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

# Reconnect parameters
RECONNECT_MIN_DELAY = 1  # seconds
RECONNECT_MAX_DELAY = 120  # seconds
RECONNECT_BACKOFF = 2  # multiplier


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
        self._reconnect_delay = RECONNECT_MIN_DELAY
        self._pending_acks: dict[int, int] = {}  # mqtt_mid -> buffer_msg_id
        self._pending_lock = threading.Lock()

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

        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

        logger.info(
            "MQTT client initialized",
            extra={"device_id": device_id, "endpoint": endpoint},
        )

    @property
    def connected(self) -> bool:
        return self._connected

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
            self._reconnect_delay = RECONNECT_MIN_DELAY
            logger.info("MQTT connected to %s", self.endpoint)

            # Replay buffered messages
            threading.Thread(
                target=self.replay_buffer, daemon=True
            ).start()
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
        """Called when disconnected. Handles reconnection."""
        self._connected = False

        if self._stop_event.is_set():
            return  # Intentional disconnect

        logger.warning("MQTT disconnected: rc=%d, reconnecting...", rc)
        self._reconnect_with_backoff()

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

    def _reconnect_with_backoff(self) -> None:
        """Attempt reconnection with exponential backoff."""

        def _reconnect_loop() -> None:
            delay = self._reconnect_delay
            while not self._stop_event.is_set() and not self._connected:
                logger.info("Reconnecting in %d seconds...", delay)
                self._stop_event.wait(delay)
                if self._stop_event.is_set():
                    return
                try:
                    self._client.reconnect()
                    return
                except Exception:
                    logger.warning("Reconnect attempt failed")
                    delay = min(delay * RECONNECT_BACKOFF, RECONNECT_MAX_DELAY)

        threading.Thread(target=_reconnect_loop, daemon=True).start()

    def __enter__(self) -> "MQTTClient":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()

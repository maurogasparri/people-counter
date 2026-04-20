"""Tests for MQTT client.

Tests client construction and publish-to-buffer logic without
requiring a real MQTT broker or AWS IoT Core endpoint.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.mqtt.buffer import MessageBuffer


class TestMQTTClientConstruction:
    """Test MQTTClient initialization and configuration."""

    def _make_certs(self, tmpdir: str) -> tuple[str, str, str]:
        """Create dummy certificate files."""
        cert = str(Path(tmpdir) / "device.pem.crt")
        key = str(Path(tmpdir) / "device.pem.key")
        ca = str(Path(tmpdir) / "AmazonRootCA1.pem")
        for f in [cert, key, ca]:
            Path(f).write_text("dummy")
        return cert, key, ca

    def test_missing_cert_raises(self):
        """Should raise FileNotFoundError if certs don't exist."""
        from src.mqtt.client import MQTTClient

        tmpdir = tempfile.mkdtemp()
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        with pytest.raises(FileNotFoundError, match="cert"):
            MQTTClient(
                device_id="test-001",
                endpoint="test.iot.us-east-1.amazonaws.com",
                port=8883,
                cert_path="/nonexistent/cert.pem",
                key_path="/nonexistent/key.pem",
                ca_path="/nonexistent/ca.pem",
                buffer=buffer,
            )

    @patch("src.mqtt.client.mqtt.Client")
    def test_construction_success(self, mock_mqtt_cls):
        """Client initializes correctly with valid config."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="test.iot.us-east-1.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
            topics={"counting": "store/001/counting"},
        )

        assert client.device_id == "test-001"
        assert not client.connected
        mock_client.tls_set.assert_called_once()

    @patch("src.mqtt.client.mqtt.Client")
    def test_publish_buffers_first(self, mock_mqtt_cls):
        """Publish always buffers locally, even if disconnected."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="test.iot.us-east-1.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        # Publish while disconnected — should buffer only
        msg_id = client.publish("test/topic", {"count": 42})
        assert msg_id is not None

        # Verify it's in the buffer
        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0][1] == "test/topic"
        assert pending[0][2]["count"] == 42

    @patch("src.mqtt.client.mqtt.Client")
    def test_publish_event_adds_metadata(self, mock_mqtt_cls):
        """publish_event wraps data with device_id and timestamp."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="test.iot.us-east-1.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
            topics={"counting": "store/001/counting"},
        )

        client.publish_event("counting", {"direction": "in"})

        pending = buffer.get_pending()
        assert len(pending) == 1
        payload = pending[0][2]
        assert payload["device_id"] == "test-001"
        assert payload["type"] == "counting"
        assert "timestamp" in payload
        assert payload["data"]["direction"] == "in"

    @patch("src.mqtt.client.mqtt.Client")
    def test_subscribe_shadow_delta_subscribes_and_routes(self, mock_mqtt_cls):
        """subscribe_shadow_delta wires to the correct topic and invokes callback."""
        import json as _json

        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_client.subscribe.return_value = (0, 1)  # MQTT_ERR_SUCCESS, mid
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="test.iot.us-east-1.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        received: list[dict] = []
        client.subscribe_shadow_delta("test-001", received.append)

        expected_topic = "$aws/things/test-001/shadow/update/delta"
        mock_client.subscribe.assert_called_once_with(expected_topic, qos=1)

        # Simulate an inbound delta message through paho's on_message hook.
        msg = MagicMock()
        msg.topic = expected_topic
        msg.payload = _json.dumps(
            {"state": {"telemetry": {"interval_seconds": 60}}, "version": 9}
        ).encode("utf-8")
        client._on_message(mock_client, None, msg)

        assert received == [{"telemetry": {"interval_seconds": 60}}]

    @patch("src.mqtt.client.mqtt.Client")
    def test_on_message_invalid_json_is_ignored(self, mock_mqtt_cls):
        """Malformed JSON on a shadow topic must not raise."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_client.subscribe.return_value = (0, 1)
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        received: list[dict] = []
        client.subscribe_shadow_delta("test-001", received.append)

        msg = MagicMock()
        msg.topic = "$aws/things/test-001/shadow/update/delta"
        msg.payload = b"{not-json"
        # Must not raise.
        client._on_message(mock_client, None, msg)
        assert received == []

    @patch("src.mqtt.client.mqtt.Client")
    def test_publish_shadow_reported_envelope(self, mock_mqtt_cls):
        """publish_shadow_reported wraps state in the AWS reported envelope."""
        import json as _json

        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_client.publish.return_value = MagicMock(rc=0)
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        client.publish_shadow_reported("test-001", {"applied": ["a", "b"]})

        mock_client.publish.assert_called_once()
        args, kwargs = mock_client.publish.call_args
        topic = args[0]
        payload = _json.loads(args[1])
        assert topic == "$aws/things/test-001/shadow/update"
        assert kwargs["qos"] == 1
        assert kwargs["retain"] is False
        assert payload == {"state": {"reported": {"applied": ["a", "b"]}}}

    @patch("src.mqtt.client.mqtt.Client")
    def test_disconnect_count_increments_on_callback(self, mock_mqtt_cls):
        """_on_disconnect increments the counter (once per disconnect)."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        assert client.disconnect_count == 0
        # Simulate two unexpected disconnects (stop_event not set).
        client._on_disconnect(mock_client, None, None, rc=7)
        client._on_disconnect(mock_client, None, None, rc=7)
        assert client.disconnect_count == 2

    @patch("src.mqtt.client.mqtt.Client")
    def test_disconnect_count_not_incremented_on_graceful_stop(self, mock_mqtt_cls):
        """When disconnect() was called, callback must not bump counter."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        client._stop_event.set()
        client._on_disconnect(mock_client, None, None, rc=0)
        assert client.disconnect_count == 0

    @patch("src.mqtt.client.mqtt.Client")
    def test_reconnect_ts_set_on_connect(self, mock_mqtt_cls):
        """_on_connect stamps reconnect_ts with epoch time."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        import time as _time
        assert client.reconnect_ts is None
        t0 = _time.time()
        client._on_connect(mock_client, None, None, rc=0)
        t1 = _time.time()
        assert client.reconnect_ts is not None
        assert t0 <= client.reconnect_ts <= t1

    @patch("src.mqtt.client.mqtt.Client")
    def test_reconnect_ts_not_set_on_failed_connect(self, mock_mqtt_cls):
        """rc != 0 means auth/handshake failure — do not stamp reconnect_ts."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        client._on_connect(mock_client, None, None, rc=5)
        assert client.reconnect_ts is None

    @patch("src.mqtt.client.mqtt.Client")
    def test_on_connected_hook_fires_on_successful_connect(self, mock_mqtt_cls):
        """_on_connect with rc=0 must invoke the on_connected callback."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        calls = []

        def hook():
            calls.append(1)

        client.on_connected = hook
        # Fire twice to simulate a reconnect.
        client._on_connect(mock_client, None, None, rc=0)
        client._on_connect(mock_client, None, None, rc=0)
        assert len(calls) == 2

    @patch("src.mqtt.client.mqtt.Client")
    def test_on_connected_hook_not_fired_on_failed_connect(self, mock_mqtt_cls):
        """rc != 0 must not invoke the on_connected callback."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        calls = []
        client.on_connected = lambda: calls.append(1)
        client._on_connect(mock_client, None, None, rc=5)
        assert calls == []

    @patch("src.mqtt.client.mqtt.Client")
    def test_on_connected_hook_exceptions_swallowed(self, mock_mqtt_cls):
        """A raising on_connected hook must not break the paho callback."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="e.iot.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
        )

        def boom():
            raise RuntimeError("nope")

        client.on_connected = boom
        # Must not raise.
        client._on_connect(mock_client, None, None, rc=0)
        assert client.connected is True

    @patch("src.mqtt.client.mqtt.Client")
    def test_publish_event_unknown_type(self, mock_mqtt_cls):
        """Unknown event type returns None without buffering."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        tmpdir = tempfile.mkdtemp()
        cert, key, ca = self._make_certs(tmpdir)
        db_path = str(Path(tmpdir) / "buf.db")
        buffer = MessageBuffer(db_path)

        client = MQTTClient(
            device_id="test-001",
            endpoint="test.iot.us-east-1.amazonaws.com",
            port=8883,
            cert_path=cert,
            key_path=key,
            ca_path=ca,
            buffer=buffer,
            topics={"counting": "store/001/counting"},
        )

        result = client.publish_event("unknown_type", {})
        assert result is None
        assert len(buffer.get_pending()) == 0

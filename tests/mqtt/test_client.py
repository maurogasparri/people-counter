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

        with tempfile.TemporaryDirectory() as tmpdir:
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

        with tempfile.TemporaryDirectory() as tmpdir:
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

        with tempfile.TemporaryDirectory() as tmpdir:
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

        with tempfile.TemporaryDirectory() as tmpdir:
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
    def test_publish_event_unknown_type(self, mock_mqtt_cls):
        """Unknown event type returns None without buffering."""
        from src.mqtt.client import MQTTClient

        mock_client = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
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

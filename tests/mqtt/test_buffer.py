"""Tests for MQTT message buffer."""
import tempfile
from pathlib import Path

from src.mqtt.buffer import MessageBuffer


def test_enqueue_and_retrieve():
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    buf = MessageBuffer(db_path)

    msg_id = buf.enqueue("test/topic", {"count": 42})
    assert msg_id == 1

    pending = buf.get_pending()
    assert len(pending) == 1
    assert pending[0][1] == "test/topic"
    assert pending[0][2]["count"] == 42


def test_mark_sent():
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    buf = MessageBuffer(db_path)

    msg_id = buf.enqueue("test/topic", {"count": 1})
    buf.mark_sent(msg_id)

    pending = buf.get_pending()
    assert len(pending) == 0

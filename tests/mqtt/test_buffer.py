"""Tests para el buffer de mensajes MQTT."""
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


def test_count_unsent_reports_backlog():
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    buf = MessageBuffer(db_path)

    assert buf.count_unsent() == 0

    ids = [buf.enqueue("t", {"i": i}) for i in range(5)]
    assert buf.count_unsent() == 5

    buf.mark_sent(ids[0])
    buf.mark_sent(ids[1])
    assert buf.count_unsent() == 3


def test_enforce_backlog_limit_drops_oldest_unsent():
    """Cuando el backlog excede ``max_backlog``, los mensajes unsent más
    viejos se dropean. Bounded data loss > disk fill cuando AWS está
    unreachable días."""
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    buf = MessageBuffer(db_path, max_backlog=5)

    # Encolar 10 unsent.
    for i in range(10):
        buf.enqueue("t", {"i": i})
    assert buf.count_unsent() == 10

    dropped = buf.enforce_backlog_limit()
    assert dropped == 5
    assert buf.count_unsent() == 5

    # Los 5 que quedan deben ser los más nuevos (i=5..9).
    remaining = buf.get_pending(limit=10)
    payloads = sorted([p[2]["i"] for p in remaining])
    assert payloads == [5, 6, 7, 8, 9]


def test_enforce_backlog_limit_disabled_with_zero():
    """``max_backlog=0`` deshabilita el cap (back-compat para users que
    prefieren age-only)."""
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    buf = MessageBuffer(db_path, max_backlog=0)

    for i in range(100):
        buf.enqueue("t", {"i": i})
    dropped = buf.enforce_backlog_limit()
    assert dropped == 0
    assert buf.count_unsent() == 100


def test_vacuum_runs_without_error():
    """VACUUM no debería crashear sobre un DB válido — verificación
    smoke de wiring + WAL checkpoint."""
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    buf = MessageBuffer(db_path)
    buf.enqueue("t", {"i": 1})
    buf.vacuum()  # no exception
    # DB sigue funcional post-VACUUM.
    assert buf.count_unsent() == 1


def test_wal_mode_enabled_on_init():
    """WAL mode se setea al init y persiste en el archivo. Verifica
    leyendo la PRAGMA después de crear el buffer."""
    import sqlite3 as _sql

    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    MessageBuffer(db_path)
    with _sql.connect(db_path) as conn:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        # SQLite devuelve el mode actual; WAL persiste en el archivo.
        assert row[0].lower() == "wal"

"""SQLite local buffer for MQTT message resilience."""
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MessageBuffer:
    """Buffers MQTT messages locally for resilience against connectivity loss."""

    def __init__(self, db_path: str, max_age_hours: int = 72) -> None:
        self.db_path = db_path
        self.max_age_hours = max_age_hours
        self._ensure_db()

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    sent INTEGER DEFAULT 0
                )
            """)

    def enqueue(self, topic: str, payload: dict[str, Any]) -> int:
        """Add message to buffer. Returns message ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (topic, payload, created_at) VALUES (?, ?, ?)",
                (topic, json.dumps(payload), time.time()),
            )
            return cursor.lastrowid

    def mark_sent(self, message_id: int) -> None:
        """Mark message as successfully sent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE messages SET sent = 1 WHERE id = ?", (message_id,))

    def get_pending(self, limit: int = 100) -> list[tuple[int, str, dict]]:
        """Get unsent messages, oldest first."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, topic, payload FROM messages WHERE sent = 0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
            return [(r[0], r[1], json.loads(r[2])) for r in rows]

    def purge_old(self) -> int:
        """Delete messages older than max_age_hours. Returns count deleted."""
        cutoff = time.time() - (self.max_age_hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
            return cursor.rowcount

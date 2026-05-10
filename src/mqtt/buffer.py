"""Buffer SQLite local para resiliencia de mensajes MQTT."""
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MessageBuffer:
    """Buffer local de mensajes MQTT para resiliencia ante pérdida de conectividad."""

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
            # Acelera get_pending() en buffers grandes después de períodos largos offline.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sent_id ON messages(sent, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON messages(created_at)"
            )

    def enqueue(self, topic: str, payload: dict[str, Any]) -> int:
        """Agrega mensaje al buffer. Devuelve el ID del mensaje."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (topic, payload, created_at) VALUES (?, ?, ?)",
                (topic, json.dumps(payload), time.time()),
            )
            return cursor.lastrowid

    def mark_sent(self, message_id: int) -> None:
        """Marca el mensaje como enviado con éxito."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE messages SET sent = 1 WHERE id = ?", (message_id,))

    def get_pending(self, limit: int = 100) -> list[tuple[int, str, dict]]:
        """Devuelve los mensajes no enviados, los más viejos primero."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, topic, payload FROM messages WHERE sent = 0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
            return [(r[0], r[1], json.loads(r[2])) for r in rows]

    def purge_old(self) -> int:
        """Borra mensajes más viejos que max_age_hours. Devuelve la cantidad borrada.

        Defensivo ante errores de SQLite: el loop principal llama a esto cada
        60s. Si el DB es read-only (smoke run como ``pi`` con el archivo
        owned por ``root`` post-service-stop) o está corrupto, no queremos
        que la rutina de mantenimiento mate el pipeline. Logueamos y
        seguimos — el peor caso es que el buffer crezca un poco hasta el
        próximo restart con permisos correctos.
        """
        cutoff = time.time() - (self.max_age_hours * 3600)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE created_at < ?", (cutoff,)
                )
                return cursor.rowcount
        except sqlite3.Error:
            logger.exception("purge_old failed")
            return 0

    def count_unsent(self) -> int:
        """Devuelve la cantidad de mensajes no enviados actualmente en el buffer.

        Seguro de llamar frecuentemente — usa el índice ``idx_sent_id`` y devuelve
        0 ante cualquier error de SQLite (así la recolección de telemetría no crashea).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE sent = 0"
                ).fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error:
            logger.exception("count_unsent failed")
            return 0

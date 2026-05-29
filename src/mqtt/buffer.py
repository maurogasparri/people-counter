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

    def __init__(
        self,
        db_path: str,
        max_age_hours: int = 72,
        max_backlog: int = 50000,
    ) -> None:
        self.db_path = db_path
        self.max_age_hours = max_age_hours
        # Cap absoluto del backlog: si el buffer crece más allá de este
        # número de mensajes unsent (cloud unreachable por días), se
        # dropean los más viejos para evitar que el disco se llene.
        # 50k a ~500 bytes/msg ≈ 25MB — bounded data loss es mejor que
        # disco lleno que crashea el device. 0 deshabilita el cap.
        self.max_backlog = max(0, int(max_backlog))
        self._ensure_db()

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # WAL mode: mejor concurrencia (readers no bloquean writers) y
            # más resilient ante crashes. La PRAGMA persiste en el archivo
            # tras la primera vez que se setea, así que es safe re-correrla.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error:
                logger.exception("PRAGMA WAL setup failed")
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

    def mark_sent(self, message_id: int) -> bool:
        """Marca el mensaje como enviado con éxito. Devuelve True si se marcó.

        Se llama desde ``_on_publish`` (thread de red de paho) al recibir el
        PUBACK. Defensivo ante errores de SQLite: si el write falla (DB locked,
        read-only) NO dejamos que la excepción burbujee al loop de paho —
        logueamos y devolvemos False. El peor caso es que la fila quede
        ``sent=0`` y se re-publique en el próximo replay (QoS 1 = el broker
        dedupe), mucho mejor que romper el dispatch de callbacks de paho.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE messages SET sent = 1 WHERE id = ?", (message_id,)
                )
            return True
        except sqlite3.Error:
            logger.exception("mark_sent failed", extra={"message_id": message_id})
            return False

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

    def enforce_backlog_limit(self) -> int:
        """Si el backlog (unsent) crece más allá de ``max_backlog``,
        dropea los más viejos para mantener el cap. Devuelve la cantidad
        borrada.

        Es una salvaguarda dura: si AWS / Internet están unreachable por
        días, preferimos perder los eventos más viejos (probablemente ya
        irrecuperables operacionalmente) a llenar el disco y crashear el
        device. ``max_backlog=0`` deshabilita.
        """
        if self.max_backlog <= 0:
            return 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE sent = 0"
                ).fetchone()
                unsent = int(row[0]) if row else 0
                if unsent <= self.max_backlog:
                    return 0
                excess = unsent - self.max_backlog
                # Borrar los `excess` más viejos sin enviar — ORDER BY
                # created_at asegura FIFO. ``id`` está correlacionado
                # pero created_at es más semánticamente correcto si
                # hubiera clock skew.
                cursor = conn.execute(
                    "DELETE FROM messages WHERE id IN ("
                    "SELECT id FROM messages WHERE sent = 0 "
                    "ORDER BY created_at ASC LIMIT ?"
                    ")",
                    (excess,),
                )
                dropped = int(cursor.rowcount)
                if dropped > 0:
                    logger.warning(
                        "buffer_backlog_capped",
                        extra={
                            "dropped": dropped,
                            "remaining": self.max_backlog,
                        },
                    )
                return dropped
        except sqlite3.Error:
            logger.exception("enforce_backlog_limit failed")
            return 0

    def vacuum(self) -> None:
        """Reclaim disk space tras un período largo de purges.

        SQLite no libera páginas eliminadas hasta un VACUUM explícito;
        en runs largos (semanas de operación con purge diario) el
        archivo crece sin reflejarlo en row count. Llamar después de
        un purge significativo (no cada minuto — VACUUM es I/O heavy).
        También hace ``wal_checkpoint(TRUNCATE)`` para que el archivo
        ``-wal`` no acumule. Errores se loguean y se ignoran (worst
        case: el archivo crece hasta el próximo VACUUM).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("VACUUM")
                logger.info("buffer_vacuum_completed")
        except sqlite3.Error:
            logger.exception("vacuum failed")

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

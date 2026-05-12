"""Dedup WiFi/BLE: intra-protocolo y cross-protocolo."""
import logging
import sqlite3
import time
from pathlib import Path

from src.wifi_ble.hasher import hash_mac

logger = logging.getLogger(__name__)


class DedupEngine:
    """Maneja las capas de dedup 1 (intra-protocolo) y 2 (cross-protocolo).

    La capa 3 (inter-cámara) se maneja del lado cloud en Lambda.
    """

    def __init__(
        self,
        db_path: str,
        cross_window_seconds: float = 2.0,
        cross_rssi_delta: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self.cross_window = cross_window_seconds
        self.cross_rssi_delta = cross_rssi_delta
        self._ensure_db()

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_hashes (
                    hash TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    rssi REAL,
                    PRIMARY KEY (hash, protocol)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS unified_hashes (
                    unified_hash TEXT PRIMARY KEY,
                    wifi_hash TEXT,
                    ble_hash TEXT,
                    created_at REAL NOT NULL
                )
            """)

    def process_detection(
        self, mac: str, protocol: str, rssi: float, salt: str = ""
    ) -> dict:
        """Procesa una detección individual de WiFi o BLE.

        Returns:
            {"is_new": bool, "hash": str, "unified": bool}
        """
        mac_hash = hash_mac(mac, salt)
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Capa 1: dedup intra-protocolo
            existing = conn.execute(
                "SELECT hash FROM seen_hashes WHERE hash = ? AND protocol = ?",
                (mac_hash, protocol),
            ).fetchone()

            if existing:
                return {"is_new": False, "hash": mac_hash, "unified": False}

            # Nueva detección — la guardamos
            conn.execute(
                "INSERT INTO seen_hashes (hash, protocol, first_seen, rssi) VALUES (?, ?, ?, ?)",
                (mac_hash, protocol, now, rssi),
            )

            # Capa 2: correlación cross-protocolo
            other_protocol = "ble" if protocol == "wifi" else "wifi"
            candidates = conn.execute(
                """SELECT hash, rssi FROM seen_hashes
                   WHERE protocol = ? AND first_seen > ?""",
                (other_protocol, now - self.cross_window),
            ).fetchall()

            for other_hash, other_rssi in candidates:
                if other_rssi is not None and abs(rssi - other_rssi) <= self.cross_rssi_delta:
                    # Match cross-protocolo — generamos hash unificado
                    unified = hash_mac(f"{mac_hash}{other_hash}")
                    conn.execute(
                        """INSERT OR IGNORE INTO unified_hashes
                           (unified_hash, wifi_hash, ble_hash, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (
                            unified,
                            mac_hash if protocol == "wifi" else other_hash,
                            mac_hash if protocol == "ble" else other_hash,
                            now,
                        ),
                    )
                    logger.debug(
                        "cross_protocol_match",
                        extra={
                            "protocol": protocol,
                            "hash_prefix": mac_hash[:8],
                            "other_protocol": other_protocol,
                            "other_hash_prefix": other_hash[:8],
                            "unified_prefix": unified[:8],
                        },
                    )
                    return {"is_new": True, "hash": unified, "unified": True}

            return {"is_new": True, "hash": mac_hash, "unified": False}

    def get_unique_count(self) -> int:
        """Devuelve el total de visitantes únicos del día actual.

        Cuenta: hashes individuales que NO son parte de un par unificado,
        más los hashes unificados.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Todos los hashes unificados
            unified = conn.execute("SELECT COUNT(*) FROM unified_hashes").fetchone()[0]

            # Hashes individuales que no están en ningún par unificado
            individual = conn.execute("""
                SELECT COUNT(*) FROM seen_hashes sh
                WHERE NOT EXISTS (
                    SELECT 1 FROM unified_hashes uh
                    WHERE uh.wifi_hash = sh.hash OR uh.ble_hash = sh.hash
                )
            """).fetchone()[0]

            return unified + individual

    def get_traffic_counts(
        self,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
        protocol: str | None = None,
    ) -> dict:
        """Clasifica detecciones WiFi/BLE por RSSI en passersby vs shoppers.

        Modelo de doble umbral:
            - rssi_passerby (-75 dBm): el dispositivo está "presente" (tráfico exterior)
            - rssi_shopper  (-55 dBm): el dispositivo está "muy cerca" (entró al local)

        Turn In Rate = shoppers / passersby.

        Args:
            rssi_passerby: RSSI mínimo para contar como passerby (default -75 dBm).
            rssi_shopper: RSSI mínimo para contar como shopper (default -55 dBm).
            protocol: Si se setea ("wifi" o "ble"), cuenta solo ese protocolo.
                WiFi y BLE tienen distintos niveles de referencia de RSSI y
                características de antena, así que un breakdown per-protocolo
                puede ser más interpretable que el total mezclado.

        Returns:
            {"passersby": int, "shoppers": int, "turn_in_rate": float}
        """
        where = "WHERE rssi >= ?"
        params_passerby: tuple = (rssi_passerby,)
        params_shopper: tuple = (rssi_shopper,)
        if protocol is not None:
            where += " AND protocol = ?"
            params_passerby = (rssi_passerby, protocol)
            params_shopper = (rssi_shopper, protocol)

        with sqlite3.connect(self.db_path) as conn:
            passerby_count = conn.execute(
                f"SELECT COUNT(DISTINCT hash) FROM seen_hashes {where}",
                params_passerby,
            ).fetchone()[0]

            shopper_count = conn.execute(
                f"SELECT COUNT(DISTINCT hash) FROM seen_hashes {where}",
                params_shopper,
            ).fetchone()[0]

        turn_in = (shopper_count / passerby_count) if passerby_count > 0 else 0.0

        return {
            "passersby": passerby_count,
            "shoppers": shopper_count,
            "turn_in_rate": round(turn_in, 4),
        }

    def get_recent_hashes(
        self,
        since_ts: float,
        until_ts: float | None = None,
        protocol: str | None = None,
    ) -> list[str]:
        """Devuelve los hashes vistos por primera vez en una ventana temporal.

        Usado por el publisher para armar el payload periódico que va a
        Lambda L3 (dedup inter-cámara). Filtra por ``first_seen`` así cada
        ventana es disjunta — un hash sólo se reporta en la ventana en la
        que fue detectado por primera vez ese día.

        Args:
            since_ts: epoch seconds — borde inferior inclusivo.
            until_ts: epoch seconds — borde superior exclusivo. None = ``time.time()``.
            protocol: ``"wifi"`` o ``"ble"`` para filtrar por protocolo.
                None = ambos.

        Returns:
            Lista de hashes hex (deduplicada implícitamente por la PK de
            ``seen_hashes``).
        """
        if until_ts is None:
            until_ts = time.time()

        params: list[float | str] = [since_ts, until_ts]
        where = "WHERE first_seen >= ? AND first_seen < ?"
        if protocol is not None:
            where += " AND protocol = ?"
            params.append(protocol)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT hash FROM seen_hashes {where}",
                params,
            ).fetchall()
        return [r[0] for r in rows]

    def get_window_summary(
        self,
        since_ts: float,
        until_ts: float | None = None,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
    ) -> dict[str, int]:
        """Agregados WiFi+BLE post-L2 dedup para una ventana cerrada.

        Es lo que publica el bridge a cloud — sin hashes, solo counts.
        Cuenta visitantes únicos: un dispositivo detectado por WiFi Y BLE
        en la ventana cuenta como 1 (gracias a los unified_hashes de L2).

        Args:
            since_ts: epoch seconds — borde inferior inclusivo del ``first_seen``.
            until_ts: epoch seconds — borde superior exclusivo. None = ``time.time()``.
            rssi_passerby: RSSI mínimo para "pasó por la zona" (default -75 dBm).
            rssi_shopper:  RSSI mínimo para "muy cerca / probable entrada" (-55 dBm).

        Returns:
            {"passersby": N, "shoppers": M} — ``shoppers ⊆ passersby``.
        """
        if until_ts is None:
            until_ts = time.time()

        # Las queries cuentan dispositivos únicos: si un MAC fue unificado en L2
        # (visto por WiFi y BLE en ventana corta), aparece en seen_hashes dos
        # veces pero unified_hashes lo agrupa — la lógica sigue siendo idéntica
        # a la de ``get_unique_count`` pero scoped a la ventana por ``first_seen``.
        with sqlite3.connect(self.db_path) as conn:
            # Hashes individuales NO unificados, dentro de la ventana, con RSSI
            # >= threshold. Cuenta cada uno una sola vez.
            def _count(rssi_threshold: float) -> int:
                # Componentes:
                # 1) hashes individuales que no son parte de ningún unified
                #    y que cumplen el filtro de RSSI en la ventana.
                # 2) unified_hashes cuyo created_at cae en la ventana — si
                #    alguno de los dos lados del par cumple el RSSI threshold
                #    el unified cuenta (un walk-by detectado fuerte por uno de
                #    los radios alcanza).
                individual = conn.execute(
                    """
                    SELECT COUNT(*) FROM seen_hashes sh
                    WHERE sh.first_seen >= ? AND sh.first_seen < ?
                      AND sh.rssi IS NOT NULL AND sh.rssi >= ?
                      AND NOT EXISTS (
                          SELECT 1 FROM unified_hashes uh
                          WHERE uh.wifi_hash = sh.hash OR uh.ble_hash = sh.hash
                      )
                    """,
                    (since_ts, until_ts, rssi_threshold),
                ).fetchone()[0]

                unified = conn.execute(
                    """
                    SELECT COUNT(*) FROM unified_hashes uh
                    WHERE uh.created_at >= ? AND uh.created_at < ?
                      AND EXISTS (
                          SELECT 1 FROM seen_hashes sh
                          WHERE (sh.hash = uh.wifi_hash OR sh.hash = uh.ble_hash)
                            AND sh.rssi IS NOT NULL
                            AND sh.rssi >= ?
                      )
                    """,
                    (since_ts, until_ts, rssi_threshold),
                ).fetchone()[0]

                return individual + unified

            passersby = _count(rssi_passerby)
            shoppers = _count(rssi_shopper)

        return {"passersby": passersby, "shoppers": shoppers}

    def reset_daily(self) -> None:
        """Limpia todos los hashes para un nuevo día comercial."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM seen_hashes")
            conn.execute("DELETE FROM unified_hashes")
        logger.info("dedup_daily_reset")

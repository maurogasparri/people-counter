"""WiFi/BLE deduplication: intra-protocol and cross-protocol."""
import logging
import sqlite3
import time
from pathlib import Path

from src.wifi_ble.hasher import hash_mac

logger = logging.getLogger(__name__)


class DedupEngine:
    """Handles dedup layers 1 (intra-protocol) and 2 (cross-protocol).

    Layer 3 (inter-camera) is handled cloud-side in Lambda.
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
        """Process a single WiFi or BLE detection.

        Returns:
            {"is_new": bool, "hash": str, "unified": bool}
        """
        mac_hash = hash_mac(mac, salt)
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Layer 1: intra-protocol dedup
            existing = conn.execute(
                "SELECT hash FROM seen_hashes WHERE hash = ? AND protocol = ?",
                (mac_hash, protocol),
            ).fetchone()

            if existing:
                return {"is_new": False, "hash": mac_hash, "unified": False}

            # New detection — store it
            conn.execute(
                "INSERT INTO seen_hashes (hash, protocol, first_seen, rssi) VALUES (?, ?, ?, ?)",
                (mac_hash, protocol, now, rssi),
            )

            # Layer 2: cross-protocol correlation
            other_protocol = "ble" if protocol == "wifi" else "wifi"
            candidates = conn.execute(
                """SELECT hash, rssi FROM seen_hashes
                   WHERE protocol = ? AND first_seen > ?""",
                (other_protocol, now - self.cross_window),
            ).fetchall()

            for other_hash, other_rssi in candidates:
                if other_rssi is not None and abs(rssi - other_rssi) <= self.cross_rssi_delta:
                    # Cross-protocol match — create unified hash
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
                        "Cross-protocol match: %s(%s) + %s(%s) → %s",
                        protocol, mac_hash[:8], other_protocol, other_hash[:8], unified[:8],
                    )
                    return {"is_new": True, "hash": unified, "unified": True}

            return {"is_new": True, "hash": mac_hash, "unified": False}

    def get_unique_count(self) -> int:
        """Get total unique visitors for current day.

        Counts: individual hashes that are NOT part of a unified pair,
        plus unified hashes.
        """
        with sqlite3.connect(self.db_path) as conn:
            # All unified hashes
            unified = conn.execute("SELECT COUNT(*) FROM unified_hashes").fetchone()[0]

            # Individual hashes not in any unified pair
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
    ) -> dict:
        """Classify WiFi/BLE detections by RSSI into passersby vs shoppers.

        Dual-threshold model:
            - rssi_passerby (-75 dBm): device is "present" (outside traffic)
            - rssi_shopper  (-55 dBm): device is "very close" (entered store)

        Turn In Rate = shoppers / passersby.

        Args:
            rssi_passerby: Minimum RSSI to count as passerby (default -75 dBm).
            rssi_shopper: Minimum RSSI to count as shopper (default -55 dBm).

        Returns:
            {"passersby": int, "shoppers": int, "turn_in_rate": float}
        """
        with sqlite3.connect(self.db_path) as conn:
            passerby_count = conn.execute(
                "SELECT COUNT(DISTINCT hash) FROM seen_hashes WHERE rssi >= ?",
                (rssi_passerby,),
            ).fetchone()[0]

            shopper_count = conn.execute(
                "SELECT COUNT(DISTINCT hash) FROM seen_hashes WHERE rssi >= ?",
                (rssi_shopper,),
            ).fetchone()[0]

        turn_in = (shopper_count / passerby_count) if passerby_count > 0 else 0.0

        return {
            "passersby": passerby_count,
            "shoppers": shopper_count,
            "turn_in_rate": round(turn_in, 4),
        }

    def reset_daily(self) -> None:
        """Clear all hashes for new business day."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM seen_hashes")
            conn.execute("DELETE FROM unified_hashes")
        logger.info("Daily dedup reset")

"""Dedup WiFi/BLE: stitching de MACs randomizadas en grupos de identidad.

iOS y Android randomizan la MAC en probes cada ~2-15 min, lo que infla el
conteo de visitantes unicos si tratamos cada MAC observada como una persona
distinta. Este modulo mantiene la abstraccion de "hash" (privacy-first: solo
hashes salen del device en agregados) pero asocia multiples hashes al mismo
``group_id`` cuando senales mecanicas sugieren que vienen del mismo chip.

Reglas de stitching (por orden de aplicacion):

1. **Seqnum continuity (WiFi-only)**: el sequence number del header 802.11 es
   un contador del chip que tipicamente es continuo cross-MAC-rotation. Dos
   probes con seqnum cercano (mod 4096) + RSSI cercano + dt corto = mismo
   chip. Defeated por chipsets que resetean seqnum on MAC change (Apple H1+
   en iPhone 12+) — sigue funcionando en Android.

2. **Cross-protocol L2 (short window)**: WiFi MAC W y BLE MAC B observados
   dentro de ``cross_window_seconds`` con RSSI similar son el mismo
   dispositivo. Es el L2 dedup historico.

3. **BLE anchoring (long window)**: dentro de la vida de UN BLE address
   (~15min para iOS RPA), si vemos multiples WiFi MACs con RSSI similar al
   BLE en cuestion, son el mismo dispositivo. Extiende #2 a la ventana real
   de rotacion de BLE.

Cada hash entra a un ``group_id`` la primera vez que se ve. Hashes futuros
hacen match contra cualquier miembro del grupo (no contra el group_id en
abstracto). Los counts publicados (``passersby``, ``shoppers``) son distinct
``group_id``, no distinct hashes.

Privacy: el ``seqnum`` y los timestamps quedan SOLO en SQLite local (rotado
diario via ``reset_daily``). El MQTT publish sigue mandando counts agregados,
nunca hashes ni seqnums. La capa 3 (inter-camara) no aplica al PoC con 1
device/sucursal.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path

from src.wifi_ble.hasher import hash_mac

logger = logging.getLogger(__name__)

# Modulo del seqnum 802.11 (12 bits).
SEQNUM_MOD = 4096


def _seqnum_delta(a: int, b: int) -> int:
    """Distancia minima entre seqnums considerando wrap a 4096.

    Devuelve siempre el valor positivo (0..2048). Si la separacion linear
    es <2048, esa es la distancia. Si es >=2048, asumimos wrap-around.
    """
    raw = abs(a - b)
    return min(raw, SEQNUM_MOD - raw)


class DedupEngine:
    """Stitching de hashes WiFi/BLE en grupos de identidad por device.

    Las 3 reglas de stitching estan documentadas en el modulo docstring.
    Cada llamada a ``process_detection`` decide si la observacion es:

    - duplicada (hash + protocolo ya vistos) -> actualiza last_seen/rssi/seqnum
    - nueva pero pertenece a un grupo existente (alguna regla matchea)
      -> agrega el hash al grupo
    - genuinamente nueva -> arranca un grupo nuevo

    El ``group_id`` es un UUID hex. Los queries downstream cuentan ``DISTINCT
    group_id``, no hashes.
    """

    def __init__(
        self,
        db_path: str,
        cross_window_seconds: float = 2.0,
        cross_rssi_delta: float = 5.0,
        seqnum_stitch_enabled: bool = True,
        seqnum_stitch_window_seconds: float = 30.0,
        seqnum_max_delta: int = 100,
        seqnum_rssi_delta: float = 5.0,
        ble_anchor_enabled: bool = True,
        ble_anchor_window_seconds: float = 900.0,
    ) -> None:
        """
        Args:
            db_path: archivo SQLite del state.
            cross_window_seconds: ventana corta para correlacion cross-protocol
                (regla 2). Default 2s = mismo dispositivo emitiendo WiFi+BLE
                casi simultaneo.
            cross_rssi_delta: tolerancia de RSSI para correlacion. Default 5dBm.
            seqnum_stitch_enabled: kill switch de la regla 1.
            seqnum_stitch_window_seconds: ventana para stitching por seqnum.
                Default 30s — chipsets rotan MAC tipicamente cada 2-15min, asi
                que 30s captura la transicion sin agarrar dispositivos
                distintos.
            seqnum_max_delta: maxima distancia (mod 4096) entre seqnums para
                considerar continuidad. Default 100 — un chip activo emite
                ~10 probes/seg, asi que 100 cubre ~10s de actividad entre
                rotaciones.
            seqnum_rssi_delta: tolerancia de RSSI para stitching por seqnum.
                Default 5dBm — igual que cross_rssi_delta.
            ble_anchor_enabled: kill switch de la regla 3.
            ble_anchor_window_seconds: ventana para BLE anchoring. Default
                900s = 15min ≈ vida tipica de una iOS BLE RPA.
        """
        self.db_path = db_path
        self.cross_window = cross_window_seconds
        self.cross_rssi_delta = cross_rssi_delta
        self.seqnum_stitch_enabled = seqnum_stitch_enabled
        self.seqnum_stitch_window = seqnum_stitch_window_seconds
        self.seqnum_max_delta = seqnum_max_delta
        self.seqnum_rssi_delta = seqnum_rssi_delta
        self.ble_anchor_enabled = ble_anchor_enabled
        self.ble_anchor_window = ble_anchor_window_seconds
        self._ensure_db()

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # hash_groups: una fila por (hash, protocol). group_id es comun
            # entre miembros del mismo grupo. seqnum solo aplica a WiFi (None
            # para BLE).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hash_groups (
                    hash         TEXT NOT NULL,
                    protocol     TEXT NOT NULL,
                    group_id     TEXT NOT NULL,
                    first_seen   REAL NOT NULL,
                    last_seen    REAL NOT NULL,
                    rssi         REAL,
                    seqnum       INTEGER,
                    PRIMARY KEY (hash, protocol)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash_groups_group ON hash_groups(group_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash_groups_last_seen ON hash_groups(last_seen)"
            )

    # ----- API publica -----

    def process_detection(
        self,
        mac: str,
        protocol: str,
        rssi: float,
        salt: str = "",
        seqnum: int | None = None,
    ) -> dict:
        """Procesa una deteccion individual de WiFi o BLE.

        Args:
            mac: MAC observada (formato AA:BB:CC:DD:EE:FF, mayusculas o no).
            protocol: ``"wifi"`` o ``"ble"``.
            rssi: signal strength en dBm (negativo, e.g. -60).
            salt: salt opcional para el hash (se usa rotacion diaria).
            seqnum: seqnum 802.11 (0..4095). None si no aplica.

        Returns:
            {"is_new": bool, "hash": str, "unified": bool, "group_id": str}

            - is_new: True si (hash, protocol) no existia previamente.
            - hash: el hash SHA256 truncado a 16 bytes.
            - unified: True si esta deteccion se incorporo a un grupo
              existente (ya sea recien creado en esta call o pre-existente).
            - group_id: UUID del grupo final.
        """
        mac_hash = hash_mac(mac, salt)
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # 1. Hash ya conocido para este protocolo? -> update y volver.
            existing = conn.execute(
                "SELECT group_id FROM hash_groups WHERE hash = ? AND protocol = ?",
                (mac_hash, protocol),
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE hash_groups
                       SET last_seen = ?, rssi = ?, seqnum = COALESCE(?, seqnum)
                       WHERE hash = ? AND protocol = ?""",
                    (now, rssi, seqnum, mac_hash, protocol),
                )
                return {
                    "is_new": False,
                    "hash": mac_hash,
                    "unified": False,
                    "group_id": existing[0],
                }

            # 2. Hash nuevo. Buscar grupos candidatos para hacer stitching.
            group_id = self._find_candidate_group(
                conn, protocol, rssi, seqnum, now
            )
            unified = group_id is not None
            if group_id is None:
                group_id = uuid.uuid4().hex

            conn.execute(
                """INSERT INTO hash_groups
                   (hash, protocol, group_id, first_seen, last_seen, rssi, seqnum)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (mac_hash, protocol, group_id, now, now, rssi, seqnum),
            )

            if unified:
                logger.debug(
                    "stitch_match protocol=%s hash_prefix=%s group_prefix=%s",
                    protocol,
                    mac_hash[:8],
                    group_id[:8],
                )

            return {
                "is_new": True,
                "hash": mac_hash,
                "unified": unified,
                "group_id": group_id,
            }

    def _find_candidate_group(
        self,
        conn: sqlite3.Connection,
        protocol: str,
        rssi: float,
        seqnum: int | None,
        now: float,
    ) -> str | None:
        """Encuentra el group_id mas reciente que matchea alguna regla de stitching.

        Devuelve None si ninguna regla matchea. Si hay multiple matches,
        devuelve el grupo con el ``last_seen`` mas reciente — heuristica que
        prioriza la observacion mas fresca como mas probable de ser la misma
        persona.

        Las 3 reglas se evaluan en una sola query union — devuelve el match
        de cualquier regla, ordenado por recencia.
        """
        other_protocol = "ble" if protocol == "wifi" else "wifi"
        rules: list[tuple[str, tuple]] = []

        # Regla 1: seqnum continuity (WiFi-only, requiere seqnum no-None).
        if (
            self.seqnum_stitch_enabled
            and protocol == "wifi"
            and seqnum is not None
        ):
            # Filtramos por (a) protocol=wifi, (b) tienen seqnum, (c) rssi
            # cercano, (d) dentro de la ventana de stitching. La distancia de
            # seqnum la calculamos en Python despues (mod 4096 no es trivial
            # de expresar en SQL portable).
            rules.append(
                (
                    """SELECT group_id, seqnum, last_seen FROM hash_groups
                       WHERE protocol = 'wifi'
                         AND seqnum IS NOT NULL
                         AND rssi IS NOT NULL
                         AND ABS(rssi - ?) <= ?
                         AND last_seen >= ?""",
                    (rssi, self.seqnum_rssi_delta, now - self.seqnum_stitch_window),
                )
            )

        # Regla 2: cross-protocol L2 short window.
        rules.append(
            (
                """SELECT group_id, NULL AS seqnum, last_seen FROM hash_groups
                   WHERE protocol = ?
                     AND rssi IS NOT NULL
                     AND ABS(rssi - ?) <= ?
                     AND last_seen >= ?""",
                (other_protocol, rssi, self.cross_rssi_delta, now - self.cross_window),
            )
        )

        # Regla 3: BLE anchor long window — un grupo con algun BLE miembro
        # activo en los ultimos N seg, RSSI compatible con esta deteccion.
        # Aplica tanto a una nueva WiFi MAC (ancla al BLE existente) como
        # a una nueva BLE addr (ancla al WiFi existente si el dispositivo
        # rota ambos en paralelo).
        if self.ble_anchor_enabled:
            anchor_protocol = "ble" if protocol == "wifi" else "wifi"
            rules.append(
                (
                    """SELECT group_id, NULL AS seqnum, last_seen FROM hash_groups
                       WHERE protocol = ?
                         AND rssi IS NOT NULL
                         AND ABS(rssi - ?) <= ?
                         AND last_seen >= ?""",
                    (
                        anchor_protocol,
                        rssi,
                        self.cross_rssi_delta,
                        now - self.ble_anchor_window,
                    ),
                )
            )

        # Ejecutar las reglas y juntar candidates. Cada regla devuelve
        # (group_id, seqnum, last_seen).
        candidates: list[tuple[str, int | None, float]] = []
        for sql, params in rules:
            for row in conn.execute(sql, params).fetchall():
                candidates.append(row)

        if not candidates:
            return None

        # Para la regla 1, validar seqnum delta en Python (mod 4096).
        # Para las otras reglas, todo candidato es valido — son matches por RSSI+time.
        # Aplicamos el filtro de seqnum solo a los rows que VINIERON con seqnum
        # (regla 1) — los de regla 2/3 vienen con seqnum=None del NULL en SQL
        # y pasan tal cual.
        valid: list[tuple[str, float]] = []
        for group_id, candidate_seqnum, last_seen in candidates:
            if candidate_seqnum is not None and seqnum is not None:
                if _seqnum_delta(int(candidate_seqnum), seqnum) > self.seqnum_max_delta:
                    continue
            valid.append((group_id, last_seen))

        if not valid:
            return None

        # Multi-match: el grupo con last_seen mas reciente.
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[0][0]

    def get_unique_count(self) -> int:
        """Devuelve el total de grupos unicos del dia actual.

        Cuenta DISTINCT group_id en hash_groups — un dispositivo con N MAC
        rotaciones stiched + 1 BLE addr cuenta como 1.
        """
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT group_id) FROM hash_groups"
            ).fetchone()[0]

    def get_traffic_counts(
        self,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
        protocol: str | None = None,
    ) -> dict:
        """Clasifica grupos por RSSI maximo observado en passersby vs shoppers.

        Un grupo cuenta como passerby/shopper si ALGUN miembro tuvo RSSI por
        encima del threshold — la lectura mas fuerte gana, asi un dispositivo
        que entro y salio del rango cuenta correctamente como shopper.

        Args:
            rssi_passerby: RSSI minimo para contar como passerby (default -75 dBm).
            rssi_shopper: RSSI minimo para contar como shopper (default -55 dBm).
            protocol: Si se setea ("wifi" o "ble"), cuenta solo grupos cuyo
                MAX-RSSI viene de un miembro de ese protocolo. Util para
                breakdown per-radio.

        Returns:
            {"passersby": int, "shoppers": int, "turn_in_rate": float}
        """
        with sqlite3.connect(self.db_path) as conn:
            def _count(threshold: float) -> int:
                where = "rssi IS NOT NULL AND rssi >= ?"
                params: list = [threshold]
                if protocol is not None:
                    where += " AND protocol = ?"
                    params.append(protocol)
                return conn.execute(
                    f"SELECT COUNT(DISTINCT group_id) FROM hash_groups WHERE {where}",
                    params,
                ).fetchone()[0]

            passerby_count = _count(rssi_passerby)
            shopper_count = _count(rssi_shopper)

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
        """Devuelve hashes vistos por primera vez en una ventana temporal.

        Mantenido para compat con el publisher viejo y eventual L3 cloud dedup.
        Filtra por ``first_seen`` — un hash solo se reporta en la ventana en
        la que fue stitcheado por primera vez ese dia.

        Args:
            since_ts: epoch seconds — borde inferior inclusivo.
            until_ts: epoch seconds — borde superior exclusivo. None = ``now()``.
            protocol: ``"wifi"`` o ``"ble"`` para filtrar. None = ambos.

        Returns:
            Lista de hashes hex (deduplicada implicita por PK (hash, protocol)).
        """
        if until_ts is None:
            until_ts = time.time()

        params: list[float | str] = [since_ts, until_ts]
        where = "first_seen >= ? AND first_seen < ?"
        if protocol is not None:
            where += " AND protocol = ?"
            params.append(protocol)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT hash FROM hash_groups WHERE {where}", params
            ).fetchall()
        return [r[0] for r in rows]

    def get_window_summary(
        self,
        since_ts: float,
        until_ts: float | None = None,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
    ) -> dict[str, int]:
        """Agregados WiFi+BLE post-stitching para una ventana cerrada.

        Cuenta DISTINCT group_id donde al menos un miembro del grupo tuvo
        first_seen en [since_ts, until_ts) Y rssi >= threshold (en cualquier
        observacion). Un dispositivo stiched en N MACs cuenta como 1.

        Args:
            since_ts: epoch seconds — borde inferior inclusivo del first_seen.
            until_ts: epoch seconds — borde superior exclusivo. None = ``now()``.
            rssi_passerby: RSSI minimo para "paso por la zona" (default -75 dBm).
            rssi_shopper:  RSSI minimo para "muy cerca / probable entrada" (-55).

        Returns:
            {"passersby": N, "shoppers": M} — invariante shoppers <= passersby.
        """
        if until_ts is None:
            until_ts = time.time()

        with sqlite3.connect(self.db_path) as conn:
            def _count(threshold: float) -> int:
                # Un grupo cuenta si ALGUN miembro tiene first_seen en la
                # ventana Y ALGUN miembro tiene rssi >= threshold. No
                # requerimos que el mismo miembro cumpla ambas — un grupo
                # que arranco con un probe debil pero despues tuvo un BLE
                # fuerte cuenta como shopper.
                return conn.execute(
                    """
                    SELECT COUNT(DISTINCT g.group_id) FROM hash_groups g
                    WHERE g.group_id IN (
                        SELECT group_id FROM hash_groups
                        WHERE first_seen >= ? AND first_seen < ?
                    )
                    AND g.group_id IN (
                        SELECT group_id FROM hash_groups
                        WHERE rssi IS NOT NULL AND rssi >= ?
                    )
                    """,
                    (since_ts, until_ts, threshold),
                ).fetchone()[0]

            passersby = _count(rssi_passerby)
            shoppers = _count(rssi_shopper)

        return {"passersby": passersby, "shoppers": shoppers}

    def get_stitching_ratio(self) -> float | None:
        """Ratio de compresion del stitching: groups / hashes.

        - 1.0 = ningun stitch ocurrio (cada hash es su propio grupo).
        - 0.5 = la mitad de los hashes se mergearon (avg 2 hashes/grupo).
        - 0.0 (no posible) o None si no hay hashes.

        Util para telemetria: rastrear si el stitching esta efectivamente
        contrarrestando la rotacion de MAC, o si la flota viene con OS que
        defeatean las reglas (Apple H1+ con seqnum reset, BLE off, etc).
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT COUNT(*), COUNT(DISTINCT group_id) FROM hash_groups"""
            ).fetchone()
            hashes, groups = row
            if hashes == 0:
                return None
            return groups / hashes

    def reset_daily(self) -> None:
        """Limpia todos los grupos para un nuevo dia comercial."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM hash_groups")
        logger.info("dedup_daily_reset")

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
        fingerprint_stitch_enabled: bool = True,
        fingerprint_stitch_window_seconds: float = 900.0,
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
        # Regla 4: stitching por fingerprint (mismo protocolo). Re-une
        # rotaciones que el seqnum no cubre — iOS resetea el seqnum al rotar
        # la MAC, pero el fingerprint de IEs/manufacturer-data es estable.
        self.fingerprint_stitch_enabled = fingerprint_stitch_enabled
        self.fingerprint_stitch_window = fingerprint_stitch_window_seconds
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
                    max_rssi     REAL,
                    seqnum       INTEGER,
                    fingerprint  TEXT,
                    PRIMARY KEY (hash, protocol)
                )
                """
            )
            # Migración para DBs creadas antes de columnas nuevas (SQLite no
            # tiene ADD COLUMN IF NOT EXISTS).
            cols = {row[1] for row in conn.execute("PRAGMA table_info(hash_groups)")}
            if "fingerprint" not in cols:
                conn.execute("ALTER TABLE hash_groups ADD COLUMN fingerprint TEXT")
            if "max_rssi" not in cols:
                conn.execute("ALTER TABLE hash_groups ADD COLUMN max_rssi REAL")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash_groups_group ON hash_groups(group_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash_groups_last_seen ON hash_groups(last_seen)"
            )
            # Índice para la regla 4 (lookup por fingerprint + protocolo).
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash_groups_fp "
                "ON hash_groups(protocol, fingerprint)"
            )

    # ----- API publica -----

    def process_detection(
        self,
        mac: str,
        protocol: str,
        rssi: float,
        salt: str = "",
        seqnum: int | None = None,
        fingerprint: str = "",
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
                # rssi = última lectura (la usan las reglas de stitching, que
                # quieren la RSSI reciente). max_rssi = la más fuerte vista
                # (la usa get_traffic_counts: un device que UNA vez estuvo
                # cerca cuenta como shopper de forma estable, sin flapping
                # cuando la RSSI oscila alrededor del threshold).
                conn.execute(
                    """UPDATE hash_groups
                       SET last_seen = ?, rssi = ?,
                           max_rssi = MAX(COALESCE(max_rssi, ?), ?),
                           seqnum = COALESCE(?, seqnum)
                       WHERE hash = ? AND protocol = ?""",
                    (now, rssi, rssi, rssi, seqnum, mac_hash, protocol),
                )
                return {
                    "is_new": False,
                    "hash": mac_hash,
                    "unified": False,
                    "group_id": existing[0],
                }

            # 2. Hash nuevo. Buscar grupos candidatos para hacer stitching.
            group_id = self._find_candidate_group(
                conn, protocol, rssi, seqnum, now, fingerprint
            )
            unified = group_id is not None
            if group_id is None:
                group_id = uuid.uuid4().hex

            conn.execute(
                """INSERT INTO hash_groups
                   (hash, protocol, group_id, first_seen, last_seen, rssi,
                    max_rssi, seqnum, fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mac_hash, protocol, group_id, now, now, rssi, rssi, seqnum,
                 fingerprint),
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
        fingerprint: str = "",
    ) -> str | None:
        """Encuentra el group_id mas reciente que matchea alguna regla de stitching.

        Devuelve None si ninguna regla matchea. Si hay multiple matches,
        devuelve el grupo con el ``last_seen`` mas reciente — heuristica que
        prioriza la observacion mas fresca como mas probable de ser la misma
        persona.

        4 reglas (ver docstring del modulo). Cada candidato es ``(group_id,
        last_seen)``; se juntan de todas las reglas y gana el mas reciente.
        """
        other_protocol = "ble" if protocol == "wifi" else "wifi"
        candidates: list[tuple[str, float]] = []

        # Regla 1: seqnum continuity (WiFi-only). El seqnum delta se valida en
        # Python (mod 4096). Filtro DURO por fingerprint: si el candidato y la
        # deteccion tienen fingerprint conocido y DISTINTO, son dispositivos
        # distintos aunque el seqnum coincida (12 bits → colisiones posibles).
        if (
            self.seqnum_stitch_enabled
            and protocol == "wifi"
            and seqnum is not None
        ):
            rows = conn.execute(
                """SELECT group_id, seqnum, last_seen, fingerprint FROM hash_groups
                   WHERE protocol = 'wifi'
                     AND seqnum IS NOT NULL
                     AND rssi IS NOT NULL
                     AND ABS(rssi - ?) <= ?
                     AND last_seen >= ?""",
                (rssi, self.seqnum_rssi_delta, now - self.seqnum_stitch_window),
            ).fetchall()
            for group_id, cand_seqnum, last_seen, cand_fp in rows:
                if _seqnum_delta(int(cand_seqnum), seqnum) > self.seqnum_max_delta:
                    continue
                if fingerprint and cand_fp and cand_fp != fingerprint:
                    continue  # filtro duro: fingerprints distintos = otro device
                candidates.append((group_id, last_seen))

        # Regla 2: cross-protocol L2 short window. Sin filtro de fingerprint:
        # WiFi y BLE tienen namespaces de fingerprint distintos, no comparables.
        for group_id, last_seen in conn.execute(
            """SELECT group_id, last_seen FROM hash_groups
               WHERE protocol = ?
                 AND rssi IS NOT NULL
                 AND ABS(rssi - ?) <= ?
                 AND last_seen >= ?""",
            (other_protocol, rssi, self.cross_rssi_delta, now - self.cross_window),
        ).fetchall():
            candidates.append((group_id, last_seen))

        # Regla 3: BLE anchor long window — grupo con miembro del otro protocolo
        # activo en los ultimos N seg, RSSI compatible.
        if self.ble_anchor_enabled:
            anchor_protocol = "ble" if protocol == "wifi" else "wifi"
            for group_id, last_seen in conn.execute(
                """SELECT group_id, last_seen FROM hash_groups
                   WHERE protocol = ?
                     AND rssi IS NOT NULL
                     AND ABS(rssi - ?) <= ?
                     AND last_seen >= ?""",
                (anchor_protocol, rssi, self.cross_rssi_delta, now - self.ble_anchor_window),
            ).fetchall():
                candidates.append((group_id, last_seen))

        # Regla 4: fingerprint continuity (MISMO protocolo). Re-une rotaciones
        # que el seqnum no cubre — iOS resetea el seqnum al rotar la MAC, pero
        # el fingerprint de IEs (WiFi) / manufacturer-data (BLE) es estable.
        # Mismo fingerprint + RSSI compatible + dentro de la ventana = mismo
        # aparato. Requiere fingerprint no vacio. (Caveat: dos devices identicos
        # co-presentes pueden mergearse → leve subconteo; aceptado vs el
        # sobreconteo que arregla, y el gate de RSSI lo acota.)
        if self.fingerprint_stitch_enabled and fingerprint:
            for group_id, last_seen in conn.execute(
                """SELECT group_id, last_seen FROM hash_groups
                   WHERE protocol = ?
                     AND fingerprint = ?
                     AND rssi IS NOT NULL
                     AND ABS(rssi - ?) <= ?
                     AND last_seen >= ?""",
                (
                    protocol,
                    fingerprint,
                    rssi,
                    self.cross_rssi_delta,
                    now - self.fingerprint_stitch_window,
                ),
            ).fetchall():
                candidates.append((group_id, last_seen))

        if not candidates:
            return None

        # Multi-match: el grupo con last_seen mas reciente.
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

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
                # COALESCE(max_rssi, rssi): cuenta sobre la RSSI MÁS FUERTE
                # vista (no la última) → un device que estuvo cerca cuenta como
                # shopper de forma estable. Fallback a rssi para filas viejas
                # sin max_rssi.
                where = "COALESCE(max_rssi, rssi) IS NOT NULL AND COALESCE(max_rssi, rssi) >= ?"
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

    def get_recent_records(
        self,
        limit: int = 10,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
    ) -> list[dict]:
        """Últimos ``limit`` registros (hash groups) vistos, más recientes
        primero. Para el live preview / debug del tráfico exterior.

        Devuelve por registro: ``hash`` truncado (8 hex, ya es un hash —
        nunca MAC cruda), ``protocol`` (wifi/ble), ``last_seen`` (epoch s),
        ``rssi`` (dBm o None), y ``classification`` derivada del RSSI
        (shopper >= ``rssi_shopper``, passerby >= ``rssi_passerby``, si no
        ``weak``; ``unknown`` sin RSSI).
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT hash, protocol, last_seen, rssi FROM hash_groups "
                "ORDER BY last_seen DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: list[dict] = []
        for h, proto, ts, rssi in rows:
            if rssi is None:
                cls = "unknown"
            elif rssi >= rssi_shopper:
                cls = "shopper"
            elif rssi >= rssi_passerby:
                cls = "passerby"
            else:
                cls = "weak"
            out.append(
                {
                    "hash": h[:8],
                    "protocol": proto,
                    "last_seen": ts,
                    "rssi": rssi,
                    "classification": cls,
                }
            )
        return out

    def get_window_records(
        self,
        since_ts: float,
        until_ts: float | None = None,
    ) -> list[dict]:
        """Per-device records de los visitors OBSERVADOS en la ventana.

        Devuelve UN dict por ``group_id`` cuyo MAX(last_seen) cae dentro de
        ``[since_ts, until_ts)`` — es decir, cualquier grupo que tuvo
        actividad observada durante la ventana, no sólo los nuevos. Esto
        permite que un visitor presente en N ventanas seguidas se emita N
        veces (uno por window), con su ``rssi_max`` LIFETIME (max de todas
        las observaciones del group hasta el cierre de la ventana).

        Semántica resultante en el cloud:
          - Visitor pasa lejos en ventana N, se acerca en ventana N+1:
            ventana N emite con rssi_max=-75 (passerby), ventana N+1 emite
            con rssi_max=-50 (shopper). Promoción capturada limpio.
          - Visitor presente en ventanas N..N+5: 6 rows con su lifetime
            max (no se "desclasifica" una vez que se acercó — match con
            la semántica retail estándar de funnel).
          - Visitor observado una vez en ventana N y se va: 1 row, simple.

        Cada record:

            {
                "visitor_hash":   "<32 hex chars>" (group_id),
                "protocol":       "wifi" | "ble" (del miembro con MAX rssi),
                "rssi_max":       int dBm (MAX lifetime sobre miembros del group),
                "first_seen_ts":  float epoch (MIN sobre miembros),
                "last_seen_ts":   float epoch (MAX sobre miembros — cae en
                                  la ventana actual),
            }

        Filtrado:
          - Grupos sin ninguna lectura de RSSI se descartan (rssi_max None
            no es persistible — la columna RDS es NOT NULL).
          - Grupos cuyo MAX rssi cae por debajo de -90 dBm se descartan:
            eco lejano que no aporta señal y sería filtrado por
            ``rssi_class`` server-side igualmente.

        Reemplaza ``get_window_summary`` (removida en PR 2). En la Lambda
        cada row del array es idempotente por (device_id, visitor_hash,
        period_start) — retries del mismo período colapsan via ON CONFLICT
        DO UPDATE con GREATEST.
        """
        if until_ts is None:
            until_ts = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Filtramos por MAX(last_seen) del group EN la ventana — captura
            # tanto visitors nuevos como los "continuos" que siguen presentes.
            # last_seen del miembro se actualiza con cada observación, así que
            # max(last_seen) >= since indica actividad reciente del group.
            rows = conn.execute(
                """
                WITH groups_active_in_window AS (
                    SELECT group_id, MAX(last_seen) AS group_last_seen
                    FROM hash_groups
                    GROUP BY group_id
                    HAVING MAX(last_seen) >= ? AND MAX(last_seen) < ?
                )
                SELECT
                    g.group_id,
                    -- protocolo del miembro con max_rssi más alto (argmax,
                    -- equivalente al protocolo "dominante" del visitor).
                    (SELECT protocol FROM hash_groups
                     WHERE group_id = g.group_id AND max_rssi IS NOT NULL
                     ORDER BY max_rssi DESC LIMIT 1) AS protocol,
                    MAX(COALESCE(max_rssi, rssi))    AS rssi_max,
                    MIN(first_seen)                  AS first_seen,
                    g.group_last_seen                AS last_seen
                FROM hash_groups h
                JOIN groups_active_in_window g ON h.group_id = g.group_id
                GROUP BY g.group_id, g.group_last_seen
                """,
                (since_ts, until_ts),
            ).fetchall()

        out: list[dict] = []
        for group_id, protocol, rssi_max, first_seen, last_seen in rows:
            if rssi_max is None or protocol is None:
                continue  # grupo sin lectura de RSSI — no persistible
            if rssi_max < -90:
                continue  # eco lejano: se descartaría en rssi_class() anyway
            # group_id es UUID hex (32 chars = 16 bytes). Lo emitimos como-is
            # — la Lambda lo convierte a BYTEA via bytes.fromhex.
            out.append({
                "visitor_hash":  group_id,
                "protocol":      protocol,
                "rssi_max":      int(rssi_max),
                "first_seen_ts": float(first_seen),
                "last_seen_ts":  float(last_seen),
            })
        return out

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

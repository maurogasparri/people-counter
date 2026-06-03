"""Tests para el engine de dedup WiFi/BLE con stitching."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from src.wifi_ble.dedup import DedupEngine, _seqnum_delta


def _make_engine(
    seqnum_enabled: bool = True,
    ble_anchor_enabled: bool = True,
    seqnum_window: float = 30.0,
    ble_window: float = 900.0,
) -> tuple[DedupEngine, str]:
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "dedup.db")
    return (
        DedupEngine(
            db_path,
            cross_window_seconds=2.0,
            cross_rssi_delta=5.0,
            seqnum_stitch_enabled=seqnum_enabled,
            seqnum_stitch_window_seconds=seqnum_window,
            seqnum_max_delta=100,
            seqnum_rssi_delta=5.0,
            ble_anchor_enabled=ble_anchor_enabled,
            ble_anchor_window_seconds=ble_window,
        ),
        tmpdir,
    )


# ---------------------------------------------------------------------------
# Comportamiento basico (compat con la API publica anterior)
# ---------------------------------------------------------------------------

def test_first_detection_is_new():
    engine, _ = _make_engine()
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert result["is_new"] is True
    assert result["unified"] is False
    assert "group_id" in result


def test_duplicate_same_protocol():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert result["is_new"] is False


def test_same_mac_different_protocol_unifies():
    """Mismo MAC visto en WiFi y BLE en ventana corta -> mismo grupo."""
    engine, _ = _make_engine()
    r1 = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    r2 = engine.process_detection("AA:BB:CC:DD:EE:FF", "ble", -58.0)
    assert r2["is_new"] is True
    assert r2["unified"] is True
    assert r2["group_id"] == r1["group_id"]


def test_cross_protocol_rssi_too_different():
    """RSSI lejos -> dos grupos distintos."""
    engine, _ = _make_engine()
    r1 = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    r2 = engine.process_detection("11:22:33:44:55:66", "ble", -30.0)
    assert r2["unified"] is False
    assert r2["group_id"] != r1["group_id"]


def test_unique_count_counts_groups_not_hashes():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    engine.process_detection("11:22:33:44:55:66", "wifi", -55.0)
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)  # duplicate
    assert engine.get_unique_count() == 2


def test_reset_daily():
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert engine.get_unique_count() == 1
    engine.reset_daily()
    assert engine.get_unique_count() == 0
    result = engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)
    assert result["is_new"] is True


# ---------------------------------------------------------------------------
# Traffic counts (RSSI dual-threshold)
# ---------------------------------------------------------------------------

def test_traffic_counts_dual_threshold():
    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -70.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -65.0)
    engine.process_detection("AA:BB:CC:DD:EE:03", "wifi", -72.0)
    engine.process_detection("AA:BB:CC:DD:EE:04", "wifi", -50.0)
    engine.process_detection("AA:BB:CC:DD:EE:05", "wifi", -45.0)
    engine.process_detection("AA:BB:CC:DD:EE:06", "wifi", -80.0)

    counts = engine.get_traffic_counts(rssi_passerby=-75, rssi_shopper=-55)
    # 5 grupos arriba de -75 (los -70/-65/-72/-50/-45). 2 arriba de -55 (-50/-45).
    # El de -80 cae afuera.
    assert counts["passersby"] == 5
    assert counts["shoppers"] == 2
    assert counts["turn_in_rate"] == round(2 / 5, 4)


def test_traffic_counts_empty():
    engine, _ = _make_engine()
    counts = engine.get_traffic_counts()
    assert counts["passersby"] == 0
    assert counts["shoppers"] == 0
    assert counts["turn_in_rate"] == 0.0


def test_traffic_counts_filter_by_protocol():
    """Protocol filter cuenta grupos donde algun miembro de ese protocol
    pasa el threshold. Para no ambiguar el test, el BLE viene con RSSI
    lejos de los WiFi para que NO se unifique con ninguno."""
    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -40.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -45.0)
    # RSSI -20 esta a 25 dBm de los WiFi (mucho mas que cross_rssi_delta=5)
    # -> no unifica, queda como su propio grupo.
    engine.process_detection("11:22:33:44:55:66", "ble", -20.0)

    mixed = engine.get_traffic_counts()
    wifi_only = engine.get_traffic_counts(protocol="wifi")
    ble_only = engine.get_traffic_counts(protocol="ble")

    assert mixed["shoppers"] == 3
    assert wifi_only["shoppers"] == 2
    assert ble_only["shoppers"] == 1


# ---------------------------------------------------------------------------
# Window summaries
# ---------------------------------------------------------------------------

def test_get_recent_hashes_returns_window():
    engine, _ = _make_engine()
    t0 = time.time()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -60.0)
    engine.process_detection("AA:BB:CC:DD:EE:02", "wifi", -60.0)

    recent = engine.get_recent_hashes(since_ts=t0 - 1, until_ts=t0 + 10)
    assert len(recent) == 2
    assert all(len(h) == 32 for h in recent)


# get_window_records (post PR 2: per-device events para el cloud)
# ---------------------------------------------------------------------------

def test_get_window_records_one_per_group():
    """Un record por group_id — múltiples MACs stiched cuentan como 1 visitor."""
    engine, _ = _make_engine()
    t0 = time.time()
    # Mismo dispositivo emitiendo cross-protocol dentro de la ventana corta
    # → mismo group_id, un solo record.
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -55.0)
    engine.process_detection("AA:BB:CC:DD:EE:01", "ble", -57.0)
    # Otro dispositivo distinto.
    engine.process_detection("11:22:33:44:55:66", "wifi", -70.0)

    records = engine.get_window_records(since_ts=t0 - 1, until_ts=t0 + 10)
    assert len(records) == 2
    # Cada record tiene el shape esperado.
    for r in records:
        assert set(r.keys()) == {
            "visitor_hash", "protocol", "rssi_max",
            "first_seen_ts", "last_seen_ts",
        }
        assert r["protocol"] in ("wifi", "ble")
        assert isinstance(r["rssi_max"], int)
    # visitor_hash es UUID hex de 32 chars (group_id).
    assert all(len(r["visitor_hash"]) == 32 for r in records)


def test_get_window_records_rssi_max_aggregates_over_group():
    """rssi_max sale del miembro con la lectura más fuerte del grupo.

    Para que se stitchen cross-protocol el RSSI debe estar dentro del gate
    (cross_rssi_delta=5 dBm). Probamos rotación intra-protocolo (WiFi MAC
    rotation con seqnum continuity) con dos RSSI distintos del mismo chip.
    """
    engine, _ = _make_engine()
    t0 = time.time()
    # Dos WiFi MACs distintas, seqnum cercano + RSSI cercano → mismo grupo.
    engine.process_detection("AA:00:00:00:00:01", "wifi", -55.0, seqnum=1000)
    engine.process_detection("BB:00:00:00:00:01", "wifi", -52.0, seqnum=1005)

    records = engine.get_window_records(since_ts=t0 - 1, until_ts=t0 + 10)
    assert len(records) == 1
    # rssi_max = MAX(-55, -52) = -52.
    assert records[0]["rssi_max"] == -52
    assert records[0]["protocol"] == "wifi"


def test_get_window_records_filters_weak_eco():
    """Grupos con rssi_max < -90 dBm se descartan (eco lejano)."""
    engine, _ = _make_engine()
    t0 = time.time()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -95.0)  # eco
    engine.process_detection("11:22:33:44:55:66", "wifi", -60.0)  # válido

    records = engine.get_window_records(since_ts=t0 - 1, until_ts=t0 + 10)
    assert len(records) == 1
    assert records[0]["rssi_max"] == -60


def test_get_window_records_excludes_groups_outside_window():
    """Grupos cuyo MAX(last_seen) cae fuera de la ventana se excluyen."""
    engine, _ = _make_engine()
    t0 = time.time()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -55.0)

    # Ventana en el futuro — el group quedó "viejo", no debe incluirse.
    records = engine.get_window_records(since_ts=t0 + 100, until_ts=t0 + 200)
    assert records == []


def test_get_window_records_emits_continuous_visitor_each_window():
    """Visitor presente en varias ventanas se emite UNA vez por ventana.

    Caso clave del PR 2.1: el visitor que persiste entre ventanas (no es
    "nuevo" en la ventana N+1, pero sigue ahí) debe seguir emitiéndose
    para que el funnel del bucket N+1 lo refleje, y para capturar
    promociones passerby→shopper cross-window.
    """
    engine, _ = _make_engine()
    # Observación 1 dentro de la ventana A.
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -75.0)
    win_a_end = time.time() + 0.01
    # Esperamos a que el reloj pase (ventana A es pasada). Cambia last_seen
    # del MISMO group (rotación virtual: misma MAC), simulando observación
    # del visitor que sigue presente.
    import time as _t
    _t.sleep(0.05)
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -50.0)  # promovió
    win_b_now = time.time() + 0.01

    # Ventana B: el group sigue activo (su last_seen cae en B).
    records_b = engine.get_window_records(
        since_ts=win_a_end, until_ts=win_b_now,
    )
    assert len(records_b) == 1
    assert records_b[0]["rssi_max"] == -50  # lifetime max — promoción capturada


def test_get_window_records_visitor_inactive_not_re_emitted():
    """Visitor cuyo last_seen quedó en ventana A NO se emite en ventana B.

    El filtro es "MAX(last_seen) IN window", no "first_seen". Una vez que
    el visitor dejó de aparecer (sin actualizar last_seen), no se re-emite
    eternamente — sólo mientras esté activo.
    """
    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:01", "wifi", -55.0)
    win_a_end = time.time() + 0.01

    # Ventana B en el futuro, sin nuevas observaciones del visitor.
    records_b = engine.get_window_records(
        since_ts=win_a_end + 10, until_ts=win_a_end + 20,
    )
    assert records_b == []


# ---------------------------------------------------------------------------
# Seqnum stitching (regla 1)
# ---------------------------------------------------------------------------

def test_seqnum_stitch_continuous_within_window():
    """Dos WiFi MACs con seqnum cercano + RSSI cercano + tiempo cercano -> 1 grupo."""
    engine, _ = _make_engine()
    r1 = engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=1005)
    assert r2["unified"] is True
    assert r2["group_id"] == r1["group_id"]
    assert engine.get_unique_count() == 1


def test_seqnum_stitch_discontinuous_seqnum():
    """Seqnum lejos -> grupos distintos aunque RSSI sea cercano."""
    engine, _ = _make_engine()
    r1 = engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=2500)
    assert r2["unified"] is False
    assert r2["group_id"] != r1["group_id"]
    assert engine.get_unique_count() == 2


def test_seqnum_stitch_rssi_too_different():
    """Seqnum cercano pero RSSI lejos -> grupos distintos."""
    engine, _ = _make_engine()
    r1 = engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -45.0, seqnum=1005)
    assert r2["unified"] is False
    assert r2["group_id"] != r1["group_id"]


def test_seqnum_stitch_outside_window():
    """Seqnum cercano pero >30s aparte -> grupos distintos."""
    engine, _ = _make_engine(seqnum_window=1.0)  # 1s window — easy to exceed
    engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000)
    time.sleep(1.2)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=1005)
    assert r2["unified"] is False


def test_seqnum_stitch_disabled():
    """Con seqnum_stitch_enabled=False, MACs distintas son grupos distintos."""
    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=1005)
    assert r2["unified"] is False


def test_seqnum_stitch_requires_seqnum_present():
    """Sin seqnum (e.g. scapy no pudo parsear), no aplica regla 1."""
    engine, _ = _make_engine(ble_anchor_enabled=False)
    engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=None)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=None)
    assert r2["unified"] is False  # ningun seqnum -> regla 1 no aplica


def test_seqnum_wrap_around():
    """Seqnum 4090 + 5 con wrap = delta 11, dentro de max_delta=100."""
    engine, _ = _make_engine()
    engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=4090)
    r2 = engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=5)
    assert r2["unified"] is True


def test_seqnum_delta_helper():
    """Sanity de la funcion _seqnum_delta — wrap simetrico."""
    assert _seqnum_delta(100, 105) == 5
    assert _seqnum_delta(4090, 5) == 11
    assert _seqnum_delta(5, 4090) == 11
    assert _seqnum_delta(0, 2048) == 2048  # antipodes


# ---------------------------------------------------------------------------
# BLE anchoring (regla 3)
# ---------------------------------------------------------------------------

def test_ble_anchor_links_late_wifi_to_existing_ble_group():
    """Un BLE addr es seen en t=0; 5 min despues aparece una WiFi MAC con
    RSSI cercano. Sin ble_anchor caerian en grupos distintos; con anchor
    on, se unen al mismo grupo BLE."""
    # Ventana de anchor = 10 min (mas que los 5 que vamos a esperar simulado)
    engine, _ = _make_engine(seqnum_enabled=False, ble_window=600.0)
    # Cross-window default es 2s — no aplica.
    # ble_anchor_window=600s deberia agarrar nuestra deteccion 5min despues.

    # Pero como time.sleep(5min) es prohibitivo, simulamos manipulando el
    # SQLite directo: insertamos una observacion vieja a t=now-300s.
    import sqlite3 as _sqlite3
    db = engine.db_path
    fake_t = time.time() - 300.0
    with _sqlite3.connect(db) as conn:
        from src.wifi_ble.hasher import hash_mac
        ble_hash = hash_mac("AA:BB:CC:DD:EE:01", "")
        conn.execute(
            """INSERT INTO hash_groups
               (hash, protocol, group_id, first_seen, last_seen, rssi, seqnum)
               VALUES (?, 'ble', 'fake-ble-group', ?, ?, ?, NULL)""",
            (ble_hash, fake_t, fake_t, -60.0),
        )

    # Ahora una nueva WiFi MAC con RSSI cercano al BLE viejo.
    r = engine.process_detection("11:22:33:44:55:66", "wifi", -62.0)
    assert r["unified"] is True
    assert r["group_id"] == "fake-ble-group"


def test_ble_anchor_disabled_means_no_long_window_stitch():
    """Con ble_anchor_enabled=False, BLE viejo no agarra WiFi nuevo (solo
    el cross_window_seconds de 2s aplica)."""
    engine, _ = _make_engine(
        seqnum_enabled=False, ble_anchor_enabled=False, ble_window=600.0
    )
    import sqlite3 as _sqlite3
    from src.wifi_ble.hasher import hash_mac

    db = engine.db_path
    fake_t = time.time() - 300.0
    with _sqlite3.connect(db) as conn:
        ble_hash = hash_mac("AA:BB:CC:DD:EE:01", "")
        conn.execute(
            """INSERT INTO hash_groups
               (hash, protocol, group_id, first_seen, last_seen, rssi, seqnum)
               VALUES (?, 'ble', 'fake-ble-group', ?, ?, ?, NULL)""",
            (ble_hash, fake_t, fake_t, -60.0),
        )

    r = engine.process_detection("11:22:33:44:55:66", "wifi", -62.0)
    # Sin anchor + sin cross_window match (BLE muy viejo) -> nuevo grupo.
    assert r["unified"] is False
    assert r["group_id"] != "fake-ble-group"


# ---------------------------------------------------------------------------
# Stitching ratio (telemetria)
# ---------------------------------------------------------------------------

def test_stitching_ratio_no_data_returns_none():
    engine, _ = _make_engine()
    assert engine.get_stitching_ratio() is None


def test_stitching_ratio_no_stitch_is_one():
    """Sin stitches efectivos -> ratio = 1.0 (cada hash = su grupo)."""
    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0)
    engine.process_detection("BB:00:00:00:00:01", "wifi", -45.0)  # RSSI lejos
    ratio = engine.get_stitching_ratio()
    assert ratio == 1.0


def test_stitching_ratio_with_seqnum_stitch():
    """2 WiFi MACs stitched + 1 BLE solo -> 3 hashes, 2 grupos, ratio = 2/3."""
    engine, _ = _make_engine(ble_anchor_enabled=False)
    engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000)
    engine.process_detection("BB:00:00:00:00:01", "wifi", -61.0, seqnum=1005)
    engine.process_detection("CC:00:00:00:00:01", "ble", -30.0)  # RSSI lejos
    ratio = engine.get_stitching_ratio()
    assert ratio == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# Self-validation: "1 device en cuarto vacio" no debe ser N personas
# ---------------------------------------------------------------------------

def test_single_device_with_mac_rotations_counts_as_one():
    """Simulacion: un Android emite 10 probes con MAC distinta cada 3s
    (rotation agresiva), pero seqnum continuo y RSSI estable -> 1 grupo."""
    engine, _ = _make_engine()
    seqnum = 500
    for i in range(10):
        # MAC distinta cada vez (rotacion agresiva del OS)
        mac = f"AA:BB:CC:00:00:{i:02x}"
        # Seqnum incrementa ~30 por iteracion (chip activo emite ~10/s)
        seqnum = (seqnum + 30) % 4096
        engine.process_detection(mac, "wifi", -55.0, seqnum=seqnum)

    assert engine.get_unique_count() == 1


def test_three_phones_close_together_count_as_three():
    """Tres dispositivos distintos cerca uno del otro (RSSI similar) pero
    cada uno con su propio seqnum stream -> 3 grupos."""
    engine, _ = _make_engine()
    # Phone A: seqnum 1000s
    engine.process_detection("AA:00:00:00:00:01", "wifi", -55.0, seqnum=1000)
    engine.process_detection("AA:00:00:00:00:02", "wifi", -56.0, seqnum=1010)
    # Phone B: seqnum 3000s (offset claro)
    engine.process_detection("BB:00:00:00:00:01", "wifi", -54.0, seqnum=3000)
    engine.process_detection("BB:00:00:00:00:02", "wifi", -55.0, seqnum=3010)
    # Phone C: seqnum 200s
    engine.process_detection("CC:00:00:00:00:01", "wifi", -57.0, seqnum=200)
    engine.process_detection("CC:00:00:00:00:02", "wifi", -56.0, seqnum=210)

    assert engine.get_unique_count() == 3


# ---------------------------------------------------------------------------
# Regla 4 — fingerprint continuity (mismo protocolo) + filtro duro
# ---------------------------------------------------------------------------


def test_fingerprint_stitch_merges_rotation_without_seqnum():
    """Mismo fingerprint + RSSI compatible + ventana, SIN seqnum continuity
    (iOS lo resetea al rotar) -> mismo group_id (regla 4)."""
    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    fp = "deadbeefcafe0001"
    r1 = engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, fingerprint=fp)
    r2 = engine.process_detection("BB:00:00:00:00:02", "wifi", -61.0, fingerprint=fp)
    assert r2["unified"] is True
    assert r1["group_id"] == r2["group_id"]
    assert engine.get_unique_count() == 1


def test_fingerprint_distinct_no_stitch():
    """Fingerprints distintos -> grupos distintos (no se mergean)."""
    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    r1 = engine.process_detection("AA:00:00:00:00:01", "wifi", -60.0, fingerprint="fp_a")
    r2 = engine.process_detection("BB:00:00:00:00:02", "wifi", -61.0, fingerprint="fp_b")
    assert r2["unified"] is False
    assert r1["group_id"] != r2["group_id"]


def test_fingerprint_hard_filter_blocks_seqnum_coincidence():
    """Seqnums continuos (delta chico) PERO fingerprints distintos -> NO mergea
    (filtro duro): dos devices cuyos seqnums coincidieron por casualidad."""
    engine, _ = _make_engine(seqnum_enabled=True, ble_anchor_enabled=False)
    r1 = engine.process_detection(
        "AA:00:00:00:00:01", "wifi", -60.0, seqnum=1000, fingerprint="fp_a"
    )
    r2 = engine.process_detection(
        "BB:00:00:00:00:02", "wifi", -61.0, seqnum=1005, fingerprint="fp_b"
    )
    assert r2["unified"] is False
    assert r1["group_id"] != r2["group_id"]


def test_traffic_counts_uses_max_rssi_not_latest():
    """Un device que UNA vez estuvo cerca sigue contando como shopper aunque su
    RSSI baje despues — get_traffic_counts usa el MAX, no la ultima lectura
    (evita el flapping 6->7->6 del contador)."""
    engine, _ = _make_engine()
    # Mismo hash: primero cerca (-50 >= -55 shopper), despues lejos (-80).
    engine.process_detection("AA:00:00:00:00:01", "wifi", -50.0, fingerprint="fp1")
    engine.process_detection("AA:00:00:00:00:01", "wifi", -80.0, fingerprint="fp1")
    counts = engine.get_traffic_counts(rssi_passerby=-75.0, rssi_shopper=-55.0)
    assert counts["shoppers"] == 1  # estable: cuenta por max_rssi (-50), no -80


# ---------------------------------------------------------------------------
# Salt local at-rest (privacidad) + thread-safety
# ---------------------------------------------------------------------------


def test_mac_hashes_stored_with_nonempty_salt():
    """Las MACs at-rest se hashean con el salt local, NO con salt vacio.

    Sin sal, el hash truncado de 16 bytes es brute-forceable sobre el espacio
    OUI por quien lea el .sqlite. Verificamos que la fila guardada NO matchea
    el hash sin-sal y SI matchea el hash con el salt del engine."""
    import sqlite3 as _sqlite3

    from src.wifi_ble.hasher import hash_mac

    engine, _ = _make_engine()
    mac = "AA:BB:CC:DD:EE:FF"
    engine.process_detection(mac, "wifi", -60.0)

    assert engine._salt  # no vacio
    with _sqlite3.connect(engine.db_path) as conn:
        stored = {
            row[0]
            for row in conn.execute("SELECT hash FROM hash_groups")
        }
    assert hash_mac(mac, "") not in stored  # NO se guardo sin sal
    assert hash_mac(mac, engine._salt) in stored  # SI con el salt del engine


def test_salt_persists_across_reopen():
    """El salt se persiste en el DB: reabrir el mismo path da el mismo salt
    (asi un restart del proceso no rompe la continuidad del stitching)."""
    engine, tmpdir = _make_engine()
    salt1 = engine._salt
    reopened = DedupEngine(
        engine.db_path,
        cross_window_seconds=2.0,
        cross_rssi_delta=5.0,
        seqnum_stitch_enabled=True,
        seqnum_stitch_window_seconds=30.0,
        seqnum_max_delta=100,
        seqnum_rssi_delta=5.0,
        ble_anchor_enabled=True,
        ble_anchor_window_seconds=900.0,
    )
    assert reopened._salt == salt1


def test_reset_daily_rotates_salt():
    """reset_daily rota el salt (defensa extra contra linkability cross-dia)."""
    engine, _ = _make_engine()
    salt_before = engine._salt
    engine.reset_daily()
    assert engine._salt != salt_before
    assert engine._salt  # sigue siendo no vacio


def test_concurrent_process_detection_same_mac_no_race():
    """Dos productores (WiFi/BLE en threads distintos) comparten el DB. El lock
    serializa el SELECT-then-INSERT: procesar la MISMA MAC nueva desde N threads
    a la vez NO debe tirar IntegrityError ni crear filas duplicadas."""
    import threading

    engine, _ = _make_engine(seqnum_enabled=False, ble_anchor_enabled=False)
    mac = "AA:BB:CC:DD:EE:FF"
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def _worker() -> None:
        barrier.wait()  # arrancan todos juntos para maximizar el solape
        try:
            engine.process_detection(mac, "wifi", -60.0)
        except Exception as e:  # noqa: BLE001 — el test reporta cualquier fallo
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Una sola fila para (hash, protocol) pese a las 20 calls concurrentes.
    assert engine.get_unique_count() == 1


def test_wal_mode_enabled():
    """El dedup abre en WAL: permite que el publisher LEA mientras un productor
    ESCRIBE. Sin esto, los reads chocaban con writes → 'database is locked'."""
    engine, _ = _make_engine()
    conn = engine._connect()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_read_during_open_write_does_not_lock():
    """Regresión (FPS-killer del piloto): una lectura del publisher
    (get_window_records) mientras un productor mantiene una transacción de
    escritura abierta NO debe tirar 'database is locked'. WAL deja leer el
    último snapshot commiteado sin bloqueo mutuo."""
    import sqlite3

    engine, _ = _make_engine()
    engine.process_detection("AA:BB:CC:DD:EE:FF", "wifi", -60.0)

    writer = engine._connect()
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO hash_groups (hash, protocol, group_id, first_seen, last_seen) "
            "VALUES ('deadbeef', 'wifi', 'g-test', 0.0, 0.0)"
        )
        try:
            records = engine.get_window_records(since_ts=0.0, until_ts=time.time())
        except sqlite3.OperationalError as e:
            raise AssertionError(f"lectura bloqueada por un write abierto: {e}")
        assert isinstance(records, list)
    finally:
        writer.rollback()
        writer.close()

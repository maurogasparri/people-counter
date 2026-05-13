-- People Counter — schema Postgres.
--
-- 3 tablas, una por tipo de evento. Sin foreign keys cross-tabla porque
-- counting_events, telemetry y wifi_ble_summaries son streams independientes
-- — los JOINs cuando hacen falta (ej. turn-in rate = counting.in / wifi.passersby)
-- se hacen por device_id + timestamp en las queries.
--
-- Aplicar con: psql -U people_counter -d people_counter -f schema.sql

CREATE EXTENSION IF NOT EXISTS pg_stat_statements;


-- =============================================================================
-- counting_events — un row por cruce de línea (in/out) detectado por las cámaras
-- =============================================================================
CREATE TABLE IF NOT EXISTS counting_events (
    id              BIGSERIAL PRIMARY KEY,
    device_id       TEXT NOT NULL,
    store_id        TEXT NOT NULL,
    event_time      TIMESTAMPTZ NOT NULL,           -- timestamp del evento en el device
    ingest_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    direction       TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    track_id        INTEGER,
    confidence      REAL,
    height_class    TEXT CHECK (height_class IN ('adult', 'child', 'unknown')),
    CONSTRAINT counting_events_uniq UNIQUE (device_id, event_time, track_id, direction)
);

CREATE INDEX IF NOT EXISTS idx_counting_store_time
    ON counting_events (store_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_counting_device_time
    ON counting_events (device_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_counting_direction
    ON counting_events (direction, event_time DESC);


-- =============================================================================
-- telemetry — un row por muestra periódica de salud del device (default 1h)
-- =============================================================================
CREATE TABLE IF NOT EXISTS telemetry (
    id                      BIGSERIAL PRIMARY KEY,
    device_id               TEXT NOT NULL,
    store_id                TEXT NOT NULL,
    sample_time             TIMESTAMPTZ NOT NULL,
    ingest_time             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cpu_temp_c              REAL,
    hailo_temp_c            REAL,
    disk_used_pct           REAL,
    uptime_s                INTEGER,
    fps                     REAL,
    p50_latency_ms          REAL,
    p99_latency_ms          REAL,
    detection_rate_per_min  REAL,
    active_tracks           INTEGER,
    buffer_pending          INTEGER,
    mqtt_connected          BOOLEAN,
    total_in                INTEGER,
    total_out               INTEGER,
    wifi_probe_ok           BOOLEAN,                 -- false si la captura WiFi murió
    ble_scanner_ok          BOOLEAN,                 -- false si BLE murió
    schedule_error          TEXT                     -- "invalid_schedule" cuando aplica
);

CREATE INDEX IF NOT EXISTS idx_telemetry_device_time
    ON telemetry (device_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_store_time
    ON telemetry (store_id, sample_time DESC);


-- =============================================================================
-- wifi_ble_summaries — un row por ventana de probe_interval_seconds (default 15min)
-- =============================================================================
-- Combina WiFi + BLE en counts agregados post-L2 dedup local. Un dispositivo
-- detectado por ambos protocolos cuenta como 1 visitante (no 2).
--
-- passersby (RSSI >= -75 dBm): pasó por la zona — incluye vereda.
-- shoppers  (RSSI >= -55 dBm): pasó muy cerca — proxy de "consideró entrar".
-- in real:  se cuenta en counting_events (cruce de línea por cámara).
--
-- turn_in_rate y conversion_rate se calculan en queries de Grafana, NO se
-- almacenan — así cambiar la fórmula no requiere migrar datos.
CREATE TABLE IF NOT EXISTS wifi_ble_summaries (
    id              BIGSERIAL PRIMARY KEY,
    device_id       TEXT NOT NULL,
    store_id        TEXT NOT NULL,
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    ingest_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    passersby       INTEGER NOT NULL DEFAULT 0,
    shoppers        INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT wifi_ble_summaries_uniq UNIQUE (device_id, period_start, period_end),
    CONSTRAINT shoppers_le_passersby CHECK (shoppers <= passersby)
);

CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_time
    ON wifi_ble_summaries (store_id, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_device_time
    ON wifi_ble_summaries (device_id, period_start DESC);


-- =============================================================================
-- Views útiles para Grafana — encapsulan agregaciones comunes así los queries
-- de los paneles quedan más cortos. NO son materialized; refrescan en cada read.
-- =============================================================================

-- Conteo neto por hora por local.
CREATE OR REPLACE VIEW v_counting_hourly AS
SELECT
    store_id,
    date_trunc('hour', event_time) AS hour,
    SUM(CASE WHEN direction = 'in'  THEN 1 ELSE 0 END) AS ins,
    SUM(CASE WHEN direction = 'out' THEN 1 ELSE 0 END) AS outs,
    SUM(CASE WHEN direction = 'in'  THEN 1 ELSE -1 END) AS net
FROM counting_events
GROUP BY store_id, date_trunc('hour', event_time);

-- Turn-in rate por hora: gente que entró / gente que pasó cerca.
-- Cuenta cámara (in real) sobre passersby de WiFi/BLE.
CREATE OR REPLACE VIEW v_turn_in_rate_hourly AS
WITH ins AS (
    SELECT
        store_id,
        date_trunc('hour', event_time) AS hour,
        COUNT(*) AS ins
    FROM counting_events
    WHERE direction = 'in'
    GROUP BY store_id, date_trunc('hour', event_time)
),
passers AS (
    SELECT
        store_id,
        date_trunc('hour', period_start) AS hour,
        SUM(passersby) AS passersby,
        SUM(shoppers)  AS shoppers
    FROM wifi_ble_summaries
    GROUP BY store_id, date_trunc('hour', period_start)
)
SELECT
    COALESCE(i.store_id, p.store_id) AS store_id,
    COALESCE(i.hour,     p.hour)     AS hour,
    COALESCE(i.ins, 0)               AS ins,
    COALESCE(p.passersby, 0)         AS passersby,
    COALESCE(p.shoppers, 0)          AS shoppers,
    CASE WHEN COALESCE(p.passersby, 0) > 0
         THEN i.ins::float / p.passersby
         ELSE NULL END               AS turn_in_rate,
    CASE WHEN COALESCE(p.shoppers, 0) > 0
         THEN i.ins::float / p.shoppers
         ELSE NULL END               AS conversion_rate
FROM ins i
FULL OUTER JOIN passers p ON p.store_id = i.store_id AND p.hour = i.hour;

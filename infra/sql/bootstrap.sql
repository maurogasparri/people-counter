-- =============================================================================
-- people-counter — bootstrap del schema en RDS Postgres
-- =============================================================================
-- Cómo ejecutar (desde DBeaver, conectado como master user "people_counter"):
--   1. Abrir este archivo
--   2. Right-click → Execute → Execute SQL Script (Alt+X)
--   3. Revisar que todas las sentencias hayan corrido OK
--
-- Idempotente: usa IF NOT EXISTS / OR REPLACE donde aplica.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- Reset (dev/PoC only)
-- =============================================================================
-- En produccion esto se reemplaza por migrations versionadas (flyway, alembic).
-- Para PoC, dropeamos y recreamos. Cascade tira las views que dependen de las
-- tablas; se recrean abajo. IF EXISTS hace este bloque idempotente en cold start.

DROP TABLE IF EXISTS count_events     CASCADE;
DROP TABLE IF EXISTS wifi_ble_summary CASCADE;
DROP TABLE IF EXISTS telemetry        CASCADE;
DROP TABLE IF EXISTS sales            CASCADE;

-- =============================================================================
-- Tablas raw — escritas por Lambda persist_event (3 topics IoT distintos)
-- =============================================================================

CREATE TABLE IF NOT EXISTS count_events (
    event_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id        TEXT         NOT NULL,
    store_id         TEXT         NOT NULL,
    event_ts         TIMESTAMPTZ  NOT NULL,
    event_bucket     TIMESTAMPTZ,                 -- alineado a analytics.bucket_seconds en el device
    direction        TEXT         NOT NULL CHECK (direction IN ('in', 'out')),
    track_id         INT,
    confidence       REAL,
    position_y       REAL,                        -- coordenada Y del cruce (rect coords)
    height_class     TEXT,                        -- 'adult' | 'child' | 'unknown' (segun height_m)
    height_m         REAL,
    head_depth_m     REAL,
    total_in         INT,                         -- running total en el device al momento del evento
    total_out        INT,
    scaling_factor   REAL,                        -- multiplier del shadow (escala por sensibilidad)
    scaled_in        INT,
    scaled_out       INT,
    best_frame_path  TEXT,                        -- path local en el device (no contiene la imagen)
    received_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia para reintentos del Lambda / replay del buffer del device.
    UNIQUE (device_id, event_ts, track_id, direction)
);

CREATE INDEX IF NOT EXISTS idx_count_events_store_ts
    ON count_events (store_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_count_events_device_ts
    ON count_events (device_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_count_events_bucket
    ON count_events (store_id, event_bucket DESC) WHERE event_bucket IS NOT NULL;

CREATE TABLE IF NOT EXISTS wifi_ble_summary (
    summary_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id       TEXT         NOT NULL,
    store_id        TEXT         NOT NULL,
    period_start    TIMESTAMPTZ  NOT NULL,   -- inicio de la ventana (epoch del device)
    period_end      TIMESTAMPTZ  NOT NULL,   -- fin de la ventana
    period_bucket   TIMESTAMPTZ  NOT NULL,   -- bucket alineado para joins multi-cam (= period_start alineado)
    passersby       INT          NOT NULL,   -- post L2 dedup
    shoppers        INT          NOT NULL,   -- en rango cercano (RSSI fuerte)
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia ante retries del device: una sola fila por (device, ventana).
    UNIQUE (device_id, period_start, period_end)
);

CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_period
    ON wifi_ble_summary (store_id, period_bucket DESC);

CREATE TABLE IF NOT EXISTS telemetry (
    telemetry_id     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id        TEXT         NOT NULL,
    store_id         TEXT         NOT NULL,
    event_ts         TIMESTAMPTZ  NOT NULL,
    -- OS metrics
    uptime_s                      REAL,
    cpu_temp_c                    REAL,
    hailo_temp_c                  REAL,
    disk_free_mb                  INT,
    mem_available_mb              INT,
    -- Pipeline metrics
    fps                           REAL,
    frame_latency_p50_ms          REAL,
    frame_latency_p95_ms          REAL,
    detection_rate_per_min        REAL,
    tracker_confirmed_count       INT,
    tracker_pending_count         INT,
    total_in                      INT,
    total_out                     INT,
    -- MQTT health
    mqtt_connected                BOOLEAN,
    mqtt_disconnect_count         INT,
    seconds_since_last_reconnect  REAL,
    buffer_backlog_messages       INT,
    -- WiFi/BLE subsystem health
    wifi_probe_ok                 BOOLEAN,
    ble_scanner_ok                BOOLEAN,
    -- Stitching ratio del dedup: groups / hashes en el dia (1.0 = sin stitch
    -- efectivo, baja a medida que MAC rotations se mergean en el mismo group).
    wifi_ble_stitching_ratio      REAL,
    -- Schedule / error state (mando 'error' del payload + detalle largo)
    error                         TEXT,
    schedule_error_detail         TEXT,
    received_at                   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_telemetry_device_ts
    ON telemetry (device_id, event_ts DESC);

-- =============================================================================
-- Sales — placeholder para futura ingesta vía API Gateway desde POS externo
-- =============================================================================

CREATE TABLE IF NOT EXISTS sales (
    sale_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     TEXT         NOT NULL,    -- ID en el sistema POS origen
    store_id        TEXT         NOT NULL,
    sale_ts         TIMESTAMPTZ  NOT NULL,
    amount_minor    BIGINT       NOT NULL,    -- centavos (sin floats)
    currency        CHAR(3)      NOT NULL DEFAULT 'ARS',
    item_count      INT,
    payment_method  TEXT,
    cashier_id      TEXT,
    metadata        JSONB,
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (store_id, external_id)            -- POS puede reintentar sin duplicar
);

CREATE INDEX IF NOT EXISTS idx_sales_store_ts
    ON sales (store_id, sale_ts DESC);
CREATE INDEX IF NOT EXISTS idx_sales_received
    ON sales (received_at DESC);

-- =============================================================================
-- View multi-cam: agregación por store con MAX (no SUM) para WiFi/BLE
-- Cuando un store tenga 2+ cams, WiFi/BLE range (~30-50m) excede al vision
-- range (3-5m), entonces las cams ven la misma gente. Tomar MAX en vez de SUM
-- evita doble-conteo. Con 1 cam por store, MAX = el row de esa cam.
-- =============================================================================

CREATE OR REPLACE VIEW wifi_ble_store_traffic AS
SELECT
    period_bucket,
    store_id,
    MAX(passersby)   AS passersby,
    MAX(shoppers)    AS shoppers,
    COUNT(*)         AS cams_reporting
FROM wifi_ble_summary
GROUP BY period_bucket, store_id;

-- =============================================================================
-- Counting agregado por bucket del device (analytics.bucket_seconds, 15min default)
-- =============================================================================
-- event_bucket es la columna que el device alinea al multiplo del epoch segun
-- analytics.bucket_seconds (15min por default). Agrupar por esa columna mantiene
-- semantica device-correcta: cambiar el bucket via shadow no requiere recomputar
-- nada, y rows viejos preservan su bucket original.

CREATE OR REPLACE VIEW counting_by_bucket AS
SELECT
    store_id,
    event_bucket,
    COUNT(*) FILTER (WHERE direction = 'in')  AS ins,
    COUNT(*) FILTER (WHERE direction = 'out') AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
      - COUNT(*) FILTER (WHERE direction = 'out') AS net
FROM count_events
WHERE event_bucket IS NOT NULL
GROUP BY store_id, event_bucket;

-- =============================================================================
-- Turn-in rate por bucket: gente que entro / gente que paso cerca
-- =============================================================================
-- Une counting (ins) con WiFi/BLE (passersby/shoppers). Usa wifi_ble_store_traffic
-- (MAX por store) para multi-cam dedup. FULL OUTER JOIN preserva buckets donde
-- solo una fuente reporto (e.g., WiFi/BLE OFF en el device, o un bucket sin
-- transito que no genero counting events).

CREATE OR REPLACE VIEW turn_in_rate_by_bucket AS
WITH ins AS (
    SELECT store_id, event_bucket, COUNT(*) AS ins
    FROM count_events
    WHERE direction = 'in' AND event_bucket IS NOT NULL
    GROUP BY store_id, event_bucket
)
SELECT
    COALESCE(i.store_id, w.store_id)          AS store_id,
    COALESCE(i.event_bucket, w.period_bucket) AS bucket,
    COALESCE(i.ins, 0)                        AS ins,
    COALESCE(w.passersby, 0)                  AS passersby,
    COALESCE(w.shoppers, 0)                   AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0
         THEN i.ins::float / w.passersby
         ELSE NULL END                        AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0) > 0
         THEN i.ins::float / w.shoppers
         ELSE NULL END                        AS conversion_rate
FROM ins i
FULL OUTER JOIN wifi_ble_store_traffic w
    ON w.store_id = i.store_id AND w.period_bucket = i.event_bucket;

-- =============================================================================
-- Rollups hourly encima de los views de bucket
-- =============================================================================
-- Para reportes diarios donde 15min es demasiado fino y para charts de Grafana
-- de "ultimas 24h" sin aliasing.

CREATE OR REPLACE VIEW counting_hourly AS
SELECT
    store_id,
    date_trunc('hour', event_bucket) AS hour,
    SUM(ins)  AS ins,
    SUM(outs) AS outs,
    SUM(net)  AS net
FROM counting_by_bucket
GROUP BY store_id, date_trunc('hour', event_bucket);

CREATE OR REPLACE VIEW turn_in_rate_hourly AS
SELECT
    store_id,
    date_trunc('hour', bucket) AS hour,
    SUM(ins)       AS ins,
    SUM(passersby) AS passersby,
    SUM(shoppers)  AS shoppers,
    CASE WHEN SUM(passersby) > 0
         THEN SUM(ins)::float / SUM(passersby)
         ELSE NULL END  AS turn_in_rate,
    CASE WHEN SUM(shoppers) > 0
         THEN SUM(ins)::float / SUM(shoppers)
         ELSE NULL END  AS conversion_rate
FROM turn_in_rate_by_bucket
GROUP BY store_id, date_trunc('hour', bucket);

-- =============================================================================
-- View agregada: count events + sales por hora por store
-- Útil para dashboards de conversion rate = sales_count / people_in
-- =============================================================================

CREATE OR REPLACE VIEW store_hourly_summary AS
SELECT
    DATE_TRUNC('hour', c.event_ts)              AS hour_bucket,
    c.store_id,
    COUNT(*) FILTER (WHERE c.direction = 'in')  AS people_in,
    COUNT(*) FILTER (WHERE c.direction = 'out') AS people_out,
    COALESCE(s.sales_total, 0)                  AS sales_total,
    COALESCE(s.sales_count, 0)                  AS sales_count
FROM count_events c
LEFT JOIN LATERAL (
    SELECT
        SUM(amount_minor) / 100.0  AS sales_total,
        COUNT(*)                    AS sales_count
    FROM sales sl
    WHERE sl.store_id = c.store_id
      AND DATE_TRUNC('hour', sl.sale_ts) = DATE_TRUNC('hour', c.event_ts)
) s ON true
GROUP BY DATE_TRUNC('hour', c.event_ts), c.store_id, s.sales_total, s.sales_count;

-- =============================================================================
-- User Lambda persist_event — IAM auth (sin password)
-- =============================================================================
-- La Lambda usa boto3 para generar un token IAM (rds.generate_db_auth_token) y
-- conecta como este user. El grant "rds_iam" es lo que habilita la auth IAM.
--
-- Idempotente: drop si existe (para correr este script varias veces sin error).

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lambda_writer') THEN
        -- Revocar todo y dropear, así re-creamos limpio
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM lambda_writer;
        REVOKE ALL ON SCHEMA public FROM lambda_writer;
        DROP USER lambda_writer;
    END IF;
END $$;

CREATE USER lambda_writer;
GRANT rds_iam TO lambda_writer;
GRANT USAGE ON SCHEMA public TO lambda_writer;
GRANT INSERT ON count_events, wifi_ble_summary, telemetry TO lambda_writer;
-- ON CONFLICT necesita SELECT también para evaluar la condición unique:
GRANT SELECT ON count_events, wifi_ble_summary, telemetry TO lambda_writer;

-- =============================================================================
-- Database "grafana" — separada para el state interno de Grafana
-- =============================================================================
-- Grafana guarda su config (users, dashboards, sessions, datasources) en
-- Postgres. Le damos una DB aislada para no mezclar con la data de events.
-- Owner = people_counter (master) para que pueda crear/modificar tablas
-- propias de Grafana en su primer arranque.
--
-- ECS Fargate (Grafana) se conecta como master user (people_counter) a la DB "grafana"
-- usando el password del Secrets Manager. Como datasource para queriar
-- events, Grafana conecta de vuelta a "people_counter" como master.
--
-- Idempotente: usa \gexec (psql meta-command) que solo ejecuta CREATE DATABASE
-- si el SELECT devuelve rows (= si la DB no existe). Funciona en psql; en
-- DBeaver no, pero el deploy script corre esto via docker postgres:16/psql.

SELECT 'CREATE DATABASE grafana OWNER people_counter'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'grafana')\gexec

-- =============================================================================
-- Verificación
-- =============================================================================

-- Listar tablas creadas
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' ORDER BY table_name;

-- Listar users
SELECT rolname, rolcanlogin, rolconnlimit
FROM pg_roles
WHERE rolname IN ('people_counter', 'lambda_writer')
ORDER BY rolname;

-- Listar databases
SELECT datname FROM pg_database
WHERE datname IN ('people_counter', 'grafana')
ORDER BY datname;

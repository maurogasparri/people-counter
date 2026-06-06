-- =============================================================================
-- people-counter — bootstrap del schema en RDS Postgres
-- =============================================================================
-- Cómo ejecutar (desde DBeaver, conectado como master user "people_counter"):
--   1. Abrir este archivo
--   2. Right-click → Execute → Execute SQL Script (Alt+X)
--   3. Revisar que todas las sentencias hayan corrido OK
--
-- Idempotente: usa IF NOT EXISTS / OR REPLACE donde aplica.
--
-- Este archivo es el SCHEMA CANÓNICO consolidado: incorpora el end-state de
-- todas las migraciones de `migrations/` hasta 2026-05-28 inclusive (ya
-- aplicadas a la DB del piloto y squasheadas acá). Un deploy fresco desde este
-- bootstrap produce el mismo estado que correr todas esas migraciones en orden.
-- `migrations/` solo conserva las migraciones PENDIENTES de aplicar a la DB
-- viva (hoy ninguna; ver migrations/README.md).
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- Reset (dev/PoC only)
-- =============================================================================
-- En produccion esto se reemplaza por migrations versionadas (flyway, alembic).
-- Para PoC, dropeamos y recreamos. Cascade tira las views que dependen de las
-- tablas; se recrean abajo. IF EXISTS hace este bloque idempotente en cold start.

DROP TABLE IF EXISTS count_events       CASCADE;
DROP TABLE IF EXISTS wifi_ble_events    CASCADE;
DROP TABLE IF EXISTS wifi_ble_summary   CASCADE;  -- legacy, post PR 2 reemplazada por wifi_ble_events
DROP TABLE IF EXISTS telemetry          CASCADE;
DROP TABLE IF EXISTS pos_transactions   CASCADE;
-- Capa de rollup (derivada, reconstruible via refresh_rollups) — se dropea en reset.
DROP TABLE IF EXISTS
    rollup_counting_15min, rollup_counting_hour, rollup_counting_day,
    rollup_wifi_ble_15min, rollup_wifi_ble_hour, rollup_wifi_ble_day,
    rollup_wifi_engagement_day, rollup_pos_15min, rollup_pos_hour, rollup_pos_day,
    rollup_state CASCADE;

-- =============================================================================
-- Tablas raw — escritas por Lambda persist_event (3 topics IoT distintos)
-- =============================================================================

CREATE TABLE IF NOT EXISTS count_events (
    event_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id       TEXT         NOT NULL,
    store_id        TEXT         NOT NULL,
    event_ts        TIMESTAMPTZ  NOT NULL,
    -- Bucket de 15min alineado al epoch — derivado server-side desde event_ts.
    -- El device manda event_ts crudo, no calcula el bucket. Cambiar el tamaño
    -- del bucket (ej. migrar a 5min) = ALTER COLUMN sin tocar device/MQTT/Lambda.
    bucket_15min    TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 900) * 900)) STORED,
    -- Rollups server-side derivados del event_ts. STORED para que sean indexables
    -- y los queries de Grafana hourly/daily no recomputen date_trunc en cada fila.
    bucket_hour     TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 3600) * 3600)) STORED,
    bucket_day      DATE         GENERATED ALWAYS AS ((TIMESTAMP 'epoch' + floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 86400) * INTERVAL '1 day')::date) STORED,
    direction       TEXT         NOT NULL CHECK (direction IN ('in', 'out')),
    track_id        INT,
    -- Score del detector (0-1). No se usa en views/dashboards regulares, se
    -- mantiene para debug ad-hoc: investigar falsos positivos en stores con
    -- conteos sospechosos, detectar drift entre devices.
    confidence      REAL,
    -- Altura cruda en metros (medición geométrica del par estéreo). La
    -- categorización adulto/niño/desconocido se aplica server-side via la
    -- función SQL height_class(height_m) — single source of truth del threshold.
    height_m        REAL,
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia para reintentos del Lambda / replay del buffer del device.
    UNIQUE (device_id, event_ts, track_id, direction)
);

CREATE INDEX IF NOT EXISTS idx_count_events_store_ts
    ON count_events (store_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_count_events_device_ts
    ON count_events (device_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_count_events_bucket15
    ON count_events (store_id, bucket_15min DESC);
CREATE INDEX IF NOT EXISTS idx_count_events_store_bucket_hour
    ON count_events (store_id, bucket_hour DESC);
CREATE INDEX IF NOT EXISTS idx_count_events_day
    ON count_events (store_id, bucket_day DESC);

-- Per-device WiFi/BLE events (post PR 2). Un evento por cada visitor
-- (group_id post-stitching local del device) por ventana de emisión.
-- La categorización shopper/passerby se aplica server-side via la función
-- SQL rssi_class(rssi_max) — el device solo persiste el RSSI crudo.
CREATE TABLE IF NOT EXISTS wifi_ble_events (
    event_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id       TEXT         NOT NULL,
    store_id        TEXT         NOT NULL,
    -- Identidad post-stitching local del device (group_id del DedupEngine).
    -- 16 bytes: un UUID random (uuid.uuid4().hex), NO derivado de la MAC ni
    -- del hash → opaco e inlinkeable. (La MAC se hashea con SHA-256+salt local
    -- aparte, pero ese hash nunca sale del device.) Se renueva cada día
    -- (people-counter-reset rota el salt y resetea los grupos).
    visitor_hash    BYTEA        NOT NULL,
    protocol        TEXT         NOT NULL CHECK (protocol IN ('wifi', 'ble')),
    -- RSSI máximo observado durante la ventana, en dBm (entero negativo
    -- típicamente entre -100 y -20). NOT NULL — un device sin lectura no se emite.
    rssi_max        INT          NOT NULL,
    first_seen_ts   TIMESTAMPTZ  NOT NULL,
    last_seen_ts    TIMESTAMPTZ  NOT NULL,
    period_start    TIMESTAMPTZ  NOT NULL,
    period_end      TIMESTAMPTZ  NOT NULL,
    -- Buckets derivados de last_seen_ts. Si el visitor aparece en 2 ventanas
    -- de 15min, se inserta 2 filas (una por ventana) pero ambas con el mismo
    -- visitor_hash → COUNT(DISTINCT visitor_hash) por día sigue siendo 1.
    bucket_15min    TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (last_seen_ts - TIMESTAMPTZ 'epoch')) / 900) * 900)) STORED,
    bucket_hour     TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (last_seen_ts - TIMESTAMPTZ 'epoch')) / 3600) * 3600)) STORED,
    bucket_day      DATE         GENERATED ALWAYS AS ((TIMESTAMP 'epoch' + floor(extract(epoch FROM (last_seen_ts - TIMESTAMPTZ 'epoch')) / 86400) * INTERVAL '1 day')::date) STORED,
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia: la Lambda hace ON CONFLICT...DO UPDATE refinando
    -- rssi_max al máximo y last_seen_ts al más reciente.
    UNIQUE (device_id, visitor_hash, period_start)
);

CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_bucket15
    ON wifi_ble_events (store_id, bucket_15min DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_bucket_hour
    ON wifi_ble_events (store_id, bucket_hour DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_day
    ON wifi_ble_events (store_id, bucket_day DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_day_visitor
    ON wifi_ble_events (store_id, bucket_day, visitor_hash);

CREATE TABLE IF NOT EXISTS telemetry (
    telemetry_id     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id        TEXT         NOT NULL,
    store_id         TEXT         NOT NULL,
    event_ts         TIMESTAMPTZ  NOT NULL,
    -- Rollup hourly server-side. Telemetry corre cada 5min, asi que 15min/day
    -- no aplican naturalmente; hourly es la granularidad util para dashboards
    -- de fleet ("CPU temp p95 por hora", "fps mediana por hora").
    bucket_hour      TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 3600) * 3600)) STORED,
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
    -- Stitching ratio del tracker de visión: unique_track_ids vistos cruzar
    -- la counting zone / total counts emitidos. Ideal ≈ 1.0 (1 persona = 1 ID = 1
    -- evento); >1.3 indica fragmentación de identidad.
    track_stitching_ratio         REAL,
    -- Cantidad de death-emits disparados hoy (el fallback que cuenta tracks
    -- que cruzaron pero murieron dentro de la counting zone). Combinado con el ratio
    -- diferencia 'fragmenta-y-rescata' (count alto) de 'fragmenta-y-pierde'
    -- (count bajo con ratio alto → recall del detector flojo).
    death_emit_count              INT,
    -- Cantidad de ghost adoptions del tracker desde el boot del proceso
    -- (capa 1 del rescue). Cierra el árbol diagnóstico: combinado con
    -- death_emit_count y track_stitching_ratio permite distinguir
    -- "tracker perfecto" / "fragmentación rescatada por adopción" /
    -- "fragmentación rescatada por death-emit" / "fragmentación sin rescate".
    ghost_adoption_count          INT,
    -- Canary del Device Shadow: timestamp del último delta aplicado vía
    -- apply_shadow_delta (cloud_defaults.{operating_hours, counting_enabled,
    -- external_traffic_enabled} pushados desde AWS). NULL hasta que el device
    -- reciba el primer push real (boot reconcile no cuenta). Útil en Grafana
    -- para distinguir "shadow nunca usado" vs "última push fue hace mucho".
    last_shadow_apply_ts          TIMESTAMPTZ,
    -- Schedule / error state (mando 'error' del payload + detalle largo)
    error                         TEXT,
    schedule_error_detail         TEXT,
    received_at                   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia ante retries de IoT (mismo device + mismo timestamp = misma muestra).
    UNIQUE (device_id, event_ts)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_device_ts
    ON telemetry (device_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_device_hour
    ON telemetry (device_id, bucket_hour DESC);

-- =============================================================================
-- POS transactions — ingesta via API Gateway desde el POS del cliente (T9.11)
-- =============================================================================
-- Soporta dos modos de envio a discrecion del POS:
--   1) Transaction-by-transaction: una fila por factura/ticket, type='sale'
--      o 'return'. transaction_id = numero de factura.
--   2) Aggregated: una fila por batch (turno, hora, etc.), type='sale' o
--      'return' segun corresponda al batch. items = total de items del batch,
--      amount_minor = total del batch. transaction_id = batch_id del POS.
-- En ambos casos el conversion_rate funciona — solo cambia la granularidad
-- maxima a la que se puede agrupar.

CREATE TABLE IF NOT EXISTS pos_transactions (
    pos_id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id  TEXT         NOT NULL,                  -- factura o batch id del POS
    store_id        TEXT         NOT NULL,
    event_ts        TIMESTAMPTZ  NOT NULL,                  -- timestamp de la transaccion/batch
    type            TEXT         NOT NULL CHECK (type IN ('sale', 'return')),
    items           INT          NOT NULL DEFAULT 1 CHECK (items >= 0),
    amount_minor    BIGINT       NOT NULL CHECK (amount_minor >= 0),  -- en centavos (no floats)
    currency        CHAR(3)      NOT NULL DEFAULT 'ARS',
    payment_method  TEXT,                                   -- nullable: NULL o 'mixed' para batches
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Buckets generados server-side desde event_ts (el POS no conoce nuestro shadow).
    -- Mismos nombres que count_events/wifi_ble_summary -> JOINs por nombre de columna.
    bucket_15min    TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 900) * 900)) STORED,
    bucket_hour     TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 3600) * 3600)) STORED,
    bucket_day      DATE         GENERATED ALWAYS AS ((TIMESTAMP 'epoch' + floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 86400) * INTERVAL '1 day')::date) STORED,
    -- Idempotencia: POS reintenta mismo transaction_id sin duplicar.
    UNIQUE (store_id, transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_pos_store_bucket15
    ON pos_transactions (store_id, bucket_15min DESC);
CREATE INDEX IF NOT EXISTS idx_pos_store_bucket_hour
    ON pos_transactions (store_id, bucket_hour DESC);
CREATE INDEX IF NOT EXISTS idx_pos_store_day
    ON pos_transactions (store_id, bucket_day DESC);

-- =============================================================================
-- Tablas de dimensiones — sites + devices (sembradas en provisioning)
-- =============================================================================
-- A DIFERENCIA de las tablas raw de arriba, estas NO se dropean en el bloque de
-- reset: son datos de provisioning (no telemetría replayable), así que re-correr
-- este script las preserva. Para agregarlas a una DB ya deployada con datos en
-- vivo, correr SOLO este bloque (es idempotente, sin DROP) — NO el script
-- entero, que borraría las tablas raw.
--
-- Sirven para: (a) geomap de Grafana (lat/long por sucursal), (b) template
-- variables / dropdowns de filtro que salen de una tabla chica en vez de un
-- SELECT DISTINCT sobre count_events (scan caro a volumen, y que encima no
-- muestra devices recién provisionados sin datos), (c) labels human-readable
-- ("Sucursal Centro" en vez de store-001-cam-01) vía JOIN.
--
-- Sin FK desde las tablas de hechos (count_events, etc.) hacia estas: las facts
-- las escribe la Lambda y NO debe fallar si un site todavía no se registró.
-- Grafana hace LEFT JOIN. La única FK es devices -> sites, que el seed respeta
-- (UPSERT del site antes que el device).

CREATE TABLE IF NOT EXISTS sites (
    store_id    TEXT             PRIMARY KEY,             -- matchea count_events.store_id, etc.
    store_name  TEXT             NOT NULL,                -- human-readable para dashboards
    -- DOUBLE PRECISION (no REAL): las coords tienen ~7 decimales y REAL solo
    -- preserva ~7 dígitos significativos → perdería precisión en el mapa.
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    timezone    TEXT NOT NULL DEFAULT 'UTC',              -- IANA, ej. 'America/Argentina/Buenos_Aires'. Base del bucketing local (multi-país).
    sales_area_m2 NUMERIC,                                 -- superficie de venta (m²) → métrica ventas/m²
    -- Lifecycle de la tienda (NO afecta ingesta — Lambdas/MQTT/RDS guardan igual;
    -- es solo un filtro de visualización). Grafana muestra SOLO 'operational'.
    -- 'temp_closed' (reforma: contador desenchufado) y 'perm_closed' (cerrada)
    -- quedan fuera de todos los tableros. Default operational.
    status      TEXT NOT NULL DEFAULT 'operational'
                CHECK (status IN ('operational', 'temp_closed', 'perm_closed')),
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS devices (
    device_id        TEXT        PRIMARY KEY,             -- matchea count_events.device_id, etc.
    store_id         TEXT        NOT NULL REFERENCES sites(store_id) ON UPDATE CASCADE,
    cam_label        TEXT,                                -- ej. 'puerta principal'
    firmware_version TEXT,
    installed_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_devices_store ON devices (store_id);

-- Feriados (referencia, AR). Grafana los pinta como annotations sobre las
-- series temporales (líneas verticales) para contextualizar picos/valles.
-- 'nacional' = feriado; 'puente' = día no laborable con fines turísticos.
CREATE TABLE IF NOT EXISTS holidays (
    holiday_date DATE PRIMARY KEY,
    name         TEXT NOT NULL,
    type         TEXT NOT NULL DEFAULT 'nacional' CHECK (type IN ('nacional', 'puente'))
);
INSERT INTO holidays (holiday_date, name, type) VALUES
    ('2026-01-01', 'Año Nuevo', 'nacional'),
    ('2026-02-16', 'Carnaval', 'nacional'),
    ('2026-02-17', 'Carnaval', 'nacional'),
    ('2026-03-23', 'Puente turístico', 'puente'),
    ('2026-03-24', 'Día de la Memoria por la Verdad y la Justicia', 'nacional'),
    ('2026-04-02', 'Día del Veterano y de los Caídos en Malvinas', 'nacional'),
    ('2026-04-03', 'Viernes Santo', 'nacional'),
    ('2026-05-01', 'Día del Trabajador', 'nacional'),
    ('2026-05-25', 'Día de la Revolución de Mayo', 'nacional'),
    ('2026-06-15', 'Paso a la Inmortalidad del Gral. Güemes', 'nacional'),
    ('2026-06-20', 'Paso a la Inmortalidad del Gral. Belgrano', 'nacional'),
    ('2026-07-09', 'Día de la Independencia', 'nacional'),
    ('2026-07-10', 'Puente turístico', 'puente'),
    ('2026-08-17', 'Paso a la Inmortalidad del Gral. San Martín', 'nacional'),
    ('2026-10-12', 'Día del Respeto a la Diversidad Cultural', 'nacional'),
    ('2026-11-23', 'Día de la Soberanía Nacional', 'nacional'),
    ('2026-12-07', 'Puente turístico', 'puente'),
    ('2026-12-08', 'Inmaculada Concepción de María', 'nacional'),
    ('2026-12-25', 'Navidad', 'nacional')
    ON CONFLICT (holiday_date) DO UPDATE SET name = EXCLUDED.name, type = EXCLUDED.type;

-- Grafana consulta como master user (people_counter), que es owner → ya puede
-- leer estas tablas. El seed de provisioning corre como master (mismo path que
-- este bootstrap). lambda_writer escribe hechos, no necesita las dimensiones.

-- =============================================================================
-- Función: categorización por altura (single source of truth del threshold)
-- =============================================================================
-- IMMUTABLE le permite a Postgres inlinearla en plans de query, sin overhead
-- por llamada. NULL → 'unknown' (cuando el detector no pudo medir profundidad
-- confiable). El threshold 1.55 m coincide con el default histórico del
-- config del device (counter.height_classifier.adult_min_m). Para modificar
-- globalmente: CREATE OR REPLACE — se aplica retroactivo a vistas e historia.

CREATE OR REPLACE FUNCTION height_class(h REAL) RETURNS TEXT
LANGUAGE SQL IMMUTABLE AS $$
    SELECT CASE
        WHEN h IS NULL THEN 'unknown'
        WHEN h < 1.55  THEN 'child'
        ELSE 'adult'
    END
$$;

COMMENT ON FUNCTION height_class(REAL) IS
    'Clasifica una altura en metros como adulto / niño / desconocido. Threshold: 1.55 m. Para modificar globalmente: CREATE OR REPLACE FUNCTION; se aplica retroactivo a las vistas y a toda la historia.';

-- Función rssi_class — categorización shopper/passerby/weak/unknown.
-- Mismo patrón que height_class: el device emite el RSSI crudo, la
-- función SQL lo categoriza server-side. shopper ⊆ passerby por
-- convención (las vistas filtran explícito para preservar el invariante).
CREATE OR REPLACE FUNCTION rssi_class(rssi INT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE AS $$
    SELECT CASE
        WHEN rssi IS NULL  THEN 'unknown'
        WHEN rssi >= -55   THEN 'shopper'
        WHEN rssi >= -75   THEN 'passerby'
        ELSE                    'weak'
    END
$$;

COMMENT ON FUNCTION rssi_class(INT) IS
    'Clasifica un RSSI máximo en dBm como shopper / passerby / weak / unknown. Thresholds: shopper >= -55 dBm, passerby >= -75 dBm. Para modificarlos globalmente: CREATE OR REPLACE FUNCTION; se aplica retroactivo a las vistas y al histórico.';

-- =============================================================================
-- Capa de rollup incremental (speed-layer + batch-layer)
-- =============================================================================
-- Las vistas base de abajo NO agregan el raw directo: leen rollup_* (buckets
-- cerrados, pre-computados) UNION ALL raw (sólo el bucket abierto = live tail
-- ≤5s). refresh_rollups() recomputa por watermark (received_at) sólo los buckets
-- con datos nuevos → O(reciente), escala con la flota, sin tocar el hot-path de
-- ingesta. Tras un deploy fresco: CALL refresh_rollups() (backfill) + agendar el
-- refresh (pg_cron o EventBridge→Lambda). Detalle: docs/metrics_roadmap.md.

-- Bucket LOCAL de un instante dado el tz IANA de la tienda (sites.timezone).
-- Devuelven el bucket como wall-clock-local-representado-como-UTC (hour/15min) o
-- date local (day) → multi-país: cada tienda bucketea en su hora local. STABLE
-- (AT TIME ZONE). EXECUTE es PUBLIC por default.
CREATE OR REPLACE FUNCTION lday(ts timestamptz, tz text) RETURNS date LANGUAGE sql STABLE AS $$
    SELECT (ts AT TIME ZONE tz)::date
$$;
CREATE OR REPLACE FUNCTION lhour(ts timestamptz, tz text) RETURNS timestamptz LANGUAGE sql STABLE AS $$
    SELECT to_timestamp(floor(extract(epoch FROM (ts AT TIME ZONE tz)) / 3600) * 3600)
$$;
CREATE OR REPLACE FUNCTION l15(ts timestamptz, tz text) RETURNS timestamptz LANGUAGE sql STABLE AS $$
    SELECT to_timestamp(floor(extract(epoch FROM (ts AT TIME ZONE tz)) / 900) * 900)
$$;

-- Estado del watermark (1 fila por fuente). '-infinity' → backfill total al 1er CALL.
CREATE TABLE IF NOT EXISTS rollup_state (
    source            TEXT PRIMARY KEY,
    last_refreshed_at TIMESTAMPTZ NOT NULL DEFAULT '-infinity'
);
INSERT INTO rollup_state (source) VALUES ('counting'), ('wifi_ble'), ('pos')
    ON CONFLICT (source) DO NOTHING;

-- Tablas de rollup. Tipos = los de los agregados (bigint/numeric) para que el
-- UNION ALL con el live-tail no choque tipos.
CREATE TABLE IF NOT EXISTS rollup_counting_15min (
    store_id TEXT NOT NULL, bucket_15min TIMESTAMPTZ NOT NULL,
    ins BIGINT NOT NULL, outs BIGINT NOT NULL, net BIGINT NOT NULL,
    ins_adult BIGINT, ins_child BIGINT, ins_unknown BIGINT,
    outs_adult BIGINT, outs_child BIGINT, outs_unknown BIGINT,
    PRIMARY KEY (store_id, bucket_15min)
);
CREATE TABLE IF NOT EXISTS rollup_counting_hour (
    store_id TEXT NOT NULL, bucket_hour TIMESTAMPTZ NOT NULL,
    ins BIGINT NOT NULL, outs BIGINT NOT NULL, net BIGINT NOT NULL,
    ins_adult BIGINT, ins_child BIGINT, ins_unknown BIGINT,
    outs_adult BIGINT, outs_child BIGINT, outs_unknown BIGINT,
    PRIMARY KEY (store_id, bucket_hour)
);
CREATE TABLE IF NOT EXISTS rollup_counting_day (
    store_id TEXT NOT NULL, bucket_day DATE NOT NULL,
    ins BIGINT NOT NULL, outs BIGINT NOT NULL, net BIGINT NOT NULL,
    ins_adult BIGINT, ins_child BIGINT, ins_unknown BIGINT,
    outs_adult BIGINT, outs_child BIGINT, outs_unknown BIGINT,
    PRIMARY KEY (store_id, bucket_day)
);
CREATE TABLE IF NOT EXISTS rollup_wifi_ble_15min (
    store_id TEXT NOT NULL, bucket_15min TIMESTAMPTZ NOT NULL,
    passersby BIGINT NOT NULL, shoppers BIGINT NOT NULL, visitors BIGINT NOT NULL,
    PRIMARY KEY (store_id, bucket_15min)
);
CREATE TABLE IF NOT EXISTS rollup_wifi_ble_hour (
    store_id TEXT NOT NULL, bucket_hour TIMESTAMPTZ NOT NULL,
    passersby BIGINT NOT NULL, shoppers BIGINT NOT NULL, visitors BIGINT NOT NULL,
    PRIMARY KEY (store_id, bucket_hour)
);
CREATE TABLE IF NOT EXISTS rollup_wifi_ble_day (
    store_id TEXT NOT NULL, bucket_day DATE NOT NULL,
    passersby BIGINT NOT NULL, shoppers BIGINT NOT NULL, visitors BIGINT NOT NULL,
    PRIMARY KEY (store_id, bucket_day)
);
CREATE TABLE IF NOT EXISTS rollup_wifi_engagement_day (
    store_id TEXT NOT NULL, bucket_day DATE NOT NULL,
    windows_bucket TEXT NOT NULL,   -- '1' | '2' | '3-5' | '6+'
    visitors BIGINT NOT NULL,
    PRIMARY KEY (store_id, bucket_day, windows_bucket)
);
CREATE TABLE IF NOT EXISTS rollup_pos_15min (
    store_id TEXT NOT NULL, bucket_15min TIMESTAMPTZ NOT NULL,
    sales BIGINT, returns BIGINT, transactions BIGINT,
    items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, net_amount_minor NUMERIC,
    currency CHAR(3), PRIMARY KEY (store_id, bucket_15min)
);
CREATE TABLE IF NOT EXISTS rollup_pos_hour (
    store_id TEXT NOT NULL, bucket_hour TIMESTAMPTZ NOT NULL,
    sales BIGINT, returns BIGINT, transactions BIGINT,
    items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, net_amount_minor NUMERIC,
    currency CHAR(3), PRIMARY KEY (store_id, bucket_hour)
);
CREATE TABLE IF NOT EXISTS rollup_pos_day (
    store_id TEXT NOT NULL, bucket_day DATE NOT NULL,
    sales BIGINT, returns BIGINT, transactions BIGINT,
    items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, net_amount_minor NUMERIC,
    currency CHAR(3), PRIMARY KEY (store_id, bucket_day)
);

-- received_at es append-ordenado → BRIN (chico) hace barato el "qué llegó desde
-- el watermark" del refresh.
CREATE INDEX IF NOT EXISTS idx_count_events_received_brin    ON count_events     USING brin (received_at);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_received_brin ON wifi_ble_events  USING brin (received_at);
CREATE INDEX IF NOT EXISTS idx_pos_received_brin             ON pos_transactions USING brin (received_at);
-- Para el bucketing local (join sites) + live-tail sobre el instante real.
CREATE INDEX IF NOT EXISTS idx_count_events_store_event    ON count_events    (store_id, event_ts);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_seen  ON wifi_ble_events (store_id, last_seen_ts);
CREATE INDEX IF NOT EXISTS idx_pos_store_event             ON pos_transactions (store_id, event_ts);
-- max(received_at) por sucursal → data_freshness_by_store (lateral por store): cada
-- subquery resuelve con un Index Only Scan Backward O(log n) en vez de seq scan total.
CREATE INDEX IF NOT EXISTS idx_count_events_store_received    ON count_events    (store_id, received_at);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_received ON wifi_ble_events (store_id, received_at);
CREATE INDEX IF NOT EXISTS idx_pos_store_received            ON pos_transactions (store_id, received_at);

-- refresh_rollups(): recomputa por watermark (received_at). El bucket de cada
-- evento se calcula en la zona horaria de la tienda (sites.timezone) vía
-- lday/lhour/l15 → multi-país. Prefiltro por event_ts/last_seen_ts (rango de lo
-- recibido ±1 día) para usar índice; el IN (store, bucket-local) acota a los
-- buckets afectados. Sin control de transacción interno (corre en la del caller).
CREATE OR REPLACE PROCEDURE refresh_rollups() LANGUAGE plpgsql AS $$
DECLARE wm TIMESTAMPTZ; nwm TIMESTAMPTZ := now() - INTERVAL '1 minute';
BEGIN
    -- ===== counting (bucket por event_ts) =====
    SELECT last_refreshed_at INTO wm FROM rollup_state WHERE source='counting';
    INSERT INTO rollup_counting_day (store_id,bucket_day,ins,outs,net,ins_adult,ins_child,ins_unknown,outs_adult,outs_child,outs_unknown)
    SELECT e.store_id, lday(e.event_ts,s.timezone),
        COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.event_ts BETWEEN (SELECT min(event_ts) FROM count_events WHERE received_at>wm) - INTERVAL '1 day'
                         AND (SELECT max(event_ts) FROM count_events WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.event_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.event_ts,s2.timezone) FROM count_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, lday(e.event_ts,s.timezone)
    ON CONFLICT (store_id,bucket_day) DO UPDATE SET ins=EXCLUDED.ins,outs=EXCLUDED.outs,net=EXCLUDED.net,
        ins_adult=EXCLUDED.ins_adult,ins_child=EXCLUDED.ins_child,ins_unknown=EXCLUDED.ins_unknown,
        outs_adult=EXCLUDED.outs_adult,outs_child=EXCLUDED.outs_child,outs_unknown=EXCLUDED.outs_unknown;

    INSERT INTO rollup_counting_hour (store_id,bucket_hour,ins,outs,net,ins_adult,ins_child,ins_unknown,outs_adult,outs_child,outs_unknown)
    SELECT e.store_id, lhour(e.event_ts,s.timezone),
        COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.event_ts BETWEEN (SELECT min(event_ts) FROM count_events WHERE received_at>wm) - INTERVAL '1 day'
                         AND (SELECT max(event_ts) FROM count_events WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.event_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.event_ts,s2.timezone) FROM count_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, lhour(e.event_ts,s.timezone)
    ON CONFLICT (store_id,bucket_hour) DO UPDATE SET ins=EXCLUDED.ins,outs=EXCLUDED.outs,net=EXCLUDED.net,
        ins_adult=EXCLUDED.ins_adult,ins_child=EXCLUDED.ins_child,ins_unknown=EXCLUDED.ins_unknown,
        outs_adult=EXCLUDED.outs_adult,outs_child=EXCLUDED.outs_child,outs_unknown=EXCLUDED.outs_unknown;

    INSERT INTO rollup_counting_15min (store_id,bucket_15min,ins,outs,net,ins_adult,ins_child,ins_unknown,outs_adult,outs_child,outs_unknown)
    SELECT e.store_id, l15(e.event_ts,s.timezone),
        COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.event_ts BETWEEN (SELECT min(event_ts) FROM count_events WHERE received_at>wm) - INTERVAL '1 day'
                         AND (SELECT max(event_ts) FROM count_events WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.event_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.event_ts,s2.timezone) FROM count_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, l15(e.event_ts,s.timezone)
    ON CONFLICT (store_id,bucket_15min) DO UPDATE SET ins=EXCLUDED.ins,outs=EXCLUDED.outs,net=EXCLUDED.net,
        ins_adult=EXCLUDED.ins_adult,ins_child=EXCLUDED.ins_child,ins_unknown=EXCLUDED.ins_unknown,
        outs_adult=EXCLUDED.outs_adult,outs_child=EXCLUDED.outs_child,outs_unknown=EXCLUDED.outs_unknown;
    UPDATE rollup_state SET last_refreshed_at=nwm WHERE source='counting';

    -- ===== wifi_ble (bucket por last_seen_ts) =====
    SELECT last_refreshed_at INTO wm FROM rollup_state WHERE source='wifi_ble';
    INSERT INTO rollup_wifi_ble_day (store_id,bucket_day,passersby,shoppers,visitors)
    SELECT e.store_id, lday(e.last_seen_ts,s.timezone),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max)='shopper'),
        COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.last_seen_ts BETWEEN (SELECT min(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) - INTERVAL '1 day'
                             AND (SELECT max(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.last_seen_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.last_seen_ts,s2.timezone) FROM wifi_ble_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, lday(e.last_seen_ts,s.timezone)
    ON CONFLICT (store_id,bucket_day) DO UPDATE SET passersby=EXCLUDED.passersby,shoppers=EXCLUDED.shoppers,visitors=EXCLUDED.visitors;

    INSERT INTO rollup_wifi_ble_hour (store_id,bucket_hour,passersby,shoppers,visitors)
    SELECT e.store_id, lhour(e.last_seen_ts,s.timezone),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max)='shopper'),
        COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.last_seen_ts BETWEEN (SELECT min(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) - INTERVAL '1 day'
                             AND (SELECT max(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.last_seen_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.last_seen_ts,s2.timezone) FROM wifi_ble_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, lhour(e.last_seen_ts,s.timezone)
    ON CONFLICT (store_id,bucket_hour) DO UPDATE SET passersby=EXCLUDED.passersby,shoppers=EXCLUDED.shoppers,visitors=EXCLUDED.visitors;

    INSERT INTO rollup_wifi_ble_15min (store_id,bucket_15min,passersby,shoppers,visitors)
    SELECT e.store_id, l15(e.last_seen_ts,s.timezone),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max)='shopper'),
        COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.last_seen_ts BETWEEN (SELECT min(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) - INTERVAL '1 day'
                             AND (SELECT max(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.last_seen_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.last_seen_ts,s2.timezone) FROM wifi_ble_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, l15(e.last_seen_ts,s.timezone)
    ON CONFLICT (store_id,bucket_15min) DO UPDATE SET passersby=EXCLUDED.passersby,shoppers=EXCLUDED.shoppers,visitors=EXCLUDED.visitors;

    DELETE FROM rollup_wifi_engagement_day
     WHERE (store_id, bucket_day) IN
       (SELECT DISTINCT e.store_id, lday(e.last_seen_ts,s.timezone) FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id WHERE e.received_at>wm);
    INSERT INTO rollup_wifi_engagement_day (store_id,bucket_day,windows_bucket,visitors)
    SELECT store_id, bd, CASE WHEN nw=1 THEN '1' WHEN nw=2 THEN '2' WHEN nw BETWEEN 3 AND 5 THEN '3-5' ELSE '6+' END, COUNT(*)
    FROM (
        SELECT e.store_id, lday(e.last_seen_ts,s.timezone) bd, visitor_hash, COUNT(DISTINCT period_start) nw
        FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
        WHERE e.last_seen_ts BETWEEN (SELECT min(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) - INTERVAL '1 day'
                                 AND (SELECT max(last_seen_ts) FROM wifi_ble_events WHERE received_at>wm) + INTERVAL '1 day'
          AND (e.store_id, lday(e.last_seen_ts,s.timezone)) IN
              (SELECT e2.store_id, lday(e2.last_seen_ts,s2.timezone) FROM wifi_ble_events e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
        GROUP BY e.store_id, lday(e.last_seen_ts,s.timezone), visitor_hash
    ) v GROUP BY store_id, bd, 3;
    UPDATE rollup_state SET last_refreshed_at=nwm WHERE source='wifi_ble';

    -- ===== pos (bucket por event_ts) =====
    SELECT last_refreshed_at INTO wm FROM rollup_state WHERE source='pos';
    INSERT INTO rollup_pos_day (store_id,bucket_day,sales,returns,transactions,items_sale,items_return,amount_minor_sale,amount_minor_return,net_amount_minor,currency)
    SELECT e.store_id, lday(e.event_ts,s.timezone),
        COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
        COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0)-COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0), MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
    WHERE e.event_ts BETWEEN (SELECT min(event_ts) FROM pos_transactions WHERE received_at>wm) - INTERVAL '1 day'
                         AND (SELECT max(event_ts) FROM pos_transactions WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.event_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.event_ts,s2.timezone) FROM pos_transactions e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, lday(e.event_ts,s.timezone)
    ON CONFLICT (store_id,bucket_day) DO UPDATE SET sales=EXCLUDED.sales,returns=EXCLUDED.returns,transactions=EXCLUDED.transactions,
        items_sale=EXCLUDED.items_sale,items_return=EXCLUDED.items_return,amount_minor_sale=EXCLUDED.amount_minor_sale,
        amount_minor_return=EXCLUDED.amount_minor_return,net_amount_minor=EXCLUDED.net_amount_minor,currency=EXCLUDED.currency;

    INSERT INTO rollup_pos_hour (store_id,bucket_hour,sales,returns,transactions,items_sale,items_return,amount_minor_sale,amount_minor_return,net_amount_minor,currency)
    SELECT e.store_id, lhour(e.event_ts,s.timezone),
        COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
        COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0)-COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0), MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
    WHERE e.event_ts BETWEEN (SELECT min(event_ts) FROM pos_transactions WHERE received_at>wm) - INTERVAL '1 day'
                         AND (SELECT max(event_ts) FROM pos_transactions WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.event_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.event_ts,s2.timezone) FROM pos_transactions e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, lhour(e.event_ts,s.timezone)
    ON CONFLICT (store_id,bucket_hour) DO UPDATE SET sales=EXCLUDED.sales,returns=EXCLUDED.returns,transactions=EXCLUDED.transactions,
        items_sale=EXCLUDED.items_sale,items_return=EXCLUDED.items_return,amount_minor_sale=EXCLUDED.amount_minor_sale,
        amount_minor_return=EXCLUDED.amount_minor_return,net_amount_minor=EXCLUDED.net_amount_minor,currency=EXCLUDED.currency;

    INSERT INTO rollup_pos_15min (store_id,bucket_15min,sales,returns,transactions,items_sale,items_return,amount_minor_sale,amount_minor_return,net_amount_minor,currency)
    SELECT e.store_id, l15(e.event_ts,s.timezone),
        COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
        COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0)-COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0), MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
    WHERE e.event_ts BETWEEN (SELECT min(event_ts) FROM pos_transactions WHERE received_at>wm) - INTERVAL '1 day'
                         AND (SELECT max(event_ts) FROM pos_transactions WHERE received_at>wm) + INTERVAL '1 day'
      AND (e.store_id, lday(e.event_ts,s.timezone)) IN
          (SELECT e2.store_id, lday(e2.event_ts,s2.timezone) FROM pos_transactions e2 JOIN sites s2 ON s2.store_id=e2.store_id WHERE e2.received_at>wm)
    GROUP BY e.store_id, l15(e.event_ts,s.timezone)
    ON CONFLICT (store_id,bucket_15min) DO UPDATE SET sales=EXCLUDED.sales,returns=EXCLUDED.returns,transactions=EXCLUDED.transactions,
        items_sale=EXCLUDED.items_sale,items_return=EXCLUDED.items_return,amount_minor_sale=EXCLUDED.amount_minor_sale,
        amount_minor_return=EXCLUDED.amount_minor_return,net_amount_minor=EXCLUDED.net_amount_minor,currency=EXCLUDED.currency;
    UPDATE rollup_state SET last_refreshed_at=nwm WHERE source='pos';
END $$;

-- =============================================================================
-- Capa de vistas: producto cartesiano fact × granularidad de bucket
-- =============================================================================
-- Convención de naming: <métrica>_by_bucket_<grano>. Tres granularidades
-- (15min, hour, day) sobre cada fact (counting, wifi_ble, pos, occupancy)
-- y sobre cada métrica derivada (turn_in_rate, conversion, visit_duration).
-- Además, tres vistas unificadas (metrics_unified_by_bucket_<grano>) que
-- combinan counting + wifi_ble + pos con FULL OUTER JOIN — consumidas
-- por la Lambda query_aggregates. Y data_freshness_by_store con el último
-- timestamp de ingesta cruzando las tres fuentes por sucursal.
--
-- Vistas idempotentes (CREATE OR REPLACE). Los COMMENT ON por columna/vista
-- están más abajo en este mismo archivo.

-- --- counting: ingresos y egresos con desglose demográfico ---
-- Categorización adulto/niño/desconocido aplicada via la función SQL
-- height_class(height_m) — single source of truth del threshold.
-- Cada vista base = rollup (buckets cerrados) UNION ALL raw (bucket abierto, live
-- tail ≤5s), bucketeado en la ZONA HORARIA de cada tienda (join sites + lday/lhour/l15)
-- → multi-país. El `<>`/`=` contra lday(now(), tz) aísla el bucket local abierto. Las
-- derivadas (turn_in, conversion, metrics_unified, occupancy_hour/day, visit_duration)
-- leen estas por nombre → quedan rápidas y tz-correctas sin cambios.
-- Bucket LOCAL por tienda: rollup r (cerrados, r.bucket <> bucket local actual)
-- UNION ALL raw (bucket abierto, live tail ≤5s), ambos vía join a sites.timezone.
-- El live-tail acota event_ts/last_seen_ts a [now()-2d, now()+1d] (ventana cerrada
-- ambos lados) para que el índice (store_id, event_ts) prune: sin el techo, datos
-- futuros (p.ej. demo seedeado adelante) hacen el rango no selectivo → seq scan de
-- la tabla raw. +1d cubre el día local abierto de cualquier zona horaria.
CREATE OR REPLACE VIEW counting_by_bucket_15min AS
  SELECT r.store_id, r.bucket_15min, r.ins, r.outs, r.net,
         r.ins_adult, r.ins_child, r.ins_unknown, r.outs_adult, r.outs_child, r.outs_unknown
    FROM rollup_counting_15min r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_15min AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, l15(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND e.event_ts < now() + INTERVAL '1 day' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, l15(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW counting_by_bucket_hour AS
  SELECT r.store_id, r.bucket_hour, r.ins, r.outs, r.net,
         r.ins_adult, r.ins_child, r.ins_unknown, r.outs_adult, r.outs_child, r.outs_unknown
    FROM rollup_counting_hour r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_hour AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lhour(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND e.event_ts < now() + INTERVAL '1 day' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lhour(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW counting_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.ins, r.outs, r.net,
         r.ins_adult, r.ins_child, r.ins_unknown, r.outs_adult, r.outs_child, r.outs_unknown
    FROM rollup_counting_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lday(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND e.event_ts < now() + INTERVAL '1 day' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lday(e.event_ts,s.timezone);

-- --- wifi_ble: tráfico exterior, DISTINCT visitor_hash + rssi_class (bucket por last_seen_ts) ---
CREATE OR REPLACE VIEW wifi_ble_by_bucket_15min AS
  SELECT r.store_id, r.bucket_15min, r.passersby, r.shoppers, r.visitors
    FROM rollup_wifi_ble_15min r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_15min AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, l15(e.last_seen_ts,s.timezone),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND e.last_seen_ts < now() + INTERVAL '1 day' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, l15(e.last_seen_ts,s.timezone);

CREATE OR REPLACE VIEW wifi_ble_by_bucket_hour AS
  SELECT r.store_id, r.bucket_hour, r.passersby, r.shoppers, r.visitors
    FROM rollup_wifi_ble_hour r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_hour AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lhour(e.last_seen_ts,s.timezone),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND e.last_seen_ts < now() + INTERVAL '1 day' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lhour(e.last_seen_ts,s.timezone);

CREATE OR REPLACE VIEW wifi_ble_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.passersby, r.shoppers, r.visitors
    FROM rollup_wifi_ble_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lday(e.last_seen_ts,s.timezone),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND e.last_seen_ts < now() + INTERVAL '1 day' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lday(e.last_seen_ts,s.timezone);

-- --- pos: transacciones agregadas (bucket por event_ts) ---
CREATE OR REPLACE VIEW pos_by_bucket_15min AS
  SELECT r.store_id, r.bucket_15min, r.sales, r.returns, r.transactions, r.items_sale, r.items_return,
         r.amount_minor_sale, r.amount_minor_return, r.net_amount_minor, r.currency
    FROM rollup_pos_15min r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_15min AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, l15(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND e.event_ts < now() + INTERVAL '1 day' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, l15(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW pos_by_bucket_hour AS
  SELECT r.store_id, r.bucket_hour, r.sales, r.returns, r.transactions, r.items_sale, r.items_return,
         r.amount_minor_sale, r.amount_minor_return, r.net_amount_minor, r.currency
    FROM rollup_pos_hour r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_hour AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lhour(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND e.event_ts < now() + INTERVAL '1 day' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lhour(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW pos_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.sales, r.returns, r.transactions, r.items_sale, r.items_return,
         r.amount_minor_sale, r.amount_minor_return, r.net_amount_minor, r.currency
    FROM rollup_pos_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lday(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND e.event_ts < now() + INTERVAL '1 day' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lday(e.event_ts,s.timezone);

-- --- occupancy: ocupación acumulada (cumsum por día) sobre la vista base ---
-- Lee counting_by_bucket_15min (rollup+live) → rápida e incluye el día abierto.
CREATE OR REPLACE VIEW occupancy_by_bucket_15min AS
SELECT
    store_id, bucket_15min, ins, outs,
    SUM(ins - outs) OVER (
        PARTITION BY store_id, date_trunc('day', bucket_15min)
        ORDER BY bucket_15min
    ) AS occupancy
FROM counting_by_bucket_15min;

CREATE OR REPLACE VIEW occupancy_by_bucket_hour AS
SELECT
    store_id,
    date_trunc('hour', bucket_15min) AS bucket_hour,
    SUM(ins)  AS ins,
    SUM(outs) AS outs,
    AVG(occupancy)::numeric(10, 2) AS avg_occupancy,
    MAX(occupancy) AS max_occupancy,
    MIN(occupancy) AS min_occupancy
FROM occupancy_by_bucket_15min
GROUP BY store_id, date_trunc('hour', bucket_15min);

CREATE OR REPLACE VIEW occupancy_by_bucket_day AS
SELECT
    store_id,
    date_trunc('day', bucket_15min)::date AS bucket_day,
    SUM(ins)  AS ins,
    SUM(outs) AS outs,
    AVG(occupancy)::numeric(10, 2) AS avg_occupancy,
    MAX(occupancy) AS max_occupancy,
    MIN(occupancy) AS min_occupancy
FROM occupancy_by_bucket_15min
GROUP BY store_id, date_trunc('day', bucket_15min)::date;

-- --- turn_in_rate: tasa de captación (ingresos / passersby o shoppers) ---
CREATE OR REPLACE VIEW turn_in_rate_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, w.store_id) AS store_id,
    COALESCE(c.bucket_15min, w.bucket_15min) AS bucket_15min,
    COALESCE(c.ins, 0) AS ins,
    COALESCE(w.passersby, 0) AS passersby,
    COALESCE(w.shoppers, 0) AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0 THEN c.ins::float / w.passersby ELSE NULL END AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0)  > 0 THEN c.ins::float / w.shoppers  ELSE NULL END AS turn_in_shoppers_rate
FROM counting_by_bucket_15min c
FULL OUTER JOIN wifi_ble_by_bucket_15min w
    ON c.store_id = w.store_id AND c.bucket_15min = w.bucket_15min;

CREATE OR REPLACE VIEW turn_in_rate_by_bucket_hour AS
SELECT
    COALESCE(c.store_id, w.store_id) AS store_id,
    COALESCE(c.bucket_hour, w.bucket_hour) AS bucket_hour,
    COALESCE(c.ins, 0) AS ins,
    COALESCE(w.passersby, 0) AS passersby,
    COALESCE(w.shoppers, 0) AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0 THEN c.ins::float / w.passersby ELSE NULL END AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0)  > 0 THEN c.ins::float / w.shoppers  ELSE NULL END AS turn_in_shoppers_rate
FROM counting_by_bucket_hour c
FULL OUTER JOIN wifi_ble_by_bucket_hour w
    ON c.store_id = w.store_id AND c.bucket_hour = w.bucket_hour;

CREATE OR REPLACE VIEW turn_in_rate_by_bucket_day AS
SELECT
    COALESCE(c.store_id, w.store_id) AS store_id,
    COALESCE(c.bucket_day, w.bucket_day) AS bucket_day,
    COALESCE(c.ins, 0) AS ins,
    COALESCE(w.passersby, 0) AS passersby,
    COALESCE(w.shoppers, 0) AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0 THEN c.ins::float / w.passersby ELSE NULL END AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0)  > 0 THEN c.ins::float / w.shoppers  ELSE NULL END AS turn_in_shoppers_rate
FROM counting_by_bucket_day c
FULL OUTER JOIN wifi_ble_by_bucket_day w
    ON c.store_id = w.store_id AND c.bucket_day = w.bucket_day;

-- --- conversion: ventas / ingresos ---
CREATE OR REPLACE VIEW conversion_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, p.store_id) AS store_id,
    COALESCE(c.bucket_15min, p.bucket_15min) AS bucket_15min,
    COALESCE(c.ins, 0) AS visits,
    COALESCE(p.sales, 0) AS sales,
    COALESCE(p.returns, 0) AS returns,
    COALESCE(p.items_sale, 0) AS items_sale,
    COALESCE(p.items_return, 0) AS items_return,
    COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0) AS amount_minor_return,
    COALESCE(p.net_amount_minor, 0) AS net_amount_minor,
    COALESCE(p.currency, 'ARS') AS currency,
    CASE WHEN COALESCE(c.ins, 0) > 0 THEN p.sales::float / c.ins ELSE NULL END AS conversion_rate
FROM counting_by_bucket_15min c
FULL OUTER JOIN pos_by_bucket_15min p
    ON c.store_id = p.store_id AND c.bucket_15min = p.bucket_15min;

CREATE OR REPLACE VIEW conversion_by_bucket_hour AS
SELECT
    COALESCE(c.store_id, p.store_id) AS store_id,
    COALESCE(c.bucket_hour, p.bucket_hour) AS bucket_hour,
    COALESCE(c.ins, 0) AS visits,
    COALESCE(p.sales, 0) AS sales,
    COALESCE(p.returns, 0) AS returns,
    COALESCE(p.items_sale, 0) AS items_sale,
    COALESCE(p.items_return, 0) AS items_return,
    COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0) AS amount_minor_return,
    COALESCE(p.net_amount_minor, 0) AS net_amount_minor,
    COALESCE(p.currency, 'ARS') AS currency,
    CASE WHEN COALESCE(c.ins, 0) > 0 THEN p.sales::float / c.ins ELSE NULL END AS conversion_rate
FROM counting_by_bucket_hour c
FULL OUTER JOIN pos_by_bucket_hour p
    ON c.store_id = p.store_id AND c.bucket_hour = p.bucket_hour;

CREATE OR REPLACE VIEW conversion_by_bucket_day AS
SELECT
    COALESCE(c.store_id, p.store_id) AS store_id,
    COALESCE(c.bucket_day, p.bucket_day) AS bucket_day,
    COALESCE(c.ins, 0) AS visits,
    COALESCE(p.sales, 0) AS sales,
    COALESCE(p.returns, 0) AS returns,
    COALESCE(p.items_sale, 0) AS items_sale,
    COALESCE(p.items_return, 0) AS items_return,
    COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0) AS amount_minor_return,
    COALESCE(p.net_amount_minor, 0) AS net_amount_minor,
    COALESCE(p.currency, 'ARS') AS currency,
    CASE WHEN COALESCE(c.ins, 0) > 0 THEN p.sales::float / c.ins ELSE NULL END AS conversion_rate
FROM counting_by_bucket_day c
FULL OUTER JOIN pos_by_bucket_day p
    ON c.store_id = p.store_id AND c.bucket_day = p.bucket_day;

-- --- visit_duration: Ley de Little aplicada a occupancy + arrivals ---
CREATE OR REPLACE VIEW visit_duration_by_bucket_hour AS
SELECT
    store_id, bucket_hour, avg_occupancy,
    ins AS arrivals,
    CASE WHEN ins > 0 THEN avg_occupancy / (ins / 60.0) ELSE NULL END AS visit_duration_minutes
FROM occupancy_by_bucket_hour;

CREATE OR REPLACE VIEW visit_duration_by_bucket_day AS
SELECT
    store_id,
    date_trunc('day', bucket_hour)::date AS bucket_day,
    SUM(arrivals) AS arrivals,
    AVG(avg_occupancy)::numeric(10, 2) AS avg_occupancy,
    -- Duración de visita diaria = promedio de la duración horaria ponderado por
    -- las llegadas de cada hora: SUM(W_h · λ_h) / SUM(λ_h). Equivale a
    -- 60·SUM(avg_occupancy)/SUM(arrivals) (Little: W = L/λ aplicado al día).
    CASE WHEN SUM(arrivals) > 0
         THEN SUM(visit_duration_minutes * arrivals) / SUM(arrivals)
         ELSE NULL END AS visit_duration_minutes
FROM visit_duration_by_bucket_hour
GROUP BY store_id, date_trunc('day', bucket_hour)::date;

-- --- wifi engagement: visitantes por nº de ventanas de 15 min (repetición intra-día) ---
-- El visitor_hash rota a diario → cada hash vive un solo día → sumar por día sobre
-- el rango es correcto (no hay doble conteo cross-día).
CREATE OR REPLACE VIEW wifi_engagement_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.windows_bucket, r.visitors
    FROM rollup_wifi_engagement_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT store_id, bd, CASE WHEN nw=1 THEN '1' WHEN nw=2 THEN '2' WHEN nw BETWEEN 3 AND 5 THEN '3-5' ELSE '6+' END, COUNT(*)
    FROM (
        SELECT e.store_id, lday(e.last_seen_ts,s.timezone) bd, visitor_hash, COUNT(DISTINCT period_start) AS nw
        FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
        WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND e.last_seen_ts < now() + INTERVAL '1 day' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
        GROUP BY e.store_id, lday(e.last_seen_ts,s.timezone), visitor_hash
    ) v
   GROUP BY store_id, bd, 3;

-- --- RPV: ventas por visitante = facturación ÷ ingresos (= conversión × ticket) ---
CREATE OR REPLACE VIEW revenue_per_visitor_by_bucket_day AS
SELECT COALESCE(c.store_id, p.store_id) AS store_id,
       COALESCE(c.bucket_day, p.bucket_day) AS bucket_day,
       COALESCE(c.ins, 0) AS ins,
       COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale,
       CASE WHEN COALESCE(c.ins,0) > 0
            THEN p.amount_minor_sale::float / 100.0 / c.ins ELSE NULL END AS revenue_per_visitor
FROM counting_by_bucket_day c
FULL OUTER JOIN pos_by_bucket_day p ON c.store_id=p.store_id AND c.bucket_day=p.bucket_day;

CREATE OR REPLACE VIEW revenue_per_visitor_by_bucket_hour AS
SELECT COALESCE(c.store_id, p.store_id) AS store_id,
       COALESCE(c.bucket_hour, p.bucket_hour) AS bucket_hour,
       COALESCE(c.ins, 0) AS ins,
       COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale,
       CASE WHEN COALESCE(c.ins,0) > 0
            THEN p.amount_minor_sale::float / 100.0 / c.ins ELSE NULL END AS revenue_per_visitor
FROM counting_by_bucket_hour c
FULL OUTER JOIN pos_by_bucket_hour p ON c.store_id=p.store_id AND c.bucket_hour=p.bucket_hour;

-- --- ventas/m²: facturación neta ÷ superficie de venta (sites.sales_area_m2) ---
CREATE OR REPLACE VIEW sales_per_sqm_by_bucket_day AS
SELECT p.store_id, p.bucket_day,
       (p.amount_minor_sale - p.amount_minor_return) / 100.0 AS amount_net,
       s.sales_area_m2,
       CASE WHEN s.sales_area_m2 > 0
            THEN ((p.amount_minor_sale - p.amount_minor_return) / 100.0) / s.sales_area_m2
            ELSE NULL END AS sales_per_sqm
FROM pos_by_bucket_day p JOIN sites s USING (store_id);

CREATE OR REPLACE VIEW sales_per_sqm_by_bucket_hour AS
SELECT p.store_id, p.bucket_hour,
       (p.amount_minor_sale - p.amount_minor_return) / 100.0 AS amount_net,
       s.sales_area_m2,
       CASE WHEN s.sales_area_m2 > 0
            THEN ((p.amount_minor_sale - p.amount_minor_return) / 100.0) / s.sales_area_m2
            ELSE NULL END AS sales_per_sqm
FROM pos_by_bucket_hour p JOIN sites s USING (store_id);

-- --- metrics_unified: combinación counting + wifi_ble + pos para la Lambda ---
CREATE OR REPLACE VIEW metrics_unified_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id) AS store_id,
    COALESCE(c.bucket_15min, w.bucket_15min, p.bucket_15min) AS bucket_15min,
    COALESCE(c.ins, 0) AS ins, COALESCE(c.outs, 0) AS outs, COALESCE(c.net, 0) AS net,
    COALESCE(c.ins_adult, 0) AS ins_adult, COALESCE(c.ins_child, 0) AS ins_child, COALESCE(c.ins_unknown, 0) AS ins_unknown,
    COALESCE(c.outs_adult, 0) AS outs_adult, COALESCE(c.outs_child, 0) AS outs_child, COALESCE(c.outs_unknown, 0) AS outs_unknown,
    COALESCE(w.passersby, 0) AS passersby, COALESCE(w.shoppers, 0) AS shoppers,
    COALESCE(p.sales, 0) AS sales, COALESCE(p.returns, 0) AS returns, COALESCE(p.transactions, 0) AS transactions,
    COALESCE(p.items_sale, 0) AS items_sale, COALESCE(p.items_return, 0) AS items_return,
    COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale, COALESCE(p.amount_minor_return, 0) AS amount_minor_return,
    COALESCE(p.currency, 'ARS') AS currency
FROM counting_by_bucket_15min c
FULL OUTER JOIN wifi_ble_by_bucket_15min w
    ON c.store_id = w.store_id AND c.bucket_15min = w.bucket_15min
FULL OUTER JOIN pos_by_bucket_15min p
    ON COALESCE(c.store_id, w.store_id) = p.store_id
   AND COALESCE(c.bucket_15min, w.bucket_15min) = p.bucket_15min;

CREATE OR REPLACE VIEW metrics_unified_by_bucket_hour AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id) AS store_id,
    COALESCE(c.bucket_hour, w.bucket_hour, p.bucket_hour) AS bucket_hour,
    COALESCE(c.ins, 0) AS ins, COALESCE(c.outs, 0) AS outs, COALESCE(c.net, 0) AS net,
    COALESCE(c.ins_adult, 0) AS ins_adult, COALESCE(c.ins_child, 0) AS ins_child, COALESCE(c.ins_unknown, 0) AS ins_unknown,
    COALESCE(c.outs_adult, 0) AS outs_adult, COALESCE(c.outs_child, 0) AS outs_child, COALESCE(c.outs_unknown, 0) AS outs_unknown,
    COALESCE(w.passersby, 0) AS passersby, COALESCE(w.shoppers, 0) AS shoppers,
    COALESCE(p.sales, 0) AS sales, COALESCE(p.returns, 0) AS returns, COALESCE(p.transactions, 0) AS transactions,
    COALESCE(p.items_sale, 0) AS items_sale, COALESCE(p.items_return, 0) AS items_return,
    COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale, COALESCE(p.amount_minor_return, 0) AS amount_minor_return,
    COALESCE(p.currency, 'ARS') AS currency
FROM counting_by_bucket_hour c
FULL OUTER JOIN wifi_ble_by_bucket_hour w
    ON c.store_id = w.store_id AND c.bucket_hour = w.bucket_hour
FULL OUTER JOIN pos_by_bucket_hour p
    ON COALESCE(c.store_id, w.store_id) = p.store_id
   AND COALESCE(c.bucket_hour, w.bucket_hour) = p.bucket_hour;

CREATE OR REPLACE VIEW metrics_unified_by_bucket_day AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id) AS store_id,
    COALESCE(c.bucket_day, w.bucket_day, p.bucket_day) AS bucket_day,
    COALESCE(c.ins, 0) AS ins, COALESCE(c.outs, 0) AS outs, COALESCE(c.net, 0) AS net,
    COALESCE(c.ins_adult, 0) AS ins_adult, COALESCE(c.ins_child, 0) AS ins_child, COALESCE(c.ins_unknown, 0) AS ins_unknown,
    COALESCE(c.outs_adult, 0) AS outs_adult, COALESCE(c.outs_child, 0) AS outs_child, COALESCE(c.outs_unknown, 0) AS outs_unknown,
    COALESCE(w.passersby, 0) AS passersby, COALESCE(w.shoppers, 0) AS shoppers,
    COALESCE(p.sales, 0) AS sales, COALESCE(p.returns, 0) AS returns, COALESCE(p.transactions, 0) AS transactions,
    COALESCE(p.items_sale, 0) AS items_sale, COALESCE(p.items_return, 0) AS items_return,
    COALESCE(p.amount_minor_sale, 0) AS amount_minor_sale, COALESCE(p.amount_minor_return, 0) AS amount_minor_return,
    COALESCE(p.currency, 'ARS') AS currency
FROM counting_by_bucket_day c
FULL OUTER JOIN wifi_ble_by_bucket_day w
    ON c.store_id = w.store_id AND c.bucket_day = w.bucket_day
FULL OUTER JOIN pos_by_bucket_day p
    ON COALESCE(c.store_id, w.store_id) = p.store_id
   AND COALESCE(c.bucket_day, w.bucket_day) = p.bucket_day;

-- --- data_freshness: último timestamp de ingesta por sucursal cross-fact ---
-- Última ingesta por sucursal (cualquiera de las 3 fuentes). Lateral por store:
-- cada max(received_at) usa idx (store_id, received_at) → Index Only Scan O(log n)
-- en vez de agregar las tablas enteras. GREATEST ignora NULLs (store sin alguna
-- fuente) y sólo da NULL si las 3 están vacías.
CREATE OR REPLACE VIEW data_freshness_by_store AS
SELECT s.store_id,
       GREATEST(
           (SELECT MAX(received_at) FROM count_events    WHERE store_id = s.store_id),
           (SELECT MAX(received_at) FROM wifi_ble_events WHERE store_id = s.store_id),
           (SELECT MAX(received_at) FROM pos_transactions WHERE store_id = s.store_id)
       ) AS last_received_at
FROM sites s;

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
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM lambda_writer;
        REVOKE ALL ON SCHEMA public FROM lambda_writer;
        DROP USER lambda_writer;
    END IF;
END $$;

CREATE USER lambda_writer;
GRANT rds_iam TO lambda_writer;
GRANT USAGE ON SCHEMA public TO lambda_writer;
GRANT INSERT ON count_events, wifi_ble_events, telemetry TO lambda_writer;
-- ON CONFLICT necesita SELECT también para evaluar la condición unique:
GRANT SELECT ON count_events, wifi_ble_events, telemetry TO lambda_writer;

-- =============================================================================
-- User Lambda ingest_pos_transaction — IAM auth (sin password)
-- =============================================================================
-- Lambda separada del persist_event con role IAM propio (least privilege:
-- solo INSERT/SELECT sobre pos_transactions, sin acceso a count_events ni
-- telemetry). El API Gateway invoca esta Lambda con IAM auth.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lambda_pos_writer') THEN
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM lambda_pos_writer;
        REVOKE ALL ON SCHEMA public FROM lambda_pos_writer;
        DROP USER lambda_pos_writer;
    END IF;
END $$;

CREATE USER lambda_pos_writer;
GRANT rds_iam TO lambda_pos_writer;
GRANT USAGE ON SCHEMA public TO lambda_pos_writer;
GRANT INSERT, SELECT ON pos_transactions TO lambda_pos_writer;

-- =============================================================================
-- User Lambda query_aggregates — IAM auth, SELECT-only (T9.12, US-09, RF-13)
-- =============================================================================
-- Tercer role separado para least privilege: la Lambda de consulta NO debe
-- poder insertar/modificar. Solo SELECT sobre las tablas raw que necesita el
-- query unificado (counts + wifi/ble + pos + sites/devices para enumerar).

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lambda_query_reader') THEN
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM lambda_query_reader;
        REVOKE ALL ON SCHEMA public FROM lambda_query_reader;
        DROP USER lambda_query_reader;
    END IF;
END $$;

CREATE USER lambda_query_reader;
GRANT rds_iam TO lambda_query_reader;
GRANT USAGE ON SCHEMA public TO lambda_query_reader;
-- La Lambda query_aggregates consume las vistas unificadas + data_freshness,
-- NO las tablas crudas. Simétrico con readonly_external (solo via vistas).
GRANT SELECT ON
    metrics_unified_by_bucket_15min,
    metrics_unified_by_bucket_hour,
    metrics_unified_by_bucket_day,
    data_freshness_by_store,
    sites, devices
TO lambda_query_reader;
-- Las vistas counting_* y wifi_ble_* invocan las funciones de
-- categorización en sus FILTER clauses — el planner necesita EXECUTE para
-- inlinearlas. Las funciones SQL IMMUTABLE por default son PUBLIC, pero
-- somos explícitos.
GRANT EXECUTE ON FUNCTION height_class(REAL) TO lambda_query_reader;
GRANT EXECUTE ON FUNCTION rssi_class(INT)    TO lambda_query_reader;

-- =============================================================================
-- User read-only para acceso programático externo (US-08)
-- =============================================================================
-- ``readonly_external`` permite a partners / analistas externos consultar los
-- agregados via SQL directo (DBeaver, psql, Python con psycopg) sin riesgo de
-- modificar data. Acceso vía PASSWORD (no IAM auth — el cliente externo no
-- necesita ser un principal AWS) leído de Secrets Manager (recurso
-- ``RdsReadonlyExternalSecret`` del CFN). Permisos minimos:
--   - USAGE en schema public.
--   - SELECT solo sobre las VIEWS de agregados (no sobre tablas crudas con
--     device_id / timestamps fine-grained). Si el partner necesita raw,
--     expandir granularly per vista.
-- La connectividad externa requiere ``RdsAllowExternalReadOnly=true`` en el
-- CFN (default false) — abre un SG ingress al puerto 5432 desde los CIDRs en
-- ``RdsExternalReadOnlyCidrs``. Ver docs/api_access.md.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'readonly_external') THEN
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM readonly_external;
        REVOKE ALL ON SCHEMA public FROM readonly_external;
        DROP USER readonly_external;
    END IF;
END $$;

-- Password gestionado fuera: el ``RdsReadonlyExternalSecret`` del CFN tiene
-- credentials autogeneradas; el script de provisioning sincroniza el password
-- desde el secret a este rol (idempotente, ver scripts/provision.py / o el
-- propio apply_bootstrap.py).
CREATE USER readonly_external WITH PASSWORD 'CHANGE_ME_FROM_SECRETS_MANAGER';
GRANT USAGE ON SCHEMA public TO readonly_external;
-- Acceso solo a las vistas del producto cartesiano + dimensiones. NO se
-- conceden las vistas metrics_unified_* (uso interno de la Lambda — el
-- partner prefiere las vistas individuales por fuente).
GRANT SELECT ON
    counting_by_bucket_15min,
    counting_by_bucket_hour,
    counting_by_bucket_day,
    wifi_ble_by_bucket_15min,
    wifi_ble_by_bucket_hour,
    wifi_ble_by_bucket_day,
    pos_by_bucket_15min,
    pos_by_bucket_hour,
    pos_by_bucket_day,
    occupancy_by_bucket_15min,
    occupancy_by_bucket_hour,
    occupancy_by_bucket_day,
    turn_in_rate_by_bucket_15min,
    turn_in_rate_by_bucket_hour,
    turn_in_rate_by_bucket_day,
    conversion_by_bucket_15min,
    conversion_by_bucket_hour,
    conversion_by_bucket_day,
    visit_duration_by_bucket_hour,
    visit_duration_by_bucket_day,
    wifi_engagement_by_bucket_day,
    revenue_per_visitor_by_bucket_day,
    revenue_per_visitor_by_bucket_hour,
    sales_per_sqm_by_bucket_day,
    sales_per_sqm_by_bucket_hour,
    data_freshness_by_store,
    sites, devices
TO readonly_external;
GRANT EXECUTE ON FUNCTION height_class(REAL) TO readonly_external;
GRANT EXECUTE ON FUNCTION rssi_class(INT)    TO readonly_external;

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
-- La creación de la DB ``grafana`` está fuera de este script porque
-- ``CREATE DATABASE`` no puede correr en una transacción + el bootstrap se
-- aplica vía psycopg (sin docker). El runner ``infra/sql/apply_bootstrap.py``
-- lo crea idempotentemente con autocommit antes/después de este script.
-- Para correr el script MANUALMENTE en DBeaver/psql contra la DB
-- ``people_counter``, ejecutar aparte:
--     SELECT 1 FROM pg_database WHERE datname = 'grafana';
--     -- si no existe:
--     CREATE DATABASE grafana OWNER people_counter;

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

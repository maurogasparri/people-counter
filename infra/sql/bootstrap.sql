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

DROP TABLE IF EXISTS count_events       CASCADE;
DROP TABLE IF EXISTS wifi_ble_summary   CASCADE;
DROP TABLE IF EXISTS telemetry          CASCADE;
DROP TABLE IF EXISTS pos_transactions   CASCADE;

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
    -- 'adult' | 'child' | 'unknown' (clasificacion server-side desde height_m).
    -- Usado en S10 dashboards (US-05 breakdown adult/child).
    height_class    TEXT         CHECK (height_class IN ('adult', 'child', 'unknown')),
    -- Altura cruda en metros. Se mantiene para debug operativo: detectar
    -- drift del mounting_height_m del config vs altura fisica real ("99% de
    -- los eventos del device X tienen height_m < 1.0 -> alguien movio el bracket").
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
CREATE INDEX IF NOT EXISTS idx_count_events_day
    ON count_events (store_id, bucket_day DESC);

CREATE TABLE IF NOT EXISTS wifi_ble_summary (
    summary_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id       TEXT         NOT NULL,
    store_id        TEXT         NOT NULL,
    period_start    TIMESTAMPTZ  NOT NULL,                  -- inicio de la ventana (epoch del device)
    period_end      TIMESTAMPTZ  NOT NULL,                  -- fin de la ventana
    -- Bucket de 15min — derivado server-side desde period_start (alineado al
    -- arranque de la ventana del device, NO al last_seen_ts que es no-determinístico).
    -- Server-derived facilita migrar tamaño de bucket sin tocar device.
    bucket_15min    TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (period_start - TIMESTAMPTZ 'epoch')) / 900) * 900)) STORED,
    bucket_hour     TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (period_start - TIMESTAMPTZ 'epoch')) / 3600) * 3600)) STORED,
    bucket_day      DATE         GENERATED ALWAYS AS ((TIMESTAMP 'epoch' + floor(extract(epoch FROM (period_start - TIMESTAMPTZ 'epoch')) / 86400) * INTERVAL '1 day')::date) STORED,
    passersby       INT          NOT NULL,                  -- post L2 dedup
    shoppers        INT          NOT NULL,                  -- en rango cercano (RSSI fuerte)
    -- Timestamp de la ÚLTIMA detección de un visitor dentro del período.
    -- Info diagnóstica — útil para alarmas "no se ha visto a nadie hace N min",
    -- o para diferenciar "no había nadie afuera" vs "el subsistema murió" sin
    -- depender de `received_at` (que es de cuando llegó el msg, no del último
    -- visitor real). NULLABLE: devices con firmware viejo no lo mandan.
    last_seen_ts    TIMESTAMPTZ,
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia ante retries del device: una sola fila por (device, ventana).
    UNIQUE (device_id, period_start, period_end)
);

CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_bucket15
    ON wifi_ble_summary (store_id, bucket_15min DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_day
    ON wifi_ble_summary (store_id, bucket_day DESC);

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
    -- el ROI / total counts emitidos. Ideal ≈ 1.0 (1 persona = 1 ID = 1
    -- evento); >1.3 indica fragmentación de identidad.
    track_stitching_ratio         REAL,
    -- Cantidad de death-emits disparados hoy (el fallback que cuenta tracks
    -- que cruzaron pero murieron dentro del ROI). Combinado con el ratio
    -- diferencia 'fragmenta-y-rescata' (count alto) de 'fragmenta-y-pierde'
    -- (count bajo con ratio alto → recall del detector flojo).
    death_emit_count              INT,
    -- Cantidad de ghost adoptions del tracker desde el boot del proceso
    -- (capa 1 del rescue). Cierra el árbol diagnóstico: combinado con
    -- death_emit_count y track_stitching_ratio permite distinguir
    -- "tracker perfecto" / "fragmentación rescatada por adopción" /
    -- "fragmentación rescatada por death-emit" / "fragmentación sin rescate".
    ghost_adoption_count          INT,
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
    timezone    TEXT,                                     -- IANA, ej. 'America/Argentina/Buenos_Aires'
    address     TEXT,
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

-- Grafana consulta como master user (people_counter), que es owner → ya puede
-- leer estas tablas. El seed de provisioning corre como master (mismo path que
-- este bootstrap). lambda_writer escribe hechos, no necesita las dimensiones.

-- =============================================================================
-- View multi-cam: agregación por store con MAX (no SUM) para WiFi/BLE
-- =============================================================================
-- Cuando un store tenga 2+ cams, WiFi/BLE range (~30-50m) excede al vision
-- range (3-5m), entonces las cams ven la misma gente. Tomar MAX en vez de SUM
-- evita doble-conteo. Con 1 cam por store, MAX = el row de esa cam.

CREATE OR REPLACE VIEW wifi_ble_store_traffic AS
SELECT
    bucket_15min,
    store_id,
    MAX(passersby)   AS passersby,
    MAX(shoppers)    AS shoppers,
    COUNT(*)         AS cams_reporting
FROM wifi_ble_summary
GROUP BY bucket_15min, store_id;

-- =============================================================================
-- Counting agregado por bucket (15min — server-derived via GENERATED)
-- =============================================================================
-- bucket_15min es GENERATED ALWAYS AS STORED desde event_ts (ver tabla
-- arriba). Nunca es NULL; el device manda event_ts crudo y RDS deriva.

CREATE OR REPLACE VIEW counting_by_bucket AS
SELECT
    store_id,
    bucket_15min,
    COUNT(*) FILTER (WHERE direction = 'in')  AS ins,
    COUNT(*) FILTER (WHERE direction = 'out') AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
      - COUNT(*) FILTER (WHERE direction = 'out') AS net
FROM count_events
GROUP BY store_id, bucket_15min;

-- =============================================================================
-- Turn-in rate por bucket: gente que entro / gente que paso cerca
-- =============================================================================
-- Une counting (ins) con WiFi/BLE (passersby/shoppers). Usa wifi_ble_store_traffic
-- (MAX por store) para multi-cam dedup. FULL OUTER JOIN preserva buckets donde
-- solo una fuente reporto (e.g., WiFi/BLE OFF en el device, o un bucket sin
-- transito que no genero counting events).

CREATE OR REPLACE VIEW turn_in_rate_by_bucket AS
WITH ins AS (
    SELECT store_id, bucket_15min, COUNT(*) AS ins
    FROM count_events
    WHERE direction = 'in'
    GROUP BY store_id, bucket_15min
)
SELECT
    COALESCE(i.store_id, w.store_id)        AS store_id,
    COALESCE(i.bucket_15min, w.bucket_15min) AS bucket_15min,
    COALESCE(i.ins, 0)                       AS ins,
    COALESCE(w.passersby, 0)                 AS passersby,
    COALESCE(w.shoppers, 0)                  AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0
         THEN i.ins::float / w.passersby
         ELSE NULL END                       AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0) > 0
         THEN i.ins::float / w.shoppers
         ELSE NULL END                       AS turn_in_shoppers_rate
FROM ins i
FULL OUTER JOIN wifi_ble_store_traffic w
    ON w.store_id = i.store_id AND w.bucket_15min = i.bucket_15min;

-- =============================================================================
-- Rollups hourly / daily encima de los views de bucket
-- =============================================================================
-- Usan bucket_hour / bucket_day directos (generated columns) -> sin date_trunc.

CREATE OR REPLACE VIEW counting_hourly AS
SELECT
    store_id,
    bucket_hour,
    COUNT(*) FILTER (WHERE direction = 'in')  AS ins,
    COUNT(*) FILTER (WHERE direction = 'out') AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
      - COUNT(*) FILTER (WHERE direction = 'out') AS net
FROM count_events
GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW counting_daily AS
SELECT
    store_id,
    bucket_day,
    COUNT(*) FILTER (WHERE direction = 'in')  AS ins,
    COUNT(*) FILTER (WHERE direction = 'out') AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
      - COUNT(*) FILTER (WHERE direction = 'out') AS net,
    -- Breakdown adult/child para US-05 (solo en daily, en 15min es ruido)
    COUNT(*) FILTER (WHERE direction = 'in' AND height_class = 'adult') AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in' AND height_class = 'child') AS ins_child
FROM count_events
GROUP BY store_id, bucket_day;

-- =============================================================================
-- Conversion rate: visitas (counter) vs ventas (POS) — cierra US-06
-- =============================================================================
-- Granularidad de 15min para drill-down. FULL OUTER JOIN preserva buckets donde
-- solo una fuente reporto (counter sin POS, o POS fuera del horario de counter).

CREATE OR REPLACE VIEW conversion_rate_by_store AS
WITH visits AS (
    SELECT store_id, bucket_15min,
           COUNT(*) FILTER (WHERE direction = 'in') AS visits
    FROM count_events
    GROUP BY store_id, bucket_15min
),
pos_agg AS (
    SELECT
        store_id,
        bucket_15min,
        COUNT(*) FILTER (WHERE type = 'sale')   AS sales,
        COUNT(*) FILTER (WHERE type = 'return') AS returns,
        COALESCE(SUM(items)        FILTER (WHERE type = 'sale'),   0) AS sales_items,
        COALESCE(SUM(items)        FILTER (WHERE type = 'return'), 0) AS returns_items,
        COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0) AS sales_amount_minor,
        COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0) AS returns_amount_minor
    FROM pos_transactions
    GROUP BY store_id, bucket_15min
)
SELECT
    COALESCE(v.store_id, p.store_id)         AS store_id,
    COALESCE(v.bucket_15min, p.bucket_15min) AS bucket_15min,
    COALESCE(v.visits, 0)                    AS visits,
    COALESCE(p.sales, 0)                     AS sales,
    COALESCE(p.returns, 0)                   AS returns,
    COALESCE(p.sales_items, 0)               AS sales_items,
    COALESCE(p.returns_items, 0)             AS returns_items,
    COALESCE(p.sales_amount_minor, 0)        AS sales_amount_minor,
    COALESCE(p.returns_amount_minor, 0)      AS returns_amount_minor,
    COALESCE(p.sales_amount_minor, 0)
      - COALESCE(p.returns_amount_minor, 0)  AS net_amount_minor,
    -- Conversion rate = sales / visits. NULL si no hay visits (evita div/0).
    CASE WHEN COALESCE(v.visits, 0) > 0
         THEN p.sales::float / v.visits
         ELSE NULL END                       AS conversion_rate
FROM visits v
FULL OUTER JOIN pos_agg p
    ON v.store_id = p.store_id AND v.bucket_15min = p.bucket_15min;

-- Rollups hourly / daily encima del 15min para reportes mas anchos.

CREATE OR REPLACE VIEW conversion_rate_hourly AS
SELECT
    store_id,
    date_trunc('hour', bucket_15min) AS bucket_hour,
    SUM(visits)               AS visits,
    SUM(sales)                AS sales,
    SUM(returns)              AS returns,
    SUM(sales_items)          AS sales_items,
    SUM(returns_items)        AS returns_items,
    SUM(sales_amount_minor)   AS sales_amount_minor,
    SUM(returns_amount_minor) AS returns_amount_minor,
    SUM(net_amount_minor)     AS net_amount_minor,
    CASE WHEN SUM(visits) > 0
         THEN SUM(sales)::float / SUM(visits)
         ELSE NULL END        AS conversion_rate
FROM conversion_rate_by_store
GROUP BY store_id, date_trunc('hour', bucket_15min);

CREATE OR REPLACE VIEW conversion_rate_daily AS
SELECT
    store_id,
    date_trunc('day', bucket_15min)::date AS bucket_day,
    SUM(visits)               AS visits,
    SUM(sales)                AS sales,
    SUM(returns)              AS returns,
    SUM(sales_items)          AS sales_items,
    SUM(returns_items)        AS returns_items,
    SUM(sales_amount_minor)   AS sales_amount_minor,
    SUM(returns_amount_minor) AS returns_amount_minor,
    SUM(net_amount_minor)     AS net_amount_minor,
    CASE WHEN SUM(visits) > 0
         THEN SUM(sales)::float / SUM(visits)
         ELSE NULL END        AS conversion_rate
FROM conversion_rate_by_store
GROUP BY store_id, date_trunc('day', bucket_15min)::date;

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
GRANT INSERT ON count_events, wifi_ble_summary, telemetry TO lambda_writer;
-- ON CONFLICT necesita SELECT también para evaluar la condición unique:
GRANT SELECT ON count_events, wifi_ble_summary, telemetry TO lambda_writer;

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
GRANT SELECT ON
    counting_hourly, counting_daily, counting_by_bucket,
    turn_in_rate_by_bucket,
    conversion_rate_hourly, conversion_rate_daily, conversion_rate_by_store,
    wifi_ble_store_traffic,
    sites, devices
TO readonly_external;

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

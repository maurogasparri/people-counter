-- =============================================================================
-- 2026-05-23  bucket_15min server-derived + last_seen_ts en wifi_ble_summary
-- =============================================================================
-- Desacopla device ↔ schema: el device deja de calcular el bucket_15min en el
-- payload; la columna pasa a ser GENERATED ALWAYS AS STORED desde event_ts /
-- period_start. Migrar a otro bucket size (ej. 5min) en el futuro = DROP +
-- ADD COLUMN con nueva expresión, sin tocar device / MQTT / Lambda.
--
-- Idempotente: usa DROP/ADD IF EXISTS para que un re-run no rompa.
--
-- Apply: psql ... -f 2026-05-23-bucket-server-derived.sql
--    o:  py infra/sql/apply_bootstrap.py --migration 2026-05-23-bucket-server-derived

BEGIN;

-- ---------------------------------------------------------------------------
-- Drop views dependientes (CASCADE no hace falta porque las recreamos abajo).
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS conversion_rate_daily;
DROP VIEW IF EXISTS conversion_rate_hourly;
DROP VIEW IF EXISTS conversion_rate_by_store;
DROP VIEW IF EXISTS turn_in_rate_by_bucket;
DROP VIEW IF EXISTS counting_by_bucket;
DROP VIEW IF EXISTS wifi_ble_store_traffic;

-- ---------------------------------------------------------------------------
-- count_events.bucket_15min: regular -> GENERATED
-- ---------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_count_events_bucket15;

ALTER TABLE count_events DROP COLUMN IF EXISTS bucket_15min;
ALTER TABLE count_events ADD COLUMN bucket_15min TIMESTAMPTZ
    GENERATED ALWAYS AS (
        to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 900) * 900)
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_count_events_bucket15
    ON count_events (store_id, bucket_15min DESC);

-- ---------------------------------------------------------------------------
-- wifi_ble_summary.bucket_15min: regular -> GENERATED, + last_seen_ts
-- ---------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_wifi_ble_store_bucket15;

ALTER TABLE wifi_ble_summary DROP COLUMN IF EXISTS bucket_15min;
ALTER TABLE wifi_ble_summary ADD COLUMN bucket_15min TIMESTAMPTZ
    GENERATED ALWAYS AS (
        to_timestamp(floor(extract(epoch FROM (period_start - TIMESTAMPTZ 'epoch')) / 900) * 900)
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_bucket15
    ON wifi_ble_summary (store_id, bucket_15min DESC);

-- last_seen_ts: timestamp de la última detección dentro del período.
-- NULLABLE — rows históricos no tienen back-fill (la info no existe).
ALTER TABLE wifi_ble_summary
    ADD COLUMN IF NOT EXISTS last_seen_ts TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- Recrear views (copia idéntica del bootstrap.sql, sin los WHERE
-- bucket_15min IS NOT NULL que ya no aplican — la columna ahora nunca es NULL).
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW wifi_ble_store_traffic AS
SELECT
    bucket_15min,
    store_id,
    MAX(passersby)   AS passersby,
    MAX(shoppers)    AS shoppers,
    COUNT(*)         AS cams_reporting
FROM wifi_ble_summary
GROUP BY bucket_15min, store_id;

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
    CASE WHEN COALESCE(v.visits, 0) > 0
         THEN p.sales::float / v.visits
         ELSE NULL END                       AS conversion_rate
FROM visits v
FULL OUTER JOIN pos_agg p
    ON v.store_id = p.store_id AND v.bucket_15min = p.bucket_15min;

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

COMMIT;

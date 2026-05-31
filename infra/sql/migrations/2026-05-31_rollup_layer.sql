-- =============================================================================
-- Migración: capa de rollup incremental + live tail + métricas nuevas
-- =============================================================================
-- Speed-layer + batch-layer sobre RDS Postgres:
--   * rollup_* (batch): agregados pre-computados por (store_id, bucket), chicos.
--   * vistas base re-apuntadas = rollup (buckets cerrados) UNION ALL raw (bucket
--     abierto) → historia instantánea + dato vivo ≤5s, sin tocar el hot-path.
--   * refresh_rollups(): recomputa por watermark (received_at) sólo los buckets
--     que recibieron datos nuevos → O(reciente), escala con la flota, auto-reparable.
--   * Fase 0: sites.sales_area_m2. Fase 2: revenue_per_visitor_*, sales_per_sqm_*.
--
-- Idempotente (IF NOT EXISTS / OR REPLACE). NO toca tablas raw ni ingesta.
-- Tras aplicar: CALL refresh_rollups();  (backfill — procesa toda la historia).
-- Scheduler (pg_cron o EventBridge+Lambda) se configura aparte.
-- =============================================================================

-- ───────────────────────────────────────────────────────────────────────────
-- Fase 0 — dimensión: superficie de venta por sucursal (para ventas/m²)
-- ───────────────────────────────────────────────────────────────────────────
ALTER TABLE sites ADD COLUMN IF NOT EXISTS sales_area_m2 NUMERIC;
COMMENT ON COLUMN sites.sales_area_m2 IS
    'Superficie de venta en m². Dato estático cargado al provisionar (como lat/long). Base de la métrica ventas/m².';

-- ───────────────────────────────────────────────────────────────────────────
-- Helpers: bucket actual (abierto). Espejan EXACTO las fórmulas de las columnas
-- GENERATED (epoch/UTC) → la comparación vivo/cerrado es tz-independiente y precisa.
-- STABLE: constantes dentro de un statement. EXECUTE es PUBLIC por default.
-- ───────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION cur_bucket_day() RETURNS date LANGUAGE sql STABLE AS $$
    SELECT (TIMESTAMP 'epoch' + floor(extract(epoch FROM now()) / 86400) * INTERVAL '1 day')::date
$$;
CREATE OR REPLACE FUNCTION cur_bucket_hour() RETURNS timestamptz LANGUAGE sql STABLE AS $$
    SELECT to_timestamp(floor(extract(epoch FROM now()) / 3600) * 3600)
$$;
CREATE OR REPLACE FUNCTION cur_bucket_15min() RETURNS timestamptz LANGUAGE sql STABLE AS $$
    SELECT to_timestamp(floor(extract(epoch FROM now()) / 900) * 900)
$$;

-- ───────────────────────────────────────────────────────────────────────────
-- Estado del watermark (1 fila por fuente)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rollup_state (
    source            TEXT PRIMARY KEY,
    last_refreshed_at TIMESTAMPTZ NOT NULL DEFAULT '-infinity'  -- backfill total al primer CALL
);
INSERT INTO rollup_state (source) VALUES ('counting'), ('wifi_ble'), ('pos')
    ON CONFLICT (source) DO NOTHING;

-- ───────────────────────────────────────────────────────────────────────────
-- Tablas de rollup (batch layer). Tipos = los de los agregados (bigint/numeric)
-- para que el UNION ALL con el live-tail no choque tipos.
-- ───────────────────────────────────────────────────────────────────────────
-- counting --------------------------------------------------------------------
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
-- wifi_ble --------------------------------------------------------------------
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
-- engagement (visitantes por nº de ventanas de 15 min, por día) ---------------
CREATE TABLE IF NOT EXISTS rollup_wifi_engagement_day (
    store_id TEXT NOT NULL, bucket_day DATE NOT NULL,
    windows_bucket TEXT NOT NULL,   -- '1' | '2' | '3-5' | '6+'
    visitors BIGINT NOT NULL,
    PRIMARY KEY (store_id, bucket_day, windows_bucket)
);
-- pos -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rollup_pos_15min (
    store_id TEXT NOT NULL, bucket_15min TIMESTAMPTZ NOT NULL,
    sales BIGINT, returns BIGINT, transactions BIGINT,
    items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, net_amount_minor NUMERIC,
    currency CHAR(3),
    PRIMARY KEY (store_id, bucket_15min)
);
CREATE TABLE IF NOT EXISTS rollup_pos_hour (
    store_id TEXT NOT NULL, bucket_hour TIMESTAMPTZ NOT NULL,
    sales BIGINT, returns BIGINT, transactions BIGINT,
    items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, net_amount_minor NUMERIC,
    currency CHAR(3),
    PRIMARY KEY (store_id, bucket_hour)
);
CREATE TABLE IF NOT EXISTS rollup_pos_day (
    store_id TEXT NOT NULL, bucket_day DATE NOT NULL,
    sales BIGINT, returns BIGINT, transactions BIGINT,
    items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, net_amount_minor NUMERIC,
    currency CHAR(3),
    PRIMARY KEY (store_id, bucket_day)
);

-- ───────────────────────────────────────────────────────────────────────────
-- Índices BRIN sobre received_at (append-ordenado) → el refresh por watermark
-- ("qué llegó desde la última corrida") es barato. BRIN = chico y casi gratis.
-- ───────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_count_events_received_brin    ON count_events     USING brin (received_at);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_received_brin ON wifi_ble_events  USING brin (received_at);
CREATE INDEX IF NOT EXISTS idx_pos_received_brin             ON pos_transactions USING brin (received_at);

-- ───────────────────────────────────────────────────────────────────────────
-- refresh_rollups(): recomputa por watermark. Sin control de transacción interno
-- (corre en la transacción del caller: CALL en autocommit, o el job de pg_cron).
-- ───────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE PROCEDURE refresh_rollups() LANGUAGE plpgsql AS $$
DECLARE
    wm  TIMESTAMPTZ;
    -- watermark nuevo con 1 min de solapa: cubre inserts in-flight durante el
    -- refresh (MVCC). El UPSERT es idempotente, así que reprocesar 1 min no daña.
    nwm TIMESTAMPTZ := now() - INTERVAL '1 minute';
BEGIN
    -- ===== counting =====
    SELECT last_refreshed_at INTO wm FROM rollup_state WHERE source = 'counting';

    INSERT INTO rollup_counting_day AS r (store_id, bucket_day, ins, outs, net,
        ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown)
    SELECT store_id, bucket_day,
        COUNT(*) FILTER (WHERE direction='in'),
        COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events
    WHERE bucket_day IN (SELECT DISTINCT bucket_day FROM count_events WHERE received_at > wm)
    GROUP BY store_id, bucket_day
    ON CONFLICT (store_id, bucket_day) DO UPDATE SET
        ins=EXCLUDED.ins, outs=EXCLUDED.outs, net=EXCLUDED.net,
        ins_adult=EXCLUDED.ins_adult, ins_child=EXCLUDED.ins_child, ins_unknown=EXCLUDED.ins_unknown,
        outs_adult=EXCLUDED.outs_adult, outs_child=EXCLUDED.outs_child, outs_unknown=EXCLUDED.outs_unknown;

    INSERT INTO rollup_counting_hour AS r (store_id, bucket_hour, ins, outs, net,
        ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown)
    SELECT store_id, bucket_hour,
        COUNT(*) FILTER (WHERE direction='in'),
        COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events
    WHERE bucket_hour IN (SELECT DISTINCT bucket_hour FROM count_events WHERE received_at > wm)
    GROUP BY store_id, bucket_hour
    ON CONFLICT (store_id, bucket_hour) DO UPDATE SET
        ins=EXCLUDED.ins, outs=EXCLUDED.outs, net=EXCLUDED.net,
        ins_adult=EXCLUDED.ins_adult, ins_child=EXCLUDED.ins_child, ins_unknown=EXCLUDED.ins_unknown,
        outs_adult=EXCLUDED.outs_adult, outs_child=EXCLUDED.outs_child, outs_unknown=EXCLUDED.outs_unknown;

    INSERT INTO rollup_counting_15min AS r (store_id, bucket_15min, ins, outs, net,
        ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown)
    SELECT store_id, bucket_15min,
        COUNT(*) FILTER (WHERE direction='in'),
        COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
        COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events
    WHERE bucket_15min IN (SELECT DISTINCT bucket_15min FROM count_events WHERE received_at > wm)
    GROUP BY store_id, bucket_15min
    ON CONFLICT (store_id, bucket_15min) DO UPDATE SET
        ins=EXCLUDED.ins, outs=EXCLUDED.outs, net=EXCLUDED.net,
        ins_adult=EXCLUDED.ins_adult, ins_child=EXCLUDED.ins_child, ins_unknown=EXCLUDED.ins_unknown,
        outs_adult=EXCLUDED.outs_adult, outs_child=EXCLUDED.outs_child, outs_unknown=EXCLUDED.outs_unknown;

    UPDATE rollup_state SET last_refreshed_at = nwm WHERE source = 'counting';

    -- ===== wifi_ble ===== (distinct se computa directo a cada grano)
    SELECT last_refreshed_at INTO wm FROM rollup_state WHERE source = 'wifi_ble';

    INSERT INTO rollup_wifi_ble_day AS r (store_id, bucket_day, passersby, shoppers, visitors)
    SELECT store_id, bucket_day,
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
        COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events
    WHERE bucket_day IN (SELECT DISTINCT bucket_day FROM wifi_ble_events WHERE received_at > wm)
    GROUP BY store_id, bucket_day
    ON CONFLICT (store_id, bucket_day) DO UPDATE SET
        passersby=EXCLUDED.passersby, shoppers=EXCLUDED.shoppers, visitors=EXCLUDED.visitors;

    INSERT INTO rollup_wifi_ble_hour AS r (store_id, bucket_hour, passersby, shoppers, visitors)
    SELECT store_id, bucket_hour,
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
        COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events
    WHERE bucket_hour IN (SELECT DISTINCT bucket_hour FROM wifi_ble_events WHERE received_at > wm)
    GROUP BY store_id, bucket_hour
    ON CONFLICT (store_id, bucket_hour) DO UPDATE SET
        passersby=EXCLUDED.passersby, shoppers=EXCLUDED.shoppers, visitors=EXCLUDED.visitors;

    INSERT INTO rollup_wifi_ble_15min AS r (store_id, bucket_15min, passersby, shoppers, visitors)
    SELECT store_id, bucket_15min,
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
        COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
        COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events
    WHERE bucket_15min IN (SELECT DISTINCT bucket_15min FROM wifi_ble_events WHERE received_at > wm)
    GROUP BY store_id, bucket_15min
    ON CONFLICT (store_id, bucket_15min) DO UPDATE SET
        passersby=EXCLUDED.passersby, shoppers=EXCLUDED.shoppers, visitors=EXCLUDED.visitors;

    -- engagement: el SET de windows_bucket por (store,día) puede cambiar → DELETE+INSERT
    -- de los días afectados (no UPSERT, que dejaría categorías viejas huérfanas).
    DELETE FROM rollup_wifi_engagement_day
     WHERE bucket_day IN (SELECT DISTINCT bucket_day FROM wifi_ble_events WHERE received_at > wm);
    INSERT INTO rollup_wifi_engagement_day (store_id, bucket_day, windows_bucket, visitors)
    SELECT store_id, bucket_day,
        CASE WHEN nw=1 THEN '1' WHEN nw=2 THEN '2' WHEN nw BETWEEN 3 AND 5 THEN '3-5' ELSE '6+' END,
        COUNT(*)
    FROM (
        SELECT store_id, bucket_day, visitor_hash, COUNT(DISTINCT period_start) AS nw
        FROM wifi_ble_events
        WHERE bucket_day IN (SELECT DISTINCT bucket_day FROM wifi_ble_events WHERE received_at > wm)
        GROUP BY store_id, bucket_day, visitor_hash
    ) v
    GROUP BY store_id, bucket_day, 3;

    UPDATE rollup_state SET last_refreshed_at = nwm WHERE source = 'wifi_ble';

    -- ===== pos =====
    SELECT last_refreshed_at INTO wm FROM rollup_state WHERE source = 'pos';

    INSERT INTO rollup_pos_day AS r (store_id, bucket_day, sales, returns, transactions,
        items_sale, items_return, amount_minor_sale, amount_minor_return, net_amount_minor, currency)
    SELECT store_id, bucket_day,
        COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
        COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        MAX(currency)
    FROM pos_transactions
    WHERE bucket_day IN (SELECT DISTINCT bucket_day FROM pos_transactions WHERE received_at > wm)
    GROUP BY store_id, bucket_day
    ON CONFLICT (store_id, bucket_day) DO UPDATE SET
        sales=EXCLUDED.sales, returns=EXCLUDED.returns, transactions=EXCLUDED.transactions,
        items_sale=EXCLUDED.items_sale, items_return=EXCLUDED.items_return,
        amount_minor_sale=EXCLUDED.amount_minor_sale, amount_minor_return=EXCLUDED.amount_minor_return,
        net_amount_minor=EXCLUDED.net_amount_minor, currency=EXCLUDED.currency;

    INSERT INTO rollup_pos_hour AS r (store_id, bucket_hour, sales, returns, transactions,
        items_sale, items_return, amount_minor_sale, amount_minor_return, net_amount_minor, currency)
    SELECT store_id, bucket_hour,
        COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
        COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        MAX(currency)
    FROM pos_transactions
    WHERE bucket_hour IN (SELECT DISTINCT bucket_hour FROM pos_transactions WHERE received_at > wm)
    GROUP BY store_id, bucket_hour
    ON CONFLICT (store_id, bucket_hour) DO UPDATE SET
        sales=EXCLUDED.sales, returns=EXCLUDED.returns, transactions=EXCLUDED.transactions,
        items_sale=EXCLUDED.items_sale, items_return=EXCLUDED.items_return,
        amount_minor_sale=EXCLUDED.amount_minor_sale, amount_minor_return=EXCLUDED.amount_minor_return,
        net_amount_minor=EXCLUDED.net_amount_minor, currency=EXCLUDED.currency;

    INSERT INTO rollup_pos_15min AS r (store_id, bucket_15min, sales, returns, transactions,
        items_sale, items_return, amount_minor_sale, amount_minor_return, net_amount_minor, currency)
    SELECT store_id, bucket_15min,
        COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
        COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
        MAX(currency)
    FROM pos_transactions
    WHERE bucket_15min IN (SELECT DISTINCT bucket_15min FROM pos_transactions WHERE received_at > wm)
    GROUP BY store_id, bucket_15min
    ON CONFLICT (store_id, bucket_15min) DO UPDATE SET
        sales=EXCLUDED.sales, returns=EXCLUDED.returns, transactions=EXCLUDED.transactions,
        items_sale=EXCLUDED.items_sale, items_return=EXCLUDED.items_return,
        amount_minor_sale=EXCLUDED.amount_minor_sale, amount_minor_return=EXCLUDED.amount_minor_return,
        net_amount_minor=EXCLUDED.net_amount_minor, currency=EXCLUDED.currency;

    UPDATE rollup_state SET last_refreshed_at = nwm WHERE source = 'pos';
END $$;

-- ───────────────────────────────────────────────────────────────────────────
-- Vistas base re-apuntadas: rollup (bucket != actual) UNION ALL raw (bucket = actual).
-- Sirve futuro/pasado desde el rollup (instantáneo) y SOLO el bucket abierto en
-- vivo (≤5s). Mismas columnas que las vistas previas → CREATE OR REPLACE compatible
-- (a wifi se le AGREGA 'visitors' al final, permitido). Las derivadas (turn_in,
-- conversion, metrics_unified, visit_duration) NO se tocan: leen estas por nombre.
-- ───────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW counting_by_bucket_day AS
  SELECT store_id, bucket_day, ins, outs, net,
         ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown
    FROM rollup_counting_day WHERE bucket_day <> cur_bucket_day()
  UNION ALL
  SELECT store_id, bucket_day,
         COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events WHERE bucket_day = cur_bucket_day()
   GROUP BY store_id, bucket_day;

CREATE OR REPLACE VIEW counting_by_bucket_hour AS
  SELECT store_id, bucket_hour, ins, outs, net,
         ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown
    FROM rollup_counting_hour WHERE bucket_hour <> cur_bucket_hour()
  UNION ALL
  SELECT store_id, bucket_hour,
         COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events WHERE bucket_hour = cur_bucket_hour()
   GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW counting_by_bucket_15min AS
  SELECT store_id, bucket_15min, ins, outs, net,
         ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown
    FROM rollup_counting_15min WHERE bucket_15min <> cur_bucket_15min()
  UNION ALL
  SELECT store_id, bucket_15min,
         COUNT(*) FILTER (WHERE direction='in'), COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in') - COUNT(*) FILTER (WHERE direction='out'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='in'  AND height_class(height_m)='unknown'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='adult'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='child'),
         COUNT(*) FILTER (WHERE direction='out' AND height_class(height_m)='unknown')
    FROM count_events WHERE bucket_15min = cur_bucket_15min()
   GROUP BY store_id, bucket_15min;

-- wifi: se AGREGA 'visitors' al final (lo consumen los paneles 60/61).
CREATE OR REPLACE VIEW wifi_ble_by_bucket_day AS
  SELECT store_id, bucket_day, passersby, shoppers, visitors
    FROM rollup_wifi_ble_day WHERE bucket_day <> cur_bucket_day()
  UNION ALL
  SELECT store_id, bucket_day,
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events WHERE bucket_day = cur_bucket_day()
   GROUP BY store_id, bucket_day;

CREATE OR REPLACE VIEW wifi_ble_by_bucket_hour AS
  SELECT store_id, bucket_hour, passersby, shoppers, visitors
    FROM rollup_wifi_ble_hour WHERE bucket_hour <> cur_bucket_hour()
  UNION ALL
  SELECT store_id, bucket_hour,
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events WHERE bucket_hour = cur_bucket_hour()
   GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW wifi_ble_by_bucket_15min AS
  SELECT store_id, bucket_15min, passersby, shoppers, visitors
    FROM rollup_wifi_ble_15min WHERE bucket_15min <> cur_bucket_15min()
  UNION ALL
  SELECT store_id, bucket_15min,
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) = 'shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events WHERE bucket_15min = cur_bucket_15min()
   GROUP BY store_id, bucket_15min;

-- pos
CREATE OR REPLACE VIEW pos_by_bucket_day AS
  SELECT store_id, bucket_day, sales, returns, transactions, items_sale, items_return,
         amount_minor_sale, amount_minor_return, net_amount_minor, currency
    FROM rollup_pos_day WHERE bucket_day <> cur_bucket_day()
  UNION ALL
  SELECT store_id, bucket_day,
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         MAX(currency)
    FROM pos_transactions WHERE bucket_day = cur_bucket_day()
   GROUP BY store_id, bucket_day;

CREATE OR REPLACE VIEW pos_by_bucket_hour AS
  SELECT store_id, bucket_hour, sales, returns, transactions, items_sale, items_return,
         amount_minor_sale, amount_minor_return, net_amount_minor, currency
    FROM rollup_pos_hour WHERE bucket_hour <> cur_bucket_hour()
  UNION ALL
  SELECT store_id, bucket_hour,
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         MAX(currency)
    FROM pos_transactions WHERE bucket_hour = cur_bucket_hour()
   GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW pos_by_bucket_15min AS
  SELECT store_id, bucket_15min, sales, returns, transactions, items_sale, items_return,
         amount_minor_sale, amount_minor_return, net_amount_minor, currency
    FROM rollup_pos_15min WHERE bucket_15min <> cur_bucket_15min()
  UNION ALL
  SELECT store_id, bucket_15min,
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0) - COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         MAX(currency)
    FROM pos_transactions WHERE bucket_15min = cur_bucket_15min()
   GROUP BY store_id, bucket_15min;

-- occupancy_by_bucket_15min: ahora lee la vista base (rollup+live) en vez del raw.
-- Mismas columnas → CREATE OR REPLACE. occupancy_hour/day y visit_duration_* no cambian.
CREATE OR REPLACE VIEW occupancy_by_bucket_15min AS
SELECT store_id, bucket_15min, ins, outs,
       SUM(ins - outs) OVER (
           PARTITION BY store_id, date_trunc('day', bucket_15min) ORDER BY bucket_15min
       ) AS occupancy
FROM counting_by_bucket_15min;

-- ───────────────────────────────────────────────────────────────────────────
-- Vistas nuevas (Fase 2): engagement, RPV, ventas/m²
-- ───────────────────────────────────────────────────────────────────────────
-- Engagement por día (panel "repetición intra-día"). El visitor_hash rota a
-- diario → cada hash vive un solo día → sumar por día sobre el rango es correcto.
CREATE OR REPLACE VIEW wifi_engagement_by_bucket_day AS
  SELECT store_id, bucket_day, windows_bucket, visitors
    FROM rollup_wifi_engagement_day WHERE bucket_day <> cur_bucket_day()
  UNION ALL
  SELECT store_id, bucket_day,
         CASE WHEN nw=1 THEN '1' WHEN nw=2 THEN '2' WHEN nw BETWEEN 3 AND 5 THEN '3-5' ELSE '6+' END,
         COUNT(*)
    FROM (
        SELECT store_id, bucket_day, visitor_hash, COUNT(DISTINCT period_start) AS nw
        FROM wifi_ble_events WHERE bucket_day = cur_bucket_day()
        GROUP BY store_id, bucket_day, visitor_hash
    ) v
   GROUP BY store_id, bucket_day, 3;

-- RPV = facturación ÷ entradas (= conversión × ticket). Por bucket.
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

-- Ventas/m² = facturación neta ÷ superficie de venta (sites.sales_area_m2).
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

-- ───────────────────────────────────────────────────────────────────────────
-- Grants: las vistas base/derivadas repointeadas conservan sus grants (OR REPLACE).
-- Sólo agregamos las vistas NUEVAS al read-only externo. Los rollup_* y
-- rollup_state NO se conceden (internos; las vistas corren con privilegios del owner).
-- ───────────────────────────────────────────────────────────────────────────
GRANT SELECT ON
    wifi_engagement_by_bucket_day,
    revenue_per_visitor_by_bucket_day,
    revenue_per_visitor_by_bucket_hour,
    sales_per_sqm_by_bucket_day,
    sales_per_sqm_by_bucket_hour
TO readonly_external;

-- =============================================================================
-- Post-aplicación (operaciones, fuera de este script idempotente):
--   1. Backfill:   CALL refresh_rollups();
--   2. Scheduler:  pg_cron →  SELECT cron.schedule('refresh-rollups','*/5 * * * *','CALL refresh_rollups()');
--                  (requiere pg_cron en shared_preload_libraries + reboot)
--                  o EventBridge→Lambda invocando CALL refresh_rollups() cada 5 min.
-- =============================================================================

-- =============================================================================
-- Migración: bucketing tz-aware por sucursal (multi-país)
-- =============================================================================
-- event_ts/last_seen_ts se guardan en UTC real (instante). El BUCKET de cada
-- evento se calcula en la ZONA HORARIA de la tienda (sites.timezone) → cada
-- sucursal reporta en su hora local y el sistema soporta sitios de distintos
-- países. Reemplaza el bucketing UTC fijo (que para tiendas no-UTC partía la
-- jornada en la medianoche UTC y rompía ocupación/duración).
--
-- Las columnas GENERATED bucket_* (UTC) en las tablas raw quedan VESTIGIALES
-- (ya no las usa nadie); se dropean en una limpieza posterior. El bucket real
-- vive en los rollup_* (local) y se computa al vuelo en el live-tail.
--
-- Tras aplicar: re-seed demo (instantes UTC) + TRUNCATE rollups + reset watermark
-- + CALL refresh_rollups() (re-backfill local).
-- =============================================================================

-- timezone obligatorio por sitio (default UTC para no romper inserts viejos).
UPDATE sites SET timezone = 'UTC' WHERE timezone IS NULL;
ALTER TABLE sites ALTER COLUMN timezone SET DEFAULT 'UTC';
ALTER TABLE sites ALTER COLUMN timezone SET NOT NULL;

-- Índices para el refresh/live-tail sobre el instante real (los bucket_* se
-- dropean luego; estos los reemplazan).
CREATE INDEX IF NOT EXISTS idx_count_events_store_event    ON count_events    (store_id, event_ts);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_seen  ON wifi_ble_events (store_id, last_seen_ts);
CREATE INDEX IF NOT EXISTS idx_pos_store_event             ON pos_transactions (store_id, event_ts);

-- Helpers: bucket LOCAL de un instante dado el tz IANA de la tienda. Devuelven
-- el bucket como "wall-clock local representado como instante UTC" (hour/15min)
-- o como date local (day) → Grafana en tz=UTC los muestra como hora local, y
-- distintos países alinean por hora local en el mismo eje. STABLE (AT TIME ZONE).
CREATE OR REPLACE FUNCTION lday(ts timestamptz, tz text) RETURNS date LANGUAGE sql STABLE AS $$
    SELECT (ts AT TIME ZONE tz)::date
$$;
CREATE OR REPLACE FUNCTION lhour(ts timestamptz, tz text) RETURNS timestamptz LANGUAGE sql STABLE AS $$
    SELECT to_timestamp(floor(extract(epoch FROM (ts AT TIME ZONE tz)) / 3600) * 3600)
$$;
CREATE OR REPLACE FUNCTION l15(ts timestamptz, tz text) RETURNS timestamptz LANGUAGE sql STABLE AS $$
    SELECT to_timestamp(floor(extract(epoch FROM (ts AT TIME ZONE tz)) / 900) * 900)
$$;

-- Rollups se recomputan con buckets locales → vaciar + resetear watermark.
TRUNCATE rollup_counting_15min, rollup_counting_hour, rollup_counting_day,
         rollup_wifi_ble_15min, rollup_wifi_ble_hour, rollup_wifi_ble_day,
         rollup_wifi_engagement_day, rollup_pos_15min, rollup_pos_hour, rollup_pos_day;
UPDATE rollup_state SET last_refreshed_at = '-infinity';

-- ───────────────────────────────────────────────────────────────────────────
-- refresh_rollups(): tz-aware. Recomputa por watermark; bucket vía sites.timezone.
-- Prefiltro por event_ts/last_seen_ts (rango de lo recibido ±1 día) para usar
-- índice; el IN (store, bucket-local) acota a los buckets realmente afectados.
-- ───────────────────────────────────────────────────────────────────────────
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

-- ───────────────────────────────────────────────────────────────────────────
-- Vistas base: rollup (días/horas cerrados) UNION ALL raw (bucket abierto, local).
-- Cerrado/abierto se decide por el bucket LOCAL actual de cada tienda (join sites).
-- El live-tail prefiltra por event_ts/last_seen_ts >= now()-2d para usar índice.
-- ───────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW counting_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.ins, r.outs, r.net, r.ins_adult, r.ins_child, r.ins_unknown, r.outs_adult, r.outs_child, r.outs_unknown
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
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lday(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW counting_by_bucket_hour AS
  SELECT r.store_id, r.bucket_hour, r.ins, r.outs, r.net, r.ins_adult, r.ins_child, r.ins_unknown, r.outs_adult, r.outs_child, r.outs_unknown
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
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lhour(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW counting_by_bucket_15min AS
  SELECT r.store_id, r.bucket_15min, r.ins, r.outs, r.net, r.ins_adult, r.ins_child, r.ins_unknown, r.outs_adult, r.outs_child, r.outs_unknown
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
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, l15(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW wifi_ble_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.passersby, r.shoppers, r.visitors
    FROM rollup_wifi_ble_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lday(e.last_seen_ts,s.timezone),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max)='shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lday(e.last_seen_ts,s.timezone);

CREATE OR REPLACE VIEW wifi_ble_by_bucket_hour AS
  SELECT r.store_id, r.bucket_hour, r.passersby, r.shoppers, r.visitors
    FROM rollup_wifi_ble_hour r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_hour AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lhour(e.last_seen_ts,s.timezone),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max)='shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lhour(e.last_seen_ts,s.timezone);

CREATE OR REPLACE VIEW wifi_ble_by_bucket_15min AS
  SELECT r.store_id, r.bucket_15min, r.passersby, r.shoppers, r.visitors
    FROM rollup_wifi_ble_15min r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_15min AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, l15(e.last_seen_ts,s.timezone),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max) IN ('passerby','shopper')),
         COUNT(DISTINCT visitor_hash) FILTER (WHERE rssi_class(rssi_max)='shopper'),
         COUNT(DISTINCT visitor_hash)
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
   WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, l15(e.last_seen_ts,s.timezone);

CREATE OR REPLACE VIEW pos_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.sales, r.returns, r.transactions, r.items_sale, r.items_return, r.amount_minor_sale, r.amount_minor_return, r.net_amount_minor, r.currency
    FROM rollup_pos_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lday(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0)-COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0), MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lday(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW pos_by_bucket_hour AS
  SELECT r.store_id, r.bucket_hour, r.sales, r.returns, r.transactions, r.items_sale, r.items_return, r.amount_minor_sale, r.amount_minor_return, r.net_amount_minor, r.currency
    FROM rollup_pos_hour r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_hour AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, lhour(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0)-COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0), MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, lhour(e.event_ts,s.timezone);

CREATE OR REPLACE VIEW pos_by_bucket_15min AS
  SELECT r.store_id, r.bucket_15min, r.sales, r.returns, r.transactions, r.items_sale, r.items_return, r.amount_minor_sale, r.amount_minor_return, r.net_amount_minor, r.currency
    FROM rollup_pos_15min r JOIN sites s ON s.store_id=r.store_id
   WHERE lday((r.bucket_15min AT TIME ZONE 'UTC') AT TIME ZONE s.timezone, s.timezone) <> lday(now(), s.timezone)
  UNION ALL
  SELECT e.store_id, l15(e.event_ts,s.timezone),
         COUNT(*) FILTER (WHERE type='sale'), COUNT(*) FILTER (WHERE type='return'), COUNT(*),
         COALESCE(SUM(items) FILTER (WHERE type='sale'),0), COALESCE(SUM(items) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0), COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0),
         COALESCE(SUM(amount_minor) FILTER (WHERE type='sale'),0)-COALESCE(SUM(amount_minor) FILTER (WHERE type='return'),0), MAX(currency)
    FROM pos_transactions e JOIN sites s ON s.store_id=e.store_id
   WHERE e.event_ts >= now() - INTERVAL '2 days' AND lday(e.event_ts,s.timezone) = lday(now(), s.timezone)
   GROUP BY e.store_id, l15(e.event_ts,s.timezone);

-- engagement: cerrado (rollup) + abierto (raw, local). Idéntico patrón.
CREATE OR REPLACE VIEW wifi_engagement_by_bucket_day AS
  SELECT r.store_id, r.bucket_day, r.windows_bucket, r.visitors
    FROM rollup_wifi_engagement_day r JOIN sites s ON s.store_id=r.store_id
   WHERE r.bucket_day <> lday(now(), s.timezone)
  UNION ALL
  SELECT store_id, bd, CASE WHEN nw=1 THEN '1' WHEN nw=2 THEN '2' WHEN nw BETWEEN 3 AND 5 THEN '3-5' ELSE '6+' END, COUNT(*)
  FROM (
    SELECT e.store_id, lday(e.last_seen_ts,s.timezone) bd, visitor_hash, COUNT(DISTINCT period_start) nw
    FROM wifi_ble_events e JOIN sites s ON s.store_id=e.store_id
    WHERE e.last_seen_ts >= now() - INTERVAL '2 days' AND lday(e.last_seen_ts,s.timezone) = lday(now(), s.timezone)
    GROUP BY e.store_id, lday(e.last_seen_ts,s.timezone), visitor_hash
  ) v GROUP BY store_id, bd, 3;

-- occupancy_by_bucket_15min, turn_in_rate_*, conversion_*, visit_duration_*,
-- metrics_unified_*, revenue_per_visitor_*, sales_per_sqm_*, data_freshness:
-- SIN CAMBIOS — leen las vistas base por nombre (ya tz-aware).

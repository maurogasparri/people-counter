-- Migración: restaurar vistas huérfanas dropeadas por CASCADE de PR 1.
-- Fecha: 2026-05-26
-- Sprint: S10 (capa cloud / fix operativo)
--
-- Bug: la migración 2026-05-26-drop-height-class.sql hizo DROP CASCADE de
-- counting_by_bucket_{15min,hour,day}. Esa CASCADE arrastró las vistas
-- downstream que dependen de counting:
--   - conversion_by_bucket_{15min,hour,day}  (counting + pos)
--   - occupancy_by_bucket_{15min,hour,day}   (counting con window function)
--   - visit_duration_by_bucket_{hour,day}    (depende de occupancy)
--
-- La migración SÓLO recreó las que usaba directamente (counting +
-- metrics_unified). Las otras quedaron huérfanas — el panel "Tasa de
-- conversión" de Grafana fallaba con "relation conversion_by_bucket_day
-- does not exist".
--
-- Este script recrea idempotentemente esas 8 vistas + re-grant a
-- readonly_external. NO dropea nada — sólo CREATE OR REPLACE.
--
-- Lecciones para futuras migraciones:
--   - Cuando hagamos DROP CASCADE de una vista madre, listar TODAS las
--     vistas que se arrastraron (\d+ pre-migración) y recrearlas al final
--     del mismo script.
--   - bootstrap.sql es la single source of truth — copiar de ahí los
--     CREATE OR REPLACE asegura paridad.

BEGIN;

-- =============================================================================
-- Occupancy: ocupación acumulada con cumsum por día
-- =============================================================================

CREATE OR REPLACE VIEW occupancy_by_bucket_15min AS
WITH events_per_bucket AS (
    SELECT store_id, bucket_15min,
           COUNT(*) FILTER (WHERE direction = 'in')  AS ins,
           COUNT(*) FILTER (WHERE direction = 'out') AS outs
    FROM count_events
    GROUP BY store_id, bucket_15min
)
SELECT
    store_id, bucket_15min, ins, outs,
    SUM(ins - outs) OVER (
        PARTITION BY store_id, date_trunc('day', bucket_15min)
        ORDER BY bucket_15min
    ) AS occupancy
FROM events_per_bucket;

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

-- =============================================================================
-- Conversion: ventas / ingresos (counting + pos)
-- =============================================================================

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

-- =============================================================================
-- Visit duration: Ley de Little aplicada a occupancy + arrivals
-- =============================================================================

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
    CASE WHEN SUM(arrivals) > 0
         THEN SUM(avg_occupancy * arrivals) / (SUM(arrivals) / 60.0) / SUM(arrivals)
         ELSE NULL END AS visit_duration_minutes
FROM visit_duration_by_bucket_hour
GROUP BY store_id, date_trunc('day', bucket_hour)::date;

-- =============================================================================
-- Permisos — las vistas recreadas pierden el GRANT, re-aplicar
-- =============================================================================

GRANT SELECT ON
    occupancy_by_bucket_15min,
    occupancy_by_bucket_hour,
    occupancy_by_bucket_day,
    conversion_by_bucket_15min,
    conversion_by_bucket_hour,
    conversion_by_bucket_day,
    visit_duration_by_bucket_hour,
    visit_duration_by_bucket_day
TO readonly_external;

COMMIT;

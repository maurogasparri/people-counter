-- Migración: eliminar columna height_class y centralizar la categorización
-- por altura en una función SQL inmutable.
-- Fecha: 2026-05-26
-- Sprint: S9 (capa cloud / refactor de modelo de datos)
--
-- Cambio de filosofía: en lugar de que el device categorice adulto / niño /
-- desconocido y persista la categoría, el device persiste solo la medición
-- cruda (height_m) y la categorización se aplica en las vistas mediante una
-- función SQL. Ventajas:
--
--   1. Single source of truth: el threshold vive en un solo lugar (la función),
--      modificable con CREATE OR REPLACE y aplicado retroactivo a toda la
--      historia + a todas las vistas que la usan.
--   2. Consistencia entre devices: hoy el threshold venía del config local de
--      cada Pi (adult_min_m), riesgo de drift entre devices de la misma flota.
--      Centralizado en SQL = todos los devices interpretan la altura igual.
--   3. What-if analysis: BI directo puede experimentar con thresholds
--      distintos sin tocar nada (height_class(height_m) usando una función
--      ad-hoc en sus queries).
--   4. Coherencia con el patrón que vamos a aplicar a RSSI (passerby / shopper)
--      en el siguiente PR — el modelo es "guardar crudo, categorizar al
--      consumir".
--
-- El threshold queda en 1.55 m, valor que coincide con el default histórico
-- del config (counter.height_classifier.adult_min_m = 1.55). Si en el futuro
-- se ajusta, una sola línea cambia y se recompute todo el histórico.

BEGIN;

-- =============================================================================
-- 1. Función de categorización por altura
-- =============================================================================
-- IMMUTABLE le permite a Postgres inlinearla en plans de query, sin overhead
-- por llamada. NULL → 'unknown' (cuando el detector no pudo medir profundidad
-- confiable, ej. confidence baja o sanity fuera de rango anthropometric).

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

-- =============================================================================
-- 2. Drop CASCADE de las vistas que dependen de count_events.height_class
-- =============================================================================
-- Las recreamos abajo usando height_class(height_m) en lugar del filtro
-- directo. CASCADE porque metrics_unified_* depende de counting_by_bucket_*.

DROP VIEW IF EXISTS metrics_unified_by_bucket_15min CASCADE;
DROP VIEW IF EXISTS metrics_unified_by_bucket_hour  CASCADE;
DROP VIEW IF EXISTS metrics_unified_by_bucket_day   CASCADE;
DROP VIEW IF EXISTS counting_by_bucket_15min CASCADE;
DROP VIEW IF EXISTS counting_by_bucket_hour  CASCADE;
DROP VIEW IF EXISTS counting_by_bucket_day   CASCADE;

-- =============================================================================
-- 3. Drop de la columna count_events.height_class
-- =============================================================================
-- height_m se mantiene (es la medición cruda). height_class queda derivado
-- vía la función en cada query/vista.

ALTER TABLE count_events DROP COLUMN IF EXISTS height_class;

-- =============================================================================
-- 4. Recrear vistas usando height_class(height_m)
-- =============================================================================

CREATE OR REPLACE VIEW counting_by_bucket_15min AS
SELECT
    store_id,
    bucket_15min,
    COUNT(*) FILTER (WHERE direction = 'in')                                            AS ins,
    COUNT(*) FILTER (WHERE direction = 'out')                                           AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
        - COUNT(*) FILTER (WHERE direction = 'out')                                     AS net,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'adult')      AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'child')      AS ins_child,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'unknown')    AS ins_unknown,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'adult')      AS outs_adult,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'child')      AS outs_child,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'unknown')    AS outs_unknown
FROM count_events
GROUP BY store_id, bucket_15min;

CREATE OR REPLACE VIEW counting_by_bucket_hour AS
SELECT
    store_id,
    bucket_hour,
    COUNT(*) FILTER (WHERE direction = 'in')                                            AS ins,
    COUNT(*) FILTER (WHERE direction = 'out')                                           AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
        - COUNT(*) FILTER (WHERE direction = 'out')                                     AS net,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'adult')      AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'child')      AS ins_child,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'unknown')    AS ins_unknown,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'adult')      AS outs_adult,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'child')      AS outs_child,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'unknown')    AS outs_unknown
FROM count_events
GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW counting_by_bucket_day AS
SELECT
    store_id,
    bucket_day,
    COUNT(*) FILTER (WHERE direction = 'in')                                            AS ins,
    COUNT(*) FILTER (WHERE direction = 'out')                                           AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
        - COUNT(*) FILTER (WHERE direction = 'out')                                     AS net,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'adult')      AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'child')      AS ins_child,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class(height_m) = 'unknown')    AS ins_unknown,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'adult')      AS outs_adult,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'child')      AS outs_child,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class(height_m) = 'unknown')    AS outs_unknown
FROM count_events
GROUP BY store_id, bucket_day;

-- COMMENTs en español (replicados de la migración anterior).
COMMENT ON VIEW counting_by_bucket_15min IS
    'Ingresos y egresos al local agrupados por sucursal y franjas de 15 minutos, con desglose demográfico (adulto / niño / desconocido). La categorización se calcula vía la función height_class() — modificable centralmente sin tocar las vistas.';
COMMENT ON COLUMN counting_by_bucket_15min.ins  IS 'Personas que ingresaron al local en la franja.';
COMMENT ON COLUMN counting_by_bucket_15min.outs IS 'Personas que egresaron del local en la franja.';
COMMENT ON COLUMN counting_by_bucket_15min.net  IS 'Diferencia ins menos outs. Positivo cuando entran más personas de las que salen.';
COMMENT ON COLUMN counting_by_bucket_15min.ins_unknown IS 'Ingresos sin clasificación demográfica confiable (height_m NULL).';

COMMENT ON VIEW counting_by_bucket_hour IS
    'Ingresos y egresos agregados por hora con desglose demográfico. Mismas columnas que counting_by_bucket_15min.';
COMMENT ON VIEW counting_by_bucket_day  IS
    'Ingresos y egresos agregados por día con desglose demográfico. Mismas columnas que counting_by_bucket_15min.';

-- =============================================================================
-- 5. Recrear vistas unificadas (metrics_unified_*)
-- =============================================================================
-- Mismo shape de columnas que antes — el cambio es interno, los consumidores
-- (Lambda query_aggregates, partner BI) no necesitan modificar sus queries.

CREATE OR REPLACE VIEW metrics_unified_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id)            AS store_id,
    COALESCE(c.bucket_15min, w.bucket_15min, p.bucket_15min) AS bucket_15min,
    COALESCE(c.ins, 0)                  AS ins,
    COALESCE(c.outs, 0)                 AS outs,
    COALESCE(c.net, 0)                  AS net,
    COALESCE(c.ins_adult, 0)            AS ins_adult,
    COALESCE(c.ins_child, 0)            AS ins_child,
    COALESCE(c.ins_unknown, 0)          AS ins_unknown,
    COALESCE(c.outs_adult, 0)           AS outs_adult,
    COALESCE(c.outs_child, 0)           AS outs_child,
    COALESCE(c.outs_unknown, 0)         AS outs_unknown,
    COALESCE(w.passersby, 0)            AS passersby,
    COALESCE(w.shoppers, 0)             AS shoppers,
    COALESCE(p.sales, 0)                AS sales,
    COALESCE(p.returns, 0)              AS returns,
    COALESCE(p.transactions, 0)         AS transactions,
    COALESCE(p.items_sale, 0)           AS items_sale,
    COALESCE(p.items_return, 0)         AS items_return,
    COALESCE(p.amount_minor_sale, 0)    AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0)  AS amount_minor_return,
    COALESCE(p.currency, 'ARS')         AS currency
FROM counting_by_bucket_15min c
FULL OUTER JOIN wifi_ble_by_bucket_15min w
    ON c.store_id = w.store_id AND c.bucket_15min = w.bucket_15min
FULL OUTER JOIN pos_by_bucket_15min p
    ON COALESCE(c.store_id, w.store_id) = p.store_id
   AND COALESCE(c.bucket_15min, w.bucket_15min) = p.bucket_15min;

CREATE OR REPLACE VIEW metrics_unified_by_bucket_hour AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id)        AS store_id,
    COALESCE(c.bucket_hour, w.bucket_hour, p.bucket_hour) AS bucket_hour,
    COALESCE(c.ins, 0)                  AS ins,
    COALESCE(c.outs, 0)                 AS outs,
    COALESCE(c.net, 0)                  AS net,
    COALESCE(c.ins_adult, 0)            AS ins_adult,
    COALESCE(c.ins_child, 0)            AS ins_child,
    COALESCE(c.ins_unknown, 0)          AS ins_unknown,
    COALESCE(c.outs_adult, 0)           AS outs_adult,
    COALESCE(c.outs_child, 0)           AS outs_child,
    COALESCE(c.outs_unknown, 0)         AS outs_unknown,
    COALESCE(w.passersby, 0)            AS passersby,
    COALESCE(w.shoppers, 0)             AS shoppers,
    COALESCE(p.sales, 0)                AS sales,
    COALESCE(p.returns, 0)              AS returns,
    COALESCE(p.transactions, 0)         AS transactions,
    COALESCE(p.items_sale, 0)           AS items_sale,
    COALESCE(p.items_return, 0)         AS items_return,
    COALESCE(p.amount_minor_sale, 0)    AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0)  AS amount_minor_return,
    COALESCE(p.currency, 'ARS')         AS currency
FROM counting_by_bucket_hour c
FULL OUTER JOIN wifi_ble_by_bucket_hour w
    ON c.store_id = w.store_id AND c.bucket_hour = w.bucket_hour
FULL OUTER JOIN pos_by_bucket_hour p
    ON COALESCE(c.store_id, w.store_id) = p.store_id
   AND COALESCE(c.bucket_hour, w.bucket_hour) = p.bucket_hour;

CREATE OR REPLACE VIEW metrics_unified_by_bucket_day AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id)        AS store_id,
    COALESCE(c.bucket_day, w.bucket_day, p.bucket_day)  AS bucket_day,
    COALESCE(c.ins, 0)                  AS ins,
    COALESCE(c.outs, 0)                 AS outs,
    COALESCE(c.net, 0)                  AS net,
    COALESCE(c.ins_adult, 0)            AS ins_adult,
    COALESCE(c.ins_child, 0)            AS ins_child,
    COALESCE(c.ins_unknown, 0)          AS ins_unknown,
    COALESCE(c.outs_adult, 0)           AS outs_adult,
    COALESCE(c.outs_child, 0)           AS outs_child,
    COALESCE(c.outs_unknown, 0)         AS outs_unknown,
    COALESCE(w.passersby, 0)            AS passersby,
    COALESCE(w.shoppers, 0)             AS shoppers,
    COALESCE(p.sales, 0)                AS sales,
    COALESCE(p.returns, 0)              AS returns,
    COALESCE(p.transactions, 0)         AS transactions,
    COALESCE(p.items_sale, 0)           AS items_sale,
    COALESCE(p.items_return, 0)         AS items_return,
    COALESCE(p.amount_minor_sale, 0)    AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0)  AS amount_minor_return,
    COALESCE(p.currency, 'ARS')         AS currency
FROM counting_by_bucket_day c
FULL OUTER JOIN wifi_ble_by_bucket_day w
    ON c.store_id = w.store_id AND c.bucket_day = w.bucket_day
FULL OUTER JOIN pos_by_bucket_day p
    ON COALESCE(c.store_id, w.store_id) = p.store_id
   AND COALESCE(c.bucket_day, w.bucket_day) = p.bucket_day;

COMMENT ON VIEW metrics_unified_by_bucket_15min IS
    'Vista unificada: counting, wifi_ble y pos combinados mediante FULL OUTER JOIN por sucursal y franja de 15 minutos. Consumida por la Lambda query_aggregates. Aplica COALESCE a cero en todas las métricas para garantizar un formato uniforme de salida.';
COMMENT ON VIEW metrics_unified_by_bucket_hour  IS 'Vista unificada por hora. Mismas columnas que metrics_unified_by_bucket_15min.';
COMMENT ON VIEW metrics_unified_by_bucket_day   IS 'Vista unificada por día. Mismas columnas que metrics_unified_by_bucket_15min.';

-- =============================================================================
-- 6. Permisos
-- =============================================================================
-- La función height_class la usan las vistas, NO los roles directamente.
-- Pero los roles que consultan las vistas necesitan EXECUTE para que el
-- planner pueda invocarla. Por default las funciones IMMUTABLE en SQL son
-- ejecutables por PUBLIC, pero somos explícitos para que el modelo de
-- permisos no dependa de defaults.

GRANT EXECUTE ON FUNCTION height_class(REAL) TO lambda_query_reader, readonly_external;

-- Re-grant de las vistas recreadas. Las vistas que dropearon en CASCADE
-- requieren re-GRANT (Postgres no lo mantiene tras DROP/CREATE).

GRANT SELECT ON
    counting_by_bucket_15min,
    counting_by_bucket_hour,
    counting_by_bucket_day
TO readonly_external;

GRANT SELECT ON
    metrics_unified_by_bucket_15min,
    metrics_unified_by_bucket_hour,
    metrics_unified_by_bucket_day
TO lambda_query_reader;

COMMIT;

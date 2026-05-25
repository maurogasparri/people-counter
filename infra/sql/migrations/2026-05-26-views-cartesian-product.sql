-- Migración: producto cartesiano de vistas + scoping de lambda_query_reader
-- Fecha: 2026-05-26
-- Sprint: S10 (visualización analítica)
--
-- Reorganiza la capa de vistas con dos objetivos:
--
-- 1) Naming consistente <métrica>_by_bucket_<grain> para todas las
--    combinaciones (fact × bucket). Cubre 15min, hour y day para counting,
--    wifi_ble, pos, occupancy, turn_in_rate, conversion. Visit duration
--    solo hour y day (15min es muy ruidoso para Little's Law).
--
-- 2) Tres vistas unificadas (metrics_unified_by_bucket_<grano>) que combinan
--    counting, wifi_ble y pos en cada granularidad. Las consume la Lambda
--    query_aggregates (luego del refactor) y queda como punto de entrada
--    limpio para consultas BI ad-hoc que quieran toda la data en un único
--    objeto.
--
-- 3) data_freshness_by_store con el último timestamp de ingesta cruzando las
--    tres fuentes por sucursal. Lo consume la Lambda para el campo
--    "data_freshness" del response del API.
--
-- 4) Eliminación completa de las vistas heredadas (counting_by_bucket,
--    counting_hourly, counting_daily, turn_in_rate_by_bucket,
--    conversion_rate_by_store, conversion_rate_hourly, conversion_rate_daily,
--    wifi_ble_store_traffic). NO se conservan como alias: confirmado fuera de
--    alcance para el PoC porque todavía no hay consumidores en producción.
--
-- 5) Restricción de permisos:
--    * lambda_query_reader pierde acceso a las tablas crudas (count_events,
--      wifi_ble_summary, pos_transactions). Solo accede a las vistas
--      unificadas, data_freshness y sites. Simétrico con readonly_external.
--    * readonly_external recibe SELECT sobre TODAS las vistas del producto
--      cartesiano (excepto las unificadas, que son de uso interno de la
--      Lambda; el partner externo prefiere las vistas individuales por
--      fuente, más explícitas en lo que consulta).
--
-- Se agregan COMMENT ON VIEW y COMMENT ON COLUMN en cada vista nueva para
-- que el consumidor SQL directo (DBeaver, psql, DataGrip) tenga
-- documentación inline sin necesidad de abrir el repositorio.

BEGIN;

-- =============================================================================
-- 1. Eliminación en cascada de vistas heredadas
-- =============================================================================
-- Se usa CASCADE porque hay dependencias internas:
--   conversion_rate_hourly y conversion_rate_daily dependen de
--   conversion_rate_by_store; turn_in_rate_by_bucket depende de
--   wifi_ble_store_traffic.

DROP VIEW IF EXISTS counting_by_bucket           CASCADE;
DROP VIEW IF EXISTS counting_hourly              CASCADE;
DROP VIEW IF EXISTS counting_daily               CASCADE;
DROP VIEW IF EXISTS turn_in_rate_by_bucket       CASCADE;
DROP VIEW IF EXISTS conversion_rate_by_store     CASCADE;
DROP VIEW IF EXISTS conversion_rate_hourly       CASCADE;
DROP VIEW IF EXISTS conversion_rate_daily        CASCADE;
DROP VIEW IF EXISTS wifi_ble_store_traffic       CASCADE;

-- =============================================================================
-- 2. Índices auxiliares para los agregados horarios
-- =============================================================================
-- Las vistas que agrupan por bucket_hour (counting_by_bucket_hour,
-- pos_by_bucket_hour, etc.) hoy no tienen índice propio: la consulta
-- escanea por bucket_15min y vuelve a agrupar, lo cual es funcional pero
-- subóptimo. Con un índice directo sobre bucket_hour los agregados
-- horarios se aceleran del orden de 5 a 10 veces cuando hay cientos de
-- miles de eventos.

CREATE INDEX IF NOT EXISTS idx_count_events_store_bucket_hour
    ON count_events (store_id, bucket_hour DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_bucket_hour
    ON wifi_ble_summary (store_id, bucket_hour DESC);
CREATE INDEX IF NOT EXISTS idx_pos_store_bucket_hour
    ON pos_transactions (store_id, bucket_hour DESC);

-- =============================================================================
-- 3. COUNTING — counting_by_bucket_15min / _hour / _day
-- =============================================================================
-- Incluye desglose demográfico (adulto / niño / desconocido) en TODAS las
-- granularidades, no solamente en la diaria como hacía la vista heredada.
-- La categoría 'desconocido' agrupa los registros con height_class NULL y
-- los etiquetados explícitamente como 'unknown': para el consumidor son
-- semánticamente equivalentes ("no tenemos el dato").

CREATE OR REPLACE VIEW counting_by_bucket_15min AS
SELECT
    store_id,
    bucket_15min,
    COUNT(*) FILTER (WHERE direction = 'in')                                            AS ins,
    COUNT(*) FILTER (WHERE direction = 'out')                                           AS outs,
    COUNT(*) FILTER (WHERE direction = 'in')
        - COUNT(*) FILTER (WHERE direction = 'out')                                     AS net,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class = 'adult')                AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class = 'child')                AS ins_child,
    COUNT(*) FILTER (WHERE direction = 'in'
                       AND (height_class IS NULL OR height_class = 'unknown'))          AS ins_unknown,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class = 'adult')                AS outs_adult,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class = 'child')                AS outs_child,
    COUNT(*) FILTER (WHERE direction = 'out'
                       AND (height_class IS NULL OR height_class = 'unknown'))          AS outs_unknown
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
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class = 'adult')                AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class = 'child')                AS ins_child,
    COUNT(*) FILTER (WHERE direction = 'in'
                       AND (height_class IS NULL OR height_class = 'unknown'))          AS ins_unknown,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class = 'adult')                AS outs_adult,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class = 'child')                AS outs_child,
    COUNT(*) FILTER (WHERE direction = 'out'
                       AND (height_class IS NULL OR height_class = 'unknown'))          AS outs_unknown
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
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class = 'adult')                AS ins_adult,
    COUNT(*) FILTER (WHERE direction = 'in'  AND height_class = 'child')                AS ins_child,
    COUNT(*) FILTER (WHERE direction = 'in'
                       AND (height_class IS NULL OR height_class = 'unknown'))          AS ins_unknown,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class = 'adult')                AS outs_adult,
    COUNT(*) FILTER (WHERE direction = 'out' AND height_class = 'child')                AS outs_child,
    COUNT(*) FILTER (WHERE direction = 'out'
                       AND (height_class IS NULL OR height_class = 'unknown'))          AS outs_unknown
FROM count_events
GROUP BY store_id, bucket_day;

-- =============================================================================
-- 4. WIFI/BLE — wifi_ble_by_bucket_15min / _hour / _day
-- =============================================================================
-- La granularidad de 15 minutos toma el máximo entre cámaras para evitar
-- doble conteo en sucursales multi-cámara (reemplaza la vista heredada
-- wifi_ble_store_traffic). Las granularidades horaria y diaria suman los
-- buckets de 15 minutos bajo el supuesto de que cada bucket sucesivo
-- representa tráfico distinto (gente que pasa). Advertencia conocida: un
-- dispositivo parado frente al local durante varios buckets aporta a cada
-- uno y produce sobreconteo; es aceptable para el PoC y queda documentado
-- en el TFG.

CREATE OR REPLACE VIEW wifi_ble_by_bucket_15min AS
SELECT
    store_id,
    bucket_15min,
    MAX(passersby) AS passersby,
    MAX(shoppers)  AS shoppers,
    COUNT(*)       AS cams_reporting
FROM wifi_ble_summary
GROUP BY store_id, bucket_15min;

CREATE OR REPLACE VIEW wifi_ble_by_bucket_hour AS
SELECT
    store_id,
    bucket_hour,
    SUM(passersby_max) AS passersby,
    SUM(shoppers_max)  AS shoppers
FROM (
    -- Subconsulta con MAX intra-bucket (dedup entre cámaras) antes del
    -- SUM al agrupar por hora.
    SELECT
        store_id,
        bucket_15min,
        date_trunc('hour', bucket_15min) AS bucket_hour,
        MAX(passersby) AS passersby_max,
        MAX(shoppers)  AS shoppers_max
    FROM wifi_ble_summary
    GROUP BY store_id, bucket_15min, date_trunc('hour', bucket_15min)
) AS per_15min
GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW wifi_ble_by_bucket_day AS
SELECT
    store_id,
    bucket_day,
    SUM(passersby_max) AS passersby,
    SUM(shoppers_max)  AS shoppers
FROM (
    SELECT
        store_id,
        bucket_15min,
        date_trunc('day', bucket_15min)::date AS bucket_day,
        MAX(passersby) AS passersby_max,
        MAX(shoppers)  AS shoppers_max
    FROM wifi_ble_summary
    GROUP BY store_id, bucket_15min, date_trunc('day', bucket_15min)::date
) AS per_15min
GROUP BY store_id, bucket_day;

-- =============================================================================
-- 5. POS — pos_by_bucket_15min / _hour / _day
-- =============================================================================

CREATE OR REPLACE VIEW pos_by_bucket_15min AS
SELECT
    store_id,
    bucket_15min,
    COUNT(*) FILTER (WHERE type = 'sale')                                       AS sales,
    COUNT(*) FILTER (WHERE type = 'return')                                     AS returns,
    COUNT(*)                                                                    AS transactions,
    COALESCE(SUM(items)        FILTER (WHERE type = 'sale'),   0)               AS items_sale,
    COALESCE(SUM(items)        FILTER (WHERE type = 'return'), 0)               AS items_return,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0)               AS amount_minor_sale,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0)               AS amount_minor_return,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0)
        - COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0)         AS net_amount_minor,
    MAX(currency)                                                               AS currency
FROM pos_transactions
GROUP BY store_id, bucket_15min;

CREATE OR REPLACE VIEW pos_by_bucket_hour AS
SELECT
    store_id,
    bucket_hour,
    COUNT(*) FILTER (WHERE type = 'sale')                                       AS sales,
    COUNT(*) FILTER (WHERE type = 'return')                                     AS returns,
    COUNT(*)                                                                    AS transactions,
    COALESCE(SUM(items)        FILTER (WHERE type = 'sale'),   0)               AS items_sale,
    COALESCE(SUM(items)        FILTER (WHERE type = 'return'), 0)               AS items_return,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0)               AS amount_minor_sale,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0)               AS amount_minor_return,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0)
        - COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0)         AS net_amount_minor,
    MAX(currency)                                                               AS currency
FROM pos_transactions
GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW pos_by_bucket_day AS
SELECT
    store_id,
    bucket_day,
    COUNT(*) FILTER (WHERE type = 'sale')                                       AS sales,
    COUNT(*) FILTER (WHERE type = 'return')                                     AS returns,
    COUNT(*)                                                                    AS transactions,
    COALESCE(SUM(items)        FILTER (WHERE type = 'sale'),   0)               AS items_sale,
    COALESCE(SUM(items)        FILTER (WHERE type = 'return'), 0)               AS items_return,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0)               AS amount_minor_sale,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0)               AS amount_minor_return,
    COALESCE(SUM(amount_minor) FILTER (WHERE type = 'sale'),   0)
        - COALESCE(SUM(amount_minor) FILTER (WHERE type = 'return'), 0)         AS net_amount_minor,
    MAX(currency)                                                               AS currency
FROM pos_transactions
GROUP BY store_id, bucket_day;

-- =============================================================================
-- 6. OCUPACIÓN — occupancy_by_bucket_15min / _hour / _day
-- =============================================================================
-- Ocupación acumulada (suma corrida de ingresos menos egresos) por sucursal,
-- reiniciada cada día a medianoche. La base de 15 minutos se construye con
-- una función de ventana sobre count_events; las granularidades horaria y
-- diaria agregan estadísticas (promedio, máximo y mínimo) sobre los buckets
-- de 15 minutos contenidos en cada hora o día. El promedio es la entrada
-- de la Ley de Little; el máximo permite detectar picos de aforo.

CREATE OR REPLACE VIEW occupancy_by_bucket_15min AS
WITH events_per_bucket AS (
    SELECT
        store_id,
        bucket_15min,
        COUNT(*) FILTER (WHERE direction = 'in')  AS ins,
        COUNT(*) FILTER (WHERE direction = 'out') AS outs
    FROM count_events
    GROUP BY store_id, bucket_15min
)
SELECT
    store_id,
    bucket_15min,
    ins,
    outs,
    -- Suma corrida desde el inicio del día. Se reinicia a medianoche.
    -- En condiciones ideales (sin pérdida de detección) vuelve a cero al
    -- cerrar el local. La desviación acumulada al cierre representa
    -- ingresos o egresos perdidos por el detector visual.
    SUM(ins - outs) OVER (
        PARTITION BY store_id, date_trunc('day', bucket_15min)
        ORDER BY bucket_15min
    ) AS occupancy
FROM events_per_bucket;

CREATE OR REPLACE VIEW occupancy_by_bucket_hour AS
SELECT
    store_id,
    date_trunc('hour', bucket_15min) AS bucket_hour,
    SUM(ins)                                AS ins,
    SUM(outs)                               AS outs,
    AVG(occupancy)::numeric(10, 2)          AS avg_occupancy,
    MAX(occupancy)                          AS max_occupancy,
    MIN(occupancy)                          AS min_occupancy
FROM occupancy_by_bucket_15min
GROUP BY store_id, date_trunc('hour', bucket_15min);

CREATE OR REPLACE VIEW occupancy_by_bucket_day AS
SELECT
    store_id,
    date_trunc('day', bucket_15min)::date AS bucket_day,
    SUM(ins)                                AS ins,
    SUM(outs)                               AS outs,
    AVG(occupancy)::numeric(10, 2)          AS avg_occupancy,
    MAX(occupancy)                          AS max_occupancy,
    MIN(occupancy)                          AS min_occupancy
FROM occupancy_by_bucket_15min
GROUP BY store_id, date_trunc('day', bucket_15min)::date;

-- =============================================================================
-- 7. TASA DE CAPTACIÓN — turn_in_rate_by_bucket_15min / _hour / _day
-- =============================================================================
-- Combina counting (ingresos) con wifi_ble (passersby, shoppers). El FULL
-- OUTER JOIN preserva los buckets donde solamente una fuente reportó datos.
-- Definiciones:
--   turn_in_rate          = ingresos / passersby (porcentaje del tráfico
--                           exterior que efectivamente entra al local).
--   turn_in_shoppers_rate = ingresos / shoppers  (porcentaje del tráfico
--                           cercano que entra).
-- Devuelve NULL cuando el denominador es cero (evita división por cero).

CREATE OR REPLACE VIEW turn_in_rate_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, w.store_id)            AS store_id,
    COALESCE(c.bucket_15min, w.bucket_15min)    AS bucket_15min,
    COALESCE(c.ins, 0)                          AS ins,
    COALESCE(w.passersby, 0)                    AS passersby,
    COALESCE(w.shoppers, 0)                     AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0
         THEN c.ins::float / w.passersby
         ELSE NULL END                          AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0) > 0
         THEN c.ins::float / w.shoppers
         ELSE NULL END                          AS turn_in_shoppers_rate
FROM counting_by_bucket_15min c
FULL OUTER JOIN wifi_ble_by_bucket_15min w
    ON c.store_id = w.store_id AND c.bucket_15min = w.bucket_15min;

CREATE OR REPLACE VIEW turn_in_rate_by_bucket_hour AS
SELECT
    COALESCE(c.store_id, w.store_id)            AS store_id,
    COALESCE(c.bucket_hour, w.bucket_hour)      AS bucket_hour,
    COALESCE(c.ins, 0)                          AS ins,
    COALESCE(w.passersby, 0)                    AS passersby,
    COALESCE(w.shoppers, 0)                     AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0
         THEN c.ins::float / w.passersby
         ELSE NULL END                          AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0) > 0
         THEN c.ins::float / w.shoppers
         ELSE NULL END                          AS turn_in_shoppers_rate
FROM counting_by_bucket_hour c
FULL OUTER JOIN wifi_ble_by_bucket_hour w
    ON c.store_id = w.store_id AND c.bucket_hour = w.bucket_hour;

CREATE OR REPLACE VIEW turn_in_rate_by_bucket_day AS
SELECT
    COALESCE(c.store_id, w.store_id)            AS store_id,
    COALESCE(c.bucket_day, w.bucket_day)        AS bucket_day,
    COALESCE(c.ins, 0)                          AS ins,
    COALESCE(w.passersby, 0)                    AS passersby,
    COALESCE(w.shoppers, 0)                     AS shoppers,
    CASE WHEN COALESCE(w.passersby, 0) > 0
         THEN c.ins::float / w.passersby
         ELSE NULL END                          AS turn_in_rate,
    CASE WHEN COALESCE(w.shoppers, 0) > 0
         THEN c.ins::float / w.shoppers
         ELSE NULL END                          AS turn_in_shoppers_rate
FROM counting_by_bucket_day c
FULL OUTER JOIN wifi_ble_by_bucket_day w
    ON c.store_id = w.store_id AND c.bucket_day = w.bucket_day;

-- =============================================================================
-- 8. CONVERSIÓN — conversion_by_bucket_15min / _hour / _day
-- =============================================================================
-- conversion_rate = ventas / ingresos (donde ingresos proviene del contador
-- visual). Devuelve NULL cuando ingresos es cero (sucursal cerrada o sin
-- tráfico en la franja).

CREATE OR REPLACE VIEW conversion_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, p.store_id)         AS store_id,
    COALESCE(c.bucket_15min, p.bucket_15min) AS bucket_15min,
    COALESCE(c.ins, 0)                       AS visits,
    COALESCE(p.sales, 0)                     AS sales,
    COALESCE(p.returns, 0)                   AS returns,
    COALESCE(p.items_sale, 0)                AS items_sale,
    COALESCE(p.items_return, 0)              AS items_return,
    COALESCE(p.amount_minor_sale, 0)         AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0)       AS amount_minor_return,
    COALESCE(p.net_amount_minor, 0)          AS net_amount_minor,
    COALESCE(p.currency, 'ARS')              AS currency,
    CASE WHEN COALESCE(c.ins, 0) > 0
         THEN p.sales::float / c.ins
         ELSE NULL END                       AS conversion_rate
FROM counting_by_bucket_15min c
FULL OUTER JOIN pos_by_bucket_15min p
    ON c.store_id = p.store_id AND c.bucket_15min = p.bucket_15min;

CREATE OR REPLACE VIEW conversion_by_bucket_hour AS
SELECT
    COALESCE(c.store_id, p.store_id)         AS store_id,
    COALESCE(c.bucket_hour, p.bucket_hour)   AS bucket_hour,
    COALESCE(c.ins, 0)                       AS visits,
    COALESCE(p.sales, 0)                     AS sales,
    COALESCE(p.returns, 0)                   AS returns,
    COALESCE(p.items_sale, 0)                AS items_sale,
    COALESCE(p.items_return, 0)              AS items_return,
    COALESCE(p.amount_minor_sale, 0)         AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0)       AS amount_minor_return,
    COALESCE(p.net_amount_minor, 0)          AS net_amount_minor,
    COALESCE(p.currency, 'ARS')              AS currency,
    CASE WHEN COALESCE(c.ins, 0) > 0
         THEN p.sales::float / c.ins
         ELSE NULL END                       AS conversion_rate
FROM counting_by_bucket_hour c
FULL OUTER JOIN pos_by_bucket_hour p
    ON c.store_id = p.store_id AND c.bucket_hour = p.bucket_hour;

CREATE OR REPLACE VIEW conversion_by_bucket_day AS
SELECT
    COALESCE(c.store_id, p.store_id)         AS store_id,
    COALESCE(c.bucket_day, p.bucket_day)     AS bucket_day,
    COALESCE(c.ins, 0)                       AS visits,
    COALESCE(p.sales, 0)                     AS sales,
    COALESCE(p.returns, 0)                   AS returns,
    COALESCE(p.items_sale, 0)                AS items_sale,
    COALESCE(p.items_return, 0)              AS items_return,
    COALESCE(p.amount_minor_sale, 0)         AS amount_minor_sale,
    COALESCE(p.amount_minor_return, 0)       AS amount_minor_return,
    COALESCE(p.net_amount_minor, 0)          AS net_amount_minor,
    COALESCE(p.currency, 'ARS')              AS currency,
    CASE WHEN COALESCE(c.ins, 0) > 0
         THEN p.sales::float / c.ins
         ELSE NULL END                       AS conversion_rate
FROM counting_by_bucket_day c
FULL OUTER JOIN pos_by_bucket_day p
    ON c.store_id = p.store_id AND c.bucket_day = p.bucket_day;

-- =============================================================================
-- 9. DURACIÓN DE VISITA — visit_duration_by_bucket_hour / _day (Ley de Little)
-- =============================================================================
-- Aplicación de la Ley de Little: W = L / λ
--   W = duración promedio de visita, en minutos.
--   L = ocupación promedio durante la ventana.
--   λ = tasa de arribo (ingresos por unidad de tiempo).
--
-- Devuelve NULL cuando no hubo ingresos (madrugada, post-cierre).
-- Advertencias:
--   * Asume estado estacionario; se degrada en horas pico del mediodía y
--     en las transiciones de apertura y cierre del local.
--   * Solo da el promedio: no se obtienen percentiles (p50, p95).
--   * Para tener la distribución haría falta seguimiento por visitante,
--     no soportado por restricciones de privacidad (el hash WiFi/BLE se
--     rota cada jornada).
--
-- Granularidad mínima: hora. En 15 minutos la muestra es muy chica y la
-- Ley de Little degenera por la baja tasa de arribos.

CREATE OR REPLACE VIEW visit_duration_by_bucket_hour AS
SELECT
    store_id,
    bucket_hour,
    avg_occupancy,
    ins                                                                     AS arrivals,
    -- λ en personas por minuto (ingresos por hora dividido 60). W = L / λ
    -- queda expresado en minutos.
    CASE WHEN ins > 0
         THEN avg_occupancy / (ins / 60.0)
         ELSE NULL END                                                      AS visit_duration_minutes
FROM occupancy_by_bucket_hour;

CREATE OR REPLACE VIEW visit_duration_by_bucket_day AS
-- Promedio diario de los W horarios, ponderado por la cantidad de
-- ingresos. El promedio ponderado es más estable que un AVG simple, que
-- daría el mismo peso a una hora con un solo ingreso que a una hora pico
-- con varios cientos.
SELECT
    store_id,
    date_trunc('day', bucket_hour)::date                                    AS bucket_day,
    SUM(arrivals)                                                            AS arrivals,
    AVG(avg_occupancy)::numeric(10, 2)                                      AS avg_occupancy,
    CASE WHEN SUM(arrivals) > 0
         THEN SUM(avg_occupancy * arrivals) / (SUM(arrivals) / 60.0)
              / SUM(arrivals)  -- weighted avg of W
         ELSE NULL END                                                      AS visit_duration_minutes
FROM visit_duration_by_bucket_hour
GROUP BY store_id, date_trunc('day', bucket_hour)::date;

-- =============================================================================
-- 10. VISTAS UNIFICADAS — counting + wifi_ble + pos combinados
-- =============================================================================
-- Pensadas para la Lambda query_aggregates y para consumidores BI que
-- quieren toda la data en una sola fila por (sucursal, bucket). El FULL
-- OUTER JOIN preserva los buckets donde solamente una fuente reportó. El
-- COALESCE garantiza que ninguna métrica de conteo devuelva NULL (el
-- cliente espera enteros, no opcionales).

CREATE OR REPLACE VIEW metrics_unified_by_bucket_15min AS
SELECT
    COALESCE(c.store_id, w.store_id, p.store_id)            AS store_id,
    COALESCE(c.bucket_15min, w.bucket_15min, p.bucket_15min) AS bucket_15min,
    -- counting con demografía
    COALESCE(c.ins, 0)                  AS ins,
    COALESCE(c.outs, 0)                 AS outs,
    COALESCE(c.net, 0)                  AS net,
    COALESCE(c.ins_adult, 0)            AS ins_adult,
    COALESCE(c.ins_child, 0)            AS ins_child,
    COALESCE(c.ins_unknown, 0)          AS ins_unknown,
    COALESCE(c.outs_adult, 0)           AS outs_adult,
    COALESCE(c.outs_child, 0)           AS outs_child,
    COALESCE(c.outs_unknown, 0)         AS outs_unknown,
    -- wifi/ble
    COALESCE(w.passersby, 0)            AS passersby,
    COALESCE(w.shoppers, 0)             AS shoppers,
    -- pos
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

-- =============================================================================
-- 11. FRESCURA DEL DATO — último timestamp de ingesta por sucursal
-- =============================================================================
-- Provee la información del campo "data_freshness" del API. Devuelve el
-- último timestamp de ingesta de cada sucursal cruzando las tres fuentes
-- (counting, wifi_ble, pos). Permite al cliente detectar dispositivos
-- desconectados sin necesidad de un endpoint adicional.

CREATE OR REPLACE VIEW data_freshness_by_store AS
SELECT
    store_id,
    MAX(last_received_at) AS last_received_at
FROM (
    SELECT store_id, MAX(received_at) AS last_received_at
        FROM count_events GROUP BY store_id
    UNION ALL
    SELECT store_id, MAX(received_at)
        FROM wifi_ble_summary GROUP BY store_id
    UNION ALL
    SELECT store_id, MAX(received_at)
        FROM pos_transactions GROUP BY store_id
) u
GROUP BY store_id;

-- =============================================================================
-- 12. COMMENTS — documentación inline visible desde clientes SQL
-- =============================================================================
-- Estos COMMENT ON quedan visibles para cualquier cliente SQL (DBeaver,
-- psql con \d+, DataGrip). Cubren las vistas y las columnas más
-- significativas. Para columnas obvias por nombre (store_id, bucket_15min,
-- etc.) no se documenta para evitar ruido redundante.

COMMENT ON VIEW counting_by_bucket_15min IS
    'Ingresos y egresos al local agrupados por sucursal y franjas de 15 minutos, con desglose demográfico (adulto / niño / desconocido).';
COMMENT ON COLUMN counting_by_bucket_15min.ins  IS 'Personas que ingresaron al local en la franja.';
COMMENT ON COLUMN counting_by_bucket_15min.outs IS 'Personas que egresaron del local en la franja.';
COMMENT ON COLUMN counting_by_bucket_15min.net  IS 'Diferencia ins menos outs. Positivo cuando entran más personas de las que salen.';
COMMENT ON COLUMN counting_by_bucket_15min.ins_unknown IS 'Ingresos sin clasificación demográfica confiable (NULL o height_class=unknown).';

COMMENT ON VIEW counting_by_bucket_hour IS
    'Ingresos y egresos agregados por hora con desglose demográfico. Mismas columnas que counting_by_bucket_15min.';
COMMENT ON VIEW counting_by_bucket_day  IS
    'Ingresos y egresos agregados por día con desglose demográfico. Mismas columnas que counting_by_bucket_15min.';

COMMENT ON VIEW wifi_ble_by_bucket_15min IS
    'Dispositivos WiFi/BLE detectados cerca del local en franjas de 15 minutos. Se toma el máximo entre cámaras para evitar doble conteo en sucursales multi-cámara.';
COMMENT ON COLUMN wifi_ble_by_bucket_15min.passersby IS 'Dispositivos detectados en todo el rango RSSI.';
COMMENT ON COLUMN wifi_ble_by_bucket_15min.shoppers  IS 'Subconjunto de passersby con señal fuerte (aproximación de cercanía a la entrada).';

COMMENT ON VIEW wifi_ble_by_bucket_hour IS
    'Tráfico exterior WiFi/BLE agregado por hora. Suma de los buckets de 15 minutos. Advertencia: dispositivos parados varios buckets pueden sobre-contar.';
COMMENT ON VIEW wifi_ble_by_bucket_day  IS
    'Tráfico exterior WiFi/BLE agregado por día.';

COMMENT ON VIEW pos_by_bucket_15min IS
    'Transacciones POS agregadas por sucursal en franjas de 15 minutos. Los montos están en la unidad menor de la moneda (centavos para ARS).';
COMMENT ON COLUMN pos_by_bucket_15min.amount_minor_sale   IS 'Monto vendido en unidad menor de la moneda (BIGINT entero, sin decimales para evitar errores de precisión).';
COMMENT ON COLUMN pos_by_bucket_15min.amount_minor_return IS 'Monto devuelto en unidad menor de la moneda.';
COMMENT ON COLUMN pos_by_bucket_15min.net_amount_minor    IS 'Ventas menos devoluciones, en unidad menor de la moneda.';

COMMENT ON VIEW pos_by_bucket_hour IS 'Transacciones POS agregadas por hora.';
COMMENT ON VIEW pos_by_bucket_day  IS 'Transacciones POS agregadas por día.';

COMMENT ON VIEW occupancy_by_bucket_15min IS
    'Ocupación acumulada (suma corrida de ingresos menos egresos) por sucursal y franja de 15 minutos. Reinicia cada medianoche. La desviación acumulada al cierre representa ingresos o egresos perdidos por el detector visual.';
COMMENT ON COLUMN occupancy_by_bucket_15min.occupancy IS 'Personas estimadas dentro del local al final de la franja.';

COMMENT ON VIEW occupancy_by_bucket_hour IS 'Estadísticas de ocupación por hora (promedio, máximo y mínimo sobre los 4 buckets de 15 minutos).';
COMMENT ON VIEW occupancy_by_bucket_day  IS 'Estadísticas de ocupación por día.';

COMMENT ON VIEW turn_in_rate_by_bucket_15min IS
    'Tasa de captación en franjas de 15 minutos. turn_in_rate equivale al ratio de ingresos sobre dispositivos detectados afuera.';
COMMENT ON COLUMN turn_in_rate_by_bucket_15min.turn_in_rate          IS 'Cociente ingresos / passersby. NULL cuando no se detectaron passersby.';
COMMENT ON COLUMN turn_in_rate_by_bucket_15min.turn_in_shoppers_rate IS 'Cociente ingresos / shoppers. NULL cuando no se detectaron shoppers.';

COMMENT ON VIEW turn_in_rate_by_bucket_hour IS 'Tasa de captación por hora.';
COMMENT ON VIEW turn_in_rate_by_bucket_day  IS 'Tasa de captación por día.';

COMMENT ON VIEW conversion_by_bucket_15min IS
    'Tasa de conversión en franjas de 15 minutos. conversion_rate equivale al ratio de ventas sobre ingresos al local.';
COMMENT ON COLUMN conversion_by_bucket_15min.visits          IS 'Personas que ingresaron al local en la franja (equivale a ins en counting).';
COMMENT ON COLUMN conversion_by_bucket_15min.conversion_rate IS 'Cociente ventas / ingresos. NULL cuando no se registraron ingresos.';

COMMENT ON VIEW conversion_by_bucket_hour IS 'Tasa de conversión por hora.';
COMMENT ON VIEW conversion_by_bucket_day  IS 'Tasa de conversión por día.';

COMMENT ON VIEW visit_duration_by_bucket_hour IS
    'Duración promedio de visita estimada por la Ley de Little (W = L / λ). Devuelve solo el promedio; no incluye distribución por percentiles (limitación inherente al método). Advertencias: el cálculo supone estado estacionario y se degrada en horas pico del mediodía, apertura y cierre del local.';
COMMENT ON COLUMN visit_duration_by_bucket_hour.visit_duration_minutes IS 'Tiempo promedio dentro del local, expresado en minutos. NULL cuando no hubo ingresos en la hora.';

COMMENT ON VIEW visit_duration_by_bucket_day IS 'Duración promedio de visita por día (promedio ponderado por cantidad de ingresos sobre las horas válidas del día).';

COMMENT ON VIEW metrics_unified_by_bucket_15min IS
    'Vista unificada: counting, wifi_ble y pos combinados mediante FULL OUTER JOIN por sucursal y franja de 15 minutos. Consumida por la Lambda query_aggregates. Aplica COALESCE a cero en todas las métricas para garantizar un formato uniforme de salida.';
COMMENT ON VIEW metrics_unified_by_bucket_hour  IS 'Vista unificada por hora. Mismas columnas que metrics_unified_by_bucket_15min.';
COMMENT ON VIEW metrics_unified_by_bucket_day   IS 'Vista unificada por día. Mismas columnas que metrics_unified_by_bucket_15min.';

COMMENT ON VIEW data_freshness_by_store IS
    'Último timestamp de ingesta cruzando las tres fuentes (counting, wifi_ble y pos) por sucursal. Permite al cliente del API detectar dispositivos desconectados.';

-- =============================================================================
-- 13. PERMISOS — restricción de lambda_query_reader y readonly_external
-- =============================================================================
-- Cambio importante: lambda_query_reader pierde el acceso a las tablas
-- crudas. A partir de esta migración solo lee las vistas unificadas, la
-- vista de frescura del dato y las dimensiones (sites, devices). Queda
-- simétrico con readonly_external. La Lambda se refactoriza en este
-- mismo sprint para consumir las vistas en lugar de los CTE inline.

-- Revocar el acceso previo de lambda_query_reader sobre las tablas crudas.
REVOKE SELECT ON count_events       FROM lambda_query_reader;
REVOKE SELECT ON wifi_ble_summary   FROM lambda_query_reader;
REVOKE SELECT ON pos_transactions   FROM lambda_query_reader;

-- Nuevos permisos para lambda_query_reader: solo vistas unificadas y
-- frescura del dato.
GRANT SELECT ON
    metrics_unified_by_bucket_15min,
    metrics_unified_by_bucket_hour,
    metrics_unified_by_bucket_day,
    data_freshness_by_store
TO lambda_query_reader;
-- Las dimensiones sites y devices ya estaban concedidas en el bootstrap
-- original; no se tocan aquí.

-- Permisos para readonly_external sobre TODAS las vistas del producto
-- cartesiano. Las heredadas se eliminaron en cascada al inicio, así que
-- solo se conceden las nuevas.
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
    data_freshness_by_store
TO readonly_external;
-- Las dimensiones sites y devices ya estaban concedidas en el bootstrap;
-- no se tocan aquí.
-- NO se conceden a readonly_external las vistas metrics_unified_*: están
-- pensadas como punto de entrada interno de la Lambda. El partner externo
-- prefiere las vistas individuales por fuente, más explícitas en lo que
-- consulta. Si se solicita expresamente, se conceden en una migración
-- posterior.

COMMIT;

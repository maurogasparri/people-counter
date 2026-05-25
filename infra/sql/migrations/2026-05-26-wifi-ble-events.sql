-- Migración: reemplazar wifi_ble_summary (agregados pre-categorizados) por
-- wifi_ble_events (per-device, RSSI crudo + categorización via función SQL).
-- Fecha: 2026-05-26
-- Sprint: S9 (capa cloud / refactor de modelo de datos)
--
-- Cambio de filosofía: mismo patrón que height_class (PR 1). En lugar de que
-- el device decida "passerby vs shopper vs weak" via thresholds del config y
-- pre-agregue {passersby: N, shoppers: M} por ventana, el device emite UN
-- evento por visitor_hash (post-stitching local) con el RSSI máximo crudo.
-- La categorización vive en la función SQL ``rssi_class(rssi_max)`` que se
-- aplica server-side en las vistas.
--
-- Ventajas:
--   1. Single source of truth de los thresholds (rssi_passerby=-75,
--      rssi_shopper=-55) — modificable con CREATE OR REPLACE, retroactivo a
--      todo el histórico.
--   2. Unique visitors POR DÍA cuentan limpios: COUNT(DISTINCT visitor_hash)
--      sobre los eventos del día — el mismo visitor en 3 ventanas de 15min
--      cuenta como 1, no como 3 (que era el bug del SUM cross-bucket sobre
--      wifi_ble_summary).
--   3. What-if analysis directo en BI ("¿qué pasaría con un threshold de
--      -65 para shopper?") sin re-procesar el device.
--   4. Coherencia con el patrón aplicado a height (PR 1) — "guardar crudo,
--      categorizar al consumir".
--
-- Trade-off del status quo de privacidad (confirmado con el operador
-- 2026-05-26): la categorización post-stitching la sigue haciendo el device
-- LOCAL (cada Pi tiene su namespace de visitor_hash por día — no hay
-- cross-camera dedup en multi-cam por sucursal). El visitor_hash que llega
-- a RDS es el ``group_id`` del DedupEngine local del device, no la MAC ni
-- la hash de MAC. Salt diaria local (rotación via people-counter-reset.service).
--
-- Privacy: el visitor_hash en RDS es opaco (16 bytes derivados de la
-- identidad post-stitching local) — invertirlo a MAC real es
-- computacionalmente infactible (256 bits de salt diaria + SHA-256
-- truncado). Cumple RNF-12 (hashes daily-rotating).

BEGIN;

-- =============================================================================
-- 1. Drop CASCADE de wifi_ble_summary + vistas que dependen
-- =============================================================================
-- Datos viejos del piloto se descartan: no son convertibles a per-device
-- events (la info per-device se agregó en el device antes de persistir).
-- Confirmado con el operador — PoC con piloto único, downsize aceptable a
-- cambio del modelo coherente. Para producción se haría un dual-write
-- transitorio, pero acá es overkill.

DROP VIEW IF EXISTS metrics_unified_by_bucket_15min   CASCADE;
DROP VIEW IF EXISTS metrics_unified_by_bucket_hour    CASCADE;
DROP VIEW IF EXISTS metrics_unified_by_bucket_day     CASCADE;
DROP VIEW IF EXISTS turn_in_rate_by_bucket_15min      CASCADE;
DROP VIEW IF EXISTS turn_in_rate_by_bucket_hour       CASCADE;
DROP VIEW IF EXISTS turn_in_rate_by_bucket_day        CASCADE;
DROP VIEW IF EXISTS wifi_ble_by_bucket_15min          CASCADE;
DROP VIEW IF EXISTS wifi_ble_by_bucket_hour           CASCADE;
DROP VIEW IF EXISTS wifi_ble_by_bucket_day            CASCADE;
DROP TABLE IF EXISTS wifi_ble_summary                 CASCADE;

-- =============================================================================
-- 2. Función rssi_class — categorización por RSSI máximo
-- =============================================================================
-- Thresholds estándar de probing pasivo:
--   shopper  : rssi >= -55 dBm  (muy cerca de la cámara — probable entrada)
--   passerby : rssi >= -75 dBm  (pasó por la zona; incluye shoppers)
--   weak     : rssi <  -75 dBm  (eco lejano — no se cuenta)
--   unknown  : NULL              (sin lectura — descartado downstream)
--
-- IMPORTANTE: shopper ⊆ passerby por convención — el conteo de passersby
-- INCLUYE shoppers (es "todos los que pasaron"). Las vistas filtran
-- explícitamente para preservar el invariante shoppers ≤ passersby.

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
-- 3. Tabla wifi_ble_events — un evento por visitor_hash por ventana
-- =============================================================================
-- El device emite un evento por cada group_id (post-stitching local) visto
-- durante la ventana. El cloud no re-stitchea — sólo dedupa por
-- visitor_hash dentro del bucket de tiempo deseado vía COUNT(DISTINCT ...).
--
-- Retención: SIN auto-delete (decisión del operador). Las queries de unique
-- visitors por día/semana/mes funcionan directo sobre toda la historia.

CREATE TABLE IF NOT EXISTS wifi_ble_events (
    event_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id       TEXT         NOT NULL,
    store_id        TEXT         NOT NULL,
    -- Identidad post-stitching local del device (group_id del DedupEngine).
    -- 16 bytes: derivado de SHA-256 truncado + salt diaria local. Opaco,
    -- no invertible a MAC. Se renueva cada día (people-counter-reset).
    visitor_hash    BYTEA        NOT NULL,
    protocol        TEXT         NOT NULL CHECK (protocol IN ('wifi', 'ble')),
    -- RSSI máximo observado durante la ventana, en dBm (entero negativo
    -- típicamente entre -100 y -20). NULL no permitido — un device sin
    -- ninguna lectura de RSSI no se emite.
    rssi_max        INT          NOT NULL,
    first_seen_ts   TIMESTAMPTZ  NOT NULL,
    last_seen_ts    TIMESTAMPTZ  NOT NULL,
    period_start    TIMESTAMPTZ  NOT NULL,
    period_end      TIMESTAMPTZ  NOT NULL,
    -- Buckets derivados de last_seen_ts (la observación más reciente). Si
    -- el visitor aparece en 2 ventanas de 15min, se inserta 2 filas — una
    -- por cada ventana — pero ambas con el mismo visitor_hash, así
    -- COUNT(DISTINCT visitor_hash) por día sigue siendo 1.
    bucket_15min    TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (last_seen_ts - TIMESTAMPTZ 'epoch')) / 900) * 900)) STORED,
    bucket_hour     TIMESTAMPTZ  GENERATED ALWAYS AS (to_timestamp(floor(extract(epoch FROM (last_seen_ts - TIMESTAMPTZ 'epoch')) / 3600) * 3600)) STORED,
    bucket_day      DATE         GENERATED ALWAYS AS ((TIMESTAMP 'epoch' + floor(extract(epoch FROM (last_seen_ts - TIMESTAMPTZ 'epoch')) / 86400) * INTERVAL '1 day')::date) STORED,
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Idempotencia: un mismo device emitiendo el mismo visitor en la misma
    -- ventana = misma fila. ON CONFLICT en la Lambda hace UPDATE de
    -- rssi_max (MAX) y last_seen_ts (MAX) para refinar.
    UNIQUE (device_id, visitor_hash, period_start)
);

CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_bucket15
    ON wifi_ble_events (store_id, bucket_15min DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_bucket_hour
    ON wifi_ble_events (store_id, bucket_hour DESC);
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_day
    ON wifi_ble_events (store_id, bucket_day DESC);
-- Index del visitor_hash para queries de unique visitors por día/store.
CREATE INDEX IF NOT EXISTS idx_wifi_ble_events_store_day_visitor
    ON wifi_ble_events (store_id, bucket_day, visitor_hash);

COMMENT ON TABLE wifi_ble_events IS
    'Eventos per-device WiFi/BLE: un evento por cada visitor (post-stitching local del device) por ventana de emisión. La categorización shopper/passerby se aplica server-side via rssi_class(rssi_max). Reemplaza la tabla wifi_ble_summary (que pre-agregaba en el device).';

COMMENT ON COLUMN wifi_ble_events.visitor_hash IS
    'Identidad opaca del visitor post-stitching local (group_id del DedupEngine). 16 bytes, derivada de SHA-256 + salt diaria local. No invertible a MAC. Se renueva cada día con el reset diario del device.';

-- =============================================================================
-- 4. Vistas wifi_ble_by_bucket_* — DISTINCT visitor_hash + rssi_class
-- =============================================================================
-- Reemplazo de las vistas anteriores que hacían MAX/SUM sobre wifi_ble_summary.
-- Ahora COUNT(DISTINCT visitor_hash) garantiza que el mismo visitor en N
-- ventanas cuenta como 1 al rollup del bucket más grueso.
--
-- Invariante preservado: shoppers ≤ passersby (shopper es subconjunto de
-- passerby por convención de los thresholds).

CREATE OR REPLACE VIEW wifi_ble_by_bucket_15min AS
SELECT
    store_id,
    bucket_15min,
    COUNT(DISTINCT visitor_hash) FILTER (
        WHERE rssi_class(rssi_max) IN ('passerby', 'shopper')
    ) AS passersby,
    COUNT(DISTINCT visitor_hash) FILTER (
        WHERE rssi_class(rssi_max) = 'shopper'
    ) AS shoppers
FROM wifi_ble_events
GROUP BY store_id, bucket_15min;

CREATE OR REPLACE VIEW wifi_ble_by_bucket_hour AS
SELECT
    store_id,
    bucket_hour,
    COUNT(DISTINCT visitor_hash) FILTER (
        WHERE rssi_class(rssi_max) IN ('passerby', 'shopper')
    ) AS passersby,
    COUNT(DISTINCT visitor_hash) FILTER (
        WHERE rssi_class(rssi_max) = 'shopper'
    ) AS shoppers
FROM wifi_ble_events
GROUP BY store_id, bucket_hour;

CREATE OR REPLACE VIEW wifi_ble_by_bucket_day AS
SELECT
    store_id,
    bucket_day,
    COUNT(DISTINCT visitor_hash) FILTER (
        WHERE rssi_class(rssi_max) IN ('passerby', 'shopper')
    ) AS passersby,
    COUNT(DISTINCT visitor_hash) FILTER (
        WHERE rssi_class(rssi_max) = 'shopper'
    ) AS shoppers
FROM wifi_ble_events
GROUP BY store_id, bucket_day;

COMMENT ON VIEW wifi_ble_by_bucket_15min IS
    'Tráfico exterior (WiFi/BLE) por sucursal y franja de 15 min. Cuenta DISTINCT visitor_hash, así un mismo visitor en N ventanas se dedup correctamente al rollup. Categorización via rssi_class().';
COMMENT ON VIEW wifi_ble_by_bucket_hour IS
    'Tráfico exterior por hora. Mismo shape que wifi_ble_by_bucket_15min.';
COMMENT ON VIEW wifi_ble_by_bucket_day IS
    'Tráfico exterior por día. Unique visitors diarios = passersby + shoppers (con shopper ⊆ passerby).';

-- =============================================================================
-- 5. Recrear vistas dependientes — turn_in_rate y metrics_unified
-- =============================================================================
-- El shape de las vistas wifi_ble_by_bucket_* no cambió (mismas columnas:
-- store_id, bucket_*, passersby, shoppers). Las vistas downstream son
-- recreables idénticas a la version pre-migración.

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

-- metrics_unified: counting + wifi_ble + pos
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

-- =============================================================================
-- 6. Permisos
-- =============================================================================
-- lambda_writer ya tenía INSERT sobre wifi_ble_summary — la tabla cambió
-- de nombre, hay que re-GRANT.

GRANT INSERT, SELECT ON wifi_ble_events TO lambda_writer;

GRANT EXECUTE ON FUNCTION rssi_class(INT) TO lambda_query_reader, readonly_external;

GRANT SELECT ON
    wifi_ble_by_bucket_15min,
    wifi_ble_by_bucket_hour,
    wifi_ble_by_bucket_day,
    turn_in_rate_by_bucket_15min,
    turn_in_rate_by_bucket_hour,
    turn_in_rate_by_bucket_day
TO readonly_external;

GRANT SELECT ON
    metrics_unified_by_bucket_15min,
    metrics_unified_by_bucket_hour,
    metrics_unified_by_bucket_day
TO lambda_query_reader;

COMMIT;

-- =============================================================================
-- Migración de histórico AGREGADO → tablas base de rollup
-- =============================================================================
-- Cuándo usar esto: cuando la data histórica del sistema anterior viene YA
-- AGREGADA (resúmenes por hora/día), sin eventos individuales. En ese caso NO
-- se puede poblar el crudo (count_events / wifi_ble_events / pos_transactions)
-- ni usar refresh_rollups() — se inserta DIRECTO en las tablas base rollup_*.
--
-- (Si tu histórico viene a nivel evento, NO uses esto: insertá en las tablas
--  crudas y dejá que refresh_rollups() derive los rollups. Ver scripts/seed_mock_data.py.)
--
-- Convención CRÍTICA de bucket_hour: se guarda LOCAL-AS-UTC (igual que lhour()):
-- la hora de pared local, materializada como timestamptz en UTC, de modo que
--   extract(hour FROM bucket_hour AT TIME ZONE 'UTC') == hora local.
-- Por eso el cast es:  <hora_local>::timestamp AT TIME ZONE 'UTC'.
-- Si la guardás en UTC real, los dashboards la muestran corrida por el offset.
--
-- Coexistencia con el pipeline en vivo: refresh_rollups() solo toca buckets con
-- eventos crudos cuyo received_at > watermark. El histórico no tiene crudo, así
-- que el refresh NUNCA lo pisa. ⚠️ NO resetees rollup_state.last_refreshed_at a
-- '-infinity' "para reprocesar": no reconstruiría el histórico (no hay raw) y
-- recalcularía todo el vivo al pedo.
--
-- Idempotente: todos los INSERT usan ON CONFLICT DO UPDATE → podés re-correrlo.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1) STAGING. Cargá acá tu export agregado (1 fila por store × hora local).
--    Mapeá las columnas de tu fuente a estas. Las que no tengas → dejalas NULL;
--    el transform las trata como 0 / unknown. POS y WiFi son opcionales.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stg_historical_hourly (
    store_id    TEXT      NOT NULL,
    local_hour  TIMESTAMP NOT NULL,           -- hora de pared LOCAL, truncada a la hora (sin tz)
    -- counting (footfall)
    ins         BIGINT,
    outs        BIGINT,
    ins_adult   BIGINT,                        -- NULL si el incumbente no medía altura
    ins_child   BIGINT,
    -- POS (opcional)
    sales               BIGINT,
    returns             BIGINT,
    transactions        BIGINT,
    items_sale          BIGINT,
    items_return        BIGINT,
    amount_minor_sale   NUMERIC,               -- en la unidad menor (centavos)
    amount_minor_return NUMERIC,
    currency            CHAR(3),
    -- WiFi/BLE (opcional)
    passersby   BIGINT,
    shoppers    BIGINT,
    visitors    BIGINT,
    PRIMARY KEY (store_id, local_hour)
);

-- (Cargá la staging por fuera: \copy stg_historical_hourly FROM 'export.csv' CSV HEADER
--  o usá scripts/migrate_historical.py que lo hace por lotes.)

-- Helper local: el cast canónico hora-local → bucket_hour (local-as-UTC).
-- bucket_hour := local_hour::timestamp AT TIME ZONE 'UTC'

-- -----------------------------------------------------------------------------
-- 2) COUNTING → rollup_counting_hour  (+ deriva _day)
--    net = ins - outs.  Demografía: lo conocido va a adult/child, el resto a
--    unknown. Sin demografía → todo unknown.
-- -----------------------------------------------------------------------------
INSERT INTO rollup_counting_hour
    (store_id, bucket_hour, ins, outs, net,
     ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown)
SELECT
    store_id,
    local_hour::timestamp AT TIME ZONE 'UTC'                       AS bucket_hour,
    COALESCE(ins, 0), COALESCE(outs, 0), COALESCE(ins,0) - COALESCE(outs,0),
    COALESCE(ins_adult, 0),
    COALESCE(ins_child, 0),
    COALESCE(ins, 0) - COALESCE(ins_adult, 0) - COALESCE(ins_child, 0),  -- ins_unknown
    0, 0, COALESCE(outs, 0)                                              -- outs sin demografía → unknown
FROM stg_historical_hourly
WHERE ins IS NOT NULL OR outs IS NOT NULL
ON CONFLICT (store_id, bucket_hour) DO UPDATE SET
    ins=EXCLUDED.ins, outs=EXCLUDED.outs, net=EXCLUDED.net,
    ins_adult=EXCLUDED.ins_adult, ins_child=EXCLUDED.ins_child, ins_unknown=EXCLUDED.ins_unknown,
    outs_adult=EXCLUDED.outs_adult, outs_child=EXCLUDED.outs_child, outs_unknown=EXCLUDED.outs_unknown;

INSERT INTO rollup_counting_day
    (store_id, bucket_day, ins, outs, net,
     ins_adult, ins_child, ins_unknown, outs_adult, outs_child, outs_unknown)
SELECT
    store_id, (bucket_hour AT TIME ZONE 'UTC')::date AS bucket_day,
    SUM(ins), SUM(outs), SUM(net),
    SUM(ins_adult), SUM(ins_child), SUM(ins_unknown),
    SUM(outs_adult), SUM(outs_child), SUM(outs_unknown)
FROM rollup_counting_hour
GROUP BY store_id, (bucket_hour AT TIME ZONE 'UTC')::date
ON CONFLICT (store_id, bucket_day) DO UPDATE SET
    ins=EXCLUDED.ins, outs=EXCLUDED.outs, net=EXCLUDED.net,
    ins_adult=EXCLUDED.ins_adult, ins_child=EXCLUDED.ins_child, ins_unknown=EXCLUDED.ins_unknown,
    outs_adult=EXCLUDED.outs_adult, outs_child=EXCLUDED.outs_child, outs_unknown=EXCLUDED.outs_unknown;

-- NOTA ocupación: occupancy_by_bucket_* hace cumsum sobre el grano de 15min.
-- Con histórico solo horario NO se puede reconstruir → las vistas de ocupación
-- no tendrán histórico (esperado si el incumbente no la medía). Si tuvieras
-- grano 15min, poblá rollup_counting_15min análogamente y la ocupación sale.

-- -----------------------------------------------------------------------------
-- 3) POS → rollup_pos_hour  (+ deriva _day).  Solo si tenés POS histórico.
-- -----------------------------------------------------------------------------
INSERT INTO rollup_pos_hour
    (store_id, bucket_hour, sales, returns, transactions, items_sale, items_return,
     amount_minor_sale, amount_minor_return, net_amount_minor, currency)
SELECT
    store_id, local_hour::timestamp AT TIME ZONE 'UTC',
    COALESCE(sales,0), COALESCE(returns,0), COALESCE(transactions,0),
    COALESCE(items_sale,0), COALESCE(items_return,0),
    COALESCE(amount_minor_sale,0), COALESCE(amount_minor_return,0),
    COALESCE(amount_minor_sale,0) - COALESCE(amount_minor_return,0),
    COALESCE(currency, 'ARS')
FROM stg_historical_hourly
WHERE sales IS NOT NULL OR amount_minor_sale IS NOT NULL
ON CONFLICT (store_id, bucket_hour) DO UPDATE SET
    sales=EXCLUDED.sales, returns=EXCLUDED.returns, transactions=EXCLUDED.transactions,
    items_sale=EXCLUDED.items_sale, items_return=EXCLUDED.items_return,
    amount_minor_sale=EXCLUDED.amount_minor_sale, amount_minor_return=EXCLUDED.amount_minor_return,
    net_amount_minor=EXCLUDED.net_amount_minor, currency=EXCLUDED.currency;

INSERT INTO rollup_pos_day
    (store_id, bucket_day, sales, returns, transactions, items_sale, items_return,
     amount_minor_sale, amount_minor_return, net_amount_minor, currency)
SELECT
    store_id, (bucket_hour AT TIME ZONE 'UTC')::date,
    SUM(sales), SUM(returns), SUM(transactions), SUM(items_sale), SUM(items_return),
    SUM(amount_minor_sale), SUM(amount_minor_return), SUM(net_amount_minor), MIN(currency)
FROM rollup_pos_hour
GROUP BY store_id, (bucket_hour AT TIME ZONE 'UTC')::date
ON CONFLICT (store_id, bucket_day) DO UPDATE SET
    sales=EXCLUDED.sales, returns=EXCLUDED.returns, transactions=EXCLUDED.transactions,
    items_sale=EXCLUDED.items_sale, items_return=EXCLUDED.items_return,
    amount_minor_sale=EXCLUDED.amount_minor_sale, amount_minor_return=EXCLUDED.amount_minor_return,
    net_amount_minor=EXCLUDED.net_amount_minor, currency=EXCLUDED.currency;

-- -----------------------------------------------------------------------------
-- 4) WiFi/BLE → rollup_wifi_ble_hour  (+ deriva _day).  Solo si lo tenés.
-- -----------------------------------------------------------------------------
INSERT INTO rollup_wifi_ble_hour (store_id, bucket_hour, passersby, shoppers, visitors)
SELECT store_id, local_hour::timestamp AT TIME ZONE 'UTC',
       COALESCE(passersby,0), COALESCE(shoppers,0), COALESCE(visitors,0)
FROM stg_historical_hourly
WHERE passersby IS NOT NULL OR shoppers IS NOT NULL OR visitors IS NOT NULL
ON CONFLICT (store_id, bucket_hour) DO UPDATE SET
    passersby=EXCLUDED.passersby, shoppers=EXCLUDED.shoppers, visitors=EXCLUDED.visitors;

INSERT INTO rollup_wifi_ble_day (store_id, bucket_day, passersby, shoppers, visitors)
SELECT store_id, (bucket_hour AT TIME ZONE 'UTC')::date,
       SUM(passersby), SUM(shoppers), SUM(visitors)
FROM rollup_wifi_ble_hour
GROUP BY store_id, (bucket_hour AT TIME ZONE 'UTC')::date
ON CONFLICT (store_id, bucket_day) DO UPDATE SET
    passersby=EXCLUDED.passersby, shoppers=EXCLUDED.shoppers, visitors=EXCLUDED.visitors;

-- -----------------------------------------------------------------------------
-- 5) Sanity checks (deberían dar 0 filas fuera del horario de operación, etc.)
-- -----------------------------------------------------------------------------
-- SELECT store_id, min(bucket_hour AT TIME ZONE 'UTC'), max(bucket_hour AT TIME ZONE 'UTC')
--   FROM rollup_counting_hour GROUP BY store_id;
-- SELECT count(*) FROM rollup_counting_hour
--   WHERE extract(hour FROM bucket_hour AT TIME ZONE 'UTC') NOT BETWEEN <apertura> AND <cierre>;

-- Limpieza opcional de la staging cuando termines:
-- DROP TABLE stg_historical_hourly;

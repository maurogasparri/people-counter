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
-- count_events
-- ---------------------------------------------------------------------------
-- Drop el índice que referencia bucket_15min (DROP COLUMN lo dropearía solo,
-- pero lo hacemos explícito para que el re-create posterior sea limpio).
DROP INDEX IF EXISTS idx_count_events_bucket15;

-- Recrear bucket_15min como GENERATED. Postgres recomputa para TODOS los
-- rows existentes desde event_ts (server-derived) — históricos correctos.
ALTER TABLE count_events DROP COLUMN IF EXISTS bucket_15min;
ALTER TABLE count_events ADD COLUMN bucket_15min TIMESTAMPTZ
    GENERATED ALWAYS AS (
        to_timestamp(floor(extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch')) / 900) * 900)
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_count_events_bucket15
    ON count_events (store_id, bucket_15min DESC);

-- ---------------------------------------------------------------------------
-- wifi_ble_summary
-- ---------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_wifi_ble_store_bucket15;

ALTER TABLE wifi_ble_summary DROP COLUMN IF EXISTS bucket_15min;
ALTER TABLE wifi_ble_summary ADD COLUMN bucket_15min TIMESTAMPTZ
    GENERATED ALWAYS AS (
        to_timestamp(floor(extract(epoch FROM (period_start - TIMESTAMPTZ 'epoch')) / 900) * 900)
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_wifi_ble_store_bucket15
    ON wifi_ble_summary (store_id, bucket_15min DESC);

-- last_seen_ts: timestamp de la última detección de un visitor dentro del
-- período. Info diagnóstica. NULLABLE — devices con firmware viejo no lo
-- mandan; el back-fill no es posible para rows históricos (la info no
-- existe), quedan en NULL hasta que el device empiece a reportarlo.
ALTER TABLE wifi_ble_summary
    ADD COLUMN IF NOT EXISTS last_seen_ts TIMESTAMPTZ;

COMMIT;

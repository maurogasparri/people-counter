-- =============================================================================
-- 2026-05-23  Canary del Device Shadow: telemetry.last_shadow_apply_ts
-- =============================================================================
-- Timestamp del último delta del IoT Device Shadow aplicado por el device
-- (apply_shadow_delta). NULLABLE — backfill no aplica para rows históricos
-- (la info no existe; quedan en NULL hasta que el device empiece a reportarlo
-- post-activación del shadow).
--
-- Aplicado por: debug/apply_telemetry_migration.py (o psql directo).

BEGIN;

ALTER TABLE telemetry
    ADD COLUMN IF NOT EXISTS last_shadow_apply_ts TIMESTAMPTZ;

COMMIT;

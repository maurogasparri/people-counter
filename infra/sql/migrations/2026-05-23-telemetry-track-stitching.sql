-- Migración: agregar columnas de telemetría del tracker de visión.
-- Idempotente — usa ADD COLUMN IF NOT EXISTS para que pueda re-correrse.
--
-- Contexto: el Counter ahora publica ``track_stitching_ratio`` (unique IDs
-- vistos cruzar el ROI / counts emitidos, ideal ≈ 1.0; >1.3 = fragmentación)
-- y ``death_emit_count`` (cantidad de death-emits del día, sirve para
-- diferenciar 'fragmenta-y-rescata' de 'fragmenta-y-pierde'). El Lambda
-- ``persist_event`` los persiste a partir de este commit. Ver
-- "Design philosophy del counter" en CLAUDE.md.
--
-- Run-once-per-environment. Para nuevos deploys, las columnas ya están en
-- bootstrap.sql y no hace falta correr esto.

ALTER TABLE telemetry
    ADD COLUMN IF NOT EXISTS track_stitching_ratio REAL;

ALTER TABLE telemetry
    ADD COLUMN IF NOT EXISTS death_emit_count INTEGER;

COMMENT ON COLUMN telemetry.track_stitching_ratio IS
    'Unique tracker IDs que cruzaron el ROI / total counts emitidos hoy. '
    'Ideal ≈ 1.0 (1 persona = 1 ID = 1 evento); >1.3 sugiere fragmentación '
    'de identidad del tracker. Reset diario.';

COMMENT ON COLUMN telemetry.death_emit_count IS
    'Cantidad de death-emits del counter disparados hoy (fallback que cuenta '
    'tracks que cruzaron pero murieron dentro del ROI sin alcanzar el borde). '
    'Combinado con track_stitching_ratio diferencia fragmenta-y-rescata '
    '(count alto, cuentas OK) de fragmenta-y-pierde (count bajo + ratio alto '
    '→ recall del detector flojo). Reset diario.';

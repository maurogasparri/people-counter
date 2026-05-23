-- Migración: agregar columna ``ghost_adoption_count`` a ``telemetry`` para
-- cerrar el árbol diagnóstico del tracker.
--
-- Combinado con ``track_stitching_ratio`` y ``death_emit_count`` (ya
-- existentes), permite distinguir 4 modos de operación:
--
--   stitching_ratio | adoption | death_emit | diagnóstico
--   ≈1.0            | 0        | 0          | tracker perfecto
--   ≈1.0            | >0       | 0          | fragmentación silenciosa rescatada por capa 1 (ghost adoption)
--   ≈1.0            | 0        | >0         | crossers rescatados por capa 3 (death-emit)
--   >1.3            | 0        | 0          | FRAGMENTACIÓN SIN RESCATE (alarma)
--   >1.3            | >0       | >0         | tracker flakey, dependés de ambas capas
--
-- Idempotente: ADD COLUMN IF NOT EXISTS.

ALTER TABLE telemetry
    ADD COLUMN IF NOT EXISTS ghost_adoption_count INTEGER;

COMMENT ON COLUMN telemetry.ghost_adoption_count IS
    'Capa 1 del rescue del tracker: cuántas veces un track nuevo adoptó el '
    'ID de un ghost recientemente muerto (bbox-overlap + dist gates). '
    'Acumulativo desde el boot del proceso; reset implícito al restart. '
    'Combinado con death_emit_count + track_stitching_ratio cierra el árbol '
    'diagnóstico de la flota.';

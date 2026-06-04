# Migraciones SQL

Cambios incrementales y **data-preserving** sobre la RDS del piloto (que ya
tiene datos en producción). Estilo `ADD COLUMN IF NOT EXISTS` / `CREATE OR
REPLACE` / `ALTER`, no `DROP TABLE`.

## Relación con `bootstrap.sql`

- **`../bootstrap.sql`** es el schema canónico para **deploys frescos**. Hace
  `DROP TABLE ... CASCADE` → destruye data. Un deploy fresco desde bootstrap
  produce el mismo estado que correr todas las migraciones históricas en orden.
- **Este directorio** contiene solo las migraciones **PENDIENTES** de aplicar a
  la DB viva. Una vez aplicada Y reflejada en `bootstrap.sql`, la migración se
  **squashea** (se borra de acá): su contenido ya vive en el schema canónico y
  el git history preserva el archivo.

Las migraciones hasta **2026-05-28 inclusive** ya fueron aplicadas al piloto y
consolidadas en `bootstrap.sql` (ver el header de ese archivo). Quedan
**pendientes** (aplicadas a la DB viva pero todavía sin foldear a
`bootstrap.sql`) la capa de rollup:

- `2026-05-31_rollup_layer.sql` — tablas base `rollup_*` + `refresh_rollups()`
  incremental (watermark en `rollup_state`) + las views `*_by_bucket_*` como
  UNION rollup + live-tail.
- `2026-05-31b_tz_aware_bucketing.sql` — bucketing tz-aware (local-as-UTC).

## `migrate_historical_rollups.example.sql` NO es una migración

El archivo `../migrate_historical_rollups.example.sql` (un nivel arriba, junto a
`bootstrap.sql`) **no** es una migración de schema: es un **template de carga**
de histórico AGREGADO (staging → tablas base `rollup_*`), pensado para correrse
una sola vez al importar data del sistema anterior. No va en este directorio ni
se foldea a `bootstrap.sql`.

## Cómo aplicar una migración pendiente

```powershell
# Conecta como master (people_counter) via Secrets Manager + SSL.
# Patrón: scripts/provision.py:_rds_connect(stack, region).
# El stack del piloto es "people-counter-dev".
```

Escribir un applier one-off en `debug/` (gitignoreado) que lea el `.sql` y lo
ejecute con `_rds_connect("people-counter-dev", "us-east-1")` + `autocommit`.
Tras aplicarla y foldearla a `bootstrap.sql`, borrar el `.sql` de acá.

# Migraciones SQL

Cambios incrementales y **data-preserving** sobre la RDS del PoC (que ya
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

**No hay migraciones pendientes.** Todas las migraciones aplicadas al PoC
—incl. la capa de rollup (`rollup_*` + `refresh_rollups()` incremental con
watermark en `rollup_state` + views `*_by_bucket_*` UNION rollup + live-tail),
el bucketing tz-aware (local-as-UTC) y el canary `ambiguous_reject_count`— ya
están foldeadas en `bootstrap.sql`. Un deploy fresco desde bootstrap reproduce
el estado completo; el git history preserva los `.sql` squasheados.

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
# El stack del PoC es "people-counter-dev".
```

Escribir un applier one-off en `debug/` (gitignoreado) que lea el `.sql` y lo
ejecute con `_rds_connect("people-counter-dev", "us-east-1")` + `autocommit`.
Tras aplicarla y foldearla a `bootstrap.sql`, borrar el `.sql` de acá.

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
consolidadas en `bootstrap.sql` (ver el header de ese archivo). Por eso este
directorio está vacío.

## Cómo aplicar una migración pendiente

```powershell
# Conecta como master (people_counter) via Secrets Manager + SSL.
# Patrón: scripts/provision.py:_rds_connect(stack, region).
# El stack del piloto es "people-counter-dev".
```

Escribir un applier one-off en `debug/` (gitignoreado) que lea el `.sql` y lo
ejecute con `_rds_connect("people-counter-dev", "us-east-1")` + `autocommit`.
Tras aplicarla y foldearla a `bootstrap.sql`, borrar el `.sql` de acá.

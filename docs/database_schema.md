# Schema de la base de datos

Modelo ER de las dos databases del sistema:

1. **Postgres (RDS, cloud)** — 4 tablas raw + vistas derivadas
2. **SQLite (device, local)** — 2 tablas para outbox MQTT y stitching state

DDL fuente:
- Postgres: [`infra/sql/bootstrap.sql`](../infra/sql/bootstrap.sql)
- SQLite outbox: [`src/mqtt/buffer.py`](../src/mqtt/buffer.py) `_ensure_db()`
- SQLite stitching: [`src/wifi_ble/dedup.py`](../src/wifi_ble/dedup.py) `_ensure_db()`

Snapshot del 2026-05-26 post-refactor de categorización server-side:

- `count_events.height_class` dropeada — categorización ahora vive en la
  función SQL `height_class(REAL)`, aplicada en las vistas.
- `wifi_ble_summary` reemplazada por `wifi_ble_events` (un evento per-device
  por ventana, con `visitor_hash` post-stitching local y RSSI crudo).
  Categorización passerby/shopper en función SQL `rssi_class(INT)`.

Migraciones:
- [`infra/sql/migrations/2026-05-26-drop-height-class.sql`](../infra/sql/migrations/2026-05-26-drop-height-class.sql)
- [`infra/sql/migrations/2026-05-26-wifi-ble-events.sql`](../infra/sql/migrations/2026-05-26-wifi-ble-events.sql)
- [`infra/sql/migrations/2026-05-26-restore-cascade-dropped-views.sql`](../infra/sql/migrations/2026-05-26-restore-cascade-dropped-views.sql)

---

# Parte 1 — Postgres (RDS, cloud)

## Diagrama ER

```mermaid
erDiagram
    count_events {
        uuid        event_id        PK
        text        device_id       "indexado"
        text        store_id        "join key principal"
        timestamptz event_ts        "indexado"
        timestamptz bucket_15min    "GENERATED desde event_ts (server-side)"
        timestamptz bucket_hour     "GENERATED desde event_ts"
        date        bucket_day      "GENERATED desde event_ts"
        text        direction       "CHECK in_out"
        int         track_id
        real        confidence      "debug: investigar FPs"
        real        height_m        "medicion cruda; categorizacion via SQL height_class()"
        timestamptz received_at
    }

    wifi_ble_events {
        uuid        event_id        PK
        text        device_id
        text        store_id        "join key"
        bytea       visitor_hash    "16 bytes; group_id post-stitching local (opaco)"
        text        protocol        "CHECK wifi_ble"
        int         rssi_max        "dBm crudo; categorizacion via SQL rssi_class()"
        timestamptz first_seen_ts   "MIN sobre miembros del group"
        timestamptz last_seen_ts    "MAX sobre miembros (puede caer en la ventana)"
        timestamptz period_start    "inicio ventana de emision del device"
        timestamptz period_end      "fin ventana de emision"
        timestamptz bucket_15min    "GENERATED desde last_seen_ts (server-side)"
        timestamptz bucket_hour     "GENERATED desde last_seen_ts"
        date        bucket_day      "GENERATED desde last_seen_ts"
        timestamptz received_at
    }

    telemetry {
        uuid        telemetry_id    PK
        text        device_id
        text        store_id
        timestamptz event_ts        "cada 5min"
        timestamptz bucket_hour     "GENERATED, telemetry no usa 15min ni day"
        real        cpu_temp_c
        real        hailo_temp_c
        real        fps
        real        wifi_ble_stitching_ratio "canary del stitching wifi/ble"
        real        track_stitching_ratio    "canary fragmentación del tracker"
        int         ghost_adoption_count     "capa 1 rescue: ID adoptions"
        int         death_emit_count         "capa 3 rescue: death-emits firing"
        timestamptz last_shadow_apply_ts     "canary Device Shadow: ultimo delta aplicado, NULL si nunca hubo push"
        text        error
        timestamptz received_at
    }

    pos_transactions {
        uuid        pos_id          PK
        text        transaction_id  "factura o batch_id del POS"
        text        store_id        "join key"
        timestamptz event_ts        "indexado"
        text        type            "CHECK sale_return"
        int         items           "CHECK >= 0"
        bigint      amount_minor    "CHECK >= 0, centavos"
        char        currency        "ISO 4217, default ARS"
        text        payment_method  "nullable o mixed para batches"
        timestamptz bucket_15min    "GENERATED server-side desde event_ts"
        timestamptz bucket_hour     "GENERATED"
        date        bucket_day      "GENERATED"
        timestamptz received_at
    }

    count_events     ||--o{ wifi_ble_events   : "store_id + bucket_15min (turn-in rate)"
    count_events     ||--o{ pos_transactions  : "store_id + bucket_15min (conversion rate)"
    count_events     ||--o{ telemetry         : "device_id (mismo device)"
    wifi_ble_events  ||--o{ telemetry         : "device_id (mismo device)"
```

## Categorización server-side via funciones SQL

Patrón aplicado en mayo 2026 para centralizar los thresholds que antes
vivían en el config local del device (y por ende drifteaban entre devices
de la flota): el device persiste sólo la **medición cruda** (`height_m`,
`rssi_max`), y la categorización se aplica en las vistas vía funciones
SQL `IMMUTABLE`.

| Función | Input | Output | Thresholds |
|---|---|---|---|
| `height_class(REAL)` | metros | `'adult'` / `'child'` / `'unknown'` | `< 1.55` → child, NULL → unknown, else adult |
| `rssi_class(INT)` | dBm | `'shopper'` / `'passerby'` / `'weak'` / `'unknown'` | `>= -55` → shopper, `>= -75` → passerby, NULL → unknown, else weak |

**Single source of truth**: modificar el threshold = `CREATE OR REPLACE
FUNCTION` y se aplica retroactivo a todo el histórico + a todas las
vistas downstream. What-if analysis directo en BI sin re-procesar el
device. Granted EXECUTE a `lambda_query_reader` y `readonly_external`.

**Invariante shoppers ⊆ passersby**: por convención el conteo de
`passersby` en las vistas usa el filtro `rssi_class(rssi_max) IN
('passerby', 'shopper')` — un shopper también cuenta como passerby. El
funnel queda internamente consistente bucket-por-bucket.

## Modelo de joins

**Las tablas de hechos no tienen foreign keys hacia las dimensiones** —
`store_id` y `device_id` son `TEXT` libres en `count_events`,
`wifi_ble_events`, `telemetry`, `pos_transactions`. Es deliberado: la Lambda
escribe hechos y NO debe fallar si un site todavía no se registró; Grafana hace
LEFT JOIN. Los joins de hechos son por **convención de naming**.

Sí existen dos tablas de **dimensiones** (agregadas 2026-05-22, sembradas en
provisioning vía `scripts/provision.py` / `scripts/reset_dedup.py`... ver
`provision.py create --latitude/--longitude`):

- **`sites`** (PK `store_id`): `store_name`, `latitude`/`longitude` (DOUBLE
  PRECISION, para el geomap de Grafana), `timezone`, `address`.
- **`devices`** (PK `device_id`, FK → `sites.store_id`): `cam_label`,
  `firmware_version`, `installed_at`.

Sirven para el geomap, los dropdowns de filtro de Grafana (template vars desde
una tabla chica en vez de `SELECT DISTINCT` sobre los hechos) y labels
human-readable. La única FK del schema es `devices → sites`. DDL en
`infra/sql/bootstrap.sql` (fuera del bloque de DROP — no se borran al
re-bootstrap). Los joins:

- **`store_id`**: presente en las 4 tablas. Mismo string en todos lados (ej.
  `ar-recoleta`). Es el join key principal para cualquier reporte by-store.
- **`device_id`**: presente solo en las 3 tablas IoT (counter, wifi/ble,
  telemetry). El POS no tiene device_id (viene de sistema externo). Convención:
  `<store_id>-cam-<n>` — `store_id` se infiere en Lambda con `_infer_store_id()`.
- **`bucket_15min`**: presente en `count_events`, `wifi_ble_events` y
  `pos_transactions`. Mismo TIMESTAMPTZ alineado a múltiplos de 15 min del
  epoch UTC en todas las tablas → joins temporales sin `date_trunc`.

### Modelo de buckets

Tres granularidades soportadas vía columnas dedicadas para que Grafana queries
no necesiten `date_trunc`:

| Columna | Tipo | Origen |
|---|---|---|
| `bucket_15min` | TIMESTAMPTZ | **Server-derived `GENERATED ALWAYS AS STORED`** desde `event_ts` (count_events, pos_transactions) o `last_seen_ts` (wifi_ble_events). El device manda timestamps crudos — desacopla device ↔ schema. Migrar a bucket de otro tamaño = `ALTER COLUMN` en RDS, sin tocar device/MQTT/Lambda. En `wifi_ble_events` el bucket se deriva de `last_seen_ts` (no `period_start`) para que un visitor presente en varias ventanas caiga naturalmente en el bucket más reciente al rollup; `COUNT(DISTINCT visitor_hash)` dedupa cross-window. |
| `bucket_hour` | TIMESTAMPTZ | GENERATED ALWAYS AS STORED — `date_trunc('hour', event_ts)` en todas las tablas. |
| `bucket_day` | DATE | GENERATED ALWAYS AS STORED — `date_trunc('day', event_ts)::date` en todas las tablas excepto telemetry (no aplica naturalmente a samples de 5min). |

## Vistas derivadas

Producto cartesiano `<métrica>_by_bucket_<grano>` sobre 3 granularidades
(`15min`, `hour`, `day`) y vistas unificadas consumidas por la Lambda
`query_aggregates`. Detalle completo en
[`infra/sql/migrations/2026-05-26-views-cartesian-product.sql`](../infra/sql/migrations/2026-05-26-views-cartesian-product.sql).

```
counter (count_events)                  POS (pos_transactions)
  |  height_class(height_m) FILTER       |
  +- counting_by_bucket_{15min,hour,day}  +- pos_by_bucket_{15min,hour,day}
            |     (+ breakdown adult/child via SQL function)
            |
            |   wifi_ble (wifi_ble_events)
            |     |  rssi_class(rssi_max) FILTER + COUNT(DISTINCT visitor_hash)
            |     +- wifi_ble_by_bucket_{15min,hour,day}
            |          |
            +----------+-- turn_in_rate_by_bucket_{15min,hour,day}    -> US-04
            |
            +- conversion_by_bucket_{15min,hour,day}                  -> US-06
            +- occupancy_by_bucket_{15min,hour,day} (cumsum por día)
            +- visit_duration_by_bucket_{hour,day}  (Ley de Little)

           ↓ COMBINADAS (Lambda query_aggregates consume estas)
       metrics_unified_by_bucket_{15min,hour,day}  (FULL OUTER JOIN counting+wifi_ble+pos)
       data_freshness_by_store                     (último received_at cross-fact)
```

| Vista | Granularidad | Cierra | Notas |
|---|---|---|---|
| `counting_by_bucket_*` | 15min/1h/1d | US-05 | Ins/outs/net + breakdown `ins_adult` / `ins_child` / `ins_unknown` via `height_class(height_m)` |
| `wifi_ble_by_bucket_*` | 15min/1h/1d | --- | `COUNT(DISTINCT visitor_hash)` + filter `rssi_class(rssi_max) IN ('passerby','shopper')` → dedupa cross-window |
| `turn_in_rate_by_bucket_*` | 15min/1h/1d | US-04 | `ins / passersby` y `ins / shoppers` |
| `conversion_by_bucket_*` | 15min/1h/1d | US-06 | `sales / visits` + amounts net/gross |
| `occupancy_by_bucket_*` | 15min/1h/1d | --- | Cumsum `ins - outs` window-partitioned por día |
| `visit_duration_by_bucket_{hour,day}` | 1h/1d | --- | Ley de Little: `avg_occupancy / arrivals` en minutos |
| `metrics_unified_by_bucket_*` | 15min/1h/1d | US-09 | FULL OUTER JOIN counting + wifi_ble + pos. Consumida por Lambda `query_aggregates` |
| `data_freshness_by_store` | --- | US-08 | Último `received_at` cross-fact por sucursal |

## Idempotencia (UNIQUE constraints)

| Tabla | Constraint | Caso de uso |
|---|---|---|
| `count_events` | `(device_id, event_ts, track_id, direction)` | Replay del buffer SQLite del device cuando reconecta |
| `wifi_ble_events` | `(device_id, visitor_hash, period_start)` | Retry del WifiBlePublisher cada 15min con `ON CONFLICT DO UPDATE` (GREATEST rssi_max + last_seen_ts) |
| `telemetry` | `(device_id, event_ts)` | Retry de IoT Topic Rule |
| `pos_transactions` | `(store_id, transaction_id)` | Retry del POS o re-envío del batch |

`count_events` / `telemetry` / `pos_transactions` usan `INSERT ... ON
CONFLICT DO NOTHING`. `wifi_ble_events` usa `ON CONFLICT DO UPDATE`
refinando `rssi_max` al MAX y `last_seen_ts` al MAX — un retry del mismo
período sólo mejora la observación (cubre el caso "el publisher reintentó
porque MQTT falló mid-batch y ahora rssi_max es más fuerte").

## Usuarios IAM auth (least privilege)

| User Postgres | Lambda que lo usa | Grants |
|---|---|---|
| `lambda_writer` | `persist_event` (IoT → counter + wifi/ble + telemetry) | `INSERT, SELECT` sobre `count_events`, `wifi_ble_events`, `telemetry` |
| `lambda_pos_writer` | `ingest_pos_transaction` (API Gateway → POS) | `INSERT, SELECT` sobre `pos_transactions` |
| `lambda_query_reader` | `query_aggregates` (API GW → BI/partners) | `SELECT` sobre vistas `metrics_unified_*` + `data_freshness_by_store` + dimensiones; `EXECUTE` sobre `height_class(REAL)` y `rssi_class(INT)` |
| `readonly_external` | Partners/analistas externos (SQL directo) | `SELECT` sobre todas las vistas `*_by_bucket_*` + dimensiones; `EXECUTE` sobre las funciones de categorización. Acceso vía password (Secrets Manager), no IAM |

Ambos usuarios tienen `rds_iam` GRANT (auth IAM sin password). Los tokens
duran ~15min, las Lambdas los regeneran transparentemente via
`rds.generate_db_auth_token`.

## Index strategy

Por tabla, además de los PKs y UNIQUE constraints implícitos:

- **count_events**: `(store_id, event_ts DESC)`, `(device_id, event_ts DESC)`,
  `(store_id, bucket_15min DESC)`, `(store_id, bucket_day DESC)`
- **wifi_ble_events**: `(store_id, bucket_15min DESC)`, `(store_id, bucket_hour DESC)`, `(store_id, bucket_day DESC)`, `(store_id, bucket_day, visitor_hash)` (último para queries DISTINCT visitor diario)
- **telemetry**: `(device_id, event_ts DESC)`, `(device_id, bucket_hour DESC)`
- **pos_transactions**: `(store_id, bucket_15min DESC)`, `(store_id, bucket_day DESC)`

Todos `DESC` porque las queries típicas de Grafana son "últimas N horas/días".

## Database "grafana" (separada)

Grafana guarda su state interno (users, dashboards, sessions, datasources) en
una database separada `grafana` del mismo cluster RDS. Owner =
`people_counter` (master user) para que Grafana pueda crear/modificar sus
tablas en el primer boot. No interactúa con las tablas de eventos arriba.

---

# Parte 2 — SQLite (device, local)

Cada device corre con dos archivos SQLite independientes en
`/var/lib/people-counter/`:

- `mqtt_buffer.sqlite` — outbox de mensajes pendientes de publish a IoT Core
- `wifi_ble_dedup.sqlite` — state del stitching local (hash groups,
  seqnums, RSSI windows)

Son **independientes entre sí** (no hay joins ni FKs cross-DB) y son
**locales al device** — nunca se sincronizan al cloud. La rotación es
cargo del device: el buffer purga >72h, el dedup hace `reset_daily()`.

## Diagrama ER (SQLite local)

```mermaid
erDiagram
    messages {
        integer  id           PK
        text     topic         "counting | telemetry | wifi_ble"
        text     payload       "JSON blob completo del evento"
        real     created_at    "epoch seconds, para purge >72h"
        integer  sent          "0 pending, 1 acked tras PUBACK"
    }

    hash_groups {
        text     hash          "SHA-256 truncado a 16 bytes de la MAC o BDADDR"
        text     protocol      "wifi | ble"
        text     group_id      "identidad inferida del dispositivo"
        real     first_seen    "epoch seconds"
        real     last_seen     "epoch seconds, para windows del stitching"
        real     rssi          "ULTIMA lectura — para deltas de RSSI del stitching"
        real     max_rssi      "RSSI mas fuerte vista — clasifica passerby/shopper (estable)"
        integer  seqnum        "WiFi 802.11 SC, nullable para BLE"
        text     fingerprint   "fingerprint estable (IEs WiFi / mfg-data BLE), regla 4"
    }
```

**No hay relación entre las dos tablas** — viven en archivos sqlite distintos
y tienen ciclos de vida distintos. Aparecen en el mismo diagrama solo por
documentación.

## `messages` — outbox MQTT (`src/mqtt/buffer.py`)

Cola persistente FIFO para sobrevivir reconexiones de internet. El cliente
MQTT marca `sent=1` solo después del PUBACK QoS-1 — si el device crashea
entre publish y PUBACK, el mensaje queda como `sent=0` y se reintenta en el
próximo reconnect.

| Columna | Tipo | Notas |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | Orden cronológico implícito |
| `topic` | TEXT | `counting`, `telemetry`, o `wifi_ble` |
| `payload` | TEXT | `json.dumps(envelope)` — ver [api_contracts.md](api_contracts.md) para el shape |
| `created_at` | REAL | epoch seconds, para purge >72h |
| `sent` | INTEGER | 0 = pendiente, 1 = acked |

**Indexes**:
- `idx_sent_id (sent, id)` — `get_pending()` escanea `sent=0` ordenado por id
- `idx_created_at (created_at)` — `purge_old()` borra rows con `created_at < cutoff`

**PRAGMA**: WAL mode (readers no bloquean writers, mejor resilencia a
crashes), `synchronous=NORMAL` (balance entre durabilidad y throughput).

**Capacity**: el cap del backlog se mantiene por purge >72h (con vacuum diario
adicional). En 72h offline un device típico acumula ~2-5k mensajes — cabe
holgado.

## `hash_groups` — stitching state (`src/wifi_ble/dedup.py`)

State persistente del dedup con stitching para combatir MAC randomization. Una
fila por `(hash, protocol)`; el `group_id` es la agrupación de identidad
inferida por las **4 reglas** (seqnum continuity, cross-protocol L2, BLE
anchoring, fingerprint continuity — ver CLAUDE.md).

| Columna | Tipo | Notas |
|---|---|---|
| `hash` | TEXT | Parte del PK. SHA-256 truncado a 16 bytes (nunca MAC cruda) |
| `protocol` | TEXT | Parte del PK. `wifi` o `ble` |
| `group_id` | TEXT | UUID del grupo. Múltiples (hash, protocol) comparten group_id si las reglas los mergearon |
| `first_seen` | REAL | epoch seconds — primer aparición del hash |
| `last_seen` | REAL | epoch seconds — última observación. Drives las windows (2s cross-protocol, 30s seqnum, 15min BLE anchoring/fingerprint) |
| `rssi` | REAL | RSSI de la **última** observación. La usan los deltas de RSSI del stitching (quieren la señal reciente) |
| `max_rssi` | REAL | RSSI **más fuerte** vista. `get_traffic_counts` clasifica passerby/shopper sobre esta → conteo estable (un device que estuvo cerca cuenta de forma monótona, sin flapping) |
| `seqnum` | INTEGER | 802.11 sequence number (12 bits, del header `dot11.SC >> 4`). NULL para BLE. Regla seqnum continuity chequea Δseqnum ≤ 100 mod 4096 |
| `fingerprint` | TEXT | Fingerprint estable (orden de IEs + caps en WiFi; company ID + Continuity de Apple en BLE). Regla 4: re-une rotaciones que el seqnum no agarra; filtro duro en la regla 1 |

**PK compuesto**: `(hash, protocol)` — el mismo hash en protocolos distintos
es entrada separada (un device puede emitir hashes WiFi y BLE diferentes
pero pertenece al mismo `group_id`).

**Indexes**:
- `idx_hash_groups_group (group_id)` — `get_stitching_ratio()` cuenta DISTINCT groups
- `idx_hash_groups_last_seen (last_seen)` — windows del stitching
- `idx_hash_groups_fp (protocol, fingerprint)` — lookup de la regla 4

**Rotación**: `reset_daily()` hace `DELETE FROM hash_groups` (wipe completo).
Lo dispara `people-counter-reset.timer` a las 04:00 vía
`scripts/reset_dedup.py` (config-aware: resuelve el path del sqlite igual que
el pipeline). Boundary diario de privacidad + analytics; el stitching
cross-día no tiene sentido con MAC/RPA rotando igual.

**Privacy critical**: este archivo es el ÚNICO lugar que guarda el seqnum y
las marcas temporales fine-grained. Nunca se publica por MQTT — el publisher
emite solo el count agregado de `DISTINCT group_id` (`passersby`, `shoppers`).

## ¿Algo que limpiar acá?

**No**. Ambos schemas son lean:
- `messages` tiene 5 columnas, todas usadas en el lifecycle
  enqueue → get_pending → mark_sent → purge_old.
- `hash_groups` tiene 9 columnas, cada una sirve a una regla del stitching
  (seqnum, cross-protocol, BLE anchoring, fingerprint), a la clasificación
  passerby/shopper (`max_rssi`) o a la rotación diaria.

El cleanup del payload server-side (drop de scaling_factor, total_in, etc.)
no afecta al SQLite local: `messages.payload` es JSON blob — los campos
extra que el código viejo escribe se ignoran silenciosamente cuando el
Lambda los procesa con `data.get()`. **No requiere wipe del buffer al
deployar el código nuevo**.

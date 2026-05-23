# Schema de la base de datos

Modelo ER de las dos databases del sistema:

1. **Postgres (RDS, cloud)** — 4 tablas raw + vistas derivadas
2. **SQLite (device, local)** — 2 tablas para outbox MQTT y stitching state

DDL fuente:
- Postgres: [`infra/sql/bootstrap.sql`](../infra/sql/bootstrap.sql)
- SQLite outbox: [`src/mqtt/buffer.py`](../src/mqtt/buffer.py) `_ensure_db()`
- SQLite stitching: [`src/wifi_ble/dedup.py`](../src/wifi_ble/dedup.py) `_ensure_db()`

Snapshot del 2026-05-18 post-cleanup del schema (drop de columnas dead, bucket
columns uniformizadas, `pos_transactions` agregada para T9.11 / US-06).

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
        timestamptz bucket_15min    "device-aligned, join key temporal"
        timestamptz bucket_hour     "GENERATED desde event_ts"
        date        bucket_day      "GENERATED desde event_ts"
        text        direction       "CHECK in_out"
        int         track_id
        real        confidence      "debug: investigar FPs"
        text        height_class    "CHECK adult_child_unknown, US-05"
        real        height_m        "debug: detectar drift de mounting_height_m"
        timestamptz received_at
    }

    wifi_ble_summary {
        uuid        summary_id      PK
        text        device_id
        text        store_id        "join key"
        timestamptz period_start    "inicio ventana real medida"
        timestamptz period_end      "fin ventana real medida"
        timestamptz bucket_15min    "device-aligned, join key temporal"
        timestamptz bucket_hour     "GENERATED desde period_start"
        date        bucket_day      "GENERATED desde period_start"
        int         passersby       "post stitching (4 reglas)"
        int         shoppers        "RSSI cercano (por max_rssi)"
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
        int         death_emit_count         "counts via fallback del counter"
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

    count_events     ||--o{ wifi_ble_summary  : "store_id + bucket_15min (turn-in rate)"
    count_events     ||--o{ pos_transactions  : "store_id + bucket_15min (conversion rate)"
    count_events     ||--o{ telemetry         : "device_id (mismo device)"
    wifi_ble_summary ||--o{ telemetry         : "device_id (mismo device)"
```

## Modelo de joins

**Las tablas de hechos no tienen foreign keys hacia las dimensiones** —
`store_id` y `device_id` son `TEXT` libres en `count_events`,
`wifi_ble_summary`, `telemetry`, `pos_transactions`. Es deliberado: la Lambda
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
- **`bucket_15min`**: presente en `count_events`, `wifi_ble_summary` y
  `pos_transactions`. Mismo TIMESTAMPTZ alineado a múltiplos de 15 min del
  epoch UTC en todas las tablas → joins temporales sin `date_trunc`.

### Modelo de buckets

Tres granularidades soportadas vía columnas dedicadas para que Grafana queries
no necesiten `date_trunc`:

| Columna | Tipo | Origen |
|---|---|---|
| `bucket_15min` | TIMESTAMPTZ | Device-aligned en count_events/wifi_ble (lo manda el device pre-calculado desde `analytics.bucket_seconds`). Server-derived GENERATED en pos_transactions (el POS no conoce el shadow). |
| `bucket_hour` | TIMESTAMPTZ | GENERATED ALWAYS AS STORED — `date_trunc('hour', event_ts)` en todas las tablas. |
| `bucket_day` | DATE | GENERATED ALWAYS AS STORED — `date_trunc('day', event_ts)::date` en todas las tablas excepto telemetry (no aplica naturalmente a samples de 5min). |

## Vistas derivadas

```
counter (count_events)                  POS (pos_transactions)
  |                                       |
  +- counting_by_bucket   (15min)         +- (sin vista propia, agregado
  +- counting_hourly                      |   directo en conversion_rate)
  +- counting_daily       (+breakdown adult/child)
            |
            |       wifi_ble_summary
            |         |
            |         +- wifi_ble_store_traffic (MAX multi-cam dedup)
            |              |
            +--------------+-- turn_in_rate_by_bucket     -> US-04
            |
            +------ pos_transactions ----- conversion_rate_by_store  (15min)  -> US-06
                                           conversion_rate_hourly
                                           conversion_rate_daily
```

| Vista | Granularidad | Cierra | Notas |
|---|---|---|---|
| `counting_by_bucket` | 15min | --- | Ins/outs/net por store + bucket |
| `counting_hourly` | 1h | --- | Rollup desde `bucket_hour` |
| `counting_daily` | 1d | US-05 | Incluye breakdown `ins_adult` / `ins_child` |
| `wifi_ble_store_traffic` | 15min | --- | MAX (no SUM) por store, multi-cam dedup |
| `turn_in_rate_by_bucket` | 15min | US-04 | `ins / passersby` y `ins / shoppers` |
| `conversion_rate_by_store` | 15min | US-06 | `sales / visits` + amounts net/gross |
| `conversion_rate_hourly` | 1h | US-06 | Rollup |
| `conversion_rate_daily` | 1d | US-06 | Rollup |

## Idempotencia (UNIQUE constraints)

| Tabla | Constraint | Caso de uso |
|---|---|---|
| `count_events` | `(device_id, event_ts, track_id, direction)` | Replay del buffer SQLite del device cuando reconecta |
| `wifi_ble_summary` | `(device_id, period_start, period_end)` | Retry del WifiBlePublisher cada 15min |
| `telemetry` | `(device_id, event_ts)` | Retry de IoT Topic Rule |
| `pos_transactions` | `(store_id, transaction_id)` | Retry del POS o re-envío del batch |

Todas las Lambdas usan `INSERT ... ON CONFLICT DO NOTHING`.

## Usuarios IAM auth (least privilege)

| User Postgres | Lambda que lo usa | Grants |
|---|---|---|
| `lambda_writer` | `persist_event` (IoT → counter + wifi/ble + telemetry) | `INSERT, SELECT` sobre `count_events`, `wifi_ble_summary`, `telemetry` |
| `lambda_pos_writer` | `ingest_pos_transaction` (API Gateway → POS) | `INSERT, SELECT` sobre `pos_transactions` |

Ambos usuarios tienen `rds_iam` GRANT (auth IAM sin password). Los tokens
duran ~15min, las Lambdas los regeneran transparentemente via
`rds.generate_db_auth_token`.

## Index strategy

Por tabla, además de los PKs y UNIQUE constraints implícitos:

- **count_events**: `(store_id, event_ts DESC)`, `(device_id, event_ts DESC)`,
  `(store_id, bucket_15min DESC)`, `(store_id, bucket_day DESC)`
- **wifi_ble_summary**: `(store_id, bucket_15min DESC)`, `(store_id, bucket_day DESC)`
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

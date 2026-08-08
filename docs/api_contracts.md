# Contratos de API

Documenta las dos superficies de ingreso de datos al cloud:

1. **MQTT (device → cloud)** — 3 topics que el device publica via AWS IoT Core
2. **REST (POS externo → cloud)** — `POST /pos/transactions` via API Gateway

Ambos terminan persistiendo en Postgres (RDS) vía Lambdas distintas. Ver
[`docs/database_schema.md`](database_schema.md) para el destino tabla por
tabla.

---

# Parte 1 — MQTT (device → cloud)

## Auth y transporte

- **Broker**: AWS IoT Core endpoint (un endpoint por cuenta AWS, descubrir con
  `aws iot describe-endpoint --endpoint-type iot:Data-ATS`)
- **Protocolo**: MQTT 3.1.1 sobre TLS mutuo (X.509 client cert)
- **QoS**: 1 (at-least-once, con PUBACK del broker antes de marcar `sent=1` en
  el buffer SQLite)
- **Cert provisioning**: `scripts/provision.py create` genera el cert + lo
  attacha a un Thing en IoT Core con la policy `people-counter-device-${Environment}`

## Envelope estándar

Todos los topics comparten el mismo envelope (construido por
`MQTTClient.publish_event()` en [`src/mqtt/client.py`](../src/mqtt/client.py)):

```json
{
    "device_id": "store-001-cam-01",
    "timestamp": 1779897135.123,
    "type": "counting",
    "data": { /* topic-specific payload */ }
}
```

| Campo | Tipo | Notas |
|---|---|---|
| `device_id` | string | Cargado del config del device. Convención: `<store_id>-cam-<n>` |
| `timestamp` | number | epoch seconds float — momento del publish (no del evento; el evento real lleva su propio `event_time` dentro de `data`) |
| `type` | string | `counting`, `telemetry`, o `wifi_ble` — la Topic Rule de IoT mapea por type |
| `data` | object | Payload específico del topic (shapes abajo) |

`store_id` se infiere server-side por `_infer_store_id(device_id)` (split en
`-cam-`). Si el device_id no matchea la convención, `store_id == device_id`.

## Topic: `counting` — eventos de cruce de línea

**Topic MQTT**: `store/{store_id}/counting`
**Frecuencia**: por evento (sub-segundo durante traffic alto, idle el resto)
**Tabla destino**: `count_events`

### `data` shape

```json
{
    "direction": "in",
    "track_id": 42,
    "event_time": 1779897130.456,
    "height_m": 1.78,
    "confidence": 0.91
}
```

| Campo | Tipo | Requerido | Notas |
|---|---|---|---|
| `direction` | string | ✓ | `"in"` o `"out"` — qué dirección cruzó la línea |
| `track_id` | int | recomendado | ID interno del tracker. Junto con `(device_id, event_ts, direction)` forma el UNIQUE constraint para idempotencia |
| `event_time` | number | ✓ | epoch seconds del cruce real (no del publish). Si falta, fallback al `timestamp` del envelope. **El device manda timestamp crudo** — Postgres deriva `bucket_15min`, `bucket_hour`, `bucket_day` server-side via columnas GENERATED. |
| `height_m` | number | nullable | Altura cruda del sujeto en metros. La categorización por rango de estatura (`adult`/`child`/`unknown`) la aplica la función SQL `height_class(height_m)` server-side — el device solo persiste la medición cruda. Threshold centralizado, modificable retroactivo a toda la historia con `CREATE OR REPLACE FUNCTION`. |
| `confidence` | number | nullable | Score del detector \[0, 1\]. Debug-only |

**Compatibilidad legacy**: la Lambda persist_event acepta también la key
`event_bucket` (nombre anterior a 2026-05-18) y `height_class` (campo
removido en 2026-05-26 — categorización ahora vive server-side). Durante
un rollout escalonado del firmware, devices viejos siguen ingestiando OK
con esos campos silenciosamente ignorados.

## Topic: `telemetry` — health metrics del device

**Topic MQTT**: `store/{store_id}/telemetry`
**Frecuencia**: cada 5 minutos (configurable via `telemetry.interval_seconds`)
**Tabla destino**: `telemetry`

### `data` shape

```json
{
    "uptime_s": 86400.5,
    "cpu_temp_c": 56.4,
    "hailo_temp_c": 48.2,
    "disk_free_mb": 12345,
    "mem_available_mb": 1820,
    "throttled_flags": 0,
    "arm_clock_mhz": 2400,
    "fan_rpm": 5900,
    "power_w": 6.1,
    "ext5v_v": 5.12,
    "fs_readonly": false,
    "service_restarts": 0,
    "clock_synchronized": true,
    "cam_left_ok": true,
    "cam_right_ok": true,
    "fps": 11.8,
    "frame_latency_p50_ms": 78.2,
    "frame_latency_p95_ms": 142.5,
    "detection_rate_per_min": 23.4,
    "tracker_confirmed_count": 0,
    "tracker_pending_count": 1,
    "total_in": 142,
    "total_out": 138,
    "mqtt_connected": true,
    "mqtt_disconnect_count": 0,
    "seconds_since_last_reconnect": 86230.5,
    "buffer_backlog_messages": 0,
    "wifi_probe_ok": true,
    "ble_scanner_ok": true,
    "wifi_probe_rate_per_min": 42.0,
    "wifi_ble_stitching_ratio": 0.34,
    "track_stitching_ratio": 1.05,
    "death_emit_count": 2,
    "ghost_adoption_count": 5,
    "last_shadow_apply_ts": 1779579468.4,
    "error": null,
    "schedule_error_detail": null
}
```

| Categoría | Campos |
|---|---|
| **OS** | `uptime_s`, `cpu_temp_c`, `hailo_temp_c`, `disk_free_mb`, `mem_available_mb` |
| **Hardware** | `throttled_flags` (bitmask `get_throttled`: 0=sano, undervolt/throttle/soft-temp + sticky), `arm_clock_mhz` (cae al throttlear), `fan_rpm` (0+temp alta=fan clavado), `power_w` (consumo PMIC), `ext5v_v` (sag=fuente/cable flojo). Detección de fallas de hardware silenciosas en device de techo desatendido |
| **Sistema** | `fs_readonly` (microSD fallando → root fs read-only), `service_restarts` (NRestarts del service, crash-loop si crece), `clock_synchronized` (NTP; false sostenido = timestamps en riesgo). Fallas "corre pero roto" |
| **Cámaras** | `cam_left_ok` / `cam_right_ok` — cada cámara del par estéreo viva (false = SensorTimestamp congelado por ≥10 capturas → cable CSI / sensor muerto). Atribuye una caída de FPS a la cámara culpable |
| **Pipeline** | `fps`, `frame_latency_p50_ms`, `frame_latency_p95_ms`, `detection_rate_per_min`, `tracker_confirmed_count`, `tracker_pending_count` |
| **Counts** | `total_in`, `total_out` — running totals device-side desde el último boot. Sirve como sanity check vs `COUNT(*)` server-side |
| **MQTT** | `mqtt_connected`, `mqtt_disconnect_count`, `seconds_since_last_reconnect`, `buffer_backlog_messages` |
| **WiFi/BLE** | `wifi_probe_ok`, `ble_scanner_ok`, `wifi_probe_rate_per_min` (canary del monitor mode: 0 sostenido con `wifi_probe_ok=true` = nexmon mudo / radiotap caído), `wifi_ble_stitching_ratio` (canary: groups / hashes, baja a medida que el stitching mergea MAC rotations) |
| **Tracker (visión)** | `track_stitching_ratio` (unique_track_ids cruzando counting zone / counts emitidos, ideal ≈ 1.0, >1.3 = fragmentación), `ghost_adoption_count` (capa 1: rescates por ID adoption), `death_emit_count` (capa 3: rescates por death-emit). Las tres combinadas cierran el árbol diagnóstico (tracker perfecto / fragmentación rescatada / fragmentación sin rescate) |
| **Device Shadow** | `last_shadow_apply_ts` (canary: epoch del último delta del Device Shadow aplicado vía `apply_shadow_delta`. NULL hasta el primer push real post-boot — distinguir "shadow nunca usado" vs "última push hace mucho" en Grafana) |
| **Errors** | `error` (corto), `schedule_error_detail` (detalle largo). `null` cuando todo OK |

Todos los campos son `nullable` salvo `event_ts`. La Lambda inserta cualquier
subset que llegue (`data.get()` per campo).

**Idempotencia**: `UNIQUE (device_id, event_ts)` — retries de IoT con mismo
sample timestamp no duplican.

## Topic: `wifi_ble` — eventos per-device post-stitching

**Topic MQTT**: `store/{store_id}/wifi_ble`
**Frecuencia**: cada 15 minutos (configurable via `wifi_ble.summary_interval_seconds`)
**Tabla destino**: `wifi_ble_events`

### `data` shape (post 2026-05-26)

```json
{
    "period_start": 1779896700,
    "period_end": 1779897600,
    "devices": [
        {
            "visitor_hash": "deadbeef00112233445566778899aabb",
            "protocol": "wifi",
            "rssi_max": -55,
            "first_seen_ts": 1779896712.5,
            "last_seen_ts": 1779897588.1
        },
        {
            "visitor_hash": "ffee00cc01020304050607080a0b0c0d",
            "protocol": "ble",
            "rssi_max": -70,
            "first_seen_ts": 1779896720.0,
            "last_seen_ts": 1779897595.2
        }
    ]
}
```

| Campo | Tipo | Requerido | Notas |
|---|---|---|---|
| `period_start` | number | ✓ | epoch seconds, inicio de la ventana de emisión (alineado al múltiplo de `summary_interval_seconds`). |
| `period_end` | number | ✓ | epoch seconds, fin de la ventana de emisión. |
| `devices[]` | array | ✓ | Un elemento por cada visitor observado en la ventana (post-stitching local). Puede ser `[]` si la ventana fue vacía — Lambda devuelve 200 OK sin insertar. |
| `devices[].visitor_hash` | string | ✓ | 32 hex chars (= 16 bytes) — el `group_id` UUID del `DedupEngine` local del device. Opaco, no invertible a MAC. Se renueva cada día con `reset_daily()`. |
| `devices[].protocol` | string | ✓ | `"wifi"` o `"ble"` — protocolo del miembro del group con MAX rssi (el "dominante" del visitor). |
| `devices[].rssi_max` | int | ✓ | RSSI máximo lifetime del group hasta el cierre de la ventana, en dBm. La categorización shopper/passerby la aplica la función SQL `rssi_class(rssi_max)` server-side. |
| `devices[].first_seen_ts` | number | ✓ | epoch seconds — MIN sobre los miembros del group (primera observación de cualquier rotación). |
| `devices[].last_seen_ts` | number | ✓ | epoch seconds — MAX sobre los miembros (cae dentro de la ventana actual; **Postgres deriva `bucket_15min` server-side** desde este timestamp). |

**Semántica per-window emission**: el `DedupEngine` filtra por `MAX(last_seen) IN window` —
emite UN evento por cada group observado en la ventana actual, no sólo los nuevos.
Caso clave: visitor presente en 2 ventanas seguidas → 2 rows en RDS con el mismo
`visitor_hash` (uno por ventana). Day-level `COUNT(DISTINCT visitor_hash)` dedupa
correctamente al 1 visitor único. Crucialmente, captura promociones passerby→shopper
cross-window: si en ventana N rssi_max=-75 y en N+1 rssi_max=-50, ambas filas
aparecen con sus respectivas clasificaciones.

**Privacy nota**: este es el ÚNICO output del subsystema WiFi/BLE que sale del
device. Las MACs crudas nunca salen del Pi — el hashing pasa en `process_detection`
con salt diaria local. El `visitor_hash` que llega a RDS es el `group_id` del
stitching local, opaco. Los hashes individuales pre-stitching, RSSI per-detection
y seqnums quedan en el SQLite local (`wifi_ble_dedup.sqlite`) y se purgan
diariamente con `reset_daily()`. Nunca cruzan a la nube.

**Idempotencia**: `UNIQUE (device_id, visitor_hash, period_start)` — el publisher
puede reintentar la misma ventana sin duplicar. La Lambda usa `ON CONFLICT DO
UPDATE` con `GREATEST` sobre `rssi_max` y `last_seen_ts` para que retries con
información más reciente refinen el row (cubre el caso "MQTT falló mid-batch y
en el retry tenemos una observación posterior con RSSI más fuerte").

---

# Parte 2 — REST API (POS externo → cloud)

## Auth y transporte

- **Endpoint** (custom domain): `https://api.${DomainName}/pos/transactions`
  — ej. `https://api.<tu-dominio>/pos/transactions` (output del stack:
  `IngestPosCustomUrl`). Se crea cuando se pasa `ApiCertArn` al deploy; la
  firma SigV4 funciona igual sobre el custom domain que sobre el default.
- **Endpoint** (default, fallback): `https://${ApiId}.execute-api.${Region}.amazonaws.com/pos/transactions`
  (output `IngestPosApiUrl`). Activo siempre, aunque no haya custom domain.
- **Protocolo**: HTTPS (TLS terminado por API Gateway, cert ACM regional)
- **Auth**: **AWS SigV4** (IAM) — el POS firma cada request con AccessKey +
  SecretKey de un IAM principal que tenga `execute-api:Invoke` sobre la API
- **Sin auth válida**: API Gateway devuelve 403 antes de invocar la Lambda

### Provisioning del cliente POS

1. Crear IAM user dedicado: `pos-ingest-${StoreId}`
2. Attach policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": "execute-api:Invoke",
       "Resource": "arn:aws:execute-api:${region}:${account}:${api-id}/*/POST/pos/transactions"
     }]
   }
   ```
3. Generar AccessKey + SecretKey, entregar al integrador del POS
4. El POS firma con SigV4 (la mayoría de SDKs de AWS lo hacen automático;
   con `requests` Python: usar `requests-aws4auth`)

## `POST /pos/transactions`

**Tabla destino**: `pos_transactions`
**Lambda**: `ingest_pos_transaction` ([`src/cloud/ingest_pos_transaction.py`](../src/cloud/ingest_pos_transaction.py))

### Body — single transaction

```json
{
    "transaction_id": "POS-RECOLETA-20260518-001234",
    "store_id": "ar-recoleta",
    "event_ts": "2026-05-18T14:32:15-03:00",
    "type": "sale",
    "items": 2,
    "amount_minor": 4500000,
    "currency": "ARS",
    "payment_method": "credit_card"
}
```

### Body — bulk (array)

```json
[
    { "transaction_id": "POS-001", "store_id": "ar-recoleta", "event_ts": "2026-05-18T14:32:15-03:00", "type": "sale",   "items": 1, "amount_minor": 1500000 },
    { "transaction_id": "POS-002", "store_id": "ar-recoleta", "event_ts": "2026-05-18T14:35:42-03:00", "type": "return", "items": 1, "amount_minor": 800000  },
    { "transaction_id": "POS-003", "store_id": "ar-recoleta", "event_ts": "2026-05-18T14:38:10-03:00", "type": "sale",   "items": 3, "amount_minor": 6200000 }
]
```

**A discreción del POS**: si el POS prefiere mandar transaction-by-transaction
en real-time, usa el shape single (una request por tx). Si prefiere batchear
(por cierre de turno, por hora, etc.), manda array. **El conversion rate
funciona en ambos casos** — solo cambia la granularidad máxima de drill-down
(si el batch es horario, no podés ver conversion en buckets de 15min para
ese rango).

### Fields

| Campo | Tipo | Requerido | Notas |
|---|---|---|---|
| `transaction_id` | string | ✓ | Factura individual o batch_id del POS. UNIQUE por `(store_id, transaction_id)` — retries no duplican |
| `store_id` | string | ✓ | Mismo identificador que usa el counter device en este local. Crítico para que el JOIN con `count_events` funcione |
| `event_ts` | string \| number | ✓ | ISO-8601 con offset explícito (`"2026-05-18T14:32:15-03:00"` o `"...Z"` para UTC) **o** epoch numérico (segundos o milisegundos, autodetect por magnitud). **Rechazado**: ISO-8601 sin timezone (ej. `"2026-05-18T14:32:15"`) |
| `type` | string | ✓ | `"sale"` o `"return"`. Otros valores → 400 |
| `items` | int | opcional (default 1) | Cantidad de unidades. Para batch agregado, suma del batch. CHECK >= 0 |
| `amount_minor` | int | ✓ | Monto en **centavos** (no float, para evitar precisión decimal). CHECK >= 0 |
| `currency` | string | opcional (default `"ARS"`) | ISO 4217, 3 chars, uppercase. Truncado a 3 chars |
| `payment_method` | string | opcional, nullable | Texto libre. Para batches con métodos mixtos, mandar `null` o `"mixed"` |

### Response — 200 OK

```json
{
    "total": 3,
    "inserted": 2,
    "conflicted": 1
}
```

| Campo | Notas |
|---|---|
| `total` | Cantidad de transacciones recibidas en el body |
| `inserted` | Cantidad efectivamente insertada |
| `conflicted` | Cantidad ignorada por UNIQUE constraint (retry de un `transaction_id` ya conocido). **No es error** — es idempotencia funcionando |

### Response — 400 Bad Request

```json
{"error": "type debe ser uno de ('sale', 'return'), got 'invalid_type'"}
```

Triggers:
- Body vacío o JSON inválido
- Body no es objeto ni array
- Array vacío
- Required field faltante
- `type` fuera de `{"sale", "return"}`
- `amount_minor < 0` o no-int
- `items < 0` o no-int
- `event_ts` sin timezone explícita o formato inválido

**Validación atómica**: si UNA transacción del array es inválida, **ninguna**
se inserta. El POS debe corregir el batch entero y reintentar.

### Response — 5xx (errores transitorios)

| Status | Caso | Acción del POS |
|---|---|---|
| 403 | SigV4 inválido / IAM no autorizado | Revisar credentials |
| 500/502/503 | DB unreachable, token IAM expirado mid-request, Lambda timeout | Reintentar con backoff exponencial. El UNIQUE constraint garantiza que el retry no duplica |

### Idempotencia

`INSERT ... ON CONFLICT (store_id, transaction_id) DO NOTHING`. El POS puede
reintentar el mismo body sin riesgo de duplicar. **Caveat de v1**: si el POS
manda corrección con mismo `transaction_id` pero datos diferentes (ej.
amount corregido), se ignora silenciosamente. Para correcciones reales usar
un `transaction_id` distinto (ej. `POS-001-CORRECTED`).

### Bucket columns derivadas

El POS solo manda `event_ts`. Server-side, Postgres deriva `bucket_15min`,
`bucket_hour` y `bucket_day` (UTC) como `GENERATED ALWAYS AS ... STORED`
(calculadas en INSERT, indexables). Se computan con aritmética de epoch
sobre un interval —`extract(epoch FROM (event_ts - TIMESTAMPTZ 'epoch'))`—
porque `date_trunc`/`extract` directos sobre `timestamptz` son STABLE (no
IMMUTABLE) y Postgres los rechaza en una generated column.

**Aplica idéntico** a `count_events.bucket_15min` (desde `event_ts`) y
`wifi_ble_events.bucket_15min` (desde `last_seen_ts`): todas las columnas
de bucket son server-derived, el device manda timestamps crudos. JOIN
`count_events.bucket_15min = wifi_ble_events.bucket_15min = pos_transactions.bucket_15min`
sin recomputar nada en queries.

### Ejemplos de cliente

**curl con SigV4** (`--aws-sigv4`, igual que el query de aggregates en
[`api_access.md`](api_access.md)):

```bash
curl --aws-sigv4 "aws:amz:us-east-1:execute-api" \
     --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
     -H "Content-Type: application/json" \
     -d '{"transaction_id":"POS-RECOLETA-20260518-001234","store_id":"ar-recoleta","event_ts":"2026-05-18T14:32:15-03:00","type":"sale","items":2,"amount_minor":4500000,"currency":"ARS","payment_method":"credit_card"}' \
     "https://api.<tu-dominio>/pos/transactions"
```

**Python con `requests-aws4auth`** (recomendado para integraciones):

```python
import requests
from requests_aws4auth import AWS4Auth

auth = AWS4Auth(
    "AKIA...",                 # access_key del IAM user pos-ingest-*
    "secret...",               # secret_key
    "us-east-1",
    "execute-api",
)

url = "https://api.<tu-dominio>/pos/transactions"  # o la URL execute-api default

tx = {
    "transaction_id": "POS-RECOLETA-20260518-001234",
    "store_id": "ar-recoleta",
    "event_ts": "2026-05-18T14:32:15-03:00",
    "type": "sale",
    "items": 2,
    "amount_minor": 4500000,
    "currency": "ARS",
    "payment_method": "credit_card",
}

r = requests.post(url, auth=auth, json=tx, timeout=30)
print(r.status_code, r.json())
# 200 {'total': 1, 'inserted': 1, 'conflicted': 0}
```

**Node.js con AWS SDK v3** (`@aws-sdk/signature-v4`):

```javascript
import { SignatureV4 } from "@aws-sdk/signature-v4";
import { Sha256 } from "@aws-crypto/sha256-js";
import { HttpRequest } from "@aws-sdk/protocol-http";

const signer = new SignatureV4({
    credentials: { accessKeyId: "AKIA...", secretAccessKey: "secret..." },
    region: "us-east-1",
    service: "execute-api",
    sha256: Sha256,
});

const request = new HttpRequest({
    method: "POST",
    hostname: "api.<tu-dominio>",
    path: "/pos/transactions",
    headers: { "content-type": "application/json", host: "api.<tu-dominio>" },
    body: JSON.stringify(tx),
});

const signed = await signer.sign(request);
// fetch con los headers firmados...
```

# Infra AWS — People Counter

Stack cloud que recibe los mensajes del dispositivo edge, deduplica entre cámaras y deja la serie de tiempo lista para QuickSight.

```
RPi5 ──MQTT/TLS──► IoT Core ──┬─► Rule "counting"      ─► Timestream (counting_events)
                              ├─► Rule "telemetry"     ─► Timestream (telemetry)
                              ├─► Rule "wifi_ble"      ─► Lambda dedup L3 ─► DynamoDB
                              └─► Shadow $aws/things   ◄► dispositivo (operating_hours, counting_enabled)
```

Una sola plantilla CloudFormation (`cloudformation/people-counter.yaml`) crea **todo** el stack: IoT Policy + ThingType, 3 Topic Rules, Timestream (DB + 2 tablas), DynamoDB de dedup, Lambda + rol IAM, alarmas CloudWatch. `terraform/` está reservado para si más adelante se migra.

---

## 1) Topics MQTT y schemas de payload

Los topics se interpolan en el dispositivo con `{store_id}` (ver `config/config.example.yaml → mqtt.topics`).

### `store/{store_id}/counting` — eventos de cruce de línea

Publicado en tiempo real cuando un track completa el cruce.

```json
{
  "device_id": "store-001-cam-01",
  "timestamp": 1762963200.123,
  "type": "counting",
  "data": {
    "track_id": 42,
    "direction": "in",          // "in" | "out"
    "confidence": 0.87,
    "height_class": "adult"     // "adult" | "child" | "unknown"
  }
}
```

→ Rule `IoTRuleCounting` escribe directo a Timestream `counting_events` con dimensiones `store_id`, `device_id`, `direction`.

### `store/{store_id}/telemetry` — salud del dispositivo

Cadencia default `3600s` (1h, configurable via `telemetry.interval_seconds`).

```json
{
  "device_id": "store-001-cam-01",
  "timestamp": 1762963200.0,
  "type": "telemetry",
  "data": {
    "cpu_temp_c": 56.4,
    "hailo_temp_c": 48.2,
    "disk_used_pct": 31.0,
    "uptime_s": 142800,
    "fps": 14.8,
    "p50_latency_ms": 62.0,
    "p99_latency_ms": 145.0,
    "detection_rate_per_min": 12.3,
    "active_tracks": 3,
    "buffer_pending": 0,
    "mqtt_connected": true,
    "total_in": 187,
    "total_out": 184
  }
}
```

→ Rule `IoTRuleTelemetry` escribe a Timestream `telemetry`.

### `store/{store_id}/wifi_ble` — resúmenes de probing

Cadencia `wifi_ble.probe_interval_seconds` (default 900s = 15min). Se manda un mensaje por protocolo y solo si la ventana produjo hashes (windows vacías no se publican).

```json
{
  "device_id": "store-001-cam-01",
  "timestamp": 1762963200.0,
  "type": "wifi_ble",
  "data": {
    "hashes": ["a1b2c3d4...", "e5f6..."],   // SHA-256 truncado a 16 bytes, hex
    "protocol": "wifi",                     // "wifi" | "ble"
    "period_start": 1762962300,             // epoch seconds
    "period_end":   1762963200
  }
}
```

→ Rule `IoTRuleWifiBle` invoca la Lambda de dedup L3. La Lambda hace `put_item` condicional contra DynamoDB (PK: `store_id#date`, SK: hash); cuenta `new_count` y `duplicate_count` y los devuelve.

**Reglas duras:**
- **Nunca MACs crudas.** El hashing pasa en el dispositivo antes de bufferear.
- **Hashes deduplicados localmente** (capas L1 intra-protocolo + L2 cross-protocolo en SQLite) antes de mandarse — la Lambda solo agrega la L3 inter-cámara.
- **Ventanas disjuntas.** El publisher arranca cada ventana en `last_period_end` previo, así un hash con `first_seen=T` aparece en exactamente una emisión.

### Shadow — `$aws/things/{device_id}/shadow/update` (y `/delta`)

Estado `desired` empujable desde la nube (whitelist en `src/config/loader.py`):

```json
{ "state": { "desired": { "counting_enabled": false, "operating_hours": { ... } } } }
```

Solo dos keys son pusheables — todo lo demás (thresholds, geometría de ROI, modelo) se cambia editando `config.yaml` per-device y reiniciando. El runtime ACKea con `reported` después de aplicar el delta.

---

## 2) Recursos del stack

| Recurso | Propósito |
|---------|-----------|
| `IoTDevicePolicy` | Permisos por device — connect, publish a sus 3 topics, subscribe/receive al shadow propio. Se attachea al cert X.509 que aprovisiona `scripts/provision.py`. |
| `IoTThingType: people-counter` | Atributos searcheables (`store_id`, `firmware_version`). |
| `IoTRuleCounting` / `Telemetry` / `WifiBle` | Routing MQTT → backends. |
| `TimestreamDatabase` + `TimestreamCountingTable` + `TimestreamTelemetryTable` | Memoria 7d, magnético 365d (parametrizable). |
| `DedupTable` (DynamoDB) | PK `store_date`, SK `hash`, TTL 7d. |
| `LambdaDedup` (Python 3.13) | Handler en `src/cloud/lambda_dedup.py`. |
| Alarmas CloudWatch | Lambda errors, throttles, IoT rule failures. Opcionalmente notifica a un SNS topic (param `AlarmNotificationTopicArn`). |

---

## 3) Deploy

### Pre-requisitos

- AWS CLI ≥ 2.x autenticado contra la cuenta destino (`aws sts get-caller-identity` debe responder).
- Permisos en la sesión para crear: IoT, Timestream, DynamoDB, Lambda, IAM, CloudWatch.

El template **no** requiere ningún bucket S3 — el Lambda se crea con un placeholder inline y se actualiza después con `scripts/deploy_lambda.sh`.

### Parámetros del stack

Todos opcionales (tienen default):

| Parámetro | Default | Cuándo cambiarlo |
|-----------|---------|------------------|
| `Environment` | `prod` | `dev` / `staging` / `prod` — afecta nombres de Lambda y tags. |
| `TimestreamRetentionHoursMemory` | `168` (7d) | Subir si querés queries rápidas más allá de 7 días. |
| `TimestreamRetentionDaysMagnetic` | `365` | Bajar si la cuenta tiene presupuesto chico. |
| `DedupTTLDays` | `7` | Cuántos días persisten los hashes en DynamoDB. |
| `AlarmNotificationTopicArn` | `""` | ARN de un SNS topic para recibir alertas. Vacío = alarmas silenciosas (solo visibles en consola). |

### Pasos

```bash
# 1) Deploy del stack (Lambda queda con código placeholder).
aws cloudformation deploy \
  --template-file infra/cloudformation/people-counter.yaml \
  --stack-name people-counter-dev \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides Environment=dev

# 2) Reemplazar el placeholder con el código real de la Lambda.
scripts/deploy_lambda.sh dev

# 3) Aprovisionar el primer dispositivo (cert + thing + attach a policy).
py scripts/provision.py create \
  --device-id store-pilot-01-cam-01 \
  --store-id store-pilot-01 \
  --out /etc/people-counter/certs/
```

`scripts/provision.py` también tiene `deploy` (copia certs + config al device por SSH), `harvest` (extrae keys para disaster recovery), `reprovision` (rotación de cert) y `list`.

### Outputs del stack

Después del deploy, `aws cloudformation describe-stacks --stack-name people-counter-dev --query 'Stacks[0].Outputs'` devuelve:

- `IoTEndpoint` — usar en `config.yaml → mqtt.endpoint` de cada device.
- `TimestreamDatabase` — para conectar QuickSight.
- `DedupTableName`, `DedupLambdaArn`.
- `OperationsDashboardURL` — link directo al dashboard de CloudWatch.

---

## 4) Cómo verificar end-to-end

Después del deploy, validar con `--no-mqtt=false` (el default) en un dispositivo provisionado:

```bash
# En el device — mirar logs del servicio
journalctl -u people-counter -f | grep -E "publish|wifi_ble|telemetry"

# En la cuenta AWS — query a Timestream
aws timestream-query query \
  --query-string "SELECT COUNT(*) FROM \"people_counter\".\"counting_events\" \
                  WHERE time > ago(1h)"

# Dedup L3 — invocar Lambda con un payload sintético
aws lambda invoke \
  --function-name people-counter-dedup \
  --payload '{"device_id":"store-001-cam-01","store_id":"store-001",
              "date":"2026-05-12","type":"wifi_ble",
              "data":{"hashes":["abc","def"],"protocol":"wifi",
                      "period_start":1762962300,"period_end":1762963200}}' \
  /tmp/lambda_out.json
cat /tmp/lambda_out.json
```

---

## 5) Resiliencia

- **Sin conectividad:** `MessageBuffer` (SQLite outbox en `/var/lib/people-counter/buffer.sqlite`) persiste los publishes; al reconectar, `MQTTClient.replay_buffer` los drena en orden y marca enviado sólo al PUBACK.
- **Reconexión MQTT:** la maneja paho con backoff exponencial (`reconnect_delay_set 1-120s`). El device shadow se reconcilia automáticamente en el hook `on_connected` del cliente.
- **Lambda fallida:** IoT Rule tiene `ErrorAction` que loguea a CloudWatch. La alarma de Lambda errors dispara `AlarmNotificationTopicArn` si está configurado.
- **DynamoDB conflict:** `put_item` con `ConditionExpression="attribute_not_exists(#h)"` — si el hash ya existía, cuenta como `duplicate_count` sin error.

---

## 6) Costos estimados (orden de magnitud para 1 dispositivo PoC)

| Servicio | Consumo | Costo mensual aprox (us-east-1) |
|----------|---------|--------------------------------|
| IoT Core | ~10k mensajes/día (counting + telemetría 1h + 15min wifi/ble) | < 1 USD |
| Timestream | ~300 MB/mes counting + telemetry | 1-2 USD |
| DynamoDB | < 1 GB, on-demand | < 1 USD |
| Lambda | 96 invocaciones/día (cada 15min, 2 protocols) | gratis (free tier) |
| CloudWatch | logs + alarmas | < 1 USD |
| **Total** | | **~5 USD/mes/device** |

Escalar con cuidado: a 1000 devices, Timestream pasa a ser el ítem dominante (~150 USD/mes a write-rate constante).

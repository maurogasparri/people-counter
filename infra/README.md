# Infra AWS — People Counter (PoC)

Stack cloud que recibe mensajes del dispositivo edge, los persiste en Postgres
(RDS), y los expone en Grafana (App Runner). Todo en `us-east-1`.

```
RPi5 ──MQTT/TLS──► IoT Core ──┬─► Rule "counting"   ─┐
                              ├─► Rule "telemetry"  ─┼─► Lambda persist_event ─► RDS Postgres 16
                              ├─► Rule "wifi_ble"   ─┘      (IAM auth)            (db.t4g.micro)
                              │
                              └─► Shadow $aws/things ◄─► dispositivo (operating_hours, counting_enabled)

                                                            App Runner ─── Grafana 13 ──► dashboards
                                                            (custom domain)       ▲
                                                                                  │
                                                                  Postgres datasource (RDS)
```

**Decisiones de scope** (PoC con 1 device):

- **RDS Postgres + App Runner Grafana**, no self-hosted en EC2 — operabilidad >
  costo: snapshots automaticos, parche de SO/DB managed, restart sin perder
  state, scaling vertical sin downtime. App Runner sirve Grafana con custom
  domain + ACM auto-renovado.
- **Lambda fuera de VPC, IAM auth a RDS** — sin VPC connector ($7-14/mo de VPC
  endpoints). La Lambda usa `rds.generate_db_auth_token` para autenticar como
  el DB user `lambda_writer` (sin password almacenado).
- **Sin Lambda dedup L3** — innecesaria con 1 device/sucursal. El dedup local
  L1+L2 cubre monocam. Reintroducir cuando haya 2+ cams por store.
- **Payload WiFi/BLE reducido** — el device manda `{passersby, shoppers}` post
  L2 dedup, no hashes individuales. Privacidad + payload chico.

---

## 1) Topics MQTT

Los topics se interpolan en el dispositivo con `{store_id}` (ver
`config/config.example.yaml → mqtt.topics`). Los 3 son consumidos por la misma
Lambda; el `type` dentro del envelope discrimina la tabla destino.

| Topic | Cadencia | Tabla destino |
|---|---|---|
| `store/{store_id}/counting`  | Tiempo real (por cruce de linea)        | `count_events`     |
| `store/{store_id}/telemetry` | `telemetry.interval_seconds` (300s def) | `telemetry`        |
| `store/{store_id}/wifi_ble`  | `wifi_ble.probe_interval_seconds` (15min def) | `wifi_ble_summary` |

Shape del envelope (de `src.mqtt.client.MQTTClient.publish_event`):

```json
{
  "device_id": "store-pilot-01-cam-01",
  "timestamp": 1762963200.123,
  "type":      "counting",
  "data":      { ... shape per type ... }
}
```

El shape exacto del `data` por tipo esta documentado en
[`src/cloud/persist_event.py`](../src/cloud/persist_event.py) (cada `_insert_*`
mapea field por field) y en [`sql/bootstrap.sql`](sql/bootstrap.sql) (columnas
de cada tabla).

**Reglas duras**:
- Nunca MACs crudas — hashing local en el device antes de bufferear.
- Hashes nunca salen del device — solo agregados `{passersby, shoppers}`.

### Shadow — `$aws/things/{device_id}/shadow/update`

Pusheable desde la nube (whitelist en `src/config/loader.py → CLOUD_OVERRIDABLE`):

```json
{ "state": { "desired": { "counting_enabled": false, "operating_hours": { ... } } } }
```

Solo `counting_enabled` + `operating_hours`. El resto (thresholds, geometria,
modelo) se cambia editando `/etc/people-counter/config.yaml` per-device.

---

## 2) Recursos del stack

Definidos en [`cloudformation/people-counter.yaml`](cloudformation/people-counter.yaml).

| Recurso | Proposito |
|---|---|
| `VPC` + 2 public subnets + IGW   | Red para RDS (App Runner y Lambda van fuera de VPC). |
| `RdsInstance` (db.t4g.micro)     | Postgres 16, IAM auth + `rds.force_ssl=1`, PubliclyAccessible para Lambda + DBeaver. |
| `RdsMasterSecret`                | Password autogenerado en Secrets Manager (master user `people_counter`). |
| `RdsSecurityGroup`               | 5432 abierto a `AdminCidr` (DBeaver) + `0.0.0.0/0` (Lambda) — TLS + IAM gatekeep. |
| `IoTDevicePolicy`                | Connect, publish a los 3 topics, subscribe al shadow propio. |
| `IoTThingType`                   | `people-counter-${env}` con atributos `store_id`, `firmware_version`. |
| `IoTTopicRule` x3                | Routing MQTT → Lambda. Error action → CloudWatch Logs. |
| `PersistEventLambda`             | Python 3.13, 256MB, IAM auth a RDS. Code: `src/cloud/persist_event.py`. |
| `GrafanaEcrRepo`                 | `:latest` pushed por `deploy.ps1`. `EmptyOnDelete: true` para teardown limpio. |
| `GrafanaService` (App Runner)    | Grafana OSS 13 (oficial Docker image). Default egress, no VPC connector. |
| `GrafanaInstanceRole/AccessRole` | Permisos para que App Runner pulle de ECR + corra el container. |
| `AlertTopic` (SNS)               | Subs por email a alarmas (Lambda errors, IoT disconnect, App Runner health). |

---

## 3) Schema de Postgres

Definido en [`sql/bootstrap.sql`](sql/bootstrap.sql). 4 tablas + 6 views.

| Tabla | Granularidad | Idempotencia |
|---|---|---|
| `count_events`     | 1 row por cruce de linea          | `UNIQUE (device_id, event_ts, track_id, direction)` |
| `telemetry`        | 1 row por sample (5min def)       | sin constraint — duplicados aceptables |
| `wifi_ble_summary` | 1 row por ventana (15min def)     | `UNIQUE (device_id, period_start, period_end)` |
| `sales`            | 1 row por venta (futuro POS API)  | `UNIQUE (store_id, external_id)` |

Views (todas en `bootstrap.sql`):

- **`wifi_ble_store_traffic`** — agrega `wifi_ble_summary` por `(store_id, period_bucket)` con `MAX(passersby)` y `MAX(shoppers)`. Multi-cam dedup read-time: cuando un store tiene 2+ cams, las ventanas de WiFi/BLE solapan (~30-50m vs ~3-5m de vision), tomar el MAX usa la cam que mejor lo vio como estimador. Con 1 cam/store, MAX == el row de esa cam.
- **`counting_by_bucket`** — `ins / outs / net` por `(store_id, event_bucket)`. `event_bucket` es la columna que el device alinea al multiplo de `analytics.bucket_seconds` (15min def) — cambiar el bucket via shadow no requiere recomputar nada.
- **`turn_in_rate_by_bucket`** — `counting_by_bucket` ⨝ `wifi_ble_store_traffic` por bucket, calcula `turn_in_rate = ins / passersby` y `conversion_rate = ins / shoppers`. FULL OUTER JOIN preserva buckets donde solo una fuente reporto.
- **`counting_hourly`** / **`turn_in_rate_hourly`** — rollups por hora encima de las dos anteriores. Para reportes diarios donde 15min es demasiado fino.
- **`store_hourly_summary`** — counting + sales por hora por store. Conversion rate = ventas / personas. Util cuando `sales` empiece a tener datos via API Gateway.

### DB users + auth

- `people_counter` (master) — desde DBeaver, password en Secrets Manager.
- `lambda_writer` — sin password, `GRANT rds_iam` para auth via IAM token. La Lambda role tiene `rds-db:connect` sobre este user.
- DB separada `grafana` — owner `people_counter`, solo para state interno de Grafana (users, dashboards, sessions).

---

## 4) Deploy

Orchestrado por [`deploy.ps1`](deploy.ps1) en 6 fases con `-StartFromPhase` para
resumir interrupciones.

### Pre-requisitos

```powershell
aws sts get-caller-identity            # debe ser tu cuenta target
aws configure get region               # us-east-1
docker info                            # daemon corriendo (push a ECR + bootstrap SQL via docker psql)
```

App Runner requiere **plan pago de AWS** (no Free Account). Si la cuenta esta
en Free, hay que upgradear desde la consola — sin costo fijo, solo pay-as-you-go.

### Run

```powershell
.\infra\deploy.ps1
```

Fases:

1. **[1/6]** CFN deploy core — VPC + RDS + IoT + Lambda stub + ECR (sin App Runner).
2. **[2/6]** Push imagen `grafana/grafana:latest` a ECR + bootstrap SQL via `docker run postgres:16 psql`.
3. **[3/6]** CFN deploy con `DeployAppRunner=true` — agrega App Runner Service apuntando al ECR.
4. **[4/6]** `aws apprunner associate-custom-domain` + **pause manual** para agregar CNAMEs (validation records + final CNAME → DNSTarget) en tu DNS provider. Loop hasta que custom domain → `ACTIVE`.
5. **[5/6]** `aws apprunner update-service` con `RuntimeEnvironmentVariables` (Postgres backend para Grafana). El CFN no acepta env vars dinamicas con secret refs en early validation, asi que se setean por CLI post-deploy.
6. **[6/6]** Poll hasta que App Runner Service → `RUNNING`.

El unico step manual es el DNS — los CNAMEs hay que agregarlos en el DNS provider externo (gasparri.com.ar no esta en Route53). El script printea exactamente que records y donde, y pausea con `Read-Host`.

### Deploy del codigo real de Lambda

CFN crea el Lambda con un stub inline. Para deployar el codigo real:

```powershell
.\scripts\deploy_lambda.ps1 -Environment dev
```

Empaqueta `src/cloud/persist_event.py` + `psycopg[binary]` (manylinux x86_64) y
hace `aws lambda update-function-code`.

### Aprovisionar un device

```powershell
py scripts\provision.py --thing-name store-pilot-01-cam-01
```

Crea el cert X.509, attach a la policy, y descarga el cert + key a
`provisioned/<thing-name>/` (gitignored).

---

## 5) Verificar end-to-end

```powershell
# 1) Publicar un evento de prueba al topic real (bypassea el device)
aws iot-data publish `
    --topic "store/test-cam/counting" `
    --payload "fileb://test-event.json" `
    --cli-binary-format raw-in-base64-out

# 2) Confirmar que cae en Postgres
$secret = (aws secretsmanager get-secret-value `
    --secret-id (aws cloudformation describe-stacks `
        --stack-name people-counter-dev `
        --query "Stacks[0].Outputs[?OutputKey=='RdsMasterSecretArn'].OutputValue" `
        --output text) `
    --query SecretString --output text) | ConvertFrom-Json
$host = aws cloudformation describe-stacks --stack-name people-counter-dev `
    --query "Stacks[0].Outputs[?OutputKey=='RdsEndpoint'].OutputValue" --output text
docker run --rm -e PGPASSWORD=$($secret.password) -e PGSSLMODE=require postgres:16 `
    psql -h $host -U $secret.username -d people_counter `
    -c "SELECT COUNT(*), MAX(event_ts) FROM count_events;"

# 3) Abrir Grafana en https://grafana.<your-domain> y queriar count_events
```

---

## 6) Resiliencia

- **Sin conectividad del device** — buffer SQLite local persiste publishes. Replay al reconectar, marca enviado solo tras PUBACK.
- **Lambda fallida** — IoT Rule loguea a CloudWatch (`/aws/iot/people-counter-rule-errors-${env}`). IoT reintenta 1 vez built-in; despues drop. Alarma `PersistEventLambdaErrorAlarm` dispara con >0 errors / 5min.
- **RDS** — Multi-AZ desactivado para PoC ($13/mo vs $26). Daily snapshot (`BackupRetentionPeriod` default 7d, configurable). Restore via `aws rds restore-db-instance-to-point-in-time`.
- **App Runner** — auto-restarts si el container se cae (`HealthCheckConfiguration: Protocol HTTP / Path /api/health`). Rolling redeploy en cada `update-service` (~2-3 min).
- **Teardown / re-create** — `aws cloudformation delete-stack` borra todo. ECR esta marcado `EmptyOnDelete: true` para no atascarse en imagenes pendientes.

---

## 7) Costos PoC (1 device, us-east-1)

| Recurso | $/mes | Notas |
|---|---|---|
| RDS db.t4g.micro (single-AZ) | ~$13 | Storage 20GB gp3 incluido. |
| App Runner (1 vCPU / 2GB)    | ~$5  | Pay-per-request post-traffic; idle scale-down OFF para 100% uptime. |
| IoT Core                     | <$1  | $1/M msgs — PoC genera centavos. |
| Lambda                       | $0   | 1M invocations + 400k GB-s free **forever**. |
| Secrets Manager              | $0.40| 1 secret. |
| CloudWatch                   | $0   | 5 GB logs + 10 metrics free forever. |
| **Total estimado**           | **~$20/mo** | Post free tier. |

**Producción (flota)**: migrar RDS a Multi-AZ (~$26 + storage), considerar
Amazon Managed Grafana (SSO + IAM-integrated, ~$9/user/mo) en vez de OSS, y
reintroducir Lambda dedup L3 con DynamoDB cuando haya 2+ cams por store.

---

## 8) Proximos pasos

- Construir dashboards en Grafana UI sobre las views (`counting_by_bucket`,
  `turn_in_rate_hourly`, `store_hourly_summary`).
- API Gateway para ingest de sales desde POS externo (tabla `sales` ya
  preparada con `UNIQUE (store_id, external_id)` para idempotencia).
- Migracion a Route53 delegated subdomain (`tfg.gasparri.com.ar` NS → R53)
  para que CFN gestione DNS records y el deploy sea 100% sin pausa.

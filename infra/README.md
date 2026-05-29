# Infra AWS — People Counter (PoC)

Stack cloud que recibe mensajes del dispositivo edge, los persiste en Postgres
(RDS), y los expone en Grafana (ECS Fargate + ALB). Todo en `us-east-1`.

```
RPi5 ──MQTT/TLS──► IoT Core ──┬─► Rule "counting"   ─┐
                              ├─► Rule "telemetry"  ─┼─► Lambda persist_event ─► RDS Postgres 16
                              ├─► Rule "wifi_ble"   ─┘      (IAM auth)            (db.t4g.micro)
                              │
                              └─► Shadow $aws/things ◄─► dispositivo (operating_hours, counting_enabled)

                                            ALB (HTTPS, ACM cert) ──► ECS Fargate (Grafana 13) ──► dashboards
                                            custom domain                                              ▲
                                                                                                       │
                                                                                       Postgres datasource (RDS)
```

**Decisiones de scope** (PoC con 1 device):

- **RDS Postgres + ECS Fargate Grafana detrás de ALB**, no self-hosted en EC2 —
  operabilidad > costo: snapshots automaticos, parche de SO/DB managed, restart
  sin perder state, scaling vertical sin downtime. Fargate corre el container
  oficial de Grafana 13; el ALB termina HTTPS con ACM cert custom (custom
  domain `grafana.<DomainName>`) y forwardea a port 3000 del task.
- **Cert ACM creado fuera de CFN** — `deploy.ps1` corre `aws acm
  request-certificate` antes del deploy del Grafana stack y espera DNS
  validation. El ARN entra al CFN como parametro. Razon: si el cert vive
  adentro del stack con DnsValidation, `cloudformation deploy` bloquea
  esperando que el operador agregue los CNAMEs — peor UX que pausar el
  script con un Read-Host explicito.
- **Lambda fuera de VPC, IAM auth a RDS** — sin VPC connector ($7-14/mo de VPC
  endpoints). La Lambda usa `rds.generate_db_auth_token` para autenticar como
  el DB user `lambda_writer` (sin password almacenado).
- **Sin Lambda dedup L3** — innecesaria con 1 device/sucursal. El stitching
  local del device (hash groups con 4 reglas: seqnum continuity + cross-protocol
  L2 + BLE anchoring + fingerprint continuity) cubre monocam. Reintroducir
  cuando haya 2+ cams por store.
- **Payload WiFi/BLE per-device** — el device manda un array `devices[]` (un
  evento por visitor post-stitching) con `rssi_max` crudo + `visitor_hash`
  opaco (UUID random); la categorización passerby/shopper se aplica server-side
  via la función SQL `rssi_class(rssi_max)`. Nunca hashes pre-stitching ni MACs.

---

## 1) Topics MQTT

Los topics se interpolan en el dispositivo con `{store_id}` (ver
`config/config.example.yaml → mqtt.topics`). Los 3 son consumidos por la misma
Lambda; el `type` dentro del envelope discrimina la tabla destino.

| Topic | Cadencia | Tabla destino |
|---|---|---|
| `store/{store_id}/counting`  | Tiempo real (por cruce de linea)        | `count_events`     |
| `store/{store_id}/telemetry` | `telemetry.interval_seconds` (300s def) | `telemetry`        |
| `store/{store_id}/wifi_ble`  | `wifi_ble.summary_interval_seconds` (15min def, rango [30, 900]) | `wifi_ble_events` |

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
- Nunca MACs crudas — hashing local (SHA-256 + salt) en el device antes de bufferear.
- Los hashes de MAC nunca salen del device — solo el `visitor_hash` opaco (UUID random) + `rssi_max` crudo por visitor.

### Shadow — `$aws/things/{device_id}/shadow/update`

Pusheable desde la nube (whitelist en `src/config/loader.py → CLOUD_OVERRIDABLE`):

```json
{ "state": { "desired": { "counting_enabled": false, "operating_hours": { ... } } } }
```

Solo `counting_enabled`, `operating_hours` y `external_traffic_enabled`
(whitelist `CLOUD_OVERRIDABLE`). El resto (thresholds, geometria, modelo) se
cambia editando `/etc/people-counter/config.yaml` per-device. Los deltas con
valores inválidos se rechazan antes de persistir.

---

## 2) Recursos del stack

Definidos en [`cloudformation/people-counter.yaml`](cloudformation/people-counter.yaml).

| Recurso | Proposito |
|---|---|
| `VPC` + 2 public subnets + IGW   | Red para RDS, ALB y tasks ECS (Lambda va fuera de VPC). |
| `RdsInstance` (db.t4g.micro)     | Postgres 16.6, IAM auth + `rds.force_ssl=1` + `AutoMinorVersionUpgrade=true`, PubliclyAccessible para Lambda + DBeaver. |
| `RdsMasterSecret`                | Password autogenerado en Secrets Manager (master user `people_counter`). |
| `RdsSecurityGroup`               | 5432 abierto a `AdminCidr` (DBeaver) + `0.0.0.0/0` (Lambda + Fargate tasks) — TLS + IAM gatekeep. |
| `IoTDevicePolicy`                | Connect, publish a los 3 topics, subscribe al shadow propio. |
| `IoTThingType`                   | `people-counter-${env}` con atributos `store_id`, `firmware_version`. |
| `IoTTopicRule` x3                | Routing MQTT → Lambda. Error action → CloudWatch Logs. |
| `PersistEventLambda`             | Python 3.13, 256MB, IAM auth a RDS (user `lambda_writer`). Code: `src/cloud/persist_event.py`. Persiste counting/telemetry/wifi_ble. |
| `IngestPosLambda`                | Ingesta de transacciones POS (API Gateway). IAM auth como `lambda_pos_writer`. Code: `src/cloud/ingest_pos_transaction.py`. |
| `QueryAggregatesLambda`          | API de agregados read-only (API Gateway). IAM auth como `lambda_query_reader`. Code: `src/cloud/query_aggregates.py`. |
| `GrafanaEcrRepo`                 | `:latest` pushed por `deploy.ps1`. `EmptyOnDelete: true` para teardown limpio. |
| `GrafanaCluster` (ECS)           | Capacity provider FARGATE. |
| `GrafanaTaskDefinition`          | Cpu/Memory parametrizados, env vars + secret de RDS bakeados, log driver `awslogs`. |
| `GrafanaService` (ECS)           | DesiredCount=1, MinimumHealthyPercent=0 (rolling deploy con downtime corto en PoC single-instance), AssignPublicIp=ENABLED. |
| `GrafanaAlb` (ALB)               | internet-facing, application LB, en las 2 public subnets. |
| `GrafanaListenerHttps`           | 443/HTTPS con SslPolicy TLS13-1-2-2021-06 y cert ACM (param `GrafanaCertArn`). |
| `GrafanaListenerHttp`            | 80/HTTP -> redirect 301 a HTTPS (defense in depth aparte de force_ssl en Grafana). |
| `GrafanaTaskExecutionRole`       | Pull de ECR + write a CloudWatch + inyecta password del Secrets Manager al container. |
| `AlertTopic` (SNS)               | Subs por email a alarmas (Lambda errors, IoT disconnect, RDS CPU/storage/conn, ALB 5xx, ECS tasks down). |

---

## 3) Schema de Postgres

Definido en [`sql/bootstrap.sql`](sql/bootstrap.sql) — fuente canónica. Tablas
de hechos + dimensiones (`sites`, `devices`) + vistas del producto cartesiano.

| Tabla | Granularidad | Idempotencia |
|---|---|---|
| `count_events`     | 1 row por cruce de linea          | `UNIQUE (device_id, event_ts, track_id, direction)` |
| `telemetry`        | 1 row por sample (5min def)       | sin constraint — duplicados aceptables |
| `wifi_ble_events`  | 1 row por visitor por ventana     | `UNIQUE (device_id, visitor_hash, period_start)` — `ON CONFLICT DO UPDATE` refina `rssi_max` al máx |
| `pos_transactions` | 1 row por transacción (POS API)   | `UNIQUE (store_id, transaction_id)` |

Categorización single-source-of-truth en funciones SQL: `height_class(REAL)`
(adult/child/unknown desde `height_m` crudo) y `rssi_class(INT)`
(shopper/passerby/weak desde `rssi_max` crudo). Modificables con
`CREATE OR REPLACE FUNCTION` retroactivo a todos los rows históricos.

**Columnas notables de `telemetry`**: ademas de OS metrics (`cpu_temp_c`,
`hailo_temp_c`, `disk_free_mb`, etc) y pipeline metrics (`fps`,
`frame_latency_p50/p95_ms`, `tracker_confirmed_count`, etc), tiene
**`wifi_ble_stitching_ratio`** = `groups / hashes` del dia del device. 1.0 =
ningun stitch efectivo. Canary para detectar si la flota corre con OS que
defeatean el stitching (Apple H1+ con seqnum reset, BLE off, etc). Query:
`SELECT device_id, AVG(wifi_ble_stitching_ratio) FROM telemetry WHERE event_ts > now() - interval '1 day' GROUP BY device_id;`

Views (todas en `bootstrap.sql` — ver el archivo para la lista completa y las
definiciones). El patrón es un **producto cartesiano** de las facts agregadas a
3 buckets (`_15min` / `_hour` / `_day`):

- **`counting_by_bucket_*`** — `ins / outs / net` + desglose demográfico (vía `height_class()`) por `(store_id, bucket)`. Buckets server-derived (`GENERATED` desde `event_ts`).
- **`wifi_ble_by_bucket_*`** — `passersby / shoppers / weak` (vía `rssi_class()`) + `COUNT(DISTINCT visitor_hash)` por bucket. El `DISTINCT` dedupa un visitor presente en N ventanas.
- **`pos_by_bucket_*`** — ventas/devoluciones/items/montos por bucket.
- **`turn_in_rate_by_bucket_*`** / **`conversion_by_bucket_*`** — `turn_in_rate = ins / passersby`, `conversion = ins / shoppers` (FULL OUTER JOIN preserva buckets de una sola fuente).
- **`occupancy_by_bucket_*`**, **`visit_duration_by_bucket_*`**, **`metrics_unified_by_bucket_*`** (uso interno de la Lambda), **`data_freshness_by_store`** (último `received_at` cross-fact por sucursal).

### DB users + auth

- `people_counter` (master) — desde DBeaver, password en Secrets Manager.
- `lambda_writer` / `lambda_pos_writer` / `lambda_query_reader` — sin password, `GRANT rds_iam` para auth via IAM token (least privilege: writer escribe facts, pos_writer solo POS, query_reader es SELECT-only sobre las vistas). Cada Lambda role tiene `rds-db:connect` sobre su user.
- `readonly_external` — partner externo, SELECT solo sobre las vistas `*_by_bucket_*` + `sites`/`devices` + `EXECUTE` de las 2 funciones de categorización.
- DB separada `grafana` — owner `people_counter`, solo para state interno de Grafana.

---

## 4) Deploy

Orchestrado por [`deploy.ps1`](deploy.ps1) en 5 fases con `-StartFromPhase`
para resumir interrupciones. Tiene 2 pausas manuales (CNAMEs de validacion
ACM + CNAME final al ALB).

### Pre-requisitos

```powershell
aws sts get-caller-identity            # debe ser tu cuenta target
aws configure get region               # us-east-1
docker info                            # daemon corriendo (push a ECR + bootstrap SQL via docker psql)
```

### Run

```powershell
.\infra\deploy.ps1
```

Fases:

1. **[1/5]** CFN deploy core — VPC + RDS + IoT + Lambda stub + ECR (sin Grafana).
2. **[2/5]** Push imagen `grafana/grafana:latest` a ECR + bootstrap SQL via `docker run postgres:16 psql`.
3. **[3/5]** `aws acm request-certificate` (idempotente: reusa cert si ya existe para el FQDN). Printea los CNAMEs de validacion y **pausa con Read-Host**. Una vez agregados al DNS provider, `aws acm wait certificate-validated` polea hasta ISSUED (timeout 75min). Los CNAMEs de validacion deben quedar PERMANENTES en el DNS provider — ACM los re-checkea en cada renewal.
4. **[4/5]** CFN deploy con `DeployGrafana=true` + el cert ARN como parametro. Crea cluster ECS, task definition, ALB, target group, 2 listeners (443 HTTPS forward + 80 HTTP redirect), 2 SGs y service. CFN espera a que el service estabilice antes de marcar `UPDATE_COMPLETE`. Toda env var de Grafana (incluido el password del RDS via Secrets Manager) ya esta bakeada en la task definition — no hay `update-service` post-deploy.
5. **[5/5]** Printea el ALB DNS Name y **pausa con Read-Host** para que el operador agregue el CNAME `grafana.<DomainName>` -> ALB DNS Name. Despues hace best-effort de `Resolve-DnsName` para confirmar propagacion (no falla si TTL todavia no propago).

Tiempo end-to-end ~15-25 min, dominado por validacion DNS (puede tardar segundos o minutos segun el TTL del provider). Phase 4 ~5-8 min para que CFN cree el ALB y el ECS service estabilice.

La URL final `https://grafana.<DomainName>` queda en el output `GrafanaUrl` del stack y `deploy.ps1` la imprime al cierre.

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

# 3) Abrir Grafana en https://grafana.<DomainName> y queriar count_events
```

---

## 6) Resiliencia

- **Sin conectividad del device** — buffer SQLite local persiste publishes. Replay al reconectar, marca enviado solo tras PUBACK.
- **Lambda fallida** — IoT Rule loguea a CloudWatch (`/aws/iot/people-counter-rule-errors-${env}`). IoT reintenta 1 vez built-in; despues drop. Alarma `PersistEventLambdaErrorAlarm` dispara con >0 errors / 5min.
- **RDS** — Multi-AZ desactivado para PoC ($13/mo vs $26). Daily snapshot (`BackupRetentionPeriod` default 7d, configurable). Restore via `aws rds restore-db-instance-to-point-in-time`.
- **ECS Fargate** — ECS scheduler reinicia el task si crashea. ALB health check (`/api/health` cada 30s) saca de rotacion targets unhealthy. Rolling deploy en cada CFN update (~30s downtime con MinimumHealthyPercent=0 en PoC single-instance; en prod subir a 100% + DesiredCount=2).
- **Cert ACM** — auto-renewed por AWS si los CNAMEs de validacion siguen presentes en el DNS provider. Si los borrás, el cert muere a los 13 meses. Documentado en `deploy.ps1` Phase 3.
- **Teardown / re-create** — `aws cloudformation delete-stack` borra todo el stack (ALB + listeners + target group + ECS service + SGs + cluster). El cert ACM NO se borra (vive fuera del stack); para limpieza full hay que correr `aws acm delete-certificate --certificate-arn <arn>` manualmente. ECR esta marcado `EmptyOnDelete: true` para no atascarse en imagenes pendientes.

---

## 7) Costos PoC (1 device, us-east-1)

| Recurso | $/mes | Notas |
|---|---|---|
| RDS db.t4g.micro (single-AZ)    | ~$13  | Storage 20GB gp3 incluido. |
| Fargate task (0.5 vCPU / 1 GB)  | ~$18  | 24/7 single-instance. |
| ALB                             | ~$16  | $0.0225/h fixed + LCU (PoC = LCU ~$0.5/mo). Costo amortizable agregando services al mismo LB via listener rules. |
| ACM cert                        | $0    | AWS-managed, auto-renewed mientras los CNAMEs de validacion sigan en el DNS provider. |
| IoT Core                        | <$1   | $1/M msgs — PoC genera centavos. |
| Lambda                          | $0    | 1M invocations + 400k GB-s free **forever**. |
| Secrets Manager                 | $0.40 | 1 secret. |
| CloudWatch                      | $0    | 5 GB logs + 10 metrics free forever. |
| **Total estimado**              | **~$35/mo** | Post free tier. |

**Producción (flota)**: migrar RDS a Multi-AZ (~$26 + storage), considerar
Amazon Managed Grafana (SSO + IAM-integrated, ~$9/user/mo) en vez de OSS, y
reintroducir Lambda dedup L3 con DynamoDB cuando haya 2+ cams por store.
Cuando se agregue 2da app (sales API, auth service), reutilizar el mismo ALB
via listener rules (host-header o path-based) en vez de provisionar otro
($16/mo de ahorro por service).

---

## 8) Proximos pasos

- Construir dashboards en Grafana UI sobre las views (`counting_by_bucket`,
  `turn_in_rate_hourly`, `store_hourly_summary`).
- API Gateway para ingest de sales desde POS externo (tabla `sales` ya
  preparada con `UNIQUE (store_id, external_id)` para idempotencia).
- Migracion a Route53 delegated subdomain (`tfg.gasparri.com.ar` NS → R53)
  para que CFN gestione DNS records (ALIAS record al ALB) y el deploy sea
  100% sin pausa.

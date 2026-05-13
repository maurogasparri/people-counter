# Infra AWS — People Counter (PoC)

Stack cloud que recibe mensajes del dispositivo edge, los persiste en Postgres, y los expone en Grafana.

```
RPi5 ──MQTT/TLS──► IoT Core ──┬─► Rule "counting"   ─┐
                              ├─► Rule "telemetry"  ─┼─► Lambda persist_event ─► Postgres
                              ├─► Rule "wifi_ble"   ─┘                          (EC2 t3.micro)
                              │
                              └─► Shadow $aws/things ◄─► dispositivo (operating_hours, counting_enabled)

                                              EC2 ── Grafana OSS ──► dashboards
                                                  └── cron pg_dump ──► S3 bucket
```

**Decisiones de scope** (PoC con 1 device):
- **Sin Timestream**: no está en el AWS Free Plan; usar Postgres self-hosted en EC2 free tier.
- **Sin Lambda dedup L3**: innecesaria con 1 device/sucursal (el dedup L1+L2 local cubre todo). Reintroducir cuando haya 2+ devices por sucursal.
- **Postgres self-hosted en EC2** (no RDS): $0 los primeros 12 meses, ~$8/mes después. Grafana OSS en la misma instancia para ahorrar otra VM.
- **Payload de WiFi/BLE reducido**: el device manda counts agregados (`passersby`, `shoppers`), no hashes individuales — privacidad mejor y payload chico.

---

## 1) Topics MQTT y schemas de payload

Los topics se interpolan en el dispositivo con `{store_id}` (ver `config/config.example.yaml → mqtt.topics`).

### `store/{store_id}/counting` — eventos de cruce de línea

Publicado en tiempo real cuando un track completa el cruce.

```json
{
  "device_id": "store-pilot-01-cam-01",
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

→ Rule `IoTRuleCounting` → Lambda `persist_event` → tabla `counting_events`.

### `store/{store_id}/telemetry` — salud del dispositivo

Cadencia default `3600s` (1h, configurable via `telemetry.interval_seconds`).

```json
{
  "device_id": "store-pilot-01-cam-01",
  "timestamp": 1762963200.0,
  "type": "telemetry",
  "data": {
    "cpu_temp_c": 56.4,
    "hailo_temp_c": 48.2,
    "disk_used_pct": 31.0,
    "uptime_s": 142800,
    "fps": 14.8,
    "frame_latency_p50_ms": 62.0,
    "frame_latency_p95_ms": 145.0,
    "detection_rate_per_min": 12.3,
    "tracker_confirmed_count": 3,
    "buffer_backlog_messages": 0,
    "mqtt_connected": true,
    "total_in": 187,
    "total_out": 184,
    "wifi_probe_ok": true,
    "ble_scanner_ok": false
  }
}
```

→ Rule `IoTRuleTelemetry` → Lambda → tabla `telemetry`.

### `store/{store_id}/wifi_ble` — resúmenes de probing (cada 15 min)

Cadencia `wifi_ble.probe_interval_seconds` (default 900s = 15min). Solo se publica si la ventana produjo detecciones (passersby > 0).

```json
{
  "device_id": "store-pilot-01-cam-01",
  "timestamp": 1762963200.0,
  "type": "wifi_ble",
  "data": {
    "period_start": 1762962300,
    "period_end":   1762963200,
    "passersby":    160,
    "shoppers":      27
  }
}
```

- **passersby** (RSSI ≥ `wifi_ble.rssi_passerby_threshold`, default -75 dBm): pasó por la zona.
- **shoppers** (RSSI ≥ `wifi_ble.rssi_shopper_threshold`, default -55 dBm): muy cerca / probable entrada.
- Combina WiFi+BLE post-L2 dedup local: un MAC detectado por ambos cuenta 1 vez.
- **`in` real** = `counting_events.direction='in'` (cámaras). El turn-in rate se calcula en Grafana como `ins / passersby`.

→ Rule `IoTRuleWifiBle` → Lambda → tabla `wifi_ble_summaries`.

**Reglas duras:**
- **Nunca MACs crudas.** Hashing en el device antes de bufferear.
- **Los hashes no salen del device.** Solo agregados al cloud.

### Shadow — `$aws/things/{device_id}/shadow/update` (y `/delta`)

Pusheable desde la nube (whitelist en `src/config/loader.py` — solo 2 keys):

```json
{ "state": { "desired": { "counting_enabled": false, "operating_hours": { ... } } } }
```

El resto (thresholds, geometría, modelo) se cambia editando `config.yaml` per-device y reiniciando.

---

## 2) Recursos del stack

| Recurso | Propósito |
|---------|-----------|
| `IoTDevicePolicy` | Permisos por device — connect, publish a sus 3 topics, subscribe al shadow propio. |
| `IoTThingType: people-counter-<env>` | Atributos searcheables (`store_id`, `firmware_version`). |
| 3× `AWS::IoT::TopicRule` | Routing MQTT → Lambda. Error action → CloudWatch Logs. |
| `PersistEventLambda` | Python 3.13, recibe envelope estándar, inserta en Postgres vía psycopg. |
| `PgPasswordParameter` (SSM) | Password DB, leído por la Lambda y por la EC2 al boot. |
| `DataInstance` (EC2 t3.micro Ubuntu 24.04) | Postgres 16 + Grafana OSS + nginx + cron de backup. UserData inline bootstrapea todo. |
| `DataInstanceEIP` | IP pública estable (no cambia al reiniciar). |
| `BackupBucket` (S3) | Recibe `pg_dump -Fc` diario, lifecycle 30 días. |
| 3 alarmas CloudWatch | Lambda errors, IoT disconnects, EC2 status checks. |

## 3) Schema de Postgres

Definido en `infra/postgres/schema.sql`. 3 tablas independientes:

| Tabla | Granularidad | Retención |
|-------|--------------|-----------|
| `counting_events` | 1 row por cruce de línea | Indefinida (lifecycle manual) |
| `telemetry` | 1 row por sample (default 1h) | Indefinida |
| `wifi_ble_summaries` | 1 row por ventana (default 15 min) | Indefinida |

Más 2 views útiles (`v_counting_hourly`, `v_turn_in_rate_hourly`) que pre-arman queries comunes.

---

## 4) Deploy

### Pre-requisitos

```bash
# Verificar credenciales y región.
aws sts get-caller-identity
aws configure get region    # Debería decir us-east-1

# Crear un key pair para SSH a la EC2.
aws ec2 create-key-pair --key-name people-counter-dev \
  --query 'KeyMaterial' --output text > ~/.ssh/people-counter-dev.pem
chmod 600 ~/.ssh/people-counter-dev.pem
```

### Pasos

```powershell
# 1) Deploy del stack. Tarda ~5-8 min (la EC2 corre userdata en paralelo).
aws cloudformation deploy `
  --template-file infra/cloudformation/people-counter.yaml `
  --stack-name people-counter-dev `
  --capabilities CAPABILITY_NAMED_IAM `
  --parameter-overrides `
      Environment=dev `
      KeyPairName=people-counter-dev `
      PostgresInitialPassword=$(openssl rand -base64 24)

# 2) Reemplazar el placeholder de Lambda con el código real + psycopg layer.
scripts/deploy_lambda.sh dev

# 3) Aprovisionar el primer dispositivo (cert + thing + attach a policy).
py scripts/provision.py create `
  --device-id store-pilot-01-cam-01 `
  --store-id store-pilot-01 `
  --out /etc/people-counter/certs/
```

### Outputs del stack

```bash
aws cloudformation describe-stacks --stack-name people-counter-dev \
  --query 'Stacks[0].Outputs' --output table
```

Devuelve:
- `IoTEndpointHint` — comando para obtener el endpoint MQTT (cargar en `config.yaml → mqtt.endpoint`).
- `DataInstancePublicIp` — IP estática de la EC2.
- `GrafanaURL` — `http://<ip>:3000`, login inicial `admin`/`admin`.
- `PostgresEndpoint` — para conectar DBeaver/psql.
- `SshCommand` — comando SSH listo.

### Convertir el password a SecureString (recomendado post-deploy)

CloudFormation no acepta `SecureString` en parámetros nuevos, así que el password vive como `String` plano en SSM hasta que lo convertís manualmente:

```bash
aws ssm put-parameter \
  --name /people-counter/dev/pg_password \
  --type SecureString \
  --overwrite \
  --value "$(aws ssm get-parameter --name /people-counter/dev/pg_password --query Parameter.Value --output text)"
```

---

## 5) Setup post-deploy

### Cargar dashboards de Grafana

Los dashboards están en `infra/grafana/dashboards/`. Vía SSH:

```bash
ssh -i ~/.ssh/people-counter-dev.pem ubuntu@<IP>
sudo mkdir -p /var/lib/grafana/dashboards
sudo cp /path/to/repo/infra/grafana/dashboards/*.json /var/lib/grafana/dashboards/
sudo cp /path/to/repo/infra/grafana/provisioning/dashboards.yaml \
  /etc/grafana/provisioning/dashboards/people-counter.yaml
sudo systemctl restart grafana-server
```

O alternativamente: importar los JSON desde la UI de Grafana (`+ → Import`).

### Configurar HTTPS (opcional para PoC)

nginx ya está instalado pero sin config. Para Grafana detrás de HTTPS con Let's Encrypt:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d <tu-dominio>.com
```

Hay que tener un dominio apuntando al EIP. Para PoC alcanza con `http://<ip>:3000` directo.

---

## 6) Cómo verificar end-to-end

```bash
# 1) En el device — confirmar que está publicando.
journalctl -u people-counter -f | grep -E "publish|wifi_ble|telemetry"

# 2) En AWS — verificar que IoT recibe los mensajes (rule activations).
aws logs tail /aws/iot/people-counter-rule-errors-dev --since 10m
# (debería estar vacío si todo funciona)

# 3) Conectarse a Postgres y ver los rows.
psql -h <IP> -U people_counter -d people_counter -c \
  "SELECT COUNT(*), MAX(event_time) FROM counting_events;"

# 4) Abrir Grafana en http://<IP>:3000 — el dashboard "People Counter — Overview"
#    debería mostrar datos recientes.
```

---

## 7) Resiliencia

- **Sin conectividad del device**: buffer SQLite local (`/var/lib/people-counter/buffer.sqlite`) persiste publishes. Al reconectar, `MQTTClient.replay_buffer` los drena en orden y marca enviado solo tras PUBACK.
- **Reconexión MQTT**: paho con backoff exponencial (`reconnect_delay_set 1-120s`). Shadow se reconcilia en el hook `on_connected`.
- **Lambda fallida**: IoT Rule tiene `ErrorAction` que loguea a CloudWatch. Alarma `PersistEventLambdaErrorAlarm` dispara después de 15 min de errors elevados.
- **Postgres corrupto / EC2 reventada**: backup diario en S3 (`postgres/pc-<timestamp>.dump`). Restore con `pg_restore -d people_counter <dump>` toma ~5 min.
- **EC2 reboot**: Postgres + Grafana son `systemctl enable`d → arrancan solos. Grafana mantiene dashboards (SQLite local en `/var/lib/grafana/`).
- **EC2 reemplazo total**: el EIP no cambia (asociación se mantiene). Re-deploy del stack reconstruye todo desde userdata. Restore manual del backup más reciente de S3.

---

## 8) Costos (PoC con 1 device)

| Período | Costo estimado | Notas |
|---------|---------------|-------|
| Año 1 | **$0** | Todo en free tier 12 meses + créditos. |
| Año 2+ | **~$8-12 USD/mes** | EC2 t3.micro on-demand ($7.5) + EBS 30GB gp3 (~$2.5) + EIP ($0 mientras esté attached). |

Servicios que cuestan según uso, dentro del free tier:
- IoT Core: $1/millón msgs (PoC = pocos centavos).
- Lambda: 1M invocations + 400k GB-s free **forever**.
- CloudWatch: 10 metrics + 3 dashboards + 5 GB logs free forever.
- S3: 5 GB free 12 meses, después ~$0.023/GB.
- SSM Parameter Store: gratis hasta 10k parámetros estándar.

**Si se escala a flota de devices**: el cost driver pasa a ser IoT Core (msgs) y Postgres (storage). A ~50 devices, considerar:
- Migrar Postgres a RDS db.t4g.micro (~$13/mes + storage).
- Reintroducir Lambda dedup L3 con DynamoDB (~$0 free tier por mucho tiempo).

---

## 9) Próximos pasos sugeridos

- **Producción**: VPC dedicada con private subnets para Postgres, NAT gateway para egress de Lambda. Hoy todo está en VPC default por simplicidad.
- **HTTPS**: nginx + Let's Encrypt + dominio (paso 5).
- **Backups offsite**: replicar S3 bucket a otra región con S3 replication.
- **Métricas custom**: emitir CloudWatch metrics desde la Lambda (rows insertados, latencia de query) para tener dashboards de operación además de los de negocio.

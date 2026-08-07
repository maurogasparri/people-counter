# Alerting (Grafana → email vía AWS SES)

Setup end-to-end del canal de alertas operacional. Grafana 13 evalúa reglas
contra el datasource Postgres y dispara emails a través de AWS SES.

## Arquitectura

```
Alert rule (Grafana)              SES SMTP                Operador
  ├─ query a Postgres              ↓                       ↓
  ├─ condición (>/<)         email-smtp.us-east-1     mauro@gasparri.com.ar
  ├─ for: 5m (sostenido)       (port 587, TLS)
  └─ contact point: "ops-email" ─────→
       (interno de Grafana)
```

Auth: Grafana usa SMTP user + password generados por SES (no son las
credenciales AWS normales — son derivadas via HMAC-SHA256 del IAM
SecretAccessKey + el string "SendRawEmail"). El password vive en Secrets
Manager y ECS Fargate lo inyecta como env var al container.

## Pre-requisitos (todos cubiertos por CFN + deploy.ps1 Phase 5)

| Pieza | Owner | Detalle |
|---|---|---|
| SES Domain Identity (`<tu-dominio>`) | CFN | Resource `SesDomainIdentity` con DKIM 2048-bit |
| 3 CNAMEs de DKIM en DNS externo | manual | `deploy.ps1` los muestra; agregar al DNS provider |
| SES Email Identity (`mauro@gasparri.com.ar`) | CFN | Resource `SesAlertEmailIdentity`; verificación = click en mail que envía AWS |
| IAM User `people-counter-ses-smtp-dev` | CFN | Policy: `ses:SendRawEmail` sobre identity del dominio |
| IAM AccessKey del SMTP user | CFN | Resource `SesSmtpAccessKey` |
| Secrets Manager `people-counter-ses-smtp-dev` | CFN + deploy.ps1 | CFN crea con creds raw; deploy.ps1 deriva el SMTP password y lo writeback |
| Env vars SMTP en Grafana task | CFN | `GF_SMTP_ENABLED=true`, `GF_SMTP_HOST`, etc. |
| Secret refs en Grafana task | CFN | `GF_SMTP_USER` y `GF_SMTP_PASSWORD` desde el secret |

## Operación corriente (luego del setup inicial)

### Verificar que el canal está OK

```bash
# Domain identity verificado?
aws sesv2 get-email-identity --email-identity <tu-dominio> \
  --query 'VerifiedForSendingStatus' --output text
# → True

# Email destinatario verificado? (sandbox SES exige esto)
aws sesv2 get-email-identity --email-identity mauro@gasparri.com.ar \
  --query 'VerifiedForSendingStatus' --output text
# → True
```

### Test send desde Grafana UI

1. Login en `https://grafana.<tu-dominio>`
2. **Alerting → Contact points → New contact point**
3. Tipo: **Email**
4. Addresses: `mauro@gasparri.com.ar`
5. **Test** — debe llegar el mail.

Si el test falla:
- **"Email address is not verified"** → `mauro@gasparri.com.ar` no está
  verificado en SES. Buscar mail de AWS y clickear el link de verificación.
- **"Domain not verified"** → DKIM aún no propagó. Esperar y reintentar.
- **"Authentication failed"** → el smtpPassword del secret no se derivó
  correctamente. Re-correr `deploy.ps1 -StartFromPhase 5` o re-derivar
  manual (ver "Recovery" abajo).
- **"Connection refused / timeout"** → el SG del task ECS no permite
  outbound 587. El default allow-all egress debería cubrirlo; revisar
  `GrafanaTaskSg` no haya sido restringido.

### Recovery: re-derivar el SMTP password manualmente

Si el password se desincronizó (por ej. el SecretAccessKey rotó pero el
secret no se actualizó), reejecutar:

```powershell
.\infra\deploy.ps1 -StartFromPhase 5
```

Es idempotente: detecta si el password ya está derivado y skipea, o lo
deriva si encuentra el placeholder. También fuerza un rolling deploy del
service para que tome el nuevo secret.

## Reglas de alerta iniciales recomendadas

Las queries de canaries del tracker están listas en RDS (columnas de la
tabla `telemetry`). Sugeridas como primeras alert rules en Grafana
(ver `docs/tracker_tuning.md` para el contexto operacional):

| Nombre | Query (sobre `telemetry`) | Condición | For | Severity |
|---|---|---|---|---|
| `tracker-stitching-degraded` | `SELECT AVG(track_stitching_ratio) FROM telemetry WHERE device_id = $device AND event_ts > NOW() - INTERVAL '1 hour'` | `> 1.3` | 1h | warning |
| `tracker-rescue-saturated` | `SELECT MAX(death_emit_count) - MIN(death_emit_count) FROM telemetry WHERE device_id = $device AND event_ts > NOW() - INTERVAL '1 hour'` | `> 30/hour` | 30m | warning |
| `wifi-stitching-broken` | `SELECT AVG(wifi_ble_stitching_ratio) FROM telemetry WHERE device_id = $device AND event_ts > NOW() - INTERVAL '1 hour'` | `> 0.9` | 1h | warning |
| `device-offline` | `SELECT NOW() - MAX(event_ts) FROM telemetry WHERE device_id = $device` | `> 15 min` | 15m | critical |
| `shadow-apply-stale` | `SELECT NOW() - MAX(last_shadow_apply_ts) FROM telemetry WHERE device_id = $device AND last_shadow_apply_ts IS NOT NULL` | `> 24h` | 24h | warning |

## Segundo canal: alarmas de infraestructura (CloudWatch → SNS → email)

El canal Grafana→SES de arriba cubre las alertas de **negocio/dispositivo**
(canaries del tracker, device offline, stitching) que se calculan con queries
SQL sobre `telemetry`. En **paralelo y por separado**, el CloudFormation define
10 **alarmas de CloudWatch** sobre las métricas de infraestructura AWS, que
disparan a un **SNS topic dedicado** `people-counter-alerts-${Environment}`
(subscripción por email al `AlertEmail` del stack). Es un canal **distinto** del
Grafana→SES: no depende de que Grafana ni la RDS estén sanos para avisarte
(justamente cubren cuando se caen).

| Alarma (CFN) | Métrica | Umbral |
|---|---|---|
| `persist-event-errors` | Lambda `persist_event` Errors | `>= 3 / 5min` |
| `ingest-pos-errors` | Lambda `ingest_pos_transaction` Errors | sostenido |
| `query-aggregates-errors` | Lambda `query_aggregates` Errors | sostenido |
| `query-aggregates-latency` | Lambda `query_aggregates` Duration | latencia alta |
| `rds-cpu` | RDS CPUUtilization | `> 75%` sostenido (3×5min) |
| `rds-storage` | RDS FreeStorageSpace | `< 2 GB` |
| `rds-connections` | RDS DatabaseConnections | `> 60` (max ~85 en t4g.micro) |
| `iot-errors` | IoT RuleMessageThrottled | `> 5 / 5min` |
| `grafana-5xx` | ALB HTTPCode_Target_5XX | `> 5 / 5min` |
| `grafana-tasks` | ALB HealthyHostCount del target group | `< 1` (Grafana caído) |

Todas con `AlarmActions: [!Ref AlertTopic]`. Definidas en
`infra/cloudformation/people-counter.yaml` (sección *SNS + CloudWatch alarms*).

> **Falso positivo esperable durante un bulk-load**: un COPY/re-seed masivo o
> una migración de histórico satura RDS y dispara transitoriamente
> `persist-event-errors` por `ConnectionTimeout`. Es esperable durante el
> mantenimiento y se resuelve solo al terminar — ver el runbook de carga masiva
> en [`cloud_dr.md`](cloud_dr.md).

## Sandbox vs Producción

SES en cuenta nueva arranca en **sandbox**:

- Max 200 emails/día (suficiente para alertas, no para emails masivos).
- Max 1 email/segundo.
- **Destinatario debe estar verificado** (cada email que querramos enviar
  necesita un `SesEmailIdentity` correspondiente + click de verificación).

Para piloto + flota inicial el sandbox alcanza (1-2 operadores
notificados). Cuando se sumen N clientes con destinatarios variables,
**pedir production access**:

1. AWS Console → SES → Account dashboard → "Request production access".
2. Justificar use case ("notificaciones operacionales de un sistema de
   conteo de personas; volumen estimado <500 emails/día").
3. Aprobación 24-48h (a veces más rápido).
4. Una vez aprobado, no hay que verificar destinatarios individuales —
   se puede mandar a cualquier email.

## Tradeoff conocido: SMTP rate limit del sandbox

SES sandbox cap = 1 email/segundo. Si un alert rule de Grafana matchea
muchos targets simultáneamente (ej. multi-device dashboard con 1 rule por
device), pueden encolarse. Grafana hace backoff y reintenta — los emails
llegan pero con delay.

Mitigación: usar `group_by` en notification policies para colapsar alertas
similares en un solo mail. Ej:
```yaml
group_by: [alertname, severity]
group_wait: 30s
group_interval: 5m
```

## Histórico

- **2026-05-25** — Setup inicial. SES sandbox, dominio verificado vía
  DKIM, IAM user dedicado, smtpPassword derivado offline. Contact point
  email configurado en la org.
- **Cierre** — Las 12 reglas de alerta quedaron versionadas en
  `infra/grafana/alerting/alert-rules.json` y se provisionan como código
  (idempotente) vía `import_alerts.ps1`, que `deploy.ps1` invoca
  automáticamente en su Phase 6.

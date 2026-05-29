# Política de backups y disaster recovery (cloud)

Plan de protección de datos para los componentes AWS del sistema. Cubre el
PoC y la transición a producción multi-sucursal.

> 📌 **Backup del device** (calibration, certs, config local) está en
> [`setup_guide.md §14`](setup_guide.md). Este documento cubre solo la
> capa cloud.

## Qué se protege

| Recurso | Tipo de data | Riesgo si se pierde | Backup |
|---|---|---|---|
| **RDS Postgres (`people_counter`)** | Histórico de count_events, telemetry, wifi_ble_events, pos_transactions | Pérdida de meses/años de analytics | Snapshots automáticos + PITR |
| **RDS Postgres (`grafana`)** | Dashboards, users, datasources de Grafana | Re-armar dashboards a mano | Mismo snapshot que arriba (es la misma instancia) |
| **Lambda code** | persist_event, ingest_pos_transaction, query_aggregates | Cero — se redeploya desde el repo | Repo Git (source of truth) |
| **CloudFormation stack** | IaC del sistema entero | Cero — IaC en repo | Repo Git |
| **AWS IoT certs (per-device)** | X.509 certs de cada RPi | Re-emisión de cert + re-provisioning del device | `scripts/provision.py harvest` ya guarda copia local; opcional re-issue via AWS IoT |
| **ECR image (Grafana)** | Snapshot de la imagen | Cero — se re-pullea de docker hub | docker hub |

**Prioridad uno**: la RDS Postgres. Es el único recurso con data
acumulativa imposible de regenerar.

## RDS Postgres — configuración actual

Provisionada por `infra/cloudformation/people-counter.yaml`, sección
`RdsInstance`. Settings de DR:

| Setting | Valor | Comentario |
|---|---|---|
| `Engine` | postgres 16 | `db.t4g.micro` para PoC |
| `StorageEncrypted` | `true` | KMS encryption at-rest, gestionado por AWS |
| `MultiAZ` | parametrizable, default `false` (PoC); `true` en producción | Standby sincrónico en otra AZ — failover automático en ~60s |
| `BackupRetentionPeriod` | parametrizable, default `0` (PoC); `7-35` en producción | Días de snapshots automáticos retenidos |
| `PreferredBackupWindow` | `03:00-04:00 UTC` (00:00-01:00 ARG) | Off-hours del piloto |
| `DeletionPolicy` / `UpdateReplacePolicy` | `Snapshot` | Si se borra el stack o se reemplaza la DB, se toma snapshot antes (resguardo CFN) |
| `EnablePerformanceInsights` | `false` (PoC); recomendado `true` en producción | Métricas extendidas para troubleshooting |

### Snapshots automáticos

Cuando `BackupRetentionPeriod > 0`, RDS toma un snapshot **diariamente**
durante la `PreferredBackupWindow` y los retiene por N días. Storage de
snapshots dentro de la retención es **gratuito** hasta el storage size
del DB; más allá se cobra estándar S3.

### Point-in-time recovery (PITR)

Habilitado **automáticamente** cuando `BackupRetentionPeriod > 0`. RDS
guarda transaction logs cada **5 minutos** y permite restaurar la DB a
cualquier momento dentro de la ventana de retención, con granularidad de
segundos.

→ El **RPO efectivo** (Recovery Point Objective, máxima data que se
puede perder) es **5 minutos**.

### Manual snapshots

Independientes de los automáticos, NO se borran con el stack. Útiles
antes de cambios riesgosos (migración de schema, upgrade de versión):

```bash
aws rds create-db-snapshot \
    --db-instance-identifier people-counter-rds-<env> \
    --db-snapshot-identifier people-counter-pre-<descripcion>-$(date +%Y%m%d)
```

## Targets de RPO/RTO

| Métrica | PoC actual | Producción (objetivo) |
|---|---|---|
| **RPO** (Recovery Point Objective) | 24h (snapshot diario sin PITR si retention=0) o 5min (con PITR) | **5min** (PITR siempre activo) |
| **RTO** (Recovery Time Objective) | Manual restore: ~30-60min para una instancia chica | **<10min** con Multi-AZ failover automático; ~30min para restore desde snapshot |

PoC actual: `BackupRetentionPeriod=0` → **sin RPO/RTO formal**. Aceptable
solo en PoC (data se puede regenerar dejando los devices reportando un
día más). Para producción se eleva a `7-30` y `MultiAZ=true`.

## Procedimiento de restore (drill)

Validar al menos **una vez por trimestre** que el restore funciona,
contra un stack de staging (no prod):

### Restore desde snapshot automático

```bash
# 1. Listar snapshots disponibles
aws rds describe-db-snapshots \
    --db-instance-identifier people-counter-rds-<env> \
    --snapshot-type automated \
    --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,Status]' \
    --output table

# 2. Restore a una instancia NUEVA (no machaca la existente)
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier people-counter-restored-test \
    --db-snapshot-identifier <snapshot-id> \
    --db-subnet-group-name <subnet-group-existente> \
    --vpc-security-group-ids <sg-id>

# 3. Esperar a que esté "available" (~10-20min)
aws rds wait db-instance-available \
    --db-instance-identifier people-counter-restored-test

# 4. Validar: connect + count rows
psql "host=<new-endpoint> dbname=people_counter user=... sslmode=require" \
     -c "SELECT count(*) FROM count_events WHERE event_ts > NOW() - INTERVAL '1 day';"

# 5. Cleanup
aws rds delete-db-instance \
    --db-instance-identifier people-counter-restored-test \
    --skip-final-snapshot
```

### PITR a un timestamp específico

```bash
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier people-counter-rds-<env> \
    --target-db-instance-identifier people-counter-pitr-test \
    --restore-time 2026-05-23T10:00:00Z \
    --db-subnet-group-name <subnet-group-existente> \
    --vpc-security-group-ids <sg-id>
```

## Escenarios de fallo cubiertos

| Escenario | Detección | Recuperación |
|---|---|---|
| Hardware AZ falla (instancia inaccesible) | CloudWatch alarm `RDS Status != "available"` | **Multi-AZ ON**: failover automático ~60s, sin pérdida de data. **Multi-AZ OFF**: re-crear desde snapshot, RPO 5min con PITR |
| Query erróneo borra/corrompe data | Operador detecta, antes del próximo snapshot diario si es posible | PITR al timestamp justo antes del query erróneo |
| Storage corruption (raro, gestionado por AWS) | RDS alarms automáticos | AWS recupera desde su replicación interna (capa S3 detrás de RDS) |
| Stack CloudFormation borrado por error | `aws cloudformation delete-stack` | `DeletionPolicy: Snapshot` toma un snapshot final automáticamente; re-deploy del stack y restore desde ese snapshot |
| Account compromise / accidental drop | Auditoría de CloudTrail | Snapshots automáticos se preservan ~35 días; account recovery + restore. Producción: replicar snapshots cross-region o cross-account para resiliencia adicional |

## Checklist para passar de PoC a producción

- [ ] `RdsMultiAZ=true` en el deploy → standby sincrónico, failover automático.
- [ ] `RdsBackupRetentionDays >= 7` (recomendado 14-30).
- [ ] `EnablePerformanceInsights=true` en el CFN.
- [ ] Habilitar **cross-region snapshot copy** (Lambda+EventBridge o backup vault).
- [ ] Cron de **restore drill trimestral** (puede ser un script en GitHub Actions
      que arma una instancia restored, valida row count, y la borra).
- [ ] **Alarma CloudWatch** `RDS FreeStorageSpace < 1GB` (storage autoscale ya
      está configurado en el CFN hasta `RdsMaxAllocatedStorage`).
- [ ] Documentar **runbook de restore** con captures de pantalla y tiempos
      reales medidos en el drill.

## Costo aproximado de DR (cuando se activa)

| Item | Costo mensual estimado |
|---|---|
| Snapshots automáticos retention 7 días (db.t4g.micro ~20GB) | ~$1-2 |
| Multi-AZ (standby sincrónico) | +$13 (duplica el costo de la instancia) |
| Performance Insights (default retention 7 días) | $0 (free tier) |
| Cross-region snapshot copy | ~$0.5/GB transferido + storage en región destino |
| **Total DR pleno producción** | ~$15-20/mes adicionales sobre el PoC |

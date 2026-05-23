# Acceso programático a los datos (US-08)

Cómo conectarse a RDS Postgres como cliente externo (partner, analista) para
consumir las vistas de agregados del sistema vía SQL.

## Rol y permisos

El rol **`readonly_external`** (creado por `bootstrap.sql` y la migración
`infra/sql/migrations/2026-05-23-readonly-external-role.sql`) tiene
`SELECT` exclusivamente sobre las **vistas de agregados**, no sobre las
tablas crudas. Esto es deliberado: los analistas externos no necesitan
`device_id` ni timestamps fine-grained — necesitan métricas por store/hora/día.

### Vistas expuestas

| Vista | Granularidad | Métricas |
|---|---|---|
| `counting_by_bucket` | 15min | `ingress`, `egress`, neto |
| `counting_hourly` | hora | `ingress`, `egress` por hora |
| `counting_daily` | día | `ingress`, `egress` por día |
| `turn_in_rate_by_bucket` | 15min | `passersby`, `shoppers`, `turn_in_rate` |
| `wifi_ble_store_traffic` | 15min | passersby/shoppers post-stitching |
| `conversion_rate_by_store` | acumulado | `visitors`, `transactions`, `conversion_rate` |
| `conversion_rate_hourly` | hora | conversion_rate por hora |
| `conversion_rate_daily` | día | conversion_rate por día |
| `sites`, `devices` | dimensión | catálogo (lat/long, nombres) |

Si un partner necesita granularidad mayor (per-device, per-evento), se le
crea un rol específico con grants adicionales — el rol genérico se
mantiene mínimo a propósito.

## Setup operativo (one-time por entorno)

### 1. Asignar password fuerte

El bootstrap crea el rol con un placeholder. Asignar un password real
gestionado por Secrets Manager o equivalente:

```sql
ALTER USER readonly_external WITH PASSWORD '<password-fuerte-de-16+-chars>';
```

Mejor: crear un secret en AWS Secrets Manager con rotación automática y
sincronizar el password en cada rotación (Lambda rotator).

### 2. Habilitar conectividad externa

Por default la RDS está en una subnet privada del VPC y NO acepta tráfico
externo. Para habilitar acceso de partners hay dos opciones:

**Opción A — RDS publicly accessible + SG whitelist (más simple, menos seguro)**:
1. Modificar el CFN: `RdsInstance.PubliclyAccessible: true` + atributo
   `PubliclySubnets` en lugar de las privadas.
2. Agregar un `AWS::EC2::SecurityGroupIngress` al SG del RDS con los
   CIDRs autorizados (los de los partners):
   ```yaml
   RdsExternalIngress:
     Type: AWS::EC2::SecurityGroupIngress
     Properties:
       GroupId: !Ref RdsSecurityGroup
       IpProtocol: tcp
       FromPort: 5432
       ToPort: 5432
       CidrIp: <partner-IP>/32  # un recurso por CIDR
       Description: "Partner X read-only access"
   ```

**Opción B — Bastion / VPN (más seguro, más complejo)**:
- Bastion EC2 público con el SG del RDS allowando 5432 desde el bastion.
- Partner se conecta vía SSH tunnel: `ssh -L 5432:rds-endpoint:5432 user@bastion`.
- O VPC peering / VPN site-to-site para partners corporativos.

Para el PoC se recomienda Opción A con whitelist estricta. Para producción
multi-partner, evaluar VPN/PrivateLink.

## Conectarse (cliente externo)

Endpoint RDS, usuario `readonly_external`, base `people_counter`, port 5432,
SSL **required** (la instancia tiene `rds.force_ssl=1`).

### psql

```bash
psql "host=<rds-endpoint> port=5432 dbname=people_counter \
      user=readonly_external password=<password> sslmode=require"
```

### Python (psycopg)

```python
import psycopg

conn = psycopg.connect(
    host="<rds-endpoint>",
    port=5432,
    dbname="people_counter",
    user="readonly_external",
    password="<password>",  # del Secrets Manager
    sslmode="require",
)

with conn.cursor() as cur:
    cur.execute("""
        SELECT store_id, bucket_hour, ingress, egress
        FROM counting_hourly
        WHERE bucket_hour >= NOW() - INTERVAL '7 days'
        ORDER BY bucket_hour DESC
        LIMIT 100
    """)
    for row in cur.fetchall():
        print(row)
```

### DBeaver / DataGrip

Connection type: PostgreSQL.
- Host: `<rds-endpoint>`, Port: 5432, Database: `people_counter`.
- User: `readonly_external`, Password: `<del Secrets Manager>`.
- Driver properties → `sslmode = require`.

## Queries de ejemplo

### Footfall diario por store

```sql
SELECT s.store_name, c.bucket_day, c.ingress, c.egress
FROM counting_daily c
JOIN sites s ON c.store_id = s.store_id
WHERE c.bucket_day >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY s.store_name, c.bucket_day;
```

### Turn-in rate por hora — última semana

```sql
SELECT store_id, bucket_15min, passersby, shoppers, turn_in_rate
FROM turn_in_rate_by_bucket
WHERE bucket_15min >= NOW() - INTERVAL '7 days'
  AND store_id = '<store-id>'
ORDER BY bucket_15min;
```

### Conversion rate por store (acumulado)

```sql
SELECT store_id, visitors, transactions, conversion_rate
FROM conversion_rate_by_store
ORDER BY conversion_rate DESC NULLS LAST;
```

### Mapa de stores (lat/long)

```sql
SELECT store_id, store_name, latitude, longitude, address
FROM sites
WHERE latitude IS NOT NULL;
```

## Notas de seguridad

- **Nunca commitear el password** del `readonly_external` en este repo.
  Gestionarlo desde Secrets Manager o el secret store del partner.
- **El rol NO puede modificar nada** — `SELECT` único, sin `INSERT/UPDATE/
  DELETE` (verificable con `\dp` en psql). Si un query intenta escribir,
  Postgres lo rechaza con permission denied.
- **SSL es required** — `rds.force_ssl=1` en el parameter group de RDS
  rechaza conexiones plain TCP.
- **Auditar acceso**: las queries del rol se loguean en `pg_stat_activity`
  + CloudWatch Logs del RDS (si se habilita `log_statement=all`, que está
  off por default por overhead).
- **Rotar password** mínimo trimestralmente. Con Secrets Manager rotation
  Lambda lo hace automático.

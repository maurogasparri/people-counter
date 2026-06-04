# Acceso programático a los datos (US-08 + US-09)

Hay dos formas de consumir datos del sistema sin pasar por Grafana:

1. **REST API** (`GET https://api.tfg.gasparri.com.ar/v1/aggregates`) — **US-09**
   (RF-13). Recomendado para integraciones con sistemas externos (ERP del
   retailer, CRM, BI corporativo). Auth: AWS SigV4. Ver sección **REST API**
   abajo.
2. **SQL directo a Postgres** vía rol `readonly_external` — **US-08** (RF-12).
   Recomendado para analistas / equipos de BI que necesitan ad-hoc SQL
   (DBeaver, psql, notebooks). Ver sección **SQL directo (read-only)** abajo.

---

## REST API (`v1/aggregates`)

Endpoint unificado que devuelve, en una sola respuesta, counts (con
breakdown adult/child/unknown), tráfico externo WiFi/BLE y POS transactions
agrupados por bucket × site.

**Spec OpenAPI 3.1** disponible en `GET https://api.tfg.gasparri.com.ar/v1/openapi.json`
(sin auth — para que el cliente pueda generar su SDK ANTES de tener credenciales).

### Auth

AWS SigV4 con un IAM principal (user o role) autorizado con
`execute-api:Invoke` sobre el ARN del endpoint. Misma política y patrón que
el ingest POS (`POST /pos/transactions`) — un solo IAM principal por
cliente, cubre ambos endpoints.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "execute-api:Invoke",
    "Resource": [
      "arn:aws:execute-api:us-east-1:<account>:<api-id>/$default/GET/v1/aggregates",
      "arn:aws:execute-api:us-east-1:<account>:<api-id>/$default/POST/pos/transactions"
    ]
  }]
}
```

### Query parameters

| Param | Tipo | Obligatorio | Default | Notas |
|---|---|---|---|---|
| `from` | ISO 8601 | sí | — | Inicio inclusivo, con timezone (`Z` u offset). |
| `to` | ISO 8601 | sí | — | Fin exclusivo. `to > from`. |
| `sites` | CSV | no | todos | Lista de site IDs (`[a-zA-Z0-9_-]+`). Omitido = todos los sites. |
| `bucket` | enum | no | `15min` | Valores: `15min`, `1h`, `1d`. |
| `cursor` | opaco | no | — | Token de paginación (base64) del header `Link; rel="next"`. |
| `limit` | int | no | `1000` | Rango `[1, 5000]`. |

**Caps de rango por bucket** (devuelve `400 range-too-large` si se excede):

| Bucket | Máx |
|---|---|
| `15min` | 7 días |
| `1h` | 90 días |
| `1d` | 365 días |

### Response 200

```json
{
  "bucket": "15min",
  "data_freshness": {
    "site_54_21": "2026-05-25T13:42:18Z"
  },
  "rows": [
    {
      "site_id": "site_54_21",
      "bucket_start": "2026-05-25T10:00:00Z",
      "counts": {
        "in":  {"adult": 12, "child": 0, "unknown": 2, "total": 14},
        "out": {"adult": 11, "child": 0, "unknown": 3, "total": 14}
      },
      "external_traffic": {"passersby": 87, "shoppers": 22},
      "pos": {
        "sales": 5, "returns": 1, "transactions": 6,
        "items_sale": 18, "items_return": 2,
        "amount_minor_sale": 142350, "amount_minor_return": 12500,
        "currency": "ARS"
      }
    }
  ]
}
```

**Headers**:
- `ETag: "..."` — usable en `If-None-Match` (devuelve 304 si no cambió).
- `Cache-Control: public, max-age=86400, immutable` — solo si `to < now-1h`
  (datos cerrados). Caso contrario: `no-cache`.
- `Link: <...>; rel="first", <...&cursor=...>; rel="next"` — RFC 8288.
  `rel="next"` se omite cuando se llegó al final.

**Buckets vacíos**: si para `(bucket, site)` no hubo data, la fila se
devuelve igual con ceros (`currency` defaultea a `ARS`).

**`amount_minor`**: monto en la **unidad menor** de la moneda (ISO 4217).
Centavos para ARS/USD, mils para BHD/KWD, sin división para JPY/KRW.
BIGINT entero — evita errores de precisión de floats al sumar decimales.

### Errores — RFC 7807 (`application/problem+json`)

```json
{
  "type": "https://api.tfg.gasparri.com.ar/errors/range-too-large",
  "title": "Date range exceeds maximum for bucket size",
  "status": 400,
  "detail": "Requested 180.0 days with bucket=15min, max allowed is 7d",
  "instance": "/v1/aggregates",
  "parameter": "bucket",
  "requested_days": 180.0,
  "max_days_for_bucket": 7
}
```

Slugs del `type`:

| Status | `type` slug | Cuándo |
|---|---|---|
| 400 | `missing-parameter` | falta `from`/`to`. |
| 400 | `invalid-datetime` | datetime malformado o sin timezone. |
| 400 | `invalid-range` | `from >= to`. |
| 400 | `range-too-large` | excede el cap del bucket. |
| 400 | `invalid-bucket` | bucket inválido. |
| 400 | `invalid-site-id` | site no matchea `[a-zA-Z0-9_-]+`. |
| 400 | `out-of-range` | `limit` fuera de `[1, 5000]`. |
| 400 | `invalid-cursor` | cursor mal formado / expirado. |
| 403 | (sin body) | SigV4 inválido o IAM sin permiso. |
| 429 | (sin body) | throttle (10 req/s, burst 20). |
| 500 | `internal-error` | error transitorio — reintentar con backoff. |

### Throttling

10 req/s steady + burst 20, **a nivel API GW (no per-cliente)**. Devuelve
429 cuando se excede. Para per-cliente quotas habría que migrar a REST API
v1 (UsagePlans + API keys) — no implementado en el PoC.

### Ejemplos

**curl con SigV4**:
```bash
curl --aws-sigv4 "aws:amz:us-east-1:execute-api" \
     --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
     "https://api.tfg.gasparri.com.ar/v1/aggregates?from=2026-05-24T00:00:00Z&to=2026-05-25T00:00:00Z&bucket=1h"
```

**Python (boto3-style sin SDK custom)**:
```python
import boto3, requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

session = boto3.Session()
creds = session.get_credentials().get_frozen_credentials()

url = "https://api.tfg.gasparri.com.ar/v1/aggregates"
params = {"from": "2026-05-24T00:00:00Z", "to": "2026-05-25T00:00:00Z", "bucket": "1h"}

req = AWSRequest(method="GET", url=url, params=params)
SigV4Auth(creds, "execute-api", "us-east-1").add_auth(req)

resp = requests.get(url, params=params, headers=dict(req.headers))
resp.raise_for_status()
data = resp.json()

# Paginar siguiendo Link rel=next
next_link = resp.links.get("next", {}).get("url")
while next_link:
    req = AWSRequest(method="GET", url=next_link)
    SigV4Auth(creds, "execute-api", "us-east-1").add_auth(req)
    resp = requests.get(next_link, headers=dict(req.headers))
    data["rows"].extend(resp.json()["rows"])
    next_link = resp.links.get("next", {}).get("url")
```

**Re-consultar usando ETag** (ahorra bandwidth y compute para datos cerrados):
```bash
# Primera consulta — guardar el ETag del response
ETAG=$(curl -sS --aws-sigv4 "..." -D - "https://...?from=...&to=..." \
       | grep -i ^etag | awk '{print $2}' | tr -d '\r')

# Posterior — devuelve 304 sin body si no cambió
curl --aws-sigv4 "..." -H "If-None-Match: $ETAG" "https://...?from=...&to=..."
```

### Endpoint relacionado: `POST /pos/transactions`

Ingesta de transacciones del POS del cliente. Auth SigV4 igual que el query.
Idempotente por `(store_id, transaction_id)`. Spec completo en el OpenAPI.
Detalle del payload en `src/cloud/ingest_pos_transaction.py`.

---

## SQL directo (read-only)

Cómo conectarse a RDS Postgres como cliente externo (partner, analista) para
consumir las vistas de agregados del sistema vía SQL.

## Rol y permisos

El rol **`readonly_external`** (creado por `bootstrap.sql`) tiene
`SELECT` exclusivamente sobre las **vistas de agregados**, no sobre las
tablas crudas. Esto es deliberado: los analistas externos no necesitan
`device_id` ni timestamps fine-grained — necesitan métricas por store/hora/día.

### Vistas expuestas

Todas las vistas siguen el patrón `<fact>_by_bucket_{15min,hour,day}` (producto
cartesiano de granularidades). Lista completa de grants en
`bootstrap.sql` (`GRANT SELECT ON ... TO readonly_external`):

| Familia de vistas | Granularidades | Métricas |
|---|---|---|
| `counting_by_bucket_*` | 15min / hora / día | `ins`, `outs`, `net` + desglose demográfico (`ins_adult`/`ins_child`/`ins_unknown`) |
| `wifi_ble_by_bucket_*` | 15min / hora / día | `passersby`, `shoppers`, `weak` (categorizado server-side por `rssi_class`) |
| `pos_by_bucket_*` | 15min / hora / día | `transactions`, `sales`, `returns`, montos |
| `turn_in_rate_by_bucket_*` | 15min / hora / día | `turn_in_rate = ins / passersby` |
| `conversion_by_bucket_*` | 15min / hora / día | `conversion = ins / shoppers` |
| `occupancy_by_bucket_*` | 15min / hora / día | ocupación estimada (cumsum `ins - outs`) |
| `visit_duration_by_bucket_*` | hora / día‡ | duración de visita (Ley de Little) |
| `wifi_engagement_by_bucket_day` | día‡ | visitors WiFi/BLE por nº de ventanas presentes (`'1'`/`'2'`/`'3-5'`/`'6+'`) |
| `revenue_per_visitor_by_bucket_*` | hora / día‡ | `net_amount_minor / ins` (ticket por visitante) |
| `sales_per_sqm_by_bucket_*` | hora / día‡ | `net_amount_minor / sales_area_m2` (ventas por m²) |
| `data_freshness_by_store` | — | último `received_at` cross-fact por sucursal |
| `sites`, `devices` | dimensión | catálogo (lat/long, nombres) |

‡ `visit_duration`, `revenue_per_visitor` y `sales_per_sqm` solo exponen
`_hour` y `_day`; `wifi_engagement` solo `_day`. Las vistas internas
`metrics_unified_by_bucket_*` **NO** se conceden a `readonly_external` (uso de la
Lambda — el partner prefiere las vistas individuales por fuente). Las funciones
`height_class(REAL)` y `rssi_class(INT)` también tienen `GRANT EXECUTE`.

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
        SELECT store_id, bucket_hour, ins, outs
        FROM counting_by_bucket_hour
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
SELECT s.store_name, c.bucket_day, c.ins, c.outs, c.net
FROM counting_by_bucket_day c
JOIN sites s ON c.store_id = s.store_id
WHERE c.bucket_day >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY s.store_name, c.bucket_day;
```

### Turn-in rate por hora — última semana

```sql
SELECT store_id, bucket_15min, passersby, shoppers, turn_in_rate
FROM turn_in_rate_by_bucket_15min
WHERE bucket_15min >= NOW() - INTERVAL '7 days'
  AND store_id = '<store-id>'
ORDER BY bucket_15min;
```

### Conversion rate diario por store

```sql
SELECT store_id, bucket_day, visits, sales, conversion_rate
FROM conversion_by_bucket_day
WHERE bucket_day >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY bucket_day DESC, conversion_rate DESC NULLS LAST;
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

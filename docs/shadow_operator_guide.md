# Device Shadow — guía operativa

Cómo cambiar la configuración de un device en vivo, sin SSH ni restart, usando
AWS IoT Device Shadow.

## Qué se puede pushear

El device tiene una whitelist acotada de keys cloud-overridables —
deliberadamente chica, solo feature toggles end-user (no knobs técnicos).
Cualquier otra key del config sigue requiriendo SSH + edit + restart.

| Key | Default | Efecto |
|---|---|---|
| `operating_hours` | `{monday:"10:00-22:00", ..., sunday:"10:00-21:00"}` | Horario de la tienda por día. Fuera de horario, el conteo se pausa (a menos que `--ignore-operating-hours` en CLI). |
| `counting_enabled` | `true` | Toggle del conteo de personas (visión). `false` = pipeline corre pero no emite count events. |
| `external_traffic_enabled` | `true` | Toggle de la captura WiFi/BLE (passersby + shoppers). `false` = subsystem deshabilitado, ideal para pausa de privacidad por sucursal. |

Cualquier otra key pusheada se ignora silenciosamente (loggeada como
`shadow_delta_requires_restart` en el device).

## Cómo pushear — 3 caminos

### A) AWS Console (operator no técnico)

1. **IoT Core** → Manage → All devices → Things.
2. Click en el `thing_name` del device (ej. `store-pilot-01-cam-01`).
3. Tab **Device Shadows** → seleccionar el Classic Shadow.
4. Click **Edit** sobre el JSON.
5. Editar el bloque `state.desired` con los valores nuevos:

```json
{
  "state": {
    "desired": {
      "counting_enabled": false
    }
  }
}
```

6. Click **Update**.
7. AWS publica el delta al device; el device aplica (típicamente <2s si
   está online) y publica el `reported` confirmando.

### B) AWS CLI (operator técnico)

```bash
aws iot-data update-thing-shadow \
    --thing-name store-pilot-01-cam-01 \
    --cli-binary-format raw-in-base64-out \
    --payload '{"state":{"desired":{"counting_enabled":false}}}' \
    /tmp/shadow_response.json

cat /tmp/shadow_response.json
```

La respuesta confirma la versión nueva del shadow. El delta llega al device
vía MQTT en el siguiente ciclo del broker.

**Leer el state actual del shadow** (incluye `desired` + `reported`):

```bash
aws iot-data get-thing-shadow \
    --thing-name store-pilot-01-cam-01 \
    /tmp/shadow_get.json
cat /tmp/shadow_get.json
```

### C) boto3 (automatización / scripts)

```python
import boto3
import json

iot_data = boto3.client("iot-data", region_name="us-east-1")

# Pushear cambio
payload = {"state": {"desired": {"counting_enabled": False}}}
iot_data.update_thing_shadow(
    thingName="store-pilot-01-cam-01",
    payload=json.dumps(payload),
)

# Leer state actual
resp = iot_data.get_thing_shadow(thingName="store-pilot-01-cam-01")
shadow = json.loads(resp["payload"].read())
print(json.dumps(shadow, indent=2))
```

## Verificar que el cambio llegó al device

### 1. En el device (SSH si querés ver logs)

```bash
sudo journalctl -u people-counter -f | grep -E "shadow_delta_applied|shadow_reconciliation_published"
```

Esperás ver una línea como:

```
shadow_delta_applied keys=['cloud_defaults.counting_enabled'] count=1
```

### 2. Desde RDS / Grafana (sin SSH)

El device publica el campo `last_shadow_apply_ts` en su telemetry (cada
5 min). Query directa en Postgres:

```sql
SELECT device_id, event_ts, last_shadow_apply_ts
FROM telemetry
WHERE device_id = 'store-pilot-01-cam-01'
  AND last_shadow_apply_ts IS NOT NULL
ORDER BY event_ts DESC
LIMIT 3;
```

Si `last_shadow_apply_ts` está dentro de los últimos minutos = push aplicado.
Si es NULL en todas las filas o muy viejo = el push NO llegó (device offline
o flag `mqtt.shadow_enabled=false`).

## Política ante shadow inválido

Los deltas con valores inválidos se **rechazan en `apply_shadow_delta`
antes de persistir** al `config.yaml`. El último valor válido sigue
activo (sin pérdida de servicio).

Ejemplo: si pusheás `{operating_hours: {monday: "22:00-10:00"}}` (end <=
start, inválido), el log del device va a mostrar:

```
shadow_delta_rejected_invalid rejections=[{"key":"cloud_defaults.operating_hours",
  "reason":"monday: end '10:00' must be after start '22:00'"}]
```

El `config.yaml` no se reescribe; el `operating_hours` anterior sigue
operando. AWS sigue viendo `desired != reported` y va a re-publicar el
delta — la rechazo es continuo hasta que pushes un valor válido.

No hay modo `fail_open` / `fail_closed`: desde que el shadow persiste al
mismo `config.yaml` que el operator SSH-edita, no hay separación entre
"local válido" y "cloud inválido". La validación es 1 sola, fail-fast.

## Reconciliación post-reconnect

Cuando el device se conecta a IoT Core (boot o reconnect), publica
automáticamente su estado actual como `reported` en el shadow. Esto sirve:

- AWS sabe qué config tiene corriendo cada device.
- Si el `desired` del shadow no matchea el `reported`, AWS emite un delta
  inmediato — el device aplica al toque sin esperar al próximo push.

Esto es transparente al operator: no requiere acción.

## Troubleshooting

| Síntoma | Causa probable | Fix |
|---|---|---|
| Push hecho desde CLI pero `last_shadow_apply_ts` sigue NULL | `mqtt.shadow_enabled=false` en config del device | SSH al device, editar `/etc/people-counter/config.yaml` para flippearlo a `true`, restart |
| Push hecho pero el delta no aplica una key específica | Key fuera de `CLOUD_OVERRIDABLE` | Verificar log del device: `shadow_delta_requires_restart`. Las keys overridables están listadas arriba — el resto requiere SSH+restart |
| Device no responde a pushes y está online | Conexión MQTT al broker está degradada | `aws iot-data publish` test directo + revisar `mqtt_disconnect_count` en telemetry |
| Cambio fue rechazado por validación | Valor del shadow tiene formato inválido (ej. `operating_hours` con string mal formado) | Revisar log del device buscando `invalid_operating_hours_format`. Pushear un valor válido — el device queda con el last-known-good del config local mientras tanto |

## Resumen de keys NO overridables

Si necesitás cambiar alguna de estas, **SSH + edit `/etc/people-counter/config.yaml` + restart**:

- Cualquier knob de `vision.*` (resolución, FPS, SGBM tuning, AE lock).
- Cualquier knob de `detection.*` (model_path, thresholds, NMS, static suppressor).
- Cualquier knob de `tracking.*` (Kalman tuning, state machine, reid_gate_px).
- Cualquier knob de `counter.*` (ROI, líneas, height_classifier).
- Cualquier knob de `wifi_ble.*` que no sea `external_traffic_enabled` (interfaces,
  thresholds RSSI, stitching params).
- `mqtt.*`, `buffer.*`, `logging.*`, `status_led.*`, `telemetry.*`.

Es deliberado: estos knobs requieren visualización o análisis técnico para
calibrar bien — y si los configurás mal vía shadow, pueden romper el conteo
sin que te enteres hasta que veas el dashboard. SSH es la guardia natural.

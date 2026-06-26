# Diseño de OTA (actualización remota de la flota) — extensión post-PoC

Diseño de la actualización Over-The-Air para la flota de devices edge. **Estado:
propuesto, no implementado.** Este documento fija las decisiones de arquitectura
antes de escribir código; cada decisión lista la alternativa rechazada y por qué.

> Principio rector: los devices operan **desatendidos 12h/día, 365 días/año**. El
> peor caso de un OTA es **brickear un equipo que justamente no querés visitar**.
> Por eso el diseño prioriza **atomicidad + rollback automático** sobre simplicidad.

## 1. Alcance — qué se actualiza y qué NO

| Artefacto | ¿OTA? | Cadencia | Notas |
|---|---|---|---|
| Código de la app (`src/`) | ✅ | frecuente | El grueso del OTA. |
| Modelo HEF (`people-counter-detector`) | ✅ | media | Versionado **aparte** del código (un HEF nuevo puede requerir postproc nuevo → check de compatibilidad). |
| Config de negocio (3 toggles) | ya existe | — | Device Shadow, en caliente. Fuera de este OTA. |
| `config.yaml` técnico | ❌ (por ahora) | rara | SSH. Podría sumarse después como artefacto firmado. |
| **Calibración per-device** | ❌ **NUNCA** | — | Única por equipo. El OTA jamás la toca (vive en estado persistente, ver §4). |
| OS / deps de sistema / nexmon | ❌ | muy rara | Track separado e infrecuente (`unattended-upgrades` para parches; re-provisioning para deps nuevas). Ver §3. |

## 2. Decisiones de arquitectura

### 2.1 Transporte: AWS IoT Jobs + S3 (presigned URL)
Reusa el broker MQTT + los certs X.509 que la flota **ya tiene**. Dos planos:
- **Control (MQTT)**: IoT Jobs notifica y trackea estado por device. Tope ~128KB/msg.
- **Datos (HTTPS/S3)**: el bundle (MB) se descarga de S3 con **presigned URL de vida
  corta** scopeada al objeto. Nunca por MQTT.

*Rechazado:* mecanismo propio (reinventar transporte + auth ya resueltos); descarga
por MQTT (límite de tamaño, no es para bulk).

### 2.2 Packaging: file-swap de release dir + symlink (NO venv/container)
El bundle es **código `src/` + modelos HEF**. Se despliega a `releases/<version>/` y
se activa con un swap de symlink `current` (§4). **Asume que las deps ya están en el
device** (instaladas en provisioning).

*Rechazado — venv por release:* las deps pesadas (`hailo_platform`, `picamera2`,
`opencv-contrib`) son **system-level / apt**, no empaquetables limpio en un venv.
*Rechazado — contenedores:* Hailo + cámara CSI + 2GB de RAM hacen el passthrough y el
overhead dolorosos; cambio arquitectural grande sin beneficio para 1 binario.
**Consecuencia:** si un update necesita una dep nueva, NO va por este OTA → es un
re-provisioning (raro). El job doc declara `min_base_version`; si el device no la
cumple, **rechaza el job** en vez de romperse.

### 2.3 Firma de código: Ed25519 (lib `cryptography`), pubkey embebida
Se firma el bundle con una clave privada (en CI/Secrets Manager). El device verifica
**sha256 + firma Ed25519** con la pubkey embebida en la imagen base **antes** de
aplicar. Sin firma válida → descarta.

*Rechazado — AWS Signer:* válido y managed, pero más infra/curva; Ed25519 con
`cryptography` es ~30 líneas y suficiente. (Migrable a Signer si la flota crece.)

### 2.4 Swap atómico + A/B en disco
```
/opt/people-counter/
  releases/
    1.4.1/            ← versión anterior (rollback sin re-descarga)
    1.4.2/            ← versión nueva
  current -> releases/1.4.2     ← symlink atómico; systemd ExecStart lo usa
  ota/
    download/         ← staging del bundle (verify antes de promover)
    pubkey.pem        ← verificación de firma
  shared/             ← ESTADO PERSISTENTE, nunca tocado por un update
    outbox.sqlite
    dedup.sqlite
```
El swap es `ln -sfn releases/<new> current.tmp && mv -T current.tmp current` (rename
atómico POSIX) → **sobrevive corte de energía** (nunca hay un `current` a medias). Se
retienen las últimas **N=2 releases** para rollback. `/etc/people-counter/` (config +
certs + **calibración**) queda intacto.

*Rechazado — A/B de particiones de OS (RAUC/Mender):* robusto pero pesado;
sobredimensionado para OTA de app. Reevaluable a nivel flota.

### 2.5 Health-check + rollback automático (la parte crítica)
Tras el swap, `systemctl restart` y se evalúa salud con timeout. **Triggers de rollback:**
1. El servicio no llega a `READY=1` (sd_notify) en `boot_timeout` (ej. 90s).
2. **Crash-loop**: ≥3 reinicios en `crash_window` (ej. 5min) — distinto del
   `Restart=always`, que reiniciaría infinito una versión rota.
3. Health probes en rojo sostenido (captura/inferencia/MQTT) post-boot.

**Mecánica del rollback:** re-apuntar `current` a la release anterior (ya en disco,
sin re-descarga) → restart → reportar `FAILED` + razón por el Job. Un device que
rollbackeó **sigue operando con la versión vieja** (no se cae del conteo).

### 2.6 Ventana de update: solo en horario cerrado
El device ya conoce `operating_hours` → el agente **difiere la aplicación** hasta que
la tienda esté cerrada (descarga+verifica antes; aplica fuera de horario). Evita
cortar el conteo a mitad de turno.

### 2.7 Preservación de estado
`outbox.sqlite` (eventos in-flight sin PUBACK) y `dedup.sqlite` + salt viven en
`shared/` (fuera del release dir) → **sobreviven el swap sin migración**. La
calibración vive en `/etc/people-counter/` → intacta. El estado in-memory del Counter
se resetea diario igual; un update en ventana cerrada no pierde conteo del día.

### 2.8 Observabilidad del propio OTA
Sumar a `telemetry`: `app_version`, `model_version`, `last_ota_status`
(`success`/`failed`/`rolled_back`), `last_ota_ts`. El tablero ⑤ gana un panel de
**skew de versiones** de la flota → ves el avance del rollout y qué device falló.

## 3. Rollout staged (canario)
IoT Jobs targetea **thing-groups**:
1. **`canary`** (1 device) → se observa N horas en ⑤ (FPS, `track_stitching_ratio`,
   temps, `last_ota_status`, crash count).
2. **`fleet`** → recién si el canario quedó verde. Job con `rolloutConfig`
   (rate-limited) + `abortConfig` (corta el rollout si el % de fallas supera un umbral).

## 4. Flujo end-to-end
```
CI: build bundle (src/ + HEF) → firma Ed25519 → sube a S3 (versionado) → crea IoT Job (group=canary)
IoT Core: notifica por MQTT → device baja el job doc (con presigned S3 URL)
Device: descarga de S3 (HTTPS) → verifica sha+firma → stage en ota/download/
        → ESPERA ventana cerrada → swap atómico de `current` → restart
        → health-check: OK → reporta SUCCEEDED + versiones | FALLA → rollback + FAILED
Operador: observa el canario en ⑤ → promueve el Job al group `fleet`
```

## 5. Job document (esquema propuesto)
```json
{
  "ota": {
    "type": "app" | "model" | "bundle",
    "version": "1.4.2",
    "model_version": "detector-v2",
    "min_base_version": "1.0.0",
    "artifact_url": "${aws:iot:s3-presigned-url:...}",
    "sha256": "<hex>",
    "signature": "<base64 ed25519>",
    "apply_window": "closed_hours" | "immediate"
  }
}
```

## 6. Seguridad
- Bucket S3 **privado + versionado**; acceso solo vía presigned URL de vida corta.
- **Firma Ed25519** verificada en el device antes de aplicar (clave privada en
  Secrets Manager, nunca en el repo).
- Job autenticado por el cert X.509 del device (lo que ya hay).
- Principio de mínimo privilegio en el rol de presigned URLs.

## 7. Plan de fases (estimación ≈ 45–73 h)

| Fase | Entregable | Horas |
|---|---|---:|
| **0** | Este design doc + esquema acordado | ✅ |
| **1** | Infra AWS: S3 bucket + IoT Jobs + thing-groups + firma (CFN) | 8–12 |
| **2** | Build/CI: empaquetar + firmar + subir + crear Job | 4–6 |
| **3** | Agente de Jobs en el device (paho): lifecycle + download + verify | 12–18 |
| **4** | Swap atómico A/B + ventana + preservación de estado | 8–12 |
| **5** | Health-check + rollback automático | 6–10 |
| **6** | Versión en telemetría + panel de skew en ⑤ | 2–3 |
| **7** | Testing: harness de fallas + E2E en HW (canario, update malo) | 10–18 |

## 8. Riesgos / decisiones abiertas
- **Migración de layout** `/usr/src/people-counter` → `/opt/people-counter/releases/` +
  symlink. Cambia `setup_device.sh`, el `.service` (ExecStart al symlink), y los
  `ReadWritePaths`. Es un cambio de provisioning a coordinar.
- **Deps nuevas** quedan fuera del OTA (re-provisioning). Aceptable si son raras;
  si no, replantear hacia un base-image versionado.
- **Testing de fallas reales** (power-loss mid-swap, crash-loop) **requiere HW** e
  inyección deliberada — es el cuello de botella de validación.

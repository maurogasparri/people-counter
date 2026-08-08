# CLAUDE.md — People Counter Edge System

## Descripción general

Sistema de conteo de personas de bajo costo para locales comerciales. Visión estéreo + IA en el borde + detección pasiva de tráfico WiFi/BLE.

**Proyecto de producción real**: dispositivos operan desatendidos 12h/día, 365 días/año. Calidad de código, manejo de errores y resiliencia son críticos.

## Arquitectura

```
+---------------------------------------------+
|        Dispositivo edge (por local)          |
|  RPi5 2GB + Hailo-8L 13T + 2x IMX708         |
|                                              |
|  +----------+  +----------+  +--------+      |
|  |  Visión  |  | WiFi/BLE |  |  MQTT  |      |
|  | Stereo → |  | Monitor  |  | QoS 1  |      |
|  | YOLOv8n →|  | Probe →  |  | Buffer |      |
|  | Track →  |  | Hash →   |  | SQLite |      |
|  | Count    |  | Dedup    |  |        |      |
|  +----------+  +----------+  +--------+      |
+------------------+---------------------------+
                   | MQTT (TLS + X.509)
                   v
+--------------------------------------------------+
|              AWS Cloud (PoC, 1 device)            |
|                                                   |
|  IoT Core (3 Topic Rules) ──► Lambda             |
|                               persist_event       |
|                               (out of VPC,        |
|                                IAM auth a RDS)    |
|                                       │           |
|                                       ▼           |
|                              RDS Postgres 16      |
|                              (db.t4g.micro,       |
|                               force_ssl)          |
|                                       ▲           |
|                                       │ datasource|
|  ECS Fargate + ALB ── Grafana 13 ─────┘           |
|  (custom domain HTTPS,                            |
|   ACM cert auto-renewed)                          |
|                                                   |
|  CloudFormation orquesta TODO (infra/deploy.ps1)  |
+--------------------------------------------------+
```

## Hardware por unidad

- Raspberry Pi 5 2GB + Active Cooler (RAM dimensionada empíricamente — working set ~270 MB sostenido, peak 281 MB; ver `docs/hardware_sizing.md`)
- Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) — stackeado (vía PCIe)
- 2× Arducam IMX708 12MP HDR M12 120° HFOV (B0310) vía CSI — par estéreo, baseline 14cm
- Waveshare PoE HAT (G) 25.5W — alimentación PoE, stackeado (orden: AI HAT+ → Pi → Active Cooler → PoE HAT)
- LED RGB 3mm common cathode — GPIO 17/18/27 + GND pin 14, resistores 150/100/100Ω
- SanDisk Extreme 64GB microSD

## Decisiones técnicas clave

### Pipeline de visión

- **Calibración estéreo**: ChArUco A3 (9×6 / 45mm / 33mm / DICT_4X4_100), modelo fisheye Kannala-Brandt (`cv2.fisheye.calibrate`). Lab universal (mount 2.0-3.5m): foco a 1.5m, calibración a 1.0/2.0/3.0m. El wizard de calibración corre por **default en barrido libre** (sweep: auto-selección por novedad/quietud + mapa de cobertura en vivo; `--guided` = 20 poses-silueta clásicas) y `focus_assist` por **default en modo mapa** (acumula nitidez máx por zona paseando el board; `--static` = un frame al centro) — ambos con reporte HTML, pensados para espacios chicos / luz difícil. Validar con `scripts/diagnose_depth.py` (error centro <5% a 2m, <10% a 3m). `detect_charuco_dual_pass` reintenta con sharpen 3×3 si quedó <8 corners. Salud de calibración post-cal: `scripts/diagnose_calibration.py` rectifica con la calib guardada y mide error epipolar L↔R (sub-px sano, >1px → recalibrar). El QC de bracket pre-cal se retiró — medir geometría fisheye con modelo pinhole sin distorsión da yaw/offsets fantasma; la geometría real la valida el reporte de calibración (RMS estéreo + alineación sobre extrínsecos reales).
- **Óptica**: Arducam B0310 fisheye real, no rectilíneo. Focal pinhole-equivalente `f_px = 2050` a full-res 4608×2592. La fórmula `f = (W/2)/tan(HFOV/2)` NO aplica.
- **Sensor mode canónico**: 2304×1296 binned 2×2, full FOV, 16:9, hasta 56 fps. Runtime puede usar rescale lineal (ej. 1152×648 vía `scripts/rescale_calibration.py`) — K-B es resolución-independiente en intrínsecos angulares. **CRITICAL**: en TODO call site de picamera2 pasar `raw={"size": CANONICAL_RAW_SIZE}` (importar de `src.vision.capture`) para seleccionar Mode 1 (2304×1296, HFOV 120°); sin ese hint picamera2 elige Mode 0 (1536×864 cropeado, HFOV ~80°) que reduce la cobertura a la mitad. Overrideable per-device vía `sensor.default_res` del config de hardware (el viejo `vision.sensor_raw_size` quedó deprecado — `src/config/loader.py` lo elimina con warning).
- **Rectificación**: `cv2.fisheye.initUndistortRectifyMap` (balance=0.0) + `cv2.remap`.
- **Profundidad**: SGBM + matcher derecho + WLS filter. `vision.num_disparities: auto` deriva el rango desde `mounting_height_m`.
- **Detección**: YOLOv8n fine-tuneado (cenital), HEF para Hailo-8L. NMS on-chip. VStream API con scheduler ROUND_ROBIN. Modelo activo: `people-counter-detector`. Pipeline en `scripts/training/README.md`.
- **Tracking**: euclidiano 3D (x, y, profundidad) con Kalman per-track + state machine corta (CANDIDATE → CONFIRMED → PENDING → LOST).
- **Conteo**: línea virtual + counting zone rectangular. Track entra a la counting zone → cruza línea → sale = evento `direction='in'`/`'out'` (terminología canónica fleet-wide, schema y MQTT). Publicación inmediata vía MQTT.

### Captura WiFi/BLE

- **WiFi**: CYW43455 monitor mode vía nexmon. Captura probes en 2.4 + 5 GHz. **WiFi solo probing — red por Ethernet**. Monitor mode requiere `nexutil -m2` (radiotap en el firmware) — sin eso el netdev entrega frames Ethernet (DLT EN10MB) y scapy no ve 802.11; el capture loop re-parsea los bytes crudos como `RadioTap()`. nexutil se compila aparte (`setup_device.sh`). El pipeline corre como `User=pi`: `/dev/rfkill` necesita estar en grupo `netdev` (udev rule) para que el unblock funcione.
- **BLE**: bleak (D-Bus de BlueZ), escaneo pasivo.
- **Solo dispositivos portátiles** (`wifi_ble.randomized_only`, default true): se cuentan solo MACs WiFi randomizadas (locally-administered bit 0x02) y BLE `AddressType=random` (RPA iOS / aleatoria Android). Las MAC globales WiFi y los BLE `public` (OUI real) son infra/IoT fijo (APs, smart-TVs, beacons, parlantes) y se descartan antes de hashear. Apagable per-site. Nota: las rotaciones de RPA BLE de un mismo teléfono son todas `random` — eso lo resuelve el stitching, no este filtro.
- **Hashing**: SHA-256 truncado a 16 bytes, con **salt local persistido** (`dedup_meta` del SQLite, rotado en `reset_daily`) — **nunca MACs crudas**, nunca hash sin sal at-rest. `process_detection` corre bajo un `threading.Lock` (los dos productores —WiFi en el thread de scapy, BLE en el de bleak— comparten el mismo SQLite). El SQLite del dedup se abre vía el helper `_connect()` con `journal_mode=WAL` + `busy_timeout=5000` + `synchronous=NORMAL` (gotcha no obvio: sin WAL la contención del BLE bloqueaba el event loop de bleak y mataba el FPS de visión — readers y writer dejan de excluirse mutuamente).
- **Dedup → hash groups con stitching** (`src/wifi_ble/dedup.py`): los hashes se asocian a un `group_id` por identidad del dispositivo. Cada `process_detection` aplica 4 reglas y joina al grupo más reciente que matchee:
  1. **Seqnum continuity (WiFi-only)** — el seqnum 802.11 (12 bits, `dot11.SC >> 4`) es contador del chip, continuo cross-MAC-rotation. Match: Δseqnum ≤ `max_delta` (100, wrap mod 4096) + ΔRSSI ≤ 5dBm + Δt ≤ 30s. Defeated por Apple H1+ (iPhone 12+) que resetea seqnum on MAC change; OK en Android.
  2. **Cross-protocol L2 (short window)** — WiFi MAC y BLE addr dentro de `cross_protocol_window_seconds` (2s) con ΔRSSI ≤ 5dBm = mismo dispositivo.
  3. **BLE anchoring (long window)** — durante la vida de un BLE RPA (~15min iOS), nuevas WiFi MACs con RSSI compatible se mergean al grupo del BLE. Cubre "WiFi rota cada 2min, BLE cada 15min" donde la regla 2 no alcanza.
  4. **Fingerprint continuity (mismo protocolo)** — fingerprint estable (orden de IEs + HT/VHT/HE caps WiFi; company ID + subtipos Continuity de Apple + service UUIDs + TX power BLE; ver `fingerprint.py`) que sobrevive la rotación. Cubre lo que el seqnum NO agarra (Apple H1+ resetea seqnum pero el fingerprint es estable). Además es **filtro duro** en la regla 1 (seqnums que coinciden por azar pero con fingerprint distinto = devices distintos). Caveat: dos devices idénticos co-presentes pueden mergearse (leve subconteo); el gate de RSSI lo acota.
- **Privacy del stitching**: seqnum y timestamps quedan SOLO en `wifi_ble_dedup.sqlite` local (rotado diario). El MQTT publish manda solo el `group_id` opaco — un **UUID random** (`uuid.uuid4().hex`, NO derivado del hash → inlinkeable) — nunca hashes pre-stitching, seqnums ni MACs crudas.
- **Stitching canary**: `get_stitching_ratio()` = `groups / hashes` del día. 1.0 = ningún stitch, 0.5 = mitad mergeada. Va en telemetry (`wifi_ble_stitching_ratio`) — detecta si la flota corre con OS que defeatean las reglas.
- **Publish per-window events** (`publisher.py`): cada ventana (15 min) emite UN array `devices[]` con UN evento por group, shape `{ visitor_hash, protocol, rssi_max, first_seen_ts, last_seen_ts }`. RSSI crudo — la cloud aplica `rssi_class(rssi_max)` server-side (single source of truth de los thresholds, modificable con `CREATE OR REPLACE FUNCTION` retroactivo). Si la query del dedup falla NO avanza la ventana (reintenta; recién tras N fallos consecutivos la da por perdida) — sin esto un lock transitorio perdía 15min de tráfico irrecuperable.

### Comunicación

- **MQTT** 3.1.1 sobre AWS IoT Core, X.509, QoS 1.
- Eventos de conteo en tiempo real, resúmenes WiFi/BLE cada 15min, telemetría cada 5min.
- **Buffer SQLite local** (outbox): replay al reconectar, marca enviado SOLO tras PUBACK. El replay saltea mensajes ya in-flight (anti-duplicado en reconnect). `mark_sent` es defensivo — un fallo de SQLite no burbujea al callback de paho.

### Status LED

- 6 estados = 3 colores puros (rojo / verde / azul) × 2 modos (fijo / blink), en cascada worst-first: HW > pipeline > internet > cloud > unprovisioned > OK. Mapeo: rojo fijo = HARDWARE_FAULT (incluye crash/wedge de init), rojo blink = SOFTWARE_FAULT, verde blink = sin internet, verde fijo = sin cloud, azul blink = sin provisioning, azul fijo = OK. Sin amarillo (R+G se veía verdoso por el balance de brillo). No hay estado "boot" dedicado: el LED no puede señalizar un fallo previo a su propio init (SO/intérprete), así que un init colgado/crasheado cae en HARDWARE_FAULT (rojo fijo) vía `init_failed`/grace en el monitor.
- Health probes: CPU/Hailo temp, disco, calibración, captura/inferencia, watchdog (`last_loop_ts` <5s), internet TCP a 1.1.1.1:53, MQTT connected.
- Monitor en thread separado (probes blocking no estresan el hot path). Los dos probes lentos —internet (socket 3s) y Hailo temp (subprocess `hailortcli` hasta 5s)— corren con cadencia propia cacheada, NO en cada tick. Fail-safe: sin GPIO → no-op + log INFO.
- LEDs onboard ACT/power/Ethernet/audio apagados via dtparam — el RGB externo es la única fuente visual.

### Cloud (AWS)

- **PoC actual (1 device, deployado)**: IoT Core (broker + 3 Topic Rules SQL) → Lambda `persist_event` (`src/cloud/persist_event.py`, fuera de VPC, IAM auth a RDS via `rds.generate_db_auth_token`) → RDS Postgres 16 (db.t4g.micro, single-AZ, IAM auth + `rds.force_ssl=1`, `AutoMinorVersionUpgrade=true`). Grafana 13 en ECS Fargate detrás de ALB con ACM cert custom (image desde ECR, `grafana.<tu-dominio>`) lee el mismo RDS. Orquestado por CloudFormation (`infra/cloudformation/people-counter.yaml`) + `infra/deploy.ps1` (6 fases, `-StartFromPhase`). El cert ACM se crea fuera de CFN para que el deploy no bloquee esperando validación DNS (el ARN entra como parámetro). Schema en `infra/sql/bootstrap.sql`. **Caveat de sizing en bulk-load**: `db.t4g.micro` (1GB) OOM-ea en cargas masivas (~1.5M filas en una sola transacción — re-seed, migración de histórico) → escalar temporal a `db.t4g.small` (2GB) o batchear los commits (es lo que hace `scripts/migrate_historical.py`).
- **Producción (rollout de flota)**: RDS single-AZ → Multi-AZ. Los 2 pasos manuales de DNS del deploy (CNAMEs de validación ACM + CNAME final al ALB) son por usar un DNS externo en el PoC; con DNS gestionado no aplican.
- **Servicios cloud compartibles**: al sumar 2+ services (sales API, auth) se comparte el ALB via listener rules (en vez de provisionar uno por service).
- **Multi-cam por sucursal**: al agregar 2+ cámaras por local, la consolidación del tráfico WiFi/BLE se resuelve por **agregación MAX en la nube** (vistas `wifi_ble_by_bucket_*`: cada unidad publica su propio conteo y la vista toma el máximo entre devices, NO la suma — evita doble-contar a la misma persona vista por 2 cámaras, sin coordinación entre dispositivos).

## Convenciones de código

- **Lenguaje**: Python 3.13 (RPi OS Trixie). Black 88 chars, Ruff, type hints obligatorios.
- **Logging**: módulo `logging` JSON estructurado. DEBUG dev / INFO prod.
- **Config**: archivo único per-device.
  - `/etc/people-counter/config.yaml` (per-device): fuente única de verdad en runtime. `load_config()` lee SOLO este archivo (sin merge con el example) y valida todas las keys requeridas en `_validate`.
  - `config/config.example.yaml` (repo): TEMPLATE documentado de la flota. NO se mergea en runtime — sirve para provisionar un device nuevo (operator copia → edita) y auditar qué keys son válidas. Cambios fleet-wide = editar el template + redeployar a cada device. Una whitelist chica de toggles end-user (`CLOUD_OVERRIDABLE`: `operating_hours`, `counting_enabled`, `external_traffic_enabled`) se pushea vía AWS IoT Device Shadow sin restart (workflow en `docs/shadow_operator_guide.md`). Los deltas se persisten al MISMO `config.yaml` (el operator SSH ve lo que corre). Deltas inválidos se rechazan en `apply_shadow_delta` ANTES de escribir — cada toggle tiene su validador (los bool exigen bool real; `"false"` string se rechaza porque sería truthy).
  - **Hardware-agnostic**: los parámetros que dependen del hardware/setup (sensor, lens, bracket, board ChArUco, AE timings) están en `src/config/hardware.py` (`HardwareParams` + `load_hardware_params()`). Cambiar de hardware = editar keys del config; ningún script tiene constantes hardware hardcodeadas. Coerción lenient (scalar no-numérico → fleet default, no crash). Los setup tools y el runtime leen los mismos valores.
- **Secrets**: certificados X.509 en `/etc/people-counter/certs/`. **Nunca commitear**.
- **Tests**: pytest, estructura espejo de src.
- **No usar clases salvo que haya estado.** Tracker, MQTTClient justifican clases. Resto = funciones.
- **Todo I/O wrapeado** con manejo de errores.

## Estructura del directorio

```
people-counter/
├── CLAUDE.md, README.md, LICENSE, pyproject.toml, .gitignore, .gitattributes
├── src/
│   ├── vision/         <- capture, calibration, depth, detect, static_suppressor, world_coords, best_frame
│   ├── tracking/       <- tracker (Kalman + state machine), counter (counting zone + line crossings)
│   ├── wifi_ble/       <- wifi_probe, ble_scan, fingerprint, hasher, dedup, publisher
│   ├── mqtt/           <- client (AWS IoT), buffer (SQLite outbox)
│   ├── cloud/          <- persist_event / ingest_pos_transaction / query_aggregates (Lambdas)
│   ├── status/         <- led, health, monitor (background thread)
│   ├── config/         <- loader (strict) + hardware (HardwareParams dataclass)
│   └── main.py         <- orquestador del pipeline
├── tests/              <- 1103 tests (1096 pasan y 7 se saltean en Windows: 5 son
│                          POSIX-only y 2 piden el WLS de opencv-contrib, ausente
│                          en el wheel de Windows), espejo de src
├── scripts/
│   ├── calibrate.py    <- wizard end-to-end (browser-driven, ChArUco). Default barrido libre (sweep); --guided = 20 poses-silueta
│   ├── focus_assist.py <- asistente de foco (browser-driven). Default modo mapa (nitidez máx por zona); --static = un frame
│   ├── diagnose_depth.py <- valida calibración (5 zonas, error centro <5% a 2m)
│   ├── diagnose_calibration.py <- monitor de salud de calibración post-cal (error epipolar; ¿recalibrar?)
│   ├── preview.py      <- preview live MJPEG L|R (browser-driven)
│   ├── download_model.py, capture_baseline_frames.py, rescale_calibration.py
│   ├── provision.py    <- create/deploy/harvest/reprovision/list + seed sites/devices en RDS (psycopg+boto3)
│   ├── migrate_historical.py <- loader CSV→staging batcheado (commits incrementales) para histórico AGREGADO; doc en el header del archivo
│   ├── reset_dedup.py  <- reset diario del dedup + rotación de salt (lo llama people-counter-reset.service). El estado in-memory del Counter lo resetea main.py en el rollover de fecha (el script externo no puede tocarlo).
│   ├── verify_hardware.py, setup_device.sh
│   └── training/       <- pipeline X-AnyLabeling + active learning + Kaggle (ver scripts/training/README.md)
├── training_data/      <- gitignoreado salvo README + sites.yaml.example. Workspace local.
├── calibration/        <- board ChArUco A3 PDF (calib.io)
├── infra/              <- cloudformation/ + sql/ (bootstrap + migrations + migrate_historical_rollups.example.sql, template staging→tablas base rollup_*) + deploy.ps1
├── docs/               <- setup_guide, lab_calibration_guide, operator_guide, tracker_tuning, counter_test_matrix
└── config/             <- config.example.yaml + people-counter.service (systemd)
```

## Plan de sprints

| Sprint | Foco | Estado |
|--------|------|--------|
| S1 | Análisis y diseño inicial | **DONE** — repo, dependencias, arquitectura base |
| S2 | Captura estéreo y servicios | **DONE** — picamera2, dual cam, servicios systemd |
| S3 | Calibración estéreo | **DONE** — fisheye K-B (ChArUco A3), `diagnose_depth`, wizard browser-driven |
| S4 | Profundidad y región de interés | **DONE** — SGBM + WLS, counting zone rect, num_disparities auto |
| S5 | Detección neuronal | **DONE** — YOLOv8n cenital fine-tuneado, HEF Hailo-8L, `people-counter-detector` v2 (544 imgs train + 245 val) |
| S6 | Seguimiento y conteo | **DONE** — tracker Kalman + state machine + ghost pool/ID adoption + counter net-balance + death-emit con guards (rescue-with-guardrails) |
| S7 | Captura WiFi y BLE | **DONE** — nexmon + nexutil/radiotap, hopping ponderado, bleak, filtro de infraestructura fija, hashing + dedup 4 reglas, MAX-RSSI para shoppers |
| S8 | Mensajería y telemetría | **DONE** — IoT Core + buffer SQLite + replay + canaries (`track_stitching_ratio`, `death_emit_count`, `ghost_adoption_count`, `wifi_ble_stitching_ratio`, `last_shadow_apply_ts`) + Device Shadow con 3 toggles overridables |
| S9 | Servicios cloud y APIs | **DONE** — CloudFormation + Lambdas `persist_event`/`ingest_pos_transaction`/`query_aggregates` + RDS Postgres 16 (bucket server-derived via GENERATED) + funciones SQL `height_class(REAL)` y `rssi_class(INT)` + tabla `wifi_ble_events` (per-device post-stitching, RSSI crudo, `visitor_hash` opaco) + tablas `sites`/`devices` + Grafana 13 en ECS Fargate detrás de ALB con HTTPS |
| S10 | Visualización analítica | **DONE** — Grafana 13: 5 tableros en 2 carpetas por público (Analítica comercial: Panorama/Comparativa/Detalle/Patrones · Operación y flota: Salud de la flota) + comparativas vs período anterior y vs promedio del grupo (② y ③) + feriados + sidecar grafana-image-renderer (PNG) + alert rules versionadas en 3 grupos (Hardware/Operación/Negocio, incl. pico de tráfico por franja US-02). Import idempotente en `infra/grafana/` (`import_dashboards.ps1` / `import_alerts.ps1`). |
| S11 | Validación y documentación | **DONE** — rename de claves de config `counter.roi`→`counting_zone` + `pre_filter`→`tracking_zone` (los módulos siguen siendo `pre_filter.py` etc.; solo cambió la clave) + 4 guards anti-FP (`min_count_height_m`, `min_real_inside_frames`, `height_confidence_gate`, `tracking_zone` polygon) + keepalive condicional a entry real + SGBM depth cache (`vision.depth_skip_stable_tracks`) + `--profile-slow-threshold-ms`. Runbook `docs/tracker_tuning.md` (6 patrones) + matriz `docs/counter_test_matrix.md` (17 dimensiones). Pasada de hardening (review-driven): reset diario del Counter en rollover de fecha (los canaries eran lifetime, no daily), salt local rotado + lock en dedup, validación de toggles bool del shadow, Hailo temp probe cacheado fuera del tick del LED, publisher con retry de ventana, fix de `data_freshness_by_store` (referenciaba `wifi_ble_summary` dropeada). 922 tests verde. **Tooling de setup (2026-06-19, ~6.5h)**: `diagnose_bracket`→`diagnose_calibration` (salud post-cal por error epipolar) + modo barrido (sweep) default en calibración + modo mapa default en foco, ambos con reporte; validado en HW (barrido baseline 143.2mm + 0.59px epipolar out-of-sample; mapa de foco PASS). |
| S12 | Cierre del prototipo | **DONE** — TRACKDBG promovido a diagnóstico permanente (marcador `[REVERT]` removido; logs DEBUG `TRACKDBG …` + runbook `tracker_tuning.md` intactos). Hardening final: audit confirmó código ya sólido + sandboxing systemd **verificado en hardware** (exposure 4.4 OK, probing WiFi/BLE/MQTT intactos; `ProtectClock` removido pre-deploy — su `DeviceAllow` implícito bloqueaba cámaras/Hailo/GPIO). Revisión cross-proyecto multi-agente (2026-06-09): 19 hallazgos fixeados en `d209689` (ALTA: drain completo del outbox MQTT + cap de body del viewer; MEDIA: salt del dedup recargado en proceso vivo, mantenimiento del main loop al tope del loop + escalación de watchdog, BLE stop-event por generación, sesiones admin random+expiry, persist_event descarta constraint violations, shadow persist atómico, LED de fallo de init cableado, best_frame con crops, regla 3 stitching unidireccional, tracker pass-2 consumidas + sentinel z=0; y 7 BAJAS). Perf CPU (`d209689`): 162%→102% (RGB888 sin cvtColor — nombres de formato picamera2 invertidos, rect_r lazy, pre-filtro de bytes en scapy: BPF kernel NO compila en DLT nexmon) →**~50% en escena vacía** con `vision.idle_throttle` (10 FPS idle, wake instantáneo), **validado de extremo a extremo en las instalaciones** (9 cruces contados y persistidos en RDS bajo throttle, alturas y conf plausibles). Status LED simplificado a 6 estados (3 colores puros × fijo/blink): se eliminó el amarillo (R+G se veía verdoso por balance de brillo) y el estado `BOOT_FAILURE` se fundió en `HARDWARE_FAULT` (rojo fijo) — el LED no puede señalizar un fallo previo a su propio init, y la acción del operador es la misma; señales internas `boot_*`→`init_*`. S10 cerrado (regla de alerta `death_emit/total > 0.3` + 12 reglas en 3 grupos). Coverage report (`docs/coverage_report.md`, 81% total; núcleo 82–98%) + pasada de hardening de tests: guards del death-emit aislados (counter 97%) + resiliencia de las Lambdas (persist_event/ingest_pos 98%) + fix de race latente en el test del viewer. Gantt + plan de sprints (~240h plan). **Cierre final (2026-06-26)**: L1 (compresión de estatura a mount bajo) MITIGADO y validado A/B en las instalaciones (techo `max_head_height` + SGBM); TC-01…07 dirigido (registro reescrito 2026-08: 17 de 19 casos cumplen; TC-03 LIMITACIÓN DOCUMENTADA = sub-conteo en cruces simultáneos bidireccionales, causa NO aislada con el registro disponible; TC-06 CUMPLIMIENTO PARCIAL por la limitación L5); canary `ambiguous_reject_count` (device→Lambda→RDS→panel Grafana); freq-cap del CPU integrado al deploy (`config/cpu-freq-cap.service`); re-seed demo a 2026-06-30 con columnas de salud nuevas. **1103 tests** (1096 pasan y 7 se saltean en Windows: 5 POSIX-only y 2 que piden el WLS de opencv-contrib, ausente en el wheel de Windows; sobre el dispositivo Linux no hay omisiones). Entregables documentales del TFG fuera del repo. |
| Ext. | Rollout de flota (post-PoC, fuera del plan original) | PENDIENTE — OTA con AWS IoT Jobs + S3 (firma Ed25519, swap atómico A/B + rollback automático, canario, ventana de horario cerrado). Diseño en `docs/ota_design.md` (7 fases, ~45-73h). Junto con Multi-AZ del backlog. |

## Reglas duras

- **No transmitir video/imágenes.** Solo metadatos. (El `best_frame` opcional escribe JPGs SOLO a disco local para auditoría del operador — nunca al MQTT.)
- **No almacenar MACs crudas.** Hashear primero (con sal), siempre.
- **WiFi = solo probing.** Red = Ethernet.
- **Stack de HATs**: AI HAT+ (PCIe) y PoE HAT (G) ambos stackeados. Orden físico: AI HAT+ → Pi → Active Cooler → PoE HAT.
- **No hardcodear config.** Todo en YAML.
- **Siempre buffear localmente.** Conectividad puede fallar.

## Entorno

- Raspberry Pi OS Trixie 64-bit, Python 3.13
- Hailo SDK 4.23+ (`hailo_platform`), Picamera2 (rpicam-* CLI), OpenCV 4.10+ (contrib para ArUco/ChArUco)
- paho-mqtt 2.1+, SciPy 1.13+, sqlite3 (stdlib)

## Pipeline runtime — knobs

`src/main.py` orquestra capture → rectify → SGBM → detect (Hailo) → track → count → MQTT. La lista exhaustiva de knobs con defaults vive comentada en `config/config.example.yaml`; acá van solo los que tienen gotchas no obvios.

- `vision.num_disparities: auto` — deriva rango SGBM desde `mounting_height_m`. Override con int múltiplo de 16.
- `vision.sgbm.downscale: 4` — SGBM a resolución reducida (1=full, 4=quarter). 4 default (remueve speckle que infla head-height); 1 solo para diagnósticos.
- `vision.sgbm.wls.{enabled, lambda, sigma}` — WLS post-filter (opencv-contrib ximgproc): suaviza speckle + rellena agujeros usando el matcher derecho como confidence. Defaults `true / 4000 / 1.0`.
- `vision.depth_skip_stable_tracks` — cachea el depth_map cuando todos los tracks tienen height estable (≥`height_samples_threshold` samples, default **8**) Y todas las detecciones matchean tracks existentes Y no hay candidatos nuevos. SGBM (~50-70ms) domina el budget; el cache subió el prototipo de 8-12 FPS a 28-30. TTL `cache_ttl_seconds` (1s) acota staleness. Trade-off: el bbox puede moverse 10-20px entre frames (la mediana de `detection_history` lo absorbe). Default `enabled: true`. **El cache casi no engancha en un IN** (el track nace en el borde y bootstrapea altura JUSTO al cruzar la línea → SGBM fresh); sí en un OUT (track ya estable). Es inherente y solo cosmético — el cruce usa centroide 2D, no depth. Bajar el threshold 20→8 (A/B en las instalaciones 2026-05-30) sube el cache hit en cruces de ~55%→~79% acortando el bootstrap, sin tocar precisión de depth (sigue a downscale=4). **No subir `sgbm.downscale` a 8 para ganar FPS**: el head-height se desvía ~7% (>5% del budget de calibración) — rompe el rango de estatura adult/child sin beneficio al conteo.
- `vision.idle_throttle` — FPS adaptativo en escena vacía (default **OFF** fleet-wide; validado de extremo a extremo en las instalaciones 2026-06-09, ON ahí). Sin actividad sostenida `idle_after_seconds` (2s), el loop baja a `idle_fps` (10) y TODA la cadena baja con él (el productor de captura está backpressured al consumidor): CPU ~102%→~50%. Busy conservador: track no-estacionario (un track nuevo NUNCA clasifica estacionario), `ghost_count > 0` (a FPS bajo el IoU de adopción cae), detecciones > tracks estacionarios, o preview con subscribers. **Gotcha**: el `fps` de telemetría muestra ~10 en horas vacías — es la señal del throttle trabajando, NO una falla; no crear alertas de FPS bajo sin excluir este estado. El sleep va después del registro de latencia (no infla `frame_latencies_ms`).
- `--profile`, `--profile-every-n` (30), `--profile-slow-threshold-ms` (0=off) — diagnóstico de bottleneck por stage. `--profile` loguea `PROFILE frame=N cap/rect/detect/depth/track ... mode=fresh/cache/none`. Con slow-threshold>0 emite `PROFILE_SLOW` por cada frame que supere el umbral (caza FPS drops sporadicos sin inundar).
- `--no-mqtt` (CLI) — reemplaza MQTT con un no-op que loguea a stdout. Testing local sin AWS.
- `detection.{confidence_threshold 0.20, new_track_threshold 0.35, low_confidence_threshold 0.10}` — banda triple: <low descartado, [low,conf) re-asocia tracks existentes (ByteTrack-style), [conf,new) re-asocia + display, ≥new spawnea track nuevo.
- `detection.cluster_distance_px: 150` — mergea bboxes post-NMS por centroide. 150px en 1152×648 deja cabezas de 50-80px holgadas; dos personas adyacentes (~150-180px) NO se mergean.
- `detection.static_suppressor` — defense-in-depth contra clutter estructural (FPs persistentes en mismas celdas). Ventana en segundos reales (independiente del FPS). Exenta la counting zone (`exempt_counting_zone`).
- `vision.max_exposure_us: 16000` — shutter cap 16ms para limitar motion blur (~5cm a 3 m/s). Mismo cap en TODOS los setup tools.
- `vision.camera_sync.enabled` — sync de frames L/R por **software** (libcamera `SyncMode`, sin cableado): la izquierda (principal) es `Server` y la derecha el `Client` que la sigue. Las 2× IMX708 corren libres y su fase deriva (medido: ~88% de pares <5ms pero ~12% hasta ~19ms = un período de frame); con sync se alinean a **~20-400µs** (validado HW 2026-06-21). Mejora depth/estatura en sujetos en movimiento (el conteo usa centroide 2D, no se afecta). **Es feature de RUNTIME** (+ calibrate sweep + preview live, donde hay movimiento). NO se usa en focus/diagnose/capture_baseline (escena estática, no aporta). **Gotchas (los 3 importan, validados en HW — cada uno fue un callejón)**: (1) requiere config de **video** (`create_video_configuration`) — con still configuration el IPA de sync NO ajusta el frame timing (queda ~18ms); (2) requiere un **`FrameRate` fijo y ALCANZABLE** (NO un rango en `FrameDurationLimits` → tira `Sync algorithm enabled with variable framerate`, error por frame; NI `1e6/max_exposure_us`=62.5fps que pide más que el piso ~56fps → no converge). Solución: `apply_sync_framerate()` lee el piso real (`FrameDurationLimits[0]`) tras el start y setea `FrameRate=1e6/(piso·1.05)` (~53fps, exposure ~18.1ms ≈ sin tradeoff de blur); por eso `_build_camera_controls` para sync sólo pone `SyncMode`; (3) **converge-then-hold**: el IPA hace su ajuste inicial de fase SÓLO con GAPS en el dequeue — la captura continua del productor lo IMPIDE (queda ~18ms desde el frame 1). `_converge_sync()` (StereoCapture) / `converge_camera_sync()` (preview) corren una fase BLOQUEANTE de ~8-15s (bursts con pausas, corta al bajar de 1ms) ANTES del productor continuo; una vez convergido, el productor lo MANTIENE (holdea a ~20-40µs). Loguea `camera_sync_converged delta_us`. Cold-start con sync ~+10s (sigue < 90s). El Client (derecha) arranca primero (espera al Server). **Default ON fleet-wide** (`camera_sync_enabled` default-ea True; apagar con `enabled:false` per-site). La constructora de `StereoCapture` igual default-ea a False, así que los setup tools estáticos (focus/diagnose/baseline) que no leen el flag siguen sin sync.
- `vision.ae_lock.{initial_settle_seconds, resettle_seconds}` — timings del AE lock canónico (settle → lock provisional → re-settle on Comenzar → re-lock final). Subir en sites con luz fluctuante.
- `tracking.state_machine.confirm_frames: 2` — CANDIDATE→CONFIRMED. 2 balancea filtrar FPs efímeros vs responsividad; el detector fine-tuneado no necesita 3.
- `tracking.state_machine.reid_gate_px: 200` + `pending_velocity_decay: 0.85` — gate de re-id en PENDING; el decay reduce el predict del Kalman cerca de la última observación, así el gate cubre gaps de ~1s a velocidad de caminata.
- `tracking.state_machine.pending_grace_frames: 3` — frames iniciales de PENDING con predict full-velocity antes de aplicar el decay. Cubre gaps cortos del detector preservando la trayectoria.
- `tracking.ambiguous_match_ratio: 0.8` — ratio test estilo Lowe post-Hungarian: si el 2do mejor candidato está dentro del 80% del mejor, la match se rechaza por ambigua. Filtra cruces arbitrarios cuando dos detecciones y dos tracks están cerca.
- **Tracking point del counter = centroide del bbox.** Montaje cenital sobre el umbral de la puerta → el cruce ocurre cerca del nadir donde el paralaje es ~cero (el centroide es el foot-point efectivo). **La corrección de paralaje image-space (`foot_projection`) se retiró — NO re-agregarla**: comprimía la trayectoria hacia el principal point y rompía los INs en puerta central. La altura 3D de SGBM se sigue usando, pero solo para clasificar adult/child, no para la posición del cruce. Una corrección world-space queda como opción futura.

## Design philosophy del counter: rescue con guardrails

El counter pierde counts en dos modos: (1) **crosser perdido en la zona de la línea** (el detector lo dropea justo al atravesar); (2) **crosser parqueado adentro de la counting zone** (cruza con detección real, se pierde, el `pending_velocity_decay` frena el Kalman antes del borde → la salida nunca dispara). La elección de diseño es **rescue agresivo con guardrails**, no "solo contar lo observado directo"; el trade-off es explícito y tuneable per-site. **Detalle completo + runbook operacional en `docs/tracker_tuning.md`** (6 patrones síntoma→fix con comandos exactos).

**Tres capas de rescue, en cascada** (de la primera a la última que dispara):

1. **Ghost pool / ID adoption** (`EuclideanTracker._try_adopt_ghost`): un track LOST deja `track_id`+`meta` en `_ghosts` por `adoption_window_frames` (~30 ≈ 1.5s); un track nuevo con `IoU ≥ adoption_iou_min` (0.3) Y `dist ≤ adoption_max_dist_px` (100) lo adopta → el counter ve continuidad y emite natural en el exit. **Guardrail**: si `last_outside_pos` del ghost está a >`GHOST_OUTSIDE_INVALIDATE_PX` (150) del nuevo centroide, ese campo NO se hereda (un outside_pos alucinado produciría `had_outside_pos` espurio + cross artificial). El resto del meta sí se hereda.
2. **Decisive Kalman cross at exit** (`Counter._decisive_kalman_cross`): si el exit es pura extrapolación Kalman, se acepta el cruce solo si `disappeared ≤ MAX_KALMAN_CROSS_FRAMES` (15) Y desplazamiento real `≥ MIN_KALMAN_CROSS_DISPLACEMENT_PX` (30) Y `had_outside_pos`. **El gate inside-was-inside se mantiene estricto: NINGÚN cross registra en frames de predicción adentro de la counting zone** — la relajación es SOLO en la transición de exit (sin esto vuelve el sitter cuyo Kalman alucina un exit lateral; ver `test_kalman_exit_skipped_when_track_born_inside_counting_zone`).
3. **Death-emit-if-crossed** (`Counter._emit_on_death`): track que muere adentro con cruce registrado (`net≠0`) emite si pasa DOS guards: `had_outside_pos` (filtra sitters/clutter/re-id que nacen adentro) y `visit_range ≥ MIN_VISIT_RANGE_FOR_DEATH_EMIT` (80px). Diferido `death_emit_grace_frames` (= `adoption_window+2`) — si la capa 1 resucita el track dentro del grace, se cancela (no double-count).

**Guardrails que NO hay que re-romper** (cada uno fixeó un bug hallado en la validación en las instalaciones; los detalles + fechas viven en comentarios de código y tests nombrados):

- **Cross solo con detección real** (`disappeared==0`), nunca en frame de pura predicción Kalman — pero el cruce real ya registrado SÍ se emite aunque la salida sea por extrapolación. (Reemplazó un "freeze" del track que clavaba a los crossers reales perdidos mid-zone.)
- **Entry-fresca solo con detección real**: si el primer frame inside es Kalman (`is_real=False`), NO se inicia ciclo de visita (sin esto: zigzag + COUNT espurio; log `TRACKDBG entry_kalman_skipped`).
- **No U-turn cancellation**: un round-trip real cuenta 1 IN + 1 OUT. La cancelación previa se removió (cancelaba tráfico legítimo). Una "duda" sin segundo cruce completo produce un solo evento al salir igual.
- **Cancelación NETA dentro de la counting zone**: el balance neto de cruces por línea decide el signo al salir (±1 cuenta, 0 = fue-y-vino → no cuenta). Reemplaza "gana el último cruce". Gate one-way = sticky.
- **Keep-alive dentro de la counting zone** (`tracker.keepalive_counting_zone`): un PENDING cuya predicción cae dentro NO muere por timeout (extrapola con Kalman, no se congela) hasta re-matchear o salir. Cap `keepalive_max_frames` (600 ≈ 24s) garbage-collectea huérfanos. El preview esconde los fantasmas de larga ausencia (`disappeared > pending_max_frames`).

**Knobs para mover el balance "agresivo → conservador" per-site** (config-driven, sin redeploy):

| Knob | Default | Efecto al subirlo |
|---|---|---|
| `MIN_KALMAN_CROSS_DISPLACEMENT_PX` | 30 | Capa 2 más estricta. ∞ = no rescue de Kalman crosses. |
| `min_visit_range_for_death_emit` | 80 | Capa 3 más estricta. Bajarlo (~50) si `death_emit_count=0 + ratio>1.3` (guard rechaza crossers reales con poca observación). |
| `min_count_height_m` | 1.0 | Filtro anti-FP no-humanos por altura mediana; 1.0 (el valor evaluado, ahora default del template) filtra perros/objetos <1m sin perder personas de más de 1 m. 0.0 lo apaga. Track sin SGBM (None) NUNCA se filtra — ver limitación L5 en `docs/benchmark_results.md`. |
| `min_real_inside_frames` | 0 (off) | Frames con detección real inside antes de emitir. 2 filtra single-frame entries al borde (flicker + Kalman). |
| `height_confidence_gate` | 0.5 | Umbral de median(conf) para reportar el rango de estatura. NO afecta el conteo, solo `height_m`/`head_depth_m` (van NULL si cae debajo → `height_class()` los mapea a `unknown`). |
| `tracking.tracking_zone` | disabled | Filtro pre-tracker por polígono ("modo estricto"): descarta detecciones fuera del polígono ANTES del tracker. Modos: `polygon` manual, `frame_margin_px` (recomendado), `auto_margin_px`. Debe ser más amplia que counting_zone (preserva lead-in). El preview blurea lo de afuera. Para sites con clutter (perchero, mostrador, vidriera con tráfico exterior). |
| `adoption_window_frames` | 30 | ↑ = más adopción ID, ↓ = más fragmentación. 0 = sin ghost pool. |
| `adoption_iou_min` | 0.3 | ↑ = adopción más conservadora (anti ID-swap). |

Los tres primeros al máximo + adoption en 0 = modelo "puro observacional". Defaults = híbrido agresivo en sites flakey, conservador frente a sitters/clutter.

**Telemetría — árbol diagnóstico de 3 métricas** (todas reset diario):

- `track_stitching_ratio` = `unique_track_ids / total_counts`. Ideal ≈ 1.0. >1.3 = fragmentación de identidad.
- `ghost_adoption_count` = capa 1 (adopciones). `death_emit_count` = capa 3 (death-emits). Acumulativos del día.

| `stitching_ratio` | `adoption` | `death_emit` | Diagnóstico |
|---|---|---|---|
| ≈ 1.0 | 0 | 0 | Tracker perfecto |
| ≈ 1.0 | > 0 | 0 | Fragmentación rescatada por capa 1 |
| ≈ 1.0 | 0 | > 0 | Crossers rescatados por capa 3 |
| > 1.3 | 0 | 0 | 🚨 FRAGMENTACIÓN SIN RESCATE (alarma) |
| > 1.3 | > 0 | > 0 | Tracker flakey, dependés de ambas capas (recall del detector flojo) |

**Matriz de cobertura** (`docs/counter_test_matrix.md`): mapea las ~17 dimensiones que el counter+tracker bifurcan a los tests, con justificación de las celdas "structurally void". Mantener viva: agregar la celda al matrix al introducir una bifurcación nueva, antes de commitear.

## Pipeline del detector

YOLOv8n fine-tuneado para detección cenital de cabezas. Pipeline ONNX → HEF compilado para Hailo-8L.

- **Bench tooling**: `bench_detector.py` (modelos locales `.pt`/`.onnx`), `compare_detectors.py` (side-by-side anti-regresión v_actual→v_next), `analyze_eval_summary.py` (distribución conf + breakdown por site).
- **Validation set**: `capture_mjpeg.py` multi-site con motion-trigger + background sampling (filenames `_motion_`/`_bg_`).
- **Postproceso geométrico** post-NMS: cluster por centroide, containment filter (bbox chico contenido en grande), static suppressor (celdas hot).
- **Labeling**: local en **X-AnyLabeling** (no Roboflow). Convención: bbox cabeza+hombros, single-class `person` (ver `scripts/training/label_guide.md`). Sampleo via `sample_for_labeling.py` (estratificado) o `mine_active_learning.py` (informativo para v_next). Conversión a YOLO via `labelme_to_yolo.py`. Upload a Kaggle dataset privado vía CLI.
- **Training**: notebook `scripts/training/train_head_detector.ipynb` en Kaggle T4 (~20 min). v1→v2 con active learning subió mAP50 0.805→0.956 (deployed; eval contra val held-out 245 imgs / 174 cajas). La 2da ronda de AL (v3) bajó a 0.939 — sweet spot en v2.
- **Boundary fuerte**: compilación HEF (`hailomz compile`, calibration set 200 imgs) SOLO en WSL2/Linux x86 (Hailo no soporta Windows). Inferencia en la Pi: HEF en `/usr/src/people-counter/models/`, `detection.model_path` en config.

## Convenciones de tools de visión

- **Config reading**: `focus_assist`, `calibrate`, `preview`, `diagnose_depth`, `diagnose_calibration` leen `/etc/people-counter/config.yaml` para resolver `vision.resolution` (todos) y `vision.mounting_height_m` (focus_assist deriva el target distance). Pasar `--resolution` explícito solo en dev workstation sin config. `diagnose_calibration` además carga la calibración `.npz` y captura a la resolución de sus rectify maps.
- **Browser-driven**: nada de `input()` blocking. Pantalla "Comenzar", AudioContext unlocked on click, beeps cortos diferenciados (start/pose/captura/undo/fin) en lugar de TTS, reporte HTML auto-open al finalizar + cierre del servidor.
- **AE flags compartidos**: `--meter matrix|centre|spot` (default matrix), `--lock-ae`. Patrón canónico: settle 2s → lock provisional → re-settle 1.5s on click → re-lock final con la escena real.
- **Exposure cap uniforme**: `--max-exposure-us 16000` default en todos los setup tools (mismo cap que el runtime — freezea micro-vibración del bracket que rompe ArUco asimétricamente L/R). Pasar 0 para deshabilitar.
- **Dual-pass ChArUco**: `detect_charuco_dual_pass` intenta el frame original; si <8 corners reintenta con sharpen y se queda con el mejor. Aplicado en los live loops; el fit downstream (`calibrate_stereo`) sigue single-pass (no introducir sub-pixel noise en el solve).
- **CLI args homogéneos**: `--board-cols/--board-rows/--square-mm/--marker-mm` canónicos en todos los scripts (`calibrate.py` acepta `--columns/--rows/--square-length/--marker-length` como alias).
- **Wizard guardrails**: pre-calibration sanity gate (re-detección de pares capturados), coverage critical block, L/R asymmetric alert, `--resume` valida resolución, `reset --yes` para restart limpio.
- **Lens locking**: holder M12 sin set screw → fijado con esmalte de uñas transparente al seam barrel↔holder después del foco (touch-dry 15min, cura full 30-60min). Suficiente para PoC; para flota evaluar Trabasil AM3 + activador anaeróbico. Ver `docs/lab_calibration_guide.md`.
- **Interpretación reporte calibración**: RMS estéreo <0.5px es necesario pero no suficiente. **El verdict depende SOLO del ground-truth en centro** (cinta/láser). El **baseline es INFORMATIVO, no gate**: el baseline óptico (pupila de entrada del fisheye M12 + tolerancias de impresión del case) difiere legítimamente del nominal mecánico 140mm — medido repetidamente en ~146mm (143–147 en 3 cals). `--baseline-tol-mm` (default 15mm) es solo un bound de sanidad para cazar un solve degenerado, NO un QA de calidad (anclar un gate fino a 140 tiraba false-FAILs). El ratio borde/centro también es informativo, no gate.
- `debug/` está gitignoreado — para reportes, screenshots, logs de tests.

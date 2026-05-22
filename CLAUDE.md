# CLAUDE.md — People Counter Edge System

## Descripción general

Sistema de conteo de personas de bajo costo para locales comerciales. Visión estéreo + IA en el borde + detección pasiva de tráfico WiFi/BLE.

**Proyecto de producción real**: dispositivos operan desatendidos 12h/día, 363 días/año. Calidad de código, manejo de errores y resiliencia son críticos.

## Arquitectura

```
+---------------------------------------------+
|        Dispositivo edge (por local)          |
|  RPi5 4GB + Hailo-8L 13T + 2x IMX708         |
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

- Raspberry Pi 5 4GB + Active Cooler
- Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) — único HAT stackeado
- 2× Arducam IMX708 12MP HDR M12 120° HFOV (B0310) vía CSI — par estéreo, baseline 14cm
- Waveshare PoE HAT (H) 25.5W — alimentación, conectado por dupont (no stackeado)
- LED RGB 3mm common cathode — GPIO 17/18/27 + GND pin 14, resistores 150/100/100Ω
- SanDisk Extreme 64GB microSD

## Decisiones técnicas clave

### Pipeline de visión

- **Calibración estéreo**: ChArUco A3 (9×6 / 45mm / 33mm / DICT_4X4_100), modelo fisheye Kannala-Brandt (`cv2.fisheye.calibrate`). Lab universal (mount 2.0-3.5m): foco a 1.5m, calibración a 1.0/2.0/3.0m. Validar con `scripts/diagnose_depth.py` (error centro <5% a 2m, <10% a 3m). `detect_charuco_dual_pass` (en `src/vision/calibration.py`) intenta primero el frame original y, si quedó por debajo de 8 corners, retry con sharpen 3×3 — recovery transparente para feedback live y gates de captura del wizard. QC de ensamble del bracket: `scripts/diagnose_bracket.py` mide pitch/yaw/roll/offset entre L y R sin requerir calibración previa.
- **Óptica**: Arducam B0310 fisheye real, no rectilíneo. Focal pinhole-equivalente `f_px = 2050` a full-res 4608×2592. La fórmula `f = (W/2)/tan(HFOV/2)` NO aplica.
- **Sensor mode canónico**: 2304×1296 binned 2×2, full FOV, 16:9, hasta 56 fps. Runtime puede usar rescale lineal (ej. 1152×648 vía `scripts/rescale_calibration.py`) — K-B es resolución-independiente en intrínsecos angulares. **CRITICAL**: en TODO call site de picamera2 hay que pasar `raw={"size": CANONICAL_RAW_SIZE}` (importar desde `src.vision.capture`) para seleccionar Mode 1 (2304×1296, HFOV 120°); sin ese hint picamera2 elige Mode 0 (1536×864 cropeado, HFOV ~80°) que reduce la cobertura a la mitad. Overrideable per-device vía `vision.sensor_raw_size` del config.
- **Rectificación**: `cv2.fisheye.initUndistortRectifyMap` (balance=0.0) + `cv2.remap`.
- **Profundidad**: SGBM + matcher derecho + WLS filter. `vision.num_disparities: auto` deriva el rango desde `mounting_height_m`.
- **Detección**: YOLOv8n fine-tuneado (cenital), HEF para Hailo-8L. NMS on-chip. VStream API con scheduler ROUND_ROBIN. Modelo activo: `people-counter-detector`. Pipeline detallado en `scripts/training/README.md`.
- **Tracking**: euclidiano 3D (x, y, profundidad) con Kalman per-track + state machine corta (CANDIDATE → CONFIRMED → PENDING → LOST).
- **Conteo**: línea virtual + ROI rectangular. Track entra al ROI → cruza línea → sale del ROI = evento `direction='in'` o `'out'` (terminología canónica fleet-wide, schema y MQTT). Publicación inmediata vía MQTT.

### Captura WiFi/BLE

- **WiFi**: CYW43455 monitor mode vía nexmon. Captura probes en 2.4 + 5 GHz. **WiFi solo probing — red por Ethernet**. Monitor mode requiere `nexutil -m2` (radiotap en el firmware) — sin eso el netdev entrega frames Ethernet (DLT EN10MB) y scapy no ve 802.11; el capture loop re-parsea los bytes crudos como `RadioTap()`. nexutil se compila aparte (ver `setup_device.sh`). El pipeline corre como `User=pi`: `/dev/rfkill` necesita estar en grupo `netdev` (udev rule) para que el unblock funcione.
- **BLE**: bleak (D-Bus de BlueZ), escaneo pasivo.
- **Solo dispositivos "humanos"** (`wifi_ble.randomized_only`, default true): se cuentan solo MACs WiFi randomizadas (locally-administered bit 0x02) y BLE con `AddressType=random` (RPA iOS / aleatoria Android). Las MAC globales WiFi y los BLE `public` (OUI real) son infra/IoT fijo (APs-como-cliente, smart-TVs, beacons, parlantes) y se descartan antes de hashear. Apagable per-site. Nota: las rotaciones de RPA BLE de un mismo teléfono son todas `random` — eso lo resuelve el stitching, no este filtro.
- **Hashing**: SHA-256 truncado a 16 bytes. **Nunca MACs crudas**.
- **Dedup → hash groups con stitching** (`src/wifi_ble/dedup.py`): los hashes se asocian a un `group_id` por identidad del dispositivo. Cada call de `process_detection` aplica 4 reglas y joina al grupo más reciente que matchee:
  1. **Seqnum continuity (WiFi-only)** — el seqnum 802.11 (12 bits, del header `dot11.SC >> 4`) es contador del chip y tiende a ser continuo cross-MAC-rotation. Match: Δseqnum ≤ `max_delta` (default 100, considerando wrap mod 4096) + ΔRSSI ≤ 5dBm + Δt ≤ 30s. Defeated por Apple H1+ (iPhone 12+) que resetea seqnum on MAC change; sigue funcionando en Android.
  2. **Cross-protocol L2 (short window)** — WiFi MAC y BLE addr observados dentro de `cross_protocol_window_seconds` (default 2s) con ΔRSSI ≤ 5dBm = mismo dispositivo. Es el L2 histórico.
  3. **BLE anchoring (long window)** — durante la vida de un BLE RPA (~15min iOS), nuevas WiFi MACs con RSSI compatible se mergean al grupo del BLE existente. Cubre el caso "WiFi rota cada 2min, BLE cada 15min" donde la regla 2 (2s) no alcanza.
  4. **Fingerprint continuity (mismo protocolo)** — fingerprint estable (orden de IEs + HT/VHT/HE caps en WiFi; company ID + subtipos Continuity de Apple + service UUIDs + TX power en BLE; ver `src/wifi_ble/fingerprint.py`) que sobrevive la rotación de MAC/RPA. Mismo fingerprint + RSSI compatible + ventana = mismo aparato. Cubre lo que el seqnum NO agarra: **Apple H1+ resetea el seqnum al rotar la MAC** pero el fingerprint es estable. Además actúa de **filtro duro** en la regla 1 (seqnums que coinciden por azar pero con fingerprint distinto = dispositivos distintos). Caveat: dos devices idénticos co-presentes pueden mergearse (leve subconteo); el gate de RSSI lo acota.
  Los counts publicados (`passersby`, `shoppers`) son `DISTINCT group_id`, no distinct hashes. L3 inter-cámara queda reservado para deploys multi-cam (no aplica al PoC con 1 device/sucursal).
- **Privacy del stitching**: el seqnum y los timestamps quedan SOLO en `wifi_ble_dedup.sqlite` local (rotado diario via `reset_daily`). El MQTT publish sigue mandando counts agregados, nunca hashes ni seqnums ni MACs crudas.
- **Stitching canary**: `dedup.get_stitching_ratio()` = `groups / hashes` del día. 1.0 = ningún stitch (cada hash es su propio "visitor"), 0.5 = mitad de los hashes se mergearon. Va en el payload de telemetry (`wifi_ble_stitching_ratio`) y la columna homónima de la tabla `telemetry` — canary para detectar si la flota corre con OS que defeatean las reglas.
- **Publish**: el `WifiBlePublisher` (`src/wifi_ble/publisher.py`) emite un único summary por ventana (default 15 min) con `{passersby, shoppers}` post-stitching. Privacy stronger + Lambda más simple que dedup individual.

### Comunicación

- **MQTT** 3.1.1 sobre AWS IoT Core, X.509, QoS 1.
- Eventos de conteo en tiempo real, resúmenes WiFi/BLE cada 15min, telemetría cada 5min.
- **Buffer SQLite local**: replay al reconectar, marca enviado solo tras PUBACK.

### Status LED

- 8 estados (apagado / rojo / amarillo / amarillo blink / verde blink / verde / azul / azul blink) en cascada worst-first: HW > pipeline > internet > cloud > OK.
- Health probes: CPU/Hailo temp, disco, calibración, captura/inferencia, watchdog (`last_loop_ts` <5s), internet TCP a 1.1.1.1:53, MQTT connected.
- Monitor en thread separado (probes blocking no estresan el hot path). Fail-safe: sin GPIO → no-op + log INFO.
- LEDs onboard ACT/power/Ethernet/audio apagados via dtparam — el RGB externo es la única fuente visual.

### Cloud (AWS)

- **PoC actual (1 device, deployado)**: IoT Core (broker + 3 Topic Rules SQL) → Lambda `persist_event` (`src/cloud/persist_event.py`, fuera de VPC, IAM auth a RDS via `rds.generate_db_auth_token`) → RDS Postgres 16 (db.t4g.micro, single-AZ, IAM auth + `rds.force_ssl=1`, `AutoMinorVersionUpgrade=true`). Grafana 13 en ECS Fargate detrás de ALB con ACM cert custom (image desde ECR, custom domain `grafana.tfg.gasparri.com.ar`) lee el mismo RDS como datasource. Orquestado por CloudFormation (`infra/cloudformation/people-counter.yaml`) + `infra/deploy.ps1` (5 fases con `-StartFromPhase`). El cert ACM se crea fuera de CFN para que el deploy no bloquee esperando validación DNS — el ARN entra al stack como parámetro. Schema en `infra/sql/bootstrap.sql`. Lambda dedup L3 no aplica con 1 device/sucursal (el stitching local del device cubre el caso).
- **Producción (rollout de flota)**: RDS single-AZ → Multi-AZ ($26/mo en vez de $13). Considerar Amazon Managed Grafana (SSO + IAM-integrated, $9/user) en vez de OSS si se integra a auth corporativa. Migrar DNS a Route53 delegated subdomain para que CFN gestione DNS records (ALIAS al ALB) y el deploy sea 100% sin pause — hoy hay 2 steps manuales: agregar CNAMEs de validación ACM (permanentes) y el CNAME final al ALB en el DNS provider externo.
- **Costos PoC ~$35/mo**: RDS db.t4g.micro $13 + Fargate task 0.5vCPU/1GB $18 + ALB $16 + ACM cert free + IoT/Lambda/SecretsManager/CloudWatch <$2. Al sumar 2+ services en el futuro (sales API, auth), se puede compartir el ALB via listener rules y amortizar el costo fijo del LB.
- **Multi-cam por sucursal**: cuando se agregue 2+ cámaras por local, reintroducir L3 (Lambda + DynamoDB de hashes). El stitching local del device cubre monocam pero no inter-cam.

## Convenciones de código

- **Lenguaje**: Python 3.13 (RPi OS Trixie). Black 88 chars, Ruff, type hints obligatorios.
- **Logging**: módulo `logging` JSON estructurado. DEBUG dev / INFO prod.
- **Config**: archivo único per-device.
  - `/etc/people-counter/config.yaml` (per-device): fuente única de verdad en runtime. `load_config()` lee SOLO este archivo, sin merge con el example. Tiene que contener todas las keys requeridas (validadas en `_validate`).
  - `config/config.example.yaml` (en repo): TEMPLATE documentado de la flota. NO se mergea en runtime — sirve para provisionar un device nuevo (operator copia → `/etc/people-counter/config.yaml` y edita) y para auditar qué keys son válidas. Cambios fleet-wide se shippean editando el template + redeployando el config a cada device. Una whitelist chica de keys (`CLOUD_OVERRIDABLE`: operating_hours + counting_enabled) puede pushearse vía AWS IoT Device Shadow sin restart.
  - **Hardware-agnostic**: los parámetros que dependen del hardware/setup del device (sensor, lens, bracket, board ChArUco, AE timings) están consolidados en `src/config/hardware.py` (dataclass `HardwareParams` + `load_hardware_params()`). Cambiar de sensor / bracket / lens / board se hace editando keys del config; ningún script tiene constantes hardware hardcodeadas. Los setup tools (focus_assist, calibrate, preview, roi_picker, diagnose_bracket, diagnose_depth) leen `HardwareParams` al startup; el runtime también plumb-ea los mismos valores a `StereoCapture`.
- **Secrets**: certificados X.509 en `/etc/people-counter/certs/`. **Nunca commitear**.
- **Tests**: pytest, estructura espejo de src.
- **No usar clases salvo que haya estado.** Tracker, MQTTClient justifican clases. Resto = funciones.
- **Todo I/O wrapeado** con manejo de errores.

## Estructura del directorio

```
people-counter/
├── CLAUDE.md, README.md, pyproject.toml, .gitignore
├── src/
│   ├── vision/         <- capture, calibration, depth, detect, static_suppressor, world_coords, best_frame
│   ├── tracking/       <- tracker (Kalman + state machine), counter (ROI + line crossings)
│   ├── wifi_ble/       <- wifi_probe, ble_scan, fingerprint, hasher, dedup, publisher
│   ├── mqtt/           <- client (AWS IoT), buffer (SQLite outbox)
│   ├── cloud/          <- persist_event Lambda (IoT Rules → Postgres)
│   ├── status/         <- led, health, monitor (background thread)
│   ├── config/         <- loader (strict, lee solo /etc/people-counter/config.yaml) + hardware (HardwareParams dataclass)
│   └── main.py         <- orquestador del pipeline
├── tests/              <- 765 tests, estructura espejo de src
├── scripts/
│   ├── calibrate.py    <- CLI con wizard end-to-end (browser-driven, ChArUco, ground-truth check)
│   ├── focus_assist.py <- asistente de foco guiado (browser-driven)
│   ├── diagnose_depth.py <- valida calibración (5 zonas, error centro <5% a 2m)
│   ├── diagnose_bracket.py <- QC de ensamble del bracket (pitch/yaw/roll/offset L↔R sin calib previa)
│   ├── preview.py      <- preview live MJPEG L|R (browser-driven)
│   ├── download_model.py, capture_baseline_frames.py, rescale_calibration.py
│   ├── provision.py    <- create/deploy/harvest/reprovision/list (disaster recovery) + seed sites/devices en RDS (psycopg+boto3)
│   ├── reset_dedup.py  <- reset diario del dedup (config-aware, lo llama people-counter-reset.service)
│   ├── verify_hardware.py, setup_device.sh
│   └── training/       <- train_head_detector.ipynb (Kaggle T4, descarga directo de Roboflow),
│                          bench_detector.py, bench_roboflow_api.py, capture_mjpeg.py,
│                          record_clips.py, sample_for_roboflow.py, sample_for_calib.py,
│                          polys_to_bboxes.py, eval_yolo.py
├── training_data/      <- gitignoreado salvo README + sites.yaml.example. Workspace local:
│                          sites.yaml inline (matrices + IPs), captures rectificadas, manifest
│                          de los frames subidos a Roboflow.
├── calibration/        <- board ChArUco A3 PDF (calib.io)
├── infra/cloudformation/ <- people-counter.yaml (stack completo)
├── docs/               <- setup-guide, lab-calibration-guide, pilot-operator-guide
└── config/             <- config.example.yaml + people-counter.service (systemd)
```

## Plan de sprints

| Sprint | Foco | Estado |
|--------|------|--------|
| S3 | PoC visión | **DONE** — capture (picamera2), detect (Hailo) |
| S4 | Calibración | **DONE** — fisheye K-B, board en `calibration/`, diagnose_depth |
| S5 | Detección | **DONE** — YOLOv8n single-class cenital, HEF compilado, deployable. Modelo activo: `people-counter-detector` (multi-site, ~945 imgs, labeling con Smart Polygon click-por-imagen + hard negatives explícitos) |
| S6 | Tracking + counting | **DONE** — tracker Kalman + counter ROI/línea (E2E validado) |
| S7 | WiFi/BLE | **DONE** — nexmon + nexutil/radiotap, hopping ponderado, bleak, filtro de humanos (randomized), hashing + dedup 4 reglas (incl. fingerprint) |
| S8 | MQTT | **DONE** — IoT Core + buffer SQLite + replay |
| S9 | Cloud | **DONE** — CloudFormation + Lambda persist_event + EC2 Postgres/Grafana |
| S10 | Integración | **DONE** — pipeline E2E en RPi5 |
| S11 | Piloto | PENDIENTE — deploy 3 locales |
| S12 | Estabilización | PENDIENTE — post-piloto |

## Reglas duras

- **No transmitir video/imágenes.** Solo metadatos.
- **No almacenar MACs crudas.** Hashear primero, siempre.
- **WiFi = solo probing.** Red = Ethernet.
- **Stack de HATs**: AI HAT+ es el único stackeado. PoE HAT por dupont.
- **No hardcodear config.** Todo en YAML.
- **Siempre buffear localmente.** Conectividad puede fallar.

## Entorno

- Raspberry Pi OS Trixie 64-bit, Python 3.13
- Hailo SDK 4.23+ (`hailo_platform`)
- Picamera2 (rpicam-* CLI tools)
- OpenCV 4.10+ (contrib para ArUco/ChArUco)
- paho-mqtt 2.1+, SciPy 1.13+, sqlite3 (stdlib)

## Pipeline runtime — knobs

`src/main.py` orquestra capture → rectify → SGBM → detect (Hailo) → track → count → MQTT.

- `vision.num_disparities: auto` — deriva rango SGBM desde `mounting_height_m`. Override con int múltiplo de 16.
- `vision.sgbm.downscale: 4` — SGBM a resolución reducida (1=full, 2=half, 4=quarter), upscale del disparity post-match. 4 default (~4× costo de 8 pero remueve speckle que infla head-height); 1 solo para diagnósticos.
- `--no-mqtt` (CLI) — reemplaza MQTT con no-op que loguea a stdout. Para testing local sin AWS.
- `detection.confidence_threshold` (0.20) / `new_track_threshold` (0.35) / `low_confidence_threshold` (0.10) — banda triple del detector: <low descartado, [low, conf) re-asocia tracks existentes (ByteTrack-style), [conf, new_track) re-asocia + display, ≥new_track spawnea tracks nuevos. Tuneado para `people-counter-detector` (YOLOv8n fine-tuneado).
- `detection.cluster_distance_px: 150` — mergea bboxes post-NMS por centroide. 150px en 1152×648 deja cabezas de 50-80px holgadas; dos personas adyacentes (~150-180px) NO se mergean.
- `detection.static_suppressor` — defense-in-depth contra clutter estructural (FPs persistentes en mismas celdas). Ventana medida en segundos reales con timestamps internos (independiente del FPS instantáneo). Configurable cell/window/threshold/min_samples.
- `vision.max_exposure_us: 16000` — shutter cap 16ms para reducir motion blur a ~5cm a 3 m/s. AE compensa con AnalogueGain. Default 16000us si la key falta del config. Mismo cap en TODOS los setup tools (`focus_assist`, `calibrate`, `preview`, `diagnose_depth`, `diagnose_bracket`).
- `vision.ae_lock.{initial_settle_seconds, resettle_seconds}` — timings del AE lock pattern canónico, compartido por todos los setup tools browser-driven (settle → lock provisional → re-settle on Comenzar → re-lock final). Subir en sites con luz fluctuante donde AE necesita más tiempo para converger.
- `vision.sgbm.wls.{enabled, lambda, sigma}` — WLS post-filter del disparity (opencv-contrib ximgproc). Suaviza el speckle de SGBM y rellena agujeros del WTA usando el matcher derecho como confidence map. Defaults `enabled: true, lambda: 4000, sigma: 1.0`. Apagar solo para comparar contra disparity raw.
- `vision.charuco.*` — board ChArUco + dual-pass detection. Si cambia el board de calibración (kit distinto, tamaño distinto), se edita acá y los setup tools lo recogen.
- `tracking.state_machine.confirm_frames: 2` — frames hasta promover CANDIDATE→CONFIRMED. 2 balancea filtrar FPs efímeros con responsividad para pasadas rápidas; el detector fine-tuneado es estable enough para no necesitar 3.
- `tracking.state_machine.reid_gate_px: 200` — gate de distancia para re-id en PENDING. Pareado con `pending_velocity_decay: 0.85` que reduce el predict del Kalman cerca de la última observación, así el gate cubre gaps de ~1s a velocidad de caminata sin necesidad de ser más ancho.
- `tracking.state_machine.pending_grace_frames: 3` — frames iniciales de PENDING en los que el predict del Kalman se mantiene full velocity antes de aplicar `pending_velocity_decay`. Cubre gaps cortos del detector (1-3 frames) preservando la trayectoria; después de la gracia el decay arranca para evitar drift del predictor.
- `tracking.ambiguous_match_ratio: 0.8` — ratio test estilo Lowe post-Hungarian. Si el segundo mejor candidato de una asignación está dentro del 80% del mejor, la match se rechaza por ambigua. Filtra el caso "dos detecciones cerca, dos tracks cerca, Hungarian arma un cruce arbitrario".
- **Tracking point del counter = centroide del bbox.** El montaje es cenital sobre el umbral de la puerta a medir, así que el cruce ocurre cerca del nadir donde el paralaje es ~cero (cabeza y pies proyectan al mismo pixel) — el centroide es el foot-point efectivo. La corrección de paralaje image-space que existía (`counter.foot_projection_enabled` + `world_coords.project_to_floor`) se retiró: comprimía la trayectoria hacia el principal point y rompía los INs en puerta central, sin aportar en la zona del cruce. La altura 3D de SGBM se sigue usando, pero solo para clasificar adult/child (`vision/depth.py`), no para la posición del cruce. Una corrección de paralaje en world-space (counter en mm sobre el plano del piso) queda como opción futura — bajo ROI para geometría de puerta central.
- **Conteo = entrar al ROI → cruzar la línea → SALIR del ROI** (semántica de gate, en `Counter._process_track`). No hay "salida sintética" por muerte del track dentro del ROI: una persona que cruza pero se queda parada/sentada/dudando en el ROI NO cuenta (evita el FP de lingering en el umbral). No hay U-turn cancellation: un round-trip real (entrar+cruzar+salir, luego re-entrar+cruzar+salir por el otro lado) cuenta 1 IN + 1 OUT — la cancelación previa se removió porque cancelaba tráfico legítimo, no solo dudas en la puerta. Una "duda" sin segundo cruce completo produce un único evento al salir, así que igual no se sobre-cuenta.
- **Cancelación neta dentro del ROI**: durante una visita al ROI el counter acumula un balance NETO de cruces por línea (cada cruce in-segment con label hacia un lado suma, hacia el opuesto resta). Al salir emite según el SIGNO del neto: ±1 = sentido contado, 0 = la persona "fue y vino" (cruzó y re-cruzó dentro del ROI) y NO cuenta. Reemplaza el viejo "gana el último cruce". Gate one-way = sticky (el cruce de vuelta sin label no resta).
- **Keep-alive del tracker dentro del ROI**: `EuclideanTracker.keepalive_roi` (lo setea `main` desde `counter.roi`) — un track PENDING cuya última posición predicha cae dentro del ROI NO muere por timeout (`pending_max_frames`/`max_disappeared`); se mantiene vivo hasta re-matchear o hasta que su predicción Kalman salga del ROI. Cubre "cruzó la línea y se quedó adentro mirando algo, el detector lo pierde, después sale" — sin keep-alive el track moría adentro y el cruce no se contaba. Complementa la exención del `static_suppressor` (que ya no suprime detecciones dentro del ROI vía `exempt_roi`). Tres salvaguardas para que no genere artefactos: (1) **freeze** — pasada la `pending_grace_frames`, un track kept-alive se congela (no se empuja la predicción Kalman) para que un track parado con velocidad residual no drift-cruce la línea y salga solo del ROI (doble conteo); la grace preserva la extrapolación para crossers rápidos dropeados 1-3 frames. (2) **cap `keepalive_max_frames`** (default 600 ≈ 24s) — misses consecutivos máximos; garbage-collectea fantasmas huérfanos (re-id falló, persona ya no está) en vez de acumularlos. Como `disappeared` se resetea con cualquier hit, el lingering real nunca llega al cap. (3) el preview esconde los fantasmas congelados (PENDING con `disappeared > pending_max_frames`) además del clutter estático. El historial `positions` se capa a 512 para que un track inmortal no crezca sin límite.

## Pipeline del detector

YOLOv8n fine-tuneado para detección cenital de cabezas. Pipeline ONNX → HEF compilado para Hailo-8L.

### Decisiones del pipeline

- **Bench tooling**:
  - `bench_detector.py` para modelos locales `.pt`/`.onnx`
  - `bench_roboflow_api.py` para triage de modelos publicados en Roboflow Universe via REST
- **Capturador de validation set**: `capture_mjpeg.py` multi-site con motion-trigger + background sampling. Filenames `_motion_` / `_bg_`.
- **Postproceso geométrico** post-NMS: cluster por centroide (`cluster_distance_px`), containment filter (bbox chico contenido en otro grande), static suppressor (celdas hot).
- **Compilación HEF**: `hailomz compile` en WSL2/Docker (x86 only, Hailo no soporta Windows nativo). Calibration set = 200 imgs representativas.
- **Training**: notebook `scripts/training/train_head_detector.ipynb` en Kaggle T4 (~20 min). Iteración: actualizar URL Roboflow + name del run.

### Boundary fuerte

- **Compilación**: WSL2/Linux x86 only.
- **Inferencia**: en la Pi. HEF en `/usr/src/people-counter/models/`, `detection.model_path` en config.

## Convenciones de tools de visión

- **Config reading**: `focus_assist`, `calibrate`, `preview`, `diagnose_depth` leen `/etc/people-counter/config.yaml` para resolver `vision.resolution` (todos) y `vision.mounting_height_m` (focus_assist deriva el target distance de ahí). Pasar `--resolution` explícito solo en dev workstation sin config per-device. `diagnose_bracket` corre sin config porque hace QC de ensamble pre-calibración.
- **Browser-driven**: nada de `input()` blocking. Pantalla "Comenzar", AudioContext unlocked on click, beeps cortos diferenciados (start / pose / captura / undo / fin) en lugar de TTS, reporte HTML auto-open al finalizar + cierre del servidor.
- **AE flags compartidos**: `--meter matrix|centre|spot` (default matrix; centre/spot para periferia brillante), `--lock-ae` (uniforme en todos los setup tools). Patrón canónico: settle 2s → lock provisional → re-settle 1.5s on click → re-lock final con la escena real de medición.
- **Exposure cap uniforme**: `--max-exposure-us 16000` default en todos los setup tools, mismo cap que el runtime — freezea micro-vibración del bracket que rompe ArUco asimétricamente entre L/R. Pasar 0 para deshabilitar.
- **Dual-pass ChArUco**: `detect_charuco_dual_pass` (en `src/vision/calibration.py`) intenta primero el frame original; si quedó <8 corners reintenta con sharpen y se queda con el mejor. Aplicado en los live loops de calibrate, focus_assist y diagnose_bracket. El fit downstream (`_detect_all_pairs` → `calibrate_stereo`) sigue single-pass para no introducir sub-pixel noise en el solve.
- **CLI args homogéneos**: `--board-cols/--board-rows/--square-mm/--marker-mm` son los nombres canónicos en todos los scripts. `calibrate.py` también acepta `--columns/--rows/--square-length/--marker-length` como alias para back-compat con docs viejas.
- **Wizard guardrails**: pre-calibration sanity gate (re-detección de pares capturados), coverage critical block (banda/grupo faltante), L/R asymmetric alert, `--resume` valida resolución, `reset --yes` para restart limpio.
- **Lens locking**: holder M12 sin set screw → fijado con esmalte de uñas transparente aplicado al seam barrel↔holder después del foco (touch-dry 15min, cura full 30-60min). Llave dedicada al barrel durante el foco, se retira antes de pintar. Suficiente para PoC; para producción/flota evaluar Trabasil AM3 + activador anaeróbico. Ver `docs/lab-calibration-guide.md`.
- **Interpretación reporte calibración**: RMS estéreo <0.5px es necesario pero no suficiente. **El verdict depende del ground-truth en centro** (cinta/láser). Baseline estimada debe caer ±1-2mm del diseño 140mm con 20 poses diversas; ratio borde/centro es informativo, no gate.
- `debug/` está gitignoreado — para reportes, screenshots, logs de tests.

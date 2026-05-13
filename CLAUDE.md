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
+---------------------------------------------+
|              AWS Cloud (PoC, 1 device)       |
|  IoT Core  ──► Lambda persist_event ──┐      |
|  (3 rules)                            ▼      |
|                              EC2 t3.micro    |
|                              ├ Postgres 16   |
|                              └ Grafana OSS   |
|                                              |
|  S3 bucket ◄── pg_dump diario (cron)         |
+---------------------------------------------+
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
- **Conteo**: línea virtual + ROI rectangular. Track entra al ROI → cruza línea → sale del ROI = evento ingress/egress. Publicación inmediata vía MQTT.

### Captura WiFi/BLE

- **WiFi**: CYW43455 monitor mode vía nexmon. Captura probes en 2.4 + 5 GHz. **WiFi solo probing — red por Ethernet**.
- **BLE**: bleak (D-Bus de BlueZ), escaneo pasivo.
- **Hashing**: SHA-256 truncado a 16 bytes. **Nunca MACs crudas**.
- **Dedup**: L1 intra-protocolo (SQLite por día), L2 cross-protocolo (ventana 2s + ΔRSSI ≤5dBm) — todo local en el device. L3 inter-cámara queda reservado para deploys multi-cam future work (no aplica al PoC con 1 device/sucursal).
- **Publish**: en vez de mandar hashes, el `WifiBlePublisher` (`src/wifi_ble/publisher.py`) emite un único summary por ventana (default 15 min) con `{passersby, shoppers}` post-L2 dedup. Privacy stronger + Lambda más simple que dedup individual.

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

- **PoC (1 device)**: IoT Core (broker + 3 rules SQL) → Lambda `persist_event` (`src/cloud/persist_event.py`) → Postgres 16 en EC2 t3.micro (free tier). Grafana OSS en la misma EC2 lee el Postgres. pg_dump diario a S3. **Sin Timestream** (no está en AWS Free Plan). **Sin Lambda dedup L3** (innecesaria con 1 device/sucursal).
- **Multi-device futuro**: cuando se agregue 2+ cámaras por sucursal, reintroducir L3 (Lambda + DynamoDB de hashes) y separar Postgres a RDS si el volumen lo exige.

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
│   ├── wifi_ble/       <- wifi_probe, ble_scan, hasher, dedup
│   ├── mqtt/           <- client (AWS IoT), buffer (SQLite outbox)
│   ├── cloud/          <- persist_event Lambda (IoT Rules → Postgres)
│   ├── status/         <- led, health, monitor (background thread)
│   ├── config/         <- loader (strict, lee solo /etc/people-counter/config.yaml) + hardware (HardwareParams dataclass)
│   └── main.py         <- orquestador del pipeline
├── tests/              <- 721 tests, estructura espejo de src
├── scripts/
│   ├── calibrate.py    <- CLI con wizard end-to-end (browser-driven, ChArUco, ground-truth check)
│   ├── focus_assist.py <- asistente de foco guiado (browser-driven)
│   ├── diagnose_depth.py <- valida calibración (5 zonas, error centro <5% a 2m)
│   ├── diagnose_bracket.py <- QC de ensamble del bracket (pitch/yaw/roll/offset L↔R sin calib previa)
│   ├── preview.py      <- preview live MJPEG L|R (browser-driven)
│   ├── download_model.py, capture_baseline_frames.py, rescale_calibration.py
│   ├── provision.py    <- create/deploy/harvest/reprovision/list (disaster recovery)
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
| S7 | WiFi/BLE | **DONE** — nexmon + bleak, hashing + dedup L1/L2 |
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
- `detection.confidence_threshold` (0.30) / `new_track_threshold` (0.50) / `low_confidence_threshold` (0.15) — banda triple del detector: <low descartado, [low, conf) re-asocia tracks existentes (ByteTrack-style), [conf, new_track) re-asocia + display, ≥new_track spawnea tracks nuevos. Tuneado para `people-counter-detector` (YOLOv8n fine-tuneado).
- `detection.cluster_distance_px: 120` — mergea bboxes post-NMS por centroide. 120px en 1152×648 deja cabezas de 50-80px holgadas; dos personas adyacentes (~130-150px) NO se mergean.
- `detection.static_suppressor` — defense-in-depth contra clutter estructural (FPs persistentes en mismas celdas). Ventana medida en segundos reales con timestamps internos (independiente del FPS instantáneo). Configurable cell/window/threshold/min_samples.
- `vision.max_exposure_us: 16000` — shutter cap 16ms para reducir motion blur a ~5cm a 3 m/s. AE compensa con AnalogueGain. Default 16000us si la key falta del config. Mismo cap en TODOS los setup tools (`focus_assist`, `calibrate`, `preview`, `diagnose_depth`, `diagnose_bracket`).
- `vision.ae_lock.{initial_settle_seconds, resettle_seconds}` — timings del AE lock pattern canónico, compartido por todos los setup tools browser-driven (settle → lock provisional → re-settle on Comenzar → re-lock final). Subir en sites con luz fluctuante donde AE necesita más tiempo para converger.
- `vision.charuco.*` — board ChArUco + dual-pass detection. Si cambia el board de calibración (kit distinto, tamaño distinto), se edita acá y los setup tools lo recogen.
- `tracking.state_machine.confirm_frames: 3` — frames hasta promover CANDIDATE→CONFIRMED. 3 filtra FPs efímeros sobre el detector fine-tuneado. Bajar a 1 para detectores con dropeos frame-a-frame.
- `tracking.state_machine.reid_gate_px: 180` — gate de distancia para re-id en PENDING. Pareado con `pending_velocity_decay: 0.5` que congela el predict del Kalman cerca de la última observación, así el gate cubre gaps de ~1s a velocidad de caminata sin necesidad de ser más ancho.
- `counter.foot_projection_enabled` — proyección parallax-corrected del foot pixel. Default off; activar solo después de validar calibración con diagnose_depth.

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

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
|              AWS Cloud                       |
|  IoT Core → Timestream (series temporales)   |
|          → Lambda (WiFi/BLE dedup)           |
|          → DynamoDB (hashes dedup)           |
|          → API Gateway → QuickSight          |
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

- **Calibración estéreo**: ChArUco A3 (9×6 / 45mm / 33mm / DICT_4X4_100), modelo fisheye Kannala-Brandt (`cv2.fisheye.calibrate`). Lab universal (mount 2.0-3.5m): foco a 1.5m, calibración a 1.0/2.0/3.0m. Validar con `scripts/diagnose_depth.py` (error centro <5% a 2m, <10% a 3m).
- **Óptica**: Arducam B0310 fisheye real, no rectilíneo. Focal pinhole-equivalente `f_px = 2050` a full-res 4608×2592. La fórmula `f = (W/2)/tan(HFOV/2)` NO aplica.
- **Sensor mode canónico**: 2304×1296 binned 2×2, full FOV, 16:9, hasta 56 fps. Runtime puede usar rescale lineal (ej. 1152×648 vía `scripts/rescale_calibration.py`) — K-B es resolución-independiente en intrínsecos angulares.
- **Rectificación**: `cv2.fisheye.initUndistortRectifyMap` (balance=0.0) + `cv2.remap`.
- **Profundidad**: SGBM + matcher derecho + WLS filter. `vision.num_disparities: auto` deriva el rango desde `mounting_height_m`.
- **Detección**: YOLOv8n fine-tuneado (cenital), HEF para Hailo-8L. NMS on-chip. VStream API con scheduler ROUND_ROBIN. Modelo activo: `people-counter-detector`. Pipeline detallado en `scripts/training/README.md`.
- **Tracking**: euclidiano 3D (x, y, profundidad) con Kalman per-track + state machine corta (CANDIDATE → CONFIRMED → PENDING → LOST).
- **Conteo**: línea virtual + ROI rectangular. Track entra al ROI → cruza línea → sale del ROI = evento ingress/egress. Publicación inmediata vía MQTT.

### Captura WiFi/BLE

- **WiFi**: CYW43455 monitor mode vía nexmon. Captura probes en 2.4 + 5 GHz. **WiFi solo probing — red por Ethernet**.
- **BLE**: bleak (D-Bus de BlueZ), escaneo pasivo.
- **Hashing**: SHA-256 truncado a 16 bytes. **Nunca MACs crudas**.
- **Dedup**: L1 intra-protocolo (SQLite por día), L2 cross-protocolo (ventana 2s + ΔRSSI ≤5dBm), L3 inter-cámara (Lambda + DynamoDB).

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

- IoT Core (broker + rules) → Timestream (conteo) + Lambda (dedup L3) + DynamoDB (hashes) + API Gateway → QuickSight.

## Convenciones de código

- **Lenguaje**: Python 3.13 (RPi OS Trixie). Black 88 chars, Ruff, type hints obligatorios.
- **Logging**: módulo `logging` JSON estructurado. DEBUG dev / INFO prod.
- **Config**: archivo único per-device.
  - `config/config.example.yaml` (en repo): defaults canónicos de la flota. `load_config()` lo lee como base y deep-mergea encima `/etc/people-counter/config.yaml` (per-device override).
  - **No hay separación fleet vs site**. Cambios fleet-wide se hacen en el example o se pushean por shadow (RUNTIME_SAFE_KEYS en `src/config/loader.py`).
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
│   ├── cloud/          <- lambda_dedup (L3 inter-cam)
│   ├── status/         <- led, health, monitor (background thread)
│   ├── config/         <- loader (deep-merge example + per-device)
│   └── main.py         <- orquestador del pipeline
├── tests/              <- 716 tests, estructura espejo de src
├── scripts/
│   ├── calibrate.py    <- CLI con wizard end-to-end (browser-driven, ChArUco, ground-truth check)
│   ├── focus_assist.py <- asistente de foco guiado (browser-driven)
│   ├── diagnose_depth.py <- valida calibración (5 zonas, error centro <5% a 2m)
│   ├── preview.py      <- preview live MJPEG L|R (browser-driven)
│   ├── download_model.py, capture_baseline_frames.py, rescale_calibration.py
│   ├── provision.py    <- create/deploy/harvest/reprovision/list (disaster recovery)
│   ├── verify_hardware.py, setup_device.sh
│   └── training/       <- train_head_detector.ipynb, download_roboflow.py, bench_detector.py,
│                          bench_roboflow_api.py, capture_mjpeg.py, sample_for_roboflow.py,
│                          polys_to_bboxes.py, eval_yolo.py
├── dataset/            <- gitignoreado salvo README. Datasets descargados de Roboflow.
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
| S5 | Detección | **DONE** — YOLOv8n single-class cenital, HEF compilado, deployable. Modelo activo: `people-counter-detector` (multi-site, ~945 imgs, hard negatives explícitos vía SAM3 pre-label) |
| S6 | Tracking + counting | **DONE** — tracker Kalman + counter ROI/línea (E2E validado) |
| S7 | WiFi/BLE | **DONE** — nexmon + bleak, hashing + dedup L1/L2 |
| S8 | MQTT | **DONE** — IoT Core + buffer SQLite + replay |
| S9 | Cloud | **DONE** — CloudFormation + Lambda dedup L3 |
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
- `vision.sgbm_downscale: 2` — SGBM a resolución reducida (1=full, 2=half, 4=quarter), upscale del disparity post-match. 2 default; 4 para más FPS, 1 para diagnósticos.
- `--no-mqtt` (CLI) — reemplaza MQTT con no-op que loguea a stdout. Para testing local sin AWS.
- `detection.confidence_threshold` (0.20) / `new_track_threshold` (0.35) — ajustados para recall en pasadas rápidas con motion blur.
- `detection.static_suppressor` — defense-in-depth contra clutter estructural (FPs persistentes en mismas celdas). Configurable cell/window/threshold.
- `vision.max_exposure_us: 8000` — shutter cap 8ms para reducir motion blur. AE compensa con AnalogueGain.
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

- `focus_assist`, `calibrate`, `preview`, `diagnose_depth` son **standalone** — no leen `config.yaml`. Todo por CLI o defaults de `config/config.example.yaml`.
- **Browser-driven**: nada de `input()` blocking. Pantalla "Comenzar", AudioContext unlocked on click, beeps cortos diferenciados (start / pose / captura / undo / fin) en lugar de TTS, reporte HTML auto-open al finalizar.
- **AE flags compartidos**: `--meter matrix|centre|spot` (default matrix; centre/spot para periferia brillante), `--lock-ae` (calibrate + diagnose_depth, default off — AE auto típicamente mejor).
- **Wizard guardrails**: pre-calibration sanity gate (re-detección de pares capturados), coverage critical block (banda/grupo faltante), L/R asymmetric alert, `--resume` valida resolución, `reset --yes` para restart limpio.
- **Lens locking**: holder M12 sin set screw → fijado con Trabasil AM3 + activador anaeróbico (cura parcial 15min, total horas). Llave dedicada queda puesta durante foco + calibración. Ver `docs/lab-calibration-guide.md`.
- **Interpretación reporte calibración**: RMS estéreo <0.5px es necesario pero no suficiente. **El verdict depende del ground-truth en centro** (cinta/láser). Baseline estimada debe caer ±1-2mm del diseño 140mm con 20 poses diversas; ratio borde/centro es informativo, no gate.
- `debug/` está gitignoreado — para reportes, screenshots, logs de tests.

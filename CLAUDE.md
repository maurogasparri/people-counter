# CLAUDE.md — People Counter Edge System

## Descripción general

Sistema de conteo de personas de bajo costo para locales comerciales. Visión estéreo + IA en el borde + detección pasiva de tráfico WiFi/BLE.

**Este es un proyecto de producción real.** La calidad del código, el manejo de errores y la resiliencia son críticos. Los dispositivos operan desatendidos 12h/día, 363 días/año.

## Arquitectura

```
+---------------------------------------------+
|        Dispositivo edge (por local)          |
|  RPi5 4GB + Hailo-8L 13T + 2x IMX708       |
|                                              |
|  +----------+  +----------+  +--------+     |
|  |  Visión   |  | WiFi/BLE |  |  MQTT  |    |
|  |           |  |          |  | Client |    |
|  |           |  |          |  |        |    |
|  | Stereo -> |  | Monitor  |  | QoS 1  |    |
|  | YOLOv8n ->|  | Probe -> |  | Buffer |    |
|  | Track ->  |  | Hash ->  |  | SQLite |    |
|  | Count     |  | Dedup    |  |        |    |
|  +----------+  +----------+  +--------+     |
+------------------+---------------------------+
                   | MQTT (TLS + X.509)
                   v
+---------------------------------------------+
|              AWS Cloud                       |
|                                              |
|  IoT Core -> Timestream (series temporales)  |
|           -> Lambda (WiFi/BLE dedup)         |
|           -> DynamoDB (hashes dedup)         |
|           -> API Gateway -> QuickSight       |
+---------------------------------------------+
```

## Hardware por unidad

- Raspberry Pi 5 4GB — SBC principal
- Raspberry Pi Active Cooler — fan PWM + disipador para gestión térmica
- Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) — acelerador neuronal
- 2x Arducam IMX708 12MP HDR, lente M12 120 HFOV (B0310) vía CSI — par estéreo, baseline 14cm
- Waveshare PoE HAT (H) 25.5W (802.3at) conectado por dupont (no stackeado) — alimentación por Ethernet
- SanDisk Extreme 64GB microSD — boot + almacenamiento

## Decisiones técnicas clave

### Pipeline de visión
- **Calibración estéreo**: patrón ChArUco (A3 landscape, 11x7 squares, checker 35mm / marker 26mm, DICT_5X5_100, 60 esquinas internas), modelo pinhole con `cv2.calibrateCamera` (CALIB_RATIONAL_MODEL). Los parámetros del board son requeridos en todos los subcomandos CLI. Intrínsecos/extrínsecos guardados como `.npz` por dispositivo. Captura a 0.5–3m (cubriendo todo el rango operativo, no solo el sweet spot). Validar con `scripts/diagnose_depth.py` a múltiples distancias — chequea 5 zonas (centro + 4 esquinas), exige error centro <5% a 2m / <10% a 3m y ratio borde/centro <2×.
- **Rectificación**: mapas precomputados vía `cv2.initUndistortRectifyMap`. Aplicados por par de frames.
- **Profundidad**: Semi-Global Block Matching (`cv2.StereoSGBM`) sobre par rectificado + matcher derecho + filtro WLS (`cv2.ximgproc.DisparityWLSFilter`).
- **Detección**: YOLOv8n compilado a HEF vía Hailo Model Zoo. Corre en Hailo-8L a 30+ FPS. Usa API VStream de `hailo_platform` con activación persistente, VDevice compartido (`group_id="SHARED"`, scheduling `ROUND_ROBIN`), y NMS on-chip.
- **Tracking**: tracker por distancia euclidiana en espacio 3D (x, y, profundidad). ID único por trayectoria.
- **Conteo**: línea virtual en coordenadas de profundidad. Dirección de cruce = evento ingreso/egreso. Publicación inmediata vía MQTT.

### Captura WiFi/BLE
- **WiFi**: CYW43455 en monitor mode vía nexmon (firmware-nexmon + brcmfmac-nexmon-dkms de paquetes Kali) + airmon-ng. Captura probe requests en 2.4 Y 5 GHz. **WiFi es EXCLUSIVO para probing — la conectividad de red es solo por Ethernet.**
- **BLE**: Mismo CYW43455 vía bleak (API D-Bus de BlueZ). Escaneo pasivo de advertising.
- **Hashing**: SHA-256 truncado a 16 bytes sobre cada MAC antes de almacenar. Nunca se guardan MACs crudas.
- **Dedup L1 (intra-protocolo)**: set SQLite de hashes por día por protocolo. Reset al inicio del día comercial.
- **Dedup L2 (cross-protocolo)**: WiFi + BLE dentro de ventana de 2s Y delta RSSI <= 5dBm -> hash unificado.
- **Dedup L3 (inter-cámara)**: Cloud Lambda + DynamoDB por store_id + fecha.

### Comunicación
- **MQTT**: AWS IoT Core, certificados cliente X.509, QoS 1.
- **Eventos de conteo**: en tiempo real en cada cruce.
- **Resúmenes WiFi/BLE**: cada 15 min.
- **Telemetría**: cada 5 min (temp CPU, temp Hailo, RAM, disco, uptime).
- **Buffer SQLite**: todos los eventos se almacenan localmente. Replay al reconectar. Se marca enviado solo después de PUBACK.

### Cloud (AWS)
- IoT Core: broker MQTT + rules engine.
- Timestream: series temporales de conteo. 7 días en memoria, magnético para historial.
- Lambda: dedup WiFi/BLE entre cámaras por local.
- DynamoDB: tabla de hashes de dedup, particionada por store_id + fecha.
- API Gateway: API REST para consultas.
- QuickSight: dashboards.

## Convenciones de código

- **Lenguaje**: Python 3.13 (RPi OS Trixie)
- **Formatter**: Black, 88 chars
- **Linter**: Ruff
- **Type hints**: requeridos en todas las firmas de funciones
- **Logging**: módulo `logging`, JSON estructurado. DEBUG para dev, INFO para prod.
- **Config**: YAML en `/etc/people-counter/config.yaml`. Ver `config/config.example.yaml`.
- **Secrets**: certificados X.509 en `/etc/people-counter/certs/`. Nunca commitear.
- **Tests**: pytest, estructura espejo de src.
- **No usar clases salvo que haya estado.** Tracker y MQTTClient justifican clases. Preferir funciones en el resto.
- **Todo I/O debe tener manejo de errores.** Lectura de cámara, publicación MQTT, escritura de archivo — todo wrapeado.

## Estructura del directorio

```
people-counter/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/
│   ├── vision/
│   │   ├── capture.py     <- adquisición de frames estéreo (CSI en vivo + replay de archivos)
│   │   ├── calibration.py <- calibración ChArUco + rectificación
│   │   ├── depth.py       <- disparidad SGBM + conversión a profundidad
│   │   └── detect.py      <- inferencia YOLOv8n (backends Hailo + OpenCV)
│   ├── tracking/
│   │   ├── tracker.py     <- tracker euclidiano 3D
│   │   └── counter.py     <- lógica de cruce de línea virtual
│   ├── wifi_ble/
│   │   ├── wifi_probe.py  <- captura nexmon/airmon-ng de probes
│   │   ├── ble_scan.py    <- captura de advertising BLE vía bleak
│   │   ├── hasher.py      <- hashing SHA-256 truncado
│   │   └── dedup.py       <- dedup intra + cross-protocolo
│   ├── mqtt/
│   │   ├── client.py      <- cliente MQTT AWS IoT Core
│   │   └── buffer.py      <- buffer local SQLite
│   ├── cloud/
│   │   └── lambda_dedup.py <- dedup inter-cámara L3
│   ├── config/
│   │   └── loader.py      <- carga y validación de config YAML
│   └── main.py            <- orquestador del pipeline completo
├── tests/                 <- 180 tests en todos los módulos
├── scripts/
│   ├── calibrate.py       <- herramienta CLI de calibración (4 subcomandos, headless)
│   ├── focus_assist.py    <- asistente de foco guiado con preview HTTP
│   ├── diagnose_depth.py  <- diagnóstico de estimación de profundidad
│   ├── provision.py       <- provisioning de dispositivos (create/deploy/list)
│   ├── verify_hardware.py <- verificación de hardware
│   └── setup_device.sh    <- setup automático del dispositivo (pasos 4-10)
├── calibration/
│   └── charuco_11x7_sq35mm_mk26mm_dict5X5_a3_calibio.pdf <- board ChArUco (PDF vectorial calib.io, A3)
├── infra/
│   └── cloudformation/
│       └── people-counter.yaml <- stack completo (IoT, Timestream, DynamoDB, Lambda)
├── docs/
│   └── setup-guide.md     <- guía de ensamblaje + setup RPi
└── config/
    ├── config.example.yaml
    └── people-counter.service <- servicio systemd
```

## Plan de sprints (tareas de desarrollo)

| Sprint | Foco | Entregable | Estado |
|--------|------|-----------|--------|
| S3 | PoC | Captura estéreo + YOLOv8n en RPi5. Probar que funciona. | **HARDWARE VALIDADO** — capture.py adaptado a picamera2, captura estéreo verificada en RPi5. detect.py (backends Hailo + OpenCV). |
| S4 | Calibración | Pipeline ChArUco. Rectificación. Mapa de profundidad. | **DONE** — calibration.py modelo pinhole (CALIB_RATIONAL_MODEL). Board: 11x7 / 35mm / 26mm / DICT_5X5_100 (A3, en `calibration/`). Baseline 142.8mm. diagnose_depth.py valida 5 zonas con umbrales PASS/FAIL. |
| S5 | Detección | Compilación HEF. Integración Hailo SDK. 30+ FPS. | **SOFTWARE READY** — detect.py con backends Hailo + OpenCV, pre/postproceso testeado (10 tests). Hailo-8L verificado (fw 4.23.0, PCIe Gen 3). Compilación HEF pendiente. |
| S6 | Tracking | Tracker 3D. Línea virtual. Eventos ingreso/egreso. | **DONE** — tracker.py + counter.py (12 tests). main.py conectado E2E (17 tests). |
| S7 | WiFi/BLE | nexmon + captura BLE. Hashing. Dedup L1+L2. | **HARDWARE VALIDADO** — wifi_probe.py (nexmon + airmon-ng + scapy, probes capturadas), ble_scan.py (bleak, 343 adverts/8 dispositivos únicos). hasher.py + dedup.py (11 tests). |
| S8 | MQTT | Cliente IoT Core. Buffer SQLite. Reconexión. | **DONE** — client.py con TLS, replay de buffer, backoff (7 tests). |
| S9 | Cloud | Lambda dedup L3. CloudFormation. | **DONE** — lambda_dedup.py (9 tests). Template CloudFormation con IoT Core, Timestream, DynamoDB, Lambda, IAM. QuickSight/API GW diferidos post-MVP. |
| S10 | Integración | End-to-end. Todos los módulos juntos. | **E2E VALIDADO** — pipeline testeado en RPi5: capture -> rectify -> depth (SGBM) -> detect (Hailo) -> depth por persona. |
| S11 | Piloto | Deploy en 3 locales. Monitorear. Corregir. | PENDIENTE |
| S12 | Estabilización | Correcciones post-piloto. | PENDIENTE |

## Estado de implementación

**180 tests pasando.** Módulos por estado:

- COMPLETO + VALIDADO: capture (picamera2), detect (Hailo-8L HEF), wifi_probe (nexmon), ble_scan (bleak), calibration, depth, tracker, counter, hasher, dedup, buffer, client, lambda_dedup, loader, main
- INFRA READY: template CloudFormation, servicio systemd, provision.py, logrotate, timer de reset diario
- PENDIENTE: test de detección cenital, ajuste SGBM con cámaras IMX708

## Limitaciones conocidas (MVP)

- **Tracker**: matching greedy por distancia de píxeles 2D + gating por profundidad. Sin histéresis, sin multi-estado (approaching/crossed/invalidated), sin reidentificación. Suficiente para montaje cenital en puerta simple, necesita trabajo para entradas amplias con oclusión.
- **Shadow config**: bootstrap con caché local desde archivo `.shadow.json`. Sin suscripción delta en vivo — planificado post-MVP.
- **Horario operativo fail-open**: si el formato del horario es inválido, el conteo continúa (prefiere falsos positivos a pérdida de datos). Fail-closed configurable planificado para producción.

## Reglas duras

- **No transmitir video/imágenes.** Solo metadatos.
- **No almacenar MACs crudas.** Hashear primero, siempre.
- **WiFi = solo probing.** Red = Ethernet.
- **Stack de HATs**: AI HAT+ es el único HAT stackeado. PoE HAT (H) se conecta por dupont.
- **No hardcodear config.** Todo en YAML.
- **Siempre buffear localmente.** Asumir que la conectividad va a fallar.

## Entorno

- Raspberry Pi OS Trixie 64-bit, Python 3.13
- Hailo SDK: hailo_platform 4.23+
- Picamera2: para captura de cámaras CSI (herramientas CLI rpicam-*)
- OpenCV: 4.8+ (con contrib para ArUco/ChArUco)
- MQTT: paho-mqtt 2.0+
- DB: sqlite3 (stdlib)

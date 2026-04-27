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
- Waveshare PoE HAT (H) 25.5W (802.3at) conectado por dupont (2× 5V + 2× GND para repartir corriente, no stackeado) — alimentación por Ethernet
- LED RGB 3mm common cathode — status visual al operador, dupont 2x2 al header (R/G/B vía GPIO 17/18/27 con resistores 150/100/100Ω, cátodo a GND pin 14)
- SanDisk Extreme 64GB microSD — boot + almacenamiento

## Decisiones técnicas clave

### Pipeline de visión
- **Calibración estéreo**: patrón ChArUco (A3 landscape, 9x6 squares, checker 45mm / marker 33mm, DICT_4X4_100, 40 esquinas internas), **modelo fisheye Kannala-Brandt** con `cv2.fisheye.calibrate` (4 coef angulares k1–k4), apropiado para el lens Arducam 120°×152° que opera en zona útil hasta ±40% horizontal del frame (cobertura tipo FootfallCam). El flag `--dict` en `calibrate.py` permite usar boards alternativos. R y T del par se derivan de los extrínsecos per-pose que devuelve `fisheye.calibrate` (promedio + proyección SO(3)), manejando counts de puntos variables entre poses. Intrínsecos/extrínsecos guardados como `.npz` por dispositivo. **Protocolo de lab (universal para toda la flota)**: foco y calibración se hacen una sola vez en laboratorio con distancias fijas, no por sitio. Foco a 2.0m ±20cm (el DoF del M12 120° cubre 1.15–3.30m, o sea mount 3–4.5m). Calibración con poses a 1.0/2.0/3.0m (interpola hasta el operativo 3.0m, extrapola 30cm hasta 3.30m; fisheye Kannala-Brandt tolera bien esa extrapolación). `mounting_height_m` en `config.yaml` es solo para runtime (SGBM auto-tune, head-height gating). Validar con `scripts/diagnose_depth.py` a múltiples distancias — chequea 5 zonas (centro + 4 esquinas), exige error centro <5% a 2m / <10% a 3m y ratio borde/centro <2×.
- **Óptica de las cámaras**: Arducam B0310 con M12 120° HFOV (fisheye). Focal física 2.87mm / pixel pitch 1.4μm → pinhole-equivalente `f_px = 2050` a full-res 4608x2592 (`NOMINAL_FOCAL_PX` en `src/vision/calibration.py`). El FOV es fisheye real, no rectilíneo — la fórmula `f = (W/2)/tan(HFOV/2)` no aplica y daría valores erróneos.
- **Sensor modes (IMX708)**: `2304×1296 @ 56fps` (2x2 binned, full FOV, 16:9), `2304×1296 @ 30fps HDR`, `1536×864 @ 120fps` (partial-FOV crop). `focus_assist.py` pina el modo binned con `raw={"size": (2304, 1296)}` para evitar que picamera2 elija un sensor mode de FOV parcial. `calibrate.py` usa full-res 4608x2592 para máxima precisión.
- **Rectificación**: mapas precomputados vía `cv2.fisheye.initUndistortRectifyMap` (balance=0.0 para cero bordes negros, que ensucian SGBM). Aplicados por par de frames con `cv2.remap`.
- **Profundidad**: Semi-Global Block Matching (`cv2.StereoSGBM`) sobre par rectificado + matcher derecho + filtro WLS (`cv2.ximgproc.DisparityWLSFilter`). El config `vision.num_disparities` acepta `"auto"` — el pipeline deriva el rango de búsqueda de disparidades desde `mounting_height_m`, cubriendo exactamente las distancias donde van a aparecer cabezas + piso. Sites altos corren SGBM más rápido, sites bajos obtienen más rango automáticamente. Un int explícito override es posible si hace falta (`192`, `256`, etc.).
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
- **MQTT**: protocolo 3.1.1 sobre AWS IoT Core, certificados cliente X.509, QoS 1.
- **Eventos de conteo**: en tiempo real en cada cruce.
- **Resúmenes WiFi/BLE**: cada 15 min.
- **Telemetría**: cada 5 min (temp CPU, temp Hailo, RAM, disco, uptime).
- **Buffer SQLite**: todos los eventos se almacenan localmente. Replay al reconectar. Se marca enviado solo después de PUBACK.

### Status LED
- **Hardware**: RGB 3mm common cathode en GPIO 17 (R, 150Ω) / 18 (G, 100Ω) / 27 (B, 100Ω) + GND pin 14, dupont 2x2. Resistencias asimétricas porque G y B (InGaN, Vf≈3.1V) tienen apenas 0.2V de headroom contra los 3.3V del GPIO mientras que R (AlGaInP, Vf≈2.1V) tiene 1.2V — los valores apuntan a brillo perceptualmente parejo entre canales, no a corrientes iguales. Sin esa asimetría las mezclas tiran al verde por la mayor eficiencia luminosa del eye response.
- **Esquema**: 8 estados alineados con el código de FootfallCam (apagado / rojo fijo / amarillo fijo / amarillo parpadeante / verde parpadeante / verde fijo / azul fijo / azul parpadeante) — el operador del local interpreta sin SSH. Cascada de prioridad worst-first: HW > pipeline > internet > cloud > OK.
- **Health checks** (`src/status/health.py`): CPU temp <80°C, Hailo temp <85°C, disco free >10%, calibración cargable, captura/inferencia OK, pipeline watchdog (`last_loop_ts` <5s), internet TCP a 1.1.1.1:53 (cacheado 30s), MQTT `connected` flag.
- **Monitor en thread separado** (`src/status/monitor.py`): probes blocking (socket connect 3s timeout) no estresan el hot path del pipeline. `HealthSignals` es shared state mutable; el pipeline escribe, el monitor lee — atomicidad garantizada por el GIL para tipos primitivos.
- **Fail-safe**: si `gpiozero` no está disponible o los pines no se pueden abrir, `StatusLED` cae a no-op + log INFO. Permite correr el servicio en bench sin LED conectado y los tests sin GPIO real.
- **Bajo consumo asociado**: LEDs onboard ACT/power (vía `dtparam=*_led_trigger=none` + `*_led_activelow=off`) + LEDs del jack Ethernet (`eth_led0=4`, `eth_led1=4`) + audio PWM (`dtparam=audio=off`) apagados — el RGB externo es la única fuente visual de estado.

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
│   ├── status/
│   │   ├── led.py         <- driver RGB LED + state machine + blink thread
│   │   ├── health.py      <- probes (CPU/Hailo temp, disco, internet, cloud) + decide_state
│   │   └── monitor.py     <- thread background que mapea HealthSignals -> LedState
│   ├── config/
│   │   └── loader.py      <- carga y validación de config YAML
│   └── main.py            <- orquestador del pipeline completo
├── tests/                 <- 456 tests en todos los módulos
├── scripts/
│   ├── calibrate.py       <- herramienta CLI (generate-board, capture, calibrate, verify, wizard).
│   │                         wizard es el flujo end-to-end con UI web: start overlay,
│   │                         captura guiada con ghost silueta + audio TTS,
│   │                         bootstrap de intrínsecos, residuales por par,
│   │                         ground-truth check, reporte auto-open.
│   │                         Tolerance presets loose/normal/strict, --dict configurable.
│   ├── focus_assist.py    <- asistente de foco guiado, UI web: start overlay,
│   │                         barras de nitidez central + corners (absoluto) + simetría L/R,
│   │                         peak tracker, masking de zonas de bajo contraste,
│   │                         audio TTS, reporte auto-open. Captura a 2304×1296 (binned).
│   │                         Auto-detecta "escena compacta" (bbox del board >25% del frame)
│   │                         y omite el check de corners en esa geometría. --scene=
│   │                         auto|compact|full override, --min-corner-score N ajusta umbral.
│   ├── diagnose_depth.py  <- diagnóstico de estimación de profundidad
│   ├── provision.py       <- provisioning + disaster recovery (create/deploy/harvest/reprovision/list)
│   ├── verify_hardware.py <- verificación de hardware
│   └── setup_device.sh    <- setup automático del dispositivo (pasos 4-10)
├── calibration/
│   └── calib.io_charuco_420x297_6x9_45_33_DICT_4X4.pdf <- board ChArUco (PDF vectorial calib.io, A3 landscape)
├── infra/
│   └── cloudformation/
│       └── people-counter.yaml <- stack completo (IoT, Timestream, DynamoDB, Lambda)
├── docs/
│   ├── setup-guide.md          <- guía de ensamblaje + setup RPi (13 pasos)
│   ├── lab-calibration-guide.md <- protocolo de foco + calibración en lab (universal)
│   └── pilot-operator-guide.md <- guía del operador en sitio
└── config/
    ├── config.example.yaml
    └── people-counter.service <- servicio systemd
```

## Plan de sprints (tareas de desarrollo)

| Sprint | Foco | Entregable | Estado |
|--------|------|-----------|--------|
| S3 | PoC | Captura estéreo + YOLOv8n en RPi5. Probar que funciona. | **HARDWARE VALIDADO** — capture.py adaptado a picamera2, captura estéreo verificada en RPi5. detect.py (backends Hailo + OpenCV). |
| S4 | Calibración | Pipeline ChArUco. Rectificación. Mapa de profundidad. | **DONE** — calibration.py modelo fisheye Kannala-Brandt (`cv2.fisheye.*`). Board: 9x6 / 45mm / 33mm / DICT_4X4_100 (A3, en `calibration/`). Baseline 140mm por diseño. diagnose_depth.py valida 5 zonas con umbrales PASS/FAIL. |
| S5 | Detección | Compilación HEF. Integración Hailo SDK. 30+ FPS. | **SOFTWARE READY** — detect.py con backends Hailo + OpenCV, pre/postproceso testeado (16 tests). Hailo-8L verificado (fw 4.23.0, PCIe Gen 3). Compilación HEF pendiente. |
| S6 | Tracking | Tracker 3D. Línea virtual. Eventos ingreso/egreso. | **DONE** — tracker.py (13 tests) + counter.py (24 tests). main.py conectado E2E (21 tests). |
| S7 | WiFi/BLE | nexmon + captura BLE. Hashing. Dedup L1+L2. | **HARDWARE VALIDADO** — wifi_probe.py (nexmon + airmon-ng + scapy, probes capturadas, 13 tests), ble_scan.py (bleak, 343 adverts/8 dispositivos únicos, 11 tests). hasher.py (5 tests) + dedup.py (12 tests). |
| S8 | MQTT | Cliente IoT Core. Buffer SQLite. Reconexión. | **DONE** — client.py con TLS, replay de buffer, backoff (15 tests) + buffer.py SQLite (3 tests). |
| S9 | Cloud | Lambda dedup L3. CloudFormation. | **DONE** — lambda_dedup.py (9 tests). Template CloudFormation con IoT Core, Timestream, DynamoDB, Lambda, IAM. |
| S10 | Integración | End-to-end. Todos los módulos juntos. | **E2E VALIDADO** — pipeline testeado en RPi5: capture -> rectify -> depth (SGBM) -> detect (Hailo) -> depth por persona. |
| S11 | Piloto | Deploy en 3 locales. Monitorear. Corregir. | PENDIENTE |
| S12 | Estabilización | Correcciones post-piloto. | PENDIENTE |

## Estado de implementación

**456 tests pasando.** Módulos por estado:

- COMPLETO + VALIDADO: capture (picamera2), detect (Hailo-8L HEF), wifi_probe (nexmon), ble_scan (bleak), calibration, depth, tracker, counter, hasher, dedup, buffer, client, lambda_dedup, loader, main, status (led + health + monitor)
- INFRA READY: template CloudFormation, servicio systemd, provision.py, logrotate, timer de reset diario
- POR VALIDAR EN PILOTO: detección cenital con people counting real en los 3 locales del S11

## Alcance del diseño

- **Tracker**: matching greedy por distancia de píxeles 2D + gating por profundidad. Diseñado para montaje cenital en puerta simple — el caso de uso objetivo para esta generación del producto.
- **Shadow config**: bootstrap con caché local desde archivo `.shadow.json`. Lee el estado al inicio, sin suscripción delta en vivo — los cambios de config se aplican en el próximo boot del servicio.
- **Horario operativo fail-open**: si el formato del horario es inválido, el conteo continúa (prefiere falsos positivos a pérdida de datos).
- **Sync L/R**: picamera2 con dos instancias independientes tiene offset típico ~60ms (1 frame a 15fps). Irrelevante para calibración (board quieto) y dentro de tolerancia para tracking humano (±60ms = ±6cm a 1m/s).
- **Clasificador adulto/niño**: threshold único 1.55m (`adult_min_m`), cerca del P25 de mujeres adultas en Argentina. La métrica agregada prioriza estabilidad de totales diarios sobre precisión per-evento.
- **Backup / disaster recovery**: config y `calibration.npz` se preservan en el workstation (`provisioned/<id>/`) — config queda durante `create`, calibration se trae con `harvest` post-calibración. Los certs **no se respaldan**: ante restore (SD muerta) `provision.py reprovision` revoca el cert viejo en IoT Core y emite uno nuevo asociado a la misma thing. Trade-off elegido: rotación efectiva del cert vale el ~30s extra de AWS API; las credenciales admin de AWS nunca tocan la Pi.

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
- OpenCV: 4.10+ (con contrib para ArUco/ChArUco; `CharucoParameters` requiere 4.8+)
- MQTT: paho-mqtt 2.1+
- SciPy: 1.13+ (residuales de calibración)
- DB: sqlite3 (stdlib)

## Convenciones para setup tools

- `focus_assist.py` y `calibrate.py` son **standalone** — no leen `config.yaml`. Todo se pasa por CLI porque tienen que correr durante la instalación inicial, antes de que exista config.
- **Ambos son 100% browser-driven**. No hay `input()` blocking ni prompts de terminal — la consola solo muestra logs. Cualquier interacción operativa (comenzar, confirmar diversidad, ingresar distancia ground-truth, finalizar) sucede en el UI.
- Ambos comparten el mismo flujo de UI web: pantalla "Comenzar" al abrir, AudioContext unlocked on click, audio TTS opcional, barras verdes, reporte HTML auto-abierto en pestaña nueva al finalizar.
- **TTS deduplicado por semántica**: los hints de voz comparan la frase sin dígitos, así fluctuaciones menores (ej. `315<200` vs `318<200`) no re-triggereean el mismo mensaje.
- **Pose announcement = bloque atómico**: `Pose N. <label>. A Xcm de la cámara` se reproduce entero. El capture y otros audios están gateados por una flag server-side `_announce_pending` que se limpia solo cuando el browser confirma el fin del utterance via `SpeechSynthesisUtterance.onend` → POST `/announce-done`. Timing basado en señal, no en timer estimado.
- **Banner sincronizado con audio**: la cadena visible del hint de movimiento se pinea al texto del último audio dicho (`state.locked_alignment_text`) — evita drift de 1cm entre lo que se escucha y lo que aparece en pantalla cuando la medición del offset fluctúa entre frames.
- **Prompts interactivos del wizard** via threading.Event: `_ask_operator_ui()` emite un estado `data-phase="prompt"` con `data-prompt-type`, el JS renderiza los controles apropiados (botones para confirmación binaria, input numérico para valores) y deja el spinner mientras el backend procesa. El POST a `/wizard-input` signal-ea el event y el wizard continúa.
- **Acciones que pueden interrumpir el audio** (flush queue + POST `/announce-done`): Saltar pose, Deshacer última, Finalizar. Todo lo demás espera al onend.
- **Puerto HTTP con SO_REUSEADDR** y Ctrl+C limpio en ambos tools (server.shutdown() en el cleanup path) — no quedan TIME_WAITs entre runs.
- **focus_assist: corner sharpness absoluta + escena compacta**. La métrica de uniformidad es la varianza Laplaciana media de los 4 corners válidos (umbral absoluto `MIN_CORNER_SCORE = 100`), no el ratio bordes/centro. Auto-detecta "escena compacta" cuando el bbox del ChArUco cubre >25% del frame (ambiente chico donde el board domina la vista) y omite el check porque en esa geometría los corners ven paredes a distancia no relacionada con el board y el ratio fallaría por razón física, no óptica. `--scene=auto|compact|full` override manual; `--min-corner-score N` ajusta el umbral.
- El directorio `debug/` en la raíz del repo está gitignoreado — sirve para dumpear reportes, screenshots, logs de sesiones de test sin ensuciar git status.

## Interpretación del reporte de calibración

- **RMS estéreo <0.5px** es condición necesaria pero NO suficiente. Mide self-consistency: si el solver encuentra parámetros que fitean bien los datos. Con capturas degeneradas (pocas poses, similares entre sí) hay infinitos sets de parámetros que fitean igual — el solver elige uno, pero puede ser geométricamente incorrecto.
- **Baseline estimada** (se calcula del set, no se mide): con 20 poses diversas sobre trípode debe caer a ±1-2mm del diseño 140mm. Si cae ±5-70mm, el problema es capturas insuficientes / poco diversas, no el bracket. El warning del reporte y del log lo explica en ese orden.
- **Validación ground-truth** (depth check contra distancia conocida) es el test real: error centro <5% a 2m, <10% a 3m, edge/center ratio <2×. Si la calibración pasa esto, sabés que la calibración sirve.
- Protocolo recomendado para el piloto: **20 poses estándar del wizard en todos los dispositivos** (el wizard las genera deterministically). Misma secuencia → números comparables entre unidades, outliers detectables por QA.

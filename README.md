# people-counter

Sistema de conteo de personas de bajo costo para locales comerciales, basado en visión estéreo e IA en el borde.

## Qué hace

- **Cuenta personas** que entran y salen de un local en tiempo real usando profundidad por cámara estéreo + YOLOv8n en acelerador Hailo-8L
- **Detecta tráfico exterior** vía captura pasiva de probe requests WiFi y advertising BLE
- **Clasifica tráfico** con umbrales duales de RSSI: transeúntes (-75 dBm) vs compradores (-55 dBm), calculando Turn In Rate
- **Deduplica** señales WiFi/BLE entre protocolos (L1+L2 en dispositivo) y entre cámaras del mismo local (L3 en Lambda)
- **Transmite metadatos** a AWS vía MQTT con buffer local SQLite para resiliencia offline
- **Respeta horarios operativos** vía AWS IoT Device Shadow (configuración pushada desde la nube)

## Hardware

Cada unidad consiste en:

| Componente | Spec | Rol |
|-----------|------|------|
| Raspberry Pi 5 | 4GB RAM, ARM Cortex-A76 | SBC principal |
| Raspberry Pi Active Cooler | fan PWM + disipador | Gestión térmica |
| Raspberry Pi AI HAT+ | 13 TOPS (Hailo-8L) | Inferencia neuronal |
| 2x Arducam IMX708 | 12MP HDR, 120° HFOV, lente M12, CSI, baseline 14cm | Par estéreo |
| Waveshare PoE HAT (H) | 25.5W, 802.3at | Alimentación por dupont (2× 5V + 2× GND, no stackeado) |
| LED RGB 3mm | common cathode, 4 patas | Status visual al operador (R/G/B vía GPIO 17/18/27 con 150/100/100Ω) |
| MicroSD | SanDisk Extreme 64GB | Boot + almacenamiento |

## Arquitectura

```
Dispositivo edge (por puerta)         AWS Cloud
+--------------------------+         +-------------------------+
| Capture → Rectify → SGBM |  MQTT   | IoT Core → Timestream   |
| YOLOv8n → Track → Count  |--TLS-->| Lambda → DynamoDB       |
| WiFi/BLE → Hash → Dedup  |  QoS1  | API GW → QuickSight     |
| SQLite buffer (72h)      |         | CloudWatch + S3 OTA     |
+--------------------------+         +-------------------------+
```

### Procesos en el edge

El dispositivo corre tres servicios systemd independientes:

| Servicio | Proceso | Qué hace |
|---------|---------|----------|
| `people-counter.service` | `src/main.py` | Pipeline de visión: capture → rectify → depth → detect → track → count → MQTT |
| `wifi-monitor.service` | `airmon-ng` | Pone el WiFi en monitor mode para captura de probe requests |
| `people-counter-reset.timer` | Diario a las 04:00 | Resetea contadores de dedup y totales de conteo para el nuevo día comercial |

El probing WiFi/BLE corre como servicio separado porque requiere acceso exclusivo al hardware WiFi (monitor mode). Visión y WiFi nunca compiten por recursos. Ambos publican independientemente a MQTT, y la dedup L3 entre cámaras se hace en la nube (Lambda).

La config cloud usa una estrategia de **caché local de shadow**: al bootear, `main.py` lee un archivo `.shadow.json` si existe (actualizado por un proceso de fondo o en el boot anterior). Los cambios cloud-pusheados se aplican en el próximo boot del servicio.

### Status LED para diagnóstico en sitio

Un LED RGB en el frente del enclosure le da al operador del local un código visual del estado del dispositivo sin SSH. Esquema alineado con FootfallCam (cascada worst-first por capa: hardware > pipeline > internet > cloud > OK):

| LED | Patrón | Significado |
|-----|--------|-------------|
| Apagado | — | Sin power (PoE caído) |
| Rojo | Fijo | Boot failure (servicio no levanta) |
| Amarillo | Fijo | Hardware roto (cámara, Hailo, temp >80°C, disco lleno) |
| Amarillo | Parpadeante | Pipeline stalled o software crasheó |
| Verde | Parpadeante | Sin internet (ethernet up pero no llega afuera) |
| Verde | Fijo | Internet OK, AWS IoT no responde |
| Azul | Fijo | Operación normal |
| Azul | Parpadeante | Sin provisioning (certs ausentes — solo en install) |

`src/status/health.py` corre los probes (CPU/Hailo temp, disco free, calibración cargable, internet TCP a 1.1.1.1:53, MQTT connected flag, watchdog del pipeline) y `src/status/monitor.py` los agrega cada 2s en un thread separado para no estresar el hot path. Configurable vía `status_led:` en `config.yaml` (pines GPIO, intervalos, enabled flag para bench sin LED).

## Estado del proyecto

| Área | Estado | Detalles |
|------|--------|---------|
| Código fuente | 21 módulos en `src/` | Visión + tracking + wifi/ble + mqtt + cloud + config + status + main + telemetry |
| Tests | 492/492 pasando | Visión, tracking, MQTT, WiFi/BLE, config (hardware + user), cloud, main, provision (incl. disaster recovery), reports, wizard, status LED + health monitor, clasificador adulto/niño, training pipeline (download_roboflow + bench_detector) |
| Config | Hardware + Local + Cloud | Tres niveles: `config/hardware.yaml` (en repo, inmutable — bracket geometry + sensor invariants), `/etc/people-counter/config.yaml` (per-device, mutable — mounting_height, paths, MQTT), AWS IoT Shadow (cloud, business-driven — schedule, scaling, toggles). Runtime-safe prefixes para cambios cloud-pusheados sin reinicio |
| Hardware | Ensamblado + verificado | RPi5 + Hailo-8L (fw 4.23, PCIe Gen 3) + 2x Arducam IMX708 120° HFOV |
| Captura estéreo | Validada | picamera2, ambas cámaras funcionando. Sensor mode canónico 2304×1296 (binned full-FOV, 16:9) para foco, calibración y runtime — elegido por velocidad de detección ChArUco (≥8 FPS en Pi 5), mejor SNR del binning 2x2, y para que rectify+SGBM quepan en el budget runtime de 30+ FPS |
| Detección | Software validado, modelo en fine-tuning | YOLOv8n HEF en Hailo-8L, VDevice persistente con scheduling ROUND_ROBIN. El detector se entrena específicamente para geometría cenital (no se usa el stock COCO porque CrowdHuman entrena vistas frontales/laterales). Phase A en marcha: fine-tune con dataset Roboflow `coding-compass-nmjfb/overhead-head-detection-cwetj v2` (15.4k imgs `head-top-view`), entreno en Kaggle T4 → ONNX → `hailomz compile` en WSL2 → HEF a la Pi. Pipeline en `scripts/training/` |
| Calibración | Validada | **Fisheye Kannala-Brandt** (`cv2.fisheye.*`, 4 coef angulares k1–k4), baseline 140mm por diseño. ChArUco 9x6/45mm/33mm/DICT_4X4_100 A3. Protocolo lab: poses a 1.0/2.0/3.0m, foco único a 2.0m ±20cm para toda la flota. `calibrate.py wizard` 100% browser-driven: start overlay, ghost silueta, audio TTS con pose-announce atómico gateado por `SpeechSynthesisUtterance.onend`, tolerance preset (`loose`/`normal`/`strict`), ground-truth en UI con spinner, reporte HTML con rectificación epipolar + depth heatmap embebidos. Salvaguardas anti-degeneración: pre-calibration sanity gate (re-detección ≥70% en ambas cámaras), coverage critical block (banda completa o grupo entero faltante = abort), L/R asymmetric detection alert en panel. Preview L durante captura guiada **sin overlay de ChArUco** (badge "N esquinas" en lugar de los 40 puntitos+IDs que tapaban el ghost), R sí mantiene overlay como diagnóstico. Subcomando `reset --yes` para restart limpio. Flag `--low-light` para PoC en cuarto chico/oscuro (afloja gates de quality, NO produce calibración válida) |
| Asistente de foco | Validado | `focus_assist.py` UI web: header + side panel, start overlay, peak tracker, masking de zonas de bajo contraste, audio TTS opcional, auto-open del reporte. Target range lab protocol 1.80–2.20m por default. Lens locking con Trabasil AM3 + activador anaeróbico (cura parcial 15min) y llave dedicada en el barrel — habilita foco + calib en una sola sesión de lab. **L/R parity check**: pill verde "OK" / roja "INVERTIDO" / ámbar "magnitud rara" basada en disparidad medida vs esperada por baseline+depth — detecta wiring swapped antes de calibrar. Flag `--low-light` para PoC en cuarto chico/oscuro (preset que afloja todos los gates y fuerza scene=compact). Flag `--meter centre/spot` para luz baja con zonas brillantes en periferia |
| Preview en vivo | Disponible | `preview.py` — tool minimal browser-driven con UX consistente con focus / calib (start overlay, header). MJPEG side-by-side L|R con grid de tercios + crosshair central. Para apuntar el bracket, verificar oclusiones, o sanity check del wiring antes de correr foco/calibración. Sin detección, sin análisis. Flag `--meter centre/spot` |
| Validación de profundidad | Reformulada | `diagnose_depth.py` y la fase ground-truth del wizard reportan **verdict basado solo en error del centro** (única zona con distancia conocida). Las 4 zonas perimetrales se clasifican con tags: ✓ Coincide / ● Otro plano / ⚠ SGBM falló según `std × fill_rate` — distinción honesta entre "calibración errada" vs "está midiendo otro objeto" vs "SGBM no puede matchear esta superficie". Reporte HTML con verdict card prominente arriba + tabla por zonas con tags de confianza |
| Clasificador adulto/niño | Implementado | Head-height por stereo depth (`mount_height - min_depth_at_bbox`). Threshold `adult_min_m: 1.55` (cerca de P25 de mujeres adultas en Argentina). Majority vote por track |
| WiFi probe | Validada | nexmon + airmon-ng + scapy, probe requests capturadas en RPi5 |
| BLE scan | Validado | bleak, 343 adverts, 8 dispositivos únicos, dedup + turn-in rate |
| Infra cloud | CloudFormation | IoT Core, Timestream, DynamoDB, Lambda (dedup L3) |
| Deployment | Listo | provision.py (create/deploy/harvest/reprovision), servicios systemd (pipeline + wifi-monitor + reset diario), logrotate, preflight |
| Disaster recovery | Listo | `harvest` baja `calibration.npz` al workstation; `reprovision` revoca cert viejo en IoT Core y emite uno nuevo. Certs nunca se respaldan — rotan en cada restore |
| Guía de setup | Completa | Guía de 14 pasos desde microSD hasta backup/disaster recovery (docs/setup-guide.md). Guía para operadores en campo (docs/pilot-operator-guide.md) |

## Quick start

```bash
git clone https://github.com/maurogasparri/people-counter.git
cd people-counter
pip install -e ".[dev]"
pytest
```

### Dependencias

| Paquete | Instalar vía | Notas |
|---------|------------|-------|
| opencv-contrib-python, numpy, scipy, paho-mqtt, pyyaml, scapy, bleak | `pip install -e ".[dev]"` | Multiplataforma, funciona en máquinas de desarrollo |
| python3-numpy, python3-scipy, python3-opencv, python3-yaml, python3-paho-mqtt | `apt` (binarios precompilados) | En la Pi se instalan vía apt — pip-compilar scipy/opencv en la Pi tarda mucho. Ver `setup_device.sh` |
| picamera2, libcamera | `apt` (python3-picamera2) | Solo RPi, provisto por RPi OS Trixie |
| hailo_platform | `apt` (hailort + hailort-pcie-driver + python3-hailort) | Solo RPi, requiere Hailo-8L + PCIe |
| aircrack-ng, nexmon | `apt` + paquetes `.deb` | Solo RPi, WiFi monitor mode |

En máquinas de desarrollo (Windows/Mac/Linux), `pip install -e ".[dev]"` es suficiente para correr tests. Los paquetes del sistema RPi solo se necesitan en el dispositivo target — ver [docs/setup-guide.md](docs/setup-guide.md) para la instalación completa.

## Configuración

El sistema usa una estrategia de doble config:

- **Local** (`config/config.yaml`): settings intrínsecos al hardware — IDs de cámara, archivo de calibración, parámetros SGBM (`num_disparities: auto` deriva el rango de disparidad desde `mounting_height_m` para cada sitio), path del modelo, certificados MQTT
- **Cloud** (AWS IoT Device Shadow): settings del negocio — horarios operativos, factor de escala, toggles de habilitación

Ver [`config/config.example.yaml`](config/config.example.yaml) para el config anotado completo.

## Instalación en sitio

Los tools de setup (`focus_assist.py`, `calibrate.py`) son **standalone** — no leen
`config.yaml`, todo se pasa por CLI. Esto permite correrlos durante la
instalación inicial antes de que exista config.

### 1. Ajuste de foco

```bash
sudo PYTHONPATH=. python3 scripts/focus_assist.py
```

Abre UI en `http://people-counter.local:8080`. Flujo:
1. Pantalla "Comenzar" (posicionar board + activa audio)
2. Captura en vivo con barras de nitidez central + corners (absoluto) + simetría L/R
3. Peak tracker para ajustar el M12 sin pasarse del óptimo
4. Masking automático de zonas de bajo contraste
5. Audio TTS lee los hints (opcional, toggle en la UI; OFF corta la voz al toque)
6. Finalizar → reporte HTML auto-abierto en nueva pestaña

Target de foco: **1.80–2.20m** por default (lab protocol — focar a 2.0m ±20cm
cubre con el DoF del M12 todo el rango operativo 1.15–3.30m de la flota).
Overridear con `--target-distance-min-mm` / `--target-distance-max-mm` si hace
falta. El flag `--mount-height-m` es solo informativo.

**Escena compacta**: si el bounding-box del ChArUco cubre >25% del frame
(ambiente chico donde el board llena la vista) el tool auto-detecta
la situación y omite el check de corners — en esa geometría los bordes
ven superficies a distancia no relacionada con el board y el check
fallaría por razón física, no óptica. Forzá el modo con `--scene=compact`
(siempre omite) o `--scene=full` (siempre enforza). La métrica de corners
es absoluta (varianza Laplaciana media ≥ 100 por default), ajustable con
`--min-corner-score N`.

**Modo PoC (`--low-light`)**: para validar el flujo en cuarto chico /
luz baja. Afloja umbrales (centro 80, corners 30, L/R diff 50%, zonas
100%, distancia 0.5–5.0m) y fuerza `--scene=compact`. Los flags
explícitos siguen ganándole al preset. Un PASS en este modo **no
valida foco para producción** — solo confirma que el tool corre.

### 2. Calibración estéreo

```bash
sudo PYTHONPATH=. python3 scripts/calibrate.py wizard \
    --device-id DEV-001 \
    --output /etc/people-counter/calibration.npz \
    --tolerance strict
```

Wizard de una sola corrida, **todo desde el browser** (terminal solo para logs):
1. Pre-flight (puerto, disco, backup)
2. Pantalla "Comenzar" (activa audio del browser, posicionar trípode)
3. Captura guiada con ghost silueta (20 poses), audio TTS
4. Cada pose se anuncia como un bloque atómico — `Pose N. <label>. A Xcm de la cámara` — y el capture queda bloqueado hasta que el browser confirma que el audio terminó (`SpeechSynthesisUtterance.onend` → POST `/announce-done`)
5. Hints de movimiento en cm (`"movelo izquierda 4cm"`), texto en pantalla sincronizado con el último audio dicho para evitar drift de 1cm
6. Bootstrap de intrínsecos tras las primeras 6 capturas
7. Calibración estéreo con modelo fisheye Kannala-Brandt (`--min-captures` configurable)
8. Residuales por par + chequeo de baseline (estimada del set, no medida físicamente)
9. Confirmación UI si la diversidad es limitada (botones Continuar/Cancelar)
10. Ground-truth opcional con input + spinner mientras procesa (botón Saltear)
11. Reporte HTML auto-abierto en nueva pestaña — incluye rectificación epipolar y mapa de profundidad ground-truth embedded

Presets de tolerancia: `loose` (ambientes con poca diversidad de poses),
`normal` (default, tuned para A3 canónico), `strict` (producción, con
board sobre trípode).

Flags útiles:
- `--min-captures N`: baja el mínimo para terminar calibración (default 15)
- `--pose-timeout-sec N`: timeout por pose antes de auto-skip (default 180, tuned para board sobre trípode; bajar a 60 si hand-held)
- `--tolerance loose|normal|strict`: preset de tolerancia
- `--align-tol-{loose,tight}-px`: overrides finos de tolerancia
- `--dist-near-mm` / `--dist-mid-mm` / `--dist-far-mm`: distancias de las 3 bandas de poses (default 1000/2000/3000mm — lab protocol)
- `--legacy-pattern` / `--no-legacy-pattern`: enumeración de markers ChArUco (default `--legacy-pattern` matches calib.io)
- `--resume`: continúa una sesión previa (si no se pasa, descarta y arranca limpio)
- `--force-degenerate-coverage`: bypass del coverage critical block (banda/grupo faltante)
- `--min-detect-rate F`: umbral del pre-calibration sanity gate (default 0.7)
- `--low-light`: preset PoC para cuarto chico/oscuro — afloja los gates de `assess_frame_quality` (exposure, blur, corner-sharp, L/R balance). El `.npz` resultante **NO es válido para producción**, solo valida que el wizard corra end-to-end

Subcomando aparte para restart limpio:
```bash
python scripts/calibrate.py reset --yes   # borra captures + session.json + .npz
```

**Protocolo recomendado para el piloto**: las mismas 20 poses en cada dispositivo
(el wizard las genera deterministically). Comparabilidad entre unidades para QA,
mismo procedimiento para entrenar operadores, detección de outliers entre locales.

## Estructura del repo

```
src/
├── vision/          # Captura estéreo (picamera2), calibración ChArUco, profundidad SGBM + WLS, detección YOLOv8n (Hailo + OpenCV), world_coords para altura de cabeza, report HTML
├── tracking/        # Tracker euclidiano 3D + contador por línea virtual / ROI con height_class por track
├── wifi_ble/        # Captura de probes WiFi (nexmon), scan BLE (bleak), hashing de MAC, dedup (L1+L2)
├── mqtt/            # Cliente AWS IoT Core + buffer SQLite con replay
├── cloud/           # Lambda dedup L3 (inter-cámara)
├── status/          # Driver RGB LED + health probes + thread monitor que mapea HealthSignals → LedState
├── config/          # Loader de hardware.yaml + user config.yaml + merge con IoT Shadow + runtime-safe prefixes
├── telemetry.py     # Reporte periódico: CPU/Hailo temp, RAM, disco, uptime
└── main.py          # Orquestador del pipeline (captura → depth → detect → track → count → MQTT). Flag --no-mqtt para debug local sin AWS
tests/               # 451 tests espejando src/ + tests/scripts/ para el wizard
scripts/
├── calibrate.py           # CLI: generate-board, capture, calibrate, verify, wizard, reset
│                          # wizard = pipeline end-to-end browser-driven: start overlay,
│                          # ghost silueta, pose-announce atómico, tolerance preset,
│                          # ground-truth en UI, reporte HTML con viz embedded.
│                          # Flags: --meter centre/spot, --lock-ae, --low-light
├── focus_assist.py        # Asistente de foco browser-driven: start overlay, barras
│                          # visuales, peak tracker, masking, audio TTS, reporte auto-open.
│                          # Corner sharpness absoluta + auto-detección de escena compacta
│                          # (bbox board/frame >25% → omite check de corners). L/R parity
│                          # check. Flags: --meter centre/spot, --low-light
├── preview.py             # Preview en vivo browser-driven (start overlay, header).
│                          # Side-by-side L|R con grid + crosshair. Para apuntar el bracket
│                          # antes de correr foco / calib. Flag --meter centre/spot
├── diagnose_depth.py      # Validación de profundidad: 5 zonas con tags de confianza
│                          # (✓ Coincide / ● Otro plano / ⚠ SGBM falló). Verdict solo
│                          # por error del centro. Flags: --meter, --lock-ae
├── preflight.py           # Chequeo pre-install (cámaras + Hailo + hardware)
├── roi_picker.py          # Seleccionador de ROI + línea virtual
├── export_events.py       # Export de eventos desde el buffer local
├── provision.py           # Provisioning + disaster recovery: create, deploy, harvest, reprovision, list
├── deploy_lambda.sh       # Packaging del Lambda dedup L3
├── download_model.py      # Descarga YOLOv8n HEF (Hailo Model Zoo) o ONNX (ultralytics) — usar el HEF fine-tuneado de scripts/training/, no el stock
├── capture_baseline_frames.py  # Captura frames rectificados de la Pi para validation bench (no training)
├── training/
│   ├── README.md          # Walkthrough Phase A → B end-to-end
│   ├── download_roboflow.py    # Pull de dataset Roboflow Universe a dataset/
│   ├── bench_detector.py       # Bench de inferencia + diff de reportes (baseline vs fine-tuned)
│   └── .env.example       # Convención del env var ROBOFLOW_API_KEY
├── verify_hardware.py     # Verificación de hardware
└── setup_device.sh        # Setup automático del dispositivo (pasos 4-10)
notebooks/
└── train_yolov8n_heads.ipynb    # Notebook Kaggle T4 (training Phase A)
config/
├── config.example.yaml           # User config anotado con estrategia local/cloud
├── hardware.yaml                 # Hardware design constants (baseline, CSI L/R, sensor) — inmutable
├── people-counter.service        # Servicio systemd principal
├── people-counter-reset.*        # Timer de reset diario de dedup (04:00)
└── logrotate.conf                # Rotación de logs
infra/
└── cloudformation/people-counter.yaml  # Stack completo de AWS
docs/
├── setup-guide.md                # Ensamblaje de hardware + setup RPi (13 pasos)
├── lab-calibration-guide.md      # Protocolo de foco + calibración en lab (universal para la flota)
└── pilot-operator-guide.md       # Guía para el operador en sitio (foco → calibración → verificación)
dataset/                          # Drop-zone gitignoreado para datasets de training (Roboflow, WEPDTOF)
debug/                            # Drop-zone gitignoreado para reportes, capturas y logs de test
```

## Referencias clave

- [CLAUDE.md](CLAUDE.md) — Documentación completa de arquitectura para Claude Code
- [docs/setup-guide.md](docs/setup-guide.md) — Guía de ensamblaje de hardware + setup RPi
- [config/config.example.yaml](config/config.example.yaml) — Configuración anotada con estrategia

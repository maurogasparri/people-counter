# people-counter

Sistema de conteo de personas de bajo costo para locales comerciales, basado en visión estéreo e IA en el borde.

## Qué hace

- **Cuenta personas** que entran y salen de un local en tiempo real usando profundidad por cámara estéreo + YOLOv8n en acelerador Hailo-8L
- **Detecta tráfico exterior** vía captura pasiva de probe requests WiFi y advertising BLE
- **Clasifica tráfico** con umbrales duales de RSSI: transeúntes (-75 dBm) vs compradores (-55 dBm), calculando Turn In Rate
- **Cuenta solo dispositivos "humanos"** en el tráfico exterior: WiFi con MAC randomizada (locally-administered bit) y BLE con address type *random* — los identificadores globales (OUI real) son infra/IoT fijo (APs, smart-TVs, beacons) y se descartan.
- **Deduplica** WiFi/BLE local en el device via hash groups con stitching de 4 reglas: seqnum continuity 802.11 (anti MAC-randomization), cross-protocol L2 (WiFi+BLE simultáneo), BLE anchoring (durante la vida de un RPA ~15min), y fingerprint continuity (IEs en WiFi / manufacturer-data en BLE, estable a la rotación — cubre lo que el seqnum no agarra, ej. iOS que resetea el seqnum). Los counts publicados son distinct grupos, no distinct hashes. La L3 inter-cámara queda reservada para deploys multi-cam por local (no aplica al PoC con 1 device/sucursal).
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
Dispositivo edge (por puerta)         AWS Cloud (PoC, 1 device)
+--------------------------+         +-------------------------------+
| Capture → Rectify → SGBM |  MQTT   | IoT Core (3 IoT Rules)        |
| YOLOv8n → Track → Count  |--TLS-->| → Lambda persist_event         |
| WiFi/BLE → Hash → Stitch |  QoS1  |    (IAM auth a RDS, out-VPC)   |
| SQLite buffer + dedup    |         | → RDS Postgres 16 (db.t4g.μ)  |
+--------------------------+         | → ECS Fargate + ALB + Grafana |
                                     |    (custom domain HTTPS)      |
                                     +-------------------------------+
```

### Procesos en el edge

El dispositivo corre tres servicios systemd independientes:

| Servicio | Proceso | Qué hace |
|---------|---------|----------|
| `people-counter.service` | `src/main.py` | Pipeline de visión: capture → rectify → depth → detect → track → count → MQTT |
| `wifi-monitor.service` | `rfkill unblock` + `airmon-ng check kill` | Libera wlan0 al boot (destraba rfkill + mata NM/wpa); el pipeline lo pasa a monitor mode (`iw` + `nexutil -m2`) |
| `people-counter-reset.timer` | Diario a las 04:00 | Resetea contadores de dedup y totales de conteo para el nuevo día comercial |

El probing WiFi/BLE corre como servicio separado porque requiere acceso exclusivo al hardware WiFi (monitor mode). Visión y WiFi nunca compiten por recursos. Ambos publican independientemente a MQTT. La dedup L3 inter-cámara queda reservada para deploys multi-cam future work (no aplica al PoC con 1 device/sucursal).

La config cloud usa una estrategia de **caché local de shadow**: al bootear, `main.py` lee un archivo `.shadow.json` si existe (actualizado por un proceso de fondo o en el boot anterior). Los cambios cloud-pusheados se aplican en el próximo boot del servicio.

### Status LED para diagnóstico en sitio

Un LED RGB en el frente del enclosure le da al operador del local un código visual del estado del dispositivo sin SSH. Cascada worst-first por capa (hardware > pipeline > internet > cloud > OK):

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
| Código fuente | 30 módulos en 9 paquetes de `src/` | `vision/` (8: capture, calibration, depth, detect, world_coords, static_suppressor, best_frame, report) + `wifi_ble/` (6: wifi_probe, ble_scan, fingerprint, hasher, dedup, publisher) + `tracking/` (3: tracker, kalman, counter) + `status/` (3: health, led, monitor) + `mqtt/` (2: client, buffer) + `cloud/` (2: persist_event, ingest_pos_transaction) + `config/` (2: loader, hardware) + `web/` (2: viewer, annotate) + `main.py` + `telemetry.py` |
| Tests | 842 pasando (455 funciones, 42 archivos) | Visión, tracking (incluye rescue cascade — ghost pool / decisive Kalman / death-emit con guards), MQTT, WiFi/BLE (4 reglas de stitching incl. fingerprint), config (defaults + per-device + HardwareParams), cloud, main, provision (incl. disaster recovery), reports, wizard, status LED + health monitor, clasificador adulto/niño, training pipeline (bench_detector), static suppressor (timestamp-based window) |
| Config | Defaults + Per-device + Cloud + Hardware-agnostic | `config/config.example.yaml` (defaults canónicos), `/etc/people-counter/config.yaml` (per-device override), AWS IoT Shadow (business cloud). Parámetros de hardware (sensor, lens, bracket, board ChArUco, AE timings) consolidados en `src/config/hardware.py` (HardwareParams) y leídos por el runtime + todos los setup tools — swap de sensor / bracket / board = solo editar config.yaml, ningún script tiene constantes hardware hardcodeadas |
| Hardware | Ensamblado + verificado | RPi5 + Hailo-8L (fw 4.23, PCIe Gen 3) + 2x Arducam IMX708 120° HFOV |
| Captura estéreo | Validada | picamera2, ambas cámaras funcionando. Sensor mode canónico 2304×1296 (binned full-FOV, 16:9) para foco, calibración y runtime — elegido por velocidad de detección ChArUco (≥8 FPS en Pi 5), mejor SNR del binning 2x2, y para que rectify+SGBM quepan en el budget runtime de 30+ FPS |
| Detección | Pipeline activo | YOLOv8n HEF en Hailo-8L, VDevice persistente con scheduling ROUND_ROBIN. El detector se entrena específicamente para geometría cenital (no se usa el stock COCO porque CrowdHuman entrena vistas frontales/laterales). Modelo activo: `people-counter-detector` — fine-tune sobre dataset propio multi-site (945 imgs sampleadas con `sample_for_roboflow.py` desde 5 sites capturados en paralelo, ratio post-screening ~2:1 positivos:hard-negatives, labeling con Smart Polygon de Roboflow click-por-imagen en 1h 45min). Defense-in-depth runtime: containment filter post-NMS + `StaticSuppressor` por celda. Pipeline en `scripts/training/`: notebook único `train_head_detector.ipynb` para iteraciones, compile HEF en WSL2, deploy a la Pi |
| Calibración | Validada | **Fisheye Kannala-Brandt** (`cv2.fisheye.*`, 4 coef angulares k1–k4), baseline 140mm por diseño. ChArUco 9x6/45mm/33mm/DICT_4X4_100 A3. Protocolo lab universal (mount-independent, sirve para flota mount 2.0–3.5m): poses a 1.0/2.0/3.0m, foco único a 1.5m ±20cm. `calibrate.py wizard` 100% browser-driven: start overlay, ghost silueta, beeps cortos diferenciados (start / pose nueva / tick de hold / captura / undo / fin) con pose-announce gateado (capture queda bloqueado hasta que el browser confirma fin del beep vía POST `/announce-done`), tolerance preset (`loose`/`normal`/`strict`), ground-truth en UI con spinner, reporte HTML con rectificación epipolar + depth heatmap embebidos. Salvaguardas anti-degeneración: pre-calibration sanity gate (re-detección ≥70% en ambas cámaras), coverage critical block (banda completa o grupo entero faltante = abort), L/R asymmetric detection alert en panel. Preview L durante captura guiada **sin overlay de ChArUco** (badge "N esquinas" en lugar de los 40 puntitos+IDs que tapaban el ghost), R sí mantiene overlay como diagnóstico. Subcomando `reset --yes` para restart limpio. Flag `--low-light` para PoC en cuarto chico/oscuro (afloja gates de quality, NO produce calibración válida) |
| Asistente de foco | Validado | `focus_assist.py` UI web: header + side panel, start overlay, peak tracker, masking de zonas de bajo contraste, beeps en eventos (start / fin) + **pulso adaptativo tipo detector** (tap corto que acelera a medida que el score del centro se acerca al threshold de paso, lock holgado a >1.5×MIN_SCORE), auto-open del reporte. Target range lab protocol 1.30–1.70m (foco a 1.5m ±20cm) por default — universal para mount 2.0–3.5m. Lens locking con esmalte de uñas transparente aplicado al seam barrel↔holder (touch-dry 15min, cura full 30-60min) + llave dedicada en el barrel durante el foco — habilita foco + calib en una sola sesión de lab. **L/R parity check**: pill verde "OK" / roja "INVERTIDO" / ámbar "magnitud rara" basada en disparidad medida vs esperada por baseline+depth — detecta wiring swapped antes de calibrar. Flag `--low-light` para PoC en cuarto chico/oscuro (preset que afloja todos los gates y fuerza scene=compact). Flag `--meter centre/spot` para luz baja con zonas brillantes en periferia |
| Preview en vivo | Disponible | `preview.py` — tool minimal browser-driven con UX consistente con focus / calib (start overlay, header). MJPEG side-by-side L|R con grid de tercios + crosshair central. Para apuntar el bracket, verificar oclusiones, o sanity check del wiring antes de correr foco/calibración. Sin detección, sin análisis. Flag `--meter centre/spot` |
| Validación de profundidad | Reformulada | `diagnose_depth.py` y la fase ground-truth del wizard reportan **verdict basado solo en error del centro** (única zona con distancia conocida). Las 4 zonas perimetrales se clasifican con tags: ✓ Coincide / ● Otro plano / ⚠ SGBM falló según `std × fill_rate` — distinción honesta entre "calibración errada" vs "está midiendo otro objeto" vs "SGBM no puede matchear esta superficie". Reporte HTML con verdict card prominente arriba + tabla por zonas con tags de confianza |
| Clasificador adulto/niño | Implementado | Head-height por stereo depth (`mount_height - min_depth_at_bbox`). Threshold `adult_min_m: 1.55` (cerca de P25 de mujeres adultas en Argentina). Majority vote por track |
| WiFi probe | Validada | nexmon + airmon-ng + scapy, probe requests capturadas en RPi5 |
| BLE scan | Validado | bleak, 343 adverts, 8 dispositivos únicos, dedup + turn-in rate |
| Infra cloud | CloudFormation deployada (`infra/deploy.ps1`, 5 fases) | VPC + RDS Postgres 16.6 (db.t4g.micro, IAM auth + force_ssl + auto minor upgrades) + IoT Core (3 Topic Rules) + Lambda persist_event (out of VPC, psycopg + RDS token) + ECR + ECS Fargate Grafana 13 detrás de ALB con ACM cert custom (`grafana.tfg.gasparri.com.ar`) + SNS alarms. ~$35/mo PoC. Stitching ratio canary en `telemetry.wifi_ble_stitching_ratio` |
| Deployment | Listo | provision.py (create/deploy/harvest/reprovision), servicios systemd (pipeline + wifi-monitor + reset diario), logrotate, preflight |
| Disaster recovery | Listo | `harvest` baja `calibration.npz` al workstation; `reprovision` revoca cert viejo en IoT Core y emite uno nuevo. Certs nunca se respaldan — rotan en cada restore |
| Guía de setup | Completa | Guía de 14 pasos desde microSD hasta backup/disaster recovery (docs/setup_guide.md). Guía para operadores en campo (docs/pilot_operator_guide.md) |

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

En máquinas de desarrollo (Windows/Mac/Linux), `pip install -e ".[dev]"` es suficiente para correr tests. Los paquetes del sistema RPi solo se necesitan en el dispositivo target — ver [docs/setup_guide.md](docs/setup_guide.md) para la instalación completa.

## Configuración

El sistema usa **defaults canónicos + override per-device + cloud channel**:

- **Defaults** ([`config/config.example.yaml`](config/config.example.yaml)): el config canónico de la flota — bracket geometry, sensor mode, vision pipeline (SGBM, rectify), detection, tracking, counter, MQTT topics, buffer paths, status LED, etc. Todos los devices heredan estos valores.
- **Per-device** (`/etc/people-counter/config.yaml`): override mínimo con lo que cambia por unidad — `device.id`, `mounting_height_m`, ROI/lines, MQTT endpoint + certs. Cualquier key ausente cae al default. El loader hace deep-merge al boot.
- **Cloud** (AWS IoT Device Shadow): settings de negocio — horarios operativos, factor de escala, toggles de habilitación. Se aplican vía `RUNTIME_SAFE_KEYS` sin reinicio.

`mounting_height_m` (per-device) alimenta el SGBM auto-tune (`num_disparities: auto` deriva el rango de disparidad por sitio) y el head-height gating del clasificador adulto/niño. La calibración estéreo es mount-independent (un único `.npz` factory sirve para mount 2.0–3.5m).

## Instalación en sitio

Los tools de setup (`focus_assist.py`, `calibrate.py`, `preview.py`, `diagnose_depth.py`)
leen `/etc/people-counter/config.yaml` para `vision.resolution` y
`vision.mounting_height_m` (focus_assist deriva el target distance de ahí).
Pasar `--resolution` / `--mount-height-m` explícitos solo en dev workstation
sin config per-device. `diagnose_bracket.py` corre sin config porque hace QC
de ensamble pre-calibración.

Todos los setup tools comparten:

- `--max-exposure-us 16000` (default) — mismo cap que el runtime, freezea
  micro-vibración que rompe el decoder ArUco asimétricamente entre L/R.
- `--lock-ae` con patrón canónico: settle 2s → lock provisional → re-settle
  1.5s on click → re-lock final.
- `--meter matrix|centre|spot` (default matrix; usar centre/spot cuando la
  periferia es brillante).
- Browser-driven (Comenzar → trabajo → Finalizar) + reporte HTML auto-open + exit.

### 1. Ajuste de foco

```bash
sudo PYTHONPATH=. python3 scripts/focus_assist.py
```

Abre UI en `http://people-counter.local:8080`. Flujo:
1. Pantalla "Comenzar" (posicionar board + activa audio)
2. Captura en vivo con barras de nitidez central + corners (absoluto) + simetría L/R
3. Peak tracker para ajustar el M12 sin pasarse del óptimo
4. Masking automático de zonas de bajo contraste
5. Pulso de audio adaptativo (tipo detector — tap corto a 700 Hz que acelera de ~1.2s a ~130ms según mejora el score del centro; silencio cuando el board no está visible). Beeps separados de start y fin de sesión. Toggle en la UI.
6. Finalizar → reporte HTML auto-abierto en nueva pestaña

Target de foco: **1.30–1.70m** por default (lab protocol universal — focar a
1.5m ±20cm peakea el DoF sobre el rango operativo bbox 1.0–3.5m de la flota
mount 2.0–3.5m, simétrico en ambos extremos). Overridear con
`--target-distance-min-mm` / `--target-distance-max-mm` si hace falta. El flag
`--mount-height-m` es solo informativo.

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
3. Captura guiada con ghost silueta (20 poses), beeps cortos diferenciados por evento
4. Cada pose se anuncia con un doble-tap agudo (1100 Hz × 2) y el texto `Pose N. <label>. A Xcm de la cámara` aparece en el banner — el capture queda bloqueado hasta que el browser confirma que el patrón de beep terminó (POST `/announce-done` vía `setTimeout`)
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

### 3. QC de ensamble del bracket (opcional)

```bash
sudo PYTHONPATH=. python3 scripts/diagnose_bracket.py
```

Tool factory para validar el ensamble físico del bracket estéreo
**antes** de la calibración óptica. Mide pitch / yaw / roll / offset Y /
offset Z entre L y R usando solvePnP con K nominal del IMX708 (no
necesita `.npz` previo). Thresholds factory: pitch ±0.5° / yaw ±1.0° /
roll ±0.5° / offsets ±2mm. Browser-driven igual que focus/calib, ghost
del board para alineación, reporte HTML auto-open. Útil para detectar
brackets mal cortados / cámaras flojas antes de meterlas al wizard.

## Entrenamiento del detector

El modelo activo es `people-counter-detector`: YOLOv8n single-class
fine-tuneado para geometría cenital, compilado a HEF para Hailo-8L.
**No se usa el HEF stock del Hailo Model Zoo** — está entrenado en
COCO/CrowdHuman con vistas frontales que no transfieren a top-down.

Pipeline end-to-end (todo en `scripts/training/` — ver
[`scripts/training/README.md`](scripts/training/README.md) para el
walkthrough completo):

```
captura multi-site (motion-trigger)  →  sampling estratificado a Roboflow
                                    →  labeling con Smart Polygon (AI-Assisted)
                                    →  Generate Version (incluir nulls)
                                    →  Notebook Kaggle T4 (~20 min)
                                    →  best.onnx  →  hailomz compile (Docker x86)
                                    →  HEF en la Pi
```

Pasos resumidos:

1. **Captura de validation set**: `capture_mjpeg.py` multi-site con
   motion-trigger + background sampling. Filenames `_motion_` / `_bg_`
   para sampling balanceado downstream.
2. **Sampling para Roboflow**: `sample_for_roboflow.py` arma un subset
   estratificado por site (~75 motion + ~65 bg por site, con `--site-cap`
   para capear sites con sesgo conocido).
3. **Labeling con Smart Polygon** (Roboflow AI-Assisted Labeling, click-por-imagen
   sobre el project type **Object Detection**). Click-per-image consume menos
   credits que una pasada batch sobre todo el dataset. Revisión manual:
   promover bg con persona a positivo, dejar el resto como hard negatives.
4. **Generate Version**: confirmar `Filter Null = Use / Include Null
   Images` (Roboflow descarta sin labels por default). Augmentations:
   flip H, rotate ±10°, brightness ±20%, blur ligero.
5. **Training en Kaggle T4**: `train_head_detector.ipynb` (notebook
   único — para iterar a v3/v4/... basta cambiar la URL de Roboflow
   en Cell 2 y el `name` del run en Cell 3). ~20 min por iteración.
   Descarga vía Kaggle CLI con Save & Run All.
6. **Compilación HEF**: `hailomz compile` en Docker x86 Linux
   (WSL2 desde Windows). Receta exacta en
   `scripts/training/README.md` — flags críticos: `--classes 1` y
   `--end-node-names` obligatorio. Calibration set = 200 imgs
   muestreadas con `sample_for_calib.py` (sin leak del train).
7. **Deploy**: `scp` del `.hef` a `/usr/src/people-counter/models/`
   en la Pi + edición de `detection.model_path` en
   `/etc/people-counter/config.yaml`.

Bench tooling:

- `bench_detector.py` — inferencia + diff de reportes (baseline vs
  fine-tuned) sobre carpeta de frames. Subcomandos `bench` y `diff`.
- `bench_roboflow_api.py` — triage de modelos publicados en Roboflow
  Universe vía REST, sin descargar pesos. Útil para evaluar
  candidatos antes de fine-tunear.
- `eval_yolo.py` — corre un modelo sobre una carpeta y dumpea
  bboxes + summary para sanity-check rápido.

Defense-in-depth runtime (independiente del modelo):
- **Containment filter** post-NMS: descarta bbox chico contenido
  >50% en otro con mayor confidence (NMS por IoU no agarra esto en
  geometría cenital).
- **Cluster por centroide** (`cluster_distance_px: 150`): mergea
  detecciones multi-firing (cabeza + hombro como cajas distintas).
- **`StaticSuppressor`** (`cell_size_px: 30`, `window_seconds: 3`):
  suprime detecciones en celdas hot ≥70% de los últimos 3s — clutter
  estructural (maniquíes, ropa colgada, sombras) que sobrevive NMS.

## Estructura del repo

```
src/
├── vision/          # Captura estéreo (picamera2), calibración ChArUco, profundidad SGBM + WLS, detección YOLOv8n (Hailo + OpenCV), world_coords para altura de cabeza, static_suppressor + best_frame, report HTML
├── tracking/        # Tracker euclidiano 3D (Kalman) + ghost pool / ID adoption + contador por línea virtual / ROI con net-balance + rescue cascade de 3 capas (ghost adoption + decisive Kalman cross at exit + death-emit-if-crossed con guards anti-FP)
├── wifi_ble/        # Captura de probes WiFi (nexmon + radiotap, hopping ponderado 1/6/11), scan BLE (bleak), fingerprint (IEs/manufacturer-data), hashing, dedup a hash groups con stitching de 4 reglas (seqnum + cross-protocol L2 + BLE anchoring + fingerprint), publisher de summaries 15min
├── mqtt/            # Cliente AWS IoT Core + buffer SQLite con replay
├── cloud/           # Lambdas: persist_event (IoT Rules → RDS Postgres via IAM auth, out of VPC) + ingest_pos_transaction (POS API → tabla sales)
├── status/          # Driver RGB LED + health probes + thread monitor que mapea HealthSignals → LedState
├── config/          # Loader (deep-merge defaults + per-device + IoT Shadow) + hardware (HardwareParams dataclass — sensor/lens/bracket/charuco/ae_lock leídos del config, hardware-agnostic)
├── web/             # Live preview HTTP/MJPEG (viewer + annotate); gateado por has_subscribers — counting es sync-deterministic independiente del viewer
├── telemetry.py     # Reporte periódico: CPU/Hailo temp, RAM, disco, uptime + canaries (track_stitching_ratio, ghost_adoption_count, death_emit_count, wifi_ble_stitching_ratio)
└── main.py          # Orquestador del pipeline (captura → depth → detect → track → count → MQTT). Flag --no-mqtt para debug local sin AWS
tests/               # 842 tests (455 funciones en 42 archivos) espejando src/ + tests/scripts/ para el wizard
scripts/
├── calibrate.py           # CLI: generate-board, capture, calibrate, verify, wizard, reset
│                          # wizard = pipeline end-to-end browser-driven: start overlay,
│                          # ghost silueta, pose-announce atómico, tolerance preset,
│                          # ground-truth en UI, reporte HTML con viz embedded.
│                          # Flags: --meter centre/spot, --lock-ae, --low-light
├── focus_assist.py        # Asistente de foco browser-driven: start overlay, barras
│                          # visuales, peak tracker, masking, beeps en start / fin, reporte auto-open.
│                          # Corner sharpness absoluta + auto-detección de escena compacta
│                          # (bbox board/frame >25% → omite check de corners). L/R parity
│                          # check. Flags: --meter centre/spot, --low-light
├── preview.py             # Preview en vivo browser-driven (start overlay, header).
│                          # Side-by-side L|R con grid + crosshair. Para apuntar el bracket
│                          # antes de correr foco / calib. Flag --meter centre/spot
├── diagnose_depth.py      # Validación de profundidad: 5 zonas con tags de confianza
│                          # (✓ Coincide / ● Otro plano / ⚠ SGBM falló). Verdict solo
│                          # por error del centro. Flags: --meter, --lock-ae
├── diagnose_bracket.py    # QC factory del ensamble del bracket estéreo (pitch/yaw/
│                          # roll/offset L↔R), browser-driven, no necesita .npz previo.
│                          # Thresholds factory: ±0.5°/1.0°/0.5°/2mm/2mm.
├── preflight.py           # Chequeo pre-install (cámaras + Hailo + hardware)
├── roi_picker.py          # Seleccionador de ROI + línea virtual
├── provision.py           # Provisioning + disaster recovery: create, deploy, harvest, reprovision, list
├── deploy_lambda.ps1      # Packaging del Lambda persist_event (psycopg binary + handler). Versión .sh tambien disponible.
├── download_model.py      # Descarga YOLOv8n HEF (Hailo Model Zoo) o ONNX (ultralytics) — usar el HEF fine-tuneado de scripts/training/, no el stock
├── capture_baseline_frames.py  # Captura frames rectificados de la Pi para validation bench (no training)
├── training/
│   ├── README.md          # Walkthrough end-to-end del pipeline de detector
│   ├── train_head_detector.ipynb  # Notebook Kaggle T4 (descarga el dataset directo de Roboflow vía signed URL)
│   ├── bench_detector.py       # Bench de inferencia + diff de reportes (baseline vs fine-tuned)
│   ├── bench_roboflow_api.py   # Triage de modelos en Roboflow Universe vía REST sin descargar pesos
│   ├── capture_mjpeg.py        # Captura multi-site de streams MJPEG (random-interval + motion-trigger)
│   ├── record_clips.py         # Grabación continua de clips MP4 multi-site (validation E2E con tracker)
│   ├── sample_for_roboflow.py  # Sampling estratificado de capturas para subir a Roboflow
│   ├── sample_for_calib.py     # Sampling balanceado para el calib set del QAT de Hailo
│   ├── polys_to_bboxes.py      # Conversión de polígonos a bboxes YOLO (legacy, no aplica al pipeline actual)
│   ├── eval_yolo.py            # Corre un modelo YOLO sobre una carpeta de frames
│   └── .env.example       # Convención del env var ROBOFLOW_API_KEY
├── verify_hardware.py     # Verificación de hardware
└── setup_device.sh        # Setup automático del dispositivo (pasos 4-10)
config/
├── config.example.yaml           # Defaults canónicos para toda la flota (bracket, sensor, vision, detection, tracking, counter, mqtt, etc.). Cualquier key ausente del per-device cae a los valores de acá.
├── people-counter.service        # Servicio systemd principal
├── people-counter-reset.*        # Timer de reset diario de dedup (04:00)
└── logrotate.conf                # Rotación de logs
infra/
├── README.md                              # Walkthrough del deploy + costos + verificación E2E
├── cloudformation/people-counter.yaml     # Stack completo de AWS
├── deploy.ps1                             # Orquestador 5 fases (RDS + IoT + Lambda + ECR + cert ACM + ECS Fargate + ALB Grafana + CNAME). -StartFromPhase para resumir
└── sql/bootstrap.sql                      # Schema (count_events / wifi_ble_summary / telemetry / sales + 6 views + lambda_writer con rds_iam)
docs/
├── setup_guide.md                # Ensamblaje de hardware + setup RPi (13 pasos)
├── lab_calibration_guide.md      # Protocolo de foco + calibración en lab (universal para la flota)
└── pilot_operator_guide.md       # Guía para el operador en sitio (foco → calibración → verificación)
training_data/                    # Workspace gitignoreado de training (sites.yaml inline + captures rectificadas)
debug/                            # Drop-zone gitignoreado para reportes, capturas y logs de test
```

## Referencias clave

- [CLAUDE.md](CLAUDE.md) — Documentación completa de arquitectura para Claude Code
- [docs/setup_guide.md](docs/setup_guide.md) — Guía de ensamblaje de hardware + setup RPi
- [config/config.example.yaml](config/config.example.yaml) — Configuración anotada con estrategia

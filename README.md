# people-counter

Sistema de conteo de personas de bajo costo para locales comerciales, basado en visión estéreo e IA en el borde.

## Versión

Este repositorio corresponde a la versión **1.0.0** del prototipo, publicada bajo el tag **`v1.0-tfg`**. El identificador de commit correspondiente se obtiene con `git rev-parse v1.0-tfg` y figura además en la página del release.

Ese tag identifica el estado exacto del código, de las plantillas de infraestructura y de los registros de validación entregados como Trabajo Final de Grado de la Licenciatura en Administración de Infraestructura Tecnológica de la Universidad Siglo 21. Cualquier desarrollo posterior no forma parte de esa entrega.

## Qué hace

- **Cuenta personas** que entran y salen de un local en tiempo real usando profundidad por cámara estéreo + YOLOv8n en acelerador Hailo-8L
- **Detecta tráfico exterior** vía captura pasiva de probe requests WiFi y advertising BLE
- **Clasifica tráfico** con umbrales duales de RSSI: transeúntes (-75 dBm) vs compradores (-55 dBm), calculando Turn In Rate
- **Cuenta solo dispositivos "humanos"** en el tráfico exterior: WiFi con MAC randomizada (locally-administered bit) y BLE con address type *random* — los identificadores globales (OUI real) son infra/IoT fijo (APs, smart-TVs, beacons) y se descartan.
- **Deduplica** WiFi/BLE local en el device via hash groups con stitching de 4 reglas: seqnum continuity 802.11 (anti MAC-randomization), cross-protocol L2 (WiFi+BLE simultáneo), BLE anchoring (durante la vida de un RPA ~15min), y fingerprint continuity (IEs en WiFi / manufacturer-data en BLE, estable a la rotación — cubre lo que el seqnum no agarra, ej. iOS que resetea el seqnum). Los counts publicados son distinct grupos, no distinct hashes.
- **Transmite metadatos** a AWS vía MQTT con buffer local SQLite para resiliencia offline
- **Respeta horarios operativos** vía AWS IoT Device Shadow (configuración pushada desde la nube)

## Hardware

Cada unidad consiste en:

| Componente | Spec | Rol |
|-----------|------|------|
| Raspberry Pi 5 | 2GB RAM, ARM Cortex-A76 | SBC principal (working set ~270 MB sostenido; ver [`docs/hardware_sizing.md`](docs/hardware_sizing.md)) |
| Raspberry Pi Active Cooler | fan PWM + disipador | Gestión térmica |
| Raspberry Pi AI HAT+ | 13 TOPS (Hailo-8L) | Inferencia neuronal |
| 2x Arducam IMX708 | 12MP HDR, 120° HFOV, lente M12, CSI, baseline 14cm | Par estéreo |
| Waveshare PoE HAT (G) | 25.5W, 802.3at | Alimentación PoE, stackeado (orden: AI HAT+ → Pi → Active Cooler → PoE HAT) |
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
| `people-counter-reset.timer` | Diario a las 04:00 | Resetea el estado de dedup + rota el salt local. Los totales/canaries del counter se resetean in-process en el rollover de medianoche (`main.py`) |

El probing WiFi/BLE corre como servicio separado porque requiere acceso exclusivo al hardware WiFi (monitor mode). Visión y WiFi nunca compiten por recursos. Ambos publican independientemente a MQTT.

La config cloud usa una estrategia de **caché local de shadow**: al bootear, `main.py` lee un archivo `.shadow.json` si existe (actualizado por un proceso de fondo o en el boot anterior). Los cambios cloud-pusheados se aplican en el próximo boot del servicio.

### Status LED para diagnóstico en las instalaciones

Un LED RGB en el frente del enclosure le da al operador del local un código visual del estado del dispositivo sin SSH. **6 estados = 3 colores puros (rojo/verde/azul) × 2 modos (fijo/parpadeante)**, en cascada worst-first por capa (hardware > pipeline > internet > cloud > unprovisioned > OK). Sin amarillo (R+G se veía verdoso por el balance de brillo):

| LED | Patrón | Significado |
|-----|--------|-------------|
| Apagado | — | Sin power (PoE caído) |
| Rojo | Fijo | HARDWARE_FAULT — cámara/Hailo/disco/temp sobre umbral (config-driven, default 80°C), o crash/wedge del init |
| Rojo | Parpadeante | SOFTWARE_FAULT — pipeline stalled o crash del software |
| Verde | Parpadeante | Sin internet (ethernet up pero no llega afuera) |
| Verde | Fijo | Internet OK, AWS IoT no responde (sin cloud) |
| Azul | Parpadeante | Sin provisioning (certs ausentes — solo en install) |
| Azul | Fijo | Operación normal (OK) |

`src/status/health.py` corre los probes (CPU/Hailo temp, disco free, calibración cargable, internet TCP a 1.1.1.1:53, MQTT connected flag, watchdog del pipeline) y `src/status/monitor.py` los agrega cada 2s en un thread separado para no estresar el hot path. Configurable vía `status_led:` en `config.yaml` (pines GPIO, intervalos, enabled flag para bench sin LED).

## Estado del proyecto

**Prototipo completo.** Las 12 etapas del plan (S1–S12) están cerradas y todas las áreas validadas — visión, tracking/conteo, WiFi/BLE, mensajería, cloud, visualización y validación. Detalle por sprint en [CLAUDE.md](CLAUDE.md); resultados de las pruebas en [docs/benchmark_results.md](docs/benchmark_results.md).

| Área | Estado | Detalles |
|------|--------|---------|
| Código fuente | 33 módulos en 8 subpaquetes de `src/` | `vision/` (9: capture, calibration, depth, detect, world_coords, static_suppressor, pre_filter, best_frame, report) + `wifi_ble/` (6: wifi_probe, ble_scan, fingerprint, hasher, dedup, publisher) + `tracking/` (3: tracker, kalman, counter) + `status/` (3: health, led, monitor) + `mqtt/` (2: client, buffer) + `cloud/` (3: persist_event, ingest_pos_transaction, query_aggregates) + `config/` (2: loader, hardware) + `web/` (3: viewer, annotate, admin_auth) + `main.py` + `telemetry.py` |
| Tests | 1083 funciones de test en 48 archivos (1103 casos, 81% coverage — ver `docs/coverage_report.md`) | Visión, tracking (incluye rescue cascade — ghost pool / decisive Kalman / death-emit con guards + invalidación de outside_pos lejano del ghost + knobs config-driven per-site + matriz de cobertura discriminante en `docs/counter_test_matrix.md` + tracking_zone polygon filter pre-tracker + guards min_count_height_m / min_real_inside_frames anti-FP no-humanos), MQTT, WiFi/BLE (4 reglas de stitching incl. fingerprint), config (defaults + per-device + HardwareParams + shadow delta validation), cloud (incl. persist + ingest POS), main, provision (incl. disaster recovery), reports, wizard, status LED + health monitor, clasificador adulto/niño, training pipeline (bench_detector), static suppressor (timestamp-based window) |
| Config | Defaults + Per-device + Cloud + Hardware-agnostic | `config/config.example.yaml` (defaults canónicos), `/etc/people-counter/config.yaml` (per-device override), AWS IoT Shadow (business cloud). Parámetros de hardware (sensor, lens, bracket, board ChArUco, AE timings) consolidados en `src/config/hardware.py` (HardwareParams) y leídos por el runtime + todos los setup tools — swap de sensor / bracket / board = solo editar config.yaml, ningún script tiene constantes hardware hardcodeadas |
| Hardware | Ensamblado + verificado | RPi5 + Hailo-8L (fw 4.23, PCIe Gen 3) + 2x Arducam IMX708 120° HFOV |
| Captura estéreo | Validada | picamera2, ambas cámaras funcionando. Sensor mode canónico 2304×1296 (binned full-FOV, 16:9) para foco, calibración y runtime — elegido por velocidad de detección ChArUco (≥8 FPS en Pi 5), mejor SNR del binning 2x2, y para que rectify+SGBM quepan en el budget runtime de 30+ FPS |
| Detección | Pipeline activo | YOLOv8n HEF en Hailo-8L, VDevice persistente con scheduling ROUND_ROBIN. El detector se entrena específicamente para geometría cenital (no se usa el stock COCO porque CrowdHuman entrena vistas frontales/laterales). Modelo activo: `people-counter-detector` **v2** — fine-tune sobre dataset propio multi-site (544 imgs / 438 cajas, bbox cabeza+hombros, 5 sites en paralelo, validation set held-out de 245 imgs). Pipeline canónico: `sample_for_labeling.py` (estratificado) o `mine_active_learning.py` (informativo) → labeling local en **X-AnyLabeling** → `labelme_to_yolo.py` → Kaggle dataset privado → notebook único `train_head_detector.ipynb` en Kaggle T4 (~20 min). Defense-in-depth runtime: containment filter post-NMS + `StaticSuppressor` por celda + `tracking_zone` polygon opcional. **v1→v2 con active learning subió mAP50 de 0.805 → 0.956 contra val held-out** (la 2da ronda de AL bajó a 0.939 — sweet spot detectado, ver tabla completa en `scripts/training/README.md`). |
| Calibración | Validada | **Fisheye Kannala-Brandt** (`cv2.fisheye.*`, 4 coef angulares k1–k4), baseline 140mm por diseño. ChArUco 9x6/45mm/33mm/DICT_4X4_100 A3. Protocolo lab universal (mount-independent, sirve para flota mount 2.0–3.5m): poses a 1.0/2.0/3.0m, foco único a 1.5m ±20cm. `calibrate.py wizard` 100% browser-driven: start overlay, ghost silueta, beeps cortos diferenciados (start / pose nueva / tick de hold / captura / undo / fin) con pose-announce gateado (capture queda bloqueado hasta que el browser confirma fin del beep vía POST `/announce-done`), tolerance preset (`loose`/`normal`/`strict`), ground-truth en UI con spinner, reporte HTML con rectificación epipolar + depth heatmap embebidos. Salvaguardas anti-degeneración: pre-calibration sanity gate (re-detección ≥70% en ambas cámaras), coverage critical block (banda completa o grupo entero faltante = abort), L/R asymmetric detection alert en panel. Preview L durante captura guiada **sin overlay de ChArUco** (badge "N esquinas" en lugar de los 40 puntitos+IDs que tapaban el ghost), R sí mantiene overlay como diagnóstico. Subcomando `reset --yes` para restart limpio. Flag `--low-light` para PoC en cuarto chico/oscuro (afloja gates de quality, NO produce calibración válida) |
| Asistente de foco | Validado | `focus_assist.py` UI web: header + side panel, start overlay, peak tracker, masking de zonas de bajo contraste, beeps en eventos (start / fin) + **pulso adaptativo tipo detector** (tap corto que acelera a medida que el score del centro se acerca al threshold de paso, lock holgado a >1.5×MIN_SCORE), auto-open del reporte. Target range lab protocol 1.30–1.70m (foco a 1.5m ±20cm) por default — universal para mount 2.0–3.5m. Lens locking con esmalte de uñas transparente aplicado al seam barrel↔holder (touch-dry 15min, cura full 30-60min) + llave dedicada en el barrel durante el foco — habilita foco + calib en una sola sesión de lab. **L/R parity check**: pill verde "OK" / roja "INVERTIDO" / ámbar "magnitud rara" basada en disparidad medida vs esperada por baseline+depth — detecta wiring swapped antes de calibrar. Flag `--low-light` para PoC en cuarto chico/oscuro (preset que afloja todos los gates y fuerza scene=compact). Flag `--meter centre/spot` para luz baja con zonas brillantes en periferia |
| Preview en vivo | Disponible | `preview.py` — tool minimal browser-driven con UX consistente con focus / calib (start overlay, header). MJPEG side-by-side L|R con grid de tercios + crosshair central. Para apuntar el bracket, verificar oclusiones, o sanity check del wiring antes de correr foco/calibración. Sin detección, sin análisis. Flag `--meter centre/spot` |
| Validación de profundidad | Reformulada | `diagnose_depth.py` y la fase ground-truth del wizard reportan **verdict basado solo en error del centro** (única zona con distancia conocida). Las 4 zonas perimetrales se clasifican con tags: ✓ Coincide / ● Otro plano / ⚠ SGBM falló según `std × fill_rate` — distinción honesta entre "calibración errada" vs "está midiendo otro objeto" vs "SGBM no puede matchear esta superficie". Reporte HTML con verdict card prominente arriba + tabla por zonas con tags de confianza |
| Clasificador adulto/niño | Implementado | Head-height por stereo depth (`mount_height - min_depth_at_bbox`). Threshold `adult_min_m: 1.55` (cerca de P25 de mujeres adultas en Argentina). Majority vote por track |
| WiFi probe | Validada | nexmon + airmon-ng + scapy, probe requests capturadas en RPi5 |
| BLE scan | Validado | bleak, 343 adverts, 8 dispositivos únicos, dedup + turn-in rate |
| Infra cloud | CloudFormation deployada (`infra/deploy.ps1`, 6 fases) | VPC + RDS Postgres 16 (db.t4g.micro, IAM auth + force_ssl + auto minor upgrades) + IoT Core (3 Topic Rules) + Lambda persist_event (out of VPC, psycopg + RDS token) + ECR + ECS Fargate Grafana 13 detrás de ALB con ACM cert custom (`grafana.<tu-dominio>`) + SNS alarms. Stitching ratio canary en `telemetry.wifi_ble_stitching_ratio` |
| Deployment | Listo | provision.py (create/deploy/harvest/reprovision), servicios systemd (pipeline + wifi-monitor + reset diario), logrotate, preflight |
| Disaster recovery | Listo | `harvest` baja `calibration.npz` al workstation; `reprovision` revoca cert viejo en IoT Core y emite uno nuevo. Certs nunca se respaldan — rotan en cada restore |
| Guía de setup | Completa | Guía de 14 pasos desde microSD hasta backup/disaster recovery (docs/setup_guide.md). Guía para operadores en las instalaciones (docs/operator_guide.md) |

## Acceso a los dashboards (evaluación)

El backend cloud expone los tableros de Grafana en
**https://grafana.<tu-dominio>** (HTTPS con cert ACM, dominio propio).

Para revisión externa hay un usuario **read-only** (`user`, rol *Viewer*):
ve los 5 dashboards de las 2 carpetas (Analítica comercial + Operación y
flota), sin poder editar, ejecutar queries crudas (Explore deshabilitado)
ni acceder a administración. **La contraseña está disponible bajo solicitud.**

## Quick start

Desde cero, en una máquina de desarrollo (Windows, macOS o Linux). No requiere
la Raspberry Pi ni el acelerador: la suite de pruebas corre íntegra sin
hardware.

```bash
# 1. Clonar
git clone https://github.com/maurogasparri/people-counter.git
cd people-counter

# 2. Entorno aislado (Python >= 3.11; el dispositivo usa 3.13)
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Dependencias, incluidas las de desarrollo
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 4. Suite de pruebas
pytest -q
```

Resultado esperado sobre Windows: **1096 pruebas superadas y 7 omitidas**
(1103 en total). Cinco omisiones comprueban permisos de archivo POSIX; las
otras dos requieren el filtro WLS de `opencv-contrib`, ausente en el wheel de
Windows. Sobre el dispositivo (Linux aarch64) no hay omisiones: **1102
superadas y 1 fallida**, y esa falla es artefacto del entorno —la prueba
presupone que no existe el archivo de configuración de producción, que en un
equipo aprovisionado sí existe—. Ninguna omisión indica fallo.

El entorno resuelto por estas instrucciones se verificó el 2026-08-04 con
OpenCV 4.14, NumPy 2.5 y SciPy 1.18. La dependencia de OpenCV está acotada a
la serie 4.x de forma deliberada: la 5.0 removió una constante que usa el
solve de calibración estéreo.

Para medir cobertura:

```bash
pytest --cov=src --cov-report=term -q
```

Los artefactos del modelo de detección **no** son necesarios para la suite. Si
querés además reproducir las métricas del detector, seguí
[Artefactos del modelo](#artefactos-del-modelo).

### Dependencias

| Paquete | Instalar vía | Notas |
|---------|------------|-------|
| opencv-contrib-python, numpy, scipy, paho-mqtt, pyyaml, scapy, bleak | `pip install -e ".[dev]"` | Multiplataforma, funciona en máquinas de desarrollo |
| python3-numpy, python3-scipy, python3-opencv, python3-yaml, python3-paho-mqtt | `apt` (binarios precompilados) | En la Pi se instalan vía apt — pip-compilar scipy/opencv en la Pi tarda mucho. Ver `setup_device.sh` |
| picamera2, libcamera | `apt` (python3-picamera2) | Solo RPi, provisto por RPi OS Trixie |
| hailo_platform | `apt` (hailort + hailort-pcie-driver + python3-hailort) | Solo RPi, requiere Hailo-8L + PCIe |
| aircrack-ng, nexmon | `apt` + paquetes `.deb` | Solo RPi, WiFi monitor mode |

En máquinas de desarrollo (Windows/Mac/Linux), `pip install -e ".[dev]"` es suficiente para correr tests. Los paquetes del sistema RPi solo se necesitan en el dispositivo target — ver [docs/setup_guide.md](docs/setup_guide.md) para la instalación completa.

La tabla anterior indica qué versiones son **admisibles** para instalar el proyecto. No indica cuáles se usaron: un rango describe una familia de entornos posibles, no un entorno. El registro de las versiones exactas sobre las que se ejecutaron las mediciones que reporta el trabajo —dispositivo, plataforma cloud y contenedores, con sus digests— está en [docs/dependencies.md](docs/dependencies.md), junto con el congelado completo del entorno del dispositivo en [requirements-lock.txt](requirements-lock.txt).

## Configuración

El sistema usa **defaults canónicos + override per-device + cloud channel**:

- **Defaults** ([`config/config.example.yaml`](config/config.example.yaml)): el config canónico de la flota — bracket geometry, sensor mode, vision pipeline (SGBM, rectify), detection, tracking, counter, MQTT topics, buffer paths, status LED, etc. Todos los devices heredan estos valores.
- **Per-device** (`/etc/people-counter/config.yaml`): **única fuente de verdad en runtime**. El loader (`load_config`) lee SOLO este archivo — **no** mergea con el template. Tiene que estar completo (todas las keys requeridas, validadas en `_validate`). `config.example.yaml` es el TEMPLATE de provisioning: el operator lo copia, edita lo que cambia por unidad (`device.id`, `mounting_height_m`, counting zone/lines, MQTT endpoint + certs) y lo deja como config del device.
- **Cloud** (AWS IoT Device Shadow): settings de negocio — horarios operativos, factor de escala, toggles de habilitación. Se aplican vía `RUNTIME_SAFE_KEYS` sin reinicio.

`mounting_height_m` (per-device) alimenta el SGBM auto-tune (`num_disparities: auto` deriva el rango de disparidad por sitio) y el head-height gating del clasificador adulto/niño. La calibración estéreo es mount-independent (un único `.npz` factory sirve para mount 2.0–3.5m).

## Instalación en las instalaciones

Los tools de setup (`focus_assist.py`, `calibrate.py`, `preview.py`, `diagnose_depth.py`,
`diagnose_calibration.py`) leen `/etc/people-counter/config.yaml` para `vision.resolution` y
`vision.mounting_height_m` (focus_assist deriva el target distance de ahí).
Pasar `--resolution` / `--mount-height-m` explícitos solo en dev workstation
sin config per-device. `diagnose_calibration.py` además carga la calibración
`.npz` y captura a la resolución con la que se armaron sus rectify maps.

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

Abre UI en `http://people-counter.local:8080`. **Default: modo MAPA** — paseás
el board ChArUco por todo el cuadro y la herramienta acumula la nitidez máxima
por zona (grilla 3×3); el panel muestra la cobertura (verde = cubierta) y qué
zonas faltan. Cuando las 9 zonas están cubiertas en L y R, evalúa el mapa
completo. Así cada zona se mide con board real (no fondo) y el check de
simetría/corners por zona es honesto. `--static` vuelve al modo clásico de un
solo frame con el board fijo al centro. Flujo:
1. Pantalla "Comenzar" (posicionar board + activa audio)
2. Captura en vivo con barras de nitidez (frame actual) + acumulación del mapa
3. Peak tracker para ajustar el M12 sin pasarse del óptimo
4. Masking automático de zonas de bajo contraste
5. Pulso de audio adaptativo (tap corto que acelera según mejora el score; toggle en la UI)
6. Finalizar → reporte HTML (mapa acumulado) auto-abierto en nueva pestaña

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

Wizard de una sola corrida, **todo desde el browser** (terminal solo para logs).
**Default: modo BARRIDO (sweep)** — movés el board libremente y la herramienta
auto-selecciona los frames diversos que necesita (gate de novedad + quietud +
calidad) con un mapa de cobertura en vivo; mucho más fácil en espacios chicos /
luz difícil. **`--guided`** usa el modo clásico de 20 poses-silueta (más preciso
con buen espacio + luz; `--manual` lo hace captura-por-botón). Ambos terminan en
calibrar → verificar → reporte. El flujo guiado:
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
- `--guided`: modo clásico de 20 poses-silueta (el default es barrido libre); `--manual` lo hace captura-por-botón
- `--sweep-novelty F`: (modo barrido) umbral de novedad para aceptar un frame (default 0.12)
- `--dist-near-mm` / `--dist-mid-mm` / `--dist-far-mm`: distancias de las 3 bandas de poses (default 1000/2000/3000mm — lab protocol)
- `--legacy-pattern` / `--no-legacy-pattern`: enumeración de markers ChArUco (default `--legacy-pattern` matches calib.io)
- `--resume`: continúa una sesión previa (si no se pasa, descarta y arranca limpio)
- `--force-degenerate-coverage`: bypass del coverage critical block del modo guiado (en barrido el gate por-pose se omite solo — usa su propia cobertura de grilla/distancia/tilt)
- `--min-detect-rate F`: umbral del pre-calibration sanity gate (default 0.7)
- `--low-light`: preset PoC para cuarto chico/oscuro — afloja los gates de `assess_frame_quality` (exposure, blur, corner-sharp, L/R balance). El `.npz` resultante **NO es válido para producción**, solo valida que el wizard corra end-to-end

Subcomando aparte para restart limpio:
```bash
python scripts/calibrate.py reset --yes   # borra captures + session.json + .npz
```

**Protocolo de calibración recomendado**: el modo de **barrido libre** (default) es
el recomendado para la puesta en marcha — es el que se usó y validó en la calibración
del prototipo (RMS epipolar 0,12 px) y el más práctico en espacios chicos / luz
difícil; su cobertura la garantiza el mapa de zonas/distancia/tilt y se valida con el
RMS + `diagnose_calibration`. El modo `--guided` (20 poses deterministas) queda como
opción para QA comparable entre unidades cuando hay buen espacio + luz, p. ej. en un
eventual despliegue de flota.

### 3. Monitor de salud de calibración (post-calibración)

```bash
PYTHONPATH=. python3 scripts/diagnose_calibration.py
```

Responde "¿la calibración guardada sigue válida o hay que recalibrar?".
Rectifica un board ChArUco con la calibración del device (`.npz`) y mide el
**error epipolar** = disparidad vertical residual `|y_L − y_R|` de las
esquinas correspondientes. Un par bien calibrado da sub-píxel; >1px (default)
= la geometría estéreo cambió (bracket movido, lente corrido, drift térmico)
→ recalibrar. No necesita ground-truth (a diferencia de `diagnose_depth`, que
mide exactitud absoluta en metros y sí requiere distancia conocida). Check de
campo / periódico. El QC de bracket *pre*-calibración se retiró: medir la
geometría de un fisheye de 120° con modelo pinhole sin coeficientes de
distorsión produce yaw/offsets fantasma.

### 4. Vista previa en vivo del pipeline (diagnóstico, apagada por omisión)

El runtime puede servir un visor HTTP que streamea, como MJPEG, un compuesto de
tres paneles: **izquierda con overlay** (cajas, IDs de track, counting zone,
línea virtual, destellos +IN/+OUT), **derecha rectificada sin overlay** y **mapa
de profundidad**. Sirve para verificar en el lugar que el conteo se comporta
como se espera mientras alguien camina bajo las cámaras.

**Viene deshabilitada.** El parámetro es `--web-viewer-port`, cuyo valor por
omisión es `0` = apagado, y la unidad de systemd **no** lo pasa: en producción el
visor no arranca. Para una sesión de diagnóstico hay que habilitarlo a mano:

```bash
# Detener el servicio y correr el pipeline a mano con el visor en un puerto alto
sudo systemctl stop people-counter
cd /usr/src/people-counter
sudo -u pi PYTHONPATH=. python3 -m src.main \
    --config /etc/people-counter/config.yaml \
    --web-viewer-port 8080
# abrir http://<IP-del-dispositivo>:8080 y, al terminar, Ctrl-C +
sudo systemctl start people-counter
```

> ⚠️ **Expone imagen en vivo de personas, sin autenticación.** El flujo
> (`/stream`) y las métricas (`/stats`) responden a **cualquiera** que alcance
> el puerto: la única credencial del visor protege los botones de reinicio y
> apagado, no la imagen. Además el servidor bindea **todas** las interfaces
> (`0.0.0.0`) y el sistema no aplica ninguna restricción por dirección de
> origen — el aislamiento depende enteramente de la red donde esté el
> dispositivo. Usarlo **solo en una red controlada**, el tiempo que dure el
> diagnóstico, y apagarlo después. No dejarlo habilitado de forma permanente ni
> exponerlo a través de un reenvío de puertos.

El visor **no escribe nada a disco**: los fotogramas viven en una cola en
memoria que descarta los más viejos si el cliente va lento. Si el bind falla
(puerto ocupado, o el 80 sin `CAP_NET_BIND_SERVICE`) se registra una
advertencia y el pipeline sigue corriendo sin visor.

## Entrenamiento del detector

El modelo activo es `people-counter-detector`: YOLOv8n single-class
fine-tuneado para geometría cenital, compilado a HEF para Hailo-8L.
**No se usa el HEF stock del Hailo Model Zoo** — está entrenado en
COCO/CrowdHuman con vistas frontales que no transfieren a top-down.

Pipeline end-to-end (todo en `scripts/training/` — ver
[`scripts/training/README.md`](scripts/training/README.md) para el
walkthrough completo):

```
captura multi-site (motion-trigger)   →  sample_for_labeling.py (estratificado)
                                      →  X-AnyLabeling (local, bbox cabeza+hombros)
                                      →  labelme_to_yolo.py (dataset YOLO)
                                      →  Kaggle dataset privado (upload vía API)
                                      →  Notebook Kaggle T4 (~20 min)
                                      →  best.onnx  →  hailomz compile (Docker x86)
                                      →  HEF en la Pi
```

Pasos resumidos:

1. **Captura del pool de training**: `capture_mjpeg.py` multi-site con
   motion-trigger + background sampling. Filenames `_motion_` / `_bg_`
   para sampling balanceado downstream.
2. **Sampling estratificado para labeling**: `sample_for_labeling.py`
   arma un batch diverso por site (motion + bg balanceados) en una
   carpeta plana lista para X-AnyLabeling. `--exclude-manifest` saltea
   imgs ya usadas en otro batch (anti leak train/val).
3. **Active learning** (a partir de v2): `mine_active_learning.py`
   identifica frames informativos vía disagreement v1↔v_actual +
   uncertainty del modelo actual. Selecciona top-N en vez de muestrear
   al azar. v1→v2 con esta técnica subió mAP50 de 0.805 → 0.956 (la 2da
   ronda de AL —v3— bajó a 0.939; sweet spot en v2).
4. **Labeling con X-AnyLabeling** (local, no SaaS): bbox cabeza+hombros
   (convención canónica en `scripts/training/label_guide.md`).
   Export YOLO format directo desde la app.
5. **Conversión a dataset YOLO**: `labelme_to_yolo.py` toma la carpeta
   labeleada (`.json` formato labelme) y produce la estructura
   canónica de Ultralytics (`images/`, `labels/`, `data.yaml`).
   Imgs sin `.json` se tratan como background revisado (`.txt` vacío).
6. **Subir a Kaggle dataset privado** vía Kaggle CLI:
   `kaggle datasets version -p training_data/dataset_v_next ...`. El
   notebook consume el dataset desde ahí.
7. **Training en Kaggle T4**: `train_head_detector.ipynb`. Para iterar
   a v2/v3/... basta cambiar el slug del dataset + name del run.
   ~20 min por iteración. Descarga del modelo entrenado vía Kaggle CLI.
8. **Compilación HEF**: `hailomz compile` en Docker x86 Linux
   (WSL2 desde Windows). Receta exacta en
   [`scripts/training/README.md`](scripts/training/README.md) — flags
   críticos: `--classes 1` y `--end-node-names` obligatorio.
   Calibration set = 200 imgs muestreadas con `sample_for_calib.py`
   (sin leak del train).
9. **Deploy**: `scp` del `.hef` a `/usr/src/people-counter/models/`
   en la Pi + edición de `detection.model_path` en
   `/etc/people-counter/config.yaml`.

Bench tooling:

- `bench_detector.py` — inferencia + diff de reportes (baseline vs
  fine-tuned) sobre carpeta de frames. Subcomandos `bench` y `diff`.
- `eval_yolo.py` — corre un modelo sobre una carpeta y dumpea
  bboxes + summary para sanity-check rápido.
- `compare_detectors.py` — side-by-side de dos modelos sobre el mismo
  set (visual + métricas), valida que v_next no regresione casos del
  v_actual.
- `analyze_eval_summary.py` — análisis estadístico del output del
  eval (distribución de confidence, breakdown por site).

Defense-in-depth runtime (independiente del modelo):
- **Containment filter** post-NMS: descarta bbox chico contenido
  >50% en otro con mayor confidence (NMS por IoU no agarra esto en
  geometría cenital).
- **Cluster por centroide** (`cluster_distance_px: 150`): mergea
  detecciones multi-firing (cabeza + hombro como cajas distintas).
- **`StaticSuppressor`** (`cell_size_px: 30`, `window_seconds: 15`):
  suprime detecciones en celdas hot ≥70% de los últimos 15s — clutter
  estructural (maniquíes, ropa colgada, sombras) que sobrevive NMS.

## Estructura del repo

```
src/
├── vision/          # Captura estéreo (picamera2), calibración ChArUco, profundidad SGBM + WLS, detección YOLOv8n (Hailo + OpenCV), world_coords para altura de cabeza, static_suppressor + best_frame, report HTML
├── tracking/        # Tracker euclidiano 3D (Kalman) + ghost pool / ID adoption + contador por línea virtual / counting zone con net-balance + rescue cascade de 3 capas (ghost adoption + decisive Kalman cross at exit + death-emit-if-crossed con guards anti-FP)
├── wifi_ble/        # Captura de probes WiFi (nexmon + radiotap, hopping ponderado 1/6/11), scan BLE (bleak), fingerprint (IEs/manufacturer-data), hashing, dedup a hash groups con stitching de 4 reglas (seqnum + cross-protocol L2 + BLE anchoring + fingerprint), publisher de summaries 15min
├── mqtt/            # Cliente AWS IoT Core + buffer SQLite con replay
├── cloud/           # Lambdas: persist_event (IoT Rules → RDS Postgres via IAM auth, out of VPC) + ingest_pos_transaction (POS API → tabla pos_transactions) + query_aggregates (API read-only de agregados sobre las vistas)
├── status/          # Driver RGB LED + health probes + thread monitor que mapea HealthSignals → LedState
├── config/          # Loader (per-device strict + IoT Shadow overrides) + hardware (HardwareParams dataclass — sensor/lens/bracket/charuco/ae_lock leídos del config, hardware-agnostic)
├── web/             # Live preview HTTP/MJPEG (viewer + annotate) + panel admin con login (admin_auth). DESHABILITADO por default (--web-viewer-port 0): sirve imagen sin auth, se enciende solo para diagnóstico. Gateado por has_subscribers — counting es sync-deterministic independiente del viewer
├── telemetry.py     # Reporte periódico: CPU/Hailo temp, RAM, disco, uptime + canaries (track_stitching_ratio, ghost_adoption_count, death_emit_count, ambiguous_reject_count, wifi_ble_stitching_ratio)
└── main.py          # Orquestador del pipeline (captura → depth → detect → track → count → MQTT). Flag --no-mqtt para debug local sin AWS
tests/               # 1083 funciones de test en 48 archivos espejando src/ + tests/scripts/ para el wizard
scripts/
├── calibrate.py           # CLI: generate-board, capture, calibrate, verify, wizard, reset
│                          # wizard = pipeline end-to-end browser-driven. Default
│                          # BARRIDO (sweep): cobertura por novedad/quietud + mapa en vivo.
│                          # --guided = 20 poses-silueta clásicas. Ambos: ground-truth
│                          # en UI + reporte HTML. Flags: --guided, --sweep-novelty,
│                          # --meter, --lock-ae, --low-light
├── focus_assist.py        # Asistente de foco browser-driven. Default MAPA: paseás el
│                          # board, acumula nitidez máx por zona (grilla 3×3) + cobertura;
│                          # evalúa el mapa completo. --static = un solo frame al centro.
│                          # Peak tracker, masking, L/R parity, reporte auto-open.
│                          # Flags: --static, --map-coverage-min, --meter, --low-light
├── preview.py             # Preview en vivo browser-driven (start overlay, header).
│                          # Side-by-side L|R con grid + crosshair. Para apuntar el bracket
│                          # antes de correr foco / calib. Flag --meter centre/spot
├── diagnose_depth.py      # Validación de profundidad: 5 zonas con tags de confianza
│                          # (✓ Coincide / ● Otro plano / ⚠ SGBM falló). Verdict solo
│                          # por error del centro. Flags: --meter, --lock-ae
├── diagnose_calibration.py # Salud de calibración post-cal: rectifica con la .npz y mide
│                          # error epipolar L↔R. Veredicto OK / recalibrar (>1px). Sin
│                          # ground-truth — check de campo/periódico de drift del bracket.
├── preflight.py           # Chequeo pre-install (cámaras + Hailo + hardware)
├── counting_zone_picker.py          # Seleccionador de counting zone + línea virtual
├── provision.py           # Provisioning + disaster recovery: create, deploy, harvest, reprovision, list
├── migrate_historical.py  # Loader CSV→staging batcheado (commits incrementales) para histórico AGREGADO; corre el transform de infra/sql/migrate_historical_rollups.example.sql
├── deploy_lambda.ps1      # Packaging del Lambda persist_event (psycopg binary + handler). Versión .sh tambien disponible.
├── download_model.py      # Descarga YOLOv8n HEF (Hailo Model Zoo) o ONNX (ultralytics) — usar el HEF fine-tuneado de scripts/training/, no el stock
├── capture_baseline_frames.py  # Captura frames rectificados de la Pi para validation bench (no training)
├── training/
│   ├── README.md              # Walkthrough end-to-end del pipeline de detector (X-AnyLabeling + active learning + Kaggle)
│   ├── label_guide.md         # Convención canónica de labeling (bbox cabeza+hombros) + setup X-AnyLabeling
│   ├── train_head_detector.ipynb  # Notebook Kaggle T4 (consume Kaggle dataset privado, ~20 min)
│   ├── sample_for_labeling.py # Muestreo estratificado del pool de captures para X-AnyLabeling (anti leak train/val via --exclude-manifest)
│   ├── mine_active_learning.py # Active learning v_next: minado por disagreement v1↔v_actual + uncertainty
│   ├── labelme_to_yolo.py     # Conversión de labels X-AnyLabeling (.json labelme) a dataset YOLO + data.yaml
│   ├── capture_mjpeg.py       # Captura multi-site de streams MJPEG (random-interval + motion-trigger)
│   ├── record_clips.py        # Grabación continua de clips MP4 multi-site (validation E2E con tracker)
│   ├── sample_for_calib.py    # Sampling balanceado para el calib set del QAT de Hailo
│   ├── bench_detector.py      # Bench de inferencia + diff de reportes (baseline vs fine-tuned)
│   ├── compare_detectors.py   # Side-by-side de dos modelos (visual + métricas) — anti regresión
│   ├── analyze_eval_summary.py # Análisis estadístico del eval (distribución conf, breakdown por site)
│   ├── eval_yolo.py           # Corre un modelo YOLO sobre una carpeta de frames
├── verify_hardware.py     # Verificación de hardware
└── setup_device.sh        # Setup automático del dispositivo (pasos 4-10)
config/
├── config.example.yaml           # Defaults canónicos para toda la flota (bracket, sensor, vision, detection, tracking, counter, mqtt, etc.). Cualquier key ausente del per-device cae a los valores de acá.
├── people-counter.service        # Servicio systemd principal
├── people-counter-reset.*        # Timer de reset diario de dedup (04:00)
└── logrotate.conf                # Rotación de logs
infra/
├── README.md                              # Walkthrough del deploy + verificación E2E
├── cloudformation/people-counter.yaml     # Stack completo de AWS
├── deploy.ps1                             # Orquestador 6 fases (RDS + IoT + Lambdas + ECR + SES + certs ACM + ECS Fargate + ALB + Grafana datasource/dashboards/alerts). -StartFromPhase para resumir
├── sql/bootstrap.sql                      # Schema (count_events / wifi_ble_events / telemetry / pos_transactions + funciones SQL height_class() y rssi_class() + vistas cartesian + lambda_writer con rds_iam)
├── sql/migrations/                        # Migraciones incrementales data-preserving sobre la RDS viva (squasheadas a bootstrap.sql una vez aplicadas)
└── sql/migrate_historical_rollups.example.sql  # Template staging→tablas base rollup_* para histórico AGREGADO (no es migración de schema)
docs/
├── setup_guide.md                # Ensamblaje de hardware + setup RPi (13 pasos)
├── lab_calibration_guide.md      # Protocolo de foco + calibración en lab (universal para la flota)
└── operator_guide.md             # Guía para el operador en las instalaciones (foco → calibración → verificación)
training_data/                    # Workspace gitignoreado de training (sites.yaml inline + captures rectificadas)
debug/                            # Drop-zone gitignoreado para reportes, capturas y logs de test
```

## Referencias clave

- [CLAUDE.md](CLAUDE.md) — Documentación completa de arquitectura para Claude Code
- [docs/setup_guide.md](docs/setup_guide.md) — Guía de ensamblaje de hardware + setup RPi
- [config/config.example.yaml](config/config.example.yaml) — Configuración anotada con estrategia

## Artefactos del modelo

Los pesos del detector **no se versionan**: se distribuyen como adjuntos de la
versión etiquetada del repositorio. El manifiesto de sumas de verificación sí
está versionado, de modo que se puede comprobar que el archivo descargado es
exactamente el que se usó.

| Archivo | Tamaño | Qué es |
|---|---:|---|
| `people-counter-detector.pt` | 6.242.019 B | Pesos afinados en formato PyTorch, tal como los produjo el entrenamiento. Es el punto de partida de toda la cadena y lo que permite re-exportar o re-compilar |
| `people-counter-detector.onnx` | 12.265.214 B | Exportación ONNX de esos mismos pesos, precisión completa. Es el artefacto que permite reproducir la inferencia en cualquier equipo, sin acelerador |
| `people-counter-detector.hef` | 9.390.809 B | Compilación cuantizada para el acelerador Hailo-8L. Es el artefacto que corre en el dispositivo |

La cadena `.pt` → `.onnx` → `.hef` está verificada: el ONNX publicado es
byte-idéntico al que produjo la misma corrida de entrenamiento que el `.pt`, y
ambos devuelven la misma inferencia sobre una entrada fija (coincidencia del
score máximo hasta 2·10⁻⁷ y de las coordenadas de la caja hasta 0,06 px).

### Cómo obtenerlos

1. Descargar los tres archivos desde la sección *Releases* del repositorio, en
   la versión etiquetada correspondiente.
2. Colocar los tres en `models/training/people-counter-detector/`, creando el
   directorio si no existe:

   ```bash
   mkdir -p models/training/people-counter-detector
   # mover aquí los tres archivos descargados
   ```

3. Verificar las sumas **antes de ejecutar nada**, desde la raíz del
   repositorio:

   ```bash
   sha256sum -c models/CHECKSUMS.txt
   ```

   Salida esperada: una línea `OK` por cada uno de los tres archivos y código
   de salida 0. En
   PowerShell, sin `sha256sum` disponible:

   ```powershell
   Get-FileHash models\training\people-counter-detector\people-counter-detector.onnx -Algorithm SHA256
   ```

Una vez colocados y verificados, la caracterización del detector se reproduce
con `python scripts/analysis/eval_detector_valset.py` (requiere además el
conjunto de validación, que no se distribuye — ver más abajo).

### Modelo base

El afinado parte de **YOLOv8n con pesos COCO** de Ultralytics, que **no se
redistribuye** en este repositorio ni en sus adjuntos. Lo descarga
automáticamente la biblioteca `ultralytics` al invocar `YOLO("yolov8n.pt")`,
desde los adjuntos de publicación del propio proyecto Ultralytics. El archivo
obtenido de ese modo durante el desarrollo tiene 6.549.796 B y suma SHA-256:

```
f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
```

Se consigna para que un tercero pueda confirmar que partió del mismo archivo.
La versión exacta de la publicación de origen no está fijada por este
repositorio: la resuelve `ultralytics` en el momento de la descarga.

### Material que no se distribuye

El conjunto de imágenes de entrenamiento y validación, sus anotaciones y las
matrices de calibración por sucursal **no forman parte del repositorio ni de
los adjuntos**: proceden de cámaras instaladas en locales de la organización y
contienen personas. Los resultados que dependen de ese material se reportan en
el trabajo escrito con la evidencia bruta que sí se publica, bajo
`docs/validacion/`.

## Despliegue de la infraestructura en la nube

El procedimiento completo de despliegue en AWS —plantilla de CloudFormation,
las seis fases del script de orquestación, los pasos manuales de DNS y el
desmantelamiento— está documentado en
[`infra/README.md`](infra/README.md). El esquema de la base de datos y sus
vistas se describen en [`docs/database_schema.md`](docs/database_schema.md), y
la configuración de alertas en [`docs/alerting.md`](docs/alerting.md).

Reproducir el sistema completo no es necesario para evaluar el trabajo: la
suite de pruebas y los bancos de análisis corren sin la nube.

## Licencia

Este repositorio se distribuye bajo una **licencia de uso académico con
derechos reservados**. Se permite descargar, examinar, compilar y ejecutar el
contenido con la finalidad de evaluar, verificar o reproducir los resultados
del Trabajo Final de Grado, así como para estudio e investigación sin fines
comerciales, conservando la nota de licencia y atribuyendo la autoría.

Quedan reservados todos los demás derechos: no se autoriza el uso comercial,
la redistribución del código o de obras derivadas, ni su incorporación a
productos o servicios de terceros sin autorización previa y por escrito del
titular.

El texto completo está en [LICENSE](LICENSE), e incluye la ausencia de
garantía y el alcance del material no incluido en el repositorio.

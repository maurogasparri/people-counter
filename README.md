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
| Waveshare PoE HAT (H) | 25.5W, 802.3at | Alimentación por dupont (no stackeado) |
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

La config cloud usa una estrategia de **caché local de shadow**: al bootear, `main.py` lee un archivo `.shadow.json` si existe (actualizado por un proceso de fondo o en el boot anterior). Suscripción delta en vivo vía AWS IoT Shadow planificada post-MVP.

## Estado del proyecto

| Área | Estado | Detalles |
|------|--------|---------|
| Código fuente | 22 módulos | Todos los módulos implementados y validados en hardware |
| Tests | 180/180 pasando | Visión, tracking, MQTT, WiFi/BLE, config, cloud, main, provision |
| Config | Local + Cloud | YAML (hardware) + IoT Shadow (negocio) |
| Hardware | Ensamblado + verificado | RPi5 + Hailo-8L (fw 4.23, PCIe Gen 3) + 2x Arducam IMX708 |
| Captura estéreo | Validada | picamera2 en RPi5, ambas cámaras funcionando |
| Detección | Validada | YOLOv8n HEF en Hailo-8L, VDevice persistente con scheduling ROUND_ROBIN |
| Calibración | Validada | Pinhole (CALIB_RATIONAL_MODEL), baseline 140mm por diseño. ChArUco 9x6/45mm/33mm/DICT_4X4_100 A3 (en `calibration/`). Wizard guiado con ghost silueta, tolerance preset (`loose`/`normal`/`strict`), auto-open de reporte HTML. Validada vía chequeo de profundidad en 5 zonas con umbrales PASS/FAIL |
| Asistente de foco | Validado | `focus_assist.py` con UI web (header + side panel, auto-open de reporte). Captura a 2304×1296 (modo full-FOV binned del IMX708), derivación automática del target de foco desde `--mount-height-m`, peak tracker y masking de zonas de bajo contraste |
| WiFi probe | Validada | nexmon + airmon-ng + scapy, probe requests capturadas en RPi5 |
| BLE scan | Validado | bleak, 343 adverts, 8 dispositivos únicos, dedup + turn-in rate |
| Infra cloud | CloudFormation | IoT Core, Timestream, DynamoDB, Lambda |
| Deployment | Listo | provision.py, servicios systemd (pipeline + wifi-monitor + reset diario), logrotate |
| Guía de setup | Completa | Guía de 13 pasos desde microSD hasta overlayfs (docs/setup-guide.md) |

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
| opencv-contrib-python, numpy, paho-mqtt, pyyaml, scapy, bleak | `pip install -e ".[dev]"` | Multiplataforma, funciona en máquinas de desarrollo |
| picamera2, libcamera | `apt` (python3-picamera2) | Solo RPi, provisto por RPi OS Trixie |
| hailo_platform | `apt` (hailort + hailort-pcie-driver + python3-hailort) | Solo RPi, requiere Hailo-8L + PCIe |
| aircrack-ng, nexmon | `apt` + paquetes `.deb` | Solo RPi, WiFi monitor mode |

En máquinas de desarrollo (Windows/Mac/Linux), `pip install -e ".[dev]"` es suficiente para correr tests. Los paquetes del sistema RPi solo se necesitan en el dispositivo target — ver [docs/setup-guide.md](docs/setup-guide.md) para la instalación completa.

## Configuración

El sistema usa una estrategia de doble config:

- **Local** (`config/config.yaml`): settings intrínsecos al hardware — IDs de cámara, archivo de calibración, parámetros SGBM, path del modelo, certificados MQTT
- **Cloud** (AWS IoT Device Shadow): settings del negocio — horarios operativos, factor de escala, toggles de habilitación

Ver [`config/config.example.yaml`](config/config.example.yaml) para el config anotado completo.

## Instalación en sitio

Los tools de setup (`focus_assist.py`, `calibrate.py`) son **standalone** — no leen
`config.yaml`, todo se pasa por CLI. Esto permite correrlos durante la
instalación inicial antes de que exista config.

### 1. Ajuste de foco

```bash
sudo PYTHONPATH=. python3 scripts/focus_assist.py --mount-height-m 3.0
```

Abre UI en `http://people-counter.local:8080`. Flujo:
1. Pantalla "Comenzar" (posicionar board + activa audio)
2. Captura en vivo con barras de nitidez central + uniformidad + simetría L/R
3. Peak tracker para ajustar el M12 sin pasarse del óptimo
4. Masking automático de zonas de bajo contraste
5. Audio TTS lee los hints (opcional, toggle en la UI)
6. Finalizar → reporte HTML auto-abierto en nueva pestaña

El board puede estar a cualquier distancia dentro del rango de cabezas
derivado del mount height (por default `3.0m` → target `1.15–1.80m`).

### 2. Calibración estéreo

```bash
sudo PYTHONPATH=. python3 scripts/calibrate.py wizard \
    --device-id DEV-001 \
    --output /etc/people-counter/calibration.npz \
    --tolerance strict
```

Wizard de una sola corrida, **todo desde el browser** (terminal solo para logs):
1. Pre-flight (puerto, disco, backup)
2. Pantalla "Comenzar" (activa audio del browser)
3. Captura guiada con ghost silueta (20 poses), audio TTS opcional, hints en cm
4. Anuncio por voz de la distancia objetivo de cada pose ("A 100cm de la cámara")
5. Bootstrap de intrínsecos tras las primeras 6 capturas
6. Calibración estéreo con CALIB_RATIONAL_MODEL
7. Residuales por par + chequeo de baseline (estimada, no medida físicamente)
8. Confirmación UI si la diversidad es limitada (botones Continuar/Cancelar)
9. Ground-truth opcional con input de distancia en la UI (botón Saltear)
10. Reporte HTML auto-abierto en nueva pestaña

Presets de tolerancia: `loose` (PoC / board chico), `normal` (default, tuned
para A3 final), `strict` (producción, recomendado con board sobre trípode).

Flags de debug útiles:
- `--min-captures N`: baja el mínimo para terminar calibración (default 15)
- `--align-tol-{loose,tight}-px`: overrides finos de tolerancia
- `--resume`: continúa una sesión previa (si no se pasa, descarta y arranca limpio)

## Estructura del repo

```
src/
├── vision/          # Captura estéreo (picamera2), calibración, profundidad SGBM, detección YOLOv8n (Hailo + OpenCV)
├── tracking/        # Tracker euclidiano 3D + contador por línea virtual
├── wifi_ble/        # Captura de probes WiFi, scan BLE, hashing de MAC, dedup (L1+L2)
├── mqtt/            # Cliente AWS IoT Core + buffer SQLite
├── cloud/           # Lambda dedup L3 (inter-cámara)
├── config/          # Carga de YAML + merge con IoT Shadow
└── main.py          # Orquestador del pipeline (17 tests)
tests/               # 180 tests espejando la estructura de src/
scripts/
├── calibrate.py      # CLI: generate-board, capture, calibrate, verify, wizard
│                     # wizard = pipeline end-to-end con UI web, ghost silueta,
│                     # start overlay, tolerance preset, reporte auto-open
├── focus_assist.py   # Asistente de foco guiado, UI web con peak tracker,
│                     # masking de zonas de bajo contraste, reporte auto-open
├── diagnose_depth.py # Validación de profundidad: análisis de 5 zonas + PASS/FAIL vs distancia conocida
├── provision.py      # Provisioning de dispositivos: create, deploy, list
├── download_model.py # Descarga YOLOv8n HEF/ONNX
├── verify_hardware.py # Script de verificación de hardware
└── setup_device.sh   # Setup automático del dispositivo (pasos 4-10)
config/
├── config.example.yaml       # Config anotado con documentación de estrategia
├── people-counter.service    # Servicio systemd (auto-restart, hardening)
├── people-counter-reset.*    # Timer de reset diario de dedup (04:00)
└── logrotate.conf            # Rotación de logs
infra/
└── cloudformation/people-counter.yaml  # Stack completo de AWS
docs/
└── setup-guide.md            # Ensamblaje de hardware + setup RPi (13 pasos)
```

## Referencias clave

- [CLAUDE.md](CLAUDE.md) — Documentación completa de arquitectura para Claude Code
- [docs/setup-guide.md](docs/setup-guide.md) — Guía de ensamblaje de hardware + setup RPi
- [config/config.example.yaml](config/config.example.yaml) — Configuración anotada con estrategia

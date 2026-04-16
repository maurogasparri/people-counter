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
| Calibración | Validada | Pinhole (CALIB_RATIONAL_MODEL), baseline 142.8mm. ChArUco 11x7/35mm/26mm/DICT_5X5_100 A3 (en `calibration/`). Validada vía chequeo de profundidad en 5 zonas con umbrales PASS/FAIL |
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
├── calibrate.py      # CLI: generate-board, capture (headless), calibrate, verify
├── focus_assist.py   # Asistente de foco guiado con preview HTTP
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

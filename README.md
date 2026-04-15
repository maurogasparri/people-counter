# people-counter

Low-cost people counting system for retail stores, based on stereo vision and edge AI.

**TFG (Trabajo Final de Grado)** — Lic. en Administracion de Infraestructura Tecnologica, Universidad Siglo 21.

## What it does

- **Counts people** entering and exiting a store in real-time using stereo camera depth + YOLOv8n on a Hailo-8L accelerator
- **Detects exterior foot traffic** via passive WiFi probe request and BLE advertising capture
- **Classifies traffic** with dual RSSI thresholds: passersby (-75 dBm) vs shoppers (-55 dBm), calculating Turn In Rate
- **Deduplicates** WiFi/BLE signals across protocols (L1+L2 on device) and across cameras in the same store (L3 in Lambda)
- **Streams metadata** to AWS via MQTT with local SQLite buffering for offline resilience
- **Respects operating hours** via AWS IoT Device Shadow (cloud-pushed site config)

## Hardware

Each unit costs ~USD 416 and consists of:

| Component | Spec | Role |
|-----------|------|------|
| Raspberry Pi 5 | 4GB RAM, ARM Cortex-A76 | Main SBC |
| Hailo-8L | 13 TOPS, M.2 via Raspberry Pi AI HAT+ | Neural inference |
| 2x Arducam IMX708 | 12MP, 120 HFOV, M12 lens, CSI, 14cm baseline | Stereo pair |
| Waveshare PoE HAT (H) | 25.5W, 802.3at | Power via dupont (no stack) |
| MicroSD | 32GB | Boot + storage |

## Architecture

```
Edge Device (per store door)          AWS Cloud
+--------------------------+         +-------------------------+
| Capture → Rectify → SGBM |  MQTT   | IoT Core → Timestream   |
| YOLOv8n → Track → Count  |--TLS-->| Lambda → DynamoDB       |
| WiFi/BLE → Hash → Dedup  |  QoS1  | API GW → QuickSight     |
| SQLite buffer (72h)      |         | CloudWatch + S3 OTA     |
+--------------------------+         +-------------------------+
```

### Edge processes

The device runs three independent systemd services:

| Service | Process | What it does |
|---------|---------|-------------|
| `people-counter.service` | `src/main.py` | Vision pipeline: capture → rectify → depth → detect → track → count → MQTT |
| `wifi-monitor.service` | `airmon-ng` | Puts WiFi into monitor mode for probe request capture |
| `people-counter-reset.timer` | Daily at 04:00 | Resets dedup counters and counting totals for the new business day |

WiFi/BLE probing runs as a separate service because it requires exclusive WiFi hardware access (monitor mode). Vision and WiFi never contend for resources. Both publish independently to MQTT, and L3 dedup across cameras happens in the cloud (Lambda).

Cloud config uses a **local shadow cache** strategy: on boot, `main.py` reads a `.shadow.json` file if present (updated by a background process or on previous boot). Live delta subscription via AWS IoT Shadow is planned post-MVP.

## Project status

| Area | Status | Details |
|------|--------|---------|
| Source code | 22 modules | All modules implemented and validated on hardware |
| Tests | 180/180 passing | Vision, tracking, MQTT, WiFi/BLE, config, cloud, main, provision |
| Config | Local + Cloud | YAML (hardware) + IoT Shadow (business) |
| Hardware | Assembled + verified | RPi5 + Hailo-8L (fw 4.23, PCIe Gen 3) + 2x Arducam IMX708 |
| Stereo capture | Validated | picamera2 on RPi5, both cameras working |
| Detection | Validated | YOLOv8n HEF on Hailo-8L, persistent VDevice with ROUND_ROBIN scheduling |
| Calibration | Validated | Pinhole (CALIB_RATIONAL_MODEL), baseline 142.8mm. ChArUco 11x7/35mm/26mm/DICT_5X5_100 A3 (in `calibration/`). Validated via 5-zone depth check with PASS/FAIL thresholds |
| WiFi probe | Validated | nexmon + airmon-ng + scapy, probe requests captured on RPi5 |
| BLE scan | Validated | bleak, 343 adverts, 8 unique devices, dedup + turn-in rate |
| Cloud infra | CloudFormation | IoT Core, Timestream, DynamoDB, Lambda |
| Deployment | Ready | provision.py, systemd services (pipeline + wifi-monitor + daily reset), logrotate |
| Setup guide | Complete | 12-step guide from MicroSD to overlayfs (docs/setup-guide.md) |
| TFG document | 90+ pages | 27 references, 13 tables, 6 figures |

## Quick start

```bash
git clone https://github.com/maurogasparri/people-counter.git
cd people-counter
pip install -e ".[dev]"
pytest
```

### Dependencies

| Package | Install via | Notes |
|---------|------------|-------|
| opencv-contrib-python, numpy, paho-mqtt, pyyaml, scapy, bleak | `pip install -e ".[dev]"` | Cross-platform, works on dev machines |
| picamera2, libcamera | `apt` (python3-picamera2) | RPi only, provided by RPi OS Trixie |
| hailo_platform | `apt` (hailo-all) | RPi only, requires Hailo-8L + PCIe |
| aircrack-ng, nexmon | `apt` + `.deb` packages | RPi only, WiFi monitor mode |

On development machines (Windows/Mac/Linux), `pip install -e ".[dev]"` is sufficient to run tests. RPi system packages are only needed on the target device — see [docs/setup-guide.md](docs/setup-guide.md) for full installation.

## Configuration

The system uses a dual-config strategy inspired by FootfallCam:

- **Local** (`config/config.yaml`): hardware-intrinsic settings — camera IDs, calibration file, SGBM params, model path, MQTT certs
- **Cloud** (AWS IoT Device Shadow): business-driven settings — operating hours, scaling factor, enable/disable toggles

See [`config/config.example.yaml`](config/config.example.yaml) for the full annotated config.

## Repo structure

```
src/
├── vision/          # Stereo capture (picamera2), calibration, SGBM depth, YOLOv8n detection (Hailo + OpenCV)
├── tracking/        # 3D Euclidean tracker + virtual line counter
├── wifi_ble/        # WiFi probe capture, BLE scan, MAC hashing, dedup (L1+L2)
├── mqtt/            # AWS IoT Core client + SQLite buffer
├── cloud/           # Lambda dedup L3 (inter-camera)
├── config/          # YAML loader + IoT Shadow merge
└── main.py          # Pipeline orchestrator (17 tests)
tests/               # 180 tests mirroring src/ structure
scripts/
├── calibrate.py      # CLI: generate-board, capture (headless), calibrate, verify
├── focus_assist.py   # Live focus scoring with HTTP preview
├── diagnose_depth.py # Depth validation: 5-zone analysis + PASS/FAIL vs known distance
├── provision.py      # Device provisioning: create, deploy, list
├── download_model.py # Download YOLOv8n HEF/ONNX
├── verify_hardware.py # Hardware verification script
└── setup_device.sh   # Automated device setup (steps 4-9)
config/
├── config.example.yaml       # Annotated config with strategy docs
├── people-counter.service    # systemd service (auto-restart, hardening)
├── people-counter-reset.*    # Daily dedup reset timer (04:00)
└── logrotate.conf            # Log rotation
infra/
└── cloudformation/people-counter.yaml  # Full AWS stack
docs/
└── setup-guide.md            # Hardware assembly + RPi setup (12 steps)
```

## Key references

- [CLAUDE.md](CLAUDE.md) — Full architecture documentation for Claude Code
- [docs/setup-guide.md](docs/setup-guide.md) — Hardware assembly + RPi setup guide
- [config/config.example.yaml](config/config.example.yaml) — Annotated configuration with strategy

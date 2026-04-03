# people-counter

Low-cost people counting system for retail stores, based on stereo vision and edge AI.

**TFG (Trabajo Final de Grado)** — Lic. en Administración de Infraestructura Tecnológica, Universidad Siglo 21.

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
| Hailo-8L | 13 TOPS, M.2 via PoE HAT+ | Neural inference |
| 2× OV5647 | 160° fisheye, CSI, 14cm baseline | Stereo pair |
| PoE HAT+ | Waveshare, 30W | Power + network |
| MicroSD | 32GB, overlayfs | Boot + storage |

## Architecture

```
Edge Device (per store door)          AWS Cloud
┌──────────────────────────┐         ┌─────────────────────────┐
│ Capture → Rectify → SGBM │  MQTT   │ IoT Core → Timestream   │
│ YOLOv8n → Track → Count  │──TLS──→│ Lambda → DynamoDB       │
│ WiFi/BLE → Hash → Dedup  │  QoS1  │ API GW → QuickSight     │
│ SQLite buffer (72h)      │         │ CloudWatch + S3 OTA     │
└──────────────────────────┘         └─────────────────────────┘
```

## Project status

| Area | Status | Details |
|------|--------|---------|
| Source code | ✅ 22 files | All modules implemented |
| Tests | ✅ 180/180 passing | Vision, tracking, MQTT, WiFi/BLE, config, cloud, main, provision |
| Config | ✅ Local + Cloud | YAML (hardware) + IoT Shadow (business) |
| Hardware | ✅ Assembled + verified | RPi5 + Hailo-8L (fw 4.23, PCIe Gen 3) + 2× OV5647 |
| Stereo capture | ✅ Validated | picamera2 on RPi5, both cameras working |
| Detection | ✅ Validated | YOLOv8n HEF on Hailo-8L, person detected at 91% confidence |
| Calibration | 🔧 Ready to run | `scripts/calibrate.py` headless — pending ChArUco capture |
| WiFi probe | ✅ Validated | nexmon + airmon-ng + scapy, probe requests captured on RPi5 |
| BLE scan | ✅ Validated | bleak, 343 adverts, 8 unique devices, dedup + turn-in rate |
| Cloud infra | ✅ CloudFormation | IoT Core, Timestream, DynamoDB, Lambda |
| Deployment | ✅ Ready | provision.py, systemd service, logrotate, daily reset timer |
| TFG document | ✅ 90+ pages | 27 references, 13 tables, 6 figures |

## Quick start

```bash
git clone https://github.com/maurogasparri/people-counter.git
cd people-counter
pip install -e ".[dev]"
pytest
```

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
├── calibrate.py     # CLI: generate-board, capture (headless), calibrate, verify
├── provision.py     # Device provisioning: create, deploy, list
└── download_model.py # Download YOLOv8n HEF/ONNX
config/
├── config.example.yaml       # Annotated config with strategy docs
├── people-counter.service    # systemd service (auto-restart, hardening)
├── people-counter-reset.*    # Daily dedup reset timer (04:00)
└── logrotate.conf            # Log rotation
infra/
└── cloudformation/people-counter.yaml  # Full AWS stack
docs/
└── setup-guide.md            # Hardware assembly + RPi setup (14 steps)
```

## Key references

- [CLAUDE.md](CLAUDE.md) — Full architecture documentation for Claude Code
- [docs/setup-guide.md](docs/setup-guide.md) — Hardware assembly + RPi setup guide
- [config/config.example.yaml](config/config.example.yaml) — Annotated configuration with strategy

## License

MIT — Copyright (c) 2026 Mauro Gasparri

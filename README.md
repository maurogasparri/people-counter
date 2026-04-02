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
| Source code | ✅ 20 files, 2684 lines | All modules implemented |
| Tests | ✅ 129/129 passing | Vision, tracking, MQTT, WiFi/BLE, config, cloud |
| Config | ✅ Local + Cloud | YAML (hardware) + IoT Shadow (business) |
| Hardware | 📦 In box | RPi5 + Hailo-8L + cameras, unboxing pending |
| Calibration | 🔧 Code ready | `scripts/calibrate.py` — needs real cameras |
| WiFi/BLE capture | 🔧 Stub | Needs nexmon on RPi5 + real hardware |
| Cloud infra | 🔧 Stub | CloudFormation templates pending |
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
├── vision/          # Stereo capture, calibration, SGBM depth, YOLOv8n detection
├── tracking/        # 3D Euclidean tracker + virtual line counter
├── wifi_ble/        # MAC hashing, dedup (L1 intra + L2 cross-protocol)
├── mqtt/            # AWS IoT Core client + SQLite buffer
├── cloud/           # Lambda dedup L3 (inter-camera)
├── config/          # YAML loader + IoT Shadow merge
└── main.py          # Pipeline orchestrator
tests/               # 129 tests mirroring src/ structure
scripts/
└── calibrate.py     # CLI: generate-board, capture, calibrate, verify
config/
└── config.example.yaml  # Annotated config with strategy docs
```

## Key references

- [CLAUDE.md](CLAUDE.md) — Full architecture documentation for Claude Code
- [config/config.example.yaml](config/config.example.yaml) — Annotated configuration with strategy

## License

MIT — Copyright (c) 2026 Mauro Gasparri

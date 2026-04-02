# CLAUDE.md — People Counter Edge System

## Project Overview

Low-cost people counting system for retail stores. Stereo vision + edge AI + passive WiFi/BLE traffic detection. Replaces a commercial Axis/Cognimatics system across 30 stores in Argentina.

**This is a real production project.** Code quality, error handling, and resilience are critical. Devices run unattended 12h/day, 363 days/year.

## Architecture

```
┌─────────────────────────────────────────┐
│           Edge Device (per store)        │
│  RPi5 4GB + Hailo-8L 13T + 2×OV5647    │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  Vision   │  │ WiFi/BLE │  │  MQTT  │ │
│  │  Module   │  │  Module  │  │ Client │ │
│  │           │  │          │  │        │ │
│  │ Stereo →  │  │ Monitor  │  │ QoS 1  │ │
│  │ YOLOv8n → │  │ Probe →  │  │ Buffer │ │
│  │ Track →   │  │ Hash →   │  │ SQLite │ │
│  │ Count     │  │ Dedup    │  │        │ │
│  └──────────┘  └──────────┘  └────────┘ │
└──────────────────┬───────────────────────┘
                   │ MQTT (TLS + X.509)
                   ▼
┌─────────────────────────────────────────┐
│              AWS Cloud                   │
│                                          │
│  IoT Core → Timestream (time series)     │
│           → Lambda (WiFi/BLE dedup)      │
│           → DynamoDB (dedup hashes)      │
│           → API Gateway → QuickSight     │
└─────────────────────────────────────────┘
```

## Hardware per Unit (BOM ~USD 416)

- Raspberry Pi 5 4GB — main SBC
- Hailo-8L 13 TOPS via PoE M.2 HAT+ (Waveshare) — neural accelerator
- 2× OV5647 160° fisheye cameras via CSI — stereo pair, 14cm baseline
- PoE 30W injector — power via existing Ethernet cabling
- PETG 3D-printed enclosure — ceiling mount on metal L-bracket

## Key Technical Decisions

### Vision Pipeline
- **Stereo calibration**: ChArUco pattern (A4), OpenCV `calibrateCamera` + `stereoCalibrate`. Store intrinsics/extrinsics as `.npz` per device.
- **Rectification**: Precomputed maps applied per frame pair. Fisheye undistort via `cv2.fisheye`.
- **Depth**: Semi-Global Block Matching (`cv2.StereoSGBM`) on rectified pair.
- **Detection**: YOLOv8n compiled to HEF via Hailo Model Zoo. Run on Hailo-8L at 30+ FPS. Use `hailo_platform` Python SDK.
- **Tracking**: Euclidean distance tracker in 3D space (x, y, depth). Unique ID per trajectory.
- **Counting**: Virtual line in depth coordinates. Crossing direction = ingress/egress event. Publish immediately via MQTT.

### WiFi/BLE Capture
- **WiFi**: CYW43455 in monitor mode via nexmon. Capture probe requests on 2.4 AND 5 GHz. **WiFi is EXCLUSIVE for probing — network connectivity is Ethernet only.**
- **BLE**: Same CYW43455, passive advertising on channels 37/38/39.
- **Hashing**: SHA-256 truncated to 16 bytes on every MAC before storage. Never store raw MACs.
- **Dedup L1 (intra-protocol)**: SQLite set of hashes per day per protocol. Reset at business day start.
- **Dedup L2 (cross-protocol)**: WiFi + BLE within 2s window AND RSSI delta ≤ 5dBm → unified hash.
- **Dedup L3 (inter-camera)**: Cloud Lambda + DynamoDB per store_id + date.

### Communication
- **MQTT**: AWS IoT Core, X.509 client certs, QoS 1.
- **Counting events**: Real-time on each crossing.
- **WiFi/BLE summaries**: Every 15 min.
- **Telemetry**: Every 5 min (CPU temp, Hailo temp, RAM, disk, uptime).
- **SQLite buffer**: All events buffered locally. Replay on reconnect. Mark sent only after PUBACK.

### Cloud (AWS)
- IoT Core: MQTT broker + rules engine.
- Timestream: Counting time series. 7-day memory, magnetic for history.
- Lambda: WiFi/BLE dedup across cameras per store.
- DynamoDB: Dedup hash table, partitioned by store_id + date.
- API Gateway: REST API for queries.
- QuickSight: Dashboards.

## Code Conventions

- **Language**: Python 3.11+ (RPi OS Bookworm)
- **Formatter**: Black, 88 chars
- **Linter**: Ruff
- **Type hints**: Required on all function signatures
- **Logging**: `logging` module, structured JSON. DEBUG for dev, INFO for prod.
- **Config**: YAML at `/etc/people-counter/config.yaml`. See `config/config.example.yaml`.
- **Secrets**: X.509 certs in `/etc/people-counter/certs/`. Never commit.
- **Tests**: pytest, mirroring src structure.
- **No classes unless stateful.** Tracker and MQTTClient justify classes. Prefer functions elsewhere.
- **Every I/O must have error handling.** Camera read, MQTT publish, file write — all wrapped.

## Directory Structure

```
people-counter/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/
│   ├── vision/
│   │   ├── capture.py     ← stereo frame acquisition (live CSI + file replay)
│   │   ├── calibration.py ← ChArUco calibration + rectification (IMPLEMENTED)
│   │   ├── depth.py       ← SGBM disparity + depth conversion (IMPLEMENTED)
│   │   └── detect.py      ← YOLOv8n inference (Hailo + OpenCV backends)
│   ├── tracking/
│   │   ├── tracker.py     ← 3D Euclidean tracker
│   │   └── counter.py     ← virtual line crossing logic
│   ├── wifi_ble/
│   │   ├── wifi_probe.py  ← nexmon monitor mode capture (NOT STARTED)
│   │   ├── ble_scan.py    ← BLE advertising capture (NOT STARTED)
│   │   ├── hasher.py      ← SHA-256 truncated hashing
│   │   └── dedup.py       ← intra + cross-protocol dedup
│   ├── mqtt/
│   │   ├── client.py      ← AWS IoT Core MQTT client (IMPLEMENTED)
│   │   └── buffer.py      ← SQLite local buffer
│   ├── cloud/
│   │   └── lambda_dedup.py ← inter-camera dedup L3 (IMPLEMENTED)
│   ├── config/
│   │   └── loader.py      ← YAML config loading + validation
│   └── main.py            ← full pipeline orchestrator (IMPLEMENTED)
├── tests/                 ← 98 tests across all modules
├── scripts/
│   ├── calibrate.py       ← CLI calibration tool (4 subcommands)
│   └── provision.py       ← device provisioning (NOT STARTED)
├── calibration/
├── infra/
├── docs/
└── config/
    └── config.example.yaml
```

## Sprint Plan (dev-only tasks)

| Sprint | Focus | Deliverable | Status |
|--------|-------|------------|--------|
| S3 | PoC | Stereo capture + YOLOv8n on RPi5. Prove it works. | **SOFTWARE READY** — capture.py (live + file replay), detect.py (Hailo + OpenCV backends). Hardware pending. |
| S4 | Calibration | ChArUco pipeline. Rectification. Depth map. | **DONE** — calibration.py (362 lines, 15 tests), depth.py (201 lines, 16 tests), scripts/calibrate.py CLI. |
| S5 | Detection | HEF compilation. Hailo SDK integration. 30+ FPS. | **SOFTWARE READY** — detect.py with Hailo + OpenCV backends, preprocess/postprocess tested (10 tests). HEF compilation pending hardware. |
| S6 | Tracking | 3D tracker. Virtual line. Ingress/egress events. | **DONE** — tracker.py + counter.py (12 tests). main.py wired E2E. |
| S7 | WiFi/BLE | nexmon + BLE capture. Hashing. Dedup L1+L2. | **PARTIAL** — hasher.py + dedup.py done (11 tests). wifi_probe.py + ble_scan.py need hardware. |
| S8 | MQTT | IoT Core client. SQLite buffer. Reconnect. | **DONE** — client.py with TLS, buffer replay, backoff (7 tests). |
| S9 | Cloud | Lambda dedup L3. QuickSight. API Gateway. | **PARTIAL** — lambda_dedup.py done (9 tests). CloudFormation/QuickSight/API GW pending. |
| S10 | Integration | End-to-end. All modules together. | **SOFTWARE READY** — main.py orchestrates full pipeline with --replay-dir for testing. |
| S11 | Pilot | Deploy 3 stores. Monitor. Fix. | PENDING |
| S12 | Stabilize | Post-pilot fixes. | PENDING |

## Implementation Status

**98 tests passing.** Modules by status:

- ✅ COMPLETE: calibration, depth, tracker, counter, hasher, dedup, buffer, client, lambda_dedup, loader, main
- 🔧 SOFTWARE READY (need hardware to finish): capture (has FileCapture for dev), detect (has OpenCV backend for dev)
- ❌ NOT STARTED: wifi_probe.py, ble_scan.py, scripts/provision.py, infra/ (CloudFormation)

## Hard Rules

- **No video/image transmission.** Only metadata.
- **No raw MAC storage.** Hash first, always.
- **WiFi = probe only.** Network = Ethernet.
- **No HAT stacking.** PoE M.2 HAT+ is the only HAT.
- **No hardcoded config.** Everything in YAML.
- **Always buffer locally.** Assume connectivity will fail.

## Environment

- Raspberry Pi OS Bookworm 64-bit, Python 3.11
- Hailo SDK: hailo_platform 4.17+
- OpenCV: 4.8+ (with contrib for ArUco/ChArUco)
- MQTT: paho-mqtt 2.0+
- DB: sqlite3 (stdlib)

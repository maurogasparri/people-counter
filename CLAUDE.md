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
  **IMPORTANT**: OV5647 cameras must have IR cut filter (NOT "NoIR" / "night vision"). NoIR cameras produce violet tint and reduce SGBM depth accuracy.
- Hailo-8L 13 TOPS via PoE M.2 HAT+ (Waveshare) — neural accelerator
- 2× OV5647 160° fisheye cameras via CSI — stereo pair, 14cm baseline
- PoE 30W injector — power via existing Ethernet cabling
- PETG 3D-printed enclosure — ceiling mount on metal L-bracket

## Key Technical Decisions

### Vision Pipeline
- **Stereo calibration**: ChArUco pattern (A4, calib.io 5x7 DICT_5X5), `--mode pinhole` (default, center-crop + `cv2.calibrateCamera` with CALIB_RATIONAL_MODEL) or `--mode fisheye` (full FOV, `cv2.fisheye`). Board params (columns, rows, square-length, marker-length) required on all CLI subcommands. Store intrinsics/extrinsics as `.npz` per device. Calibrate at 0.5-1.5m.
- **Rectification**: Precomputed maps via `cv2.initUndistortRectifyMap` (pinhole) or `cv2.fisheye.initUndistortRectifyMap` (fisheye). Applied per frame pair. Pinhole mode auto-crops frames by stored crop_ratio before rectification.
- **Depth**: Semi-Global Block Matching (`cv2.StereoSGBM`) on rectified pair + right matcher + WLS filter (`cv2.ximgproc.DisparityWLSFilter`).
- **Detection**: YOLOv8n compiled to HEF via Hailo Model Zoo. Run on Hailo-8L at 30+ FPS. Use `hailo_platform` Python SDK.
- **Tracking**: Euclidean distance tracker in 3D space (x, y, depth). Unique ID per trajectory.
- **Counting**: Virtual line in depth coordinates. Crossing direction = ingress/egress event. Publish immediately via MQTT.

### WiFi/BLE Capture
- **WiFi**: CYW43455 in monitor mode via nexmon (firmware-nexmon + brcmfmac-nexmon-dkms from Kali packages) + airmon-ng. Capture probe requests on 2.4 AND 5 GHz. **WiFi is EXCLUSIVE for probing — network connectivity is Ethernet only.**
- **BLE**: Same CYW43455 via bleak (BlueZ D-Bus API). Passive advertising scan.
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

- **Language**: Python 3.13 (RPi OS Trixie)
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
│   │   ├── wifi_probe.py  ← nexmon/airmon-ng probe capture (VALIDATED)
│   │   ├── ble_scan.py    ← BLE advertising capture via bleak (VALIDATED)
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
├── tests/                 ← 180 tests across all modules
├── scripts/
│   ├── calibrate.py       ← CLI calibration tool (4 subcommands, headless)
│   ├── focus_assist.py    ← live focus scoring with HTTP preview
│   ├── diagnose_depth.py  ← depth estimation diagnostic
│   ├── provision.py       ← device provisioning (create/deploy/list)
│   ├── verify_hardware.py ← hardware verification
│   └── setup_device.sh    ← automated device setup (steps 4-9)
├── calibration/
│   └── charuco_board.pdf  ← reference ChArUco pattern (calib.io 5x7 DICT_5X5)
├── infra/
│   └── cloudformation/
│       └── people-counter.yaml ← full stack (IoT, Timestream, DynamoDB, Lambda)
├── docs/
│   └── setup-guide.md     ← hardware assembly + RPi setup guide
└── config/
    ├── config.example.yaml
    └── people-counter.service ← systemd service file
```

## Sprint Plan (dev-only tasks)

| Sprint | Focus | Deliverable | Status |
|--------|-------|------------|--------|
| S3 | PoC | Stereo capture + YOLOv8n on RPi5. Prove it works. | **HARDWARE VALIDATED** — capture.py adapted to picamera2, stereo capture verified on RPi5 with OV5647 pair. detect.py (Hailo + OpenCV backends). |
| S4 | Calibration | ChArUco pipeline. Rectification. Depth map. | **DONE** — calibration.py dual mode: pinhole (center crop, CALIB_RATIONAL_MODEL) + fisheye (full FOV). Board params required on CLI. Baseline 142.8mm. Depth accuracy pending good-light validation. |
| S5 | Detection | HEF compilation. Hailo SDK integration. 30+ FPS. | **SOFTWARE READY** — detect.py with Hailo + OpenCV backends, preprocess/postprocess tested (10 tests). Hailo-8L verified (fw 4.23.0, PCIe Gen 3). HEF compilation pending. |
| S6 | Tracking | 3D tracker. Virtual line. Ingress/egress events. | **DONE** — tracker.py + counter.py (12 tests). main.py wired E2E (17 tests). |
| S7 | WiFi/BLE | nexmon + BLE capture. Hashing. Dedup L1+L2. | **HARDWARE VALIDATED** — wifi_probe.py (nexmon + airmon-ng + scapy, probes captured), ble_scan.py (bleak, 343 adverts/8 unique devices). hasher.py + dedup.py (11 tests). |
| S8 | MQTT | IoT Core client. SQLite buffer. Reconnect. | **DONE** — client.py with TLS, buffer replay, backoff (7 tests). |
| S9 | Cloud | Lambda dedup L3. CloudFormation. | **DONE** — lambda_dedup.py (9 tests). CloudFormation template with IoT Core, Timestream, DynamoDB, Lambda, IAM. QuickSight/API GW deferred post-MVP. |
| S10 | Integration | End-to-end. All modules together. | **E2E VALIDATED** — pipeline tested on RPi5: capture → rectify → depth (SGBM) → detect (Hailo) → depth per person. Person detected at 2.2m, depth reported 1.4m (SGBM tuning pending). |
| S11 | Pilot | Deploy 3 stores. Monitor. Fix. | PENDING |
| S12 | Stabilize | Post-pilot fixes. | PENDING |

## Implementation Status

**180 tests passing.** Modules by status:

- ✅ COMPLETE + VALIDATED: capture (picamera2), detect (Hailo-8L HEF), wifi_probe (nexmon), ble_scan (bleak), calibration, depth, tracker, counter, hasher, dedup, buffer, client, lambda_dedup, loader, main
- 🔧 INFRA READY: CloudFormation template, systemd service, provision.py, logrotate, daily reset timer
- ⏳ PENDING: OV5647 160° with IR filter (current NoIR cameras affect depth), cenital detection test, SGBM tuning

## Known Limitations (MVP)

- **Tracker**: Greedy matching by 2D pixel distance + depth gating. No histéresis, no multi-state (approaching/crossed/invalidated), no reidentification. Sufficient for single-door ceiling-mount, needs work for wide entrances with occlusion.
- **Shadow config**: Local cache bootstrap from `.shadow.json` file. No live delta subscription yet — planned post-MVP.
- **Operating hours fail-open**: If schedule format is invalid, counting continues (prefers false positives over lost data). Configurable fail-closed planned for production.

## Hard Rules

- **No video/image transmission.** Only metadata.
- **No raw MAC storage.** Hash first, always.
- **WiFi = probe only.** Network = Ethernet.
- **No HAT stacking.** PoE M.2 HAT+ is the only HAT.
- **No hardcoded config.** Everything in YAML.
- **Always buffer locally.** Assume connectivity will fail.

## Environment

- Raspberry Pi OS Trixie 64-bit, Python 3.13
- Hailo SDK: hailo_platform 4.23+
- Picamera2: for CSI camera capture (rpicam-* CLI tools)
- OpenCV: 4.8+ (with contrib for ArUco/ChArUco)
- MQTT: paho-mqtt 2.0+
- DB: sqlite3 (stdlib)

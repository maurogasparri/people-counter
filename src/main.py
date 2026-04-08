"""People Counter — Main entry point.

Orchestrates the full edge pipeline:
    1. Stereo capture → rectification
    2. Disparity → depth map
    3. YOLOv8n person detection
    4. 3D tracking + virtual line counting
    5. Event publishing via MQTT (buffered)
    6. Telemetry reporting
    7. WiFi/BLE probing (optional)
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

from src.config.loader import (
    get_scaling_factor,
    is_counting_enabled,
    is_within_operating_hours,
    load_config,
    merge_cloud_config,
)
from src.mqtt.buffer import MessageBuffer
from src.mqtt.client import MQTTClient
from src.tracking.counter import LineCounter
from src.tracking.tracker import EuclideanTracker
from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import FileCapture, StereoCapture
from src.vision.depth import compute_disparity, create_sgbm, depth_at_bbox, disparity_to_depth
from src.vision.detect import detect_persons, load_model

logger = logging.getLogger(__name__)


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))

    if log_cfg.get("format") == "json":
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_file = log_cfg.get("file")
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def build_capture(config: dict[str, Any], replay_dir: str | None = None):
    """Build the appropriate capture source."""
    if replay_dir:
        logger.info("Using file replay from %s", replay_dir)
        cap = FileCapture(
            directory=replay_dir,
            loop=True,
            fps=config["vision"].get("fps", 15),
        )
    else:
        cap = StereoCapture(
            cam_left_id=config["vision"]["camera_left"],
            cam_right_id=config["vision"]["camera_right"],
            resolution=tuple(config["vision"]["resolution"]),
            fps=config["vision"].get("fps", 15),
        )
    return cap


def build_mqtt(config: dict[str, Any]) -> tuple[MQTTClient, MessageBuffer]:
    """Build MQTT client and buffer from config."""
    buf_cfg = config["buffer"]
    buffer = MessageBuffer(
        db_path=buf_cfg["db_path"],
        max_age_hours=buf_cfg.get("max_age_hours", 72),
    )

    mqtt_cfg = config["mqtt"]
    store_id = config["device"]["store_id"]
    device_id = config["device"]["id"]

    # Expand topic templates
    topics = {}
    for key, template in mqtt_cfg.get("topics", {}).items():
        topics[key] = template.replace("{store_id}", store_id).replace("{device_id}", device_id)

    client = MQTTClient(
        device_id=config["device"]["id"],
        endpoint=mqtt_cfg["endpoint"],
        port=mqtt_cfg.get("port", 8883),
        cert_path=mqtt_cfg["cert_path"],
        key_path=mqtt_cfg["key_path"],
        ca_path=mqtt_cfg["ca_path"],
        buffer=buffer,
        topics=topics,
    )
    return client, buffer


def get_telemetry() -> dict[str, Any]:
    """Collect device telemetry."""
    telemetry: dict[str, Any] = {"uptime_s": 0}

    try:
        with open("/proc/uptime") as f:
            telemetry["uptime_s"] = float(f.read().split()[0])
    except Exception:
        pass

    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            telemetry["cpu_temp_c"] = int(f.read().strip()) / 1000.0
    except Exception:
        pass

    try:
        import shutil
        usage = shutil.disk_usage("/")
        telemetry["disk_free_mb"] = usage.free // (1024 * 1024)
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    telemetry["mem_available_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass

    return telemetry


def run_pipeline(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Run the main processing pipeline."""
    device_id = config["device"]["id"]
    store_id = config["device"]["store_id"]
    vision_cfg = config["vision"]
    detect_cfg = config["detection"]
    track_cfg = config.get("tracking", {})
    telem_cfg = config.get("telemetry", {})

    # --- Load calibration ---
    cal_file = vision_cfg.get("calibration_file")
    if cal_file:
        logger.info("Loading calibration from %s", cal_file)
        calibration = load_calibration(cal_file)
    else:
        calibration = None
        logger.warning("No calibration file configured — skipping rectification")

    # --- Load detection model ---
    model_path = detect_cfg["model_path"]
    backend = args.detection_backend if hasattr(args, "detection_backend") else "auto"
    logger.info("Loading model: %s (backend=%s)", model_path, backend)
    model = load_model(model_path, backend=backend)

    # --- Build SGBM ---
    sgbm = create_sgbm(
        num_disparities=vision_cfg.get("num_disparities", 128),
        block_size=vision_cfg.get("block_size", 9),
    )

    # --- Build tracker + counter ---
    tracker = EuclideanTracker(
        max_disappeared=track_cfg.get("max_disappeared", 30),
        max_distance=track_cfg.get("max_distance", 50.0),
    )
    line_y = vision_cfg.get("counting_line_y", 0.5)
    # Convert relative line position to pixels if needed
    # (actual pixel value computed after first frame read)
    counter: LineCounter | None = None

    # --- Build MQTT ---
    mqtt_client, buffer = build_mqtt(config)
    mqtt_client.connect()

    # --- Build capture ---
    capture = build_capture(config, replay_dir=getattr(args, "replay_dir", None))
    capture.open()

    # --- Focal length + baseline for depth ---
    # Use calibration values if available, config as fallback
    focal_length_px = None
    baseline_mm = vision_cfg.get("baseline_cm", 14) * 10.0

    # --- Telemetry timer ---
    telem_interval = telem_cfg.get("interval_seconds", 300)
    last_telem = time.time()

    # --- Graceful shutdown ---
    running = True

    def _signal_handler(sig: int, frame: Any) -> None:
        nonlocal running
        logger.info("Shutdown signal received (%d)", sig)
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "Pipeline started: device=%s store=%s",
        device_id,
        store_id,
    )

    frame_count = 0
    fps_start = time.time()
    telem_frame_count = 0
    telem_fps_start = time.time()
    last_hours_check = 0.0
    last_purge = time.time()
    within_hours = True  # assume open until first check

    try:
        while running:
            # --- Check operating hours every 60 seconds ---
            now = time.time()
            if now - last_hours_check >= 60.0:
                dt = datetime.now()
                day_name = dt.strftime("%A").lower()
                within_hours = is_within_operating_hours(
                    config, day_name, dt.hour, dt.minute
                )
                if not within_hours:
                    logger.debug("Outside operating hours (%s %02d:%02d) — paused",
                                 day_name, dt.hour, dt.minute)
                last_hours_check = now

            # --- Check if counting is enabled (cloud toggle) ---
            if not is_counting_enabled(config) or not within_hours:
                time.sleep(1.0)
                continue
            try:
                frame_l, frame_r = capture.read()
            except StopIteration:
                logger.info("File replay exhausted")
                break
            except RuntimeError as e:
                logger.error("Capture error: %s", e)
                time.sleep(0.1)
                continue

            # --- Rectification ---
            if calibration is not None:
                rect_l, rect_r = rectify_pair(frame_l, frame_r, calibration)
            else:
                rect_l, rect_r = frame_l, frame_r

            # --- Initialize counter with actual frame height ---
            if counter is None:
                h = rect_l.shape[0]
                actual_line_y = line_y * h if line_y <= 1.0 else line_y
                counter = LineCounter(line_y=actual_line_y)
                logger.info("Counting line at y=%.1f", actual_line_y)

            # --- Set focal length + baseline from calibration ---
            if focal_length_px is None and calibration is not None:
                focal_length_px = calibration["P1"][0, 0]
                # Use actual baseline from calibration (T vector magnitude)
                T = calibration["T"]
                baseline_mm = float(np.linalg.norm(T))
                logger.info("Focal length: %.1f px, Baseline: %.1f mm", focal_length_px, baseline_mm)

            # --- Depth map ---
            if calibration is not None and focal_length_px is not None:
                disparity = compute_disparity(rect_l, rect_r, sgbm=sgbm)
                depth_map = disparity_to_depth(
                    disparity, focal_length_px, baseline_mm
                )
            else:
                depth_map = None

            # --- Detection ---
            detections = detect_persons(
                rect_l,
                model,
                confidence_threshold=detect_cfg.get("confidence_threshold", 0.5),
                nms_threshold=detect_cfg.get("nms_threshold", 0.45),
            )

            # --- Build 3D positions from detections + depth ---
            positions = []
            for det in detections:
                cx, cy = det.centroid
                if depth_map is not None:
                    z = depth_at_bbox(depth_map, det.bbox)
                else:
                    z = 0.0
                positions.append(np.array([cx, cy, z]))

            # --- Tracking ---
            tracks = tracker.update(positions)

            # --- Counting ---
            events = counter.check_all(tracks)

            # --- Publish counting events ---
            scaling = get_scaling_factor(config)
            for event in events:
                mqtt_client.publish_event(
                    "counting",
                    {
                        "direction": event.direction,
                        "track_id": event.track_id,
                        "position_y": event.position_y,
                        "event_time": event.timestamp,
                        "total_in": counter.total_in,
                        "total_out": counter.total_out,
                        "scaling_factor": scaling,
                        "scaled_in": round(counter.total_in * scaling),
                        "scaled_out": round(counter.total_out * scaling),
                    },
                )

            # --- FPS tracking ---
            frame_count += 1
            telem_frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 10.0:
                fps = frame_count / elapsed
                logger.info(
                    "Pipeline: %.1f FPS, %d detections, in=%d out=%d",
                    fps,
                    len(detections),
                    counter.total_in,
                    counter.total_out,
                )
                frame_count = 0
                fps_start = time.time()

            # --- Telemetry ---
            now = time.time()
            if now - last_telem >= telem_interval:
                telem_elapsed = now - telem_fps_start
                telem = get_telemetry()
                telem["fps"] = telem_frame_count / max(telem_elapsed, 1)
                telem["total_in"] = counter.total_in
                telem["total_out"] = counter.total_out
                mqtt_client.publish_event("telemetry", telem)
                last_telem = now
                telem_frame_count = 0
                telem_fps_start = now

            # --- Buffer maintenance (every 60s, not every frame) ---
            if now - last_purge >= 60.0:
                buffer.purge_old()
                last_purge = now

    finally:
        capture.close()
        mqtt_client.disconnect()
        logger.info(
            "Pipeline stopped. Final counts: in=%d out=%d",
            counter.total_in if counter else 0,
            counter.total_out if counter else 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="People Counter Edge Device")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--replay-dir",
        help="Replay from saved stereo pairs instead of live cameras",
    )
    parser.add_argument(
        "--detection-backend",
        choices=["auto", "hailo", "opencv"],
        default="auto",
        help="Detection backend (default: auto-detect from model extension)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    # --- Attempt to merge cloud config from IoT Shadow ---
    # In production this would fetch from AWS IoT via MQTT $aws/things/{id}/shadow/get.
    # For the MVP, we read a local shadow cache file if it exists (updated by a
    # background process or on previous boot). If not available, local defaults apply.
    shadow_path = args.config.replace(".yaml", ".shadow.json")
    try:
        import json
        from pathlib import Path

        shadow_file = Path(shadow_path)
        if shadow_file.exists():
            shadow_data = json.loads(shadow_file.read_text())
            desired = shadow_data.get("state", {}).get("desired", {})
            config = merge_cloud_config(config, desired)
            logger.info("Cloud shadow merged from %s", shadow_path)
        else:
            logger.info("No shadow cache at %s — using local defaults", shadow_path)
    except Exception as e:
        logger.warning("Failed to load shadow cache: %s — using local defaults", e)

    logger.info(
        "Starting people-counter",
        extra={"device_id": config["device"]["id"]},
    )

    run_pipeline(config, args)


if __name__ == "__main__":
    main()

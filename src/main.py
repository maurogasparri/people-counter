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
import math
import os
import queue
import signal
import socket
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from src.config.hardware import load_hardware_config
from src.config.loader import (
    apply_shadow_delta,
    build_reported_state,
    get_invalid_schedule_mode,
    get_scaling_factor,
    has_schedule_error,
    is_counting_enabled,
    is_within_operating_hours,
    load_config,
    merge_cloud_config,
)
from src.mqtt.buffer import MessageBuffer
from src.mqtt.client import MQTTClient
from src.status.led import StatusLED
from src.status.monitor import HealthMonitor, HealthSignals
from src.telemetry import collect_telemetry
from src.tracking.counter import LineCounter, ROICounter, build_counter
from src.tracking.tracker import EuclideanTracker
from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import FileCapture, StereoCapture
from src.vision.depth import (
    compute_disparity, create_sgbm, depth_at_bbox, disparity_to_depth,
    min_depth_at_bbox,
)
from src.vision.world_coords import classify_height, head_height_above_floor
from src.vision.detect import detect_persons, load_model

# Size of the rolling windows used for frame-latency percentiles and
# detection-rate calculations. 100 covers ~7s at 15 FPS — enough to smooth
# noise without drowning brief stalls.
TELEMETRY_WINDOW_SIZE = 100

logger = logging.getLogger(__name__)


def sd_notify(message: str) -> None:
    """Send a notification to systemd via NOTIFY_SOCKET.

    No-op if the socket env var is unset (i.e. running outside systemd).
    Used to send READY=1, WATCHDOG=1, STOPPING=1 for Type=notify services.
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):  # Abstract namespace on Linux
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(sock_path)
            s.sendall(message.encode())
    except OSError as e:
        logger.debug("sd_notify(%s) failed: %s", message, e)


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
        # camera_left / camera_right come from hardware.yaml — they're
        # determined by the bracket assembly procedure (which CSI port the
        # left/right ribbon plugs into) and are invariant across the fleet.
        # See config/hardware.yaml.
        hw = load_hardware_config()
        cap = StereoCapture(
            cam_left_id=hw["bracket"]["camera_left_csi"],
            cam_right_id=hw["bracket"]["camera_right_csi"],
            resolution=tuple(
                config["vision"].get("resolution",
                                     hw["sensor"]["default_res"])
            ),
            fps=config["vision"].get("fps", hw["sensor"]["default_fps"]),
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


def _build_telemetry_state(
    frame_latencies_ms: "deque[float]",
    detection_counts: "deque[int]",
    detection_window_start_ts: float,
    fps: float | int | None,
    tracker: EuclideanTracker,
    mqtt_client: MQTTClient,
    buffer: MessageBuffer,
) -> dict[str, Any]:
    """Snapshot pipeline runtime state into the dict expected by
    :func:`collect_telemetry`. Each lookup is wrapped so a single faulty
    probe never blocks the telemetry emission.
    """
    state: dict[str, Any] = {
        "frame_latencies_ms": list(frame_latencies_ms),
        "detection_counts": list(detection_counts),
        "detection_window_start_ts": detection_window_start_ts,
        "fps": fps,
    }

    try:
        track_counts = tracker.count_by_state()
        state["tracker_confirmed"] = track_counts.get("confirmed", 0)
        state["tracker_pending"] = track_counts.get("pending", 0)
    except Exception:
        logger.exception("tracker.count_by_state failed")
        state["tracker_confirmed"] = None
        state["tracker_pending"] = None

    try:
        state["mqtt_disconnect_count"] = mqtt_client.disconnect_count
    except Exception:
        state["mqtt_disconnect_count"] = None
    try:
        state["mqtt_reconnect_ts"] = mqtt_client.reconnect_ts
    except Exception:
        state["mqtt_reconnect_ts"] = None

    try:
        state["buffer_backlog"] = buffer.count_unsent()
    except Exception:
        logger.exception("buffer.count_unsent failed")
        state["buffer_backlog"] = None

    return state


def get_telemetry(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect device telemetry.

    Thin wrapper over :func:`src.telemetry.collect_telemetry` kept for
    backwards compatibility. The original implementation returned
    ``uptime_s = 0`` when ``/proc/uptime`` was unreadable; the new module
    returns ``None``. Callers that must see a numeric uptime (legacy tests
    running on Windows) coerce here.
    """
    telem = collect_telemetry(state)
    if telem.get("uptime_s") is None:
        telem["uptime_s"] = 0
    return telem


def _auto_num_disparities(
    vision_cfg: dict[str, Any], logger: logging.Logger,
) -> int:
    """Derive a num_disparities that matches the head-distance range at this
    site. Uses the calibration focal length (from the .npz loaded elsewhere
    or a nominal if unknown), design baseline 140mm, and the operator-set
    mounting_height_m. Rounded up to the next multiple of 16 (SGBM constraint).
    """
    mount_m = float(vision_cfg.get("mounting_height_m", 3.0) or 3.0)
    # f_px and baseline — we don't yet have the calibration loaded at this
    # point in the pipeline, so we use the nominal optical values. Fine for
    # sizing the disparity search; the actual depth will use the real K/T.
    f_px = 2050.0
    baseline_mm = 140.0
    head_max_m = 1.85  # tallest adult head
    floor_margin_m = 0.5  # a bit beyond the floor so SGBM covers the full bg

    z_min = max(0.3, mount_m - head_max_m)
    z_max = mount_m + floor_margin_m
    disp_max = f_px * baseline_mm / (z_min * 1000)
    disp_min = f_px * baseline_mm / (z_max * 1000)

    # SGBM searches [minDisparity, minDisparity + numDisparities). We keep
    # minDisparity=0 (OpenCV default) for simplicity, so numDisparities
    # needs to cover 0..disp_max. Round up to next multiple of 16 + one
    # extra bucket for margin.
    raw = int(math.ceil(disp_max / 16.0) + 1) * 16
    # Clamp to a sane envelope.
    num_disparities = max(64, min(512, raw))
    logger.info(
        "num_disparities auto: mount=%.2fm → z=[%.2f,%.2f]m → disp=[%.0f,%.0f]"
        " → num_disparities=%d",
        mount_m, z_min, z_max, disp_min, disp_max, num_disparities,
    )
    return num_disparities


def run_pipeline(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Run the main processing pipeline."""
    device_id = config["device"]["id"]
    store_id = config["device"]["store_id"]
    vision_cfg = config["vision"]
    detect_cfg = config["detection"]
    track_cfg = config.get("tracking", {})
    telem_cfg = config.get("telemetry", {})
    status_cfg = config.get("status_led", {}) or {}

    # --- Status LED + health monitor (started before any heavy init so it can
    # surface hardware faults during model load / camera open). The LED falls
    # back to a no-op when gpiozero is missing, so this is safe off-RPi.
    health_signals = HealthSignals(
        provisioned=True,
        calibration_path=vision_cfg.get("calibration_file"),
    )
    status_led: StatusLED | None = None
    health_monitor: HealthMonitor | None = None
    if bool(status_cfg.get("enabled", True)):
        status_led = StatusLED()
        health_monitor = HealthMonitor(led=status_led, signals=health_signals)
        health_monitor.start()

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
    detection_backend = getattr(args, "detection_backend", "auto")
    logger.info("Loading model: %s (backend=%s)", model_path, detection_backend)
    model = load_model(model_path, backend=detection_backend)

    # --- Build SGBM ---
    # num_disparities can be a concrete int or the literal "auto" — in which
    # case we derive the search range from mounting_height_m so the solver
    # covers exactly the depth band where heads will appear (1.2-1.85m
    # below the camera) plus a small floor margin. Saves compute at high
    # mounts and gives wider search at low mounts, transparently per site.
    num_disp_cfg = vision_cfg.get("num_disparities", 192)
    if isinstance(num_disp_cfg, str) and num_disp_cfg.lower() == "auto":
        num_disp_resolved = _auto_num_disparities(vision_cfg, logger)
    else:
        num_disp_resolved = int(num_disp_cfg)
    sgbm = create_sgbm(
        num_disparities=num_disp_resolved,
        block_size=vision_cfg.get("block_size", 9),
    )

    # --- Build tracker + counter ---
    counter_cfg = config.get("counter", {})
    tracker_cfg = counter_cfg.get("tracker", {})
    tracker = EuclideanTracker(
        max_disappeared=track_cfg.get("max_disappeared", 30),
        max_distance=track_cfg.get("max_distance", 50.0),
        max_depth_delta=tracker_cfg.get("depth_gate_m", 0.5) * 1000.0,
        confirm_frames=tracker_cfg.get("confirm_frames", 3),
        pending_max_frames=tracker_cfg.get("pending_max_frames", 5),
        reid_gate_px=tracker_cfg.get("reid_gate_px", 60.0),
    )
    line_y = vision_cfg.get("counting_line_y", 0.5)
    # Counter built lazily once frame height is known (needed for legacy line_y relative values)
    counter: LineCounter | ROICounter | None = None

    # --- Build MQTT ---
    mqtt_client, buffer = build_mqtt(config)

    # --- Shadow delta wiring -------------------------------------------------
    # Deltas arrive in the paho network thread; the queue is the thread-safe
    # hand-off to the main loop, which is the only writer for `config`. The
    # same queue also carries a sentinel ``{"__reconcile__": True}`` item
    # posted by the MQTT on_connected hook, so reported-state publishing runs
    # on the main thread (no paho-thread network calls).
    shadow_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=32)
    _RECONCILE_SENTINEL = {"__reconcile__": True}

    def _shadow_delta_handler(state: dict[str, Any]) -> None:
        try:
            shadow_queue.put_nowait(state)
        except queue.Full:
            logger.warning("Shadow delta queue full — dropping delta")

    def _on_mqtt_connected() -> None:
        # Runs in paho thread — enqueue, don't block.
        try:
            shadow_queue.put_nowait(_RECONCILE_SENTINEL)
        except queue.Full:
            logger.warning("Shadow queue full — dropping reconcile request")

    mqtt_client.on_connected = _on_mqtt_connected
    mqtt_client.connect()

    try:
        mqtt_client.subscribe_shadow_delta(device_id, _shadow_delta_handler)
    except Exception:
        logger.exception("Shadow delta subscription failed")

    config_path_arg = getattr(args, "config", None)

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
    last_watchdog = 0.0
    within_hours = True  # assume open until first check

    # --- Observability state (sliding windows) ---
    # Rolling buffers of per-frame latency and detection counts. Cleared after
    # each telemetry emission so each sample represents the previous interval.
    frame_latencies_ms: deque[float] = deque(maxlen=TELEMETRY_WINDOW_SIZE)
    detection_counts: deque[int] = deque(maxlen=TELEMETRY_WINDOW_SIZE)
    detection_window_start_ts = time.time()

    # Operational-hours fail mode — only log the startup warning once.
    schedule_invalid = has_schedule_error(config)
    invalid_mode = get_invalid_schedule_mode(config)
    if schedule_invalid:
        err = config.get("_schedule_error", "unknown")
        if invalid_mode == "fail_closed":
            logger.critical(
                "Invalid operating_hours (%s) and on_invalid_schedule=fail_closed "
                "— counting paused until a valid schedule is pushed",
                err,
            )
        else:
            logger.warning(
                "Invalid operating_hours (%s) and on_invalid_schedule=fail_open "
                "— continuing to count (may produce false positives)",
                err,
            )

    # Signal systemd that startup is complete. Pair with WatchdogSec in the
    # unit file — we ping WATCHDOG=1 inside the loop below.
    sd_notify("READY=1")
    health_signals.boot_complete = True

    try:
        while running:
            # --- Drain pending shadow deltas (main-thread side) ---
            while True:
                try:
                    delta_state = shadow_queue.get_nowait()
                except queue.Empty:
                    break
                # Reconcile sentinel (on (re)connect): publish current effective
                # state as reported. Non-fatal on error.
                if delta_state is _RECONCILE_SENTINEL or delta_state.get(
                    "__reconcile__"
                ):
                    try:
                        reported = build_reported_state(config, calibration)
                        mqtt_client.publish_shadow_reported(device_id, reported)
                        logger.info(
                            "shadow_reconciliation_published",
                            extra={"keys": sorted(reported.keys())},
                        )
                    except Exception:
                        logger.exception("Failed to publish shadow reconciliation")
                    continue
                try:
                    new_config, applied_keys = apply_shadow_delta(
                        config, delta_state, config_path=config_path_arg
                    )
                except Exception:
                    logger.exception("apply_shadow_delta failed")
                    continue
                if not applied_keys:
                    continue
                config = new_config
                # Re-evaluate schedule flag after a shadow update.
                schedule_invalid = has_schedule_error(config)
                invalid_mode = get_invalid_schedule_mode(config)
                last_hours_check = 0.0  # force recheck on next iteration

                # Selective re-init based on which keys changed.
                # counter.tracker.* and counter.height_classifier.* are read
                # live each frame — no rebuild needed.
                if any(
                    k.startswith("counter.")
                    and not k.startswith("counter.tracker.")
                    and not k.startswith("counter.height_classifier.")
                    for k in applied_keys
                ):
                    counter = None
                    logger.info("Counter will be rebuilt after shadow delta")
                if any(k.startswith("counter.tracker.") for k in applied_keys):
                    logger.info(
                        "Tracker params updated; new values apply to future tracks"
                    )
                if any(
                    k in ("vision.num_disparities", "vision.block_size",
                          "vision.mounting_height_m")
                    for k in applied_keys
                ):
                    nd_cfg = config["vision"].get("num_disparities", 192)
                    if isinstance(nd_cfg, str) and nd_cfg.lower() == "auto":
                        nd_resolved = _auto_num_disparities(
                            config["vision"], logger,
                        )
                    else:
                        nd_resolved = int(nd_cfg)
                    sgbm = create_sgbm(
                        num_disparities=nd_resolved,
                        block_size=config["vision"].get("block_size", 9),
                    )
                    logger.info("SGBM rebuilt after shadow delta")
                if any(k == "telemetry.interval_seconds" for k in applied_keys):
                    telem_interval = config.get("telemetry", {}).get(
                        "interval_seconds", telem_interval
                    )

                try:
                    mqtt_client.publish_shadow_reported(
                        device_id, {k: True for k in applied_keys}
                    )
                except Exception:
                    logger.exception("Publishing shadow reported failed")

            # --- Check operating hours every 60 seconds ---
            now = time.time()
            if now - last_hours_check >= 60.0:
                if schedule_invalid:
                    # fail_closed: treat as outside hours; fail_open: count.
                    within_hours = invalid_mode != "fail_closed"
                else:
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
                # Keep telemetry + watchdog alive so ops can re-push config.
                if schedule_invalid and invalid_mode == "fail_closed":
                    telem_now = time.time()
                    if telem_now - last_telem >= telem_interval:
                        telem = collect_telemetry(
                            _build_telemetry_state(
                                frame_latencies_ms,
                                detection_counts,
                                detection_window_start_ts,
                                config.get("vision", {}).get("fps"),
                                tracker,
                                mqtt_client,
                                buffer,
                            )
                        )
                        telem["error"] = "invalid_schedule"
                        telem["schedule_error_detail"] = config.get(
                            "_schedule_error", ""
                        )
                        mqtt_client.publish_event("telemetry", telem)
                        last_telem = telem_now
                    if telem_now - last_watchdog >= 60.0:
                        sd_notify("WATCHDOG=1")
                        last_watchdog = telem_now
                time.sleep(1.0)
                continue

            t_iter_start = time.perf_counter()
            try:
                frame_l, frame_r = capture.read()
                health_signals.capture_ok = True
            except StopIteration:
                logger.info("File replay exhausted")
                break
            except RuntimeError as e:
                logger.error("Capture error: %s", e)
                health_signals.capture_ok = False
                time.sleep(0.1)
                continue

            # --- Rectification ---
            if calibration is not None:
                rect_l, rect_r = rectify_pair(frame_l, frame_r, calibration)
            else:
                rect_l, rect_r = frame_l, frame_r

            # --- Initialize counter with actual frame height ---
            if counter is None:
                counter = build_counter(config, frame_height=rect_l.shape[0])
                logger.info("Counter initialized: %s", type(counter).__name__)

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
            try:
                detections = detect_persons(
                    rect_l,
                    model,
                    confidence_threshold=detect_cfg.get("confidence_threshold", 0.5),
                    nms_threshold=detect_cfg.get("nms_threshold", 0.45),
                )
                health_signals.detect_ok = True
            except Exception:
                logger.exception("Detection failed")
                health_signals.detect_ok = False
                time.sleep(0.1)
                continue

            # --- Build 3D positions + per-detection metadata ---
            vision_cfg = config.get("vision", {})
            counter_cfg_live = config.get("counter", {})
            hc_cfg = counter_cfg_live.get("height_classifier", {}) or {}
            hc_enabled = bool(hc_cfg.get("enabled", False))
            mount_height_mm = (
                float(vision_cfg.get("mounting_height_m", 0.0) or 0.0) * 1000.0
            )
            adult_min_mm = float(hc_cfg.get("adult_min_m", 1.55)) * 1000.0

            positions: list[np.ndarray] = []
            metas: list[dict] = []
            for det in detections:
                cx, cy = det.centroid
                if depth_map is not None:
                    z = depth_at_bbox(depth_map, det.bbox)
                    near_z = min_depth_at_bbox(depth_map, det.bbox)
                else:
                    z = 0.0
                    near_z = 0.0
                positions.append(np.array([cx, cy, z]))

                if hc_enabled and mount_height_mm > 0 and near_z > 0:
                    head_mm = head_height_above_floor(near_z, mount_height_mm)
                    height_class = classify_height(head_mm, adult_min_mm)
                else:
                    head_mm = None
                    height_class = "unknown"
                metas.append({
                    "near_depth_mm": near_z,
                    "head_height_mm": head_mm,
                    "height_class": height_class,
                })

            # --- Tracking ---
            tracks = tracker.update(positions, detection_metas=metas)

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
                        "height_class": event.height_class,
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

            # --- Observability: record per-frame latency + detection count ---
            frame_latencies_ms.append(
                (time.perf_counter() - t_iter_start) * 1000.0
            )
            detection_counts.append(len(detections))

            # --- Telemetry ---
            now = time.time()
            if now - last_telem >= telem_interval:
                telem_elapsed = now - telem_fps_start
                telem = collect_telemetry(
                    _build_telemetry_state(
                        frame_latencies_ms,
                        detection_counts,
                        detection_window_start_ts,
                        config.get("vision", {}).get("fps"),
                        tracker,
                        mqtt_client,
                        buffer,
                    )
                )
                telem["fps"] = telem_frame_count / max(telem_elapsed, 1)
                telem["total_in"] = counter.total_in
                telem["total_out"] = counter.total_out
                mqtt_client.publish_event("telemetry", telem)
                last_telem = now
                telem_frame_count = 0
                telem_fps_start = now
                # Reset rolling windows so the next sample is independent.
                frame_latencies_ms.clear()
                detection_counts.clear()
                detection_window_start_ts = now

            # --- Buffer maintenance (every 60s, not every frame) ---
            if now - last_purge >= 60.0:
                buffer.purge_old()
                last_purge = now

            # --- systemd watchdog keepalive (every 60s; WatchdogSec=300) ---
            if now - last_watchdog >= 60.0:
                sd_notify("WATCHDOG=1")
                last_watchdog = now

            # --- Health signals (read by the LED monitor thread) ---
            health_signals.last_loop_ts = now
            health_signals.mqtt_connected = mqtt_client.connected

    finally:
        sd_notify("STOPPING=1")
        if health_monitor is not None:
            health_monitor.stop()
        if status_led is not None:
            status_led.close()
        capture.close()
        mqtt_client.disconnect()
        # Release Hailo resources if backend supports it
        backend_impl = model.get("backend")
        if hasattr(backend_impl, "close"):
            backend_impl.close()
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
    from pathlib import Path
    import json

    config_path = Path(args.config)
    shadow_file = Path(str(config_path.with_suffix("")) + ".shadow.json")
    shadow_path = str(shadow_file)
    try:
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

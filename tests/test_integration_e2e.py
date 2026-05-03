"""End-to-end integration test for the full pipeline.

Drives ``src.main.run_pipeline`` with all external boundaries mocked
(capture, calibration/disparity, detector, MQTT client, time) and asserts
that the orchestration wires each stage correctly:

    capture -> rectify -> depth -> detect -> track -> count -> MQTT publish
    plus periodic telemetry.

Unlike the per-module tests in ``tests/test_main.py``, this file scripts a
multi-frame scenario for a single simulated person and verifies the
counting/telemetry side-effects produced by the whole loop.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.main import run_pipeline
from src.vision.detect import Detection


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


FRAME_W = 640
FRAME_H = 480
FOCAL_PX = 1330.0
BASELINE_MM = 140.0


def _make_config(tmpdir: str) -> dict[str, Any]:
    """Build a minimally-valid pipeline config with an ROI counter.

    The ROI spans y=[180, 300] with a horizontal counting line at y=240.
    Tracks entering at y<240 ("side_a") and exiting at y>240 ("side_b")
    produce ingress events; the mirror direction produces egress.
    """
    cert = str(Path(tmpdir) / "cert.pem")
    key = str(Path(tmpdir) / "key.pem")
    ca = str(Path(tmpdir) / "ca.pem")
    cal = str(Path(tmpdir) / "cal.npz")
    for f in [cert, key, ca, cal]:
        Path(f).write_text("dummy")

    return {
        "device": {"id": "dev-e2e", "store_id": "store-e2e"},
        "vision": {
            "camera_left": 0,
            "camera_right": 1,
            "resolution": [FRAME_W, FRAME_H],
            "fps": 15,
            "baseline_cm": 14,
            "calibration_file": cal,
            "num_disparities": 192,
            "block_size": 9,
        },
        "detection": {
            "model_path": "/tmp/model.onnx",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.45,
        },
        "tracking": {
            "max_disappeared": 30,
            "max_distance": 120.0,
        },
        "counter": {
            "roi": {"x_min": 100, "x_max": 540, "y_min": 180, "y_max": 300},
            "line": {"orientation": "horizontal", "position": 240},
            "direction_labels": {
                "side_a_to_b": "ingress",
                "side_b_to_a": "egress",
            },
            "tracker": {
                "confirm_frames": 3,
                "pending_max_frames": 5,
                "reid_gate_px": 60,
                "depth_gate_m": 0.5,
            },
        },
        "telemetry": {"interval_seconds": 9999},
        "mqtt": {
            "endpoint": "test.iot.amazonaws.com",
            "port": 8883,
            "cert_path": cert,
            "key_path": key,
            "ca_path": ca,
            "topics": {
                "counting": "store/store-e2e/counting",
                "telemetry": "store/store-e2e/telemetry",
            },
        },
        "buffer": {
            "db_path": str(Path(tmpdir) / "buf.db"),
            "max_age_hours": 72,
        },
        "cloud_defaults": {
            "counting_enabled": True,
            "operating_hours": {
                "monday": "00:00-23:59",
                "tuesday": "00:00-23:59",
                "wednesday": "00:00-23:59",
                "thursday": "00:00-23:59",
                "friday": "00:00-23:59",
                "saturday": "00:00-23:59",
                "sunday": "00:00-23:59",
            },
        },
    }


def _identity_calibration() -> dict[str, np.ndarray]:
    """Build an identity-ish calibration dict.

    ``map_l_*`` / ``map_r_*`` are identity remaps so ``rectify_pair`` is a
    pass-through.  ``P1[0,0]`` = focal length in pixels and ``T`` encodes
    the baseline so ``disparity_to_depth`` yields predictable values.
    """
    xs, ys = np.meshgrid(np.arange(FRAME_W), np.arange(FRAME_H))
    map_x = xs.astype(np.float32)
    map_y = ys.astype(np.float32)
    p1 = np.zeros((3, 4), dtype=np.float64)
    p1[0, 0] = FOCAL_PX
    p1[1, 1] = FOCAL_PX
    p1[2, 2] = 1.0
    t = np.array([[-BASELINE_MM], [0.0], [0.0]], dtype=np.float64)
    return {
        "map_l_x": map_x,
        "map_l_y": map_y,
        "map_r_x": map_x.copy(),
        "map_r_y": map_y.copy(),
        "P1": p1,
        "T": t,
    }


def _fixed_disparity_map() -> np.ndarray:
    """Return a disparity map that yields depth = 3m everywhere.

    Z = f * B / d  →  d = 1330 * 140 / 3000 ≈ 62.07 px.
    """
    disp = np.full((FRAME_H, FRAME_W), 62.07, dtype=np.float32)
    return disp


def _det_at(cx: float, cy: float, half_w: int = 40, half_h: int = 60) -> Detection:
    """Build a Detection with a bbox centered at (cx, cy)."""
    x1 = int(cx - half_w)
    y1 = int(cy - half_h)
    x2 = int(cx + half_w)
    y2 = int(cy + half_h)
    return Detection(
        bbox=(x1, y1, x2, y2),
        confidence=0.9,
        centroid=(float(cx), float(cy)),
    )


def _make_scripted_capture(n_frames: int) -> MagicMock:
    """Mock capture: returns dummy BGR frame pairs and raises StopIteration
    after ``n_frames`` to let ``run_pipeline`` exit cleanly."""
    dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    mock = MagicMock()
    call_idx = [0]

    def _read():
        idx = call_idx[0]
        call_idx[0] += 1
        if idx >= n_frames:
            raise StopIteration("scripted capture exhausted")
        return dummy, dummy

    mock.read.side_effect = _read
    return mock


def _install_common_patches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    detections_per_frame: list[list[Detection]],
    capture_mock: MagicMock,
    mqtt_mock: MagicMock,
    now_func=None,
) -> list[list[Detection]]:
    """Wire up mocks for every external boundary of ``run_pipeline``.

    Returns the (possibly padded) detections script actually used.
    """
    # Capture
    monkeypatch.setattr("src.main.build_capture", lambda *a, **k: capture_mock)

    # MQTT (+ buffer)
    monkeypatch.setattr(
        "src.main.build_mqtt", lambda *a, **k: (mqtt_mock, MagicMock())
    )

    # Calibration: load_calibration returns our identity dict
    monkeypatch.setattr(
        "src.main.load_calibration", lambda path: _identity_calibration()
    )

    # Detection model: bypass load_model — a dummy dict is enough since
    # detect_persons is also patched.
    monkeypatch.setattr(
        "src.main.load_model",
        lambda *a, **k: {"backend": MagicMock(), "type": "opencv"},
    )

    # Disparity: fixed map for predictable depth.
    monkeypatch.setattr(
        "src.main.compute_disparity",
        lambda left, right, sgbm=None, downscale=1, use_clahe=True:
            _fixed_disparity_map(),
    )

    # Detector: scripted per-frame detections. Pad with [] so the loop can
    # run as many iterations as the capture feeds.
    script = list(detections_per_frame)
    frame_idx = [0]

    def _fake_detect(frame, model, confidence_threshold=0.5, nms_threshold=0.45):
        i = frame_idx[0]
        frame_idx[0] += 1
        if i < len(script):
            return list(script[i])
        return []

    monkeypatch.setattr("src.main.detect_persons", _fake_detect)

    # Disable SIGINT/SIGTERM handler installation — pytest runs on the main
    # thread on Windows, where signal.signal(SIGTERM, ...) raises.
    monkeypatch.setattr("src.main.signal.signal", lambda *a, **k: None)

    if now_func is not None:
        monkeypatch.setattr("src.main.time.time", now_func)

    return script


def _counting_events(mqtt_mock: MagicMock) -> list[dict[str, Any]]:
    """Extract all publish_event('counting', payload) payloads."""
    out: list[dict[str, Any]] = []
    for call in mqtt_mock.publish_event.call_args_list:
        args, kwargs = call
        topic = args[0] if args else kwargs.get("event_type")
        data = args[1] if len(args) > 1 else kwargs.get("data")
        if topic == "counting":
            out.append(data)
    return out


def _telemetry_payloads(mqtt_mock: MagicMock) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in mqtt_mock.publish_event.call_args_list:
        args, _kwargs = call
        if args and args[0] == "telemetry":
            out.append(args[1])
    return out


# ---------------------------------------------------------------------------
# Test 1 — ingress event flow
# ---------------------------------------------------------------------------


def test_e2e_ingress_event_published(monkeypatch: pytest.MonkeyPatch) -> None:
    """A person walking top→bottom through the ROI emits one ingress event."""
    tmpdir = tempfile.mkdtemp()
    config = _make_config(tmpdir)

    # Scripted trajectory (cx=320 kept constant; cy walks top→bottom).
    # 3 frames outside (above) to reach CONFIRMED, then enter ROI from
    # side_a (y<240), cross line, exit ROI below at side_b (y>300).
    trajectory_y = [50, 100, 150, 190, 230, 260, 295, 320]
    detections = [[_det_at(320, y)] for y in trajectory_y]

    capture = _make_scripted_capture(len(detections))
    mqtt = MagicMock()

    _install_common_patches(
        monkeypatch,
        detections_per_frame=detections,
        capture_mock=capture,
        mqtt_mock=mqtt,
    )

    args = argparse.Namespace(
        config="/tmp/ignored.yaml",
        replay_dir=None,
        detection_backend="opencv",
    )

    run_pipeline(config, args)

    events = _counting_events(mqtt)
    assert len(events) == 1, f"expected 1 counting event, got {events}"
    event = events[0]
    assert event["direction"] == "ingress"
    assert event["total_in"] == 1
    assert event["total_out"] == 0
    assert event["scaled_in"] == 1
    assert "event_time" in event
    assert "track_id" in event


# ---------------------------------------------------------------------------
# Test 2 — indeciso (enters ROI, turns back) emits NO counting event
# ---------------------------------------------------------------------------


def test_e2e_indeciso_no_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A person entering but turning back on the same side is not counted.

    Trajectory: approaches from above (3 frames to reach CONFIRMED), dips
    into the ROI on side_a only, never crosses the line, then exits on
    the same side. ROICounter must emit zero counting events.
    """
    tmpdir = tempfile.mkdtemp()
    config = _make_config(tmpdir)

    trajectory_y = [50, 100, 150, 190, 210, 230, 220, 200, 150]
    detections = [[_det_at(320, y)] for y in trajectory_y]

    capture = _make_scripted_capture(len(detections))
    mqtt = MagicMock()

    _install_common_patches(
        monkeypatch,
        detections_per_frame=detections,
        capture_mock=capture,
        mqtt_mock=mqtt,
    )

    args = argparse.Namespace(
        config="/tmp/ignored.yaml",
        replay_dir=None,
        detection_backend="opencv",
    )

    run_pipeline(config, args)

    assert _counting_events(mqtt) == [], (
        "indeciso (same-side entry/exit) must not emit a counting event"
    )

    # Pipeline still ran normally — capture and detector were driven.
    assert capture.read.call_count >= len(detections)


# ---------------------------------------------------------------------------
# Test 3 — telemetry fires on interval
# ---------------------------------------------------------------------------


def test_e2e_telemetry_fires_on_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry is published once the configured interval elapses.

    We drive ``src.main.time.time`` with a synthetic monotonically-
    increasing clock (10s per call) and configure a 5s telemetry interval
    so at least one telemetry message must be emitted.
    """
    tmpdir = tempfile.mkdtemp()
    config = _make_config(tmpdir)
    config["telemetry"]["interval_seconds"] = 5

    # No persons visible — keep the scenario simple.
    detections = [[] for _ in range(6)]

    capture = _make_scripted_capture(len(detections))
    mqtt = MagicMock()

    # Fake clock: advances 10 s per call starting at t=1_000_000.
    clock = [1_000_000.0]

    def _now():
        clock[0] += 10.0
        return clock[0]

    _install_common_patches(
        monkeypatch,
        detections_per_frame=detections,
        capture_mock=capture,
        mqtt_mock=mqtt,
        now_func=_now,
    )

    args = argparse.Namespace(
        config="/tmp/ignored.yaml",
        replay_dir=None,
        detection_backend="opencv",
    )

    run_pipeline(config, args)

    telem = _telemetry_payloads(mqtt)
    assert len(telem) >= 1, f"expected >=1 telemetry publish, got {telem}"

    payload = telem[0]
    # Keys produced unconditionally by run_pipeline telemetry branch.
    for key in ("fps", "total_in", "total_out"):
        assert key in payload, f"telemetry payload missing {key}: {payload}"
    # Keys from get_telemetry() — the set depends on the host OS, but on
    # any platform at least uptime_s is attempted.
    assert "uptime_s" in payload

    # No counting events — nobody was detected.
    assert _counting_events(mqtt) == []

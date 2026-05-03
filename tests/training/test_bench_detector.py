"""Tests for scripts/training/bench_detector.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "bench_detector",
    Path(__file__).resolve().parents[2]
    / "scripts" / "training" / "bench_detector.py",
)
bench_detector = importlib.util.module_from_spec(_SPEC)  # type: ignore
sys.modules["bench_detector"] = bench_detector
_SPEC.loader.exec_module(bench_detector)  # type: ignore


# ---------------------------------------------------------------------------
# _zone_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xc,yc,expected", [
    (640, 360, "center"),   # dead center of a 1280x720 frame
    (100, 100, "tl"),
    (1100, 100, "tr"),
    (100, 600, "bl"),
    (1100, 600, "br"),
])
def test_zone_of_classifies_5_regions(xc, yc, expected):
    assert bench_detector._zone_of(xc, yc, 1280, 720) == expected


def test_zone_of_handles_zero_dimensions():
    """No div-by-zero if a degenerate frame slips through."""
    assert bench_detector._zone_of(0, 0, 0, 0) == "tl"


# ---------------------------------------------------------------------------
# run_bench (mocked ultralytics)
# ---------------------------------------------------------------------------


def _make_image_dir(tmp_path: Path, n: int = 3) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        (d / f"f{i}.jpg").write_bytes(b"x")
    # Non-image must be skipped
    (d / "notes.txt").write_text("ignore")
    return d


def _fake_box_set(boxes_xyxy, classes, confidences, w=1280, h=720):
    """Build a MagicMock that mimics ultralytics' Boxes object."""
    import numpy as np

    boxes = MagicMock()
    boxes.__len__ = lambda self: len(boxes_xyxy)
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        boxes_xyxy, dtype=float)
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array(
        classes, dtype=float)
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array(
        confidences, dtype=float)
    result = MagicMock()
    result.boxes = boxes
    result.orig_shape = (h, w)
    return result


def test_run_bench_aggregates_per_zone(tmp_path):
    frames = _make_image_dir(tmp_path, n=3)

    # Fake ultralytics: 3 frames, with detections at 3 different zones.
    fake_results = [
        [_fake_box_set([[600, 340, 680, 380]], [0], [0.92])],   # center
        [_fake_box_set([[50, 50, 150, 150]], [0], [0.55])],     # tl
        [_fake_box_set([[1100, 600, 1200, 700]], [0], [0.71])], # br
    ]
    fake_model = MagicMock()
    fake_model.predict.side_effect = fake_results
    fake_yolo = MagicMock(return_value=fake_model)

    fake_module = MagicMock()
    fake_module.YOLO = fake_yolo

    with patch.dict(sys.modules, {"ultralytics": fake_module}):
        report = bench_detector.run_bench(
            weights=tmp_path / "fake.pt",
            frames_dir=frames, conf=0.25, imgsz=640,
        )

    s = report["summary"]
    assert s["n_frames"] == 3
    assert s["frames_with_detections"] == 3
    assert s["detection_rate"] == 1.0
    assert s["total_detections"] == 3
    assert s["zone_counts"]["center"] == 1
    assert s["zone_counts"]["tl"] == 1
    assert s["zone_counts"]["br"] == 1
    assert s["zone_counts"]["tr"] == 0
    assert s["zone_counts"]["bl"] == 0
    assert s["confidence"]["min"] == pytest.approx(0.55)
    assert s["confidence"]["max"] == pytest.approx(0.92)


def test_run_bench_handles_empty_detections(tmp_path):
    """A frame with zero detections must not crash, and should be counted
    as a miss in `frames_with_detections`."""
    frames = _make_image_dir(tmp_path, n=2)

    empty = MagicMock()
    empty.boxes = None
    empty.orig_shape = (720, 1280)

    fake_model = MagicMock()
    fake_model.predict.side_effect = [[empty], [empty]]
    fake_module = MagicMock()
    fake_module.YOLO = MagicMock(return_value=fake_model)

    with patch.dict(sys.modules, {"ultralytics": fake_module}):
        report = bench_detector.run_bench(
            weights=tmp_path / "fake.pt",
            frames_dir=frames, conf=0.25,
        )

    s = report["summary"]
    assert s["frames_with_detections"] == 0
    assert s["detection_rate"] == 0.0
    assert s["total_detections"] == 0
    assert s["confidence"] is None


def test_run_bench_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    fake_module = MagicMock()
    fake_module.YOLO = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"ultralytics": fake_module}):
        with pytest.raises(SystemExit, match="No images"):
            bench_detector.run_bench(
                weights=tmp_path / "f.pt",
                frames_dir=empty, conf=0.25,
            )


def test_run_bench_missing_dir_raises(tmp_path):
    fake_module = MagicMock()
    fake_module.YOLO = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"ultralytics": fake_module}):
        with pytest.raises(SystemExit, match="not found"):
            bench_detector.run_bench(
                weights=tmp_path / "f.pt",
                frames_dir=tmp_path / "does-not-exist", conf=0.25,
            )


# ---------------------------------------------------------------------------
# diff_reports
# ---------------------------------------------------------------------------


def test_diff_reports_runs_without_error(tmp_path, capsys):
    """Smoke test — diff reads two real JSON files and prints a summary."""
    a = {
        "summary": {
            "weights": "a.pt", "n_frames": 10, "frames_with_detections": 4,
            "detection_rate": 0.4, "total_detections": 5,
            "zone_counts": {"center": 2, "tl": 1, "tr": 1, "bl": 1, "br": 0},
            "confidence": {"mean": 0.5, "median": 0.5,
                           "min": 0.3, "max": 0.7, "stdev": 0.1},
        },
        "per_frame": [],
    }
    b = {
        "summary": {
            "weights": "b.pt", "n_frames": 10, "frames_with_detections": 9,
            "detection_rate": 0.9, "total_detections": 11,
            "zone_counts": {"center": 5, "tl": 2, "tr": 2, "bl": 1, "br": 1},
            "confidence": {"mean": 0.78, "median": 0.8,
                           "min": 0.4, "max": 0.95, "stdev": 0.12},
        },
        "per_frame": [],
    }
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))

    bench_detector.diff_reports(tmp_path / "a.json", tmp_path / "b.json")
    out = capsys.readouterr().out
    assert "frames_with_detections" in out
    assert "detection_rate" in out
    assert "zone.center" in out
    assert "+0.500" in out  # detection_rate delta 0.4 -> 0.9 = +0.500

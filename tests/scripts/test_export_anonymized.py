"""Tests for scripts/export_anonymized.py.

The anonymisation script is the last technical guardrail before frames
leave the device — the targeted-blur path must measurably alter the
bbox region, and the fallback must heavily blur the entire frame.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_anonymized.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_anonymized",
        _SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


anon = _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sharp_image(h: int = 100, w: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed=123)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _write_jpg(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_blur_bbox_modifies_bbox_only():
    img = _sharp_image()
    bbox = (50, 25, 150, 75)
    out = anon.blur_bbox(img, bbox, kernel=21)
    # Outside bbox: identical pixels.
    np.testing.assert_array_equal(out[:25, :], img[:25, :])
    np.testing.assert_array_equal(out[75:, :], img[75:, :])
    np.testing.assert_array_equal(out[:, :50], img[:, :50])
    np.testing.assert_array_equal(out[:, 150:], img[:, 150:])
    # Inside bbox: should differ from original (blur changed values).
    assert not np.array_equal(out[25:75, 50:150], img[25:75, 50:150])


def test_blur_bbox_off_screen_falls_back_to_full_blur():
    img = _sharp_image()
    bbox = (-100, -100, -50, -50)  # entirely off-screen
    out = anon.blur_bbox(img, bbox, kernel=21)
    # Result is heavily processed — many pixels differ.
    diff = (out.astype(int) - img.astype(int)).any(axis=2)
    # More than half the pixels changed (full-frame fallback).
    assert diff.mean() > 0.5


def test_blur_full_with_edges_obscures_image():
    img = _sharp_image()
    out = anon.blur_full_with_edges(img, kernel=31)
    assert out.shape == img.shape
    # The bulk of pixels differ from the original.
    diff = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 10
    assert diff.mean() > 0.5


def test_ensure_odd_pads_evens():
    assert anon._ensure_odd(4) == 5
    assert anon._ensure_odd(31) == 31
    assert anon._ensure_odd(0) == 3  # min clamp


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


def test_extract_bbox_list_form():
    assert anon.extract_bbox({"bbox": [1, 2, 3, 4]}) == (1, 2, 3, 4)


def test_extract_bbox_dict_form():
    bb = {"bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}
    assert anon.extract_bbox(bb) == (1, 2, 3, 4)


def test_extract_bbox_missing_returns_none():
    assert anon.extract_bbox({}) is None
    assert anon.extract_bbox({"bbox": "garbage"}) is None
    assert anon.extract_bbox({"bbox": [1, 2]}) is None


def test_load_sidecar_picks_up_double_suffix(tmp_path: Path):
    jpg = tmp_path / "frame.jpg"
    jpg.write_bytes(b"")
    sidecar = tmp_path / "frame.jpg.json"
    sidecar.write_text(json.dumps({"bbox": [1, 2, 3, 4]}))
    meta = anon.load_sidecar(jpg)
    assert meta == {"bbox": [1, 2, 3, 4]}


def test_load_sidecar_picks_up_simple_stem(tmp_path: Path):
    jpg = tmp_path / "frame.jpg"
    jpg.write_bytes(b"")
    sidecar = tmp_path / "frame.json"
    sidecar.write_text(json.dumps({"bbox": [10, 20, 30, 40]}))
    meta = anon.load_sidecar(jpg)
    assert meta == {"bbox": [10, 20, 30, 40]}


def test_load_sidecar_malformed_returns_none(tmp_path: Path):
    jpg = tmp_path / "frame.jpg"
    jpg.write_bytes(b"")
    (tmp_path / "frame.jpg.json").write_text("{not valid json")
    assert anon.load_sidecar(jpg) is None


# ---------------------------------------------------------------------------
# anonymise_one
# ---------------------------------------------------------------------------


def test_anonymise_one_targeted(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = _sharp_image()
    in_jpg = in_dir / "f.jpg"
    _write_jpg(in_jpg, img)
    (in_dir / "f.jpg.json").write_text(json.dumps({"bbox": [50, 25, 150, 75]}))

    status = anon.anonymise_one(
        in_jpg,
        out_dir / "f.jpg",
        blur_kernel=21,
        canny_low=50,
        canny_high=150,
        quality=85,
        dry_run=False,
    )
    assert status == "targeted"
    assert (out_dir / "f.jpg").exists()


def test_anonymise_one_fallback_when_no_sidecar(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = _sharp_image()
    in_jpg = in_dir / "f.jpg"
    _write_jpg(in_jpg, img)
    status = anon.anonymise_one(
        in_jpg,
        out_dir / "f.jpg",
        blur_kernel=21,
        canny_low=50,
        canny_high=150,
        quality=85,
        dry_run=False,
    )
    assert status == "fallback"
    assert (out_dir / "f.jpg").exists()


def test_anonymise_one_dry_run_writes_nothing(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = _sharp_image()
    in_jpg = in_dir / "f.jpg"
    _write_jpg(in_jpg, img)
    status = anon.anonymise_one(
        in_jpg,
        out_dir / "f.jpg",
        blur_kernel=21,
        canny_low=50,
        canny_high=150,
        quality=85,
        dry_run=True,
    )
    assert status == "fallback"
    assert not (out_dir / "f.jpg").exists()


def test_anonymise_one_unreadable_returns_error(tmp_path: Path):
    in_jpg = tmp_path / "missing.jpg"
    status = anon.anonymise_one(
        in_jpg,
        tmp_path / "out.jpg",
        blur_kernel=21,
        canny_low=50,
        canny_high=150,
        quality=85,
        dry_run=False,
    )
    assert status == "error"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_processes_tree(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = _sharp_image()
    _write_jpg(in_dir / "a" / "1.jpg", img)
    _write_jpg(in_dir / "b" / "2.jpg", img)
    rc = anon.main(
        [
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "a" / "1.jpg").exists()
    assert (out_dir / "b" / "2.jpg").exists()


def test_main_returns_1_for_missing_input(tmp_path: Path):
    rc = anon.main(
        [
            "--input-dir",
            str(tmp_path / "nope"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1

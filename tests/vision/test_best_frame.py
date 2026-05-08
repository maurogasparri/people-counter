"""Tests for src/vision/best_frame.py.

Covers:
  - score_frame component math (confidence / area / centrality / sharpness).
  - pick_best on empty / single-element / multi-element buffers.
  - BufferedFrame container behaviour.
  - BestFrameManager.observe + commit + forget + gc lifecycle.
  - JPG write path with on-disk tmp dir.
  - Privacy invariants: feature stays inert when the manager isn't built;
    the manager never silently writes to the wrong place.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.best_frame import (
    BestFrameManager,
    BufferedFrame,
    pick_best,
    score_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 100, w: int = 200, *, sharp: bool = True) -> np.ndarray:
    """Synthetic BGR frame. ``sharp=False`` produces a flat gray frame so
    the Laplacian variance is ~0 (motion blur fixture)."""
    if sharp:
        # High-contrast checkerboard — Laplacian responds.
        rng = np.random.default_rng(seed=42)
        return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return np.full((h, w, 3), 128, dtype=np.uint8)


WEIGHTS = {
    "confidence_weight": 0.4,
    "bbox_area_weight": 0.2,
    "centrality_weight": 0.2,
    "sharpness_weight": 0.2,
}


# ---------------------------------------------------------------------------
# score_frame
# ---------------------------------------------------------------------------


def test_score_components_in_unit_range():
    img = _make_frame(100, 200)
    bbox = (50, 25, 150, 75)  # centered, half the frame
    score, comps = score_frame(img, bbox, 0.9, img.shape[:2], WEIGHTS)
    for k, v in comps.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
    assert 0.0 <= score <= 1.0


def test_score_higher_confidence_wins_ceteris_paribus():
    img = _make_frame()
    bbox = (50, 25, 150, 75)
    s_low, _ = score_frame(img, bbox, 0.2, img.shape[:2], WEIGHTS)
    s_high, _ = score_frame(img, bbox, 0.9, img.shape[:2], WEIGHTS)
    assert s_high > s_low


def test_score_centred_bbox_beats_corner():
    img = _make_frame(100, 200)
    centred = (90, 40, 110, 60)
    corner = (0, 0, 20, 20)
    _, c_centred = score_frame(img, centred, 0.5, img.shape[:2], WEIGHTS)
    _, c_corner = score_frame(img, corner, 0.5, img.shape[:2], WEIGHTS)
    assert c_centred["centrality"] > c_corner["centrality"]


def test_score_larger_bbox_higher_area_component():
    img = _make_frame(100, 200)
    big = (10, 10, 190, 90)
    small = (90, 40, 110, 60)
    _, c_big = score_frame(img, big, 0.5, img.shape[:2], WEIGHTS)
    _, c_small = score_frame(img, small, 0.5, img.shape[:2], WEIGHTS)
    assert c_big["bbox_area"] > c_small["bbox_area"]


def test_sharpness_drops_for_flat_frame():
    sharp_img = _make_frame(sharp=True)
    flat_img = _make_frame(sharp=False)
    bbox = (10, 10, 190, 90)
    _, c_sharp = score_frame(sharp_img, bbox, 0.5, sharp_img.shape[:2], WEIGHTS)
    _, c_flat = score_frame(flat_img, bbox, 0.5, flat_img.shape[:2], WEIGHTS)
    assert c_sharp["sharpness"] > c_flat["sharpness"]
    # Flat frame's sharpness is ~0
    assert c_flat["sharpness"] < 0.05


def test_score_handles_oob_bbox_gracefully():
    img = _make_frame()
    # Bbox completely outside the frame: the sharpness crop has no
    # pixels (clamped to empty) so its component goes to 0; area is
    # still computed from raw bbox extent (we don't try to "intersect"
    # with the frame) and may be non-zero. The whole call must just not
    # raise.
    bbox = (-50, -50, -10, -10)
    score, comps = score_frame(img, bbox, 0.5, img.shape[:2], WEIGHTS)
    assert comps["sharpness"] == 0.0
    assert comps["bbox_area"] >= 0.0
    assert score >= 0.0


def test_score_inverted_bbox_area_zero():
    """An inverted bbox (x2<x1) has zero width — area must clamp to 0."""
    img = _make_frame()
    bbox = (150, 75, 50, 25)  # inverted
    _, comps = score_frame(img, bbox, 0.5, img.shape[:2], WEIGHTS)
    assert comps["bbox_area"] == 0.0


def test_score_zero_frame_dim_returns_zeros():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    score, comps = score_frame(img, (0, 0, 1, 1), 0.5, (0, 0), WEIGHTS)
    assert score == 0.5 * WEIGHTS["confidence_weight"]
    assert comps["bbox_area"] == 0.0
    assert comps["centrality"] == 0.0


def test_confidence_clipped_to_unit_range():
    img = _make_frame()
    bbox = (10, 10, 90, 90)
    s_above, _ = score_frame(img, bbox, 5.0, img.shape[:2], WEIGHTS)
    s_one, _ = score_frame(img, bbox, 1.0, img.shape[:2], WEIGHTS)
    assert s_above == pytest.approx(s_one)
    s_neg, _ = score_frame(img, bbox, -0.5, img.shape[:2], WEIGHTS)
    s_zero, _ = score_frame(img, bbox, 0.0, img.shape[:2], WEIGHTS)
    assert s_neg == pytest.approx(s_zero)


# ---------------------------------------------------------------------------
# pick_best
# ---------------------------------------------------------------------------


def test_pick_best_empty_returns_none():
    assert pick_best([]) is None


def test_pick_best_single_returns_it():
    bf = BufferedFrame(
        frame_image=_make_frame(),
        bbox=(0, 0, 10, 10),
        confidence=0.5,
        score=0.4,
    )
    assert pick_best([bf]) is bf


def test_pick_best_picks_highest_score():
    bfs = [
        BufferedFrame(_make_frame(), (0, 0, 5, 5), 0.5, 0.2, timestamp=1.0),
        BufferedFrame(_make_frame(), (0, 0, 5, 5), 0.5, 0.8, timestamp=2.0),
        BufferedFrame(_make_frame(), (0, 0, 5, 5), 0.5, 0.3, timestamp=3.0),
    ]
    assert pick_best(bfs) is bfs[1]


def test_pick_best_breaks_ties_by_recency():
    bfs = [
        BufferedFrame(_make_frame(), (0, 0, 5, 5), 0.5, 0.5, timestamp=1.0),
        BufferedFrame(_make_frame(), (0, 0, 5, 5), 0.5, 0.5, timestamp=10.0),
        BufferedFrame(_make_frame(), (0, 0, 5, 5), 0.5, 0.5, timestamp=5.0),
    ]
    assert pick_best(bfs) is bfs[1]


# ---------------------------------------------------------------------------
# BestFrameManager
# ---------------------------------------------------------------------------


def test_manager_observe_buffers_per_track(tmp_path):
    mgr = BestFrameManager(tmp_path, buffer_size=3)
    img = _make_frame()
    mgr.observe(track_id=1, frame=img, bbox=(10, 10, 90, 90), confidence=0.5)
    mgr.observe(track_id=2, frame=img, bbox=(10, 10, 90, 90), confidence=0.6)
    # Internal access for testing — buffers are indexed by track_id.
    assert 1 in mgr._buffers
    assert 2 in mgr._buffers
    assert len(mgr._buffers[1].frames) == 1


def test_manager_buffer_caps_at_size(tmp_path):
    mgr = BestFrameManager(tmp_path, buffer_size=3)
    img = _make_frame()
    for _ in range(5):
        mgr.observe(1, img, (10, 10, 90, 90), 0.5)
    assert len(mgr._buffers[1].frames) == 3


def test_manager_observe_skips_empty_frame(tmp_path):
    mgr = BestFrameManager(tmp_path, buffer_size=3)
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    mgr.observe(1, empty, (0, 0, 1, 1), 0.5)
    # Empty frame returns early — no buffer created.
    assert 1 not in mgr._buffers


def test_manager_commit_writes_jpg(tmp_path):
    mgr = BestFrameManager(tmp_path, buffer_size=5, jpeg_quality=90)
    img = _make_frame()
    mgr.observe(7, img, (10, 10, 90, 90), 0.4)
    mgr.observe(7, img, (10, 10, 90, 90), 0.9)  # higher conf -> higher score

    path = mgr.commit(7, event_timestamp=1700000000.0)
    assert path is not None
    p = Path(path)
    assert p.exists()
    assert p.suffix == ".jpg"
    # Day directory is in the path (local time, format YYYY-MM-DD).
    assert p.parent.parent == tmp_path
    # Buffer is dropped after commit.
    assert 7 not in mgr._buffers


def test_manager_commit_empty_buffer_returns_none(tmp_path):
    mgr = BestFrameManager(tmp_path)
    assert mgr.commit(99, time.time()) is None


def test_manager_forget_drops_buffer(tmp_path):
    mgr = BestFrameManager(tmp_path)
    mgr.observe(3, _make_frame(), (10, 10, 90, 90), 0.5)
    assert 3 in mgr._buffers
    mgr.forget(3)
    assert 3 not in mgr._buffers


def test_manager_gc_drops_dead_tracks(tmp_path):
    mgr = BestFrameManager(tmp_path)
    img = _make_frame()
    mgr.observe(1, img, (10, 10, 90, 90), 0.5)
    mgr.observe(2, img, (10, 10, 90, 90), 0.5)
    mgr.observe(3, img, (10, 10, 90, 90), 0.5)
    dropped = mgr.gc({2})
    assert dropped == 2
    assert set(mgr._buffers) == {2}


def test_manager_filename_layout(tmp_path):
    mgr = BestFrameManager(tmp_path)
    mgr.observe(42, _make_frame(), (10, 10, 90, 90), 0.5)
    path = mgr.commit(42, event_timestamp=1700000000.0)
    assert path is not None
    p = Path(path)
    # <tmp>/<YYYY-MM-DD>/<track_id>_<ts>.jpg
    parts = p.relative_to(tmp_path).parts
    assert len(parts) == 2
    day_dir, fname = parts
    assert len(day_dir) == 10  # YYYY-MM-DD
    assert fname.startswith("42_")
    assert fname.endswith(".jpg")


def test_manager_observe_does_not_mutate_input_frame(tmp_path):
    mgr = BestFrameManager(tmp_path)
    img = _make_frame()
    snapshot = img.copy()
    mgr.observe(1, img, (10, 10, 90, 90), 0.5)
    # Pipeline may annotate `img` in-place after observe; the buffered
    # copy must not share storage.
    img[:] = 0
    buffered = mgr._buffers[1].frames[0].frame_image
    np.testing.assert_array_equal(buffered, snapshot)


def test_manager_commit_failure_returns_none(tmp_path, monkeypatch):
    """When cv2.imwrite fails (read-only fs, disk full), commit must
    return None instead of raising — pipeline must keep counting."""
    mgr = BestFrameManager(tmp_path)
    mgr.observe(1, _make_frame(), (10, 10, 90, 90), 0.5)

    def _failing_imwrite(*args, **kwargs):
        return False

    monkeypatch.setattr(cv2, "imwrite", _failing_imwrite)
    path = mgr.commit(1, event_timestamp=time.time())
    assert path is None


def test_manager_jpeg_quality_clamped(tmp_path):
    """Out-of-range jpeg_quality is silently clamped to [1, 100]."""
    mgr = BestFrameManager(tmp_path, jpeg_quality=999)
    assert mgr._jpeg_quality == 100
    mgr2 = BestFrameManager(tmp_path, jpeg_quality=-5)
    assert mgr2._jpeg_quality == 1

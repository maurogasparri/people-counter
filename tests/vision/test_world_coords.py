"""Tests for world-space helpers (head-height classification)."""

import numpy as np
import pytest

from src.vision.depth import min_depth_at_bbox
from src.vision.world_coords import (
    aggregate_height_class,
    classify_height,
    head_height_above_floor,
)


class TestHeadHeightAboveFloor:
    def test_adult_head_under_3_5m_mount(self):
        # Camera at 3500mm, head at top of box returns depth 1800mm
        # → head_height = 3500 - 1800 = 1700mm (adult)
        assert head_height_above_floor(1800.0, 3500.0) == 1700.0

    def test_child_head_gives_smaller_height(self):
        # Shorter person: depth 2200mm under same mount → 1300mm (child)
        assert head_height_above_floor(2200.0, 3500.0) == 1300.0

    def test_invalid_depth_returns_none(self):
        assert head_height_above_floor(0.0, 3500.0) is None
        assert head_height_above_floor(-100.0, 3500.0) is None

    def test_invalid_mount_returns_none(self):
        assert head_height_above_floor(1800.0, 0.0) is None

    def test_negative_height_returns_none(self):
        """Depth > mount means the 'head' is below the floor — bad data."""
        assert head_height_above_floor(4000.0, 3500.0) is None


class TestClassifyHeight:
    def test_tall_person_is_adult(self):
        assert classify_height(1700.0, adult_min_mm=1500.0) == "adult"

    def test_short_person_is_child(self):
        assert classify_height(1200.0, adult_min_mm=1500.0) == "child"

    def test_exactly_at_threshold_is_adult(self):
        """Threshold is inclusive on the adult side — avoids "unknown" gap."""
        assert classify_height(1500.0, adult_min_mm=1500.0) == "adult"

    def test_none_is_unknown(self):
        assert classify_height(None, adult_min_mm=1500.0) == "unknown"

    def test_threshold_is_configurable(self):
        # 1400mm is adult with a 1300mm threshold, child with a 1500mm one
        assert classify_height(1400.0, 1300.0) == "adult"
        assert classify_height(1400.0, 1500.0) == "child"


class TestAggregateHeightClass:
    def test_majority_adult(self):
        samples = ["adult", "adult", "adult", "child"]
        assert aggregate_height_class(samples) == "adult"

    def test_majority_child(self):
        samples = ["child", "child", "adult"]
        assert aggregate_height_class(samples) == "child"

    def test_unknown_samples_ignored(self):
        samples = ["unknown", "unknown", "child", "child", "unknown"]
        assert aggregate_height_class(samples) == "child"

    def test_all_unknown(self):
        assert aggregate_height_class(["unknown"] * 5) == "unknown"
        assert aggregate_height_class([]) == "unknown"

    def test_tie_goes_to_most_recent(self):
        # Equal counts: last non-unknown wins (bias toward latest stable bbox)
        assert aggregate_height_class(["adult", "child"]) == "child"
        assert aggregate_height_class(["child", "adult"]) == "adult"


class TestMinDepthAtBbox:
    def test_low_percentile_robust_to_speckle_noise(self):
        """Low percentile should reject a single noisy min pixel."""
        depth = np.full((100, 100), 2000.0, dtype=np.float32)
        depth[50, 50] = 100.0  # speckle — very near
        result = min_depth_at_bbox(depth, (40, 40, 60, 60), low_percentile=5.0)
        # 5% percentile of 400 pixels = 20 pixels at the low end; 1 outlier
        # doesn't move the result much
        assert result > 500, f"got {result}, expected >500 (speckle rejected)"

    def test_uniform_depth_returns_that_depth(self):
        depth = np.full((100, 100), 2400.0, dtype=np.float32)
        assert abs(min_depth_at_bbox(depth, (10, 10, 90, 90)) - 2400.0) < 1

    def test_invalid_pixels_excluded(self):
        depth = np.zeros((50, 50), dtype=np.float32)
        depth[20:30, 20:30] = 1500.0
        assert min_depth_at_bbox(depth, (0, 0, 50, 50)) > 0  # zeros skipped

    def test_no_valid_returns_zero(self):
        depth = np.zeros((50, 50), dtype=np.float32)
        assert min_depth_at_bbox(depth, (0, 0, 50, 50)) == 0.0

    def test_is_smaller_than_median(self):
        """Basic sanity: near-depth (low percentile) < median depth."""
        depth = np.zeros((100, 100), dtype=np.float32)
        for row in range(100):
            # Vertical depth gradient — head at top (small depth)
            depth[row, :] = 1500.0 + row * 5.0
        from src.vision.depth import depth_at_bbox
        near = min_depth_at_bbox(depth, (0, 0, 100, 100))
        median = depth_at_bbox(depth, (0, 0, 100, 100))
        assert near < median

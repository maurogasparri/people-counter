"""Tests for world-space helpers (head-height classification)."""

import numpy as np
import pytest

from src.vision.depth import head_depth_in_bbox, min_depth_at_bbox
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

    def test_samples_central_crop_only(self):
        """Peripheral pixels (closer to camera) outside the central 50%
        of the bbox must NOT pull the near-depth down. In zenith view
        the head sits near the bbox centroid; peripheral noise from
        background pixels inside the YOLO-stock person bbox is what
        was producing 1.88 m heights for a sitting user.
        """
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        # Spurious very-near pixels along the bbox edges (peripheral).
        depth[0:10, :] = 700.0
        depth[90:100, :] = 700.0
        depth[:, 0:10] = 700.0
        depth[:, 90:100] = 700.0
        # Bbox covers the whole array. Central 50% sees only 1500.
        result = min_depth_at_bbox(depth, (0, 0, 100, 100))
        assert abs(result - 1500.0) < 1, (
            f"got {result}, expected ~1500 (peripheral noise should be "
            "ignored by central-crop sampling)"
        )

    def test_default_percentile_robust_to_speckle_cluster(self):
        """A small cluster of low-depth speckle pixels in the central
        crop must not dominate the result. The 15% default is meant to
        survive ~10% noise contamination at SGBM downscale=8.
        """
        # 50×50 central crop of a 100×100 bbox. Inject a 4×4 (16 px)
        # cluster of bad-depth speckle = ~6.4% of central pixels.
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        depth[40:44, 40:44] = 600.0
        result = min_depth_at_bbox(depth, (0, 0, 100, 100))
        # 15th percentile of [600×16, 1500×(2500-16)] is 1500.
        assert result > 1400


class TestHeadDepthInBbox:
    """head_depth_in_bbox finds the head cluster (smallest plausible
    depth with enough area) inside a YOLO-COCO bbox that is centered on
    the torso, not the head."""

    MOUNT_MM = 2560.0  # mounting height used in tests below

    def test_returns_none_on_empty_depth_map(self):
        depth = np.zeros((0, 0), dtype=np.float32)
        assert head_depth_in_bbox(
            depth, (0, 0, 10, 10), self.MOUNT_MM,
        ) is None

    def test_returns_none_when_bbox_outside_frame(self):
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        # Negative-area bbox (clipped to nothing).
        assert head_depth_in_bbox(
            depth, (50, 50, 40, 40), self.MOUNT_MM,
        ) is None

    def test_returns_none_when_only_speckle_present(self):
        """Below-floor speckle alone (no real head cluster) → None.

        At mount=2.56m, max_head_height=1.80m → near gate at 760mm.
        Speckle at 200mm is below the gate and excluded; no remaining
        pixels above the gate (whole bbox is speckle).
        """
        depth = np.full((50, 50), 200.0, dtype=np.float32)
        result = head_depth_in_bbox(depth, (0, 0, 50, 50), self.MOUNT_MM)
        assert result is None

    def test_finds_sitting_user_head_not_torso(self):
        """Sitting user: head at ~1.10 m floor (depth 1450 mm), torso at
        ~0.80 m floor (depth 1760 mm). Function must return ~1450.
        """
        depth = np.full((100, 100), 1760.0, dtype=np.float32)
        # Head blob 7×7 = 49 px, comfortably above the 40-px area gate.
        depth[20:27, 40:47] = 1450.0
        result = head_depth_in_bbox(depth, (0, 0, 100, 100), self.MOUNT_MM)
        assert result is not None
        assert abs(result - 1450.0) < 60.0  # within one slice (100mm)

    def test_finds_standing_adult_head(self):
        """Standing user 1.68 m tall: head at depth ~880 mm. Torso at
        depth ~1160 mm. Returns ~880.
        """
        depth = np.full((100, 100), 1160.0, dtype=np.float32)
        depth[10:18, 35:45] = 880.0  # head, ~80 px
        result = head_depth_in_bbox(depth, (0, 0, 100, 100), self.MOUNT_MM)
        assert result is not None
        assert abs(result - 880.0) < 60.0

    def test_rejects_below_anthropometric_floor(self):
        """A spurious very-near cluster is below the anthropometric
        ceiling and must be gated out. At mount=2.56m,
        max_head_height=1.80m → near gate at 760mm. A cluster at 300mm
        depth corresponds to a 2.26m-tall head, exceeds the cap, and
        falls outside the gate.
        """
        depth = np.full((100, 100), 1500.0, dtype=np.float32)  # plausible head
        depth[10:25, 10:25] = 300.0  # impossible (head 2.26m), excluded
        result = head_depth_in_bbox(depth, (0, 0, 100, 100), self.MOUNT_MM)
        # The 300mm cluster is gated out, so we get the 1500mm plane.
        assert result is not None
        assert abs(result - 1500.0) < 100.0

    def test_speckle_smaller_than_min_area_does_not_win(self):
        """A 3×3 cluster (9 px) of false-near depth must be rejected
        even though it sits closer to the camera than the real head —
        below the min_head_area_px (40) gate, it's noise.
        """
        depth = np.full((100, 100), 1500.0, dtype=np.float32)  # head plane
        depth[5:8, 5:8] = 1100.0  # 9 px speckle, closer (within gate)
        result = head_depth_in_bbox(depth, (0, 0, 100, 100), self.MOUNT_MM)
        assert result is not None
        # Should land on the 1500 mm head plane, not the 1100 speckle.
        assert result > 1300

    def test_picks_head_when_torso_dominates_area(self):
        """Realistic case: bbox is mostly torso (more pixels), but head
        is the closest plausible cluster. We pick head, not torso —
        that's the whole point of the algorithm.
        """
        depth = np.full((100, 100), 1700.0, dtype=np.float32)  # torso
        # Head cluster 7×7 = 49 px — closer, comfortably above area gate.
        depth[20:27, 43:50] = 1200.0
        result = head_depth_in_bbox(depth, (0, 0, 100, 100), self.MOUNT_MM)
        assert result is not None
        assert abs(result - 1200.0) < 100.0

    def test_invalid_pixels_ignored(self):
        """Zero pixels (SGBM invalid) must not be counted toward the
        head-area gate or pollute the median.
        """
        depth = np.zeros((100, 100), dtype=np.float32)  # all invalid
        depth[40:60, 40:60] = 1500.0  # one valid plane
        result = head_depth_in_bbox(depth, (0, 0, 100, 100), self.MOUNT_MM)
        assert result is not None
        assert abs(result - 1500.0) < 100.0

    def test_returns_none_with_no_valid_pixels(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        assert head_depth_in_bbox(
            depth, (0, 0, 100, 100), self.MOUNT_MM,
        ) is None

    def test_mount_height_zero_returns_none(self):
        # Pathological config: mount=0 collapses the gate and there's
        # no plausible region. Must not crash.
        depth = np.full((50, 50), 1500.0, dtype=np.float32)
        result = head_depth_in_bbox(depth, (0, 0, 50, 50), 0.0)
        assert result is None

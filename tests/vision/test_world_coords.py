"""Tests for world-space helpers (head-height classification)."""

import numpy as np
import pytest

from src.vision.depth import head_depth_in_bbox, min_depth_at_bbox
from src.vision.world_coords import (
    aggregate_height_class,
    classify_height,
    head_height_above_floor,
    project_to_floor,
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


class TestProjectToFloor:
    """Footpoint projection (head pixel → foot pixel using head height).

    Verifies the parallax-correction trick used by the counter: knowing
    the actual head height above the floor (via SGBM depth) lets us scale
    the head pixel toward the principal point by ``Z_head / H`` to get
    the foot pixel. At nadir the scale is irrelevant; at the periphery
    the shift is the parallax error a centroid-based tracker would
    silently accumulate.
    """

    # Realistic geometry: 3 m mount, 1.7 m person, 1152×648 frame after
    # the standard rescale of the 2304×1296 calibration. Principal
    # point sits roughly at frame centre on rectified imagery.
    MOUNT_MM = 3000.0
    HEAD_MM = 1700.0
    CX = 576.0
    CY = 324.0

    def test_nadir_no_shift(self):
        """Head exactly at the principal point ⇒ scale is irrelevant,
        feet project on the same pixel. The geometric invariant the
        whole construction rests on."""
        u, v = project_to_floor(
            (self.CX, self.CY),
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        assert u == pytest.approx(self.CX)
        assert v == pytest.approx(self.CY)

    def test_periphery_shifts_toward_principal_point(self):
        """A head pixel offset from nadir scales TOWARD the principal
        point — the foot pixel is closer to centre than the head pixel
        by exactly the parallax factor.
        """
        # Head 200 px to the right of nadir.
        head_pixel = (self.CX + 200.0, self.CY)
        u, v = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        # Z_head/H = (3000-1700)/3000 = 0.4333..., so the offset shrinks
        # from 200 px to 200*0.4333 = 86.67 px.
        expected_offset = 200.0 * (self.MOUNT_MM - self.HEAD_MM) / self.MOUNT_MM
        assert u == pytest.approx(self.CX + expected_offset)
        assert v == pytest.approx(self.CY)
        # Foot pixel is closer to principal point than head pixel.
        assert abs(u - self.CX) < abs(head_pixel[0] - self.CX)

    def test_diagonal_periphery(self):
        """2D shift: both u and v scale by the same factor."""
        head_pixel = (self.CX + 300.0, self.CY - 150.0)
        u, v = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        scale = (self.MOUNT_MM - self.HEAD_MM) / self.MOUNT_MM
        assert u == pytest.approx(self.CX + 300.0 * scale)
        assert v == pytest.approx(self.CY + (-150.0) * scale)

    def test_zero_head_height_is_no_op(self):
        """No head height = no info. Returning the input lets the caller
        fall back to centroid-based crossing logic transparently."""
        head_pixel = (700.0, 200.0)
        u, v = project_to_floor(
            head_pixel,
            0.0,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        assert (u, v) == head_pixel

    def test_zero_mount_is_no_op(self):
        """Bad/missing mount config ⇒ no projection, no crash."""
        head_pixel = (700.0, 200.0)
        u, v = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            0.0,
            self.CX,
            self.CY,
        )
        assert (u, v) == head_pixel

    def test_negative_inputs_are_no_op(self):
        head_pixel = (700.0, 200.0)
        assert (
            project_to_floor(
                head_pixel,
                -5.0,
                self.MOUNT_MM,
                self.CX,
                self.CY,
            )
            == head_pixel
        )
        assert (
            project_to_floor(
                head_pixel,
                self.HEAD_MM,
                -1.0,
                self.CX,
                self.CY,
            )
            == head_pixel
        )

    def test_head_taller_than_mount_is_no_op(self):
        """Pathological: a taller-than-mount head means the camera is
        looking at a torso, not a head — the depth estimate must be
        wrong. Falling back to the head pixel keeps the counter from
        sign-flipping the parallax shift."""
        head_pixel = (800.0, 400.0)
        # Head at 3.5 m, mount only 3 m → Z_head = -500 mm.
        u, v = project_to_floor(
            head_pixel,
            3500.0,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        assert (u, v) == head_pixel

    def test_periphery_magnitude_matches_documented_geometry(self):
        """Sanity-check against the geometric numbers documented on the
        bug report: 1.7 m person under 3.0 m mount, head pixel near the
        frame border (~60° eccentricity from optical axis at this FOV).
        The shift is not exactly 110 cm here because we are computing in
        IMAGE space, not floor-plane metres — but the scale factor must
        be exactly Z_head/H."""
        # 500 px is roughly ~80% of the half-frame width on 1152×648,
        # i.e. nearly at the border.
        head_pixel = (self.CX + 500.0, self.CY)
        u_foot, _ = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        # The shift in image space relative to the head pixel:
        shift_px = head_pixel[0] - u_foot
        # Z_head/H = 1300/3000 ≈ 0.433, so foot is at offset 500*0.433 =
        # 216.67 from centre, shift = 500 - 216.67 = 283.33 px.
        assert shift_px == pytest.approx(500.0 - 500.0 * 1300.0 / 3000.0)


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
        assert (
            head_depth_in_bbox(
                depth,
                (0, 0, 10, 10),
                self.MOUNT_MM,
            )
            is None
        )

    def test_returns_none_when_bbox_outside_frame(self):
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        # Negative-area bbox (clipped to nothing).
        assert (
            head_depth_in_bbox(
                depth,
                (50, 50, 40, 40),
                self.MOUNT_MM,
            )
            is None
        )

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
        assert (
            head_depth_in_bbox(
                depth,
                (0, 0, 100, 100),
                self.MOUNT_MM,
            )
            is None
        )

    def test_mount_height_zero_returns_none(self):
        # Pathological config: mount=0 collapses the gate and there's
        # no plausible region. Must not crash.
        depth = np.full((50, 50), 1500.0, dtype=np.float32)
        result = head_depth_in_bbox(depth, (0, 0, 50, 50), 0.0)
        assert result is None

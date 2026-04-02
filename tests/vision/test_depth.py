"""Tests for disparity and depth computation."""

import cv2
import numpy as np
import pytest

from src.vision.depth import (
    compute_disparity,
    create_sgbm,
    depth_at_bbox,
    disparity_to_depth,
)


class TestCreateSGBM:
    def test_default_creation(self):
        sgbm = create_sgbm()
        assert sgbm is not None
        assert isinstance(sgbm, cv2.StereoSGBM)

    def test_custom_params(self):
        sgbm = create_sgbm(num_disparities=64, block_size=7)
        assert sgbm is not None


class TestComputeDisparity:
    def _make_shifted_pair(
        self, shift_px: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a synthetic stereo pair with known horizontal shift.

        A textured pattern shifted by shift_px pixels simulates a scene
        at a known disparity.
        """
        rng = np.random.RandomState(123)
        # Generate a textured image (random blocks for SGBM to match)
        base = np.zeros((480, 640 + shift_px), dtype=np.uint8)
        for _ in range(500):
            x = rng.randint(0, base.shape[1] - 30)
            y = rng.randint(0, base.shape[0] - 30)
            w = rng.randint(5, 30)
            h = rng.randint(5, 30)
            color = rng.randint(50, 250)
            base[y : y + h, x : x + w] = color

        left = base[:, shift_px : shift_px + 640]
        right = base[:, :640]
        return left, right

    def test_disparity_shape(self):
        left, right = self._make_shifted_pair(20)
        disp = compute_disparity(left, right)
        assert disp.shape == (480, 640)
        assert disp.dtype == np.float32

    def test_disparity_has_valid_pixels(self):
        left, right = self._make_shifted_pair(20)
        disp = compute_disparity(left, right, num_disparities=64)
        valid = disp[disp > 0]
        # Should find at least some valid disparity values
        assert len(valid) > 0

    def test_known_shift_disparity(self):
        """Disparity in textured region should be close to known shift."""
        shift = 30
        left, right = self._make_shifted_pair(shift)
        disp = compute_disparity(left, right, num_disparities=64, block_size=5)

        # Check central region (edges are unreliable in SGBM)
        roi = disp[100:380, 200:500]
        valid = roi[roi > 0]

        if len(valid) > 100:
            median_disp = np.median(valid)
            # Median disparity should be in the ballpark of our shift
            assert abs(median_disp - shift) < 15, (
                f"Expected disparity ~{shift}, got median {median_disp:.1f}"
            )

    def test_accepts_bgr(self):
        left, right = self._make_shifted_pair(20)
        left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        disp = compute_disparity(left_bgr, right_bgr)
        assert disp.shape == (480, 640)

    def test_with_prebuilt_sgbm(self):
        left, right = self._make_shifted_pair(20)
        sgbm = create_sgbm(num_disparities=64)
        disp = compute_disparity(left, right, sgbm=sgbm)
        assert disp.shape == (480, 640)

    def test_invalid_pixels_are_negative_one(self):
        left, right = self._make_shifted_pair(20)
        disp = compute_disparity(left, right)
        # Invalid pixels should be exactly -1.0
        invalid = disp[disp < 0]
        if len(invalid) > 0:
            np.testing.assert_array_equal(invalid, -1.0)


class TestDisparityToDepth:
    def test_basic_conversion(self):
        # f=500px, B=140mm, d=20px → Z = 500*140/20 = 3500mm
        disp = np.full((100, 100), 20.0, dtype=np.float32)
        depth = disparity_to_depth(disp, focal_length_px=500.0, baseline_mm=140.0)
        np.testing.assert_allclose(depth, 3500.0, atol=0.1)

    def test_invalid_disparity_gives_zero_depth(self):
        disp = np.full((100, 100), -1.0, dtype=np.float32)
        depth = disparity_to_depth(disp, 500.0, 140.0)
        np.testing.assert_array_equal(depth, 0.0)

    def test_zero_disparity_gives_zero_depth(self):
        disp = np.zeros((100, 100), dtype=np.float32)
        depth = disparity_to_depth(disp, 500.0, 140.0)
        np.testing.assert_array_equal(depth, 0.0)

    def test_min_max_clipping(self):
        # Very high disparity → very close → below min_depth → clipped
        disp = np.full((100, 100), 200.0, dtype=np.float32)
        depth = disparity_to_depth(
            disp, 500.0, 140.0, min_depth_mm=500.0, max_depth_mm=5000.0
        )
        # Z = 500*140/200 = 350mm < 500 min → should be 0
        np.testing.assert_array_equal(depth, 0.0)

    def test_far_distance_clipped(self):
        # Very low disparity → very far → above max_depth → clipped
        disp = np.full((100, 100), 1.0, dtype=np.float32)
        depth = disparity_to_depth(
            disp, 500.0, 140.0, min_depth_mm=500.0, max_depth_mm=5000.0
        )
        # Z = 500*140/1 = 70000mm > 5000 max → should be 0
        np.testing.assert_array_equal(depth, 0.0)

    def test_mixed_valid_invalid(self):
        disp = np.array([[20.0, -1.0], [0.0, 10.0]], dtype=np.float32)
        depth = disparity_to_depth(disp, 500.0, 140.0)
        assert depth[0, 0] == pytest.approx(3500.0, abs=0.1)
        assert depth[0, 1] == 0.0  # invalid
        assert depth[1, 0] == 0.0  # zero disparity
        assert depth[1, 1] == pytest.approx(7000.0, abs=0.1)


class TestDepthAtBbox:
    def test_returns_median_of_valid(self):
        depth = np.full((100, 100), 3000.0, dtype=np.float32)
        d = depth_at_bbox(depth, (10, 10, 90, 90))
        assert d == pytest.approx(3000.0, abs=1.0)

    def test_returns_zero_for_all_invalid(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        d = depth_at_bbox(depth, (10, 10, 90, 90))
        assert d == 0.0

    def test_custom_percentile(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[40:60, 40:60] = 2500.0
        depth[45:55, 45:55] = 3000.0
        d = depth_at_bbox(depth, (20, 20, 80, 80), percentile=75)
        assert d > 0

    def test_bbox_at_edge(self):
        """Should handle bboxes near image boundaries."""
        depth = np.full((100, 100), 2000.0, dtype=np.float32)
        d = depth_at_bbox(depth, (0, 0, 20, 20))
        assert d > 0

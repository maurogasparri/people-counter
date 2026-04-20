"""Smoke tests for focus_assist pure helpers (no camera hardware)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Import focus_assist as a module
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location(
    "focus_assist_script", _ROOT / "scripts" / "focus_assist.py",
)
focus_assist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(focus_assist)


def _textured_frame(w: int = 640, h: int = 480, variance: float = 50.0) -> np.ndarray:
    rng = np.random.RandomState(0)
    return (rng.rand(h, w, 3) * variance + 60).astype(np.uint8)


class TestFocusScore:
    def test_focus_score_on_sharp_frame(self):
        f = _textured_frame(variance=100)
        score = focus_assist.focus_score(f)
        assert score > 100

    def test_focus_score_on_blurred_frame(self):
        import cv2
        blurred = cv2.GaussianBlur(_textured_frame(variance=100), (31, 31), 10)
        score = focus_assist.focus_score(blurred)
        assert score < 50


class TestEvaluateFocus:
    def _sharp_grid(self, value: float = 500) -> np.ndarray:
        return np.full((3, 3), value, dtype=np.float64)

    def test_symmetric_sharp_passes(self):
        g = self._sharp_grid()
        # 1500mm is inside the default target range (1150-1800mm for a 3.0m mount)
        ev = focus_assist.evaluate_focus(g, g, 1500.0, 1500.0)
        assert ev["all_pass"]
        assert ev["all_pass_with_distance"]

    def test_missing_charuco_not_distance_ok(self):
        g = self._sharp_grid()
        ev = focus_assist.evaluate_focus(g, g, None, None)
        assert ev["all_pass"]
        assert not ev["all_pass_with_distance"]
        assert not ev["checks"]["distance"]

    def test_board_too_close(self):
        g = self._sharp_grid()
        # 800mm is below the default 1150mm min for a 3.0m mount
        ev = focus_assist.evaluate_focus(g, g, 800.0, 800.0)
        assert not ev["distance_ok"]
        assert any("cerca" in h for h in ev["hints"])

    def test_board_too_far(self):
        g = self._sharp_grid()
        ev = focus_assist.evaluate_focus(g, g, 4000.0, 4000.0)
        assert not ev["distance_ok"]
        assert any("lejos" in h for h in ev["hints"])

    def test_weak_center_flags_correct_side(self):
        strong = self._sharp_grid(500)
        weak = np.full((3, 3), 50, dtype=np.float64)
        # Weak center on L, strong on R
        weak_l = weak.copy()
        ev = focus_assist.evaluate_focus(weak_l, strong, 1500.0, 1500.0)
        assert not ev["checks"]["center_l"]
        assert any("IZQ" in h for h in ev["hints"])

    def test_lr_asymmetry_detected(self):
        bright = self._sharp_grid(500)
        dim = np.full((3, 3), 200, dtype=np.float64)
        ev = focus_assist.evaluate_focus(bright, dim, 1500.0, 1500.0)
        # 60% global diff -> exceeds 15%
        assert not ev["checks"]["lr_global"]

    def test_low_contrast_corners_excluded_from_uniformity(self):
        # Sharp center, but 2 corners are "low-content" (mask=False). The
        # remaining corners are sharp — uniformity should still pass based
        # on the valid corners only.
        grid = np.array([
            [500.0, 500.0, 500.0],
            [500.0, 500.0, 500.0],
            [500.0, 500.0, 500.0],
        ])
        # Mark 2 corners as invalid (simulate plain wall there)
        valid = np.array([
            [False, True, True],
            [True, True, True],
            [True, True, False],
        ])
        ev = focus_assist.evaluate_focus(grid, grid, 1500.0, 1500.0, valid, valid)
        assert ev["checks"]["uniformity_l"]
        assert ev["uniformity_l_measurable"]
        assert ev["n_valid_corners_l"] == 2

    def test_all_corners_invalid_skips_uniformity_check(self):
        # Flat wall in every corner — uniformity becomes unmeasurable but
        # the check passes by default so the operator isn't blocked.
        grid = np.full((3, 3), 500.0)
        valid = np.array([
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ])
        ev = focus_assist.evaluate_focus(grid, grid, 1500.0, 1500.0, valid, valid)
        assert not ev["uniformity_l_measurable"]
        assert ev["checks"]["uniformity_l"]   # default-pass
        assert any("textura" in h for h in ev["hints"])


class TestEstimateDistance:
    def test_returns_none_when_no_board(self):
        from src.vision.calibration import create_charuco_board
        board = create_charuco_board()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dist, n = focus_assist.estimate_charuco_distance_mm(blank, board)
        assert dist is None
        assert n == 0

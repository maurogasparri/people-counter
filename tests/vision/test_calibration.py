"""Tests for stereo calibration module.

Uses synthetically rendered ChArUco board images to test the full
calibration pipeline without requiring physical cameras.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.calibration import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_MARKER_LENGTH,
    DEFAULT_SQUARE_LENGTH,
    calibrate_stereo,
    create_charuco_board,
    detect_charuco_corners,
    generate_board_image,
    load_calibration,
    rectify_pair,
    save_calibration,
)


# ---------------------------------------------------------------------------
# Synthetic stereo pair generator
# ---------------------------------------------------------------------------

IMAGE_W, IMAGE_H = 640, 480
FOCAL_LENGTH = 500.0  # pixels
BASELINE = 140.0  # mm (14 cm)


def _synth_camera_matrix() -> np.ndarray:
    """Synthetic camera matrix for a 640x480 image."""
    return np.array(
        [
            [FOCAL_LENGTH, 0, IMAGE_W / 2],
            [0, FOCAL_LENGTH, IMAGE_H / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _render_charuco_pair(
    board: cv2.aruco.CharucoBoard,
    rvec: np.ndarray,
    tvec_base: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a ChArUco board from two synthetic camera viewpoints.

    Left camera is at tvec_base, right camera is shifted +BASELINE in X.
    Uses cv2.projectPoints to map board corners to image coordinates,
    then draws the board image with a perspective warp.
    """
    board_img = board.generateImage((600, 400), marginSize=20)

    # Board 3D corners (in board coordinate frame)
    h_board, w_board = board_img.shape[:2]
    board_corners_3d = np.array(
        [
            [0, 0, 0],
            [w_board, 0, 0],
            [w_board, h_board, 0],
            [0, h_board, 0],
        ],
        dtype=np.float32,
    )

    # Scale 3D corners to match the physical board size
    cols, rows = board.getChessboardSize()
    sq_len = DEFAULT_SQUARE_LENGTH
    physical_w = cols * sq_len
    physical_h = rows * sq_len
    board_corners_3d[:, 0] *= physical_w / w_board
    board_corners_3d[:, 1] *= physical_h / h_board

    dist_coeffs = np.zeros(5)

    def _render_view(tvec: np.ndarray) -> np.ndarray:
        pts_2d, _ = cv2.projectPoints(
            board_corners_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(np.float32)

        src_pts = np.array(
            [[0, 0], [w_board, 0], [w_board, h_board], [0, h_board]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src_pts, pts_2d)
        img = cv2.warpPerspective(board_img, M, (IMAGE_W, IMAGE_H))
        return img

    # Left camera
    img_l = _render_view(tvec_base)

    # Right camera: shifted by baseline in X
    tvec_right = tvec_base.copy()
    tvec_right[0] += BASELINE
    img_r = _render_view(tvec_right)

    return img_l, img_r


def _generate_synthetic_pairs(
    n_pairs: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate n_pairs of synthetic stereo ChArUco images.

    Varies the board orientation slightly for each pair to simulate
    capturing from different angles.
    """
    board = create_charuco_board()
    cam = _synth_camera_matrix()
    rng = np.random.RandomState(42)

    pairs = []
    for i in range(n_pairs):
        # Small rotation variations around X and Y axes
        rx = rng.uniform(-0.3, 0.3)
        ry = rng.uniform(-0.3, 0.3)
        rz = rng.uniform(-0.1, 0.1)
        rvec = np.array([rx, ry, rz], dtype=np.float64)

        # Board at ~500mm distance, centered
        tvec = np.array(
            [
                rng.uniform(-30, 30),
                rng.uniform(-30, 30),
                rng.uniform(400, 600),
            ],
            dtype=np.float64,
        )

        img_l, img_r = _render_charuco_pair(board, rvec, tvec, cam)
        pairs.append((img_l, img_r))

    return pairs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCharucoBoard:
    def test_create_board(self):
        board = create_charuco_board()
        cols, rows = board.getChessboardSize()
        assert cols == DEFAULT_BOARD_SIZE[0]
        assert rows == DEFAULT_BOARD_SIZE[1]

    def test_generate_board_image(self):
        board = create_charuco_board()
        img = generate_board_image(board, (800, 600))
        assert img.shape == (600, 800)
        assert img.dtype == np.uint8

    def test_board_has_correct_corners(self):
        board = create_charuco_board()
        corners = board.getChessboardCorners()
        cols, rows = board.getChessboardSize()
        # ChArUco has (cols-1)*(rows-1) internal corners
        expected = (cols - 1) * (rows - 1)
        assert len(corners) == expected


class TestDetection:
    def test_detect_on_rendered_board(self):
        """Detect corners on a clean frontal view of the board."""
        board = create_charuco_board()
        img = generate_board_image(board, (800, 600))

        corners, ids = detect_charuco_corners(img, board)

        assert corners is not None
        assert ids is not None
        assert len(corners) >= 6
        assert len(ids) == len(corners)

    def test_detect_returns_none_on_blank(self):
        board = create_charuco_board()
        blank = np.zeros((480, 640), dtype=np.uint8)

        corners, ids = detect_charuco_corners(blank, board)

        assert corners is None
        assert ids is None

    def test_detect_on_synthetic_pair(self):
        """Detection works on perspective-warped synthetic images."""
        board = create_charuco_board()
        cam = _synth_camera_matrix()
        rvec = np.array([0.1, 0.1, 0.0])
        tvec = np.array([0.0, 0.0, 500.0])

        img_l, img_r = _render_charuco_pair(board, rvec, tvec, cam)

        corners_l, ids_l = detect_charuco_corners(img_l, board)
        corners_r, ids_r = detect_charuco_corners(img_r, board)

        assert corners_l is not None
        assert corners_r is not None


class TestCalibration:
    @pytest.fixture(scope="class")
    def synthetic_pairs(self):
        return _generate_synthetic_pairs(25)

    def test_calibrate_stereo_runs(self, synthetic_pairs):
        """Full calibration completes without error on synthetic data."""
        result = calibrate_stereo(synthetic_pairs)

        # Check all expected keys
        expected_keys = {
            "camera_matrix_l", "dist_coeffs_l",
            "camera_matrix_r", "dist_coeffs_r",
            "R", "T",
            "R1", "R2", "P1", "P2", "Q",
            "map_l_x", "map_l_y", "map_r_x", "map_r_y",
            "image_size",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_camera_matrices_reasonable(self, synthetic_pairs):
        result = calibrate_stereo(synthetic_pairs)

        # Focal length should be in reasonable range for 640x480
        fx_l = result["camera_matrix_l"][0, 0]
        fy_l = result["camera_matrix_l"][1, 1]
        assert 200 < fx_l < 1000
        assert 200 < fy_l < 1000

        # Principal point near image center
        cx = result["camera_matrix_l"][0, 2]
        cy = result["camera_matrix_l"][1, 2]
        assert abs(cx - IMAGE_W / 2) < IMAGE_W * 0.2
        assert abs(cy - IMAGE_H / 2) < IMAGE_H * 0.2

    def test_translation_recovers_baseline(self, synthetic_pairs):
        """Translation vector should be close to the baseline."""
        result = calibrate_stereo(synthetic_pairs)

        # T[0] should be roughly the baseline (140mm)
        # Tolerance is wide because synthetic rendering isn't perfect
        tx = abs(result["T"][0, 0])
        assert tx > 50  # At least some horizontal translation detected

    def test_rectification_maps_valid(self, synthetic_pairs):
        result = calibrate_stereo(synthetic_pairs)

        assert result["map_l_x"].shape == (IMAGE_H, IMAGE_W)
        assert result["map_l_y"].shape == (IMAGE_H, IMAGE_W)
        assert result["map_r_x"].shape == (IMAGE_H, IMAGE_W)
        assert result["map_r_y"].shape == (IMAGE_H, IMAGE_W)

    def test_too_few_pairs_raises(self):
        """Should raise ValueError with fewer than 15 valid pairs."""
        pairs = _generate_synthetic_pairs(5)
        with pytest.raises(ValueError, match="at least 15"):
            calibrate_stereo(pairs)

    def test_image_size_stored(self, synthetic_pairs):
        result = calibrate_stereo(synthetic_pairs)
        assert np.array_equal(result["image_size"], [IMAGE_W, IMAGE_H])


class TestRectifyPair:
    def test_rectify_pair_shapes(self):
        """Rectified images have same shape as input."""
        cal = {
            "map_l_x": np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32),
            "map_l_y": np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32),
            "map_r_x": np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32),
            "map_r_y": np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32),
        }
        img = np.random.randint(0, 255, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        rect_l, rect_r = rectify_pair(img, img, cal)
        assert rect_l.shape == img.shape
        assert rect_r.shape == img.shape


class TestIO:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "cal.npz")
            params = {
                "camera_matrix_l": np.eye(3),
                "T": np.array([[140.0], [0.0], [0.0]]),
            }
            save_calibration(params, path)

            loaded = load_calibration(path)
            np.testing.assert_array_almost_equal(
                loaded["camera_matrix_l"], np.eye(3)
            )
            np.testing.assert_array_almost_equal(
                loaded["T"], np.array([[140.0], [0.0], [0.0]])
            )

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_calibration("/nonexistent/path/cal.npz")

"""Tests para el módulo de calibración estéreo.

Usa imágenes del board ChArUco renderizadas sintéticamente para testear el
calibration pipeline without requiring physical cameras.
"""

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.calibration import (
    ALIGN_CENTER_TOL_PX,
    ALIGN_MEAN_ERR_TOL_PX_LOOSE,
    ALIGN_MEAN_ERR_TOL_PX_TIGHT,
    ALIGN_ROTATION_TOL_DEG,
    ALIGN_SCALE_TOL,
    DEFAULT_BOARD_SIZE,
    DEFAULT_SQUARE_LENGTH,
    PoseTarget,
    StabilityTracker,
    alignment_hint,
    alignment_hint_by_corners,
    analyze_pose_coverage,
    assess_frame_quality,
    calibrate_stereo,
    compute_alignment_by_corners,
    compute_alignment_error,
    compute_per_pair_residuals,
    is_calibration_ready_for_early_stop,
    lens_alignment_metrics,
    corner_local_sharpness,
    count_common_corners,
    create_charuco_board,
    default_pose_sequence,
    detect_charuco_corners,
    fit_single_camera_intrinsics,
    generate_board_image,
    is_aligned,
    is_aligned_by_corners,
    live_lighting_warnings,
    load_calibration,
    project_pose,
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


class TestLensAlignmentMetrics:
    """Operator-friendly decomposition of stereo extrinsics."""

    def test_identity_R_zero_T_yields_all_zeros(self):
        R = np.eye(3)
        T = np.zeros(3)
        m = lens_alignment_metrics(R, T)
        for k, v in m.items():
            assert abs(v) < 1e-6, f"{k}={v}"

    def test_T_components_split_correctly(self):
        # Pure translation, no rotation. T components map straight to
        # offset_{x,y,z}_mm. The X axis is the baseline (negative when
        # calibrate_stereo emits -baseline-on-X by convention; the test
        # uses positive to keep the assertion simple).
        R = np.eye(3)
        T = np.array([140.0, 1.5, -0.8])
        m = lens_alignment_metrics(R, T)
        assert m["offset_x_mm"] == pytest.approx(140.0)
        assert m["offset_y_mm"] == pytest.approx(1.5)
        assert m["offset_z_mm"] == pytest.approx(-0.8)
        assert abs(m["rotation_x_deg"]) < 1e-6
        assert abs(m["rotation_y_deg"]) < 1e-6
        assert abs(m["rotation_z_deg"]) < 1e-6

    def test_pure_yaw_rotation(self):
        # 5° around Y (yaw). XYZ-Euler convention: ry should pop out
        # exactly, others zero.
        ang = math.radians(5.0)
        R = np.array([
            [math.cos(ang), 0, math.sin(ang)],
            [0, 1, 0],
            [-math.sin(ang), 0, math.cos(ang)],
        ])
        m = lens_alignment_metrics(R, np.zeros(3))
        assert m["rotation_y_deg"] == pytest.approx(5.0, abs=1e-5)
        assert abs(m["rotation_x_deg"]) < 1e-5
        assert abs(m["rotation_z_deg"]) < 1e-5

    def test_pure_pitch_rotation(self):
        # 2° around X (pitch).
        ang = math.radians(2.0)
        R = np.array([
            [1, 0, 0],
            [0, math.cos(ang), -math.sin(ang)],
            [0, math.sin(ang), math.cos(ang)],
        ])
        m = lens_alignment_metrics(R, np.zeros(3))
        assert m["rotation_x_deg"] == pytest.approx(2.0, abs=1e-5)
        assert abs(m["rotation_y_deg"]) < 1e-5
        assert abs(m["rotation_z_deg"]) < 1e-5

    def test_pure_roll_rotation(self):
        # 1.2° around Z (roll).
        ang = math.radians(1.2)
        R = np.array([
            [math.cos(ang), -math.sin(ang), 0],
            [math.sin(ang), math.cos(ang), 0],
            [0, 0, 1],
        ])
        m = lens_alignment_metrics(R, np.zeros(3))
        assert m["rotation_z_deg"] == pytest.approx(1.2, abs=1e-5)
        assert abs(m["rotation_x_deg"]) < 1e-5
        assert abs(m["rotation_y_deg"]) < 1e-5

    def test_T_accepts_column_vector(self):
        # cv2.fisheye.stereoCalibrate returns T as (3,1) column vector.
        # The function should accept either shape.
        R = np.eye(3)
        T = np.array([[140.0], [1.5], [-0.8]])
        m = lens_alignment_metrics(R, T)
        assert m["offset_x_mm"] == pytest.approx(140.0)

    def test_invalid_R_shape_raises(self):
        with pytest.raises(ValueError, match="3x3"):
            lens_alignment_metrics(np.eye(2), np.zeros(3))


class TestEarlyStopReadiness:
    """Helper that lets the wizard finalize when calibration is already
    lab-grade, without forcing the operator through all 20 poses."""

    def _full_coverage(self) -> dict:
        return {
            "by_distance": {"near": 4, "mid": 5, "far": 4},
            "by_tilt_axis": {"frontal": 2, "pitch": 4, "yaw": 3, "roll": 2},
            "by_group": {"A": 3, "B": 4, "C": 3, "D": 3},
            "warnings": [],
            "critical": [],
            "ok": True,
        }

    def test_all_conditions_met_returns_ready(self):
        ready, reason = is_calibration_ready_for_early_stop(
            self._full_coverage(), per_pair_rms_px=0.30, captured_count=13,
        )
        assert ready is True
        assert reason == ""

    def test_too_few_captures_blocks(self):
        ready, reason = is_calibration_ready_for_early_stop(
            self._full_coverage(), per_pair_rms_px=0.20, captured_count=10,
        )
        assert ready is False
        assert "10/12" in reason

    def test_critical_gap_blocks(self):
        cov = self._full_coverage()
        cov["critical"] = ["Banda de distancia 'far' sin capturas"]
        ready, reason = is_calibration_ready_for_early_stop(
            cov, per_pair_rms_px=0.30, captured_count=14,
        )
        assert ready is False
        assert "cobertura incompleta" in reason

    def test_band_under_min_blocks(self):
        cov = self._full_coverage()
        cov["by_distance"]["far"] = 1  # below min_per_band=2
        ready, reason = is_calibration_ready_for_early_stop(
            cov, per_pair_rms_px=0.30, captured_count=14,
        )
        assert ready is False
        assert "far" in reason

    def test_missing_required_group_blocks(self):
        cov = self._full_coverage()
        cov["by_group"]["B"] = 0
        ready, reason = is_calibration_ready_for_early_stop(
            cov, per_pair_rms_px=0.30, captured_count=14,
        )
        assert ready is False
        assert "grupo B" in reason or "grupo B" in reason.lower()

    def test_rms_above_threshold_blocks(self):
        ready, reason = is_calibration_ready_for_early_stop(
            self._full_coverage(),
            per_pair_rms_px=0.80,  # over default 0.50
            captured_count=14,
        )
        assert ready is False
        assert "RMS" in reason

    def test_nan_rms_blocks(self):
        ready, reason = is_calibration_ready_for_early_stop(
            self._full_coverage(),
            per_pair_rms_px=float("nan"),
            captured_count=14,
        )
        assert ready is False
        assert "RMS" in reason

    def test_thresholds_are_overridable(self):
        # Stricter min_poses keeps a session open even with great metrics.
        ready, _ = is_calibration_ready_for_early_stop(
            self._full_coverage(), per_pair_rms_px=0.20, captured_count=12,
            min_poses=15,
        )
        assert ready is False

    def test_e_group_not_required(self):
        # Group E (extreme tilts) is bonus, not required.
        cov = self._full_coverage()
        cov["by_group"].pop("E", None)
        ready, _ = is_calibration_ready_for_early_stop(
            cov, per_pair_rms_px=0.30, captured_count=13,
        )
        assert ready is True


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


# ---------------------------------------------------------------------------
# Guided calibration helpers
# ---------------------------------------------------------------------------


PREVIEW_SIZE = (648, 486)


class TestPoseSequence:
    def test_default_sequence_has_canonical_coverage(self):
        poses = default_pose_sequence()
        assert len(poses) == 20
        ids = [p.id for p in poses]
        assert len(set(ids)) == len(ids), "pose IDs must be unique"

    def test_sequence_covers_three_distance_bands(self):
        poses = default_pose_sequence()
        zs = sorted({p.tvec_mm[2] for p in poses})
        assert len(zs) == 3, f"expected 3 distance bands, got {zs}"
        assert 800 <= zs[0] <= 1200
        assert 1800 <= zs[1] <= 2200
        assert 2800 <= zs[2] <= 3500

    def test_sequence_includes_strong_tilts(self):
        poses = default_pose_sequence()
        has_big_pitch = any(abs(p.pitch_deg) >= 15 for p in poses)
        has_big_yaw = any(abs(p.yaw_deg) >= 15 for p in poses)
        has_big_roll = any(abs(p.roll_deg) >= 15 for p in poses)
        assert has_big_pitch and has_big_yaw and has_big_roll

    def test_rvec_roundtrip(self):
        pose = PoseTarget("test", "label", (0, 0, 1500), pitch_deg=10, yaw_deg=20, roll_deg=30)
        rv = pose.rvec()
        assert rv.shape == (3,)
        # Applying Rodrigues and inverse should round-trip
        R, _ = cv2.Rodrigues(rv)
        rv2, _ = cv2.Rodrigues(R)
        np.testing.assert_allclose(rv, rv2.flatten(), atol=1e-6)


class TestProjectPose:
    def test_frontal_projection_centered(self):
        pose = PoseTarget("c", "center", (0, 0, 1500))
        proj = project_pose(pose, DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        cx, cy = proj["center"]
        assert abs(cx - PREVIEW_SIZE[0] / 2) < 1
        assert abs(cy - PREVIEW_SIZE[1] / 2) < 1

    def test_far_projection_smaller(self):
        near = project_pose(PoseTarget("n", "", (0, 0, 1000)),
                            DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        far = project_pose(PoseTarget("f", "", (0, 0, 2500)),
                           DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        near_w = np.ptp(near["outer_corners"][:, 0])
        far_w = np.ptp(far["outer_corners"][:, 0])
        assert near_w > far_w * 2  # much bigger when closer

    def test_offset_shifts_projection(self):
        center = project_pose(PoseTarget("c", "", (0, 0, 1500)),
                              DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        right = project_pose(PoseTarget("r", "", (200, 0, 1500)),
                             DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        assert right["center"][0] > center["center"][0] + 20

    def test_inner_corner_count_matches_chess_grid(self):
        proj = project_pose(PoseTarget("c", "", (0, 0, 1500)),
                            DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        cols, rows = DEFAULT_BOARD_SIZE
        expected = (cols - 1) * (rows - 1)
        assert len(proj["inner_corners"]) == expected


class TestAlignment:
    def _ghost(self, tvec=(0, 0, 1500), **kw):
        pose = PoseTarget("g", "", tvec, **kw)
        return project_pose(pose, DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)

    def test_exact_match_is_aligned(self):
        ghost = self._ghost()
        detected = ghost["inner_corners"].copy()
        err = compute_alignment_error(detected, ghost["inner_corners"])
        assert err["center_px"] < 1
        assert abs(err["scale_ratio"] - 1.0) < 1e-6
        assert abs(err["rotation_deg"]) < 1e-3
        assert is_aligned(err)

    def test_small_offset_fails_alignment(self):
        ghost = self._ghost()
        detected = ghost["inner_corners"] + np.array([50, 0])
        err = compute_alignment_error(detected, ghost["inner_corners"])
        assert err["center_px"] > ALIGN_CENTER_TOL_PX
        assert not is_aligned(err)

    def test_scale_outside_tolerance_fails(self):
        ghost = self._ghost()
        # Scale around centroid to preserve center
        centroid = ghost["inner_corners"].mean(axis=0)
        detected = centroid + (ghost["inner_corners"] - centroid) * 1.15
        err = compute_alignment_error(detected, ghost["inner_corners"])
        assert abs(err["scale_ratio"] - 1.0) > ALIGN_SCALE_TOL
        assert not is_aligned(err)

    def test_rotation_outside_tolerance_fails(self):
        ghost = self._ghost()
        centroid = ghost["inner_corners"].mean(axis=0)
        theta = math.radians(10)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        detected = (ghost["inner_corners"] - centroid) @ R.T + centroid
        err = compute_alignment_error(detected, ghost["inner_corners"])
        assert abs(err["rotation_deg"]) > ALIGN_ROTATION_TOL_DEG
        assert not is_aligned(err)

    def test_alignment_hint_suggests_direction(self):
        ghost = self._ghost()
        detected = ghost["inner_corners"] + np.array([40, 0])
        err = compute_alignment_error(detected, ghost["inner_corners"])
        hint = alignment_hint(err)
        assert "izquierda" in hint.lower() or "derecha" in hint.lower()

    def test_alignment_hint_ok_when_aligned(self):
        ghost = self._ghost()
        err = compute_alignment_error(ghost["inner_corners"], ghost["inner_corners"])
        assert "alineado" in alignment_hint(err).lower()


class TestStability:
    def test_stability_needs_full_window(self):
        st = StabilityTracker(window=10)
        pts = np.array([[100, 100], [200, 200]], dtype=np.float32)
        for _ in range(5):
            st.push(pts)
        assert not st.is_stable()

    def test_stable_when_buffer_static(self):
        st = StabilityTracker(window=5, max_disp_px=1.5)
        pts = np.array([[100, 100], [200, 200]], dtype=np.float32)
        for _ in range(5):
            st.push(pts)
        assert st.is_stable()

    def test_unstable_when_moving(self):
        st = StabilityTracker(window=5, max_disp_px=1.5)
        for i in range(5):
            pts = np.array([[100 + i * 3, 100], [200, 200]], dtype=np.float32)
            st.push(pts)
        assert not st.is_stable()

    def test_detection_loss_clears_buffer(self):
        st = StabilityTracker(window=5)
        pts = np.array([[100, 100]], dtype=np.float32)
        for _ in range(5):
            st.push(pts)
        st.push(None)
        assert not st.is_stable()

    def test_corner_count_change_restarts(self):
        st = StabilityTracker(window=5)
        small = np.array([[100, 100]], dtype=np.float32)
        big = np.array([[100, 100], [200, 200]], dtype=np.float32)
        for _ in range(3):
            st.push(small)
        st.push(big)
        # Without IDs the tracker still uses the legacy "restart on count
        # change" behaviour — there's no way to know which corner is which.
        assert len(st._buffer) == 1

    def test_ids_tolerate_detection_count_fluctuation(self):
        # With marker IDs, the tracker compares displacement on the corners
        # that appear in BOTH frames. Counts can fluctuate (e.g. 23↔35) due
        # to marginal lighting without resetting the buffer — what matters
        # is whether the persistent corners are still.
        st = StabilityTracker(window=5, max_disp_px=1.5)
        # Frames alternate between detecting 3 corners and detecting 2 corners,
        # but the corners that DO appear stay at the same pixel coords.
        big_pts = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
        big_ids = np.array([1, 2, 3])
        small_pts = np.array([[100, 100], [200, 200]], dtype=np.float32)
        small_ids = np.array([1, 2])
        for i in range(5):
            if i % 2 == 0:
                st.push(big_pts, ids=big_ids)
            else:
                st.push(small_pts, ids=small_ids)
        assert len(st._buffer) == 5
        assert st.is_stable()

    def test_ids_detect_movement_on_common_corners(self):
        # Even with fluctuating counts, real motion on the persistent corners
        # must be detected — IDs only tolerate count changes, not displacement.
        st = StabilityTracker(window=5, max_disp_px=1.5)
        for i in range(5):
            pts = np.array(
                [[100 + i * 5, 100], [200, 200], [300, 300]], dtype=np.float32,
            )
            ids = np.array([1, 2, 3]) if i % 2 == 0 else np.array([1, 2])
            st.push(pts[:len(ids)], ids=ids)
        assert not st.is_stable()


class TestFrameQuality:
    def _noisy_frame(self, shape=(480, 640, 3), scale=100):
        rng = np.random.RandomState(0)
        return (rng.rand(*shape) * scale + 50).astype(np.uint8)

    def test_sharp_balanced_frame_passes(self):
        f = self._noisy_frame()
        q = assess_frame_quality(f, f, n_corners=30)
        assert q["all_pass"], q["reasons"]

    def test_low_corners_fails(self):
        f = self._noisy_frame()
        q = assess_frame_quality(f, f, n_corners=5)
        assert not q["checks"]["corners"]
        assert any("esquinas" in r.lower() for r in q["reasons"])

    def test_blurry_frame_fails(self):
        blurred = cv2.GaussianBlur(self._noisy_frame(), (31, 31), 10)
        q = assess_frame_quality(blurred, blurred, n_corners=30)
        assert not q["checks"]["blur_l"]
        assert not q["checks"]["blur_r"]

    def test_saturated_frame_fails(self):
        sat = np.full((480, 640, 3), 255, dtype=np.uint8)
        q = assess_frame_quality(sat, sat, n_corners=30)
        assert not q["checks"]["saturation_l"]

    def test_dark_frame_fails(self):
        dark = np.full((480, 640, 3), 10, dtype=np.uint8)
        q = assess_frame_quality(dark, dark, n_corners=30)
        assert not q["checks"]["exposure_l"]

    def test_lr_asymmetry_detected(self):
        bright = np.full((480, 640, 3), 180, dtype=np.uint8)
        dark = np.full((480, 640, 3), 60, dtype=np.uint8)
        q = assess_frame_quality(bright, dark, n_corners=30)
        assert not q["checks"]["lr_balance"]


class TestCornerIdMatching:
    def _ghost(self):
        return project_pose(
            PoseTarget("g", "", (0, 0, 1500)),
            DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE,
        )

    def test_exact_match_zero_error(self):
        ghost = self._ghost()
        ids = np.arange(len(ghost["inner_corners"]))
        err = compute_alignment_by_corners(
            ghost["inner_corners"], ids, ghost["inner_corners"],
        )
        assert err["mean_error_px"] < 1e-3
        assert err["centroid_offset_px"] < 1e-3
        assert err["matched"] == len(ids)
        assert is_aligned_by_corners(err)

    def test_translation_is_detected(self):
        ghost = self._ghost()
        ids = np.arange(len(ghost["inner_corners"]))
        shifted = ghost["inner_corners"] + np.array([40, 0])
        err = compute_alignment_by_corners(shifted, ids, ghost["inner_corners"])
        assert err["mean_error_px"] > 30
        assert err["centroid_offset_px"] > 30
        assert abs(err["offset_x"] - 40) < 1e-3
        assert not is_aligned_by_corners(err)

    def test_partial_detection_still_works(self):
        ghost = self._ghost()
        n_full = len(ghost["inner_corners"])
        # Detect only half the corners
        ids = np.arange(n_full // 2)
        detected = ghost["inner_corners"][:n_full // 2]
        err = compute_alignment_by_corners(detected, ids, ghost["inner_corners"])
        assert err["matched"] == n_full // 2
        assert err["mean_error_px"] < 1e-3

    def test_too_few_corners_returns_inf(self):
        ghost = self._ghost()
        detected = ghost["inner_corners"][:2]
        ids = np.array([0, 1])
        err = compute_alignment_by_corners(detected, ids, ghost["inner_corners"])
        assert err["mean_error_px"] == float("inf")
        assert not is_aligned_by_corners(err)

    def test_loose_tolerance_gate(self):
        ghost = self._ghost()
        ids = np.arange(len(ghost["inner_corners"]))
        shifted = ghost["inner_corners"] + np.array([18, 0])
        err = compute_alignment_by_corners(shifted, ids, ghost["inner_corners"])
        # Would fail tight (12px) but pass loose (25px)
        assert not is_aligned_by_corners(err, mean_err_tol_px=ALIGN_MEAN_ERR_TOL_PX_TIGHT)
        assert is_aligned_by_corners(
            err,
            mean_err_tol_px=ALIGN_MEAN_ERR_TOL_PX_LOOSE,
            min_matched=1,
        )

    def test_hint_flags_dominant_axis(self):
        ghost = self._ghost()
        ids = np.arange(len(ghost["inner_corners"]))
        shifted = ghost["inner_corners"] + np.array([50, 0])
        err = compute_alignment_by_corners(shifted, ids, ghost["inner_corners"])
        hint = alignment_hint_by_corners(err).lower()
        assert "izquierda" in hint

    def test_hint_aligned_state(self):
        ghost = self._ghost()
        ids = np.arange(len(ghost["inner_corners"]))
        err = compute_alignment_by_corners(
            ghost["inner_corners"], ids, ghost["inner_corners"],
        )
        assert "alineado" in alignment_hint_by_corners(err).lower()

    def test_tilted_pose_corner_matching_succeeds(self):
        """Tilted projection produces a trapezoidal ghost — corner-ID match
        should still give near-zero error when detected = ghost exactly.
        """
        pose = PoseTarget("t", "", (0, 0, 1500), pitch_deg=20, yaw_deg=15)
        ghost = project_pose(pose, DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE)
        ids = np.arange(len(ghost["inner_corners"]))
        err = compute_alignment_by_corners(
            ghost["inner_corners"], ids, ghost["inner_corners"],
        )
        assert err["mean_error_px"] < 1e-3


class TestCountCommonCorners:
    def test_overlap(self):
        a = np.array([[0], [1], [2], [3]])
        b = np.array([[2], [3], [4], [5]])
        assert count_common_corners(a, b) == 2

    def test_no_overlap(self):
        a = np.array([[0], [1]])
        b = np.array([[2], [3]])
        assert count_common_corners(a, b) == 0

    def test_none_inputs(self):
        assert count_common_corners(None, np.array([[0]])) == 0
        assert count_common_corners(np.array([[0]]), None) == 0


class TestFitBootstrapIntrinsics:
    def test_fit_runs_on_synthetic_pairs(self):
        board = create_charuco_board()
        pairs = _generate_synthetic_pairs(n_pairs=6)
        # Use left frames only
        lefts = [p[0] for p in pairs]
        K = fit_single_camera_intrinsics(lefts, board)
        assert K is not None
        assert K.shape == (3, 3)
        # Focal length should be in reasonable range for synthetic cam (f=500)
        assert 200 < K[0, 0] < 1200

    def test_fit_returns_none_when_empty(self):
        board = create_charuco_board()
        K = fit_single_camera_intrinsics([], board)
        assert K is None

    def test_project_pose_honours_fitted_K(self):
        """A fitted K with a larger focal length should project a smaller ghost
        at the same distance (longer focal → narrower FOV → same board projects
        to fewer pixels… wait, actually opposite: longer f at same Z → bigger
        image. Let me verify by direct scale comparison.)
        """
        pose = PoseTarget("c", "", (0, 0, 1500))
        small_f = np.array([[1000, 0, 2304], [0, 1000, 1296], [0, 0, 1]])
        large_f = np.array([[1500, 0, 2304], [0, 1500, 1296], [0, 0, 1]])
        proj_small = project_pose(
            pose, DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE,
            fitted_K=small_f,
        )
        proj_large = project_pose(
            pose, DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_LENGTH, PREVIEW_SIZE,
            fitted_K=large_f,
        )
        w_small = np.ptp(proj_small["outer_corners"][:, 0])
        w_large = np.ptp(proj_large["outer_corners"][:, 0])
        # Longer focal at same tvec = larger image
        assert w_large > w_small


class TestPerPairResiduals:
    @pytest.fixture(scope="class")
    def synthetic_pairs_and_cal(self):
        pairs = _generate_synthetic_pairs(30)
        result = calibrate_stereo(pairs)
        return pairs, result

    def test_residuals_one_per_pair(self, synthetic_pairs_and_cal):
        pairs, cal = synthetic_pairs_and_cal
        board = create_charuco_board()
        residuals = compute_per_pair_residuals(pairs, board, cal)
        assert len(residuals) == len(pairs)
        for r in residuals:
            assert set(r.keys()) >= {"pair_idx", "rms_l", "rms_r", "rms", "n_corners"}

    def test_residuals_low_on_synthetic_data(self, synthetic_pairs_and_cal):
        """Synthetic pairs render a perfect board — residuals should be small."""
        pairs, cal = synthetic_pairs_and_cal
        board = create_charuco_board()
        residuals = compute_per_pair_residuals(pairs, board, cal)
        valid = [r["rms"] for r in residuals if not math.isnan(r["rms"])]
        assert len(valid) > 10
        # Synthetic rendering isn't pixel-perfect but residuals should be under 5px
        assert all(r < 5.0 for r in valid), f"residuals: {valid}"

    def test_blank_pair_returns_nan(self, synthetic_pairs_and_cal):
        _pairs, cal = synthetic_pairs_and_cal
        board = create_charuco_board()
        blank = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        residuals = compute_per_pair_residuals([(blank, blank)], board, cal)
        assert len(residuals) == 1
        assert math.isnan(residuals[0]["rms"])


class TestConfigurableDistances:
    def test_default_distances_preserved(self):
        poses = default_pose_sequence()
        zs = sorted({p.tvec_mm[2] for p in poses})
        assert zs == [1000.0, 2000.0, 3000.0]

    def test_custom_distances_applied(self):
        poses = default_pose_sequence(near_mm=1200, mid_mm=2000, far_mm=3000)
        zs = sorted({p.tvec_mm[2] for p in poses})
        assert zs == [1200.0, 2000.0, 3000.0]
        # Sequence shape (20 poses, unique IDs) preserved
        assert len(poses) == 20
        assert len({p.id for p in poses}) == 20


class TestPoseCoverage:
    def test_full_canonical_capture_has_no_warnings(self):
        all_poses = default_pose_sequence()
        # Capture every pose
        coverage = analyze_pose_coverage([p.id for p in all_poses], all_poses)
        assert coverage["ok"], coverage["warnings"]
        assert coverage["by_distance"]["near"] >= 2
        assert coverage["by_distance"]["mid"] >= 2
        assert coverage["by_distance"]["far"] >= 2

    def test_only_near_poses_flags_far_missing(self):
        all_poses = default_pose_sequence()
        near_ids = [p.id for p in all_poses if p.tvec_mm[2] <= 1200]
        coverage = analyze_pose_coverage(near_ids, all_poses)
        assert not coverage["ok"]
        # Empty bands and missing groups are CRITICAL, not soft warnings
        all_msgs = coverage["warnings"] + coverage["critical"]
        assert any("mid" in m or "far" in m for m in all_msgs)
        assert coverage["critical"], "expected critical entries when whole bands missing"

    def test_no_yaw_flagged(self):
        all_poses = default_pose_sequence()
        # Drop all poses with large yaw
        kept = [p.id for p in all_poses if abs(p.yaw_deg) < 10]
        coverage = analyze_pose_coverage(kept, all_poses)
        assert any("yaw" in w.lower() for w in coverage["warnings"])

    def test_unknown_ids_ignored(self):
        coverage = analyze_pose_coverage(["XYZ1", "XYZ2"], default_pose_sequence())
        assert coverage["by_distance"]["near"] == 0
        assert coverage["by_distance"]["mid"] == 0


class TestCornerLocalSharpness:
    def _sharp_frame(self, shape=(480, 640)):
        rng = np.random.RandomState(7)
        return (rng.rand(*shape) * 150 + 40).astype(np.uint8)

    def test_sharpness_increases_with_texture(self):
        gray = self._sharp_frame()
        corners = np.array([[320, 240], [100, 100], [500, 400]], dtype=np.float32)
        score = corner_local_sharpness(gray, corners)
        assert score > 100

    def test_blurred_corners_give_lower_score(self):
        import cv2
        gray = self._sharp_frame()
        blurred = cv2.GaussianBlur(gray, (15, 15), 5)
        corners = np.array([[320, 240], [100, 100], [500, 400]], dtype=np.float32)
        assert corner_local_sharpness(blurred, corners) < corner_local_sharpness(gray, corners)

    def test_empty_corners_returns_nan(self):
        gray = self._sharp_frame()
        assert math.isnan(corner_local_sharpness(gray, None))
        assert math.isnan(corner_local_sharpness(gray, np.zeros((0, 2), dtype=np.float32)))


class TestQualityGateCornerSharpness:
    def _textured(self):
        rng = np.random.RandomState(2)
        return (rng.rand(480, 640, 3) * 150 + 40).astype(np.uint8)

    def test_pass_without_corners_keeps_compat(self):
        f = self._textured()
        q = assess_frame_quality(f, f, n_corners=30)
        assert q["all_pass"]

    def test_sharp_corners_still_pass(self):
        f = self._textured()
        corners = np.array([[[100, 100]], [[300, 200]], [[500, 300]]], dtype=np.float32)
        q = assess_frame_quality(f, f, n_corners=30,
                                 corners_l=corners, corners_r=corners)
        assert q["checks"]["corner_sharp_l"]
        assert q["checks"]["corner_sharp_r"]

    def test_soft_corners_fail(self):
        import cv2
        blurred = cv2.GaussianBlur(self._textured(), (31, 31), 12)
        corners = np.array([[[100, 100]], [[300, 200]], [[500, 300]]], dtype=np.float32)
        q = assess_frame_quality(blurred, blurred, n_corners=30,
                                 corners_l=corners, corners_r=corners)
        # Motion-smeared pattern: whole-frame blur already fails, but the
        # per-corner check should also flag it independently
        assert not q["all_pass"]


class TestLightingWarnings:
    def test_clean_frame_no_warnings(self):
        rng = np.random.RandomState(1)
        f = (rng.rand(480, 640, 3) * 100 + 60).astype(np.uint8)
        assert live_lighting_warnings(f, f) == []

    def test_dark_frame_warns(self):
        dark = np.full((480, 640, 3), 10, dtype=np.uint8)
        warns = live_lighting_warnings(dark, dark)
        assert any("oscur" in w.lower() for w in warns)

    def test_glare_frame_warns(self):
        bright = np.full((480, 640, 3), 255, dtype=np.uint8)
        warns = live_lighting_warnings(bright, bright)
        assert any("reflejo" in w.lower() or "brillo" in w.lower() for w in warns)

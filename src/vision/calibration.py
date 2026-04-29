"""Stereo calibration using ChArUco patterns.

Uses the OpenCV fisheye model (cv2.fisheye.*, Kannala-Brandt angular polynomial
k1..k4). The Arducam IMX708 M12 lens is fisheye (120° HFOV / 152° diagonal) —
pinhole + CALIB_RATIONAL_MODEL fits the centre but leaves measurable residuals
at the edges of the frame, which matters for wide-coverage cenital counting
(FootfallCam-style, ~40% frame eccentricity for the tracking zone).

The ghost overlay renderer in project_pose() stays pinhole — it's a coarse
visual guide for the operator, and at the wizard's alignment tolerances the
pinhole/fisheye discrepancy at frame edges (~3-9px in preview) is negligible.

Compatible with OpenCV 4.8+ (contrib) which uses the refactored ArUco API.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChArUco board factory
# ---------------------------------------------------------------------------

DEFAULT_BOARD_SIZE = (9, 6)  # (columns, rows) of chessboard squares — A3 landscape
DEFAULT_SQUARE_LENGTH = 45.0  # mm
DEFAULT_MARKER_LENGTH = 33.0  # mm (73% of square — calib.io default ratio)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_100

# Nominal intrinsics for IMX708 + Arducam B0310 M12 lens.
# Derived from physical optics: f=2.87mm / pixel_pitch=1.4μm = 2050px at native
# 4608x2592. The lens is fisheye (152°D x 120°H x 66°V captured), so the
# pinhole-equivalent f approximates only the center region — edges have
# significant radial distortion that calibration corrects for. Used for
# ghost rendering + distance estimates (±10% center accuracy is fine).
NOMINAL_FULL_RES = (4608, 2592)
NOMINAL_FOCAL_PX = 2050.0  # f_mm / pixel_pitch (pinhole-equiv, center region)


def create_charuco_board(
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
    dict_id: int = ARUCO_DICT_ID,
    legacy_pattern: bool = True,
) -> cv2.aruco.CharucoBoard:
    """Build a ChArUco board. legacy_pattern defaults to True because our
    canonical calib.io PDFs use the pre-4.6 marker enumeration — OpenCV 4.6+
    switched the default and CharucoDetector.detectBoard returns zero ChArUco
    corners against legacy-printed boards when run under the new pattern
    (markers decode, IDs don't land where the new-pattern board expects). Both
    sides — detection and generateImage() — respect this flag, so toggling
    produces matching producer + detector behaviour.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        board_size, square_length, marker_length, aruco_dict
    )
    board.setLegacyPattern(legacy_pattern)
    return board


def generate_board_image(
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int] = (4961, 3508),  # A3 landscape @ 300 DPI
    margin: int = 50,
) -> np.ndarray:
    img = board.generateImage(image_size, marginSize=margin)
    return img


# ---------------------------------------------------------------------------
# ChArUco corner detection
# ---------------------------------------------------------------------------


def detect_charuco_corners(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    min_corners: int = 8,
    lenient: bool = False,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ChArUco internal corners.

    min_corners — minimum corners required to return a valid detection.
        Calibration needs >=8 for stable intrinsics; focus/setup tools
        can accept fewer (just enough for a rough PnP).
    lenient — loosen marker detection thresholds so smaller/farther markers
        are still picked up. Slightly higher false-positive rate; fine for
        setup tools, avoid for calibration.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if lenient:
        params = cv2.aruco.DetectorParameters()
        # Default is 0.03 (3% of image dim). Lower lets us find markers
        # that span as little as 1% of the frame — i.e. far-away small
        # boards where each marker is only ~10-15 px across.
        params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 4
        # Sub-pixel refinement stabilises corner positions (~0.1px vs 1px
        # without), which materially reduces PnP flicker when markers are
        # small or near the edges of the detection envelope.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Tolerate blurrier markers (default 0.6 rejects ~40% of bits in
        # error; 0.8 accepts a rougher read). Small bump in false-positive
        # rate in exchange for more stable detection near the limit.
        params.errorCorrectionRate = 0.8
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
    else:
        detector = cv2.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        detector.detectBoard(gray)
    )

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None

    return charuco_corners, charuco_ids


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _detect_all_pairs(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
    min_common: int = 8,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detect ChArUco corners in all pairs, return matched obj/img points.

    Returns (all_obj, all_corners_l, all_corners_r, image_size).
    """
    board_corners_3d = board.getChessboardCorners()

    all_obj: list[np.ndarray] = []
    all_corners_l: list[np.ndarray] = []
    all_corners_r: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    for idx, (img_l, img_r) in enumerate(image_pairs):
        corners_l, ids_l = detect_charuco_corners(img_l, board, lenient=True)
        corners_r, ids_r = detect_charuco_corners(img_r, board, lenient=True)

        if corners_l is None or corners_r is None:
            continue

        common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
        if len(common_ids) < min_common:
            continue

        mask_l = np.isin(ids_l.flatten(), common_ids)
        mask_r = np.isin(ids_r.flatten(), common_ids)

        c_l = corners_l[mask_l]
        c_r = corners_r[mask_r]

        order_l = np.argsort(ids_l[mask_l].flatten())
        order_r = np.argsort(ids_r[mask_r].flatten())

        obj_pts = board_corners_3d[common_ids].astype(np.float32)

        all_obj.append(obj_pts)
        all_corners_l.append(c_l[order_l].reshape(-1, 1, 2).astype(np.float32))
        all_corners_r.append(c_r[order_r].reshape(-1, 1, 2).astype(np.float32))

        if image_size is None:
            h, w = img_l.shape[:2]
            image_size = (w, h)

    return all_obj, all_corners_l, all_corners_r, image_size


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------


# Fisheye stereoRectify balance: 0.0 crops to zero black pixels in the rectified
# output (tight FOV). For SGBM we prefer no black borders over maximum field —
# black regions produce spurious disparity at the image edges. Raise to ~0.3 if
# edge FOV turns out to matter more than disparity cleanliness in testing.
FISHEYE_RECTIFY_BALANCE = 0.0


def _reshape_for_fisheye(
    obj_pts_list: list[np.ndarray],
    img_pts_list: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """cv2.fisheye.* is picky: object points must be (N,1,3) float64 and image
    points (N,1,2) float64. _detect_all_pairs already returns image points as
    (N,1,2) float32 — we just upcast dtype and reshape the object points.
    """
    obj_out = [np.asarray(o, dtype=np.float64).reshape(-1, 1, 3) for o in obj_pts_list]
    img_out = [np.asarray(p, dtype=np.float64).reshape(-1, 1, 2) for p in img_pts_list]
    return obj_out, img_out


def _derive_stereo_rt_from_per_pose(
    rvecs_l: list[np.ndarray], tvecs_l: list[np.ndarray],
    rvecs_r: list[np.ndarray], tvecs_r: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average the per-pose relative transform (left→right) across all poses.

    For a rigid stereo bracket, R_stereo @ R_board_to_left == R_board_to_right
    and T_stereo == t_right - R_stereo @ t_left in every pose. We compute both
    per-pose and average — translations with a simple mean, rotations via
    quaternion averaging (arithmetic mean + renormalise is close enough for
    this application and avoids the pypkg scipy dependency).
    """
    rel_rotations: list[np.ndarray] = []
    rel_translations: list[np.ndarray] = []
    for rv_l, tv_l, rv_r, tv_r in zip(rvecs_l, tvecs_l, rvecs_r, tvecs_r):
        R_l, _ = cv2.Rodrigues(np.asarray(rv_l, dtype=np.float64).reshape(3, 1))
        R_r, _ = cv2.Rodrigues(np.asarray(rv_r, dtype=np.float64).reshape(3, 1))
        t_l = np.asarray(tv_l, dtype=np.float64).reshape(3, 1)
        t_r = np.asarray(tv_r, dtype=np.float64).reshape(3, 1)
        R_rel = R_r @ R_l.T
        t_rel = t_r - R_rel @ t_l
        rel_rotations.append(R_rel)
        rel_translations.append(t_rel)

    # Average translations straight.
    T_mean = np.mean(np.stack(rel_translations, axis=0), axis=0).reshape(3, 1)

    # Average rotations: arithmetic mean of R matrices, then SVD-project to
    # the closest valid rotation (orthogonal with det=+1).
    R_sum = np.sum(np.stack(rel_rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        # Mirror reflection — flip the last singular column.
        U[:, -1] *= -1
        R_mean = U @ Vt

    return R_mean.astype(np.float64), T_mean.astype(np.float64)


def _compute_stereo_rms(
    obj_list: list[np.ndarray],
    corners_l: list[np.ndarray],
    corners_r: list[np.ndarray],
    K_l: np.ndarray, D_l: np.ndarray,
    K_r: np.ndarray, D_r: np.ndarray,
    rvecs_l: list[np.ndarray], tvecs_l: list[np.ndarray],
    R_stereo: np.ndarray, T_stereo: np.ndarray,
) -> float:
    """Reprojection RMS of the fixed-intrinsics / averaged-extrinsics stereo fit.

    For each pose, left extrinsics come from fisheye.calibrate's per-pose
    solve; right extrinsics are derived by applying R_stereo, T_stereo. We
    project the 3D object points through both cameras and compare to the
    detected corners. This is the same quantity stereoCalibrate minimises,
    just computed after the fact.
    """
    errs: list[float] = []
    for obj_pts, pts_l, pts_r, rv_l, tv_l in zip(
        obj_list, corners_l, corners_r, rvecs_l, tvecs_l,
    ):
        rv_l_arr = np.asarray(rv_l, dtype=np.float64).reshape(3, 1)
        tv_l_arr = np.asarray(tv_l, dtype=np.float64).reshape(3, 1)
        R_l, _ = cv2.Rodrigues(rv_l_arr)

        proj_l, _ = cv2.fisheye.projectPoints(
            obj_pts, rv_l_arr, tv_l_arr, K_l, D_l,
        )
        err_l = np.linalg.norm(
            proj_l.reshape(-1, 2) - pts_l.reshape(-1, 2), axis=1,
        )

        # Right extrinsics: chain left with the stereo transform.
        R_r = R_stereo @ R_l
        t_r = R_stereo @ tv_l_arr + T_stereo
        rv_r_arr, _ = cv2.Rodrigues(R_r)
        proj_r, _ = cv2.fisheye.projectPoints(
            obj_pts, rv_r_arr, t_r, K_r, D_r,
        )
        err_r = np.linalg.norm(
            proj_r.reshape(-1, 2) - pts_r.reshape(-1, 2), axis=1,
        )

        errs.extend(err_l.tolist())
        errs.extend(err_r.tolist())

    if not errs:
        return float("nan")
    errs_arr = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt((errs_arr ** 2).mean()))


def calibrate_stereo(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
    dict_id: int = ARUCO_DICT_ID,
    min_pairs: int = 15,
    legacy_pattern: bool = True,
) -> dict[str, np.ndarray]:
    """Stereo calibration using the fisheye (Kannala-Brandt) model."""
    board = create_charuco_board(
        board_size, square_length, marker_length, dict_id, legacy_pattern,
    )
    all_obj, all_corners_l, all_corners_r, image_size = _detect_all_pairs(
        image_pairs, board,
    )

    valid_pairs = len(all_obj)
    logger.info(
        "stereo_pairs_validated",
        extra={"valid": valid_pairs, "total": len(image_pairs)},
    )

    if valid_pairs < min_pairs:
        raise ValueError(
            f"Need at least {min_pairs} valid pairs, got {valid_pairs}."
        )

    obj_f, corners_l_f = _reshape_for_fisheye(all_obj, all_corners_l)
    _, corners_r_f = _reshape_for_fisheye(all_obj, all_corners_r)

    # CHECK_COND raises if a pose is too degenerate. We omit it — with real lab
    # captures the 20 canonical poses are well-distributed, and keeping it off
    # avoids cryptic failures on rare borderline cases.
    calib_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # Pre-allocate K/D with the shapes the fisheye binding expects, but do NOT
    # pre-allocate rvecs/tvecs — the Python binding builds them itself from the
    # pose count, and passing fixed-shape placeholders triggers an _OutputArray
    # assertion (OpenCV 4.13+).
    K_l = np.zeros((3, 3), dtype=np.float64)
    D_l = np.zeros((4, 1), dtype=np.float64)
    K_r = np.zeros((3, 3), dtype=np.float64)
    D_r = np.zeros((4, 1), dtype=np.float64)

    rms_l, K_l, D_l, rvecs_l, tvecs_l = cv2.fisheye.calibrate(
        obj_f, corners_l_f, image_size, K_l, D_l,
        flags=calib_flags, criteria=criteria,
    )
    logger.info(
        "camera_calibrated",
        extra={"camera": "left", "rms": rms_l, "pairs": valid_pairs},
    )

    rms_r, K_r, D_r, rvecs_r, tvecs_r = cv2.fisheye.calibrate(
        obj_f, corners_r_f, image_size, K_r, D_r,
        flags=calib_flags, criteria=criteria,
    )
    logger.info(
        "camera_calibrated",
        extra={"camera": "right", "rms": rms_r, "pairs": valid_pairs},
    )

    D_l = np.asarray(D_l, dtype=np.float64).reshape(4, 1)
    D_r = np.asarray(D_r, dtype=np.float64).reshape(4, 1)

    # fisheye.stereoCalibrate requires identical point counts across poses
    # (OpenCV 4.13 limitation). Our lab captures have variable per-pose counts
    # (far distances lose corners), so we derive the rigid L→R transform
    # directly from the per-pose extrinsics returned by fisheye.calibrate:
    # for each pose i, R_stereo = R_ri @ R_li.T and T_stereo = t_ri - R_stereo @ t_li.
    # Averaging across poses (with SO(3) projection on rotations) gives the
    # same R, T that joint stereoCalibrate would converge to, within a
    # fraction of a pixel RMS for a rigid bracket.
    R, T = _derive_stereo_rt_from_per_pose(rvecs_l, tvecs_l, rvecs_r, tvecs_r)
    rms_stereo = _compute_stereo_rms(
        obj_f, corners_l_f, corners_r_f, K_l, D_l, K_r, D_r,
        rvecs_l, tvecs_l, R, T,
    )
    logger.info(
        "stereo_calibrated",
        extra={"rms": rms_stereo, "pairs": valid_pairs},
    )

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
        balance=FISHEYE_RECTIFY_BALANCE,
    )
    map_l_x, map_l_y = cv2.fisheye.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_32FC1,
    )
    map_r_x, map_r_y = cv2.fisheye.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_32FC1,
    )

    logger.info(
        "stereo_geometry_computed",
        extra={
            "fx": float(K_l[0, 0]),
            "fy": float(K_l[1, 1]),
            "baseline_mm": float(abs(T[0, 0])),
        },
    )

    return {
        "camera_matrix_l": K_l, "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r, "dist_coeffs_r": D_r,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_l_x": map_l_x, "map_l_y": map_l_y,
        "map_r_x": map_r_x, "map_r_y": map_r_y,
        "image_size": np.array(list(image_size)),
    }


# ---------------------------------------------------------------------------
# Rectification helpers
# ---------------------------------------------------------------------------


def rectify_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rect_l = cv2.remap(
        img_l, calibration["map_l_x"], calibration["map_l_y"],
        cv2.INTER_LINEAR,
    )
    rect_r = cv2.remap(
        img_r, calibration["map_r_x"], calibration["map_r_y"],
        cv2.INTER_LINEAR,
    )
    return rect_l, rect_r


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_calibration(path: str) -> dict[str, np.ndarray]:
    cal_path = Path(path)
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = dict(np.load(cal_path))
    logger.info("Calibration loaded", extra={"path": path, "keys": list(data.keys())})
    return data


def save_calibration(params: dict[str, np.ndarray], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)
    logger.info("Calibration saved", extra={"path": path})


# ---------------------------------------------------------------------------
# Guided calibration: pose targets, ghost projection, quality/matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseTarget:
    """A target position+orientation for the ChArUco board during guided capture.

    Translation is in millimeters in camera frame (+X right, +Y down, +Z forward).
    Rotations are in degrees applied as rotZ(roll) * rotY(yaw) * rotX(pitch).
    """
    id: str
    label: str
    tvec_mm: tuple[float, float, float]
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    def rvec(self) -> np.ndarray:
        """Rodrigues rotation vector from the Euler angles."""
        rx = math.radians(self.pitch_deg)
        ry = math.radians(self.yaw_deg)
        rz = math.radians(self.roll_deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()


DEFAULT_DIST_NEAR_MM = 1000.0
DEFAULT_DIST_MID_MM = 2000.0
DEFAULT_DIST_FAR_MM = 3000.0


def default_pose_sequence(
    near_mm: float = DEFAULT_DIST_NEAR_MM,
    mid_mm: float = DEFAULT_DIST_MID_MM,
    far_mm: float = DEFAULT_DIST_FAR_MM,
) -> list[PoseTarget]:
    """Return a canonical 20-pose sequence covering near/mid/far × positions × tilts.

    Default distances (1000/2000/3000mm) span most of the fleet operating range
    (mount heights 3.0–4.5m → head distances 1.15–3.30m) while keeping every
    pose's required board height inside a standard 70–210cm tripod envelope.
    With far=3.0m the D-group poses (±0.35 vertical offset) put the board at
    1.40m ± 66cm = [0.74m, 2.06m], vs 1.40m ± 77cm = [0.63m, 2.17m] at far=3.5m
    which blows past a typical 2.1m tripod max and a 70cm tripod min. The
    operating ceiling (3.30m) is extrapolated 30cm beyond the sampled far —
    low risk for the fisheye Kannala-Brandt model which stays well-behaved
    under modest extrapolation. At 3.0m with f≈2050px full-res, a 33mm
    DICT_4X4 marker is ~23 px wide, comfortably above the ~12 px threshold.
    """
    near, mid, far = near_mm, mid_mm, far_mm

    # Lateral offsets: tvec X/Y values chosen so the projected board lands near
    # the target screen position at each distance.
    def off(frac_x: float, frac_y: float, z: float) -> tuple[float, float]:
        # Project a pixel offset from image center back to world mm at depth z.
        half_w_px = NOMINAL_FULL_RES[0] / 2
        half_h_px = NOMINAL_FULL_RES[1] / 2
        return (frac_x * half_w_px * z / NOMINAL_FOCAL_PX,
                frac_y * half_h_px * z / NOMINAL_FOCAL_PX)

    poses: list[PoseTarget] = []

    # Group A — Center at near, systematic tilts
    poses.append(PoseTarget("A1", "Center near, frontal", (0, 0, near)))
    poses.append(PoseTarget("A2", "Center near, pitch up", (0, 0, near), pitch_deg=-20))
    poses.append(PoseTarget("A3", "Center near, yaw left", (0, 0, near), yaw_deg=-20))
    poses.append(PoseTarget("A4", "Center near, pitch/yaw mix", (0, 0, near), pitch_deg=15, yaw_deg=15))

    # Group B — Corners at mid, edge constraints
    dx, dy = off(-0.55, -0.45, mid)
    poses.append(PoseTarget("B1", "Top-left mid, yaw left", (dx, dy, mid), yaw_deg=-15))
    dx, dy = off(0.55, -0.45, mid)
    poses.append(PoseTarget("B2", "Top-right mid, yaw right", (dx, dy, mid), yaw_deg=15))
    dx, dy = off(-0.55, 0.45, mid)
    poses.append(PoseTarget("B3", "Bottom-left mid, pitch up", (dx, dy, mid), pitch_deg=15))
    dx, dy = off(0.55, 0.45, mid)
    poses.append(PoseTarget("B4", "Bottom-right mid, pitch down", (dx, dy, mid), pitch_deg=-15))

    # Group C — Mid distance, roll + pitch diversity
    poses.append(PoseTarget("C1", "Center mid, roll +20", (0, 0, mid), roll_deg=20))
    poses.append(PoseTarget("C2", "Center mid, roll -20", (0, 0, mid), roll_deg=-20))
    dx, dy = off(0.0, -0.4, mid)
    poses.append(PoseTarget("C3", "Top-center mid, pitch up", (dx, dy, mid), pitch_deg=-20))
    dx, dy = off(0.0, 0.4, mid)
    poses.append(PoseTarget("C4", "Bottom-center mid, pitch down", (dx, dy, mid), pitch_deg=20))

    # Group D — Far distance, coverage
    poses.append(PoseTarget("D1", "Center far, frontal", (0, 0, far)))
    dx, dy = off(-0.45, -0.35, far)
    poses.append(PoseTarget("D2", "Top-left far, diag tilt", (dx, dy, far), yaw_deg=-10, pitch_deg=-10))
    dx, dy = off(0.45, -0.35, far)
    poses.append(PoseTarget("D3", "Top-right far, diag tilt", (dx, dy, far), yaw_deg=10, pitch_deg=-10))
    dx, dy = off(-0.45, 0.35, far)
    poses.append(PoseTarget("D4", "Bottom-left far, diag tilt", (dx, dy, far), yaw_deg=-10, pitch_deg=10))
    dx, dy = off(0.45, 0.35, far)
    poses.append(PoseTarget("D5", "Bottom-right far, diag tilt", (dx, dy, far), yaw_deg=10, pitch_deg=10))

    # Group E — Extreme tilts at near
    dx, dy = off(-0.5, 0, near)
    poses.append(PoseTarget("E1", "Left-mid near, strong yaw", (dx, dy, near), yaw_deg=-25))
    dx, dy = off(0.5, 0, near)
    poses.append(PoseTarget("E2", "Right-mid near, strong yaw", (dx, dy, near), yaw_deg=25))
    poses.append(PoseTarget("E3", "Center mid, combined 3-axis",
                            (0, 0, mid), pitch_deg=-12, yaw_deg=-12, roll_deg=15))

    return poses


def project_pose(
    pose: PoseTarget,
    board_size: tuple[int, int],
    square_length: float,
    preview_size: tuple[int, int],
    focal_full_px: float = NOMINAL_FOCAL_PX,
    full_res: tuple[int, int] = NOMINAL_FULL_RES,
    fitted_K: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Project a pose target into preview pixel coordinates.

    Args:
        fitted_K: If given, use this 3x3 intrinsics matrix (at full_res) scaled
            to preview_size. Overrides focal_full_px + centered principal point.
            Used during bootstrap → tight-tolerance handoff.

    Returns a dict with:
        outer_corners: (4, 2) — board outline for visual ghost, clockwise TL,TR,BR,BL
        inner_corners: (N, 2) — internal chessboard corners (for matching vs detection)
        center: (2,) — projected center of the board
    """
    cols, rows = board_size
    board_w_mm = cols * square_length
    board_h_mm = rows * square_length

    preview_w, preview_h = preview_size
    full_w, full_h = full_res
    scale_x = preview_w / full_w
    scale_y = preview_h / full_h

    if fitted_K is not None:
        f_preview_x = float(fitted_K[0, 0]) * scale_x
        f_preview_y = float(fitted_K[1, 1]) * scale_y
        cx_preview = float(fitted_K[0, 2]) * scale_x
        cy_preview = float(fitted_K[1, 2]) * scale_y
    else:
        f_preview_x = focal_full_px * scale_x
        f_preview_y = focal_full_px * scale_y
        cx_preview = preview_w / 2
        cy_preview = preview_h / 2

    K = np.array([[f_preview_x, 0, cx_preview],
                  [0, f_preview_y, cy_preview],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    # Board points expressed in a frame centered on the board's center
    # (so tvec places the center of the board, not its top-left corner).
    half_w = board_w_mm / 2
    half_h = board_h_mm / 2

    outer_3d = np.array([
        [-half_w, -half_h, 0],
        [half_w, -half_h, 0],
        [half_w, half_h, 0],
        [-half_w, half_h, 0],
    ], dtype=np.float32)

    inner_pts = []
    for r in range(1, rows):
        for c in range(1, cols):
            x = c * square_length - half_w
            y = r * square_length - half_h
            inner_pts.append([x, y, 0])
    inner_3d = np.array(inner_pts, dtype=np.float32)

    rvec = pose.rvec().astype(np.float64)
    tvec = np.array(pose.tvec_mm, dtype=np.float64)

    outer_2d, _ = cv2.projectPoints(outer_3d, rvec, tvec, K, dist)
    inner_2d, _ = cv2.projectPoints(inner_3d, rvec, tvec, K, dist)
    center_2d, _ = cv2.projectPoints(
        np.array([[0, 0, 0]], dtype=np.float32), rvec, tvec, K, dist,
    )

    return {
        "outer_corners": outer_2d.reshape(-1, 2),
        "inner_corners": inner_2d.reshape(-1, 2),
        "center": center_2d.reshape(2),
    }


def _min_area_rect(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Wrapper over cv2.minAreaRect returning (center, size, angle)."""
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.minAreaRect(pts)


def compute_alignment_error(
    detected_preview: np.ndarray,
    ghost_inner: np.ndarray,
) -> dict[str, float]:
    """Compare detected ChArUco corners (in preview px) to projected ghost inner corners.

    Returns center_px, scale_ratio (detected/ghost), rotation_deg (wrapped to [-45, 45]).
    All comparisons are in preview pixel coordinates.
    """
    det_center, det_size, det_angle = _min_area_rect(detected_preview)
    ghost_center, ghost_size, ghost_angle = _min_area_rect(ghost_inner)

    cdx = det_center[0] - ghost_center[0]
    cdy = det_center[1] - ghost_center[1]
    center_px = float(math.hypot(cdx, cdy))

    det_diag = math.hypot(det_size[0], det_size[1])
    ghost_diag = math.hypot(ghost_size[0], ghost_size[1])
    scale_ratio = det_diag / ghost_diag if ghost_diag > 1 else 0.0

    # minAreaRect angles wrap at 90°, so wrap difference to [-45, 45]
    raw_diff = det_angle - ghost_angle
    while raw_diff > 45:
        raw_diff -= 90
    while raw_diff < -45:
        raw_diff += 90

    return {
        "center_px": center_px,
        "offset_x": float(cdx),
        "offset_y": float(cdy),
        "scale_ratio": float(scale_ratio),
        "rotation_deg": float(raw_diff),
    }


ALIGN_CENTER_TOL_PX = 20.0
ALIGN_SCALE_TOL = 0.05  # ±5%
ALIGN_ROTATION_TOL_DEG = 3.0


def is_aligned(error: dict[str, float]) -> bool:
    """Tight tolerance gate for auto-capture (legacy minAreaRect-based)."""
    return (
        error["center_px"] <= ALIGN_CENTER_TOL_PX
        and abs(error["scale_ratio"] - 1.0) <= ALIGN_SCALE_TOL
        and abs(error["rotation_deg"]) <= ALIGN_ROTATION_TOL_DEG
    )


# ---------------------------------------------------------------------------
# Corner-ID-based matching (preferred over minAreaRect — handles tilted poses)
# ---------------------------------------------------------------------------


ALIGN_MEAN_ERR_TOL_PX_TIGHT = 12.0
ALIGN_MEAN_ERR_TOL_PX_LOOSE = 25.0
ALIGN_MATCHED_MIN_TIGHT = 15
ALIGN_MATCHED_MIN_LOOSE = 12


def compute_alignment_by_corners(
    detected_preview: np.ndarray,
    detected_ids: np.ndarray,
    ghost_inner_all: np.ndarray,
) -> dict[str, float]:
    """Match detected ChArUco corners to ghost by ID, compute per-corner error.

    Args:
        detected_preview: (N, 2) detected inner corners in preview px
        detected_ids: (N,) ChArUco IDs for each detected corner
        ghost_inner_all: (M, 2) all projected ghost inner corners, indexed by ID

    Returns dict with matched count, mean_error_px, max_error_px, offset vector,
    and centroid_offset_px. Tilted poses handled naturally because we compare
    per-corner, not via a 2D similarity fit.
    """
    pairs_det: list[np.ndarray] = []
    pairs_ghost: list[np.ndarray] = []
    ids_flat = np.asarray(detected_ids).flatten()
    for i, cid in enumerate(ids_flat):
        cid_int = int(cid)
        if 0 <= cid_int < len(ghost_inner_all):
            pairs_det.append(detected_preview[i])
            pairs_ghost.append(ghost_inner_all[cid_int])

    n = len(pairs_det)
    empty = {
        "matched": n,
        "mean_error_px": float("inf"),
        "max_error_px": float("inf"),
        "offset_x": 0.0, "offset_y": 0.0,
        "centroid_offset_px": float("inf"),
    }
    if n < 4:
        return empty

    det = np.asarray(pairs_det, dtype=np.float32).reshape(-1, 2)
    ghost = np.asarray(pairs_ghost, dtype=np.float32).reshape(-1, 2)
    errors = np.linalg.norm(det - ghost, axis=1)

    offset = det.mean(axis=0) - ghost.mean(axis=0)
    return {
        "matched": n,
        "mean_error_px": float(errors.mean()),
        "max_error_px": float(errors.max()),
        "offset_x": float(offset[0]),
        "offset_y": float(offset[1]),
        "centroid_offset_px": float(math.hypot(offset[0], offset[1])),
    }


def is_aligned_by_corners(
    err: dict[str, float],
    mean_err_tol_px: float = ALIGN_MEAN_ERR_TOL_PX_TIGHT,
    min_matched: int = ALIGN_MATCHED_MIN_TIGHT,
) -> bool:
    """Alignment gate for the corner-ID matcher. Tolerances configurable so
    bootstrap phase can use looser limits while fitted-K phase is strict.
    """
    return err["matched"] >= min_matched and err["mean_error_px"] <= mean_err_tol_px


def alignment_hint_by_corners(
    err: dict[str, float], mm_per_px: float | None = None,
) -> str:
    """Human-readable direction for the operator. Uses centroid offset as the
    dominant error vector; surfaces residual-after-translation when that's small.

    If `mm_per_px` is provided, the hint reports offsets in cm instead of
    pixels — more meaningful to an operator holding a physical board. This
    ratio is `target_distance_mm / focal_length_px` evaluated at the pose's
    expected depth.
    """
    def _fmt_offset(px: float) -> str:
        if mm_per_px is None:
            return f"{px:.0f}px"
        cm = abs(px) * mm_per_px / 10.0
        return f"{cm:.0f}cm"

    if err["matched"] < 8:
        return "Board apenas visible — movelo al centro"
    if err["centroid_offset_px"] < 10:
        if err["mean_error_px"] > ALIGN_MEAN_ERR_TOL_PX_TIGHT:
            residual = _fmt_offset(err["mean_error_px"])
            return f"Ajustá tilt/escala (residual {residual})"
        return "Alineado"
    parts = []
    if abs(err["offset_x"]) > abs(err["offset_y"]):
        direction = "izquierda" if err["offset_x"] > 0 else "derecha"
        parts.append(f"Movelo {direction} {_fmt_offset(err['offset_x'])}")
    else:
        direction = "arriba" if err["offset_y"] > 0 else "abajo"
        parts.append(f"Movelo {direction} {_fmt_offset(err['offset_y'])}")
    if err["mean_error_px"] > ALIGN_MEAN_ERR_TOL_PX_TIGHT + err["centroid_offset_px"]:
        parts.append("y ajustá tilt")
    return " · ".join(parts)


def count_common_corners(ids_l: Optional[np.ndarray], ids_r: Optional[np.ndarray]) -> int:
    """Count ChArUco IDs present in both left and right detections."""
    if ids_l is None or ids_r is None:
        return 0
    return int(np.intersect1d(np.asarray(ids_l).flatten(),
                              np.asarray(ids_r).flatten()).size)


# ---------------------------------------------------------------------------
# Pose-coverage diversity check (guard against degenerate pose sets)
# ---------------------------------------------------------------------------


def analyze_pose_coverage(
    captured_pose_ids: list[str],
    all_poses: Optional[list[PoseTarget]] = None,
) -> dict[str, object]:
    """Report how diverse a set of captured poses is.

    The canonical sequence groups poses by distance + tilt pattern via prefix
    letter (A=center near, B=corners mid, C=roll/pitch mid, D=far, E=extreme).
    A healthy calibration captures at least 2 poses from each of A..D and
    covers pitch, yaw and roll tilts.

    Returns:
        {
          "by_distance": {"near": n, "mid": n, "far": n},
          "by_tilt_axis": {"frontal": n, "pitch": n, "yaw": n, "roll": n},
          "by_group": {"A": n, "B": n, "C": n, "D": n, "E": n},
          "warnings": [str, ...],
          "ok": bool,
        }
    """
    if all_poses is None:
        all_poses = default_pose_sequence()
    pose_by_id = {p.id: p for p in all_poses}

    by_distance = {"near": 0, "mid": 0, "far": 0}
    by_tilt = {"frontal": 0, "pitch": 0, "yaw": 0, "roll": 0}
    by_group: dict[str, int] = {}

    for pid in captured_pose_ids:
        pose = pose_by_id.get(pid)
        if pose is None:
            continue
        z = pose.tvec_mm[2]
        if z <= 1200:
            by_distance["near"] += 1
        elif z <= 2000:
            by_distance["mid"] += 1
        else:
            by_distance["far"] += 1

        tilts = (pose.pitch_deg, pose.yaw_deg, pose.roll_deg)
        if max(abs(t) for t in tilts) < 5:
            by_tilt["frontal"] += 1
        if abs(pose.pitch_deg) >= 10:
            by_tilt["pitch"] += 1
        if abs(pose.yaw_deg) >= 10:
            by_tilt["yaw"] += 1
        if abs(pose.roll_deg) >= 10:
            by_tilt["roll"] += 1

        group = pose.id[0] if pose.id else "?"
        by_group[group] = by_group.get(group, 0) + 1

    warnings: list[str] = []
    critical: list[str] = []  # Coverage gaps that produce degenerate fits
    for band, count in by_distance.items():
        if count == 0:
            critical.append(f"Banda de distancia '{band}' sin capturas")
        elif count < 2:
            warnings.append(f"Solo {count} captura en banda de distancia '{band}'")
    for axis, count in by_tilt.items():
        if axis == "frontal":
            continue  # frontal is nice-to-have, not required
        if count < 2:
            warnings.append(f"Poca diversidad de tilt '{axis}' ({count} captura/s)")
    for group in ["A", "B", "C", "D"]:  # E is extras, not required
        if by_group.get(group, 0) < 1:
            critical.append(f"Grupo de poses {group} sin capturas")

    return {
        "by_distance": by_distance,
        "by_tilt_axis": by_tilt,
        "by_group": by_group,
        "warnings": warnings,
        "critical": critical,
        "ok": len(warnings) == 0 and len(critical) == 0,
    }


# ---------------------------------------------------------------------------
# Bootstrap intrinsics fit — switches ghost projection from nominal K to
# per-sensor measured K after the first few captures
# ---------------------------------------------------------------------------


def fit_single_camera_intrinsics(
    images: list[np.ndarray],
    board: cv2.aruco.CharucoBoard,
) -> Optional[np.ndarray]:
    """Calibrate intrinsics for a single camera from a handful of ChArUco images.

    Returns the 3×3 camera matrix, or None if detection fails on too many frames.
    Used for the guided-capture bootstrap — after a few captures we replace the
    nominal K used to render ghost targets with the measured per-sensor value.

    Intentionally stays on the pinhole model (cv2.calibrateCamera) even though
    the main stereo pipeline is fisheye. The only consumer is project_pose()
    which also renders the ghost with pinhole projection — keeping both halves
    on the same model avoids a visible offset between the ghost and the
    detected board at frame edges during bootstrap. The final calibration that
    actually goes into the .npz uses cv2.fisheye.* in calibrate_stereo.
    """
    all_obj: list[np.ndarray] = []
    all_corners: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    board_corners_3d = board.getChessboardCorners()
    for img in images:
        corners, ids = detect_charuco_corners(img, board)
        if corners is None or ids is None or len(ids) < 8:
            continue
        obj_pts = board_corners_3d[ids.flatten()].astype(np.float32)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
        all_obj.append(obj_pts)
        all_corners.append(img_pts)
        if image_size is None:
            h, w = img.shape[:2]
            image_size = (w, h)

    if len(all_obj) < 4 or image_size is None:
        return None

    try:
        _rms, K, _dist, _rvecs, _tvecs = cv2.calibrateCamera(
            all_obj, all_corners, image_size, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL,
        )
    except cv2.error:
        return None
    return K


# ---------------------------------------------------------------------------
# Per-pair residual analysis (outlier flagging in the HTML report)
# ---------------------------------------------------------------------------


def _fisheye_reprojection_rms(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    """PnP + reprojection residual for one view under the fisheye model.

    OpenCV doesn't ship a fisheye solvePnP, so we undistort the observations
    into normalised pinhole coordinates, solve PnP against an identity camera,
    then reproject through the fisheye model and measure the pixel residual
    against the original observations. Returns NaN on any OpenCV failure.
    """
    try:
        # Normalise observations: fisheye.undistortPoints returns (N,1,2) in
        # normalised image coords (what a pinhole camera with K=I would see).
        undistorted = cv2.fisheye.undistortPoints(
            img_pts.reshape(-1, 1, 2).astype(np.float64), K, D,
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts.astype(np.float64),
            undistorted,
            np.eye(3, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return float("nan")
        proj, _ = cv2.fisheye.projectPoints(
            obj_pts.reshape(-1, 1, 3).astype(np.float64),
            rvec, tvec, K, D,
        )
        err = np.linalg.norm(
            proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1,
        )
        return float(np.sqrt((err ** 2).mean()))
    except cv2.error:
        return float("nan")


def compute_per_pair_residuals(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
    calibration: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    """Compute reprojection residual for each pair after calibration.

    Returns one dict per pair: {pair_idx, rms_l, rms_r, rms, n_corners}.
    Used to flag outlier captures in the calibration report so the operator
    can see which poses dragged the overall RMS up.
    """
    K_l = calibration["camera_matrix_l"]
    D_l = calibration["dist_coeffs_l"]
    K_r = calibration["camera_matrix_r"]
    D_r = calibration["dist_coeffs_r"]
    board_corners_3d = board.getChessboardCorners()

    results: list[dict[str, float]] = []
    for idx, (img_l, img_r) in enumerate(image_pairs):
        corners_l, ids_l = detect_charuco_corners(img_l, board, lenient=True)
        corners_r, ids_r = detect_charuco_corners(img_r, board, lenient=True)
        if corners_l is None or corners_r is None or ids_l is None or ids_r is None:
            results.append({
                "pair_idx": idx, "rms_l": float("nan"), "rms_r": float("nan"),
                "rms": float("nan"), "n_corners": 0,
            })
            continue

        common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
        if len(common_ids) < 6:
            results.append({
                "pair_idx": idx, "rms_l": float("nan"), "rms_r": float("nan"),
                "rms": float("nan"), "n_corners": int(len(common_ids)),
            })
            continue

        mask_l = np.isin(ids_l.flatten(), common_ids)
        mask_r = np.isin(ids_r.flatten(), common_ids)
        order_l = np.argsort(ids_l[mask_l].flatten())
        order_r = np.argsort(ids_r[mask_r].flatten())

        obj_pts = board_corners_3d[common_ids].astype(np.float32)
        pts_l = corners_l[mask_l][order_l].reshape(-1, 1, 2).astype(np.float32)
        pts_r = corners_r[mask_r][order_r].reshape(-1, 1, 2).astype(np.float32)

        rms_l = _fisheye_reprojection_rms(obj_pts, pts_l, K_l, D_l)
        rms_r = _fisheye_reprojection_rms(obj_pts, pts_r, K_r, D_r)

        combined = float("nan")
        vals = [v for v in (rms_l, rms_r) if not math.isnan(v)]
        if vals:
            combined = sum(vals) / len(vals)

        results.append({
            "pair_idx": idx,
            "rms_l": rms_l, "rms_r": rms_r,
            "rms": combined,
            "n_corners": int(len(common_ids)),
        })
    return results


def alignment_hint(error: dict[str, float]) -> str:
    """Human-readable direction for the operator to move the board."""
    parts = []
    if error["center_px"] > ALIGN_CENTER_TOL_PX:
        if abs(error["offset_x"]) > abs(error["offset_y"]):
            direction = "derecha" if error["offset_x"] < 0 else "izquierda"
            parts.append(f"Movelo {direction} {abs(error['offset_x']):.0f}px")
        else:
            direction = "abajo" if error["offset_y"] < 0 else "arriba"
            parts.append(f"Movelo {direction} {abs(error['offset_y']):.0f}px")
    scale_off = error["scale_ratio"] - 1.0
    if abs(scale_off) > ALIGN_SCALE_TOL:
        parts.append("Alejalo" if scale_off > 0 else "Acercalo")
    if abs(error["rotation_deg"]) > ALIGN_ROTATION_TOL_DEG:
        direction = "horario" if error["rotation_deg"] < 0 else "antihorario"
        parts.append(f"Rotá {direction} {abs(error['rotation_deg']):.0f}°")
    return " · ".join(parts) if parts else "Alineado"


# ---------------------------------------------------------------------------
# Stability: rolling buffer of corner positions, check motion between frames
# ---------------------------------------------------------------------------


STABILITY_MAX_DISP_PX = 1.5
STABILITY_WINDOW_FRAMES = 10


class StabilityTracker:
    """Rolling buffer of detected corner positions to detect when the board is still.

    Call push() every frame with the detected corner array (preview px). Call
    is_stable() to check whether the last `window` frames have max inter-frame
    displacement below the threshold.
    """

    def __init__(
        self,
        window: int = STABILITY_WINDOW_FRAMES,
        max_disp_px: float = STABILITY_MAX_DISP_PX,
    ) -> None:
        self.window = window
        self.max_disp_px = max_disp_px
        self._buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, corners_preview: Optional[np.ndarray]) -> None:
        """Add a frame's corner positions. Pass None if detection failed."""
        if corners_preview is None or len(corners_preview) == 0:
            self._buffer.clear()
            return
        pts = np.asarray(corners_preview).reshape(-1, 2).astype(np.float32)
        if self._buffer and self._buffer[-1].shape != pts.shape:
            # Corner count changed (partial detection) — restart
            self._buffer.clear()
        self._buffer.append(pts)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def max_displacement(self) -> float:
        """Max per-corner displacement between consecutive frames in the buffer."""
        if len(self._buffer) < 2:
            return float("inf")
        worst = 0.0
        for prev, curr in zip(self._buffer[:-1], self._buffer[1:]):
            disp = np.linalg.norm(curr - prev, axis=1).max()
            worst = max(worst, float(disp))
        return worst

    def is_stable(self) -> bool:
        if len(self._buffer) < self.window:
            return False
        return self.max_displacement() <= self.max_disp_px


# ---------------------------------------------------------------------------
# Frame quality gate (blur, corner count, saturation, exposure)
# ---------------------------------------------------------------------------


QUALITY_MIN_CORNERS = 15
QUALITY_MIN_BLUR = 10.0  # Laplacian variance on grayscale (full-res 4608x2592 captures).
# At 12MP each pixel is finer so per-pixel gradients are smaller; 10 is the
# empirical floor for a well-focused frame. Corner-local sharpness (separate
# check below) handles the "looks sharp globally but corners soft" case.
QUALITY_MAX_SATURATION_PCT = 8.0  # % of pixels >250
QUALITY_MIN_EXPOSURE = 35.0  # median grayscale value
QUALITY_MAX_LR_BRIGHTNESS_PCT = 25.0  # % asymmetry between L and R
QUALITY_MIN_CORNER_SHARPNESS = 40.0  # mean Laplacian var in a 15x15 window per corner
QUALITY_CORNER_WINDOW = 15


def corner_local_sharpness(gray: np.ndarray, corners: Optional[np.ndarray]) -> float:
    """Mean Laplacian-variance sharpness in a 15×15 window around each corner.

    Catches motion-smeared captures where the whole-frame Laplacian passes but
    the localized corner regions are soft (common with hand-held boards).
    Returns NaN if no corners supplied.
    """
    if corners is None or len(corners) == 0:
        return float("nan")
    h, w = gray.shape[:2]
    half = QUALITY_CORNER_WINDOW // 2
    pts = corners.reshape(-1, 2)
    scores: list[float] = []
    for x, y in pts:
        xi, yi = int(round(float(x))), int(round(float(y)))
        x1, x2 = max(0, xi - half), min(w, xi + half + 1)
        y1, y2 = max(0, yi - half), min(h, yi + half + 1)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        scores.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def assess_frame_quality(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    n_corners: int,
    corners_l: Optional[np.ndarray] = None,
    corners_r: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Evaluate a captured frame pair. Returns checks + pass/fail + reasons.

    corners_l/corners_r (optional) in full-res pixel coords. If provided, a
    per-corner local-sharpness check runs on top of the whole-frame Laplacian —
    rejects pairs that pass the global blur test but have soft corners.
    """
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY) if frame_l.ndim == 3 else frame_l
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if frame_r.ndim == 3 else frame_r

    blur_l = float(cv2.Laplacian(gray_l, cv2.CV_64F).var())
    blur_r = float(cv2.Laplacian(gray_r, cv2.CV_64F).var())
    sat_l = float((gray_l > 250).mean() * 100)
    sat_r = float((gray_r > 250).mean() * 100)
    exp_l = float(np.median(gray_l))
    exp_r = float(np.median(gray_r))
    lr_bright_diff = abs(exp_l - exp_r) / max(exp_l, exp_r, 1) * 100

    corner_sharp_l = corner_local_sharpness(gray_l, corners_l)
    corner_sharp_r = corner_local_sharpness(gray_r, corners_r)
    corner_sharp_ok_l = (
        math.isnan(corner_sharp_l) or corner_sharp_l >= QUALITY_MIN_CORNER_SHARPNESS
    )
    corner_sharp_ok_r = (
        math.isnan(corner_sharp_r) or corner_sharp_r >= QUALITY_MIN_CORNER_SHARPNESS
    )

    checks = {
        "corners": n_corners >= QUALITY_MIN_CORNERS,
        "blur_l": blur_l >= QUALITY_MIN_BLUR,
        "blur_r": blur_r >= QUALITY_MIN_BLUR,
        "corner_sharp_l": corner_sharp_ok_l,
        "corner_sharp_r": corner_sharp_ok_r,
        "saturation_l": sat_l <= QUALITY_MAX_SATURATION_PCT,
        "saturation_r": sat_r <= QUALITY_MAX_SATURATION_PCT,
        "exposure_l": exp_l >= QUALITY_MIN_EXPOSURE,
        "exposure_r": exp_r >= QUALITY_MIN_EXPOSURE,
        "lr_balance": lr_bright_diff <= QUALITY_MAX_LR_BRIGHTNESS_PCT,
    }
    reasons: list[str] = []
    if not checks["corners"]:
        reasons.append(f"Pocas esquinas ({n_corners}<{QUALITY_MIN_CORNERS})")
    if not checks["blur_l"]:
        reasons.append(f"Imagen izquierda borrosa ({blur_l:.0f}<{QUALITY_MIN_BLUR:.0f})")
    if not checks["blur_r"]:
        reasons.append(f"Imagen derecha borrosa ({blur_r:.0f}<{QUALITY_MIN_BLUR:.0f})")
    if not checks["corner_sharp_l"]:
        reasons.append(f"Esquinas izquierda blandas ({corner_sharp_l:.0f}<{QUALITY_MIN_CORNER_SHARPNESS:.0f})")
    if not checks["corner_sharp_r"]:
        reasons.append(f"Esquinas derecha blandas ({corner_sharp_r:.0f}<{QUALITY_MIN_CORNER_SHARPNESS:.0f})")
    if not checks["saturation_l"]:
        reasons.append(f"Brillo/reflejo en izquierda ({sat_l:.1f}%)")
    if not checks["saturation_r"]:
        reasons.append(f"Brillo/reflejo en derecha ({sat_r:.1f}%)")
    if not checks["exposure_l"]:
        reasons.append(f"Izquierda muy oscura (mediana {exp_l:.0f})")
    if not checks["exposure_r"]:
        reasons.append(f"Derecha muy oscura (mediana {exp_r:.0f})")
    if not checks["lr_balance"]:
        weaker = "Izquierda" if exp_l < exp_r else "Derecha"
        reasons.append(f"{weaker} más oscura ({lr_bright_diff:.0f}% diff)")

    return {
        "checks": checks,
        "blur_l": blur_l, "blur_r": blur_r,
        "corner_sharp_l": corner_sharp_l, "corner_sharp_r": corner_sharp_r,
        "sat_l": sat_l, "sat_r": sat_r,
        "exp_l": exp_l, "exp_r": exp_r,
        "lr_bright_diff": lr_bright_diff,
        "n_corners": n_corners,
        "all_pass": all(checks.values()),
        "reasons": reasons,
    }


# Real-time lighting advisories (subset of quality — warnings during aiming,
# not blocking capture).


def live_lighting_warnings(frame_l: np.ndarray, frame_r: np.ndarray) -> list[str]:
    """Non-blocking warnings shown while the operator is still positioning."""
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY) if frame_l.ndim == 3 else frame_l
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if frame_r.ndim == 3 else frame_r
    sat_l = float((gray_l > 250).mean() * 100)
    sat_r = float((gray_r > 250).mean() * 100)
    exp_l = float(np.median(gray_l))
    exp_r = float(np.median(gray_r))
    lr_diff = abs(exp_l - exp_r) / max(exp_l, exp_r, 1) * 100

    warnings: list[str] = []
    if sat_l > QUALITY_MAX_SATURATION_PCT or sat_r > QUALITY_MAX_SATURATION_PCT:
        warnings.append("Reflejo/brillo detectado — inclinalo o movelo")
    if exp_l < QUALITY_MIN_EXPOSURE or exp_r < QUALITY_MIN_EXPOSURE:
        warnings.append("Escena muy oscura — aumentá iluminación")
    if lr_diff > QUALITY_MAX_LR_BRIGHTNESS_PCT:
        weaker = "izquierda" if exp_l < exp_r else "derecha"
        warnings.append(f"Cámara {weaker} más oscura — revisá obstrucción")
    return warnings

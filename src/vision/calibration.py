"""Stereo calibration using ChArUco patterns.

Implements a pure fisheye pipeline for wide-angle (160-170°) lenses:

  1. Robust fisheye intrinsic calibration per camera, intersect kept pairs.
  2. Filter by per-view monocular reprojection error (quality > quantity).
  3. cv2.fisheye.stereoCalibrate with RECOMPUTE_EXTRINSIC + CHECK_COND.
  4. cv2.fisheye.stereoRectify + cv2.fisheye.initUndistortRectifyMap.

Compatible with OpenCV 4.8+ (contrib) which uses the refactored ArUco API.

References:
    - Zhang (2000): Flexible camera calibration technique.
    - Garrido-Jurado et al. (2014): ChArUco marker detection.
    - Hartley & Zisserman (2004): Stereo rectification.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChArUco board factory
# ---------------------------------------------------------------------------

# Default board parameters matching calib.io print (5x7, DICT_5X5)
DEFAULT_BOARD_SIZE = (7, 5)  # (columns, rows) of chessboard squares
DEFAULT_SQUARE_LENGTH = 35.0  # mm
DEFAULT_MARKER_LENGTH = 26.0  # mm
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_250


def create_charuco_board(
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
) -> cv2.aruco.CharucoBoard:
    """Create a ChArUco board for calibration.

    Args:
        board_size: (columns, rows) of chessboard squares.
        square_length: Side length of each chessboard square in mm.
        marker_length: Side length of each ArUco marker in mm.

    Returns:
        CharucoBoard instance.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        board_size, square_length, marker_length, aruco_dict
    )
    return board


def generate_board_image(
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int] = (2480, 3508),  # A4 at 300 DPI
    margin: int = 50,
) -> np.ndarray:
    """Render the ChArUco board as a printable image.

    Args:
        board: CharucoBoard instance.
        image_size: Output image (width, height) in pixels.
        margin: Border margin in pixels.

    Returns:
        Grayscale image of the board.
    """
    img = board.generateImage(image_size, marginSize=margin)
    return img


# ---------------------------------------------------------------------------
# ChArUco corner detection
# ---------------------------------------------------------------------------


def detect_charuco_corners(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ChArUco corners in a single image.

    Uses the CharucoDetector API (OpenCV 4.8+).

    Args:
        image: Grayscale or BGR image.
        board: CharucoBoard used for detection.

    Returns:
        (charuco_corners, charuco_ids) or (None, None) if detection fails.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        detector.detectBoard(gray)
    )

    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None

    return charuco_corners, charuco_ids


# ---------------------------------------------------------------------------
# Stereo calibration pipeline
# ---------------------------------------------------------------------------


def calibrate_stereo(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
) -> dict[str, np.ndarray]:
    """Run stereo calibration from ChArUco image pairs.

    Detects ChArUco corners in each image pair, calibrates each camera
    individually, then performs stereo calibration and computes
    rectification maps.

    Args:
        image_pairs: List of (left_image, right_image) tuples.
            Images can be BGR or grayscale.
        board_size: (columns, rows) of chessboard squares.
        square_length: Square side length in mm.
        marker_length: Marker side length in mm.

    Returns:
        Dict with keys:
            camera_matrix_l, dist_coeffs_l,
            camera_matrix_r, dist_coeffs_r,
            R, T, E, F,
            R1, R2, P1, P2, Q,
            map_l_x, map_l_y, map_r_x, map_r_y,
            image_size (as [w, h] array).

    Raises:
        ValueError: If fewer than 10 valid pairs are found.
    """
    board = create_charuco_board(board_size, square_length, marker_length)

    all_corners_l: list[np.ndarray] = []
    all_ids_l: list[np.ndarray] = []
    all_corners_r: list[np.ndarray] = []
    all_ids_r: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None  # (w, h)

    for idx, (img_l, img_r) in enumerate(image_pairs):
        corners_l, ids_l = detect_charuco_corners(img_l, board)
        corners_r, ids_r = detect_charuco_corners(img_r, board)

        if corners_l is None or corners_r is None:
            logger.debug("Pair %d: detection failed, skipping", idx)
            continue

        # Keep only corner IDs present in BOTH images
        common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
        if len(common_ids) < 6:
            logger.debug(
                "Pair %d: only %d common corners, skipping", idx, len(common_ids)
            )
            continue

        mask_l = np.isin(ids_l.flatten(), common_ids)
        mask_r = np.isin(ids_r.flatten(), common_ids)

        filtered_corners_l = corners_l[mask_l]
        filtered_ids_l = ids_l[mask_l]
        filtered_corners_r = corners_r[mask_r]
        filtered_ids_r = ids_r[mask_r]

        # Sort by ID so both arrays align
        order_l = np.argsort(filtered_ids_l.flatten())
        order_r = np.argsort(filtered_ids_r.flatten())

        all_corners_l.append(filtered_corners_l[order_l])
        all_ids_l.append(filtered_ids_l[order_l])
        all_corners_r.append(filtered_corners_r[order_r])
        all_ids_r.append(filtered_ids_r[order_r])

        if image_size is None:
            h, w = img_l.shape[:2]
            image_size = (w, h)

    valid_pairs = len(all_corners_l)
    logger.info("Valid calibration pairs: %d / %d", valid_pairs, len(image_pairs))

    if valid_pairs < 15:
        raise ValueError(
            f"Need at least 15 valid pairs for fisheye stereo calibration, "
            f"got {valid_pairs}. Capture more images with the ChArUco "
            f"board visible in both cameras."
        )

    obj_points_per_image = _build_object_points(all_ids_l, board)

    result = _calibrate_fisheye(
        obj_points_per_image, all_corners_l, all_corners_r, image_size
    )

    result["image_size"] = np.array(list(image_size))
    return result



def _fisheye_calibrate_robust(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    label: str,
) -> tuple[float, np.ndarray, np.ndarray, list[int]]:
    """Iteratively calibrate fisheye, removing ill-conditioned pairs.

    Returns (rms, K, D, kept_indices).
    """
    import re

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    indices = list(range(len(obj_points)))

    while len(indices) >= 10:
        cur_obj = [obj_points[i] for i in indices]
        cur_img = [img_points[i] for i in indices]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        try:
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                cur_obj, cur_img, image_size, K, D,
                flags=flags, criteria=criteria,
            )
            logger.info("%s RMS (fisheye): %.4f (%d pairs)", label, rms, len(indices))
            return rms, K, D, indices
        except cv2.error as e:
            msg = str(e)
            # Try to extract bad array index from error message
            m = re.search(r'array (\d+)', msg)
            if m:
                bad_local = int(m.group(1))
                bad_global = indices[bad_local]
                logger.debug("%s: removing pair %d (local %d): %s", label, bad_global, bad_local, msg.split('\n')[0])
                indices.pop(bad_local)
            else:
                # Can't identify which pair — remove last and retry
                removed = indices.pop()
                logger.debug("%s: removing pair %d (unknown cause): %s", label, removed, msg.split('\n')[0])

    raise ValueError(f"{label}: fewer than 10 pairs remaining after filtering")


def _calibrate_fisheye(
    obj_points_per_image: list[np.ndarray],
    all_corners_l: list[np.ndarray],
    all_corners_r: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Pure fisheye stereo calibration pipeline for 160°+ lenses.

    Single geometric model end-to-end (no pinhole mixing):
    1. Robust fisheye intrinsic calibration per camera.
    2. Intersect kept pairs, re-calibrate on common set.
    3. Filter by per-view monocular reprojection error.
    4. Truncate to uniform point count for fisheye.stereoCalibrate.
    5. fisheye.stereoCalibrate with RECOMPUTE_EXTRINSIC + CHECK_COND.
    6. fisheye.stereoRectify + fisheye.initUndistortRectifyMap.
    """
    # Fisheye needs shape (1, N, 3) and (1, N, 2)
    obj_points = [pts.reshape(1, -1, 3) for pts in obj_points_per_image]
    img_points_l = [c.reshape(1, -1, 2) for c in all_corners_l]
    img_points_r = [c.reshape(1, -1, 2) for c in all_corners_r]

    # Step 1: Initial robust calibration to find ill-conditioned pairs
    _, _, _, keep_l = _fisheye_calibrate_robust(
        obj_points, img_points_l, image_size, "Left (initial)",
    )
    _, _, _, keep_r = _fisheye_calibrate_robust(
        obj_points, img_points_r, image_size, "Right (initial)",
    )

    # Step 2: Intersect and re-calibrate on common pairs
    keep = sorted(set(keep_l) & set(keep_r))
    logger.info("Pairs valid for both cameras: %d", len(keep))
    obj_points = [obj_points[i] for i in keep]
    img_points_l = [img_points_l[i] for i in keep]
    img_points_r = [img_points_r[i] for i in keep]

    rms_l, K_l, D_l, keep2_l = _fisheye_calibrate_robust(
        obj_points, img_points_l, image_size, "Left (final)",
    )
    rms_r, K_r, D_r, keep2_r = _fisheye_calibrate_robust(
        obj_points, img_points_r, image_size, "Right (final)",
    )

    # Intersect again after re-calibration
    keep2 = sorted(set(keep2_l) & set(keep2_r))
    if len(keep2) < len(obj_points):
        logger.info("Re-calibration removed %d more pairs", len(obj_points) - len(keep2))
        obj_points = [obj_points[i] for i in keep2]
        img_points_l = [img_points_l[i] for i in keep2]
        img_points_r = [img_points_r[i] for i in keep2]

    # Step 3: Filter by per-view monocular reprojection error.
    # This is critical for stereo — views with high monocular error
    # will poison the stereo R/T estimation.
    flags_proj = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # Get rvecs/tvecs for reprojection error computation
    _, _, _, rvecs_l, tvecs_l = cv2.fisheye.calibrate(
        obj_points, img_points_l, image_size,
        K_l.copy(), D_l.copy(), flags=flags_proj, criteria=criteria,
    )
    _, _, _, rvecs_r, tvecs_r = cv2.fisheye.calibrate(
        obj_points, img_points_r, image_size,
        K_r.copy(), D_r.copy(), flags=flags_proj, criteria=criteria,
    )

    # Compute per-view reprojection error
    errors_l = []
    errors_r = []
    for i in range(len(obj_points)):
        proj_l, _ = cv2.fisheye.projectPoints(
            obj_points[i], rvecs_l[i], tvecs_l[i], K_l, D_l)
        err_l = cv2.norm(img_points_l[i], proj_l, cv2.NORM_L2) / len(proj_l)
        errors_l.append(err_l)

        proj_r, _ = cv2.fisheye.projectPoints(
            obj_points[i], rvecs_r[i], tvecs_r[i], K_r, D_r)
        err_r = cv2.norm(img_points_r[i], proj_r, cv2.NORM_L2) / len(proj_r)
        errors_r.append(err_r)

    errors_l = np.array(errors_l)
    errors_r = np.array(errors_r)
    errors_max = np.maximum(errors_l, errors_r)

    # Filter: keep pairs below threshold (start at 1.0, relax if needed)
    threshold = 1.0
    while threshold <= 3.0:
        good = [i for i in range(len(obj_points)) if errors_max[i] < threshold]
        if len(good) >= 20:
            break
        threshold += 0.5

    removed = len(obj_points) - len(good)
    logger.info(
        "Reprojection filter: keeping %d/%d pairs (threshold=%.1f, "
        "median_err_l=%.3f, median_err_r=%.3f)",
        len(good), len(obj_points), threshold,
        float(np.median(errors_l)), float(np.median(errors_r)),
    )
    obj_points = [obj_points[i] for i in good]
    img_points_l = [img_points_l[i] for i in good]
    img_points_r = [img_points_r[i] for i in good]

    # Re-calibrate intrinsics on the filtered set
    if removed > 0:
        rms_l, K_l, D_l, _ = _fisheye_calibrate_robust(
            obj_points, img_points_l, image_size, "Left (filtered)",
        )
        rms_r, K_r, D_r, _ = _fisheye_calibrate_robust(
            obj_points, img_points_r, image_size, "Right (filtered)",
        )

    # Step 4: Truncate to uniform point count for fisheye.stereoCalibrate.
    # Filter out pairs with few points first (keep >= median/2 or >= 12).
    point_counts = [o.shape[1] for o in obj_points]
    median_pts = int(np.median(point_counts))
    min_acceptable = max(12, median_pts // 2)
    good_pts = [i for i, n in enumerate(point_counts) if n >= min_acceptable]
    if len(good_pts) < 15:
        min_acceptable = min(point_counts)
        good_pts = list(range(len(obj_points)))
    logger.info("Point count filter: %d/%d pairs have >= %d points",
                len(good_pts), len(obj_points), min_acceptable)
    obj_points = [obj_points[i] for i in good_pts]
    img_points_l = [img_points_l[i] for i in good_pts]
    img_points_r = [img_points_r[i] for i in good_pts]

    min_pts = min(o.shape[1] for o in obj_points)
    logger.info("Truncating %d pairs to %d points each", len(obj_points), min_pts)
    obj_points = [o[:, :min_pts, :] for o in obj_points]
    img_points_l = [p[:, :min_pts, :] for p in img_points_l]
    img_points_r = [p[:, :min_pts, :] for p in img_points_r]

    # Step 5: Pure fisheye stereo calibration
    stereo_flags = (
        cv2.fisheye.CALIB_FIX_INTRINSIC
        | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    retval = cv2.fisheye.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r, image_size,
        flags=stereo_flags, criteria=criteria,
    )
    rms_stereo = retval[0]
    R = retval[5]
    T = retval[6]
    logger.info("Stereo RMS (fisheye): %.4f (%d pairs, %d pts each)",
                rms_stereo, len(obj_points), min_pts)

    # Step 6: Fisheye stereo rectification
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0,
    )

    # Rectification maps: compose fisheye undistortion + rectification
    map_l_x, map_l_y = cv2.fisheye.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_32FC1
    )
    map_r_x, map_r_y = cv2.fisheye.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_32FC1
    )

    return {
        "camera_matrix_l": K_l, "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r, "dist_coeffs_r": D_r,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_l_x": map_l_x, "map_l_y": map_l_y,
        "map_r_x": map_r_x, "map_r_y": map_r_y,
    }



def _build_object_points(
    all_ids: list[np.ndarray],
    board: cv2.aruco.CharucoBoard,
) -> list[np.ndarray]:
    """Build 3D object points for each image from detected corner IDs.

    Each ChArUco corner has a known 3D position on the board plane (Z=0).
    """
    board_corners = board.getChessboardCorners()  # (N, 3) float32

    obj_points = []
    for ids in all_ids:
        pts = board_corners[ids.flatten()]
        obj_points.append(pts.astype(np.float32))

    return obj_points


# ---------------------------------------------------------------------------
# Rectification helpers
# ---------------------------------------------------------------------------


def rectify_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rectification maps to a stereo image pair.

    Args:
        img_l: Left image (BGR or grayscale).
        img_r: Right image (BGR or grayscale).
        calibration: Dict from calibrate_stereo() or load_calibration().

    Returns:
        (rectified_left, rectified_right).
    """
    rect_l = cv2.remap(
        img_l,
        calibration["map_l_x"],
        calibration["map_l_y"],
        cv2.INTER_LINEAR,
    )
    rect_r = cv2.remap(
        img_r,
        calibration["map_r_x"],
        calibration["map_r_y"],
        cv2.INTER_LINEAR,
    )
    return rect_l, rect_r


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_calibration(path: str) -> dict[str, np.ndarray]:
    """Load calibration parameters from .npz file."""
    cal_path = Path(path)
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = dict(np.load(cal_path))
    logger.info(
        "Calibration loaded",
        extra={"path": path, "keys": list(data.keys())},
    )
    return data


def save_calibration(params: dict[str, np.ndarray], path: str) -> None:
    """Save calibration parameters to .npz file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)
    logger.info("Calibration saved", extra={"path": path})

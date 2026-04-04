"""Stereo calibration using ChArUco patterns.

Implements the full pipeline: ChArUco detection → individual camera
calibration → stereo calibration → rectification map generation.

Compatible with OpenCV 4.8+ (contrib) which uses the refactored ArUco API.
Uses the OpenCV fisheye model (cv2.fisheye) for the OV5647 160° lenses,
which handles wide-angle distortion much better than the pinhole model.

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

    if charuco_ids is None or len(charuco_ids) < 8:
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
    use_fisheye: bool = True,
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
        if len(common_ids) < 8:
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

    if valid_pairs < 10:
        raise ValueError(
            f"Need at least 10 valid pairs for stereo calibration, "
            f"got {valid_pairs}. Capture more images with the ChArUco "
            f"board visible in both cameras."
        )

    obj_points_per_image = _build_object_points(all_ids_l, board)

    if use_fisheye:
        try:
            result = _calibrate_fisheye(
                obj_points_per_image, all_corners_l, all_corners_r, image_size
            )
        except ValueError as e:
            logger.warning("Fisheye calibration failed: %s", e)
            logger.warning("Falling back to pinhole + rational model")
            result = _calibrate_pinhole(
                obj_points_per_image, all_corners_l, all_corners_r, image_size
            )
    else:
        result = _calibrate_pinhole(
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
    """Calibrate using cv2.fisheye model (for 160°+ lenses)."""
    # Fisheye needs shape (1, N, 3) and (1, N, 2)
    obj_points = [pts.reshape(1, -1, 3) for pts in obj_points_per_image]
    img_points_l = [c.reshape(1, -1, 2) for c in all_corners_l]
    img_points_r = [c.reshape(1, -1, 2) for c in all_corners_r]

    rms_l, K_l, D_l, keep_l = _fisheye_calibrate_robust(
        obj_points, img_points_l, image_size, "Left",
    )
    rms_r, K_r, D_r, keep_r = _fisheye_calibrate_robust(
        obj_points, img_points_r, image_size, "Right",
    )

    # Use only pairs that passed both cameras
    keep = sorted(set(keep_l) & set(keep_r))
    logger.info("Pairs valid for both cameras: %d", len(keep))
    obj_points = [obj_points[i] for i in keep]
    img_points_l = [img_points_l[i] for i in keep]
    img_points_r = [img_points_r[i] for i in keep]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms_stereo, _, _, _, _, R, T = cv2.fisheye.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l.copy(), D_l.copy(), K_r.copy(), D_r.copy(),
        image_size,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
        criteria=criteria,
    )
    logger.info("Stereo RMS (fisheye): %.4f", rms_stereo)

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.5,
        fov_scale=1.0,
    )

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
        "E": np.zeros((3, 3)), "F": np.zeros((3, 3)),
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_l_x": map_l_x, "map_l_y": map_l_y,
        "map_r_x": map_r_x, "map_r_y": map_r_y,
    }


def _calibrate_pinhole(
    obj_points_per_image: list[np.ndarray],
    all_corners_l: list[np.ndarray],
    all_corners_r: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Calibrate using standard pinhole model with rational distortion."""
    calib_flags = cv2.CALIB_RATIONAL_MODEL

    rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        obj_points_per_image, all_corners_l, image_size, None, None,
        flags=calib_flags,
    )
    logger.info("Left camera RMS (pinhole): %.4f", rms_l)

    rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        obj_points_per_image, all_corners_r, image_size, None, None,
        flags=calib_flags,
    )
    logger.info("Right camera RMS (pinhole): %.4f", rms_r)

    stereo_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
    rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points_per_image, all_corners_l, all_corners_r,
        K_l, D_l, K_r, D_r, image_size,
        flags=stereo_flags,
    )
    logger.info("Stereo RMS (pinhole): %.4f", rms_stereo)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T, alpha=1.0,
    )

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_32FC1
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_32FC1
    )

    return {
        "camera_matrix_l": K_l, "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r, "dist_coeffs_r": D_r,
        "R": R, "T": T, "E": E, "F": F,
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

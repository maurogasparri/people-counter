"""Stereo calibration using ChArUco patterns.

Supports two modes:
  - Pinhole: Crops center of image, uses standard cv2.calibrateCamera
    and cv2.stereoCalibrate. Robust for any FOV.
  - Fisheye: Full cv2.fisheye pipeline for lenses where full FOV is needed.

Compatible with OpenCV 4.8+ (contrib) which uses the refactored ArUco API.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChArUco board factory
# ---------------------------------------------------------------------------

DEFAULT_BOARD_SIZE = (7, 5)  # (columns, rows) of chessboard squares
DEFAULT_SQUARE_LENGTH = 35.0  # mm
DEFAULT_MARKER_LENGTH = 26.0  # mm
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_250


def create_charuco_board(
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
) -> cv2.aruco.CharucoBoard:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        board_size, square_length, marker_length, aruco_dict
    )
    return board


def generate_board_image(
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int] = (2480, 3508),
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
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
# Shared helpers
# ---------------------------------------------------------------------------


def _crop_center(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return image[y0:y0 + new_h, x0:x0 + new_w]


def _detect_all_pairs(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
    crop_ratio: float = 1.0,
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
        if crop_ratio < 1.0:
            img_l = _crop_center(img_l, crop_ratio)
            img_r = _crop_center(img_r, crop_ratio)

        corners_l, ids_l = detect_charuco_corners(img_l, board)
        corners_r, ids_r = detect_charuco_corners(img_r, board)

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
# Pinhole stereo calibration
# ---------------------------------------------------------------------------


def calibrate_stereo(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
    crop_ratio: float = 0.6,
) -> dict[str, np.ndarray]:
    """Pinhole stereo calibration with center crop for wide-angle lenses."""
    board = create_charuco_board(board_size, square_length, marker_length)
    all_obj, all_corners_l, all_corners_r, image_size = _detect_all_pairs(
        image_pairs, board, crop_ratio=crop_ratio,
    )

    valid_pairs = len(all_obj)
    logger.info("Valid pairs (crop %.0f%%): %d / %d",
                crop_ratio * 100, valid_pairs, len(image_pairs))

    if valid_pairs < 15:
        raise ValueError(
            f"Need at least 15 valid pairs, got {valid_pairs}."
        )

    calib_flags = cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        all_obj, all_corners_l, image_size, None, None,
        flags=calib_flags, criteria=criteria,
    )
    logger.info("Left RMS: %.4f (pinhole, %d pairs)", rms_l, valid_pairs)

    rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        all_obj, all_corners_r, image_size, None, None,
        flags=calib_flags, criteria=criteria,
    )
    logger.info("Right RMS: %.4f (pinhole, %d pairs)", rms_r, valid_pairs)

    stereo_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
    rms_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        all_obj, all_corners_l, all_corners_r,
        K_l, D_l, K_r, D_r, image_size,
        flags=stereo_flags, criteria=criteria,
    )
    logger.info("Stereo RMS: %.4f (%d pairs)", rms_stereo, valid_pairs)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_32FC1,
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_32FC1,
    )

    logger.info("Focal: fx=%.1f fy=%.1f | Baseline: %.1f mm",
                K_l[0, 0], K_l[1, 1], abs(T[0, 0]))

    return {
        "camera_matrix_l": K_l, "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r, "dist_coeffs_r": D_r,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_l_x": map_l_x, "map_l_y": map_l_y,
        "map_r_x": map_r_x, "map_r_y": map_r_y,
        "image_size": np.array(list(image_size)),
        "crop_ratio": np.array([crop_ratio]),
        "mode": np.array([0]),  # 0=pinhole
    }


# ---------------------------------------------------------------------------
# Fisheye stereo calibration
# ---------------------------------------------------------------------------


def _fisheye_calibrate_robust(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    label: str,
) -> tuple[float, np.ndarray, np.ndarray, list[int]]:
    """Fisheye calibration with iterative removal of bad pairs."""
    flags = (
        cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    w, h = image_size
    f_init = (w / 2) / (80 * np.pi / 180)
    K_init = np.array([
        [f_init, 0, w / 2.0],
        [0, f_init, h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)

    indices = list(range(len(obj_points)))

    while len(indices) >= 6:
        cur_obj = [obj_points[i] for i in indices]
        cur_img = [img_points[i] for i in indices]
        K = K_init.copy()
        D = np.zeros((4, 1))
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                cur_obj, cur_img, image_size, K, D,
                flags=flags, criteria=criteria,
            )
            logger.info("%s RMS: %.4f (%d pairs)", label, rms, len(indices))
            return rms, K, D, indices
        except cv2.error as e:
            msg = str(e)
            m = re.search(r'array (\d+)', msg)
            if m:
                bad_local = int(m.group(1))
                logger.warning("%s: removing pair %d", label, indices[bad_local])
                indices.pop(bad_local)
            else:
                # Probe to find the bad pair
                found = False
                for local_idx in range(len(indices)):
                    try:
                        Kt = K_init.copy()
                        Dt = np.zeros((4, 1))
                        cv2.fisheye.calibrate(
                            [obj_points[indices[local_idx]]],
                            [img_points[indices[local_idx]]],
                            image_size, Kt, Dt,
                            flags=flags & ~cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
                            criteria=criteria,
                        )
                    except cv2.error:
                        logger.warning("%s: removing pair %d (probed)", label, indices[local_idx])
                        indices.pop(local_idx)
                        found = True
                        break
                if not found:
                    removed = indices.pop()
                    logger.warning("%s: removing pair %d (fallback)", label, removed)

    raise ValueError(f"{label}: fewer than 6 pairs remaining")


def calibrate_stereo_fisheye(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
) -> dict[str, np.ndarray]:
    """Pure fisheye stereo calibration for 160°+ lenses."""
    board = create_charuco_board(board_size, square_length, marker_length)
    all_obj, all_corners_l, all_corners_r, image_size = _detect_all_pairs(
        image_pairs, board, crop_ratio=1.0,
    )

    valid_pairs = len(all_obj)
    logger.info("Valid pairs: %d / %d", valid_pairs, len(image_pairs))

    if valid_pairs < 15:
        raise ValueError(
            f"Need at least 15 valid pairs, got {valid_pairs}."
        )

    # Fisheye needs (1, N, 2/3) shape
    obj_f = [o.reshape(1, -1, 3) for o in all_obj]
    img_l_f = [c.reshape(1, -1, 2) for c in all_corners_l]
    img_r_f = [c.reshape(1, -1, 2) for c in all_corners_r]

    # Robust per-camera calibration
    rms_l, K_l, D_l, keep_l = _fisheye_calibrate_robust(
        obj_f, img_l_f, image_size, "Left",
    )
    rms_r, K_r, D_r, keep_r = _fisheye_calibrate_robust(
        obj_f, img_r_f, image_size, "Right",
    )

    # Intersect surviving pairs
    keep = sorted(set(keep_l) & set(keep_r))
    logger.info("Common pairs after robust calibration: %d", len(keep))

    if len(keep) < 10:
        raise ValueError(f"Only {len(keep)} common pairs — need at least 10.")

    obj_f = [obj_f[i] for i in keep]
    img_l_f = [img_l_f[i] for i in keep]
    img_r_f = [img_r_f[i] for i in keep]

    # Truncate to uniform point count (required by fisheye.stereoCalibrate)
    min_pts = min(o.shape[1] for o in obj_f)
    obj_f = [o[:, :min_pts, :] for o in obj_f]
    img_l_f = [p[:, :min_pts, :] for p in img_l_f]
    img_r_f = [p[:, :min_pts, :] for p in img_r_f]
    logger.info("Stereo input: %d pairs × %d points", len(obj_f), min_pts)

    # Stereo calibration
    stereo_flags = (
        cv2.fisheye.CALIB_FIX_INTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    retval = cv2.fisheye.stereoCalibrate(
        obj_f, img_l_f, img_r_f,
        K_l, D_l, K_r, D_r, image_size,
        flags=stereo_flags, criteria=criteria,
    )
    rms_stereo = retval[0]
    R = retval[5]
    T = retval[6]
    logger.info("Stereo RMS: %.4f (%d pairs)", rms_stereo, len(obj_f))

    # Rectification
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0,
    )
    map_l_x, map_l_y = cv2.fisheye.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_32FC1,
    )
    map_r_x, map_r_y = cv2.fisheye.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_32FC1,
    )

    logger.info("Focal: fx=%.1f fy=%.1f | Baseline: %.1f mm",
                K_l[0, 0], K_l[1, 1], abs(T[0, 0]))

    return {
        "camera_matrix_l": K_l, "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r, "dist_coeffs_r": D_r,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_l_x": map_l_x, "map_l_y": map_l_y,
        "map_r_x": map_r_x, "map_r_y": map_r_y,
        "image_size": np.array(list(image_size)),
        "crop_ratio": np.array([1.0]),
        "mode": np.array([1]),  # 1=fisheye
    }


# ---------------------------------------------------------------------------
# Rectification helpers
# ---------------------------------------------------------------------------


def rectify_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    crop_ratio = float(calibration.get("crop_ratio", [1.0])[0])
    if crop_ratio < 1.0:
        img_l = _crop_center(img_l, crop_ratio)
        img_r = _crop_center(img_r, crop_ratio)

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

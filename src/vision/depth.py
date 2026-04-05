"""Disparity map computation from rectified stereo pairs.

Uses Semi-Global Block Matching (SGBM) as described by Hirschmuller (2008),
with parameters tuned for ceiling-mounted OV5647 stereo pair at ~2-4m range
and 14cm baseline.

The depth (Z) at each pixel is: Z = f * B / disparity
where f = focal length in pixels, B = baseline in mm.
"""

import cv2
import numpy as np


# SGBM parameters for OV5647 160-170° fisheye, 14cm baseline, 2-4m range.
# Disparity range: Z=4m → ~35px, Z=1.5m → ~95px (depends on rectified fx).
DEFAULT_NUM_DISPARITIES = 128  # Covers ~1.5-6m range
DEFAULT_BLOCK_SIZE = 9  # Larger block for robust matching on wide-angle images
DEFAULT_P1_FACTOR = 12  # Smoothness penalty for ±1 disparity change
DEFAULT_P2_FACTOR = 96  # Smoothness penalty for large discontinuities (8× P1)
DEFAULT_DISP12_MAX_DIFF = 2  # Allow ±2px left-right mismatch (fisheye residual)
DEFAULT_UNIQUENESS_RATIO = 5  # Lower for IR-filter cameras with good contrast
DEFAULT_SPECKLE_WINDOW_SIZE = 150  # Filter small noise blobs
DEFAULT_SPECKLE_RANGE = 16  # Max disparity variation within a speckle
DEFAULT_PRE_FILTER_CAP = 63
DEFAULT_MIN_DISPARITY = 0


def create_sgbm(
    num_disparities: int = DEFAULT_NUM_DISPARITIES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    p1_factor: int = DEFAULT_P1_FACTOR,
    p2_factor: int = DEFAULT_P2_FACTOR,
    disp12_max_diff: int = DEFAULT_DISP12_MAX_DIFF,
    uniqueness_ratio: int = DEFAULT_UNIQUENESS_RATIO,
    speckle_window_size: int = DEFAULT_SPECKLE_WINDOW_SIZE,
    speckle_range: int = DEFAULT_SPECKLE_RANGE,
    pre_filter_cap: int = DEFAULT_PRE_FILTER_CAP,
    min_disparity: int = DEFAULT_MIN_DISPARITY,
) -> cv2.StereoSGBM:
    """Create a configured SGBM matcher.

    Args:
        num_disparities: Maximum disparity minus minimum disparity.
            Must be divisible by 16.
        block_size: Matched block size. Must be odd.
        p1_factor: Multiplier for P1 smoothness penalty.
        p2_factor: Multiplier for P2 smoothness penalty (must be > p1_factor).
        disp12_max_diff: Max allowed difference in left-right disparity check.
        uniqueness_ratio: Margin (%) by which best match must beat second best.
        speckle_window_size: Max area of connected component to filter.
        speckle_range: Max disparity variation within a connected component.
        pre_filter_cap: Truncation value for pre-filtered image pixels.
        min_disparity: Minimum disparity value (usually 0).

    Returns:
        Configured StereoSGBM instance.
    """
    channels = 1
    p1 = p1_factor * channels * block_size * block_size
    p2 = p2_factor * channels * block_size * block_size

    sgbm = cv2.StereoSGBM.create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm


def _to_gray(image: np.ndarray, use_green_channel: bool = False) -> np.ndarray:
    """Convert image to grayscale.

    Args:
        image: BGR or grayscale image.
        use_green_channel: If True, extract green channel only (better for
            NoIR cameras where IR contaminates red and blue channels).
    """
    if len(image.shape) != 3:
        return image
    if use_green_channel:
        return image[:, :, 1]  # Green in BGR
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_disparity(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    num_disparities: int = DEFAULT_NUM_DISPARITIES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sgbm: cv2.StereoSGBM | None = None,
    use_wls_filter: bool = False,
    use_green_channel: bool = False,
    use_clahe: bool = True,
    downscale: int = 1,
) -> np.ndarray:
    """Compute disparity map using SGBM.

    Calibration is resolution-independent (K, D don't change), so we can
    match at lower resolution for speed and noise reduction, then upscale
    the disparity map. The disparity values are scaled back to correspond
    to the original resolution.

    Args:
        left_rect: Left rectified image (BGR or grayscale).
        right_rect: Right rectified image (BGR or grayscale).
        num_disparities: Max disparity range. Ignored if sgbm is provided.
        block_size: Block size for matching. Ignored if sgbm is provided.
        sgbm: Pre-created SGBM matcher. If None, one is created.
        use_wls_filter: Apply WLS filter (default False — too aggressive
            for NoIR cameras with weak edges).
        use_green_channel: Use green channel only (for NoIR cameras).
        use_clahe: Apply CLAHE contrast enhancement before matching.
        downscale: Factor to reduce resolution before matching (1=full,
            2=half, 4=quarter). Disparity is upscaled back and values
            are multiplied by the factor so depth calculations remain
            correct. Higher = faster but less detail.

    Returns:
        Disparity map as float32 in pixels (at original resolution).
        Invalid pixels are -1.0.
    """
    gray_l = _to_gray(left_rect, use_green_channel)
    gray_r = _to_gray(right_rect, use_green_channel)

    if downscale > 1:
        gray_l = cv2.resize(
            gray_l, (gray_l.shape[1] // downscale, gray_l.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )
        gray_r = cv2.resize(
            gray_r, (gray_r.shape[1] // downscale, gray_r.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    if sgbm is None:
        # Scale numDisparities down for reduced resolution
        nd = max(16, (num_disparities // downscale // 16) * 16) if downscale > 1 else num_disparities
        sgbm = create_sgbm(num_disparities=nd, block_size=block_size)

    raw_disp_l = sgbm.compute(gray_l, gray_r)

    if use_wls_filter:
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        raw_disp_r = right_matcher.compute(gray_r, gray_l)

        wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)
        wls.setLambda(4000.0)
        wls.setSigmaColor(1.0)

        filtered = wls.filter(raw_disp_l, gray_l, disparity_map_right=raw_disp_r)
        disparity = filtered.astype(np.float32) / 16.0
    else:
        disparity = raw_disp_l.astype(np.float32) / 16.0

    disparity[disparity < 0] = -1.0

    if downscale > 1:
        # Upscale disparity to original resolution.
        # Disparity values scale linearly with resolution, so multiply
        # by the downscale factor: Z = fx * B / d, and both fx and d
        # scale together, but we need d at original resolution.
        orig_h, orig_w = left_rect.shape[:2]
        valid_mask = disparity > 0
        disparity = cv2.resize(
            disparity, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
        )
        upscaled_mask = cv2.resize(
            valid_mask.astype(np.uint8), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        disparity[upscaled_mask] *= downscale
        disparity[~upscaled_mask] = -1.0

    return disparity


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_mm: float,
    min_depth_mm: float = 500.0,
    max_depth_mm: float = 10000.0,
) -> np.ndarray:
    """Convert disparity map to depth map in millimeters.

    Z = f * B / d

    Args:
        disparity: Disparity map from compute_disparity() (float32, pixels).
        focal_length_px: Focal length in pixels (from calibration P1[0,0]).
        baseline_mm: Stereo baseline in mm.
        min_depth_mm: Minimum valid depth.
        max_depth_mm: Maximum valid depth.

    Returns:
        Depth map in mm as float32. Invalid pixels are 0.0.
    """
    depth = np.zeros_like(disparity)

    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_mm) / disparity[valid]

    out_of_range = (depth < min_depth_mm) | (depth > max_depth_mm)
    depth[out_of_range & valid] = 0.0
    depth[~valid] = 0.0

    return depth


def depth_at_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    percentile: float = 50.0,
) -> float:
    """Estimate depth of a detected person from their bounding box.

    Uses the median (or specified percentile) of valid depth values
    within the center 50% of the bbox.

    Args:
        depth_map: Depth map in mm from disparity_to_depth().
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        percentile: Percentile of depth values to use (50 = median).

    Returns:
        Estimated depth in mm. Returns 0.0 if no valid pixels in ROI.
    """
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    hw = (x2 - x1) // 4
    hh = (y2 - y1) // 4

    roi_x1 = max(0, cx - hw)
    roi_y1 = max(0, cy - hh)
    roi_x2 = min(depth_map.shape[1], cx + hw)
    roi_y2 = min(depth_map.shape[0], cy + hh)

    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    valid = roi[roi > 0]

    if len(valid) == 0:
        return 0.0

    return float(np.percentile(valid, percentile))

"""Disparity map computation from rectified stereo pairs.

Uses Semi-Global Block Matching (SGBM) as described by Hirschmuller (2008),
with parameters tuned for ceiling-mounted OV5647 stereo pair at ~2.5-4m range
and 14cm baseline.

The depth (Z) at each pixel is: Z = f * B / disparity
where f = focal length in pixels, B = baseline in mm.
"""

import cv2
import numpy as np


# Default SGBM parameters tuned for OV5647 160deg, 14cm baseline, 2.5-4m range.
# These are starting points; real tuning happens during PoC with actual frames.
DEFAULT_NUM_DISPARITIES = 128  # Must be divisible by 16
DEFAULT_BLOCK_SIZE = 5  # Odd number, 3-11
DEFAULT_P1_FACTOR = 8  # P1 = P1_FACTOR * channels * block_size^2
DEFAULT_P2_FACTOR = 32  # P2 = P2_FACTOR * channels * block_size^2
DEFAULT_DISP12_MAX_DIFF = 1
DEFAULT_UNIQUENESS_RATIO = 10
DEFAULT_SPECKLE_WINDOW_SIZE = 100
DEFAULT_SPECKLE_RANGE = 32
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
        block_size: Matched block size. Must be odd, in range [1, 11].
        p1_factor: Multiplier for P1 smoothness penalty.
        p2_factor: Multiplier for P2 smoothness penalty (must be > p1_factor).
        disp12_max_diff: Max allowed difference in left-right disparity check.
            Set to -1 to disable.
        uniqueness_ratio: Margin (%) by which best match must beat second best.
        speckle_window_size: Max area of connected component to filter.
        speckle_range: Max disparity variation within a connected component.
        pre_filter_cap: Truncation value for pre-filtered image pixels.
        min_disparity: Minimum disparity value (usually 0).

    Returns:
        Configured StereoSGBM instance.
    """
    # P1 and P2 are computed relative to block size and assumed grayscale
    channels = 1  # We convert to grayscale before matching
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


def compute_disparity(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    num_disparities: int = DEFAULT_NUM_DISPARITIES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sgbm: cv2.StereoSGBM | None = None,
    use_wls_filter: bool = True,
) -> np.ndarray:
    """Compute disparity map using SGBM with optional WLS filtering.

    Uses left+right matchers and a Weighted Least Squares (WLS) filter
    to reduce noise and fill holes. This significantly improves depth
    map quality, especially with fisheye lenses.

    Args:
        left_rect: Left rectified image (BGR or grayscale).
        right_rect: Right rectified image (BGR or grayscale).
        num_disparities: Max disparity range. Ignored if sgbm is provided.
        block_size: Block size for matching. Ignored if sgbm is provided.
        sgbm: Pre-created SGBM matcher. If None, one is created with
            the given num_disparities and block_size.
        use_wls_filter: Apply WLS filter with right matcher (default True).

    Returns:
        Disparity map as float32 in pixels. Invalid pixels are -1.0.
        Raw SGBM output is in fixed-point (Q4 = value/16), this function
        converts to true pixel disparity.
    """
    # Convert to grayscale if needed
    if len(left_rect.shape) == 3:
        gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    else:
        gray_l = left_rect

    if len(right_rect.shape) == 3:
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    else:
        gray_r = right_rect

    if sgbm is None:
        sgbm = create_sgbm(num_disparities=num_disparities, block_size=block_size)

    # Left disparity
    raw_disp_l = sgbm.compute(gray_l, gray_r)

    if use_wls_filter:
        # Right matcher + WLS filter for cleaner disparity
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        raw_disp_r = right_matcher.compute(gray_r, gray_l)

        wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)

        filtered = wls.filter(raw_disp_l, gray_l, disparity_map_right=raw_disp_r)
        disparity = filtered.astype(np.float32) / 16.0
    else:
        disparity = raw_disp_l.astype(np.float32) / 16.0

    # Mark invalid pixels
    disparity[disparity < 0] = -1.0

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
        baseline_mm: Stereo baseline in mm (140mm for our rig).
        min_depth_mm: Minimum valid depth. Closer points are clipped.
        max_depth_mm: Maximum valid depth. Farther points are clipped.

    Returns:
        Depth map in mm as float32. Invalid pixels are 0.0.
    """
    depth = np.zeros_like(disparity)

    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_mm) / disparity[valid]

    # Clip to valid range
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
    within the lower-center region of the bbox, which corresponds
    to the person's head/shoulders in a ceiling-mounted camera.

    Args:
        depth_map: Depth map in mm from disparity_to_depth().
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        percentile: Percentile of depth values to use (50 = median).

    Returns:
        Estimated depth in mm. Returns 0.0 if no valid pixels in ROI.
    """
    x1, y1, x2, y2 = bbox

    # Use center 50% of the bbox to avoid edges
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

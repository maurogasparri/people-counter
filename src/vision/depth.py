"""Disparity map computation from rectified stereo pairs.

Uses Semi-Global Block Matching (SGBM) as described by Hirschmuller (2008),
with parameters tuned for ceiling-mounted Arducam IMX708 stereo pair at
~1.3-6m range and 14cm baseline.

The depth (Z) at each pixel is: Z = f * B / disparity
where f = focal length in pixels, B = baseline in mm.

Focal length estimate for IMX708 120° HFOV at full resolution:
  f_px ≈ 4608 / (2 * tan(60°)) ≈ 1330 px
  disparity at 3m  = 1330 * 140 / 3000 ≈ 62 px
  disparity at 1.3m = 1330 * 140 / 1300 ≈ 143 px
"""

from typing import Optional

import cv2
import numpy as np


# SGBM parameters for Arducam IMX708 120° HFOV, 14cm baseline, 1.3-6m range.
DEFAULT_NUM_DISPARITIES = 192  # Covers disparity range up to ~143px at 1.3m
DEFAULT_BLOCK_SIZE = 9  # Robust matching on wide-angle images
DEFAULT_P1_FACTOR = 12  # Smoothness penalty for ±1 disparity change
DEFAULT_P2_FACTOR = 96  # Smoothness penalty for large discontinuities (8× P1)
DEFAULT_DISP12_MAX_DIFF = 2  # Allow ±2px left-right mismatch
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
        use_green_channel: If True, extract green channel only.
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
        use_wls_filter: Apply WLS filter for edge-preserving smoothing.
        use_green_channel: Use green channel only.
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
            gray_l,
            (gray_l.shape[1] // downscale, gray_l.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )
        gray_r = cv2.resize(
            gray_r,
            (gray_r.shape[1] // downscale, gray_r.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    if sgbm is None:
        # Scale numDisparities down for reduced resolution
        nd = (
            max(16, (num_disparities // downscale // 16) * 16)
            if downscale > 1
            else num_disparities
        )
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
            disparity,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        upscaled_mask = cv2.resize(
            valid_mask.astype(np.uint8),
            (orig_w, orig_h),
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


def head_depth_in_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    mounting_height_mm: float,
    fx_px: float,
    cx_px: float,
    cy_px: float,
    *,
    rotated_bbox: Optional[tuple[float, float, float, float, float]] = None,
    slice_thickness_mm: float = 100.0,
    min_head_area_px: int = 40,
    max_head_height_mm: float = 1800.0,
    min_head_above_floor_mm: float = 500.0,
    column_radius_mm: float = 250.0,
) -> Optional[float]:
    """Find head depth inside a bbox using histogram + connected components,
    pre-filtered by a vertical column anchored on the body's 3-D centroid.

    The 2-D bbox is the only anchor the detector hands us, but in cenital
    geometry the bbox often clips overhead structure (cables, cornices,
    shelf edges) that sits at a different metric (X, Y) than the body
    underneath. When the body partially occludes that structure the gate
    works fine; when the body shifts (sitting, crouching) and the
    structure becomes visible, the histogram-walk-from-nearest happily
    picks the structure as "head" because it really is closer to the
    camera than the actual head.

    The fix is to anchor on a small estimate of the body's metric
    centroid (X_center, Y_center) and reject pixels whose back-projected
    metric (X, Y) falls outside a vertical column of radius
    ``column_radius_mm`` around it. Real heads always sit inside that
    column (the head is part of the body that produced the bbox).
    Overhead structure offset in metric X or Y is excluded.

    For zenith-mount stereo with a YOLO-COCO person bbox the centroid
    sits on the torso, not the head, so neither the central-crop median
    (measures the torso surface) nor a low percentile of the bbox depth
    (single-pixel speckle wins) gives a reliable head-height reading.
    After the spatial column filter we run the canonical histogram
    walk + connected-components pick: walk the depth distribution from
    camera-nearest outward, return the median of the largest connected
    pixel cluster in the first 10 cm slice that has enough area to be
    a real head.

    Reference: Del Pizzo et al., "Counting people by RGB or depth
    overhead cameras", Pattern Recognition Letters 2016 (histogram +
    components). Spatial column gate is our extension — needed because
    overhead structure produces depth slices that pass the anthropometric
    gate but are nowhere near the body in the world frame.

    Args:
        depth_map: Depth map in mm from disparity_to_depth(). 0 = invalid.
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        mounting_height_mm: Camera-to-floor distance.
        fx_px: Rectified focal length in px (P1[0, 0]). Used to back-
            project pixels to metric X, Y. We use the same fx for X and
            Y because the rectified model is square-pixel and only the
            horizontal scale enters depth-to-world conversion.
        cx_px: Rectified principal point x in px (P1[0, 2]).
        cy_px: Rectified principal point y in px (P1[1, 2]).
        rotated_bbox: Optional rotated rectangle ``(cx, cy, w, h, angle_deg)``
            in image-pixel coordinates. When provided, depth sampling is
            restricted to pixels inside this polygon — strictly tighter
            than the axis-aligned ``bbox``, which the rotated rectangle's
            envelope inflates by up to 2× area at 45° rotation. Crucial
            for cenital geometry where RAPiD emits a body-aligned rotated
            rectangle but ``bbox`` overshoots into floor + neighbouring
            structure that pass the anthropometric depth gate. Pass
            ``None`` (default) for axis-aligned detectors (yolov8) where
            ``bbox`` is already the tight envelope.
        slice_thickness_mm: Histogram bin width. 10 cm is wider than
            SGBM's depth quantization at the operational range, so a
            real head fills at least one bin solidly.
        min_head_area_px: Minimum pixel count for a slice to qualify as
            a head cluster. Tuned for the disparity grid at the runtime
            resolution + downscale combo — small enough to catch a child
            but big enough to reject speckle clusters.
        max_head_height_mm: Anthropometric ceiling on head height; depths
            below ``mount - max_head_height`` are rejected as impossible.
        min_head_above_floor_mm: Anthropometric floor (ignore depths
            corresponding to a head essentially on the ground — that's
            the floor itself, not a person).
        column_radius_mm: Radius (mm) of the vertical column around the
            body's 3-D centroid that pixels must fall inside to be
            considered as candidate head material. Default 250 mm: a
            human head sits well inside a 50 cm-wide column centred on
            the torso, while overhead structure offset by >25 cm in
            metric X or Y is rejected. Set very large to disable
            (effectively reverting to the pre-spatial-filter behaviour).

    Returns:
        Estimated head depth in mm, or None if no plausible head cluster
        is found. Use ``head_height_above_floor()`` to convert to height.
    """
    if depth_map.size == 0:
        return None
    if fx_px <= 0.0:
        return None
    h, w = depth_map.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_map[y1:y2, x1:x2]

    # Anthropometric gate: heads live in [mount-max_head, mount-min_above_floor].
    near = mounting_height_mm - max_head_height_mm
    far = mounting_height_mm - min_head_above_floor_mm
    if far <= near:
        return None

    valid = (roi > 0) & (roi >= near) & (roi <= far)
    if int(valid.sum()) < min_head_area_px:
        return None

    # ---- Rotated-rectangle mask (RAPiD-style detectors) ----
    # The axis-aligned ``bbox`` is the tight envelope of a rotated
    # rectangle for architectures like RAPiD. At non-trivial angles the
    # envelope sweeps in floor + neighbouring structure; in cenital
    # geometry that structure can sit at depths inside the anthropometric
    # gate (a desk edge, a chair back, a tall cabinet at the bbox
    # margin) and the histogram-walk-from-nearest then picks it as the
    # "head". Filtering ``valid`` against the rotated polygon hands the
    # function a body-shaped ROI even when the axis-aligned bbox is
    # 2× the area of the actual body.
    if rotated_bbox is not None:
        rcx, rcy, rw, rh, rang_deg = rotated_bbox
        rang = float(np.deg2rad(rang_deg))
        cos_a, sin_a = float(np.cos(rang)), float(np.sin(rang))
        # 4 corners of the rotated rectangle in original-image coords.
        dx = np.array(
            [-rw / 2.0, rw / 2.0, rw / 2.0, -rw / 2.0], dtype=np.float32
        )
        dy = np.array(
            [-rh / 2.0, -rh / 2.0, rh / 2.0, rh / 2.0], dtype=np.float32
        )
        corners_x = rcx + dx * cos_a - dy * sin_a
        corners_y = rcy + dx * sin_a + dy * cos_a
        # Translate to ROI-local coords (origin at bbox top-left). Round
        # to int once — fillPoly takes int32.
        poly = np.empty((4, 2), dtype=np.int32)
        poly[:, 0] = np.round(corners_x - x1).astype(np.int32)
        poly[:, 1] = np.round(corners_y - y1).astype(np.int32)
        poly_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 1)
        valid = valid & poly_mask.astype(bool)
        if int(valid.sum()) < min_head_area_px:
            return None

    # ---- 3-D back-projection of every gated pixel in the ROI ----
    # Build pixel-centre (u, v) grids covering the bbox in full-frame
    # coordinates so cx/cy reference the same origin as P1. Z is the
    # gated depth; X, Y come from the standard pinhole back-projection
    # X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fx
    # (same fx for both axes — rectified pixels are square).
    us = np.arange(x1, x2, dtype=np.float32) - float(cx_px)
    vs = np.arange(y1, y2, dtype=np.float32) - float(cy_px)
    u_grid, v_grid = np.meshgrid(us, vs)
    z_full = roi.astype(np.float32)
    inv_fx = np.float32(1.0 / fx_px)
    x_full = u_grid * z_full * inv_fx
    y_full = v_grid * z_full * inv_fx

    # ---- 3-D centroid of the body inside the bbox ----
    # Median over the gated central crop is robust to the overhead
    # structure that motivates the column filter — the structure
    # typically only kisses the bbox at the top/edges, while the
    # body fills the central crop. Using the centre means we anchor
    # on the body even when the structure dominates the gated area
    # count along the histogram nearest slice.
    bw, bh = x2 - x1, y2 - y1
    cw = max(1, bw // 2)
    ch = max(1, bh // 2)
    cx_off = (bw - cw) // 2
    cy_off = (bh - ch) // 2
    central_valid = valid[cy_off : cy_off + ch, cx_off : cx_off + cw]
    if int(central_valid.sum()) >= 4:
        x_center = float(
            np.median(x_full[cy_off : cy_off + ch, cx_off : cx_off + cw][central_valid])
        )
        y_center = float(
            np.median(y_full[cy_off : cy_off + ch, cx_off : cx_off + cw][central_valid])
        )
    else:
        # Central crop has too few valid pixels — fall back to the full
        # gated set so the function still returns something for thin
        # bboxes or holes-y depth maps.
        x_center = float(np.median(x_full[valid]))
        y_center = float(np.median(y_full[valid]))

    # ---- Spatial column filter ----
    # Keep only gated pixels whose (X, Y) is within column_radius_mm of
    # the body centroid. Squared-distance comparison avoids the sqrt.
    dx = x_full - x_center
    dy = y_full - y_center
    column_mask = (dx * dx + dy * dy) <= (column_radius_mm * column_radius_mm)
    valid_col = valid & column_mask
    if int(valid_col.sum()) < min_head_area_px:
        return None

    # Despeckle: median blur over the column-gated ROI. Use NaN for
    # invalids so the blur doesn't smear depth across holes. Fall back
    # to plain median if medianBlur isn't applicable to the dtype/shape
    # (paranoia).
    roi_f = np.where(valid_col, roi.astype(np.float32), np.nan)
    try:
        smooth = cv2.medianBlur(roi_f, 3)
    except cv2.error:
        smooth = roi_f
    valid_s = np.isfinite(smooth)
    if int(valid_s.sum()) < min_head_area_px:
        return None

    d = smooth[valid_s]
    bin_edges = np.arange(near, far + slice_thickness_mm, slice_thickness_mm)
    if len(bin_edges) < 2:
        return float(np.median(d))
    hist, edges = np.histogram(d, bins=bin_edges)

    # Walk from camera-nearest outward; first slice with >= min_head_area_px is the head.
    candidates = np.where(hist >= min_head_area_px)[0]
    if len(candidates) == 0:
        return None
    bin_idx = int(candidates[0])
    d_lo = float(edges[bin_idx])
    d_hi = float(edges[bin_idx + 1])

    # Connected components within the head slice. The biggest blob is
    # the head; smaller blobs in the same depth slice are usually noise.
    head_mask = ((smooth >= d_lo) & (smooth < d_hi)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(head_mask, connectivity=4)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[biggest, cv2.CC_STAT_AREA]) < min_head_area_px:
        return None

    blob_pixels = smooth[lbl == biggest]
    if blob_pixels.size == 0:
        return None
    return float(np.median(blob_pixels))


def min_depth_at_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    low_percentile: float = 15.0,
) -> float:
    """Estimate the depth of the nearest point in a bbox (top of head for
    ceiling-mounted stereo).

    Samples the central 50% of the bbox (matching ``depth_at_bbox``):
    in zenith geometry the head sits near the bbox centroid, so the
    central crop excludes peripheral SGBM noise from background pixels
    that happen to fall inside the YOLO-stock person bbox. Then takes
    a low percentile (default 15%) to stay robust to speckle outliers
    — the raw min or 5th percentile is too easily pulled below the
    true head depth at high SGBM downscale factors.

    Args:
        depth_map: Depth map in mm from disparity_to_depth().
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        low_percentile: Percentile for the "near" value. 15 is the
            empirical floor under which speckle clusters in the SGBM
            output start dominating at downscale=8.

    Returns:
        Estimated near-depth in mm. Returns 0.0 if no valid pixels.
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

    return float(np.percentile(valid, low_percentile))

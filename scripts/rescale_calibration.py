#!/usr/bin/env python3
"""Rescale a stereo calibration .npz to a different image resolution.

The fisheye Kannala-Brandt model separates intrinsics into:
- focal length + principal point (in pixels)  → scale linearly with resolution
- angular distortion coefficients k1..k4      → invariant to resolution
- extrinsics R, T (3D bracket geometry)        → invariant to resolution

So a calibration captured at one resolution can be analytically transformed
to any other resolution of the same sensor's full FOV — no re-capture, no
new lab session. Useful when you've calibrated at a high resolution
(2304x1296) and want to deploy at a lower one (1152x648 or 640x480) for
runtime FPS.

Caveats:
- Only valid for the SAME sensor mode / FOV. Going from binned 2304x1296
  to a partial-FOV crop mode (1536x864 IMX708) is NOT a valid rescale —
  the FOV changes and intrinsics differ.
- Picamera2 with the lower resolution should use the ISP downsize on
  hardware. If it's doing software downsize, you don't gain runtime FPS.

Usage:
    PYTHONPATH=. python3 scripts/rescale_calibration.py \\
        calibration.npz --output calibration_1152x648.npz \\
        --width 1152 --height 648
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.calibration import (
    FISHEYE_RECTIFY_BALANCE,
    load_calibration,
    save_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rescale_calibration")


def rescale_calibration(
    cal: dict, new_w: int, new_h: int,
) -> dict:
    """Return a new calibration dict mathematically rescaled to new_w × new_h.

    Intrinsics K scale linearly. Distortion D and extrinsics R, T are
    invariant. Rectification (R1, R2, P1, P2, Q) and the rectify maps are
    recomputed for the new resolution.
    """
    old_size = tuple(int(v) for v in cal["image_size"])
    old_w, old_h = old_size
    sx = new_w / old_w
    sy = new_h / old_h
    logger.info(
        "Rescaling %dx%d → %dx%d (sx=%.4f, sy=%.4f)",
        old_w, old_h, new_w, new_h, sx, sy,
    )

    # Scale intrinsics. Fisheye K is the standard 3x3 [[fx,0,cx],[0,fy,cy],[0,0,1]].
    K_l = cal["camera_matrix_l"].copy().astype(np.float64)
    K_l[0, 0] *= sx
    K_l[0, 2] *= sx
    K_l[1, 1] *= sy
    K_l[1, 2] *= sy

    K_r = cal["camera_matrix_r"].copy().astype(np.float64)
    K_r[0, 0] *= sx
    K_r[0, 2] *= sx
    K_r[1, 1] *= sy
    K_r[1, 2] *= sy

    # Kannala-Brandt distortion (k1..k4) is angle-based — invariant.
    D_l = cal["dist_coeffs_l"].astype(np.float64)
    D_r = cal["dist_coeffs_r"].astype(np.float64)

    # Extrinsics are pure 3D geometry (mm, rotation matrix) — invariant.
    R = cal["R"].astype(np.float64)
    T = cal["T"].astype(np.float64)

    # Recompute rectification with the new image size. R1 / R2 are pure
    # rotations (independent of resolution); P1 / P2 / Q depend on the
    # output frame size, so re-derive them here.
    new_size = (int(new_w), int(new_h))
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l, D_l, K_r, D_r, new_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=new_size,
        balance=FISHEYE_RECTIFY_BALANCE,
    )

    # New rectify maps for cv2.remap.
    map_l_x, map_l_y = cv2.fisheye.initUndistortRectifyMap(
        K_l, D_l, R1, P1, new_size, cv2.CV_32FC1,
    )
    map_r_x, map_r_y = cv2.fisheye.initUndistortRectifyMap(
        K_r, D_r, R2, P2, new_size, cv2.CV_32FC1,
    )

    new_cal = {
        "camera_matrix_l": K_l, "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r, "dist_coeffs_r": D_r,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_l_x": map_l_x, "map_l_y": map_l_y,
        "map_r_x": map_r_x, "map_r_y": map_r_y,
        "image_size": np.array(list(new_size)),
    }

    # Sanity checks
    fx_old = float(cal["P1"][0, 0])
    fx_new = float(P1[0, 0])
    baseline_old = float(np.linalg.norm(cal["T"]))
    baseline_new = float(np.linalg.norm(T))
    logger.info(
        "P1.fx: %.1f → %.1f (ratio %.4f, expected %.4f)",
        fx_old, fx_new, fx_new / fx_old, sx,
    )
    logger.info(
        "Baseline preserved: %.2fmm → %.2fmm (Δ=%.4fmm)",
        baseline_old, baseline_new, baseline_new - baseline_old,
    )
    return new_cal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input",
                        help="Input calibration .npz (e.g. calibration.npz)")
    parser.add_argument("--output", required=True,
                        help="Output .npz path")
    parser.add_argument("--width", type=int, required=True,
                        help="Target image width in pixels")
    parser.add_argument("--height", type=int, required=True,
                        help="Target image height in pixels")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")

    in_path = Path(args.input)
    if not in_path.exists():
        parser.error(f"Input not found: {in_path}")

    cal = load_calibration(str(in_path))
    new_cal = rescale_calibration(cal, args.width, args.height)
    save_calibration(new_cal, args.output)
    logger.info("✓ Wrote %s (%dx%d)", args.output, args.width, args.height)


if __name__ == "__main__":
    main()

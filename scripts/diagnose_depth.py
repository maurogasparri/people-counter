#!/usr/bin/env python3
"""Depth pipeline diagnostic tool.

Measures depth at multiple known distances to diagnose scale errors.
Prints calibration parameters, raw disparity, and computed depth for
comparison with ground truth.

Usage:
    # Place an object at a known distance and run:
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 2200

    # Test multiple distances:
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 1500
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 2200
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 3000
"""

import argparse

import cv2
import numpy as np

from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import StereoCapture
from src.vision.depth import compute_disparity, create_sgbm, disparity_to_depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth diagnostic")
    parser.add_argument("--distance", type=float, required=True,
                        help="Actual distance to object in mm")
    parser.add_argument("--calibration", default="/etc/people-counter/calibration.npz")
    args = parser.parse_args()

    # Load calibration
    cal = load_calibration(args.calibration)

    print("=" * 60)
    print("CALIBRATION PARAMETERS")
    print("=" * 60)

    # Intrinsics from K (original)
    K_l = cal["camera_matrix_l"]
    K_r = cal["camera_matrix_r"]
    print(f"K_left  fx={K_l[0,0]:.1f}  fy={K_l[1,1]:.1f}  cx={K_l[0,2]:.1f}  cy={K_l[1,2]:.1f}")
    print(f"K_right fx={K_r[0,0]:.1f}  fy={K_r[1,1]:.1f}  cx={K_r[0,2]:.1f}  cy={K_r[1,2]:.1f}")

    # Intrinsics from P (rectified)
    P1 = cal["P1"]
    P2 = cal["P2"]
    print(f"P1 fx={P1[0,0]:.1f}  fy={P1[1,1]:.1f}  cx={P1[0,2]:.1f}  cy={P1[1,2]:.1f}")
    print(f"P2 fx={P2[0,0]:.1f}  fy={P2[1,1]:.1f}  cx={P2[0,2]:.1f}  cy={P2[1,2]:.1f}")
    print(f"P2[0,3] (Tx*fx) = {P2[0,3]:.1f}")

    # Baseline from T
    T = cal["T"]
    baseline_T = float(np.linalg.norm(T))
    print(f"T = {T.flatten()}")
    print(f"Baseline from ||T|| = {baseline_T:.1f} mm")

    # Baseline from P2
    if P1[0, 0] != 0:
        baseline_P2 = abs(P2[0, 3]) / P1[0, 0]
        print(f"Baseline from P2[0,3]/P1_fx = {baseline_P2:.1f} mm")
    else:
        baseline_P2 = baseline_T

    # Q matrix
    Q = cal["Q"]
    print(f"\nQ matrix:")
    print(Q)
    print(f"Q[3,2] (1/baseline) = {Q[3,2]:.6f} → baseline = {1/Q[3,2] if Q[3,2] != 0 else 'inf':.1f} mm")
    print(f"Q[2,3] (focal) = {Q[2,3]:.1f}")

    # Image size
    print(f"\nImage size = {cal.get('image_size', 'N/A')}")

    # Capture
    print("\n" + "=" * 60)
    print("CAPTURING...")
    print("=" * 60)

    cap = StereoCapture(cam_left_id=0, cam_right_id=1, resolution=(2592, 1944), fps=5)
    cap.open()
    left, right = cap.read()
    cap.close()
    print(f"Captured: {left.shape}")

    # Rectify
    rect_l, rect_r = rectify_pair(left, right, cal)
    print(f"Rectified: {rect_l.shape}")

    # SGBM
    sgbm = create_sgbm()
    disparity = compute_disparity(rect_l, rect_r, sgbm=sgbm)

    # Analyze disparity at center of image
    h, w = disparity.shape
    center_region = disparity[h//3:2*h//3, w//3:2*w//3]
    valid_center = center_region[center_region > 0]

    print(f"\n{'=' * 60}")
    print("DISPARITY ANALYSIS (center region)")
    print(f"{'=' * 60}")
    if len(valid_center) > 0:
        disp_median = float(np.median(valid_center))
        disp_mean = float(np.mean(valid_center))
        print(f"Valid pixels in center: {len(valid_center)}")
        print(f"Disparity median: {disp_median:.2f} px")
        print(f"Disparity mean: {disp_mean:.2f} px")
    else:
        print("No valid disparity in center!")
        disp_median = 0

    # Compute depth multiple ways
    print(f"\n{'=' * 60}")
    print("DEPTH CALCULATIONS")
    print(f"{'=' * 60}")
    print(f"Ground truth distance: {args.distance:.0f} mm ({args.distance/1000:.2f} m)")

    if disp_median > 0:
        # Method 1: P1_fx * T_baseline / disparity
        fx_p1 = P1[0, 0]
        depth_1 = fx_p1 * baseline_T / disp_median
        print(f"\nMethod 1: P1_fx * ||T|| / disp")
        print(f"  fx={fx_p1:.1f}, baseline={baseline_T:.1f}mm, disp={disp_median:.2f}")
        print(f"  Depth = {depth_1:.0f} mm ({depth_1/1000:.2f} m)")
        print(f"  Error = {abs(depth_1 - args.distance)/args.distance*100:.1f}%")

        # Method 2: P1_fx * P2_baseline / disparity
        depth_2 = fx_p1 * baseline_P2 / disp_median
        print(f"\nMethod 2: P1_fx * P2_baseline / disp")
        print(f"  fx={fx_p1:.1f}, baseline={baseline_P2:.1f}mm, disp={disp_median:.2f}")
        print(f"  Depth = {depth_2:.0f} mm ({depth_2/1000:.2f} m)")
        print(f"  Error = {abs(depth_2 - args.distance)/args.distance*100:.1f}%")

        # Method 3: using Q matrix (reprojectImageTo3D)
        # Q gives: Z = focal / (disp + Q[3,3]/Q[3,2])... simplified: Z = -Q[2,3] / (disp * Q[3,2])
        if Q[3, 2] != 0:
            depth_3 = -Q[2, 3] / (disp_median * Q[3, 2])
            print(f"\nMethod 3: Q matrix (-Q[2,3] / (disp * Q[3,2]))")
            print(f"  Q[2,3]={Q[2,3]:.1f}, Q[3,2]={Q[3,2]:.6f}, disp={disp_median:.2f}")
            print(f"  Depth = {depth_3:.0f} mm ({depth_3/1000:.2f} m)")
            print(f"  Error = {abs(depth_3 - args.distance)/args.distance*100:.1f}%")

        # Method 4: K_fx instead of P1_fx
        fx_k = K_l[0, 0]
        depth_4 = fx_k * baseline_T / disp_median
        print(f"\nMethod 4: K_fx * ||T|| / disp (using original K, NOT rectified P)")
        print(f"  fx={fx_k:.1f}, baseline={baseline_T:.1f}mm, disp={disp_median:.2f}")
        print(f"  Depth = {depth_4:.0f} mm ({depth_4/1000:.2f} m)")
        print(f"  Error = {abs(depth_4 - args.distance)/args.distance*100:.1f}%")

        # Expected disparity at this distance
        expected_disp = fx_p1 * baseline_T / args.distance
        print(f"\nExpected disparity at {args.distance:.0f}mm: {expected_disp:.2f} px")
        print(f"Actual disparity: {disp_median:.2f} px")
        print(f"Ratio actual/expected: {disp_median/expected_disp:.3f}")

    print(f"\n{'=' * 60}")
    print("SAVED")
    print(f"{'=' * 60}")
    cv2.imwrite("/tmp/diagnose_rect_l.jpg", rect_l)
    cv2.imwrite("/tmp/diagnose_rect_r.jpg", rect_r)
    depth_map = disparity_to_depth(disparity, P1[0, 0], baseline_T)
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite("/tmp/diagnose_depth.jpg", depth_vis)
    print("Rectified: /tmp/diagnose_rect_l.jpg, /tmp/diagnose_rect_r.jpg")
    print("Depth map: /tmp/diagnose_depth.jpg")


if __name__ == "__main__":
    main()

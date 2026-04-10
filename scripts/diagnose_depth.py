#!/usr/bin/env python3
"""Depth pipeline diagnostic tool.

Measures depth at known distances using multiple methods to isolate
scale errors in the stereo pipeline.

Usage:
    # Place yourself at a known distance and run:
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 2200

    # Test multiple distances (run once per distance):
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
    parser.add_argument("--no-wls", action="store_true", help="Disable WLS filter")
    parser.add_argument("--wls", action="store_true", help="Enable WLS filter (off by default)")
    parser.add_argument("--green", action="store_true",
                        help="Use green channel only (for NoIR cameras)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4],
                        help="Downscale factor for SGBM matching (1=full, 2=half, 4=quarter)")
    parser.add_argument("--delay", type=int, default=0,
                        help="Countdown in seconds before capture")
    args = parser.parse_args()

    cal = load_calibration(args.calibration)

    # --- Print calibration parameters ---
    print("=" * 60)
    print("CALIBRATION PARAMETERS")
    print("=" * 60)

    K_l = cal["camera_matrix_l"]
    K_r = cal["camera_matrix_r"]
    D_l = cal["dist_coeffs_l"]
    D_r = cal["dist_coeffs_r"]
    P1 = cal["P1"]
    P2 = cal["P2"]
    T = cal["T"]
    Q = cal["Q"]

    print(f"K_left  fx={K_l[0,0]:.1f}  fy={K_l[1,1]:.1f}  cx={K_l[0,2]:.1f}  cy={K_l[1,2]:.1f}")
    print(f"K_right fx={K_r[0,0]:.1f}  fy={K_r[1,1]:.1f}  cx={K_r[0,2]:.1f}  cy={K_r[1,2]:.1f}")
    print(f"P1 fx={P1[0,0]:.1f}  fy={P1[1,1]:.1f}  cx={P1[0,2]:.1f}  cy={P1[1,2]:.1f}")
    print(f"P2 fx={P2[0,0]:.1f}  fy={P2[1,1]:.1f}  cx={P2[0,2]:.1f}  cy={P2[1,2]:.1f}")
    print(f"P2[0,3] (Tx*fx) = {P2[0,3]:.1f}")
    print(f"D_left  ({D_l.flatten().shape[0]} coeffs): {D_l.flatten()}")
    print(f"D_right ({D_r.flatten().shape[0]} coeffs): {D_r.flatten()}")

    # Baseline calculations
    baseline_T = float(np.linalg.norm(T))
    baseline_P2 = abs(P2[0, 3]) / P1[0, 0] if P1[0, 0] != 0 else 0
    print(f"\nT = {T.flatten()}")
    print(f"Baseline from ||T|| = {baseline_T:.1f} mm")
    print(f"Baseline from P2[0,3]/P1_fx = {baseline_P2:.1f} mm")

    print(f"\nQ matrix:\n{Q}")

    # --- Sanity checks ---
    print(f"\n{'=' * 60}")
    print("SANITY CHECKS")
    print("=" * 60)

    fx_p1 = P1[0, 0]
    if fx_p1 < 50:
        print("WARNING: P1 fx < 50 — too small for rectified image")
    elif fx_p1 > 5000:
        print("WARNING: P1 fx > 5000 — too large, check resolution")
    else:
        print(f"OK: P1 fx = {fx_p1:.1f}")

    if baseline_T > 1000:
        print("WARNING: baseline > 1m — probably in wrong units")
    elif baseline_T < 10:
        print("WARNING: baseline < 10mm — probably in wrong units")
    else:
        print(f"OK: baseline = {baseline_T:.1f} mm")

    if abs(baseline_T - baseline_P2) > 5:
        print(f"WARNING: baseline mismatch: T={baseline_T:.1f} vs P2={baseline_P2:.1f}")
    else:
        print(f"OK: baseline consistent (T={baseline_T:.1f}, P2={baseline_P2:.1f})")

    # --- Capture ---
    print(f"\n{'=' * 60}")
    print("CAPTURING...")
    print("=" * 60)

    cap = StereoCapture(cam_left_id=0, cam_right_id=1, resolution=(4608, 2592), fps=5)
    cap.open()

    if args.delay > 0:
        import time
        for i in range(args.delay, 0, -1):
            print(f"  Capturing in {i}...", end="\r", flush=True)
            time.sleep(1)
        print()

    left, right = cap.read()
    cap.close()
    print(f"Captured: {left.shape}")

    # --- Rectify ---
    rect_l, rect_r = rectify_pair(left, right, cal)
    print(f"Rectified: {rect_l.shape}")

    # --- Disparity ---
    use_wls = args.wls and not args.no_wls  # off by default
    sgbm = create_sgbm()
    disparity = compute_disparity(
        rect_l, rect_r, sgbm=sgbm,
        use_wls_filter=use_wls,
        use_green_channel=args.green,
        use_clahe=not args.no_clahe,
        downscale=args.downscale,
    )

    # Center ROI analysis
    h, w = disparity.shape
    roi_y1, roi_y2 = h // 3, 2 * h // 3
    roi_x1, roi_x2 = w // 3, 2 * w // 3
    disp_roi = disparity[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_disp = disp_roi[disp_roi > 0.1]

    print(f"\n{'=' * 60}")
    print("DISPARITY (center ROI)")
    print("=" * 60)

    if len(valid_disp) == 0:
        print("ERROR: No valid disparity in center region!")
        return

    disp_mean = float(np.mean(valid_disp))
    disp_median = float(np.median(valid_disp))
    print(f"Valid pixels: {len(valid_disp)}")
    print(f"Mean:   {disp_mean:.2f} px")
    print(f"Median: {disp_median:.2f} px")

    # --- Depth: multiple methods ---
    print(f"\n{'=' * 60}")
    print("DEPTH CALCULATIONS")
    print("=" * 60)
    print(f"Ground truth: {args.distance:.0f} mm ({args.distance / 1000:.2f} m)\n")

    results = []

    # Method 1: Z = P1_fx * baseline_T / disparity
    depth_1 = fx_p1 * baseline_T / disp_median
    err_1 = (depth_1 - args.distance) / args.distance * 100
    print(f"Method 1: P1_fx × ||T|| / disp")
    print(f"  Z = {fx_p1:.1f} × {baseline_T:.1f} / {disp_median:.2f} = {depth_1:.0f} mm ({depth_1/1000:.2f} m)")
    print(f"  Error: {err_1:+.1f}%")
    results.append(("P1_fx × ||T||", depth_1, err_1))

    # Method 2: Z = P1_fx * baseline_P2 / disparity
    depth_2 = fx_p1 * baseline_P2 / disp_median
    err_2 = (depth_2 - args.distance) / args.distance * 100
    print(f"\nMethod 2: P1_fx × P2_baseline / disp")
    print(f"  Z = {fx_p1:.1f} × {baseline_P2:.1f} / {disp_median:.2f} = {depth_2:.0f} mm ({depth_2/1000:.2f} m)")
    print(f"  Error: {err_2:+.1f}%")
    results.append(("P1_fx × P2_base", depth_2, err_2))

    # Method 3: reprojectImageTo3D with Q matrix
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    z_roi = points_3d[roi_y1:roi_y2, roi_x1:roi_x2, 2]
    valid_z = z_roi[np.isfinite(z_roi) & (z_roi > 0) & (disp_roi > 0.1)]
    if len(valid_z) > 0:
        depth_3 = float(np.median(valid_z))
        # Q might output in mm or m depending on calibration units
        if depth_3 < 100:  # probably meters
            depth_3_mm = depth_3 * 1000
        else:
            depth_3_mm = depth_3
        err_3 = (depth_3_mm - args.distance) / args.distance * 100
        print(f"\nMethod 3: reprojectImageTo3D(Q)")
        print(f"  Z median = {depth_3:.2f} (raw), {depth_3_mm:.0f} mm")
        print(f"  Error: {err_3:+.1f}%")
        results.append(("Q reproject", depth_3_mm, err_3))
    else:
        print(f"\nMethod 3: reprojectImageTo3D(Q) — no valid Z values")

    # Method 4: K_fx (original, not rectified) — should be WRONG
    fx_k = K_l[0, 0]
    depth_4 = fx_k * baseline_T / disp_median
    err_4 = (depth_4 - args.distance) / args.distance * 100
    print(f"\nMethod 4: K_fx × ||T|| / disp (WRONG — uses original K, not rectified P)")
    print(f"  Z = {fx_k:.1f} × {baseline_T:.1f} / {disp_median:.2f} = {depth_4:.0f} mm ({depth_4/1000:.2f} m)")
    print(f"  Error: {err_4:+.1f}%")
    results.append(("K_fx × ||T|| (wrong)", depth_4, err_4))

    # Expected disparity
    expected_disp = fx_p1 * baseline_T / args.distance
    print(f"\nExpected disparity at {args.distance:.0f} mm: {expected_disp:.2f} px")
    print(f"Actual disparity: {disp_median:.2f} px")
    print(f"Ratio actual/expected: {disp_median / expected_disp:.3f}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25s} {'Depth (mm)':>12s} {'Error':>8s}")
    print("-" * 50)
    for name, depth, err in results:
        marker = " ◀ best" if abs(err) == min(abs(e) for _, _, e in results) else ""
        print(f"{name:<25s} {depth:>10.0f}   {err:>+6.1f}%{marker}")

    # --- Save ---
    cv2.imwrite("/tmp/diagnose_rect_l.jpg", rect_l)
    cv2.imwrite("/tmp/diagnose_rect_r.jpg", rect_r)
    depth_map = disparity_to_depth(disparity, fx_p1, baseline_T)
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # Draw center ROI rectangle
    cv2.rectangle(depth_vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)
    cv2.imwrite("/tmp/diagnose_depth.jpg", depth_vis)
    print(f"\nSaved: /tmp/diagnose_rect_l.jpg, /tmp/diagnose_rect_r.jpg, /tmp/diagnose_depth.jpg")


if __name__ == "__main__":
    main()

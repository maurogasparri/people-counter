#!/usr/bin/env python3
"""Depth pipeline diagnostic and calibration validation tool.

Measures depth at known distance across 5 zones (center + 4 corners) to
validate calibration quality for the IMX708 120° HFOV stereo pair.

For 120° wide-angle lenses, the periphery is where distortion is hardest
to model — if calibration is poor, edge zones diverge from center much
more than the radial distance alone would predict. This script makes that
divergence explicit.

Usage:
    # Place a flat object (wall, board) parallel to the cameras at a
    # known distance, run once per distance:
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 1000
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 2000
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 3000

Validation thresholds (PASS/FAIL):
    - Center error: <5% at 2m, <10% at 3m
    - Edge/center error ratio: <2.0 (edges no worse than 2× center)
"""

import argparse

import cv2
import numpy as np

from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import StereoCapture
from src.vision.depth import compute_disparity, create_sgbm, disparity_to_depth


# Validation thresholds
PASS_ERROR_PCT_AT_2M = 5.0
PASS_ERROR_PCT_AT_3M = 10.0
PASS_EDGE_CENTER_RATIO = 2.0


def analyze_zone(disparity: np.ndarray, fx: float, baseline_mm: float,
                 cy: int, cx: int, half: int) -> tuple[int, float, float, float]:
    """Compute depth statistics inside a square ROI centered at (cy, cx).

    Returns (valid_count, median_depth_mm, std_depth_mm, fill_pct).
    """
    h, w = disparity.shape
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    disp_roi = disparity[y1:y2, x1:x2]
    valid = disp_roi[disp_roi > 0.1]
    total = disp_roi.size
    if len(valid) == 0:
        return 0, float("nan"), float("nan"), 0.0
    depths = fx * baseline_mm / valid
    return (
        len(valid),
        float(np.median(depths)),
        float(np.std(depths)),
        100.0 * len(valid) / total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth diagnostic + calibration validation")
    parser.add_argument("--distance", type=float, required=True,
                        help="Actual distance to flat target in mm")
    parser.add_argument("--calibration", default="/etc/people-counter/calibration.npz")
    parser.add_argument("--wls", action="store_true", help="Enable WLS filter (off by default)")
    parser.add_argument("--green", action="store_true",
                        help="Use green channel only (for NoIR cameras)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4],
                        help="Downscale factor for SGBM matching (1=full, 2=half, 4=quarter)")
    parser.add_argument("--zone-fraction", type=float, default=0.15,
                        help="Zone size as fraction of min(H,W). Default 0.15 = ~15%% per side")
    parser.add_argument("--edge-margin", type=float, default=0.10,
                        help="Edge zone offset from corner, fraction of min(H,W). Default 0.10")
    parser.add_argument("--delay", type=int, default=0,
                        help="Countdown in seconds before capture")
    args = parser.parse_args()

    cal = load_calibration(args.calibration)

    # --- Print calibration parameters ---
    print("=" * 70)
    print("CALIBRATION PARAMETERS")
    print("=" * 70)
    K_l, K_r = cal["camera_matrix_l"], cal["camera_matrix_r"]
    P1, P2, T, Q = cal["P1"], cal["P2"], cal["T"], cal["Q"]
    fx_p1 = float(P1[0, 0])
    baseline_T = float(np.linalg.norm(T))
    baseline_P2 = abs(float(P2[0, 3])) / fx_p1 if fx_p1 != 0 else 0.0

    print(f"P1 fx={fx_p1:.1f}  fy={P1[1,1]:.1f}  cx={P1[0,2]:.1f}  cy={P1[1,2]:.1f}")
    print(f"Baseline ||T|| = {baseline_T:.1f} mm  /  P2[0,3]/fx = {baseline_P2:.1f} mm")

    if abs(baseline_T - baseline_P2) > 5:
        print(f"  WARNING: baseline mismatch (>5mm)")

    # --- Capture ---
    print(f"\n{'=' * 70}")
    print("CAPTURING...")
    print("=" * 70)

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
    print(f"Captured: {left.shape}, rectifying...")

    rect_l, rect_r = rectify_pair(left, right, cal)

    # --- Disparity ---
    sgbm = create_sgbm()
    disparity = compute_disparity(
        rect_l, rect_r, sgbm=sgbm,
        use_wls_filter=args.wls,
        use_green_channel=args.green,
        use_clahe=not args.no_clahe,
        downscale=args.downscale,
    )

    h, w = disparity.shape
    short = min(h, w)
    half = int(short * args.zone_fraction / 2)
    margin = int(short * args.edge_margin)

    # 5 zones: center + 4 corners (offset from image edge by `margin`)
    zones = {
        "center":       (h // 2, w // 2),
        "top-left":     (margin + half, margin + half),
        "top-right":    (margin + half, w - margin - half),
        "bottom-left":  (h - margin - half, margin + half),
        "bottom-right": (h - margin - half, w - margin - half),
    }

    # --- Per-zone analysis ---
    print(f"\n{'=' * 70}")
    print(f"DEPTH PER ZONE  (ground truth: {args.distance:.0f} mm = {args.distance/1000:.2f} m)")
    print(f"Zone size: {2*half}×{2*half} px  ({args.zone_fraction*100:.0f}% of frame side)")
    print("=" * 70)
    print(f"{'Zone':<14s} {'Valid':>7s} {'Fill%':>7s} {'Depth(mm)':>11s} {'Std(mm)':>9s} {'Error':>9s}")
    print("-" * 70)

    results = {}
    for name, (cy, cx) in zones.items():
        n, median, std, fill = analyze_zone(disparity, fx_p1, baseline_T, cy, cx, half)
        if n == 0:
            print(f"{name:<14s} {'NO DATA':>7s}")
            results[name] = None
            continue
        err_pct = (median - args.distance) / args.distance * 100
        results[name] = (median, std, err_pct, fill)
        print(f"{name:<14s} {n:>7d} {fill:>6.1f}% {median:>10.0f} {std:>8.0f}  {err_pct:>+7.1f}%")

    # --- Consistency analysis ---
    print(f"\n{'=' * 70}")
    print("CENTER vs EDGES CONSISTENCY")
    print("=" * 70)

    center = results.get("center")
    edges = [v for k, v in results.items() if k != "center" and v is not None]

    if center is None or not edges:
        print("Cannot compute consistency — missing zones")
        edge_ratio = float("nan")
    else:
        center_err = abs(center[2])
        edge_errs = [abs(v[2]) for v in edges]
        max_edge_err = max(edge_errs)
        mean_edge_err = sum(edge_errs) / len(edge_errs)
        edge_ratio = max_edge_err / max(center_err, 0.1)

        print(f"Center |error|:        {center_err:5.2f}%")
        print(f"Mean edge |error|:     {mean_edge_err:5.2f}%")
        print(f"Max edge |error|:      {max_edge_err:5.2f}%")
        print(f"Max edge / center:     {edge_ratio:5.2f}×")

    # --- PASS/FAIL ---
    print(f"\n{'=' * 70}")
    print("VALIDATION VERDICT")
    print("=" * 70)

    # Threshold scales linearly between 2m and 3m
    d_m = args.distance / 1000
    if d_m <= 2.0:
        center_threshold = PASS_ERROR_PCT_AT_2M
    elif d_m >= 3.0:
        center_threshold = PASS_ERROR_PCT_AT_3M
    else:
        # interpolate
        t = (d_m - 2.0) / 1.0
        center_threshold = PASS_ERROR_PCT_AT_2M + t * (PASS_ERROR_PCT_AT_3M - PASS_ERROR_PCT_AT_2M)

    checks = []
    if center is not None:
        ok = abs(center[2]) <= center_threshold
        checks.append(("Center error", f"{abs(center[2]):.2f}% ≤ {center_threshold:.1f}%", ok))
    if not np.isnan(edge_ratio):
        ok = edge_ratio <= PASS_EDGE_CENTER_RATIO
        checks.append(("Edge/center ratio", f"{edge_ratio:.2f}× ≤ {PASS_EDGE_CENTER_RATIO:.1f}×", ok))

    for name, detail, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    overall = all(ok for _, _, ok in checks) if checks else False
    print(f"\nOVERALL: {'PASS — calibration is good for depth' if overall else 'FAIL — recalibrate (more captures, better focus, check rigidity)'}")

    # --- Save annotated visualization ---
    cv2.imwrite("/tmp/diagnose_rect_l.jpg", rect_l)
    cv2.imwrite("/tmp/diagnose_rect_r.jpg", rect_r)
    depth_map = disparity_to_depth(disparity, fx_p1, baseline_T)
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    for name, (cy, cx) in zones.items():
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 3)
        if results.get(name) is not None:
            label = f"{results[name][0]:.0f}mm ({results[name][2]:+.1f}%)"
        else:
            label = "no data"
        cv2.putText(depth_vis, label, (x1 + 5, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imwrite("/tmp/diagnose_depth.jpg", depth_vis)
    print(f"\nSaved: /tmp/diagnose_rect_l.jpg, /tmp/diagnose_rect_r.jpg, /tmp/diagnose_depth.jpg")


if __name__ == "__main__":
    main()

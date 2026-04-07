#!/usr/bin/env python3
"""CLI tool for stereo camera calibration using ChArUco patterns.

Usage:
    # Step 1: Capture image pairs (interactive, with HTTP preview)
    python scripts/calibrate.py capture --count 30
    # Open http://people-counter.local:8080 to see live preview

    # Step 2: Run calibration from captured pairs
    python scripts/calibrate.py calibrate --input-dir ./calibration/captures --output calibration.npz

    # Step 3: Verify calibration (draw epipolar lines on rectified pair)
    python scripts/calibrate.py verify --calibration calibration.npz --input-dir ./calibration/captures
"""

import argparse
import logging
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.calibration import (
    calibrate_stereo,
    create_charuco_board,
    detect_charuco_corners,
    generate_board_image,
    load_calibration,
    rectify_pair,
    save_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate")

# ---------------------------------------------------------------------------
# HTTP preview for capture
# ---------------------------------------------------------------------------

_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()
_shutting_down = False


class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Calibration Capture</title>
<style>body{background:#111;margin:0;display:flex;justify-content:center;
align-items:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/stream"></body></html>""")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _shutting_down:
                    with _jpeg_lock:
                        frame = _latest_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.15)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def _update_preview(jpeg_bytes: bytes) -> None:
    global _latest_jpeg
    with _jpeg_lock:
        _latest_jpeg = jpeg_bytes


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------


GRID_RECTANGULAR = np.ones((4, 5), dtype=np.int32)

GRID_CIRCULAR = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
], dtype=np.int32)

# Weighted grid for 170° fisheye: values = captures per cell.
# Center (3): most captures — stable intrinsics.
# Mid ring (2): key for distortion estimation.
# Edge (1): adds barrel info without detection issues.
# Corners (0): too distorted, skip.
GRID_170 = np.array([
    [0, 0, 2, 2, 0, 0],
    [0, 2, 4, 4, 2, 0],
    [0, 2, 6, 6, 2, 0],
    [0, 2, 6, 6, 2, 0],
    [0, 2, 4, 4, 2, 0],
    [0, 0, 2, 2, 0, 0],
], dtype=np.int32)


def _compute_coverage_center(
    corners: np.ndarray, w: int, h: int, grid_mask: np.ndarray,
) -> tuple[int, int]:
    """Get grid cell (row, col) for the center of detected corners."""
    cx = np.mean(corners[:, 0, 0])
    cy = np.mean(corners[:, 0, 1])
    grid_rows, grid_cols = grid_mask.shape
    col = int(cx / w * grid_cols)
    row = int(cy / h * grid_rows)
    col = max(0, min(col, grid_cols - 1))
    row = max(0, min(row, grid_rows - 1))
    return row, col


def _draw_coverage(
    frame: np.ndarray, coverage: np.ndarray, grid_mask: np.ndarray,
    cell_target: np.ndarray | None = None,
) -> None:
    """Draw coverage grid overlay on frame."""
    h, w = frame.shape[:2]
    grid_rows, grid_cols = coverage.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            if grid_mask[r, c] == 0:
                # Inactive cell — dim overlay
                overlay = frame[y1:y2, x1:x2].copy()
                dark = np.full_like(overlay, (0, 0, 0))
                cv2.addWeighted(dark, 0.5, overlay, 0.5, 0, frame[y1:y2, x1:x2])
                continue
            target = cell_target[r, c] if cell_target is not None else 0
            is_full = target > 0 and coverage[r, c] >= target
            if coverage[r, c] > 0:
                overlay = frame[y1:y2, x1:x2].copy()
                tint = (0, 80, 0) if is_full else (0, 60, 60)
                cv2.addWeighted(np.full_like(overlay, tint), 0.3, overlay, 0.7, 0, frame[y1:y2, x1:x2])
            label = f"{int(coverage[r, c])}/{target}" if target > 0 else str(int(coverage[r, c]))
            color = (0, 255, 0) if is_full else (0, 200, 255)
            cv2.putText(frame, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_generate_board(args: argparse.Namespace) -> None:
    """Generate a printable ChArUco board image."""
    board = create_charuco_board(
        board_size=(args.columns, args.rows),
        square_length=args.square_length,
        marker_length=args.marker_length,
    )
    img = generate_board_image(board, (args.width, args.height))
    cv2.imwrite(args.output, img)
    logger.info("Board saved to %s (%dx%d)", args.output, args.width, args.height)


def cmd_capture(args: argparse.Namespace) -> None:
    """Interactive capture with HTTP preview and coverage tracking."""
    global _shutting_down

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.vision.capture import StereoCapture

    cap = StereoCapture(
        cam_left_id=args.left,
        cam_right_id=args.right,
        resolution=tuple(args.resolution),
        fps=args.fps,
    )
    cap.open()

    board = create_charuco_board(
        board_size=(args.columns, args.rows),
        square_length=args.square_length,
        marker_length=args.marker_length,
    )

    # Start HTTP preview
    server = HTTPServer(("0.0.0.0", args.port), _MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Coverage grid
    if args.grid == "170":
        grid_mask = GRID_170
    elif args.grid == "circular":
        grid_mask = GRID_CIRCULAR
    else:
        grid_mask = GRID_RECTANGULAR
    grid_rows, grid_cols = grid_mask.shape
    coverage = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    # Per-cell targets: for weighted grids (170) use values directly,
    # for binary grids use --per-cell if set, else no per-cell limit.
    is_weighted = grid_mask.max() > 1
    if is_weighted:
        cell_target = grid_mask.copy()
    elif args.per_cell > 0:
        cell_target = grid_mask * args.per_cell
    else:
        cell_target = None  # no per-cell limit

    count = 0
    last_capture_time = 0.0
    valid_cells = int(np.count_nonzero(grid_mask))
    total_target = int(cell_target.sum()) if cell_target is not None else args.count

    logger.info("Calibration capture — preview: http://people-counter.local:%d", args.port)
    logger.info("Grid: %s (%d×%d, %d active cells, %d total captures)", args.grid, grid_rows, grid_cols, valid_cells, total_target)
    logger.info("Move the ChArUco to cover all grid cells. Vary angles (pitch/yaw/roll) in each cell.")
    logger.info("Auto-captures when board detected in both cameras. %.1fs cooldown between captures.", args.cooldown)
    logger.info("Ctrl+C to stop.\n")
    try:
        while True:
            # Stop condition
            if cell_target is not None:
                if np.all(coverage[grid_mask > 0] >= cell_target[grid_mask > 0]):
                    break
            elif count >= args.count and np.count_nonzero(coverage * grid_mask) >= valid_cells:
                break
            frame_l, frame_r = cap.read()

            # Detect corners
            corners_l, ids_l = detect_charuco_corners(frame_l, board)
            corners_r, ids_r = detect_charuco_corners(frame_r, board)

            # Build preview (resize for HTTP)
            vis_l = cv2.resize(frame_l, (648, 486))
            vis_r = cv2.resize(frame_r, (648, 486))
            scale_x = 648 / frame_l.shape[1]
            scale_y = 486 / frame_l.shape[0]

            detected = False
            n_common = 0

            if corners_l is not None and ids_l is not None:
                # Draw corners on left preview
                scaled_corners_l = corners_l.copy()
                scaled_corners_l[:, 0, 0] *= scale_x
                scaled_corners_l[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_l, scaled_corners_l, ids_l, (0, 255, 0))

            if corners_r is not None and ids_r is not None:
                scaled_corners_r = corners_r.copy()
                scaled_corners_r[:, 0, 0] *= scale_x
                scaled_corners_r[:, 0, 1] *= scale_y
                cv2.aruco.drawDetectedCornersCharuco(vis_r, scaled_corners_r, ids_r, (0, 255, 0))

            if corners_l is not None and corners_r is not None and ids_l is not None and ids_r is not None:
                n_common = len(np.intersect1d(ids_l.flatten(), ids_r.flatten()))
                detected = n_common >= 8

            # Draw coverage grid on left preview
            _draw_coverage(vis_l, coverage, grid_mask, cell_target)

            # Status text
            color = (0, 255, 0) if detected else (0, 0, 255)
            status = f"Pair {count}/{total_target} | Common: {n_common}"
            cv2.putText(vis_l, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            covered_cells = int(np.count_nonzero(coverage * grid_mask))
            coverage_pct = int(covered_cells / valid_cells * 100)
            cov_color = (0, 255, 0) if covered_cells == valid_cells else (0, 200, 255)
            cv2.putText(vis_r, f"Coverage: {covered_cells}/{valid_cells} ({coverage_pct}%)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cov_color, 2)

            if detected:
                cv2.putText(vis_r, "BOARD DETECTED", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            combined = np.hstack([vis_l, vis_r])
            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _update_preview(jpeg.tobytes())

            # Auto-capture when board detected, with cooldown
            now = time.time()
            if detected and (now - last_capture_time) >= args.cooldown:
                # Check if this cell is already full
                row, col = _compute_coverage_center(
                    corners_l, frame_l.shape[1], frame_l.shape[0], grid_mask,
                )
                cell_max = cell_target[row, col] if cell_target is not None else 0
                if cell_max > 0 and coverage[row, col] >= cell_max:
                    # Cell full — skip but show feedback
                    cv2.putText(vis_r, f"Cell ({row},{col}) full ({cell_max}/{cell_max})", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    left_path = output_dir / f"left_{count:03d}.png"
                    right_path = output_dir / f"right_{count:03d}.png"
                    cv2.imwrite(str(left_path), frame_l)
                    cv2.imwrite(str(right_path), frame_r)

                    coverage[row, col] += 1
                    count += 1
                    last_capture_time = now
                    cell_limit = cell_target[row, col] if cell_target is not None else "∞"
                    logger.info(
                        "Pair %d/%d saved — %d common corners, coverage %d%%, cell (%d,%d): %d/%s",
                        count, total_target, n_common, coverage_pct,
                        row, col, coverage[row, col], cell_limit,
                    )

            remaining = f" | Missing {valid_cells - covered_cells} cells!" if count >= total_target and covered_cells < valid_cells else ""
            print(
                f"\r  Pairs: {count}/{total_target} | Common: {n_common:2d} | Coverage: {covered_cells}/{valid_cells}{remaining}   ",
                end="", flush=True,
            )
            time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    _shutting_down = True
    cap.close()
    print(f"\n\nCaptured {count} pairs in {output_dir}")
    print(f"Coverage: {int(np.count_nonzero(coverage * grid_mask))}/{valid_cells} cells")
    print("\nCoverage grid:")
    print(coverage)
    import os
    os._exit(0)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run stereo calibration from captured image pairs."""
    input_dir = Path(args.input_dir)

    # Find all left_*.png files and match with right_*.png
    left_files = sorted(input_dir.glob("left_*.png"))
    if not left_files:
        logger.error("No left_*.png files found in %s", input_dir)
        sys.exit(1)

    pairs = []
    for lf in left_files:
        rf = lf.parent / lf.name.replace("left_", "right_")
        if rf.exists():
            img_l = cv2.imread(str(lf))
            img_r = cv2.imread(str(rf))
            if img_l is not None and img_r is not None:
                pairs.append((img_l, img_r))
            else:
                logger.warning("Failed to read pair: %s", lf.stem)
        else:
            logger.warning("Missing right image for %s", lf.name)

    logger.info("Loaded %d image pairs from %s", len(pairs), input_dir)

    try:
        result = calibrate_stereo(
            pairs,
            board_size=(args.columns, args.rows),
            square_length=args.square_length,
            marker_length=args.marker_length,
        )
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        sys.exit(1)

    save_calibration(result, args.output)
    logger.info("Calibration saved to %s", args.output)

    # Print summary
    fx = result["camera_matrix_l"][0, 0]
    fy = result["camera_matrix_l"][1, 1]
    tx = result["T"][0, 0]
    logger.info("Left focal: fx=%.1f fy=%.1f px", fx, fy)
    logger.info("Baseline (T_x): %.1f mm", abs(tx))


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify calibration by showing rectified pairs with epipolar lines."""
    cal = load_calibration(args.calibration)
    input_dir = Path(args.input_dir)

    left_files = sorted(input_dir.glob("left_*.png"))
    if not left_files:
        logger.error("No images found in %s", input_dir)
        sys.exit(1)

    # Use first pair for verification
    lf = left_files[0]
    rf = lf.parent / lf.name.replace("left_", "right_")
    img_l = cv2.imread(str(lf))
    img_r = cv2.imread(str(rf))

    rect_l, rect_r = rectify_pair(img_l, img_r, cal)

    # Draw horizontal epipolar lines on the rectified pair
    combined = np.hstack([rect_l, rect_r])
    h = combined.shape[0]
    for y in range(0, h, 30):
        color = (0, 255, 0) if (y // 30) % 2 == 0 else (0, 200, 255)
        cv2.line(combined, (0, y), (combined.shape[1], y), color, 1)

    output_path = str(Path(args.calibration).parent / "verify_epipolar.png")
    cv2.imwrite(output_path, combined)
    logger.info("Verification image saved to %s", output_path)
    logger.info(
        "Check that corresponding features lie on the same horizontal line."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stereo calibration tool for People Counter"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate-board ---
    p_board = sub.add_parser("generate-board", help="Generate printable ChArUco board")
    p_board.add_argument("--output", default="charuco_board.png")
    p_board.add_argument("--columns", type=int, default=7)
    p_board.add_argument("--rows", type=int, default=5)
    p_board.add_argument("--square-length", type=float, default=35.0)
    p_board.add_argument("--marker-length", type=float, default=26.0)
    p_board.add_argument("--width", type=int, default=2480, help="Image width (px)")
    p_board.add_argument("--height", type=int, default=3508, help="Image height (px)")
    p_board.set_defaults(func=cmd_generate_board)

    # --- capture ---
    p_cap = sub.add_parser("capture", help="Interactive capture with HTTP preview")
    p_cap.add_argument("--left", type=int, default=0, help="Left camera index")
    p_cap.add_argument("--right", type=int, default=1, help="Right camera index")
    p_cap.add_argument("--resolution", type=int, nargs=2, default=[2592, 1944])
    p_cap.add_argument("--fps", type=int, default=5)
    p_cap.add_argument("--output-dir", default="./calibration/captures")
    p_cap.add_argument("--count", type=int, default=30, help="Minimum number of pairs")
    p_cap.add_argument("--per-cell", type=int, default=0,
                        help="Max captures per grid cell (0=unlimited). Stops cell when reached.")
    p_cap.add_argument("--cooldown", type=float, default=1.5,
                        help="Seconds to wait after each capture before next one")
    p_cap.add_argument("--port", type=int, default=8080, help="HTTP preview port")
    p_cap.add_argument("--columns", type=int, default=7)
    p_cap.add_argument("--rows", type=int, default=5)
    p_cap.add_argument("--square-length", type=float, default=35.0)
    p_cap.add_argument("--marker-length", type=float, default=26.0)
    p_cap.add_argument("--grid", choices=["rectangular", "circular", "170"], default="rectangular",
                        help="Coverage grid: rectangular (4x5), circular (6x6), 170 (6x6 weighted for 170° fisheye)")
    p_cap.set_defaults(func=cmd_capture)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Run stereo calibration")
    p_cal.add_argument("--input-dir", required=True, help="Dir with left_/right_ images")
    p_cal.add_argument("--output", default="calibration.npz")
    p_cal.add_argument("--columns", type=int, default=7)
    p_cal.add_argument("--rows", type=int, default=5)
    p_cal.add_argument("--square-length", type=float, default=35.0)
    p_cal.add_argument("--marker-length", type=float, default=26.0)
    p_cal.set_defaults(func=cmd_calibrate)

    # --- verify ---
    p_ver = sub.add_parser("verify", help="Verify calibration visually")
    p_ver.add_argument("--calibration", required=True, help="Path to calibration.npz")
    p_ver.add_argument("--input-dir", required=True, help="Dir with image pairs")
    p_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

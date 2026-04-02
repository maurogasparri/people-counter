#!/usr/bin/env python3
"""CLI tool for stereo camera calibration using ChArUco patterns.

Usage:
    # Step 1: Generate a printable ChArUco board (A4)
    python scripts/calibrate.py generate-board --output board.png

    # Step 2: Capture image pairs (live cameras or from directory)
    python scripts/calibrate.py capture --left 0 --right 1 --output-dir ./captures --count 30

    # Step 3: Run calibration from captured pairs
    python scripts/calibrate.py calibrate --input-dir ./captures --output calibration.npz

    # Step 4: Verify calibration (draw epipolar lines on rectified pair)
    python scripts/calibrate.py verify --calibration calibration.npz --input-dir ./captures
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.calibration import (
    calibrate_stereo,
    create_charuco_board,
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
    """Capture stereo image pairs from live cameras."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)

    if not cap_l.isOpened() or not cap_r.isOpened():
        logger.error("Failed to open cameras (left=%d, right=%d)", args.left, args.right)
        sys.exit(1)

    count = 0
    logger.info(
        "Press SPACE to capture, Q to quit. Need %d pairs.", args.count
    )

    while count < args.count:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            logger.warning("Frame read failed, retrying...")
            continue

        # Show live preview side by side
        h, w = frame_l.shape[:2]
        preview = np.hstack([
            cv2.resize(frame_l, (w // 2, h // 2)),
            cv2.resize(frame_r, (w // 2, h // 2)),
        ])
        cv2.putText(
            preview, f"Captured: {count}/{args.count} | SPACE=capture Q=quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("Stereo Capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            left_path = output_dir / f"left_{count:03d}.png"
            right_path = output_dir / f"right_{count:03d}.png"
            cv2.imwrite(str(left_path), frame_l)
            cv2.imwrite(str(right_path), frame_r)
            logger.info("Pair %d saved", count)
            count += 1

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    logger.info("Captured %d pairs in %s", count, output_dir)


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
    p_board.add_argument("--square-length", type=float, default=30.0)
    p_board.add_argument("--marker-length", type=float, default=22.5)
    p_board.add_argument("--width", type=int, default=2480, help="Image width (px)")
    p_board.add_argument("--height", type=int, default=3508, help="Image height (px)")
    p_board.set_defaults(func=cmd_generate_board)

    # --- capture ---
    p_cap = sub.add_parser("capture", help="Capture stereo image pairs")
    p_cap.add_argument("--left", type=int, default=0, help="Left camera index")
    p_cap.add_argument("--right", type=int, default=1, help="Right camera index")
    p_cap.add_argument("--output-dir", default="./calibration_captures")
    p_cap.add_argument("--count", type=int, default=30, help="Number of pairs")
    p_cap.set_defaults(func=cmd_capture)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Run stereo calibration")
    p_cal.add_argument("--input-dir", required=True, help="Dir with left_/right_ images")
    p_cal.add_argument("--output", default="calibration.npz")
    p_cal.add_argument("--columns", type=int, default=7)
    p_cal.add_argument("--rows", type=int, default=5)
    p_cal.add_argument("--square-length", type=float, default=30.0)
    p_cal.add_argument("--marker-length", type=float, default=22.5)
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

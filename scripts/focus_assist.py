#!/usr/bin/env python3
"""Focus assist tool for OV5647 cameras.

Continuously captures frames and prints a focus score (Laplacian variance).
Higher score = sharper image. Adjust the lens ring until the score peaks.

Also saves the latest frame to /tmp for SCP verification.

Usage:
    # Focus left camera (CAM1)
    PYTHONPATH=. python3 scripts/focus_assist.py --camera 1

    # Focus right camera (CAM0)
    PYTHONPATH=. python3 scripts/focus_assist.py --camera 0

    # Both cameras side by side
    PYTHONPATH=. python3 scripts/focus_assist.py --camera both
"""

import argparse
import sys
import time

import cv2
import numpy as np


def focus_score(frame: np.ndarray) -> float:
    """Compute focus score using Laplacian variance.

    Higher value = sharper image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def run_single(cam_id: int) -> None:
    from picamera2 import Picamera2

    cam = Picamera2(cam_id)
    config = cam.create_still_configuration(
        main={"size": (1296, 972), "format": "BGR888"},
    )
    cam.configure(config)
    cam.start()
    time.sleep(1)  # let auto-exposure settle

    name = "left" if cam_id == 1 else "right"
    print(f"Focusing camera {cam_id} ({name}). Adjust lens ring and watch the score.")
    print(f"Latest frame saved to /tmp/focus_{name}.jpg — Ctrl+C to stop.\n")

    best = 0.0
    try:
        while True:
            frame = cam.capture_array("main")
            score = focus_score(frame)
            if score > best:
                best = score
            bar = "#" * min(int(score / 10), 50)
            print(f"\r  cam{cam_id} ({name}): {score:8.1f}  best: {best:8.1f}  [{bar:<50s}]", end="", flush=True)
            cv2.imwrite(f"/tmp/focus_{name}.jpg", frame)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n\nFinal best score: {best:.1f}")
    finally:
        cam.stop()
        cam.close()


def run_both() -> None:
    from picamera2 import Picamera2

    cam_l = Picamera2(1)
    cam_r = Picamera2(0)

    for cam in [cam_l, cam_r]:
        config = cam.create_still_configuration(
            main={"size": (1296, 972), "format": "BGR888"},
        )
        cam.configure(config)
        cam.start()

    time.sleep(1)
    print("Focusing both cameras. Adjust lens rings and watch the scores.")
    print("Latest frames saved to /tmp/focus_left.jpg and /tmp/focus_right.jpg")
    print("Ctrl+C to stop.\n")

    best_l = 0.0
    best_r = 0.0
    try:
        while True:
            frame_l = cam_l.capture_array("main")
            frame_r = cam_r.capture_array("main")
            score_l = focus_score(frame_l)
            score_r = focus_score(frame_r)
            if score_l > best_l:
                best_l = score_l
            if score_r > best_r:
                best_r = score_r
            print(
                f"\r  LEFT: {score_l:8.1f} (best {best_l:8.1f})  |  RIGHT: {score_r:8.1f} (best {best_r:8.1f})",
                end="", flush=True,
            )
            cv2.imwrite("/tmp/focus_left.jpg", frame_l)
            cv2.imwrite("/tmp/focus_right.jpg", frame_r)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n\nBest — LEFT: {best_l:.1f}  RIGHT: {best_r:.1f}")
    finally:
        cam_l.stop()
        cam_l.close()
        cam_r.stop()
        cam_r.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Focus assist for OV5647 cameras")
    parser.add_argument(
        "--camera",
        default="both",
        help="Camera to focus: 0 (right), 1 (left), or both",
    )
    args = parser.parse_args()

    if args.camera == "both":
        run_both()
    else:
        run_single(int(args.camera))


if __name__ == "__main__":
    main()

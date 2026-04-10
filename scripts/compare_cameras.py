#!/usr/bin/env python3
"""Side-by-side camera comparison tool.

Mounts two cameras and shows a realtime comparison via HTTP with metrics
to determine viability and NoIR status.

Metrics per camera:
  - Sensor model and FOV label
  - Focus score (Laplacian variance, center 25%)
  - Color analysis: mean B/G/R channels + violet tint indicator
  - ChArUco corner detection count (if board visible)

NoIR detection: cameras without IR cut filter show elevated blue
relative to green under normal indoor lighting. A B/G ratio > 1.15
is a strong NoIR indicator.

Usage:
    PYTHONPATH=. python3 scripts/compare_cameras.py

    Open: http://people-counter.local:8080

    Press Ctrl+C to stop and save comparison snapshots.
"""

import argparse
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

from src.vision.calibration import create_charuco_board, detect_charuco_corners

latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()
shutting_down = False


def focus_score(frame: np.ndarray) -> float:
    """Laplacian variance on center 25% of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = 3 * h // 8
    crop = gray[m:h - m, 3 * w // 8:w - 3 * w // 8]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def color_analysis(frame: np.ndarray) -> dict:
    """Analyze color channels to detect NoIR (missing IR cut filter).

    NoIR cameras under normal lighting show:
    - Elevated blue channel relative to green
    - Overall magenta/violet tint
    - B/G ratio > ~1.15
    """
    # Use center 50% to avoid vignetting at edges
    h, w = frame.shape[:2]
    crop = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
    b, g, r = cv2.split(crop)
    mean_b = float(np.mean(b))
    mean_g = float(np.mean(g))
    mean_r = float(np.mean(r))
    bg_ratio = mean_b / mean_g if mean_g > 1 else 0
    # NoIR tends to have B/G > 1.15 and R elevated too
    is_noir = bg_ratio > 1.15
    return {
        "mean_b": mean_b,
        "mean_g": mean_g,
        "mean_r": mean_r,
        "bg_ratio": bg_ratio,
        "is_noir": is_noir,
    }


def charuco_count(frame: np.ndarray, board: cv2.aruco.CharucoBoard) -> int:
    """Count ChArUco corners detected in frame. Returns 0 if none."""
    corners, ids = detect_charuco_corners(frame, board)
    if corners is None:
        return 0
    return len(corners)


def annotate_frame(
    frame: np.ndarray,
    label: str,
    score: float,
    color: dict,
    corners: int,
    sensor: str,
) -> np.ndarray:
    """Draw metrics overlay on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Semi-transparent black bar at top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    y = 22
    dy = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)

    cv2.putText(out, f"{label} — {sensor}", (8, y), font, fs, white, 1)
    y += dy
    cv2.putText(out, f"Focus: {score:.0f}", (8, y), font, fs, green, 1)
    y += dy

    noir_text = "NoIR (no IR filter)" if color["is_noir"] else "IR filter OK"
    noir_color = red if color["is_noir"] else green
    cv2.putText(out, f"B/G: {color['bg_ratio']:.2f} — {noir_text}", (8, y), font, fs, noir_color, 1)
    y += dy

    cv2.putText(
        out,
        f"B:{color['mean_b']:.0f} G:{color['mean_g']:.0f} R:{color['mean_r']:.0f}",
        (8, y), font, fs, white, 1,
    )
    y += dy

    if corners > 0:
        cv2.putText(out, f"ChArUco: {corners} corners", (8, y), font, fs, green, 1)
    else:
        cv2.putText(out, "ChArUco: not detected", (8, y), font, fs, yellow, 1)

    return out


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Camera Comparison</title>
<style>
body{background:#111;margin:0;display:flex;justify-content:center;
align-items:center;height:100vh;flex-direction:column}
img{max-width:100%;max-height:90vh}
p{color:#888;font-family:monospace;margin:4px}
</style>
</head><body>
<p>CAM0 (left) vs CAM1 (right) &mdash; refresh: ~3 fps</p>
<img src="/stream">
</body></html>""")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not shutting_down:
                    with jpeg_lock:
                        frame = latest_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def get_sensor_model(cam) -> str:
    """Try to get sensor model from picamera2 properties."""
    try:
        props = cam.camera_properties
        model = props.get("Model", "unknown")
        return model
    except Exception:
        return "unknown"


def main() -> None:
    global latest_jpeg, shutting_down

    parser = argparse.ArgumentParser(description="Compare two cameras side-by-side")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--resolution", default="1296x972",
                        help="Capture resolution WxH (default: 1296x972)")
    parser.add_argument("--cam0-label", default="CAM0", help="Label for camera 0")
    parser.add_argument("--cam1-label", default="CAM1", help="Label for camera 1")
    args = parser.parse_args()

    w, h = [int(x) for x in args.resolution.split("x")]

    from picamera2 import Picamera2

    cam0 = Picamera2(0)
    cam1 = Picamera2(1)
    sensor0 = get_sensor_model(cam0)
    sensor1 = get_sensor_model(cam1)

    for cam in [cam0, cam1]:
        config = cam.create_still_configuration(
            main={"size": (w, h), "format": "BGR888"},
        )
        cam.configure(config)
        cam.start()
    time.sleep(1)

    board = create_charuco_board()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Camera comparison — http://people-counter.local:{args.port}")
    print(f"  CAM0: {sensor0}")
    print(f"  CAM1: {sensor1}")
    print("Ctrl+C to stop and save snapshots.\n")

    try:
        while True:
            frame0 = cam0.capture_array("main")
            frame1 = cam1.capture_array("main")

            score0 = focus_score(frame0)
            score1 = focus_score(frame1)
            color0 = color_analysis(frame0)
            color1 = color_analysis(frame1)
            corners0 = charuco_count(frame0, board)
            corners1 = charuco_count(frame1, board)

            ann0 = annotate_frame(frame0, args.cam0_label, score0, color0, corners0, sensor0)
            ann1 = annotate_frame(frame1, args.cam1_label, score1, color1, corners1, sensor1)

            # Scale to same height before combining
            combined = np.hstack([ann0, ann1])

            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            # Terminal output
            noir0 = "NoIR!" if color0["is_noir"] else "IR OK"
            noir1 = "NoIR!" if color1["is_noir"] else "IR OK"
            print(
                f"\r  {args.cam0_label}: focus={score0:7.0f} B/G={color0['bg_ratio']:.2f} {noir0:5s} corners={corners0:2d}"
                f"  |  {args.cam1_label}: focus={score1:7.0f} B/G={color1['bg_ratio']:.2f} {noir1:5s} corners={corners1:2d}",
                end="", flush=True,
            )
            time.sleep(0.3)

    except KeyboardInterrupt:
        shutting_down = True
        print("\n\nSaving comparison snapshots...")
        f0 = cam0.capture_array("main")
        f1 = cam1.capture_array("main")
        cv2.imwrite("/tmp/compare_cam0.jpg", f0)
        cv2.imwrite("/tmp/compare_cam1.jpg", f1)
        # Annotated side-by-side
        a0 = annotate_frame(f0, args.cam0_label, focus_score(f0),
                            color_analysis(f0), charuco_count(f0, board), sensor0)
        a1 = annotate_frame(f1, args.cam1_label, focus_score(f1),
                            color_analysis(f1), charuco_count(f1, board), sensor1)
        cv2.imwrite("/tmp/compare_sidebyside.jpg", np.hstack([a0, a1]))
        print("Saved:")
        print("  /tmp/compare_cam0.jpg")
        print("  /tmp/compare_cam1.jpg")
        print("  /tmp/compare_sidebyside.jpg")

        # Print final summary
        c0 = color_analysis(f0)
        c1 = color_analysis(f1)
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {args.cam0_label} ({sensor0}):")
        print(f"    B/G ratio: {c0['bg_ratio']:.3f} — {'NoIR (no IR filter)' if c0['is_noir'] else 'Has IR filter'}")
        print(f"    Focus: {focus_score(f0):.0f}")
        print(f"    BGR: {c0['mean_b']:.0f} / {c0['mean_g']:.0f} / {c0['mean_r']:.0f}")
        print(f"  {args.cam1_label} ({sensor1}):")
        print(f"    B/G ratio: {c1['bg_ratio']:.3f} — {'NoIR (no IR filter)' if c1['is_noir'] else 'Has IR filter'}")
        print(f"    Focus: {focus_score(f1):.0f}")
        print(f"    BGR: {c1['mean_b']:.0f} / {c1['mean_g']:.0f} / {c1['mean_r']:.0f}")
        print()
        if c0["is_noir"] != c1["is_noir"]:
            print("  One camera has IR filter, the other doesn't.")
        elif c0["is_noir"] and c1["is_noir"]:
            print("  Both cameras appear to be NoIR (no IR filter).")
        else:
            print("  Both cameras appear to have IR filters.")
        print(f"{'=' * 60}")

        cam0.stop(); cam0.close()
        cam1.stop(); cam1.close()
        import os
        os._exit(0)


if __name__ == "__main__":
    main()

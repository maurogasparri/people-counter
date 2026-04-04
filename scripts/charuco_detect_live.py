#!/usr/bin/env python3
"""Live ChArUco detection preview.

Continuously captures frames, detects ChArUco corners, and serves
a live preview via HTTP showing detected corners and scores.

Usage:
    PYTHONPATH=. python3 scripts/charuco_detect_live.py

    Then open: http://people-counter.local:8080
"""

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

from src.vision.calibration import create_charuco_board, detect_charuco_corners

latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()
shutting_down = False


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>ChArUco Detection</title>
<style>body{background:#111;margin:0;display:flex;justify-content:center;
align-items:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/stream"></body></html>""")
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
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def main() -> None:
    global latest_jpeg, shutting_down

    from picamera2 import Picamera2

    cam_l = Picamera2(0)
    cam_r = Picamera2(1)
    for cam in [cam_l, cam_r]:
        config = cam.create_still_configuration(
            main={"size": (2592, 1944), "format": "BGR888"},
        )
        cam.configure(config)
        cam.start()
    time.sleep(1)

    board = create_charuco_board()

    server = HTTPServer(("0.0.0.0", 8080), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("ChArUco detection — live preview: http://people-counter.local:8080")
    print("Move the ChArUco board and watch corners detected. Ctrl+C to stop.\n")

    best_common = 0
    try:
        while True:
            frame_l = cam_l.capture_array("main")
            frame_r = cam_r.capture_array("main")

            corners_l, ids_l = detect_charuco_corners(frame_l, board)
            corners_r, ids_r = detect_charuco_corners(frame_r, board)

            n_l = len(ids_l) if ids_l is not None else 0
            n_r = len(ids_r) if ids_r is not None else 0

            if ids_l is not None and ids_r is not None:
                common = len(np.intersect1d(ids_l.flatten(), ids_r.flatten()))
            else:
                common = 0

            if common > best_common:
                best_common = common

            # Draw detected corners on images
            vis_l = frame_l.copy()
            vis_r = frame_r.copy()

            if corners_l is not None and ids_l is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis_l, corners_l, ids_l, (0, 255, 0))
            if corners_r is not None and ids_r is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis_r, corners_r, ids_r, (0, 255, 0))

            # Resize for preview
            vis_l = cv2.resize(vis_l, (648, 486))
            vis_r = cv2.resize(vis_r, (648, 486))

            # Labels
            color = (0, 255, 0) if common >= 6 else (0, 0, 255)
            cv2.putText(vis_l, f"LEFT: {n_l} corners", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_r, f"RIGHT: {n_r} corners", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_l, f"Common: {common} (need 6+)", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if common >= 6:
                cv2.putText(vis_r, "CALIBRATION VIABLE", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            combined = np.hstack([vis_l, vis_r])
            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            print(
                f"\r  LEFT: {n_l:2d} corners  |  RIGHT: {n_r:2d} corners  |  Common: {common:2d} (best {best_common:2d})",
                end="", flush=True,
            )
            time.sleep(0.5)

    except KeyboardInterrupt:
        shutting_down = True
        print(f"\n\nBest common corners: {best_common}")
        if best_common >= 6:
            print("Ready to calibrate!")
        else:
            print("Need >= 6 common corners. Try better lighting or closer distance.")
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        import os
        os._exit(0)


if __name__ == "__main__":
    main()

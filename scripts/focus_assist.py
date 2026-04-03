#!/usr/bin/env python3
"""Focus assist tool for OV5647 cameras.

Continuously captures frames, prints a focus score (Laplacian variance),
and serves a live preview via HTTP for headless devices.

Usage:
    PYTHONPATH=. python3 scripts/focus_assist.py

    Then open: http://people-counter.local:8080

    Adjust the lens rings while watching the score in the terminal
    and the live image in the browser. Ctrl+C to stop and save final frames.
"""

import argparse
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()
shutting_down = False


def focus_score(frame: np.ndarray) -> float:
    """Compute focus score using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Focus Assist</title>
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
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def main() -> None:
    global latest_jpeg

    parser = argparse.ArgumentParser(description="Focus assist with live HTTP preview")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    from picamera2 import Picamera2

    cam_l = Picamera2(1)
    cam_r = Picamera2(0)
    for cam in [cam_l, cam_r]:
        config = cam.create_still_configuration(
            main={"size": (648, 486), "format": "BGR888"},
        )
        cam.configure(config)
        cam.start()
    time.sleep(1)

    # Start HTTP server in background
    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Focus assist — live preview: http://people-counter.local:{args.port}")
    print("Adjust lens rings and watch the scores. Ctrl+C to stop.\n")

    best_l = 0.0
    best_r = 0.0
    try:
        while True:
            frame_l = cam_l.capture_array("main")
            frame_r = cam_r.capture_array("main")
            # Score only center 50% to ignore fingers adjusting the lens
            h, w = frame_l.shape[:2]
            cy1, cy2 = h // 4, 3 * h // 4
            cx1, cx2 = w // 4, 3 * w // 4
            score_l = focus_score(frame_l[cy1:cy2, cx1:cx2])
            score_r = focus_score(frame_r[cy1:cy2, cx1:cx2])
            if score_l > best_l:
                best_l = score_l
            if score_r > best_r:
                best_r = score_r

            # Draw rectangle showing the scored region on full image
            cv2.rectangle(frame_l, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.rectangle(frame_r, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.putText(frame_l, f"LEFT  score:{score_l:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_r, f"RIGHT score:{score_r:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            combined = np.hstack([frame_l, frame_r])

            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            print(
                f"\r  LEFT: {score_l:8.1f} (best {best_l:8.1f})  |  RIGHT: {score_r:8.1f} (best {best_r:8.1f})",
                end="", flush=True,
            )
            time.sleep(0.3)

    except KeyboardInterrupt:
        shutting_down = True
        print(f"\n\nBest — LEFT: {best_l:.1f}  RIGHT: {best_r:.1f}")
        print("Saving final frames...")
        cv2.imwrite("/tmp/focus_left.jpg", cam_l.capture_array("main"))
        cv2.imwrite("/tmp/focus_right.jpg", cam_r.capture_array("main"))
        print("Saved to /tmp/focus_left.jpg and /tmp/focus_right.jpg")
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        import os
        os._exit(0)


if __name__ == "__main__":
    main()

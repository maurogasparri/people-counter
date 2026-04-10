#!/usr/bin/env python3
"""Focus assist tool for Arducam IMX708 cameras.

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


def focus_grid(frame: np.ndarray, rows: int = 3, cols: int = 3) -> np.ndarray:
    """Compute focus score for each cell in a grid.

    Returns array of shape (rows, cols) with Laplacian variance per cell.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols
            scores[r, c] = cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F).var()
    return scores


def draw_focus_grid(frame: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Overlay focus grid scores on frame with color coding."""
    out = frame.copy()
    rows, cols = scores.shape
    h, w = frame.shape[:2]
    max_score = scores.max() if scores.max() > 0 else 1.0

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols

            # Color: green (good) to red (bad) based on relative score
            ratio = scores[r, c] / max_score
            green = int(ratio * 255)
            red = int((1 - ratio) * 255)
            color = (0, green, red)

            # Draw cell border
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Draw score text
            text = f"{scores[r, c]:.0f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(out, text, (tx, ty), font, scale, (0, 0, 0), thick + 2)
            cv2.putText(out, text, (tx, ty), font, scale, color, thick)

    return out


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
    parser.add_argument("--no-zoom", action="store_true",
                        help="Show full frame instead of zoomed center")
    parser.add_argument("--grid", action="store_true",
                        help="Show 3x3 focus score grid overlay")
    args = parser.parse_args()

    from picamera2 import Picamera2

    cam_l = Picamera2(0)
    cam_r = Picamera2(1)
    for cam in [cam_l, cam_r]:
        config = cam.create_still_configuration(
            main={"size": (1296, 972), "format": "BGR888"},
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
            frame_l = cv2.cvtColor(cam_l.capture_array("main"), cv2.COLOR_RGB2BGR)
            frame_r = cv2.cvtColor(cam_r.capture_array("main"), cv2.COLOR_RGB2BGR)
            # Score center 25% only (ignores fingers on the lens edges)
            h, w = frame_l.shape[:2]
            margin = 3 * h // 8  # 37.5% margin each side = 25% center
            cy1, cy2 = margin, h - margin
            cx1, cx2 = 3 * w // 8, w - 3 * w // 8
            score_l = focus_score(frame_l[cy1:cy2, cx1:cx2])
            score_r = focus_score(frame_r[cy1:cy2, cx1:cx2])
            if score_l > best_l:
                best_l = score_l
            if score_r > best_r:
                best_r = score_r

            # Preview: zoomed center (default) or full frame
            if args.no_zoom:
                preview_l = frame_l.copy()
                preview_r = frame_r.copy()
            else:
                preview_l = cv2.resize(frame_l[cy1:cy2, cx1:cx2], (w, h))
                preview_r = cv2.resize(frame_r[cy1:cy2, cx1:cx2], (w, h))

            if args.grid:
                preview_l = draw_focus_grid(preview_l, focus_grid(preview_l))
                preview_r = draw_focus_grid(preview_r, focus_grid(preview_r))

            cv2.putText(preview_l, f"LEFT  score:{score_l:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview_r, f"RIGHT score:{score_r:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            combined = np.hstack([preview_l, preview_r])

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
        cv2.imwrite("/tmp/focus_left.jpg", cv2.cvtColor(cam_l.capture_array("main"), cv2.COLOR_RGB2BGR))
        cv2.imwrite("/tmp/focus_right.jpg", cv2.cvtColor(cam_r.capture_array("main"), cv2.COLOR_RGB2BGR))
        print("Saved to /tmp/focus_left.jpg and /tmp/focus_right.jpg")
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        import os
        os._exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Focus assist tool for stereo Arducam IMX708 cameras.

Guided focus tool that validates sharpness across 9 zones per camera,
checks L/R symmetry, and gives a clear PASS/FAIL verdict.

Usage:
    PYTHONPATH=. python3 scripts/focus_assist.py

    Then open: http://people-counter.local:8080

    Place a textured target (newspaper, poster) at 2.5-3m. Adjust each
    lens ring while watching the live feedback. The tool tells you when
    both cameras are balanced and ready for calibration.
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

MIN_SCORE = 200
MIN_EDGE_CENTER_RATIO = 0.25
MAX_LR_DIFF_PCT = 15.0
MAX_LR_ZONE_DIFF_PCT = 30.0


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def focus_grid(frame: np.ndarray, rows: int = 3, cols: int = 3) -> np.ndarray:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols
            scores[r, c] = cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F).var()
    return scores


def evaluate_focus(grid_l: np.ndarray, grid_r: np.ndarray) -> dict:
    """Evaluate focus quality across both cameras.

    Returns dict with metrics, per-check pass/fail, verdict, and guidance message.
    """
    center_l, center_r = grid_l[1, 1], grid_r[1, 1]
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    edges_l = np.mean([grid_l[r, c] for r, c in corners])
    edges_r = np.mean([grid_r[r, c] for r, c in corners])

    ec_l = edges_l / center_l if center_l > 0 else 0
    ec_r = edges_r / center_r if center_r > 0 else 0

    global_l = grid_l.mean()
    global_r = grid_r.mean()
    lr_diff = abs(global_l - global_r) / max(global_l, global_r, 1) * 100

    zone_diffs = np.abs(grid_l - grid_r) / np.maximum(grid_l, grid_r).clip(1) * 100
    max_zone_diff = zone_diffs.max()
    worst_zone = np.unravel_index(zone_diffs.argmax(), zone_diffs.shape)

    checks = {}
    checks["center_l"] = center_l >= MIN_SCORE
    checks["center_r"] = center_r >= MIN_SCORE
    checks["uniformity_l"] = ec_l >= MIN_EDGE_CENTER_RATIO
    checks["uniformity_r"] = ec_r >= MIN_EDGE_CENTER_RATIO
    checks["lr_global"] = lr_diff <= MAX_LR_DIFF_PCT
    checks["lr_zones"] = max_zone_diff <= MAX_LR_ZONE_DIFF_PCT

    all_pass = all(checks.values())

    guidance = []
    if not checks["center_l"]:
        guidance.append("LEFT center too low — adjust left lens")
    if not checks["center_r"]:
        guidance.append("RIGHT center too low — adjust right lens")
    if checks["center_l"] and not checks["uniformity_l"]:
        guidance.append(f"LEFT edges weak (edge/center {ec_l:.2f}) — back off slightly from peak")
    if checks["center_r"] and not checks["uniformity_r"]:
        guidance.append(f"RIGHT edges weak (edge/center {ec_r:.2f}) — back off slightly from peak")
    if not checks["lr_global"]:
        side = "LEFT" if global_l < global_r else "RIGHT"
        guidance.append(f"{side} is weaker overall (diff {lr_diff:.0f}%) — adjust {side.lower()} lens")
    if checks["lr_global"] and not checks["lr_zones"]:
        zone_names = {(0,0): "top-left", (0,2): "top-right", (2,0): "bot-left", (2,2): "bot-right",
                      (0,1): "top-mid", (1,0): "mid-left", (1,2): "mid-right", (2,1): "bot-mid", (1,1): "center"}
        zname = zone_names.get(worst_zone, f"({worst_zone[0]},{worst_zone[1]})")
        guidance.append(f"L/R mismatch at {zname} ({max_zone_diff:.0f}%) — possible tilt or mount issue")
    if all_pass:
        guidance = ["READY — lock the lenses and proceed to calibration"]

    return {
        "center_l": center_l, "center_r": center_r,
        "ec_l": ec_l, "ec_r": ec_r,
        "lr_diff": lr_diff,
        "max_zone_diff": max_zone_diff, "worst_zone": worst_zone,
        "checks": checks, "all_pass": all_pass,
        "guidance": guidance,
    }


def draw_focus_overlay(frame: np.ndarray, grid: np.ndarray, label: str,
                       eval_result: dict, side: str) -> np.ndarray:
    """Draw focus grid with pass/fail coloring and status banner."""
    out = frame.copy()
    rows, cols = grid.shape
    h, w = frame.shape[:2]
    max_score = max(grid.max(), 1.0)

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols

            ratio = grid[r, c] / max_score
            green = int(ratio * 255)
            red = int((1 - ratio) * 255)
            color = (0, green, red)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            text = f"{grid[r, c]:.0f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(out, text, (tx, ty), font, scale, (0, 0, 0), thick + 2)
            cv2.putText(out, text, (tx, ty), font, scale, color, thick)

    ec = eval_result[f"ec_{side}"]
    center = eval_result[f"center_{side}"]
    center_ok = eval_result["checks"][f"center_{side}"]
    uniform_ok = eval_result["checks"][f"uniformity_{side}"]

    banner_color = (0, 180, 0) if (center_ok and uniform_ok) else (0, 0, 220)
    cv2.rectangle(out, (0, 0), (w, 32), banner_color, -1)
    status = f"{label}  center:{center:.0f}  edge/center:{ec:.2f}"
    cv2.putText(out, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return out


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Focus Assist</title>
<style>body{background:#111;margin:0;display:flex;flex-direction:column;
justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:90vh}
#status{color:#fff;font-family:monospace;font-size:18px;padding:8px;
text-align:center;min-height:28px}</style>
</head><body><div id="status"></div><img src="/stream">
<script>
setInterval(()=>fetch('/status').then(r=>r.text()).then(t=>{
  document.getElementById('status').innerHTML=t}),500);
</script></body></html>""")
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
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            with _status_lock:
                self.wfile.write(_status_html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


_status_html = ""
_status_lock = threading.Lock()


def main() -> None:
    global latest_jpeg, shutting_down, _status_html

    parser = argparse.ArgumentParser(description="Guided focus assist for stereo cameras")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

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

    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Focus assist — http://people-counter.local:{args.port}")
    print("Place textured target at 2.5-3m. Adjust lens rings.")
    print(f"Targets: center>{MIN_SCORE}, edge/center>{MIN_EDGE_CENTER_RATIO}, "
          f"L/R diff<{MAX_LR_DIFF_PCT}%\n")

    pass_streak = 0
    try:
        while True:
            frame_l = cv2.cvtColor(cam_l.capture_array("main"), cv2.COLOR_RGB2BGR)
            frame_r = cv2.cvtColor(cam_r.capture_array("main"), cv2.COLOR_RGB2BGR)

            grid_l = focus_grid(frame_l)
            grid_r = focus_grid(frame_r)
            ev = evaluate_focus(grid_l, grid_r)

            preview_l = draw_focus_overlay(frame_l, grid_l, "LEFT", ev, "l")
            preview_r = draw_focus_overlay(frame_r, grid_r, "RIGHT", ev, "r")

            # Bottom guidance banner on combined image
            combined = np.hstack([preview_l, preview_r])
            ch, cw = combined.shape[:2]

            if ev["all_pass"]:
                pass_streak += 1
            else:
                pass_streak = 0

            if ev["all_pass"] and pass_streak >= 3:
                banner_color = (0, 200, 0)
                banner_text = "READY — lock the lenses (Loctite/esmalte) and proceed to calibration"
            elif ev["all_pass"]:
                banner_color = (0, 180, 180)
                banner_text = "Looking good — hold steady..."
            else:
                banner_color = (0, 0, 200)
                banner_text = ev["guidance"][0] if ev["guidance"] else "Adjusting..."

            cv2.rectangle(combined, (0, ch - 36), (cw, ch), banner_color, -1)
            cv2.putText(combined, banner_text, (10, ch - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            # Status for web UI overlay
            if ev["all_pass"] and pass_streak >= 3:
                html = '<span style="color:#0f0">READY — lock the lenses and calibrate</span>'
            elif ev["all_pass"]:
                html = '<span style="color:#ff0">Hold steady...</span>'
            else:
                html = '<span style="color:#f44">' + " | ".join(ev["guidance"]) + '</span>'
            with _status_lock:
                _status_html = html

            # Terminal output
            checks_str = " ".join(
                f"{'OK' if v else 'NO'}:{k}" for k, v in ev["checks"].items()
            )
            verdict = "PASS" if ev["all_pass"] else "FAIL"
            print(
                f"\r  [{verdict}] L/R:{ev['lr_diff']:4.1f}% | "
                f"ec_L:{ev['ec_l']:.2f} ec_R:{ev['ec_r']:.2f} | "
                f"ctr_L:{ev['center_l']:6.0f} ctr_R:{ev['center_r']:6.0f} | "
                f"{'READY' if pass_streak >= 3 else ev['guidance'][0] if ev['guidance'] else ''}   ",
                end="", flush=True,
            )
            time.sleep(0.3)

    except KeyboardInterrupt:
        shutting_down = True
        grid_l = focus_grid(frame_l)
        grid_r = focus_grid(frame_r)
        ev = evaluate_focus(grid_l, grid_r)

        print("\n")
        print("=" * 60)
        print("FINAL FOCUS REPORT")
        print("=" * 60)
        print(f"\n  LEFT grid:")
        for r in range(3):
            print(f"    {grid_l[r, 0]:7.0f}  {grid_l[r, 1]:7.0f}  {grid_l[r, 2]:7.0f}")
        print(f"\n  RIGHT grid:")
        for r in range(3):
            print(f"    {grid_r[r, 0]:7.0f}  {grid_r[r, 1]:7.0f}  {grid_r[r, 2]:7.0f}")
        print(f"\n  L/R global diff:    {ev['lr_diff']:.1f}% (max {MAX_LR_DIFF_PCT}%)")
        print(f"  L/R max zone diff:  {ev['max_zone_diff']:.1f}% (max {MAX_LR_ZONE_DIFF_PCT}%)")
        print(f"  Edge/center LEFT:   {ev['ec_l']:.2f} (min {MIN_EDGE_CENTER_RATIO})")
        print(f"  Edge/center RIGHT:  {ev['ec_r']:.2f} (min {MIN_EDGE_CENTER_RATIO})")
        print(f"\n  Verdict: {'PASS' if ev['all_pass'] else 'FAIL'}")
        if not ev["all_pass"]:
            for g in ev["guidance"]:
                print(f"    -> {g}")
        print()

        print("Saving final frames...")
        cv2.imwrite("/tmp/focus_left.jpg", frame_l)
        cv2.imwrite("/tmp/focus_right.jpg", frame_r)
        print("Saved to /tmp/focus_left.jpg and /tmp/focus_right.jpg")

        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        import os
        os._exit(0)


if __name__ == "__main__":
    main()

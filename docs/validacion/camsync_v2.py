#!/usr/bin/env python3
"""Re-test de sync estéreo CON la nueva implementación (camera_sync ON).

Mide el desfase L/R como lo usa el RUNTIME: StereoCapture con camera_sync=True +
async_capture=True (converge-then-hold), luego ≥5min de captura continua. Compara
contra la base (camsync.csv, sin sync). Correr en la Pi con el servicio detenido:
    cd /usr/src/people-counter && python3 camsync_v2.py
Salida: camsync_v2.csv + stats + histograma.
"""
import csv
import statistics as st
import sys
import time

sys.path.insert(0, "/usr/src/people-counter")
from src.config.loader import load_config
from src.main import _runtime_resolution
from src.vision.capture import StereoCapture

DUR = 330  # ≥5 min de captura continua
cfg = load_config("/etc/people-counter/config.yaml")
RES = tuple(_runtime_resolution(cfg))
RAW = tuple(cfg.get("sensor", {}).get("default_res") or (2304, 1296))
vision = cfg.get("vision", {})

cap = StereoCapture(
    cam_left_id=cfg["bracket"]["camera_left_csi"],
    cam_right_id=cfg["bracket"]["camera_right_csi"],
    resolution=RES,
    fps=int(vision.get("fps", 30)),
    max_exposure_us=vision.get("max_exposure_us", 16000),
    sensor_raw_size=RAW,
    camera_sync=True,        # nueva implementación
    async_capture=True,      # igual que el runtime (dispara converge-then-hold)
)
print("open() — corre la fase de convergencia (~10s)...", flush=True)
cap.open()
print(f"convergido (delta inicial ~{cap._last_sync_delta_us}us). "
      f"Capturando {DUR}s continuos...", flush=True)

deltas = []
fh = open("/home/pi/bench/20260621/camsync_v2.csv", "w", newline="")
w = csv.writer(fh)
w.writerow(["ts_l_ns", "ts_r_ns", "delta_ms"])
t_end = time.time() + DUR
last = None
while time.time() < t_end:
    fl, fr, tl, tr, cl, cr = cap.read_with_metadata()
    if tl and tr and (tl, tr) != last:  # evitar contar el mismo slot dos veces
        last = (tl, tr)
        d = abs(tl - tr) / 1e6
        deltas.append(d)
        w.writerow([tl, tr, round(d, 4)])
fh.close()
cap.close()


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)


n = len(deltas)
within5 = sum(1 for d in deltas if d <= 5.0)
episodic = sum(1 for d in deltas if d >= 15.0)  # modo a ~media trama
print(f"\n=== SYNC v2 (n={n} pares, {DUR}s) ===")
print(f"  mediana={st.median(deltas):.3f}ms p95={pct(deltas,.95):.3f}ms "
      f"max={max(deltas):.3f}ms std={st.pstdev(deltas):.3f}ms")
print(f"  % pares <=5ms: {100*within5/n:.2f}%")
print(f"  modo episodico >=15ms: {episodic} ({100*episodic/n:.2f}%)")
print("\n  Histograma (ms):")
bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1), (1, 2),
        (2, 5), (5, 10), (10, 15), (15, 20)]
for lo, hi in bins:
    c = sum(1 for d in deltas if lo <= d < hi)
    print(f"    [{lo:>5}-{hi:<5}) {c:>6} ({100*c/n:5.2f}%) {'#'*int(50*c/n)}")

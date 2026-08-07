#!/usr/bin/env python3
"""A8 — sincronización estéreo: |SensorTimestamp_L - SensorTimestamp_R| por par.

Servicio detenido (cámaras libres). Correr en la Pi:
    cd /usr/src/people-counter && python3 camsync.py
Reporta % de pares dentro de 5 ms + p50/p95/máx. CSV: camsync.csv
"""
import sys
import time
import csv
import statistics as st

sys.path.insert(0, '/usr/src/people-counter')
from src.config.loader import load_config
from src.main import _runtime_resolution, _runtime_fps
from src.vision.capture import StereoCapture

DUR = 300
cfg = load_config('/etc/people-counter/config.yaml')
vision = cfg.get('vision', {})
sensor = cfg.get('sensor', {})
raw = sensor.get('default_res')
kw = dict(
    cam_left_id=cfg['bracket']['camera_left_csi'],
    cam_right_id=cfg['bracket']['camera_right_csi'],
    resolution=_runtime_resolution(cfg),
    fps=_runtime_fps(cfg),
    max_exposure_us=vision.get('max_exposure_us', 16000),
    async_capture=False,
)
if raw:
    kw['sensor_raw_size'] = tuple(raw)
cap = StereoCapture(**kw)
cap.open()
deltas = []
fh = open('/home/pi/bench/20260621/camsync.csv', 'w', newline='')
w = csv.writer(fh)
w.writerow(['ts_l_ns', 'ts_r_ns', 'delta_ms'])
t_end = time.time() + DUR
try:
    while time.time() < t_end:
        fl, fr, tl, tr, cl, cr = cap.read_with_metadata()
        if tl and tr:
            d = abs(tl - tr) / 1e6
            deltas.append(d)
            w.writerow([tl, tr, round(d, 4)])
finally:
    fh.close()
    cap.close()


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)


n = len(deltas)
within = sum(1 for d in deltas if d <= 5.0)
print(f'pares={n} ventana={DUR}s')
print(f'sync L/R: dentro_5ms={within} ({100 * within / n:.2f}%) '
      f'p50={pct(deltas, .5):.3f}ms p95={pct(deltas, .95):.3f}ms '
      f'max={max(deltas):.3f}ms mean={st.mean(deltas):.3f}ms')

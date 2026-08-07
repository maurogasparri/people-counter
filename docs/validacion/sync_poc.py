import sys, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/usr/src/people-counter')
from libcamera import controls
from picamera2 import Picamera2
from src.config.loader import load_config
from src.main import _runtime_resolution

MODE = sys.argv[1] if len(sys.argv) > 1 else 'off'
cfg = load_config('/etc/people-counter/config.yaml')
RES = tuple(_runtime_resolution(cfg))
RAW = tuple(cfg.get('sensor', {}).get('default_res') or (2304, 1296))
FR = 30.0
L = int(cfg['bracket']['camera_left_csi']); R = int(cfg['bracket']['camera_right_csi'])

def make(idx, sm):
    cam = Picamera2(idx)
    ctr = {'FrameRate': FR, 'SyncMode': sm}
    cam.configure(cam.create_preview_configuration(
        main={'size': RES, 'format': 'RGB888'}, raw={'size': RAW}, controls=ctr))
    return cam

E = controls.rpi.SyncModeEnum
if MODE == 'sync':
    client = make(L, E.Client); server = make(R, E.Server)
    client.start(); time.sleep(0.5); server.start()
    camL, camR = client, server
else:
    camL = make(L, E.Off); camR = make(R, E.Off)
    camL.start(); camR.start()
print(f'[{MODE}] convergiendo 12s...', flush=True)
time.sleep(12)
ex = ThreadPoolExecutor(2)
def grab(cam):
    r = cam.capture_request(); m = r.get_metadata()
    ts = m.get('SensorTimestamp', 0); stv = m.get('SyncTimer'); r.release()
    return ts, stv
deltas = []; sync_us = []
end = time.time() + 45
while time.time() < end:
    fl = ex.submit(grab, camL); fr = ex.submit(grab, camR)
    tl, stl = fl.result(); tr, _ = fr.result()
    if tl and tr:
        deltas.append(abs(tl - tr) / 1e6)
        if stl is not None: sync_us.append(abs(stl))
camL.stop(); camR.stop(); camL.close(); camR.close()
def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)
n = len(deltas); within = sum(1 for d in deltas if d <= 5)
sub1 = sum(1 for d in deltas if d <= 1)
print(f'MODE={MODE} pares={n}')
print(f'  L/R delta: <1ms={100*sub1/n:.1f}% <5ms={100*within/n:.1f}% p50={pct(deltas,.5):.3f}ms p95={pct(deltas,.95):.3f}ms max={max(deltas):.3f}ms mean={st.mean(deltas):.3f}ms')
if sync_us:
    print(f'  SyncTimer |dev|: p50={pct(sync_us,.5):.0f}us p95={pct(sync_us,.95):.0f}us max={max(sync_us):.0f}us')

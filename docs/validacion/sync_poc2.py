import sys, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/usr/src/people-counter')
from libcamera import controls
from picamera2 import Picamera2
from src.config.loader import load_config
from src.main import _runtime_resolution

# arg: 'pin' (FDL 16000/16000, blur cap exacto) | 'range' (16000/18000)
MODE = sys.argv[1] if len(sys.argv) > 1 else 'pin'
cfg = load_config('/etc/people-counter/config.yaml')
RES = tuple(_runtime_resolution(cfg)); RAW = tuple(cfg.get('sensor', {}).get('default_res') or (2304, 1296))
FDL = (16000, 16000) if MODE == 'pin' else (16000, 18000)
L = int(cfg['bracket']['camera_left_csi']); R = int(cfg['bracket']['camera_right_csi'])
E = controls.rpi.SyncModeEnum

def make(idx, sm):
    cam = Picamera2(idx)
    cam.configure(cam.create_preview_configuration(
        main={'size': RES, 'format': 'RGB888'}, raw={'size': RAW},
        controls={'FrameDurationLimits': FDL, 'SyncMode': sm}))
    return cam

client = make(L, E.Client); server = make(R, E.Server)
client.start(); time.sleep(0.5); server.start()
print(f'[sync+{MODE} FDL={FDL}] convergiendo 12s...', flush=True)
time.sleep(12)
ex = ThreadPoolExecutor(2)
def grab(cam):
    r = cam.capture_request(); m = r.get_metadata(); ts = m.get('SensorTimestamp', 0)
    et = m.get('ExposureTime'); fd = m.get('FrameDuration'); r.release(); return ts, et, fd
deltas = []; exps = []; fds = []
end = time.time() + 40
while time.time() < end:
    fl = ex.submit(grab, client); fr = ex.submit(grab, server)
    tl, el, dl = fl.result(); tr, er, dr = fr.result()
    if tl and tr:
        deltas.append(abs(tl - tr) / 1e6)
        if el: exps.append(el)
        if dl: fds.append(dl)
client.stop(); server.stop(); client.close(); server.close()
def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)
n = len(deltas); sub1 = sum(1 for d in deltas if d <= 1)
print(f'MODE=sync+{MODE} pares={n}')
print(f'  L/R delta: <1ms={100*sub1/n:.1f}% p50={pct(deltas,.5):.3f}ms p95={pct(deltas,.95):.3f}ms max={max(deltas):.3f}ms')
if exps: print(f'  ExposureTime client: p50={pct(exps,.5):.0f}us max={max(exps):.0f}us (cap blur)')
if fds: print(f'  FrameDuration client: p50={pct(fds,.5):.0f}us min={min(fds):.0f} max={max(fds):.0f}us (rango que usa el sync)')

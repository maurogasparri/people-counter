import sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/usr/src/people-counter')
from libcamera import controls
from picamera2 import Picamera2
from src.config.loader import load_config
from src.main import _runtime_resolution

cfg = load_config('/etc/people-counter/config.yaml')
RES = tuple(_runtime_resolution(cfg)); RAW = tuple(cfg['sensor']['default_res'])
E = controls.rpi.SyncModeEnum
FR = float(sys.argv[1])

client = Picamera2(0); server = Picamera2(1)
def mk(cam, sm):
    return cam.create_video_configuration(
        main={'size':RES,'format':'RGB888'}, raw={'size':RAW},
        controls={'FrameRate':FR,'SyncMode':sm})
client.configure(mk(client,E.Client)); client.start()
time.sleep(0.5)
server.configure(mk(server,E.Server)); server.start()
ex=ThreadPoolExecutor(2)
def grab(c):
    r=c.capture_request(); m=r.get_metadata(); ts=m.get('SensorTimestamp',0); et=m.get('ExposureTime',0); fd=m.get('FrameDuration',0); r.release(); return ts,et,fd
time.sleep(16)
ds=[]; et=fd=0
for _ in range(20):
    fl=ex.submit(grab,client); fr=ex.submit(grab,server)
    a=fl.result(); b=fr.result()
    if a[0] and b[0]: ds.append(abs(a[0]-b[0])/1000.0)
    et=a[1]; fd=a[2]
ds.sort(); print(f'FR={FR}: median={ds[len(ds)//2]:.1f}us max={max(ds):.1f}us exp={et}us framedur={fd}us', flush=True)
client.stop(); server.stop(); client.close(); server.close()

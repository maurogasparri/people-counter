import sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'/usr/src/people-counter')
from libcamera import controls
from picamera2 import Picamera2
from src.config.loader import load_config
from src.main import _runtime_resolution
cfg=load_config('/etc/people-counter/config.yaml')
RES=tuple(_runtime_resolution(cfg)); RAW=tuple(cfg['sensor']['default_res']); E=controls.rpi.SyncModeEnum
client=Picamera2(0); server=Picamera2(1)
def mk(cam,sm): return cam.create_video_configuration(main={'size':RES,'format':'RGB888'},raw={'size':RAW},controls={'SyncMode':sm})
client.configure(mk(client,E.Client)); client.start()
time.sleep(0.4); server.configure(mk(server,E.Server)); server.start()
# leer piso real y setear FrameRate alcanzable con 5% de margen
floor=client.camera_controls.get('FrameDurationLimits')[0]
fr=1_000_000.0/(floor*1.05)
for cam in (client,server): cam.set_controls({'FrameRate':fr})
print(f'floor_us={floor} -> FrameRate={fr:.1f}fps (exp cap ~{floor*1.05/1000:.1f}ms)',flush=True)
ex=ThreadPoolExecutor(2)
def grab(c):
    r=c.capture_request(); m=r.get_metadata(); ts=m.get('SensorTimestamp',0); et=m.get('ExposureTime',0); r.release(); return ts,et
for t in (6,12,18,24):
    time.sleep(6); ds=[]; et=0
    for _ in range(12):
        a=ex.submit(grab,client).result(); b=ex.submit(grab,server).result()
        if a[0] and b[0]: ds.append(abs(a[0]-b[0])/1000.0); et=a[1]
    ds.sort(); print(f'  t={t}s median={ds[len(ds)//2]:.1f}us exp={et}us',flush=True)
client.stop(); server.stop(); client.close(); server.close()

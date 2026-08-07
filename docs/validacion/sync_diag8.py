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
floor=client.camera_controls.get('FrameDurationLimits')[0]
for cam in (client,server): cam.set_controls({'FrameRate':1_000_000.0/(floor*1.05)})
ex=ThreadPoolExecutor(2)
def grab(c):
    r=c.capture_request(); ts=r.get_metadata().get('SensorTimestamp',0); r.release(); return ts
def measure():
    a=ex.submit(grab,client); b=ex.submit(grab,server); ta,tb=a.result(),b.result()
    return abs(ta-tb)/1000.0 if ta and tb else -1
print('FASE 1: converger con gaps (bursts c/sleep)',flush=True)
for i in range(5):
    time.sleep(4)
    d=0
    for _ in range(8): d=measure()
    print(f'  burst {i} delta_us={d:.1f}',flush=True)
print('FASE 2: captura CONTINUA 20s (mantiene la convergencia?)',flush=True)
t0=time.time(); n=0
while time.time()-t0<20:
    d=measure(); n+=1
    if n%150==0: print(f'  t={int(time.time()-t0):2d}s delta_us={d:.1f}',flush=True)
client.stop(); server.stop(); client.close(); server.close()

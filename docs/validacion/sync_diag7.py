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
def grab_sync(c):
    r=c.capture_sync_request(); ts=r.get_metadata().get('SensorTimestamp',0); r.release(); return ts
print('=== CONTINUO con capture_sync_request ===',flush=True)
t0=time.time(); last=0
while time.time()-t0 < 28:
    a=ex.submit(grab_sync,client); b=ex.submit(grab_sync,server)
    ta,tb=a.result(),b.result()
    if ta and tb: last=abs(ta-tb)/1000.0
    if int(time.time()-t0)!=int(last) and int(time.time()-t0)%4==0:
        print(f'  t={int(time.time()-t0):2d}s delta_us={last:.1f}',flush=True); time.sleep(0.3)
client.stop(); server.stop(); client.close(); server.close()

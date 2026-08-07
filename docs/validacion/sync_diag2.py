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
FDL = (16000, 18000)
kind = sys.argv[1] if len(sys.argv)>1 else 'video'

client = Picamera2(0); server = Picamera2(1)
def mk(cam, sm):
    f = {'video': cam.create_video_configuration,
         'preview': cam.create_preview_configuration,
         'still4': cam.create_still_configuration}[kind]
    kw = dict(main={'size':RES,'format':'RGB888'}, raw={'size':RAW},
              controls={'FrameDurationLimits':FDL,'SyncMode':sm})
    if kind=='still4': kw['buffer_count']=4
    return f(**kw)
client.configure(mk(client,E.Client)); client.start()
time.sleep(0.5)
server.configure(mk(server,E.Server)); server.start()
ex=ThreadPoolExecutor(2)
def grab(c):
    r=c.capture_request(); ts=r.get_metadata().get('SensorTimestamp',0); r.release(); return ts
print(f'kind={kind}: muestreo continuo cada 2s x 30s', flush=True)
synced=0; total=0
for i in range(15):
    time.sleep(2)
    ds=[]
    for _ in range(10):
        fl=ex.submit(grab,client); fr=ex.submit(grab,server)
        a,b=fl.result(),fr.result()
        if a and b: ds.append(abs(a-b)/1000.0)
    ds.sort(); med=ds[len(ds)//2]
    ok='OK' if med<1000 else 'OFF'
    total+=1; synced+= (1 if med<1000 else 0)
    print(f'  t={i*2+2:2d}s median={med:8.1f}us {ok}', flush=True)
print(f'RESUMEN {kind}: {synced}/{total} ventanas sincronizadas (<1ms)', flush=True)
client.stop(); server.stop(); client.close(); server.close()

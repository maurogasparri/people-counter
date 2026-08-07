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

def run(kind):
    client = Picamera2(0); server = Picamera2(1)
    def cfgfn(cam, sm):
        mk = cam.create_still_configuration if kind=='still' else cam.create_preview_configuration
        return mk(main={'size':RES,'format':'RGB888'}, raw={'size':RAW},
                  controls={'FrameDurationLimits':FDL,'SyncMode':sm})
    client.configure(cfgfn(client, E.Client)); client.start()
    server.configure(cfgfn(server, E.Server)); server.start()
    ex = ThreadPoolExecutor(2)
    def grab(c):
        r=c.capture_request(); ts=r.get_metadata().get('SensorTimestamp',0); r.release(); return ts
    print(f'--- kind={kind} ---', flush=True)
    for t in (3,6,10,15,22):
        time.sleep(t - (3 if t==3 else 0) if False else 0)
        time.sleep(3)
        ds=[]
        for _ in range(15):
            fl=ex.submit(grab,client); fr=ex.submit(grab,server)
            a,b=fl.result(),fr.result()
            if a and b: ds.append(abs(a-b)/1000.0)
        ds.sort(); med=ds[len(ds)//2] if ds else -1
        print(f'  t~{t+3}s: median={med:.1f}us  (min={min(ds):.1f} max={max(ds):.1f})', flush=True)
    client.stop(); server.stop(); client.close(); server.close()

run('still')
time.sleep(2)
run('preview')

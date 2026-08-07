import sys, time, threading
sys.path.insert(0,'/usr/src/people-counter')
from src.config.loader import load_config
from src.main import _runtime_resolution
from src.vision.capture import StereoCapture
cfg=load_config('/etc/people-counter/config.yaml')
RES=tuple(_runtime_resolution(cfg)); RAW=tuple(cfg['sensor']['default_res'])
cap=StereoCapture(0,1,RES,max_exposure_us=16000,sensor_raw_size=RAW,camera_sync=True,async_capture=True)
cap.open()
stop=False
def reader():
    while not stop:
        try: cap.read_with_metadata()
        except Exception: break
threading.Thread(target=reader,daemon=True).start()  # consumo continuo como el servicio
for t in range(1,16):
    time.sleep(2)
    print(f'  t={t*2:2d}s last_sync_delta_us={cap._last_sync_delta_us}',flush=True)
stop=True; time.sleep(0.5); cap.close()

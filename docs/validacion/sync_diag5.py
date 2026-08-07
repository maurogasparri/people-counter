import sys, time
sys.path.insert(0,'/usr/src/people-counter')
from src.config.loader import load_config
from src.main import _runtime_resolution
from src.vision.capture import StereoCapture
cfg=load_config('/etc/people-counter/config.yaml')
RES=tuple(_runtime_resolution(cfg)); RAW=tuple(cfg['sensor']['default_res'])
cap=StereoCapture(0,1,RES,max_exposure_us=16000,sensor_raw_size=RAW,camera_sync=True,async_capture=True)
cap.open()
fdl=cap._cam_left.camera_controls.get('FrameDurationLimits')
md=cap._cam_left.capture_metadata()
print(f'post-open: FrameDurationLimits={fdl} FrameDuration_now={md.get("FrameDuration")} exp={md.get("ExposureTime")}',flush=True)
for t in (8,16,24,32):
    time.sleep(8)
    for _ in range(25): cap.read_with_metadata()
    print(f'  t={t}s last_sync_delta_us={cap._last_sync_delta_us}',flush=True)
cap.close()

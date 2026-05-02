"""Stereo frame acquisition from dual CSI cameras.

Supports three modes:
  - picamera2: RPi5 CSI cameras via libcamera/picamera2 (production).
  - opencv: USB or V4L2 cameras via OpenCV VideoCapture (fallback).
  - file: Replay from saved image pairs (for testing/development).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoCapture:
    """Manages simultaneous capture from left and right CSI cameras via picamera2."""

    def __init__(
        self,
        cam_left_id: int,
        cam_right_id: int,
        resolution: tuple[int, int],
        fps: int = 15,
        meter_mode: str = "matrix",
        lock_ae: bool = False,
    ) -> None:
        """Initialize stereo capture.

        Args:
            cam_left_id: Left camera index as listed by rpicam-hello --list-cameras.
            cam_right_id: Right camera index.
            resolution: (width, height) capture resolution.
            fps: Target frame rate.
            meter_mode: AE metering mode ('matrix', 'centre', 'spot'). Centre/spot
                ignore the frame periphery during the AE settle, useful when
                bright zones outside the working area drag exposure down on the
                target (calibration board). Default 'matrix' matches picamera2's
                default whole-frame weighting.
            lock_ae: When True, AE/AWB are locked to the values that AE settled
                on after a 1-second wait. Useful when the calibration scene has
                fluctuating light (natural daylight, doors opening) — the lock
                prevents L/R AE drift mid-session. Default False — for stable
                indoor lighting AE auto is simpler and produces representative
                images for the ground-truth report.
        """
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id
        self.resolution = resolution
        self.fps = fps
        self.meter_mode = meter_mode
        self.lock_ae = lock_ae
        self._cam_left = None
        self._cam_right = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def open(self) -> None:
        """Open both camera streams via picamera2.

        Raises:
            RuntimeError: If either camera fails to open.
        """
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError(
                "picamera2 not installed. "
                "Install with: pip install picamera2"
            )

        try:
            self._cam_left = Picamera2(self.cam_left_id)
            self._cam_right = Picamera2(self.cam_right_id)
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to open cameras: {e}") from e

        w, h = self.resolution
        for cam, name in [
            (self._cam_left, "left"),
            (self._cam_right, "right"),
        ]:
            config = cam.create_still_configuration(
                main={"size": (w, h), "format": "BGR888"},
                controls={"FrameRate": self.fps},
            )
            cam.configure(config)
            cam.start()

        # Set AE metering mode before the settle so the locked exposure
        # reflects the chosen weighting (matrix/centre/spot).
        if self.meter_mode != "matrix":
            try:
                from libcamera import controls as _libcam_controls
                meter_map = {
                    "centre": _libcam_controls.AeMeteringModeEnum.CentreWeighted,
                    "spot": _libcam_controls.AeMeteringModeEnum.Spot,
                }
                meter_value = meter_map[self.meter_mode]
                for cam in [self._cam_left, self._cam_right]:
                    cam.set_controls({"AeMeteringMode": meter_value})
                logger.info(
                    "ae_metering_mode_set",
                    extra={"mode": self.meter_mode},
                )
            except (ImportError, KeyError, Exception) as e:
                logger.warning(
                    "ae_metering_mode_failed",
                    extra={"mode": self.meter_mode, "error": str(e)},
                )

        # Optionally lock exposure, gain and white balance after a 1s settle.
        # Default behaviour (lock_ae=False) keeps AE auto throughout the
        # session — simpler, produces representative ground-truth images,
        # and works fine for stable indoor lighting. Enable lock_ae=True
        # when the scene has fluctuating light (natural daylight, doors
        # opening, mixed lighting) — the lock prevents independent L/R AE
        # drift mid-session.
        if self.lock_ae:
            import time as _time
            _time.sleep(1.0)
            for cam, name in [
                (self._cam_left, "left"),
                (self._cam_right, "right"),
            ]:
                metadata = cam.capture_metadata()
                cam.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": metadata.get("ExposureTime", 30000),
                    "AnalogueGain": metadata.get("AnalogueGain", 1.0),
                    "ColourGains": metadata.get("ColourGains", (1.0, 1.0)),
                })
                logger.info(
                    "camera_controls_locked",
                    extra={
                        "camera": name,
                        "exposure_us": metadata.get("ExposureTime", 0),
                        "analogue_gain": metadata.get("AnalogueGain", 0),
                    },
                )

        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="stereo-cap"
        )

        logger.info(
            "stereo_capture_opened",
            extra={
                "left_id": self.cam_left_id,
                "right_id": self.cam_right_id,
                "resolution": list(self.resolution),
                "fps": self.fps,
            },
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read frame pair.

        Returns:
            (left_frame, right_frame) as BGR numpy arrays.

        Raises:
            RuntimeError: If cameras not opened or read fails.
        """
        if self._cam_left is None or self._cam_right is None or self._executor is None:
            raise RuntimeError("Cameras not opened. Call open() first.")

        try:
            fut_l = self._executor.submit(self._cam_left.capture_array, "main")
            fut_r = self._executor.submit(self._cam_right.capture_array, "main")
            frame_l = fut_l.result()
            frame_r = fut_r.result()
        except Exception as e:
            raise RuntimeError(f"Frame capture failed: {e}") from e

        # picamera2 with "BGR888" format empirically delivers RGB on RPi OS
        # Trixie / libcamera builds we ship with. Convert to BGR so downstream
        # consumers (OpenCV, YOLO preprocess that assumes BGR input) see the
        # correct channel order.
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

        return frame_l, frame_r

    def read_with_timestamps(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Read frame pair along with each camera's sensor timestamp (ns).

        Returns:
            (left_frame, right_frame, ts_left_ns, ts_right_ns)
        """
        frame_l, frame_r, ts_l, ts_r, _temp_l, _temp_r = self.read_with_metadata()
        return frame_l, frame_r, ts_l, ts_r

    def read_with_metadata(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, Optional[float], Optional[float]]:
        """Read frame pair with sensor timestamps (ns) and sensor temperatures (°C).

        Temperature comes from picamera2 SensorTemperature metadata key when
        available (IMX708 on RPi5 exposes it). Returns None per camera if the
        key isn't present — older libcamera builds or non-IMX708 sensors.
        """
        if self._cam_left is None or self._cam_right is None or self._executor is None:
            raise RuntimeError("Cameras not opened. Call open() first.")

        def _grab(cam):
            req = cam.capture_request()
            try:
                frame = req.make_array("main")
                metadata = req.get_metadata()
                ts = int(metadata.get("SensorTimestamp", 0))
                temp_raw = metadata.get("SensorTemperature")
                temp = float(temp_raw) if temp_raw is not None else None
                return frame, ts, temp
            finally:
                req.release()

        try:
            fut_l = self._executor.submit(_grab, self._cam_left)
            fut_r = self._executor.submit(_grab, self._cam_right)
            frame_l, ts_l, temp_l = fut_l.result()
            frame_r, ts_r, temp_r = fut_r.result()
        except Exception as e:
            raise RuntimeError(f"Frame capture failed: {e}") from e

        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

        return frame_l, frame_r, ts_l, ts_r, temp_l, temp_r

    def close(self) -> None:
        """Release camera resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for cam, name in [
            (self._cam_left, "left"),
            (self._cam_right, "right"),
        ]:
            if cam is not None:
                try:
                    cam.stop()
                    cam.close()
                except Exception:
                    logger.warning("camera_close_failed", extra={"camera": name})
        self._cam_left = None
        self._cam_right = None
        logger.info("stereo_capture_closed")

    def __enter__(self) -> "StereoCapture":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FileCapture:
    """Replay stereo frame pairs from saved image files.

    Looks for files named left_NNN.png and right_NNN.png in the given
    directory. Useful for development and testing without hardware.
    """

    def __init__(
        self,
        directory: str,
        loop: bool = True,
        fps: int = 15,
    ) -> None:
        """Initialize file-based capture.

        Args:
            directory: Path to directory containing left_*/right_* images.
            loop: Whether to restart from beginning after all pairs.
            fps: Simulated frame rate (controls read delay).
        """
        self.directory = Path(directory)
        self.loop = loop
        self.fps = fps
        self._pairs: list[tuple[Path, Path]] = []
        self._index = 0
        self._frame_interval = 1.0 / fps if fps > 0 else 0
        self._last_read = 0.0

    def open(self) -> None:
        """Scan directory for image pairs.

        Raises:
            RuntimeError: If no valid pairs found.
        """
        left_files = sorted(self.directory.glob("left_*.png"))
        self._pairs = []

        for lf in left_files:
            rf = lf.parent / lf.name.replace("left_", "right_")
            if rf.exists():
                self._pairs.append((lf, rf))

        if not self._pairs:
            # Also try .jpg
            left_files = sorted(self.directory.glob("left_*.jpg"))
            for lf in left_files:
                rf = lf.parent / lf.name.replace("left_", "right_")
                if rf.exists():
                    self._pairs.append((lf, rf))

        if not self._pairs:
            raise RuntimeError(
                f"No stereo pairs found in {self.directory}. "
                "Expected files named left_NNN.png and right_NNN.png"
            )

        self._index = 0
        logger.info(
            "file_capture_opened",
            extra={"pairs": len(self._pairs), "path": str(self.directory)},
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read next frame pair.

        Returns:
            (left_frame, right_frame) as BGR numpy arrays.

        Raises:
            StopIteration: If all pairs consumed and loop=False.
            RuntimeError: If pairs not loaded.
        """
        if not self._pairs:
            raise RuntimeError("No pairs loaded. Call open() first.")

        if self._index >= len(self._pairs):
            if self.loop:
                self._index = 0
            else:
                raise StopIteration("All frame pairs consumed")

        # Simulate frame rate
        now = time.monotonic()
        elapsed = now - self._last_read
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)
        self._last_read = time.monotonic()

        lf, rf = self._pairs[self._index]
        img_l = cv2.imread(str(lf))
        img_r = cv2.imread(str(rf))

        if img_l is None or img_r is None:
            raise RuntimeError(f"Failed to read pair at index {self._index}")

        self._index += 1
        return img_l, img_r

    @property
    def total_pairs(self) -> int:
        return len(self._pairs)

    @property
    def current_index(self) -> int:
        return self._index

    def close(self) -> None:
        """Reset state."""
        self._pairs = []
        self._index = 0

    def __enter__(self) -> "FileCapture":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

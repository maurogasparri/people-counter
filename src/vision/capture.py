"""Stereo frame acquisition from dual CSI cameras.

Supports three modes:
  - picamera2: RPi5 CSI cameras via libcamera/picamera2 (production).
  - opencv: USB or V4L2 cameras via OpenCV VideoCapture (fallback).
  - file: Replay from saved image pairs (for testing/development).
"""

import logging
import time
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
    ) -> None:
        """Initialize stereo capture.

        Args:
            cam_left_id: Left camera index as listed by rpicam-hello --list-cameras.
            cam_right_id: Right camera index.
            resolution: (width, height) capture resolution.
            fps: Target frame rate.
        """
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id
        self.resolution = resolution
        self.fps = fps
        self._cam_left = None
        self._cam_right = None

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

        # Lock exposure, gain and white balance so both cameras match.
        # Let auto-exposure settle first, then fix the values.
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
                "%s camera locked: exposure=%d gain=%.1f",
                name,
                metadata.get("ExposureTime", 0),
                metadata.get("AnalogueGain", 0),
            )

        logger.info(
            "Stereo capture opened: left=%d, right=%d, res=%s, fps=%d",
            self.cam_left_id,
            self.cam_right_id,
            self.resolution,
            self.fps,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read frame pair.

        Returns:
            (left_frame, right_frame) as BGR numpy arrays.

        Raises:
            RuntimeError: If cameras not opened or read fails.
        """
        if self._cam_left is None or self._cam_right is None:
            raise RuntimeError("Cameras not opened. Call open() first.")

        from concurrent.futures import ThreadPoolExecutor

        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_l = ex.submit(self._cam_left.capture_array, "main")
                fut_r = ex.submit(self._cam_right.capture_array, "main")
                frame_l = fut_l.result()
                frame_r = fut_r.result()
        except Exception as e:
            raise RuntimeError(f"Frame capture failed: {e}") from e

        return frame_l, frame_r

    def close(self) -> None:
        """Release camera resources."""
        for cam, name in [
            (self._cam_left, "left"),
            (self._cam_right, "right"),
        ]:
            if cam is not None:
                try:
                    cam.stop()
                    cam.close()
                except Exception:
                    logger.warning("Error closing %s camera", name)
        self._cam_left = None
        self._cam_right = None
        logger.info("Stereo capture closed")

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
            "File capture opened: %d pairs from %s",
            len(self._pairs),
            self.directory,
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

"""Stereo frame acquisition from dual CSI cameras.

Supports two modes:
  - live: Real cameras via OpenCV VideoCapture (CSI or USB).
  - file: Replay from saved image pairs (for testing/development).

On RPi5 with libcamera, cameras appear as /dev/video0 and /dev/video2
(not sequential) depending on CSI port. Use `v4l2-ctl --list-devices`
to confirm.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoCapture:
    """Manages simultaneous capture from left and right cameras."""

    def __init__(
        self,
        cam_left_id: int,
        cam_right_id: int,
        resolution: tuple[int, int],
        fps: int = 15,
    ) -> None:
        """Initialize stereo capture.

        Args:
            cam_left_id: Left camera device index (e.g. 0).
            cam_right_id: Right camera device index (e.g. 1 or 2).
            resolution: (width, height) capture resolution.
            fps: Target frame rate.
        """
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id
        self.resolution = resolution
        self.fps = fps
        self._cap_left: Optional[cv2.VideoCapture] = None
        self._cap_right: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open both camera streams.

        Raises:
            RuntimeError: If either camera fails to open.
        """
        self._cap_left = cv2.VideoCapture(self.cam_left_id)
        self._cap_right = cv2.VideoCapture(self.cam_right_id)

        for cap, name, cam_id in [
            (self._cap_left, "left", self.cam_left_id),
            (self._cap_right, "right", self.cam_right_id),
        ]:
            if not cap.isOpened():
                self.close()
                raise RuntimeError(f"Failed to open {name} camera (id={cam_id})")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(
            "Stereo capture opened: left=%d, right=%d, res=%s, fps=%d",
            self.cam_left_id,
            self.cam_right_id,
            self.resolution,
            self.fps,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read synchronized frame pair.

        Returns:
            (left_frame, right_frame) as BGR numpy arrays.

        Raises:
            RuntimeError: If cameras not opened or read fails.
        """
        if self._cap_left is None or self._cap_right is None:
            raise RuntimeError("Cameras not opened. Call open() first.")

        ret_l, frame_l = self._cap_left.read()
        ret_r, frame_r = self._cap_right.read()

        if not ret_l or not ret_r:
            raise RuntimeError(
                f"Frame read failed: left={ret_l}, right={ret_r}"
            )

        return frame_l, frame_r

    def close(self) -> None:
        """Release camera resources."""
        if self._cap_left is not None:
            self._cap_left.release()
            self._cap_left = None
        if self._cap_right is not None:
            self._cap_right.release()
            self._cap_right = None
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

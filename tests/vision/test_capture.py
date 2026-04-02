"""Tests for stereo capture module (FileCapture only — no real cameras)."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.capture import FileCapture


def _create_test_pairs(directory: str, n: int = 5, ext: str = "png") -> None:
    """Create synthetic stereo pair files in directory."""
    for i in range(n):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(Path(directory) / f"left_{i:03d}.{ext}"), img)
        cv2.imwrite(str(Path(directory) / f"right_{i:03d}.{ext}"), img)


class TestFileCapture:
    def test_open_and_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 5)
            cap = FileCapture(tmpdir)
            cap.open()
            assert cap.total_pairs == 5
            cap.close()

    def test_read_returns_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 3)
            cap = FileCapture(tmpdir, fps=0)
            cap.open()
            left, right = cap.read()
            assert left.shape == (480, 640, 3)
            assert right.shape == (480, 640, 3)
            cap.close()

    def test_read_all_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 3)
            cap = FileCapture(tmpdir, loop=False, fps=0)
            cap.open()
            for _ in range(3):
                cap.read()
            with pytest.raises(StopIteration):
                cap.read()
            cap.close()

    def test_loop_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 2)
            cap = FileCapture(tmpdir, loop=True, fps=0)
            cap.open()
            # Read more than available — should loop
            for _ in range(5):
                left, right = cap.read()
                assert left is not None
            cap.close()

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cap = FileCapture(tmpdir)
            with pytest.raises(RuntimeError, match="No stereo pairs"):
                cap.open()

    def test_missing_right_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(Path(tmpdir) / "left_000.png"), img)
            cv2.imwrite(str(Path(tmpdir) / "left_001.png"), img)
            cv2.imwrite(str(Path(tmpdir) / "right_001.png"), img)
            # Only pair 001 has both files
            cap = FileCapture(tmpdir, fps=0)
            cap.open()
            assert cap.total_pairs == 1
            cap.close()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 2)
            with FileCapture(tmpdir, fps=0) as cap:
                left, right = cap.read()
                assert left is not None

    def test_current_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 3)
            cap = FileCapture(tmpdir, fps=0)
            cap.open()
            assert cap.current_index == 0
            cap.read()
            assert cap.current_index == 1
            cap.read()
            assert cap.current_index == 2
            cap.close()

    def test_jpg_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_pairs(tmpdir, 2, ext="jpg")
            cap = FileCapture(tmpdir, fps=0)
            cap.open()
            assert cap.total_pairs == 2
            cap.close()

    def test_read_before_open_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cap = FileCapture(tmpdir)
            with pytest.raises(RuntimeError, match="No pairs loaded"):
                cap.read()

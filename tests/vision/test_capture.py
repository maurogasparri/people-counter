"""Tests para el módulo de captura estéreo (solo FileCapture — sin cámaras reales)."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.capture import (
    CAMERA_STALL_THRESHOLD,
    FileCapture,
    StereoCapture,
    _next_stall,
)


def test_next_stall_advances_resets_and_handles_no_ts():
    assert _next_stall(100, 90, 5) == 0  # ts avanzó → reset
    assert _next_stall(100, 100, 5) == 6  # ts no avanzó → +1
    assert _next_stall(0, 100, 5) == 0  # sin SensorTimestamp → no evaluable


def test_camera_health_attributes_stalled_camera():
    cap = StereoCapture(0, 1, (640, 480))
    assert cap.camera_health() == (True, True)  # default sano
    cap._cam_r_stall = CAMERA_STALL_THRESHOLD  # derecha congelada
    assert cap.camera_health() == (True, False)


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


class _FakeReq:
    def __init__(self, frame, ts, temp):
        self._frame, self._ts, self._temp = frame, ts, temp

    def make_array(self, name):
        return self._frame.copy()

    def get_metadata(self):
        return {"SensorTimestamp": self._ts, "SensorTemperature": self._temp}

    def release(self):
        pass


class _FakeCam:
    def __init__(self, frame, ts=123, temp=40.0):
        self._frame, self._ts, self._temp = frame, ts, temp
        self.captures = 0

    def capture_request(self):
        self.captures += 1
        return _FakeReq(self._frame, self._ts, self._temp)

    def stop(self):
        pass

    def close(self):
        pass


def _wired_stereo(async_capture: bool) -> StereoCapture:
    """StereoCapture con cámaras fake inyectadas (sin picamera2/open)."""
    cap = StereoCapture(
        cam_left_id=0,
        cam_right_id=1,
        resolution=(64, 48),
        async_capture=async_capture,
    )
    cap._cam_left = _FakeCam(np.zeros((48, 64, 3), np.uint8), ts=111)
    cap._cam_right = _FakeCam(np.zeros((48, 64, 3), np.uint8), ts=222)
    cap._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-cap")
    return cap


class TestStereoCaptureAsync:
    def test_async_read_starts_thread_and_returns_latest(self):
        cap = _wired_stereo(async_capture=True)
        try:
            fl, fr, ts_l, ts_r, temp_l, temp_r = cap.read_with_metadata()
            assert fl.shape == (48, 64, 3) and fr.shape == (48, 64, 3)
            assert ts_l == 111 and ts_r == 222
            assert temp_l == 40.0
            assert cap._stream_thread is not None and cap._stream_thread.is_alive()
        finally:
            cap.close()
        assert cap._stream_thread is None  # close() frenó el thread

    def test_sync_read_uses_no_thread(self):
        cap = _wired_stereo(async_capture=False)
        try:
            fl, fr, ts_l, ts_r, _, _ = cap.read_with_metadata()
            assert fl.shape == (48, 64, 3)
            assert ts_l == 111
            assert cap._stream_thread is None  # modo sync: sin thread productor
        finally:
            cap.close()

    def test_async_read_frames_starts_thread(self):
        # El pipeline usa read() (sin metadata) — debe activar el path async.
        cap = _wired_stereo(async_capture=True)
        try:
            fl, fr = cap.read()
            assert fl.shape == (48, 64, 3) and fr.shape == (48, 64, 3)
            assert cap._stream_thread is not None and cap._stream_thread.is_alive()
        finally:
            cap.close()

    def test_close_does_not_block_on_wedged_stream_thread(self):
        """Regresión: con el thread productor wedgeado (capture_request
        colgado, cámara stalleada) el join de close() expira — el
        executor.shutdown(wait=True) de después esperaba PARA SIEMPRE al
        future colgado y close() nunca llegaba a cam.stop(); systemd
        terminaba matando el proceso por TimeoutStopSec en vez del
        shutdown limpio. Con wait=False cuando el join falló, close()
        retorna."""
        from unittest.mock import MagicMock

        cap = _wired_stereo(async_capture=True)
        # Simular el wedge: thread "vivo" cuyo join no termina.
        wedged = MagicMock()
        wedged.is_alive.return_value = True
        wedged.join.return_value = None
        cap._stream_thread = wedged
        executor = MagicMock()
        cap._executor = executor

        cap.close()  # no debe bloquear

        executor.shutdown.assert_called_once_with(wait=False)
        assert cap._stream_thread is None

    def test_stream_loop_rate_limits_error_log(self, caplog):
        """Regresión: una falla persistente de captura (cable CSI muerto)
        logueaba un traceback completo cada 50ms (~1.7M líneas/día) —
        wear de microSD + journald saturado. Ahora: traceback solo en el
        primer error, después 1 línea cada _STREAM_ERROR_LOG_EVERY."""
        import logging as _logging

        from src.vision.capture import _STREAM_ERROR_LOG_EVERY

        cap = _wired_stereo(async_capture=True)

        def _boom():
            raise RuntimeError("CSI muerto")

        cap._capture_pair_with_metadata = _boom
        # Sin esperas reales: el wait del backoff corta al instante salvo
        # las primeras N iteraciones (evento no seteado). Simulamos N
        # errores señalando stop después de un cupo de iteraciones.
        n_iters = _STREAM_ERROR_LOG_EVERY * 2
        calls = {"n": 0}
        real_wait = cap._stream_stop.wait

        def _counting_wait(timeout=None):
            calls["n"] += 1
            if calls["n"] >= n_iters:
                cap._stream_stop.set()
            return real_wait(0)

        cap._stream_stop.wait = _counting_wait
        with caplog.at_level(_logging.ERROR):
            cap._stream_loop()  # corre inline hasta que el wait setea stop

        msgs = [
            r
            for r in caplog.records
            if "stereo_stream_capture_failed" in r.getMessage()
        ]
        # 1 (primer error, con traceback) + 2 rate-limited (en 400 iteraciones)
        assert 1 <= len(msgs) <= 4
        assert len(msgs) < 10  # ni cerca de 1-por-iteración

    def test_close_stops_thread(self):
        cap = _wired_stereo(async_capture=True)
        cap.read_with_metadata()
        assert cap._stream_thread.is_alive()
        cap.close()
        assert cap._stream_thread is None

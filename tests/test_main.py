"""Tests for main.py — pipeline orchestrator."""
import argparse
import gc
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.main import (
    _auto_num_disparities,
    build_capture,
    build_mqtt,
    get_telemetry,
    run_pipeline,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_logging():
    """Reset root logger so basicConfig takes effect again."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.WARNING)  # default


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_json_format():
    _reset_logging()
    config = {"logging": {"level": "DEBUG", "format": "json"}}
    setup_logging(config)
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_setup_logging_plain_format():
    _reset_logging()
    config = {"logging": {"level": "WARNING", "format": "plain"}}
    setup_logging(config)
    root = logging.getLogger()
    assert root.level == logging.WARNING


def test_setup_logging_defaults():
    _reset_logging()
    config = {}
    setup_logging(config)
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_setup_logging_with_file():
    _reset_logging()
    tmpdir = tempfile.mkdtemp()
    log_file = str(Path(tmpdir) / "sub" / "app.log")
    config = {"logging": {"level": "INFO", "file": log_file}}
    setup_logging(config)
    assert Path(log_file).parent.exists()


# ---------------------------------------------------------------------------
# build_capture
# ---------------------------------------------------------------------------


def test_build_capture_file_replay():
    config = {"vision": {"fps": 10}}
    cap = build_capture(config, replay_dir="/some/dir")
    from src.vision.capture import FileCapture

    assert isinstance(cap, FileCapture)
    assert cap.fps == 10


def test_build_capture_live():
    config = {
        "vision": {
            "camera_left": 0,
            "camera_right": 1,
            "resolution": [640, 480],
            "fps": 15,
        }
    }
    cap = build_capture(config, replay_dir=None)
    from src.vision.capture import StereoCapture

    assert isinstance(cap, StereoCapture)
    assert cap.fps == 15


# ---------------------------------------------------------------------------
# _auto_num_disparities
# ---------------------------------------------------------------------------


def test_auto_num_disparities_scales_with_runtime_resolution():
    """f_px must scale with runtime width — same site at half-res should get
    roughly half the disparity envelope, not the full-res value."""
    logger = logging.getLogger("test")
    full = _auto_num_disparities(
        {"mounting_height_m": 2.56, "resolution": [4608, 2592]}, logger,
    )
    half = _auto_num_disparities(
        {"mounting_height_m": 2.56, "resolution": [2304, 1296]}, logger,
    )
    quarter = _auto_num_disparities(
        {"mounting_height_m": 2.56, "resolution": [1152, 648]}, logger,
    )
    # disp_max ∝ f_px ∝ width. Each step should roughly halve, give or take
    # the multiple-of-16 rounding and the +1 margin bucket.
    assert half < full
    assert quarter < half
    # At 1152x648, mount=2.56m, baseline=140mm, f_px≈512 → disp_max≈100px
    # → next multiple of 16 + margin ≈ 128. Allow some slack for clamp.
    assert quarter <= 144
    # At 4608x2592 the same site gets ~4× the disparity envelope, but the
    # function clamps to 512 max — make sure we hit the upper region.
    assert full >= 256


def test_auto_num_disparities_clamps_to_envelope():
    """Output must stay in [64, 512] regardless of mount/resolution."""
    logger = logging.getLogger("test")
    # Tiny resolution + tall mount → very low disp_max; should clamp to 64.
    out = _auto_num_disparities(
        {"mounting_height_m": 5.0, "resolution": [320, 240]}, logger,
    )
    assert out >= 64
    # Full-res + low mount → very high disp_max; should clamp to 512.
    out = _auto_num_disparities(
        {"mounting_height_m": 1.5, "resolution": [4608, 2592]}, logger,
    )
    assert out <= 512


# ---------------------------------------------------------------------------
# build_mqtt
# ---------------------------------------------------------------------------


def _make_certs_and_config(tmpdir, store_id="store-42", extra_mqtt=None):
    cert = str(Path(tmpdir) / "cert.pem")
    key = str(Path(tmpdir) / "key.pem")
    ca = str(Path(tmpdir) / "ca.pem")
    for f in [cert, key, ca]:
        Path(f).write_text("dummy")

    mqtt_cfg = {
        "endpoint": "test.iot.amazonaws.com",
        "port": 8883,
        "cert_path": cert,
        "key_path": key,
        "ca_path": ca,
    }
    if extra_mqtt:
        mqtt_cfg.update(extra_mqtt)

    return {
        "device": {"id": "dev-001", "store_id": store_id},
        "mqtt": mqtt_cfg,
        "buffer": {"db_path": str(Path(tmpdir) / "buf.db"), "max_age_hours": 24},
    }


@patch("src.mqtt.client.mqtt.Client")
def test_build_mqtt_topic_expansion(mock_mqtt_cls):
    mock_mqtt_cls.return_value = MagicMock()
    tmpdir = tempfile.mkdtemp()
    config = _make_certs_and_config(
        tmpdir,
        store_id="store-42",
        extra_mqtt={
            "topics": {
                "counting": "store/{store_id}/counting",
                "telemetry": "store/{store_id}/telemetry",
            },
        },
    )
    client, buffer = build_mqtt(config)
    assert client.topics["counting"] == "store/store-42/counting"
    assert client.topics["telemetry"] == "store/store-42/telemetry"
    # Force cleanup of sqlite references
    del buffer, client
    gc.collect()


@patch("src.mqtt.client.mqtt.Client")
def test_build_mqtt_default_port(mock_mqtt_cls):
    mock_mqtt_cls.return_value = MagicMock()
    tmpdir = tempfile.mkdtemp()
    config = _make_certs_and_config(tmpdir, store_id="store-01")
    client, buffer = build_mqtt(config)
    assert client.port == 8883
    del buffer, client
    gc.collect()


# ---------------------------------------------------------------------------
# get_telemetry
# ---------------------------------------------------------------------------


def test_get_telemetry_returns_dict():
    telem = get_telemetry()
    assert isinstance(telem, dict)
    assert "uptime_s" in telem


def test_get_telemetry_graceful_on_windows():
    """On Windows (no /proc), telemetry should still return safely."""
    telem = get_telemetry()
    assert isinstance(telem, dict)
    assert isinstance(telem["uptime_s"], (int, float))


def test_get_telemetry_forwards_state_to_collect_telemetry():
    """The shim must pass ``state`` through to collect_telemetry and return
    the augmented dict (with the new observability keys)."""
    state = {
        "frame_latencies_ms": [10.0, 20.0, 30.0],
        "tracker_confirmed": 2,
        "tracker_pending": 0,
        "mqtt_disconnect_count": 1,
        "buffer_backlog": 5,
    }
    telem = get_telemetry(state)
    assert telem["frame_latency_p50_ms"] == 20.0
    assert telem["tracker_confirmed_count"] == 2
    assert telem["mqtt_disconnect_count"] == 1
    assert telem["buffer_backlog_messages"] == 5


# ---------------------------------------------------------------------------
# run_pipeline — integration with mocks
# ---------------------------------------------------------------------------


def _make_pipeline_config(tmpdir: str) -> dict:
    cert = str(Path(tmpdir) / "cert.pem")
    key = str(Path(tmpdir) / "key.pem")
    ca = str(Path(tmpdir) / "ca.pem")
    for f in [cert, key, ca]:
        Path(f).write_text("dummy")

    return {
        "device": {"id": "test-001", "store_id": "store-01"},
        "vision": {
            "camera_left": 0,
            "camera_right": 1,
            "resolution": [640, 480],
            "fps": 15,
            "baseline_cm": 14,
            "counting_line_y": 0.5,
        },
        "detection": {
            "model_path": "/tmp/model.onnx",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.45,
        },
        "tracking": {"max_disappeared": 30, "max_distance": 50},
        "telemetry": {"interval_seconds": 9999},
        "mqtt": {
            "endpoint": "test.iot.amazonaws.com",
            "port": 8883,
            "cert_path": cert,
            "key_path": key,
            "ca_path": ca,
            "topics": {
                "counting": "store/store-01/counting",
                "telemetry": "store/store-01/telemetry",
            },
        },
        "buffer": {"db_path": str(Path(tmpdir) / "buf.db"), "max_age_hours": 72},
        "cloud_defaults": {
            "counting_enabled": True,
            "operating_hours": {
                "monday": "00:00-23:59",
                "tuesday": "00:00-23:59",
                "wednesday": "00:00-23:59",
                "thursday": "00:00-23:59",
                "friday": "00:00-23:59",
                "saturday": "00:00-23:59",
                "sunday": "00:00-23:59",
            },
        },
    }


def _make_mock_capture(frames):
    """Create a mock capture that yields N frame pairs then stops."""
    mock = MagicMock()
    call_idx = [0]

    def fake_read():
        idx = call_idx[0]
        call_idx[0] += 1
        if idx >= len(frames):
            raise StopIteration("Done")
        return frames[idx]

    mock.read.side_effect = fake_read
    return mock


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_file_replay_exhausted(mock_build_cap, mock_load_model, mock_mqtt_cls):
    """Pipeline should stop cleanly when capture raises StopIteration."""
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_backend.infer.return_value = np.zeros((1, 84, 0), dtype=np.float32)
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_build_cap.return_value = _make_mock_capture([(dummy, dummy)])

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    run_pipeline(config, args)
    assert mock_backend.infer.call_count == 1


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_no_calibration(mock_build_cap, mock_load_model, mock_mqtt_cls):
    """Pipeline runs without calibration (depth_map=None path)."""
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_backend.infer.return_value = np.zeros((1, 84, 0), dtype=np.float32)
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_build_cap.return_value = _make_mock_capture([(dummy, dummy)])

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    config["vision"].pop("calibration_file", None)
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    run_pipeline(config, args)


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_counting_disabled(mock_build_cap, mock_load_model, mock_mqtt_cls):
    """When counting_enabled=False, pipeline sleeps without processing."""
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    # Capture that would work, but shouldn't be called
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_build_cap.return_value = _make_mock_capture([(dummy, dummy)] * 10)

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    config["cloud_defaults"]["counting_enabled"] = False
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    sleep_count = [0]

    def _fake_sleep(seconds):
        sleep_count[0] += 1
        if sleep_count[0] >= 2:
            raise KeyboardInterrupt()

    with patch("src.main.time.sleep", side_effect=_fake_sleep):
        with patch("src.main.signal.signal"):
            try:
                run_pipeline(config, args)
            except KeyboardInterrupt:
                pass

    mock_backend.infer.assert_not_called()


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_publishes_counting_events(mock_build_cap, mock_load_model, mock_mqtt_cls):
    """Pipeline processes multiple frames and calls infer for each."""
    mock_mqtt_cls.return_value = MagicMock()

    call_count = [0]

    def fake_infer(blob):
        call_count[0] += 1
        output = np.zeros((1, 84, 1), dtype=np.float32)
        output[0, 0, 0] = 320.0  # cx
        output[0, 1, 0] = 200.0 if call_count[0] == 1 else 400.0  # cy
        output[0, 2, 0] = 100.0  # w
        output[0, 3, 0] = 200.0  # h
        output[0, 4, 0] = 0.9  # person confidence
        return output

    mock_backend = MagicMock()
    mock_backend.infer.side_effect = fake_infer
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_build_cap.return_value = _make_mock_capture([(dummy, dummy), (dummy, dummy)])

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    config["vision"].pop("calibration_file", None)
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    run_pipeline(config, args)
    assert mock_backend.infer.call_count == 2


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_invalid_schedule_fail_open_continues(
    mock_build_cap, mock_load_model, mock_mqtt_cls
):
    """Invalid schedule + fail_open: pipeline still counts (calls infer)."""
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_backend.infer.return_value = np.zeros((1, 84, 0), dtype=np.float32)
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_build_cap.return_value = _make_mock_capture([(dummy, dummy)])

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    config.pop("calibration_file", None)
    config["_schedule_error"] = "monday: invalid start time '25:00'"
    config["cloud_defaults"]["on_invalid_schedule"] = "fail_open"
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    caplog_ctx = _CaplogCtx()
    with caplog_ctx:
        run_pipeline(config, args)

    assert mock_backend.infer.call_count == 1
    assert any(
        "fail_open" in msg and "Invalid operating_hours" in msg
        for msg in caplog_ctx.messages
    )


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_invalid_schedule_fail_closed_pauses(
    mock_build_cap, mock_load_model, mock_mqtt_cls
):
    """Invalid schedule + fail_closed: pipeline does NOT call infer."""
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_build_cap.return_value = _make_mock_capture([(dummy, dummy)] * 10)

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    config["_schedule_error"] = "monday: end '10:00' must be after start '22:00'"
    config["cloud_defaults"]["on_invalid_schedule"] = "fail_closed"
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    sleep_count = [0]

    def _fake_sleep(seconds):
        sleep_count[0] += 1
        if sleep_count[0] >= 2:
            raise KeyboardInterrupt()

    caplog_ctx = _CaplogCtx(level=logging.CRITICAL)
    with caplog_ctx:
        with patch("src.main.time.sleep", side_effect=_fake_sleep):
            with patch("src.main.signal.signal"):
                try:
                    run_pipeline(config, args)
                except KeyboardInterrupt:
                    pass

    mock_backend.infer.assert_not_called()
    assert any(
        "fail_closed" in msg and "paused" in msg
        for msg in caplog_ctx.messages
    )


class _CaplogCtx:
    """Capture log records across all loggers without pytest's caplog (which
    only attaches to the root logger and can miss child loggers depending on
    propagation settings)."""

    def __init__(self, level=logging.WARNING):
        self.level = level
        self.records: list[logging.LogRecord] = []
        self._handler = None

    @property
    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]

    def __enter__(self):
        self._handler = logging.Handler(level=self.level)
        self._handler.emit = self.records.append  # type: ignore[assignment]
        logging.getLogger().addHandler(self._handler)
        logging.getLogger().setLevel(self.level)
        return self

    def __exit__(self, *exc):
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
        return False


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_publishes_shadow_reconciliation_on_boot(
    mock_build_cap, mock_load_model, mock_mqtt_cls
):
    """After mqtt.connect() completes the main loop must publish the
    effective config as shadow ``reported``. We simulate the paho on_connect
    firing by invoking the hook that main.py wires up.
    """
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_backend.infer.return_value = np.zeros((1, 84, 0), dtype=np.float32)
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    # Use a one-frame capture so we exit cleanly. Before returning the
    # capture, trigger the on_connected hook that main.py installed on the
    # (real) MQTTClient so the reconcile sentinel is enqueued before the
    # first drain pass.
    captured_clients: list = []

    real_build_mqtt = None

    from src import main as _main_mod

    real_build_mqtt = _main_mod.build_mqtt

    def _wrap_build_mqtt(config, no_mqtt=False):
        client, buf = real_build_mqtt(config, no_mqtt=no_mqtt)
        captured_clients.append(client)
        return client, buf

    def _capture_after_hook_fires(*args, **kwargs):
        # By the time build_capture is called, main.py has already set
        # on_connected and called connect(). Fire the hook manually to
        # emulate the paho thread.
        assert captured_clients, "build_mqtt should have been called first"
        client = captured_clients[0]
        assert client.on_connected is not None
        client.on_connected()
        return _make_mock_capture([(dummy, dummy)])

    mock_build_cap.side_effect = _capture_after_hook_fires

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    config["vision"].pop("calibration_file", None)
    config["device"]["firmware_version"] = "0.1.0-pilot"
    args = argparse.Namespace(
        replay_dir="/fake", detection_backend="opencv", config="/fake/config.yaml"
    )

    with patch.object(_main_mod, "build_mqtt", side_effect=_wrap_build_mqtt):
        run_pipeline(config, args)

    # The captured client's paho publish() should have been called with the
    # shadow update topic and a JSON payload carrying `reported`.
    assert captured_clients
    client = captured_clients[0]
    publish_calls = client._client.publish.call_args_list
    shadow_calls = [
        c for c in publish_calls
        if len(c.args) >= 1 and c.args[0] == "$aws/things/test-001/shadow/update"
    ]
    assert shadow_calls, "Expected a shadow reported publish on boot"
    import json as _json

    payload = _json.loads(shadow_calls[0].args[1])
    assert "state" in payload and "reported" in payload["state"]
    reported = payload["state"]["reported"]
    assert reported["firmware_version"] == "0.1.0-pilot"
    assert "boot_ts" in reported
    assert reported["effective_baseline_mm"] is None  # no calibration loaded


@patch("src.mqtt.client.mqtt.Client")
@patch("src.main.load_model")
@patch("src.main.build_capture")
def test_run_pipeline_capture_error_continues(mock_build_cap, mock_load_model, mock_mqtt_cls):
    """Pipeline handles capture RuntimeError gracefully and continues."""
    mock_mqtt_cls.return_value = MagicMock()

    mock_backend = MagicMock()
    mock_backend.infer.return_value = np.zeros((1, 84, 0), dtype=np.float32)
    mock_load_model.return_value = {"backend": mock_backend, "type": "opencv"}

    mock_capture = MagicMock()
    read_calls = [0]

    def fake_read():
        read_calls[0] += 1
        if read_calls[0] == 1:
            raise RuntimeError("Camera glitch")
        raise StopIteration("Done")

    mock_capture.read.side_effect = fake_read
    mock_build_cap.return_value = mock_capture

    tmpdir = tempfile.mkdtemp()
    config = _make_pipeline_config(tmpdir)
    args = argparse.Namespace(replay_dir="/fake", detection_backend="opencv")

    run_pipeline(config, args)
    assert read_calls[0] >= 2


# ---------------------------------------------------------------------------
# main() argument parsing
# ---------------------------------------------------------------------------


def test_main_missing_config_exits():
    """main() should fail if --config is not provided."""
    with patch("sys.argv", ["main.py"]):
        with pytest.raises(SystemExit):
            from src.main import main
            main()


@patch("src.main.run_pipeline")
@patch("src.main.load_config")
def test_main_loads_config_and_runs(mock_load_config, mock_run_pipeline):
    """main() loads config and calls run_pipeline."""
    mock_load_config.return_value = {
        "device": {"id": "test"},
        "logging": {"level": "INFO"},
    }

    tmpdir = tempfile.mkdtemp()
    config_path = str(Path(tmpdir) / "config.yaml")
    Path(config_path).write_text("device:\n  id: test\n")

    with patch("sys.argv", ["main.py", "--config", config_path]):
        from src.main import main
        main()

    mock_load_config.assert_called_once_with(config_path)
    mock_run_pipeline.assert_called_once()

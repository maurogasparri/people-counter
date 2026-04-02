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

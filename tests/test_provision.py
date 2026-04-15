"""Tests for device provisioning script."""
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import provision module parts directly
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from provision import (
    PROVISION_DIR,
    _build_config,
    cmd_create,
    cmd_deploy,
    cmd_list,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "device_id": "store-001-cam-01",
        "store_id": "store-001",
        "store_name": "Test Store",
        "endpoint": "test.iot.us-east-1.amazonaws.com",
        "skip_aws": True,
        "force": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _build_config
# ---------------------------------------------------------------------------


def test_build_config_generates_yaml():
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "device"
    device_dir.mkdir()

    args = _make_args()

    with patch("provision.CONFIG_TEMPLATE", Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"):
        _build_config(device_dir, args)

    config_path = device_dir / "config.yaml"
    assert config_path.exists()

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["device"]["id"] == "store-001-cam-01"
    assert config["device"]["store_id"] == "store-001"
    assert config["device"]["store_name"] == "Test Store"
    assert config["mqtt"]["endpoint"] == "test.iot.us-east-1.amazonaws.com"


def test_build_config_sets_remote_paths():
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "device"
    device_dir.mkdir()

    args = _make_args()

    with patch("provision.CONFIG_TEMPLATE", Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"):
        _build_config(device_dir, args)

    import yaml

    with open(device_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    assert config["mqtt"]["cert_path"] == "/etc/people-counter/certs/device.pem.crt"
    assert config["mqtt"]["key_path"] == "/etc/people-counter/certs/device.pem.key"
    assert config["mqtt"]["ca_path"] == "/etc/people-counter/certs/AmazonRootCA1.pem"
    assert config["buffer"]["db_path"] == "/var/lib/people-counter/buffer.db"
    assert config["logging"]["file"] == "/var/log/people-counter/app.log"


# ---------------------------------------------------------------------------
# cmd_create
# ---------------------------------------------------------------------------


def test_create_skip_aws():
    tmpdir = tempfile.mkdtemp()
    args = _make_args(skip_aws=True)

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with patch("provision.CONFIG_TEMPLATE", Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"):
            cmd_create(args)

    device_dir = Path(tmpdir) / "store-001-cam-01"
    assert device_dir.exists()
    assert (device_dir / "config.yaml").exists()
    assert (device_dir / "metadata.json").exists()
    assert (device_dir / "certs" / "device.pem.crt").exists()
    assert (device_dir / "certs" / "device.pem.key").exists()
    assert (device_dir / "certs" / "AmazonRootCA1.pem").exists()


def test_create_metadata():
    tmpdir = tempfile.mkdtemp()
    args = _make_args(skip_aws=True, store_name="TestStore")

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with patch("provision.CONFIG_TEMPLATE", Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"):
            cmd_create(args)

    meta = json.loads((Path(tmpdir) / "store-001-cam-01" / "metadata.json").read_text())
    assert meta["device_id"] == "store-001-cam-01"
    assert meta["store_id"] == "store-001"
    assert meta["store_name"] == "TestStore"


def test_create_fails_if_exists():
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    device_dir.mkdir()

    args = _make_args(skip_aws=True, force=False)

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with pytest.raises(SystemExit):
            cmd_create(args)


def test_create_force_overwrites():
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    device_dir.mkdir()

    args = _make_args(skip_aws=True, force=True)

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with patch("provision.CONFIG_TEMPLATE", Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"):
            cmd_create(args)

    assert (device_dir / "config.yaml").exists()


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


def test_list_empty(capsys):
    tmpdir = tempfile.mkdtemp()

    with patch("provision.PROVISION_DIR", Path(tmpdir) / "nonexistent"):
        cmd_list(argparse.Namespace())

    # Should not raise


def test_list_shows_devices(capsys):
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    device_dir.mkdir()
    (device_dir / "metadata.json").write_text(json.dumps({
        "device_id": "store-001-cam-01",
        "store_id": "store-001",
        "store_name": "Test Store",
        "endpoint": "test.iot.amazonaws.com",
    }))

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        cmd_list(argparse.Namespace())

    captured = capsys.readouterr()
    assert "store-001-cam-01" in captured.out
    assert "store-001" in captured.out


# ---------------------------------------------------------------------------
# cmd_deploy
# ---------------------------------------------------------------------------


@patch("provision._scp")
@patch("provision._ssh")
def test_deploy_calls_ssh_and_scp(mock_ssh, mock_scp):
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    cert_dir = device_dir / "certs"
    cert_dir.mkdir(parents=True)
    (device_dir / "config.yaml").write_text("device: {}")
    (cert_dir / "device.pem.crt").write_text("cert")
    (cert_dir / "device.pem.key").write_text("key")
    (cert_dir / "AmazonRootCA1.pem").write_text("ca")

    args = argparse.Namespace(
        device_id="store-001-cam-01",
        host="people-counter.local",
        user="pi",
    )

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        cmd_deploy(args)

    # Should have called SSH to create dirs and set permissions
    assert mock_ssh.call_count >= 2
    # Should have copied config + 3 cert files
    assert mock_scp.call_count >= 4


def test_deploy_fails_if_not_provisioned():
    tmpdir = tempfile.mkdtemp()

    args = argparse.Namespace(
        device_id="nonexistent-device",
        host="people-counter.local",
        user="pi",
    )

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with pytest.raises(SystemExit):
            cmd_deploy(args)

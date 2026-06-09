"""Tests para el script de provisioning del device."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import provision module parts directly
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from provision import (
    _build_config,
    cmd_create,
    cmd_deploy,
    cmd_harvest,
    cmd_list,
    cmd_reprovision,
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
        # Metadata de dimensiones + seed RDS (skip_db: no tocar RDS en tests).
        "latitude": None,
        "longitude": None,
        "timezone": None,
        "address": None,
        "cam_label": None,
        "skip_db": True,
        "stack_name": "people-counter-dev",
        "region": "us-east-1",
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

    with patch(
        "provision.CONFIG_TEMPLATE",
        Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
    ):
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
    """The cert paths come from REMOTE_CERT_DIR. buffer.db_path and
    logging.file are not rewritten per-device — they fall back to the
    canonical install convention from ``config/config.example.yaml``
    (same path), so they don't appear in the per-device config.yaml at
    all.
    """
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "device"
    device_dir.mkdir()

    args = _make_args()

    with patch(
        "provision.CONFIG_TEMPLATE",
        Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
    ):
        _build_config(device_dir, args)

    import yaml

    with open(device_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    assert config["mqtt"]["cert_path"] == "/etc/people-counter/certs/device.pem.crt"
    assert config["mqtt"]["key_path"] == "/etc/people-counter/certs/device.pem.key"
    assert config["mqtt"]["ca_path"] == "/etc/people-counter/certs/AmazonRootCA1.pem"
    # After the hardware↔config unification, the per-device YAML is a copy
    # of the bundled defaults with device + mqtt overridden. The buffer +
    # logging blocks ride along carrying their fleet-wide install paths;
    # operators are free to keep the defaults or edit per-site.
    assert config["buffer"]["db_path"] == "/var/lib/people-counter/buffer.db"
    assert config["logging"]["file"] == "/var/log/people-counter/app.log"


# ---------------------------------------------------------------------------
# cmd_create
# ---------------------------------------------------------------------------


def test_create_skip_aws():
    tmpdir = tempfile.mkdtemp()
    args = _make_args(skip_aws=True)

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with patch(
            "provision.CONFIG_TEMPLATE",
            Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
        ):
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
        with patch(
            "provision.CONFIG_TEMPLATE",
            Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
        ):
            cmd_create(args)

    meta = json.loads((Path(tmpdir) / "store-001-cam-01" / "metadata.json").read_text())
    assert meta["device_id"] == "store-001-cam-01"
    assert meta["store_id"] == "store-001"
    assert meta["store_name"] == "TestStore"


def test_create_refreshes_config_on_existing_device_without_force():
    """``create`` sin --force sobre un device existente refresca el
    config + metadata desde el template, sin tocar los certs ni
    contactar AWS. Es la operación normal para sincronizar el
    provisioned local con un template editado entre corridas."""
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    cert_dir = device_dir / "certs"
    cert_dir.mkdir(parents=True)
    # Cert pre-existente — el refresh NO lo debe tocar.
    (cert_dir / "device.pem.crt").write_text("EXISTING_CERT_DO_NOT_TOUCH")

    args = _make_args(skip_aws=True, force=False)

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with patch(
            "provision.CONFIG_TEMPLATE",
            Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
        ):
            cmd_create(args)

    # Config y metadata regenerados.
    assert (device_dir / "config.yaml").exists()
    assert (device_dir / "metadata.json").exists()
    # Cert preservado.
    assert (cert_dir / "device.pem.crt").read_text() == "EXISTING_CERT_DO_NOT_TOUCH"


def test_create_force_overwrites():
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    device_dir.mkdir()

    args = _make_args(skip_aws=True, force=True)

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with patch(
            "provision.CONFIG_TEMPLATE",
            Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
        ):
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
    (device_dir / "metadata.json").write_text(
        json.dumps(
            {
                "device_id": "store-001-cam-01",
                "store_id": "store-001",
                "store_name": "Test Store",
                "endpoint": "test.iot.amazonaws.com",
            }
        )
    )

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


@patch("provision._scp")
@patch("provision._ssh")
def test_deploy_pushes_calibration_when_present(mock_ssh, mock_scp):
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    cert_dir = device_dir / "certs"
    cert_dir.mkdir(parents=True)
    (device_dir / "config.yaml").write_text("device: {}")
    (device_dir / "calibration.npz").write_bytes(b"fake npz")
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

    npz_calls = [c for c in mock_scp.call_args_list if "calibration.npz" in str(c)]
    assert len(npz_calls) == 1


@patch("provision._scp")
@patch("provision._ssh")
def test_deploy_skips_calibration_when_absent(mock_ssh, mock_scp):
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

    npz_calls = [c for c in mock_scp.call_args_list if "calibration.npz" in str(c)]
    assert len(npz_calls) == 0


# ---------------------------------------------------------------------------
# cmd_harvest
# ---------------------------------------------------------------------------


@patch("provision._scp")
def test_harvest_pulls_calibration(mock_scp):
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    device_dir.mkdir(parents=True)

    args = argparse.Namespace(
        device_id="store-001-cam-01",
        host="people-counter.local",
        user="pi",
    )

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        cmd_harvest(args)

    mock_scp.assert_called_once()
    call_args = mock_scp.call_args[0]
    assert "people-counter.local:/etc/people-counter/calibration.npz" in call_args[0]
    assert str(device_dir / "calibration.npz") == call_args[1]


def test_harvest_fails_if_not_provisioned():
    tmpdir = tempfile.mkdtemp()
    args = argparse.Namespace(
        device_id="nonexistent",
        host="people-counter.local",
        user="pi",
    )

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with pytest.raises(SystemExit):
            cmd_harvest(args)


# ---------------------------------------------------------------------------
# cmd_reprovision
# ---------------------------------------------------------------------------


@patch("provision._issue_cert")
@patch("provision._revoke_certs")
def test_reprovision_revokes_then_issues(mock_revoke, mock_issue):
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    cert_dir = device_dir / "certs"
    cert_dir.mkdir(parents=True)
    (cert_dir / "device.pem.crt").write_text("old cert")

    mock_revoke.return_value = 1

    args = argparse.Namespace(device_id="store-001-cam-01")

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        cmd_reprovision(args)

    mock_revoke.assert_called_once_with("store-001-cam-01")
    mock_issue.assert_called_once()
    # Old certs archived under a sibling certs.old-* dir
    archived = list(device_dir.glob("certs.old-*"))
    assert len(archived) == 1
    assert (archived[0] / "device.pem.crt").exists()


@patch("provision._issue_cert")
@patch("provision._revoke_certs")
def test_reprovision_handles_no_existing_certs(mock_revoke, mock_issue):
    tmpdir = tempfile.mkdtemp()
    device_dir = Path(tmpdir) / "store-001-cam-01"
    device_dir.mkdir(parents=True)

    mock_revoke.return_value = 0

    args = argparse.Namespace(device_id="store-001-cam-01")

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        cmd_reprovision(args)

    mock_revoke.assert_called_once()
    mock_issue.assert_called_once()


def test_reprovision_fails_if_not_provisioned():
    tmpdir = tempfile.mkdtemp()
    args = argparse.Namespace(device_id="nonexistent")

    with patch("provision.PROVISION_DIR", Path(tmpdir)):
        with pytest.raises(SystemExit):
            cmd_reprovision(args)

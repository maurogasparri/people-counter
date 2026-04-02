#!/usr/bin/env python3
"""Device provisioning tool for People Counter edge devices.

Creates the AWS IoT Core thing, generates X.509 certificates, builds
the device-specific config YAML, and optionally deploys to the device
via SSH.

Usage:
    # Provision a new device (creates thing + certs + config)
    python scripts/provision.py create \
        --device-id store-001-cam-01 \
        --store-id store-001 \
        --store-name "Abasto Shopping" \
        --endpoint xxxxx.iot.us-east-1.amazonaws.com

    # Deploy config and certs to a device via SSH
    python scripts/provision.py deploy \
        --device-id store-001-cam-01 \
        --host people-counter.local \
        --user pi

    # List all provisioned devices
    python scripts/provision.py list
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("provision")

# Base directories
PROVISION_DIR = Path(__file__).resolve().parent.parent / "provisioned"
CONFIG_TEMPLATE = Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"

# Remote paths on the device
REMOTE_CONFIG_DIR = "/etc/people-counter"
REMOTE_CERT_DIR = "/etc/people-counter/certs"
REMOTE_DATA_DIR = "/var/lib/people-counter"
REMOTE_LOG_DIR = "/var/log/people-counter"


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new device: register IoT thing, generate certs, build config."""
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id
    cert_dir = device_dir / "certs"

    if device_dir.exists() and not args.force:
        logger.error(
            "Device %s already provisioned at %s. Use --force to overwrite.",
            device_id,
            device_dir,
        )
        sys.exit(1)

    device_dir.mkdir(parents=True, exist_ok=True)
    cert_dir.mkdir(parents=True, exist_ok=True)

    # --- Register IoT Thing ---
    if not args.skip_aws:
        _create_iot_thing(device_id, cert_dir, args.endpoint)
    else:
        logger.warning("Skipping AWS IoT registration (--skip-aws)")
        # Create placeholder cert files for testing
        for name in ["device.pem.crt", "device.pem.key", "AmazonRootCA1.pem"]:
            placeholder = cert_dir / name
            if not placeholder.exists():
                placeholder.write_text(f"# Placeholder — replace with real {name}\n")

    # --- Build config YAML ---
    _build_config(device_dir, args)

    # --- Save device metadata ---
    metadata = {
        "device_id": device_id,
        "store_id": args.store_id,
        "store_name": args.store_name,
        "endpoint": args.endpoint,
    }
    (device_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("Device %s provisioned at %s", device_id, device_dir)


def _create_iot_thing(device_id: str, cert_dir: Path, endpoint: str) -> None:
    """Register IoT thing and generate certificates via AWS CLI."""
    try:
        # Create the thing
        subprocess.run(
            ["aws", "iot", "create-thing", "--thing-name", device_id],
            check=True,
            capture_output=True,
        )
        logger.info("IoT thing created: %s", device_id)

        # Create keys and certificate
        result = subprocess.run(
            [
                "aws", "iot", "create-keys-and-certificate",
                "--set-as-active",
                "--certificate-pem-outfile", str(cert_dir / "device.pem.crt"),
                "--private-key-outfile", str(cert_dir / "device.pem.key"),
                "--public-key-outfile", str(cert_dir / "device.pem.pub"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cert_response = json.loads(result.stdout)
        cert_arn = cert_response["certificateArn"]
        logger.info("Certificate created: %s", cert_arn)

        # Attach policy to certificate
        subprocess.run(
            [
                "aws", "iot", "attach-policy",
                "--policy-name", "people-counter-device-policy",
                "--target", cert_arn,
            ],
            check=True,
            capture_output=True,
        )

        # Attach certificate to thing
        subprocess.run(
            [
                "aws", "iot", "attach-thing-principal",
                "--thing-name", device_id,
                "--principal", cert_arn,
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Certificate attached to thing %s", device_id)

        # Download Amazon Root CA
        subprocess.run(
            [
                "curl", "-s", "-o", str(cert_dir / "AmazonRootCA1.pem"),
                "https://www.amazontrust.com/repository/AmazonRootCA1.pem",
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Root CA downloaded")

        # Save cert ARN for future reference
        (cert_dir / "cert_arn.txt").write_text(cert_arn)

    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("AWS CLI error: %s", e.stderr)
        sys.exit(1)


def _build_config(device_dir: Path, args: argparse.Namespace) -> None:
    """Build device-specific config.yaml from template."""
    import yaml

    with open(CONFIG_TEMPLATE) as f:
        config = yaml.safe_load(f)

    # Device identity
    config["device"]["id"] = args.device_id
    config["device"]["store_id"] = args.store_id
    config["device"]["store_name"] = args.store_name

    # MQTT
    config["mqtt"]["endpoint"] = args.endpoint
    config["mqtt"]["cert_path"] = f"{REMOTE_CERT_DIR}/device.pem.crt"
    config["mqtt"]["key_path"] = f"{REMOTE_CERT_DIR}/device.pem.key"
    config["mqtt"]["ca_path"] = f"{REMOTE_CERT_DIR}/AmazonRootCA1.pem"

    # Buffer
    config["buffer"]["db_path"] = f"{REMOTE_DATA_DIR}/buffer.db"

    # Logging
    config["logging"]["file"] = f"{REMOTE_LOG_DIR}/app.log"

    config_path = device_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Config written to %s", config_path)


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy config and certs to a device via SSH."""
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id

    if not device_dir.exists():
        logger.error("Device %s not provisioned. Run 'create' first.", device_id)
        sys.exit(1)

    host = f"{args.user}@{args.host}"

    # Create remote directories
    _ssh(host, f"sudo mkdir -p {REMOTE_CONFIG_DIR} {REMOTE_CERT_DIR} {REMOTE_DATA_DIR} {REMOTE_LOG_DIR}")
    _ssh(host, f"sudo chown -R {args.user}:{args.user} {REMOTE_CONFIG_DIR} {REMOTE_DATA_DIR} {REMOTE_LOG_DIR}")

    # Copy config
    _scp(str(device_dir / "config.yaml"), f"{host}:{REMOTE_CONFIG_DIR}/config.yaml")

    # Copy certs
    for cert_file in (device_dir / "certs").glob("*.pem*"):
        if cert_file.suffix in (".crt", ".key", ".pem"):
            _scp(str(cert_file), f"{host}:{REMOTE_CERT_DIR}/{cert_file.name}")

    # Set cert permissions
    _ssh(host, f"chmod 600 {REMOTE_CERT_DIR}/device.pem.key")
    _ssh(host, f"chmod 644 {REMOTE_CERT_DIR}/device.pem.crt {REMOTE_CERT_DIR}/AmazonRootCA1.pem")

    # Install systemd services and logrotate
    config_dir = Path(__file__).resolve().parent.parent / "config"
    for config_file in [
        "people-counter.service",
        "people-counter-reset.service",
        "people-counter-reset.timer",
    ]:
        src = config_dir / config_file
        if src.exists():
            _scp(str(src), f"{host}:/tmp/{config_file}")
            _ssh(host, f"sudo mv /tmp/{config_file} /etc/systemd/system/")

    logrotate = config_dir / "logrotate.conf"
    if logrotate.exists():
        _scp(str(logrotate), f"{host}:/tmp/people-counter-logrotate")
        _ssh(host, "sudo mv /tmp/people-counter-logrotate /etc/logrotate.d/people-counter")

    _ssh(host, "sudo systemctl daemon-reload")
    _ssh(host, "sudo systemctl enable people-counter people-counter-reset.timer")
    logger.info("Systemd services and logrotate installed")

    logger.info("Device %s deployed to %s", device_id, args.host)


def cmd_list(args: argparse.Namespace) -> None:
    """List all provisioned devices."""
    if not PROVISION_DIR.exists():
        logger.info("No devices provisioned yet.")
        return

    for device_dir in sorted(PROVISION_DIR.iterdir()):
        if not device_dir.is_dir():
            continue
        meta_file = device_dir / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            print(
                f"  {meta['device_id']:30s} "
                f"store={meta['store_id']:15s} "
                f"{meta.get('store_name', '')}"
            )
        else:
            print(f"  {device_dir.name:30s} (no metadata)")


def _ssh(host: str, command: str) -> None:
    """Run a command on a remote host via SSH."""
    try:
        subprocess.run(
            ["ssh", host, command],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("SSH error on '%s': %s", command, e.stderr.strip())
        raise


def _scp(local: str, remote: str) -> None:
    """Copy a file to a remote host via SCP."""
    try:
        subprocess.run(
            ["scp", local, remote],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("SCP error: %s", e.stderr.strip())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="People Counter device provisioning")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- create ---
    p_create = sub.add_parser("create", help="Provision a new device")
    p_create.add_argument("--device-id", required=True, help="Unique device ID")
    p_create.add_argument("--store-id", required=True, help="Store identifier")
    p_create.add_argument("--store-name", default="", help="Human-readable store name")
    p_create.add_argument(
        "--endpoint",
        default="xxxxx.iot.us-east-1.amazonaws.com",
        help="AWS IoT Core endpoint",
    )
    p_create.add_argument("--skip-aws", action="store_true", help="Skip AWS IoT registration")
    p_create.add_argument("--force", action="store_true", help="Overwrite existing")
    p_create.set_defaults(func=cmd_create)

    # --- deploy ---
    p_deploy = sub.add_parser("deploy", help="Deploy config and certs to device")
    p_deploy.add_argument("--device-id", required=True)
    p_deploy.add_argument("--host", required=True, help="Device hostname or IP")
    p_deploy.add_argument("--user", default="pi", help="SSH user")
    p_deploy.set_defaults(func=cmd_deploy)

    # --- list ---
    p_list = sub.add_parser("list", help="List provisioned devices")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

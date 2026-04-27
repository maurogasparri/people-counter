#!/usr/bin/env python3
"""Device provisioning tool for People Counter edge devices.

Creates the AWS IoT Core thing, generates X.509 certificates, builds
the device-specific config YAML, and optionally deploys to the device
via SSH. Also covers disaster recovery: pulling calibration back to
the workstation backup, and re-issuing certs after SD failure.

Usage:
    # Provision a new device (creates thing + certs + config)
    python scripts/provision.py create \
        --device-id store-001-cam-01 \
        --store-id store-001 \
        --store-name "Store Name" \
        --endpoint xxxxx.iot.us-east-1.amazonaws.com

    # Deploy config + certs (+ calibration.npz if backed up) to a device
    python scripts/provision.py deploy \
        --device-id store-001-cam-01 \
        --host people-counter.local \
        --user pi

    # Pull calibration.npz from a device into the workstation backup
    python scripts/provision.py harvest \
        --device-id store-001-cam-01 \
        --host people-counter.local

    # Re-issue cert (revokes the old one in IoT Core). Use after SD failure.
    python scripts/provision.py reprovision \
        --device-id store-001-cam-01

    # List all provisioned devices
    python scripts/provision.py list
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
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
        _create_thing(device_id)
        _issue_cert(device_id, cert_dir)
    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("AWS CLI error: %s", e.stderr)
        sys.exit(1)


def _create_thing(device_id: str) -> None:
    """Register an IoT thing. Idempotent: skips if already exists."""
    try:
        subprocess.run(
            ["aws", "iot", "create-thing", "--thing-name", device_id],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("IoT thing created: %s", device_id)
    except subprocess.CalledProcessError as e:
        if "ResourceAlreadyExistsException" in (e.stderr or ""):
            logger.info("IoT thing %s already exists, skipping creation", device_id)
        else:
            raise


def _issue_cert(device_id: str, cert_dir: Path) -> None:
    """Generate keys + cert, attach policy, attach to thing. Thing must exist.

    Writes device.pem.crt / device.pem.key / device.pem.pub / AmazonRootCA1.pem
    / cert_arn.txt into cert_dir.
    """
    cert_dir.mkdir(parents=True, exist_ok=True)

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
    cert_arn = json.loads(result.stdout)["certificateArn"]
    logger.info("Certificate created: %s", cert_arn)

    subprocess.run(
        [
            "aws", "iot", "attach-policy",
            "--policy-name", "people-counter-device-policy",
            "--target", cert_arn,
        ],
        check=True, capture_output=True,
    )
    subprocess.run(
        [
            "aws", "iot", "attach-thing-principal",
            "--thing-name", device_id,
            "--principal", cert_arn,
        ],
        check=True, capture_output=True,
    )
    logger.info("Certificate attached to thing %s", device_id)

    subprocess.run(
        [
            "curl", "-s", "-o", str(cert_dir / "AmazonRootCA1.pem"),
            "https://www.amazontrust.com/repository/AmazonRootCA1.pem",
        ],
        check=True, capture_output=True,
    )
    logger.info("Root CA downloaded")

    (cert_dir / "cert_arn.txt").write_text(cert_arn)


def _revoke_certs(device_id: str) -> int:
    """Detach + deactivate + delete every cert currently attached to the thing.

    Returns the count of certs revoked. Safe to call when there are none.
    """
    result = subprocess.run(
        ["aws", "iot", "list-thing-principals", "--thing-name", device_id],
        check=True, capture_output=True, text=True,
    )
    principals = json.loads(result.stdout).get("principals", [])

    for arn in principals:
        cert_id = arn.split("/")[-1]
        logger.info("Revoking cert %s", cert_id)

        subprocess.run(
            ["aws", "iot", "detach-thing-principal",
             "--thing-name", device_id, "--principal", arn],
            check=True, capture_output=True,
        )

        # Detach every policy attached to the cert (don't assume just one)
        pols = subprocess.run(
            ["aws", "iot", "list-attached-policies", "--target", arn],
            check=True, capture_output=True, text=True,
        )
        for pol in json.loads(pols.stdout).get("policies", []):
            subprocess.run(
                ["aws", "iot", "detach-policy",
                 "--policy-name", pol["policyName"], "--target", arn],
                check=True, capture_output=True,
            )

        subprocess.run(
            ["aws", "iot", "update-certificate",
             "--certificate-id", cert_id, "--new-status", "INACTIVE"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["aws", "iot", "delete-certificate", "--certificate-id", cert_id],
            check=True, capture_output=True,
        )

    return len(principals)


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

    # Push calibration.npz if a backup exists for this device
    calibration = device_dir / "calibration.npz"
    if calibration.exists():
        _scp(str(calibration), f"{host}:{REMOTE_CONFIG_DIR}/calibration.npz")
        logger.info("Calibration deployed from backup")

    # Install systemd services and logrotate
    config_dir = Path(__file__).resolve().parent.parent / "config"
    for config_file in [
        "wifi-monitor.service",
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
    _ssh(host, "sudo systemctl enable wifi-monitor people-counter people-counter-reset.timer")
    logger.info("Systemd services and logrotate installed")

    logger.info("Device %s deployed to %s", device_id, args.host)


def cmd_harvest(args: argparse.Namespace) -> None:
    """Pull calibration.npz from a device into the workstation backup."""
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id

    if not device_dir.exists():
        logger.error("Device %s not provisioned. Run 'create' first.", device_id)
        sys.exit(1)

    host = f"{args.user}@{args.host}"
    remote = f"{REMOTE_CONFIG_DIR}/calibration.npz"
    local = device_dir / "calibration.npz"

    _scp(f"{host}:{remote}", str(local))
    logger.info("Calibration harvested to %s", local)


def cmd_reprovision(args: argparse.Namespace) -> None:
    """Re-issue cert for an existing thing. Revokes the old cert first.

    Use after SD failure or whenever you suspect the device cert is
    compromised. The thing keeps its identity; only the principal rotates.
    """
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id
    cert_dir = device_dir / "certs"

    if not device_dir.exists():
        logger.error("Device %s not provisioned. Run 'create' first.", device_id)
        sys.exit(1)

    # Move the old cert dir aside before overwriting (in case we need to dig)
    if cert_dir.exists() and any(cert_dir.iterdir()):
        archived = cert_dir.parent / f"certs.old-{int(time.time())}"
        cert_dir.rename(archived)
        logger.info("Old certs archived to %s", archived)

    try:
        revoked = _revoke_certs(device_id)
        logger.info("Revoked %d old cert(s) in IoT Core", revoked)
        _issue_cert(device_id, cert_dir)
    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("AWS CLI error: %s", e.stderr)
        sys.exit(1)

    logger.info(
        "Device %s reprovisioned. Run 'deploy' to push the new cert.",
        device_id,
    )


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

    # --- harvest ---
    p_harvest = sub.add_parser(
        "harvest",
        help="Pull calibration.npz from a device into the workstation backup",
    )
    p_harvest.add_argument("--device-id", required=True)
    p_harvest.add_argument("--host", required=True, help="Device hostname or IP")
    p_harvest.add_argument("--user", default="pi", help="SSH user")
    p_harvest.set_defaults(func=cmd_harvest)

    # --- reprovision ---
    p_reprov = sub.add_parser(
        "reprovision",
        help="Re-issue cert for an existing thing (revokes the old cert)",
    )
    p_reprov.add_argument("--device-id", required=True)
    p_reprov.set_defaults(func=cmd_reprovision)

    # --- list ---
    p_list = sub.add_parser("list", help="List provisioned devices")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

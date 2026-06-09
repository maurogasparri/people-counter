#!/usr/bin/env python3
"""Tool de provisioning de devices para los edge devices del People Counter.

Crea el thing en AWS IoT Core, genera certificados X.509, arma el YAML de
config específico del device, y opcionalmente despliega al device por SSH.
También cubre disaster recovery: traer la calibración de vuelta al backup
del workstation, y re-emitir certs luego de falla de SD.

Uso:
    # Provisiona un device nuevo (crea thing + certs + config + seed RDS)
    python scripts/provision.py create \
        --device-id store-001-cam-01 \
        --store-id store-001 \
        --store-name "Store Name" \
        --latitude -34.5669426 --longitude -58.4766207 \
        --timezone America/Argentina/Buenos_Aires \
        --endpoint xxxxx.iot.us-east-1.amazonaws.com
    # (lat/long van a la tabla `sites` en RDS, no al config del device.
    #  Requiere 'pip install .[provisioning]' + creds AWS; --skip-db para omitir.)

    # Despliega config + certs (+ calibration.npz si está backupeado) a un device
    python scripts/provision.py deploy \
        --device-id store-001-cam-01 \
        --host people-counter.local \
        --user pi

    # Trae calibration.npz desde un device al backup del workstation
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
CONFIG_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"
)

# Remote paths on the device
REMOTE_CONFIG_DIR = "/etc/people-counter"
REMOTE_CERT_DIR = "/etc/people-counter/certs"
REMOTE_DATA_DIR = "/var/lib/people-counter"
REMOTE_LOG_DIR = "/var/log/people-counter"

# AWS defaults — matchean infra/deploy.ps1 (stack people-counter-$Environment).
DEFAULT_STACK_NAME = "people-counter-dev"
DEFAULT_REGION = "us-east-1"


def cmd_create(args: argparse.Namespace) -> None:
    """Provisiona un device (certs + config + metadata).

    Comportamiento idempotente respecto del template — re-correr ``create``
    sobre un device existente refresca el config desde
    ``config/config.example.yaml`` sin tocar los certs. Eso permite que el
    operador edite el template y sincronice el provisioned local con un
    ``create`` plano, sin riesgo de rotar certs (operación destructiva en
    AWS IoT).

    Reglas:
      - Device nuevo: registra IoT thing + emite cert + arma config + metadata.
      - Device existente sin ``--force``: solo refresca config + metadata.
        Certs intactos. Si necesitás rotar certs, usá ``reprovision`` (no
        ``create --force``, que es más destructivo).
      - Device existente con ``--force``: regenera TODO incluído cert
        (rotación implícita). Útil para empezar de cero sobre un device
        comprometido. Equivale a ``reprovision`` + refresh de config en
        un solo comando.
    """
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id
    cert_dir = device_dir / "certs"

    existing = device_dir.exists()
    config_refresh_only = existing and not args.force

    device_dir.mkdir(parents=True, exist_ok=True)
    cert_dir.mkdir(parents=True, exist_ok=True)

    # --- Registrar IoT Thing + emitir cert ---
    # En modo refresh saltamos toda la interacción con AWS — el thing y
    # el cert ya existen y son válidos. El operador puede usar
    # ``reprovision`` si necesita rotar el cert por separado.
    if config_refresh_only:
        logger.info(
            "Device %s ya existe — refreshing config desde el template "
            "(certs intactos; usar 'reprovision' para rotarlos, o "
            "'create --force' para rotación + refresh en uno)",
            device_id,
        )
    elif not args.skip_aws:
        _create_iot_thing(device_id, cert_dir, args.endpoint, args.policy_name)
    else:
        logger.warning("Skipping AWS IoT registration (--skip-aws)")
        # Crear archivos de cert placeholder para testing
        for name in ["device.pem.crt", "device.pem.key", "AmazonRootCA1.pem"]:
            placeholder = cert_dir / name
            if not placeholder.exists():
                placeholder.write_text(f"# Placeholder — replace with real {name}\n")

    # --- Armar config YAML --- (siempre, incluso en refresh)
    _build_config(device_dir, args)

    # --- Guardar metadata del device --- (siempre)
    # lat/long se guardan acá como registro local del provisioning, pero NO van
    # al config.yaml del device (el device no necesita saber su ubicación) — su
    # fuente de verdad es la tabla `sites` en RDS, que sembramos abajo.
    metadata = {
        "device_id": device_id,
        "store_id": args.store_id,
        "store_name": args.store_name,
        "endpoint": args.endpoint,
        "latitude": args.latitude,
        "longitude": args.longitude,
    }
    (device_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- Seed de dimensiones en RDS (sites + devices) ---
    # Idempotente y no fatal: si RDS no es alcanzable, se saltea con un warning
    # y el provisioning del device igual queda OK.
    if not args.skip_db:
        _seed_dimensions(args)

    if config_refresh_only:
        logger.info("Config refreshed at %s (certs unchanged)", device_dir)
    else:
        logger.info("Device %s provisioned at %s", device_id, device_dir)


def _rds_connect(stack_name: str, region: str):
    """Abre una conexión psycopg a RDS como master user, SIN docker.

    Reusa el stack que ya usa la Lambda (psycopg + boto3): toma el endpoint y el
    ARN del secret de los outputs del stack CloudFormation, lee el master secret
    de Secrets Manager y conecta vía psycopg con SSL. No depende de docker ni de
    un psql instalado — solo de boto3 + psycopg (``pip install
    '.[provisioning]'``), que es cross-platform.
    """
    import boto3  # lazy: solo para el seed de la DB
    import psycopg

    cfn = boto3.client("cloudformation", region_name=region)
    stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks.get("Outputs", [])}
    host = outputs["RdsEndpoint"]
    port = int(outputs.get("RdsPort", "5432"))
    secret_arn = outputs["RdsMasterSecretArn"]

    sm = boto3.client("secretsmanager", region_name=region)
    secret = json.loads(sm.get_secret_value(SecretId=secret_arn)["SecretString"])

    return psycopg.connect(
        host=host,
        port=port,
        dbname="people_counter",
        user=secret["username"],
        password=secret["password"],
        # RDS force_ssl=1 ya encripta; require (no verify-full) evita tener que
        # empaquetar el CA bundle en el workstation de provisioning.
        sslmode="require",
        connect_timeout=15,
    )


def _seed_dimensions(args: argparse.Namespace) -> None:
    """UPSERT del site + device en RDS (tablas de dimensiones sites/devices).

    Las coordenadas (lat/long) viven SOLO acá, no en el config del device — son
    metadata cloud-side para el geomap y los dropdowns de Grafana. Idempotente:
    re-correr refresca los campos sin duplicar (ON CONFLICT). No fatal: si RDS no
    es alcanzable (sin creds AWS, stack no deployado, etc.) loguea un warning y
    sigue — el provisioning del device no se bloquea.
    """
    try:
        conn = _rds_connect(args.stack_name, args.region)
    except Exception as e:
        logger.warning(
            "Seed de sites/devices SALTEADO (no se pudo conectar a RDS: %s). "
            "Correr el UPSERT a mano cuando la DB esté disponible.",
            e,
        )
        return

    try:
        with conn, conn.cursor() as cur:
            # Site primero (devices.store_id tiene FK a sites). COALESCE en el
            # UPDATE evita que un refresh sin coords borre las ya cargadas.
            cur.execute(
                """
                INSERT INTO sites (store_id, store_name, latitude, longitude,
                                   timezone, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (store_id) DO UPDATE SET
                    store_name = EXCLUDED.store_name,
                    latitude   = COALESCE(EXCLUDED.latitude,  sites.latitude),
                    longitude  = COALESCE(EXCLUDED.longitude, sites.longitude),
                    timezone   = COALESCE(EXCLUDED.timezone,  sites.timezone),
                    updated_at = now();
                """,
                (
                    args.store_id,
                    args.store_name,
                    args.latitude,
                    args.longitude,
                    args.timezone,
                ),
            )
            cur.execute(
                """
                INSERT INTO devices (device_id, store_id, cam_label,
                                     installed_at, updated_at)
                VALUES (%s, %s, %s, now(), now())
                ON CONFLICT (device_id) DO UPDATE SET
                    store_id   = EXCLUDED.store_id,
                    cam_label  = COALESCE(EXCLUDED.cam_label, devices.cam_label),
                    updated_at = now();
                """,
                (args.device_id, args.store_id, args.cam_label),
            )
        logger.info(
            "RDS dimensiones: site %s + device %s upserted (lat=%s lon=%s)",
            args.store_id,
            args.device_id,
            args.latitude,
            args.longitude,
        )
    except Exception as e:
        logger.warning("Seed de sites/devices falló durante el UPSERT: %s", e)
    finally:
        conn.close()


def _create_iot_thing(
    device_id: str, cert_dir: Path, endpoint: str, policy_name: str
) -> None:
    """Registra el IoT thing y genera certificados via AWS CLI."""
    try:
        _create_thing(device_id)
        _issue_cert(device_id, cert_dir, policy_name)
    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("AWS CLI error: %s", e.stderr)
        sys.exit(1)


def _create_thing(device_id: str) -> None:
    """Registra un IoT thing. Idempotente: saltea si ya existe."""
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


def _issue_cert(device_id: str, cert_dir: Path, policy_name: str) -> None:
    """Genera keys + cert, attachea policy, attachea al thing. El thing tiene que existir.

    Escribe device.pem.crt / device.pem.key / device.pem.pub /
    AmazonRootCA1.pem / cert_arn.txt en cert_dir.

    Args:
        policy_name: Nombre de la IoT policy a attachear al cert. Tiene que
            matchear con la creada por CloudFormation (por default
            ``people-counter-device-policy-${Environment}`` — pasar el sufijo
            del environment correspondiente).
    """
    cert_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "aws",
            "iot",
            "create-keys-and-certificate",
            "--set-as-active",
            "--certificate-pem-outfile",
            str(cert_dir / "device.pem.crt"),
            "--private-key-outfile",
            str(cert_dir / "device.pem.key"),
            "--public-key-outfile",
            str(cert_dir / "device.pem.pub"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cert_arn = json.loads(result.stdout)["certificateArn"]
    logger.info("Certificate created: %s", cert_arn)

    subprocess.run(
        [
            "aws",
            "iot",
            "attach-policy",
            "--policy-name",
            policy_name,
            "--target",
            cert_arn,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "aws",
            "iot",
            "attach-thing-principal",
            "--thing-name",
            device_id,
            "--principal",
            cert_arn,
        ],
        check=True,
        capture_output=True,
    )
    logger.info("Certificate attached to thing %s", device_id)

    subprocess.run(
        [
            "curl",
            "-s",
            "-o",
            str(cert_dir / "AmazonRootCA1.pem"),
            "https://www.amazontrust.com/repository/AmazonRootCA1.pem",
        ],
        check=True,
        capture_output=True,
    )
    logger.info("Root CA downloaded")

    (cert_dir / "cert_arn.txt").write_text(cert_arn)


def _revoke_certs(device_id: str) -> int:
    """Detachea + desactiva + borra cada cert actualmente attacheado al thing.

    Devuelve el count de certs revocados. Safe para llamar cuando no
    hay ninguno.
    """
    result = subprocess.run(
        ["aws", "iot", "list-thing-principals", "--thing-name", device_id],
        check=True,
        capture_output=True,
        text=True,
    )
    principals = json.loads(result.stdout).get("principals", [])

    for arn in principals:
        cert_id = arn.split("/")[-1]
        logger.info("Revoking cert %s", cert_id)

        subprocess.run(
            [
                "aws",
                "iot",
                "detach-thing-principal",
                "--thing-name",
                device_id,
                "--principal",
                arn,
            ],
            check=True,
            capture_output=True,
        )

        # Detachear cada policy attacheada al cert (no asumir solo una)
        pols = subprocess.run(
            ["aws", "iot", "list-attached-policies", "--target", arn],
            check=True,
            capture_output=True,
            text=True,
        )
        for pol in json.loads(pols.stdout).get("policies", []):
            subprocess.run(
                [
                    "aws",
                    "iot",
                    "detach-policy",
                    "--policy-name",
                    pol["policyName"],
                    "--target",
                    arn,
                ],
                check=True,
                capture_output=True,
            )

        subprocess.run(
            [
                "aws",
                "iot",
                "update-certificate",
                "--certificate-id",
                cert_id,
                "--new-status",
                "INACTIVE",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["aws", "iot", "delete-certificate", "--certificate-id", cert_id],
            check=True,
            capture_output=True,
        )

    return len(principals)


def _build_config(device_dir: Path, args: argparse.Namespace) -> None:
    """Arma el config.yaml device-specific a partir del template."""
    import yaml

    # config.example.yaml contiene UTF-8 (acentos en español, chars
    # especiales en comentarios). En Windows el open() default usa
    # cp1252 y se atraganta con bytes UTF-8. Forzar UTF-8 en read +
    # write.
    with open(CONFIG_TEMPLATE, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Identidad del device
    config["device"]["id"] = args.device_id
    config["device"]["store_id"] = args.store_id
    config["device"]["store_name"] = args.store_name

    # MQTT
    config["mqtt"]["endpoint"] = args.endpoint
    config["mqtt"]["cert_path"] = f"{REMOTE_CERT_DIR}/device.pem.crt"
    config["mqtt"]["key_path"] = f"{REMOTE_CERT_DIR}/device.pem.key"
    config["mqtt"]["ca_path"] = f"{REMOTE_CERT_DIR}/AmazonRootCA1.pem"

    config_path = device_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    logger.info("Config written to %s", config_path)


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploya config y certs a un device via SSH."""
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id

    if not device_dir.exists():
        logger.error("Device %s not provisioned. Run 'create' first.", device_id)
        sys.exit(1)

    host = f"{args.user}@{args.host}"

    # Crear directorios remotos
    _ssh(
        host,
        f"sudo mkdir -p {REMOTE_CONFIG_DIR} {REMOTE_CERT_DIR} {REMOTE_DATA_DIR} {REMOTE_LOG_DIR}",
    )
    _ssh(
        host,
        f"sudo chown -R {args.user}:{args.user} {REMOTE_CONFIG_DIR} {REMOTE_DATA_DIR} {REMOTE_LOG_DIR}",
    )

    # Copiar config
    _scp(str(device_dir / "config.yaml"), f"{host}:{REMOTE_CONFIG_DIR}/config.yaml")

    # Copiar certs
    for cert_file in (device_dir / "certs").glob("*.pem*"):
        if cert_file.suffix in (".crt", ".key", ".pem"):
            _scp(str(cert_file), f"{host}:{REMOTE_CERT_DIR}/{cert_file.name}")

    # Setear permisos del cert
    _ssh(host, f"chmod 600 {REMOTE_CERT_DIR}/device.pem.key")
    _ssh(
        host,
        f"chmod 644 {REMOTE_CERT_DIR}/device.pem.crt {REMOTE_CERT_DIR}/AmazonRootCA1.pem",
    )

    # Pushear calibration.npz si hay backup para este device
    calibration = device_dir / "calibration.npz"
    if calibration.exists():
        _scp(str(calibration), f"{host}:{REMOTE_CONFIG_DIR}/calibration.npz")
        logger.info("Calibration deployed from backup")

    # Instalar systemd services y logrotate
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
        _ssh(
            host,
            "sudo mv /tmp/people-counter-logrotate /etc/logrotate.d/people-counter",
        )

    _ssh(host, "sudo systemctl daemon-reload")
    _ssh(
        host,
        "sudo systemctl enable wifi-monitor people-counter people-counter-reset.timer",
    )
    logger.info("Systemd services and logrotate installed")

    logger.info("Device %s deployed to %s", device_id, args.host)


def cmd_harvest(args: argparse.Namespace) -> None:
    """Pullea calibration.npz de un device al backup del workstation."""
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
    """Re-emite el cert para un thing existente. Revoca el viejo primero.

    Usar tras una falla de SD o siempre que se sospeche que el cert
    del device está comprometido. El thing mantiene su identidad;
    solo rota el principal.
    """
    device_id = args.device_id
    device_dir = PROVISION_DIR / device_id
    cert_dir = device_dir / "certs"

    if not device_dir.exists():
        logger.error("Device %s not provisioned. Run 'create' first.", device_id)
        sys.exit(1)

    # Mover el dir viejo de cert a un costado antes de sobreescribir
    # (por si hay que escarbar)
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
    """Lista todos los devices provisionados."""
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
    """Corre un comando en un host remoto via SSH.

    NO capturamos stdout/stderr/stdin para que el prompt interactivo de
    password (cuando el host no tiene key auth setup) llegue al usuario.
    El trade-off: si el comando falla, no podemos loguear el stderr
    formateado — pero el output llega directo al terminal igual.
    """
    try:
        subprocess.run(["ssh", host, command], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("SSH command failed (rc=%d): %s", e.returncode, command)
        raise


def _scp(local: str, remote: str) -> None:
    """Copia un archivo a un host remoto via SCP.

    Sin capture_output por la misma razón que :func:`_ssh`.
    """
    try:
        subprocess.run(["scp", local, remote], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("SCP failed (rc=%d): %s -> %s", e.returncode, local, remote)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provisioning de devices People Counter"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- create ---
    p_create = sub.add_parser("create", help="Provisiona un device nuevo")
    p_create.add_argument("--device-id", required=True, help="ID único del device")
    p_create.add_argument("--store-id", required=True, help="Identificador del store")
    p_create.add_argument(
        "--store-name", default="", help="Nombre human-readable del store"
    )
    p_create.add_argument(
        "--endpoint",
        default="xxxxx.iot.us-east-1.amazonaws.com",
        help="Endpoint de AWS IoT Core",
    )
    p_create.add_argument(
        "--policy-name",
        default="people-counter-device-policy-dev",
        help=(
            "Nombre de la IoT policy a attachear al cert. CloudFormation "
            "crea 'people-counter-device-policy-<Environment>'; default "
            "asume Environment=dev."
        ),
    )
    p_create.add_argument(
        "--skip-aws", action="store_true", help="Saltea el registro de AWS IoT"
    )
    p_create.add_argument(
        "--force", action="store_true", help="Sobreescribe el existente"
    )
    # --- metadata de dimensiones (van a la tabla sites/devices en RDS, NO al
    #     config del device) ---
    p_create.add_argument(
        "--latitude",
        type=float,
        default=None,
        help="Latitud del local (tabla sites — para el geomap de Grafana)",
    )
    p_create.add_argument(
        "--longitude",
        type=float,
        default=None,
        help="Longitud del local (tabla sites)",
    )
    p_create.add_argument(
        "--timezone",
        default=None,
        help="Timezone IANA del local, ej. America/Argentina/Buenos_Aires",
    )
    p_create.add_argument(
        "--cam-label",
        default=None,
        help="Etiqueta de la cámara, ej. 'puerta principal' (tabla devices)",
    )
    p_create.add_argument(
        "--skip-db",
        action="store_true",
        help="Saltea el seed de sites/devices en RDS (solo IoT + config local)",
    )
    p_create.add_argument(
        "--stack-name",
        default=DEFAULT_STACK_NAME,
        help=f"Stack CloudFormation para los outputs de RDS (default {DEFAULT_STACK_NAME})",
    )
    p_create.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"Región AWS (default {DEFAULT_REGION})",
    )
    p_create.set_defaults(func=cmd_create)

    # --- deploy ---
    p_deploy = sub.add_parser("deploy", help="Deploya config y certs al device")
    p_deploy.add_argument("--device-id", required=True)
    p_deploy.add_argument("--host", required=True, help="Hostname o IP del device")
    p_deploy.add_argument("--user", default="pi", help="Usuario SSH")
    p_deploy.set_defaults(func=cmd_deploy)

    # --- harvest ---
    p_harvest = sub.add_parser(
        "harvest",
        help="Pullea calibration.npz de un device al backup del workstation",
    )
    p_harvest.add_argument("--device-id", required=True)
    p_harvest.add_argument("--host", required=True, help="Hostname o IP del device")
    p_harvest.add_argument("--user", default="pi", help="Usuario SSH")
    p_harvest.set_defaults(func=cmd_harvest)

    # --- reprovision ---
    p_reprov = sub.add_parser(
        "reprovision",
        help="Re-emite cert para un thing existente (revoca el viejo)",
    )
    p_reprov.add_argument("--device-id", required=True)
    p_reprov.set_defaults(func=cmd_reprovision)

    # --- list ---
    p_list = sub.add_parser("list", help="Lista los devices provisionados")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

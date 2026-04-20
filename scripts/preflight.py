#!/usr/bin/env python3
"""Pre-flight validation for People Counter edge devices.

Runs a battery of checks against a device before the main service starts.
Intended for use during deployment or on-demand troubleshooting.

Each check returns one of PASS / FAIL / SKIP with a short detail string.
Exit code 0 if no FAIL, 1 otherwise.

Usage:
    cd /usr/src/people-counter
    sudo PYTHONPATH=. python3 scripts/preflight.py
    python3 scripts/preflight.py --json
    python3 scripts/preflight.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

# Make `src` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("preflight")

DEFAULT_CONFIG_PATH = "/etc/people-counter/config.yaml"
MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
TCP_TIMEOUT_SECONDS = 5.0

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(name: str, status: str, detail: str = "", trace: str = "") -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "trace": trace}


def _safe_check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Wrap a check so any unexpected exception becomes a FAIL."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"unexpected error: {exc}", traceback.format_exc())


def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return -1, "", str(exc)


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _is_root() -> bool:
    try:
        return hasattr(os, "geteuid") and os.geteuid() == 0  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Individual checks — each returns a result dict.
# ---------------------------------------------------------------------------

def check_config(config_path: str) -> dict[str, Any]:
    name = "config_file"
    try:
        from src.config.loader import load_config
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"cannot import loader: {exc}", traceback.format_exc())

    if not Path(config_path).exists():
        return _result(name, FAIL, f"not found: {config_path}")

    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"parse/validate error: {exc}", traceback.format_exc())

    device_id = cfg.get("device", {}).get("id", "?")
    return _result(name, PASS, f"loaded ({device_id}) from {config_path}")


def check_calibration(config: dict[str, Any] | None) -> dict[str, Any]:
    name = "calibration_file"
    if config is None:
        return _result(name, SKIP, "config not loaded")

    cal_path = config.get("vision", {}).get("calibration_file")
    if not cal_path:
        return _result(name, FAIL, "vision.calibration_file not set in config")

    if not Path(cal_path).exists():
        return _result(name, FAIL, f"not found: {cal_path}")

    try:
        import numpy as np
    except ImportError:
        return _result(name, SKIP, "numpy not installed; cannot inspect npz")

    required = ["camera_matrix_l", "camera_matrix_r", "dist_coeffs_l",
                "dist_coeffs_r", "R", "T"]
    try:
        data = np.load(cal_path, allow_pickle=False)
        keys = set(data.files)
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"cannot load npz: {exc}", traceback.format_exc())

    missing = [k for k in required if k not in keys]
    if missing:
        return _result(name, FAIL, f"missing keys: {missing}")

    return _result(name, PASS, f"{cal_path} ({len(keys)} arrays)")


def check_hef_model(config: dict[str, Any] | None) -> dict[str, Any]:
    name = "hef_model"
    if config is None:
        return _result(name, SKIP, "config not loaded")

    model_path = config.get("detection", {}).get("model_path")
    if not model_path:
        return _result(name, FAIL, "detection.model_path not set")

    p = Path(model_path)
    if not p.exists():
        return _result(name, FAIL, f"not found: {model_path}")

    size = p.stat().st_size
    if size == 0:
        return _result(name, FAIL, f"empty file: {model_path}")

    return _result(name, PASS, f"{model_path} ({size / 1024 / 1024:.1f} MB)")


def check_tls_certs(config: dict[str, Any] | None) -> dict[str, Any]:
    name = "tls_certs"
    if config is None:
        return _result(name, SKIP, "config not loaded")

    mqtt_cfg = config.get("mqtt", {})
    paths = {
        "cert": mqtt_cfg.get("cert_path"),
        "key": mqtt_cfg.get("key_path"),
        "ca": mqtt_cfg.get("ca_path"),
    }

    missing = [k for k, v in paths.items() if not v or not Path(v).exists()]
    if missing:
        return _result(name, FAIL, f"missing: {missing}")

    # Optional expiration check.
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        return _result(name, SKIP,
                       "files present; cryptography not installed — skipping expiry check")

    try:
        with open(paths["cert"], "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"cannot parse device cert: {exc}", traceback.format_exc())

    # Prefer timezone-aware accessor if available (cryptography >= 42).
    not_after = getattr(cert, "not_valid_after_utc", None)
    if not_after is None:
        not_after = cert.not_valid_after  # naive, in UTC

    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    if not_after.tzinfo is None:
        not_after = not_after.replace(tzinfo=_dt.timezone.utc)

    days_left = (not_after - now).days
    if days_left < 0:
        return _result(name, FAIL, f"device cert expired {abs(days_left)} day(s) ago")
    if days_left < 30:
        return _result(name, FAIL, f"device cert expires in {days_left} day(s)")

    return _result(name, PASS, f"certs present; device cert valid for {days_left} day(s)")


def check_cameras() -> dict[str, Any]:
    name = "cameras"
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return _result(name, SKIP, f"picamera2 unavailable: {exc}")

    try:
        infos = Picamera2.global_camera_info()
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"enumeration failed: {exc}", traceback.format_exc())

    count = len(infos) if infos else 0
    if count < 2:
        return _result(name, FAIL, f"expected 2 cameras, found {count}")
    return _result(name, PASS, f"{count} camera(s) enumerated")


def check_wifi_monitor() -> dict[str, Any]:
    name = "wifi_monitor"
    if not _is_linux():
        return _result(name, SKIP, "non-Linux host")
    if not _is_root():
        return _result(name, SKIP, "not running as root")

    rc, out, err = _run(["ip", "link", "show"])
    if rc != 0:
        return _result(name, FAIL, f"ip link failed: {err or 'exit {rc}'}")

    for iface in ("wlan0mon", "wlan1mon"):
        if iface in out:
            return _result(name, PASS, f"{iface} present")

    # Fallback: nexmon firmware hint.
    rc_dmesg, out_dmesg, _ = _run(["dmesg"], timeout=10)
    if rc_dmesg == 0 and "nexmon.org" in out_dmesg:
        return _result(name, FAIL, "nexmon loaded but no *mon interface — bring it up with airmon-ng")
    return _result(name, FAIL, "no monitor interface and nexmon not detected")


def check_ble_adapter() -> dict[str, Any]:
    name = "ble_adapter"
    if not _is_linux():
        return _result(name, SKIP, "non-Linux host")

    rc, out, err = _run(["hciconfig", "hci0"])
    if rc == 0 and out:
        if "UP RUNNING" in out:
            return _result(name, PASS, "hci0 UP RUNNING")
        return _result(name, FAIL, "hci0 present but not UP")

    # Fallback: rfkill / sysfs.
    if Path("/sys/class/bluetooth/hci0").exists():
        return _result(name, FAIL, "hci0 present in sysfs but hciconfig unavailable")
    return _result(name, FAIL, f"hci0 not found ({err or 'hciconfig missing'})")


def check_disk_space() -> dict[str, Any]:
    name = "disk_space"
    candidates = ["/var/lib/people-counter", "/var", "/"]
    target: str | None = None
    for p in candidates:
        if Path(p).exists():
            target = p
            break
    if target is None:
        # On Windows, fall back to the current drive.
        target = str(Path.cwd().anchor) or "."

    try:
        usage = shutil.disk_usage(target)
    except Exception as exc:  # noqa: BLE001
        return _result(name, FAIL, f"cannot read {target}: {exc}", traceback.format_exc())

    free_gb = usage.free / (1024 ** 3)
    if usage.free < MIN_FREE_BYTES:
        return _result(name, FAIL, f"{target}: {free_gb:.2f} GB free (< 1 GB)")
    return _result(name, PASS, f"{target}: {free_gb:.2f} GB free")


def check_iot_endpoint(config: dict[str, Any] | None) -> dict[str, Any]:
    name = "iot_endpoint"
    if config is None:
        return _result(name, SKIP, "config not loaded")

    mqtt_cfg = config.get("mqtt", {})
    endpoint = mqtt_cfg.get("endpoint")
    port = int(mqtt_cfg.get("port", 8883))
    if not endpoint:
        return _result(name, FAIL, "mqtt.endpoint not set")

    # Placeholder endpoints from the example config shouldn't be treated as real.
    if endpoint.startswith("xxxxx") or endpoint == "CHANGE_ME":
        return _result(name, FAIL, f"placeholder endpoint: {endpoint}")

    sock: socket.socket | None = None
    try:
        sock = socket.create_connection((endpoint, port), timeout=TCP_TIMEOUT_SECONDS)
        return _result(name, PASS, f"{endpoint}:{port} reachable")
    except (socket.timeout, OSError) as exc:
        return _result(name, FAIL, f"cannot reach {endpoint}:{port} — {exc}")
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def check_systemd_units() -> dict[str, Any]:
    name = "systemd_units"
    if not _is_linux():
        return _result(name, SKIP, "non-Linux host")

    units = [
        "/etc/systemd/system/people-counter.service",
        "/etc/systemd/system/people-counter-reset.service",
    ]
    missing = [u for u in units if not Path(u).exists()]
    if missing:
        return _result(name, FAIL, f"missing: {missing}")
    return _result(name, PASS, "both unit files present")


# ---------------------------------------------------------------------------
# Runner / output
# ---------------------------------------------------------------------------

def run_all_checks(config_path: str) -> list[dict[str, Any]]:
    """Execute every check in order and return a list of result dicts."""
    results: list[dict[str, Any]] = []

    # Config first — several downstream checks depend on it.
    cfg_res = _safe_check("config_file", lambda: check_config(config_path))
    results.append(cfg_res)

    loaded_cfg: dict[str, Any] | None = None
    if cfg_res["status"] == PASS:
        try:
            from src.config.loader import load_config
            loaded_cfg = load_config(config_path)
        except Exception:  # noqa: BLE001
            loaded_cfg = None

    results.append(_safe_check("calibration_file", lambda: check_calibration(loaded_cfg)))
    results.append(_safe_check("hef_model", lambda: check_hef_model(loaded_cfg)))
    results.append(_safe_check("tls_certs", lambda: check_tls_certs(loaded_cfg)))
    results.append(_safe_check("cameras", check_cameras))
    results.append(_safe_check("wifi_monitor", check_wifi_monitor))
    results.append(_safe_check("ble_adapter", check_ble_adapter))
    results.append(_safe_check("disk_space", check_disk_space))
    results.append(_safe_check("iot_endpoint", lambda: check_iot_endpoint(loaded_cfg)))
    results.append(_safe_check("systemd_units", check_systemd_units))

    return results


def format_human(results: list[dict[str, Any]], verbose: bool) -> str:
    lines = ["People Counter — Pre-flight Check", ""]
    for r in results:
        lines.append(f"[{r['status']:4s}] {r['name']}: {r['detail']}")
        if verbose and r["status"] == FAIL and r.get("trace"):
            for tline in r["trace"].rstrip().splitlines():
                lines.append(f"       {tline}")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == PASS)
    skipped = sum(1 for r in results if r["status"] == SKIP)
    failed = [r for r in results if r["status"] == FAIL]

    lines.append("")
    lines.append("=" * 50)
    lines.append(f"{passed}/{total} checks passed ({skipped} skipped, {len(failed)} failed)")
    if failed:
        lines.append("Failures:")
        for r in failed:
            lines.append(f"  - {r['name']}: {r['detail']}")
    else:
        lines.append("Device is ready.")

    return "\n".join(lines)


def exit_code(results: list[dict[str, Any]]) -> int:
    return 1 if any(r["status"] == FAIL for r in results) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="People Counter pre-flight check")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full tracebacks for failures")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_all_checks(args.config)
    code = exit_code(results)

    if args.json:
        # Strip trace from JSON unless verbose — keeps payload small.
        out = []
        for r in results:
            item = {"name": r["name"], "status": r["status"], "detail": r["detail"]}
            if args.verbose and r.get("trace"):
                item["trace"] = r["trace"]
            out.append(item)
        print(json.dumps({"checks": out, "exit_code": code}, indent=2))
    else:
        print(format_human(results, verbose=args.verbose))

    sys.exit(code)


if __name__ == "__main__":
    main()

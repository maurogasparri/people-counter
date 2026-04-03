#!/usr/bin/env python3
"""Hardware verification script for People Counter edge devices.

Runs a checklist of all hardware and software prerequisites.
Designed to run on the RPi5 after setup-guide steps are complete.

Usage:
    cd /usr/src/people-counter
    sudo PYTHONPATH=. python3 scripts/verify_hardware.py
"""

import glob
import os
import subprocess
import sys


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    line = f"  [{status:4s}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1, ""


def main() -> None:
    print("People Counter — Hardware Verification\n")
    failures = 0

    # --- System ---
    print("[System]")

    rc, out = run(["uname", "-m"])
    failures += not check("Architecture", out == "aarch64", out)

    rc, out = run(["uname", "-r"])
    kernel_ok = out.startswith("6.12") or out.startswith("6.13")
    failures += not check("Kernel ≥ 6.12", kernel_ok, out)

    rc, out = run(["raspi-config", "nonint", "get_boot_cli"])
    cli_boot = out.strip() == "0"
    failures += not check("Boot to CLI (headless)", cli_boot,
                          "CLI" if cli_boot else "desktop — run: sudo raspi-config nonint do_boot_behaviour B1")

    # --- config.txt ---
    print("\n[config.txt]")
    try:
        config_txt = open("/boot/firmware/config.txt").read()
    except FileNotFoundError:
        config_txt = ""

    failures += not check("gpu_mem=16", "gpu_mem=16" in config_txt)
    failures += not check("dtparam=pciex1_gen=3", "dtparam=pciex1_gen=3" in config_txt)
    failures += not check("dtparam=rtc_bbat_vchg", "dtparam=rtc_bbat_vchg" in config_txt)

    # --- PCIe Gen 3 ---
    print("\n[PCIe]")
    rc, out = run(["lspci", "-vv"])
    pcie_gen3 = "Speed 8GT/s" in out
    failures += not check("PCIe Gen 3 (8GT/s)", pcie_gen3)

    # --- Hailo ---
    print("\n[Hailo]")
    rc, out = run(["lspci"])
    hailo_detected = "Hailo" in out
    failures += not check("Hailo detected (lspci)", hailo_detected)

    rc, out = run(["hailortcli", "fw-control", "identify"])
    hailo_fw = "Hailo-8" in out
    fw_version = ""
    for line in out.splitlines():
        if "Firmware Version" in line:
            fw_version = line.split(":")[-1].strip()
    failures += not check("Hailo firmware", hailo_fw, fw_version)

    # --- Cameras ---
    print("\n[Cameras]")
    rc, out = run(["rpicam-hello", "--list-cameras"])
    cam_count = out.count("ov5647")
    failures += not check("OV5647 cameras", cam_count >= 2, f"{cam_count} found")

    # --- RTC Battery ---
    print("\n[RTC]")
    rtc_paths = glob.glob("/sys/devices/platform/soc*/soc*:rpi_rtc/rtc/rtc0/charging_voltage")
    if rtc_paths:
        try:
            voltage = open(rtc_paths[0]).read().strip()
            failures += not check("RTC charging", voltage == "3000000", f"{voltage} µV")
        except Exception:
            failures += not check("RTC charging", False, "read error")
    else:
        failures += not check("RTC charging", False, "sysfs path not found")

    # --- Temperature ---
    print("\n[Temperature]")
    rc, out = run(["vcgencmd", "measure_temp"])
    if "temp=" in out:
        temp = float(out.split("=")[1].split("'")[0])
        failures += not check("CPU temperature", temp < 70, f"{temp}°C")
    else:
        failures += not check("CPU temperature", False, "could not read")

    # --- Watchdog ---
    print("\n[Watchdog]")
    rc, out = run(["systemctl", "is-active", "watchdog"])
    failures += not check("Watchdog service", out.strip() == "active", out.strip())

    # --- WiFi / nexmon ---
    print("\n[WiFi / nexmon]")
    rc, out = run(["dmesg"])
    nexmon_loaded = "nexmon.org" in out
    failures += not check("Nexmon firmware loaded", nexmon_loaded)

    rc, out = run(["iw", "phy", "phy0", "info"])
    monitor_support = "monitor" in out.lower()
    failures += not check("Monitor mode supported", monitor_support)

    # --- BLE ---
    print("\n[BLE]")
    rc, out = run(["hciconfig", "hci0"])
    ble_up = "UP RUNNING" in out
    failures += not check("BLE adapter (hci0)", ble_up)

    # --- Python ---
    print("\n[Software]")
    failures += not check("Python", True, f"{sys.version.split()[0]}")

    try:
        import cv2
        failures += not check("OpenCV", True, cv2.__version__)
    except ImportError:
        failures += not check("OpenCV", False, "not installed")

    try:
        import numpy
        failures += not check("NumPy", True, numpy.__version__)
    except ImportError:
        failures += not check("NumPy", False, "not installed")

    try:
        from picamera2 import Picamera2
        failures += not check("picamera2", True)
    except ImportError:
        failures += not check("picamera2", False, "not installed — sudo apt install python3-picamera2")

    try:
        import hailo_platform
        failures += not check("hailo_platform", True, hailo_platform.__version__)
    except ImportError:
        failures += not check("hailo_platform", False, "not installed")

    try:
        import scapy
        failures += not check("scapy", True)
    except ImportError:
        failures += not check("scapy", False, "pip install scapy")

    try:
        import bleak
        failures += not check("bleak", True)
    except ImportError:
        failures += not check("bleak", False, "pip install bleak")

    # --- Model ---
    print("\n[Model]")
    model_path = "/usr/src/people-counter/models/yolov8n.hef"
    model_exists = os.path.exists(model_path)
    if model_exists:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        failures += not check("YOLOv8n HEF", True, f"{size_mb:.1f} MB")
    else:
        failures += not check("YOLOv8n HEF", False, "run: PYTHONPATH=. python3 scripts/download_model.py hef")

    # --- Config ---
    print("\n[Config]")
    config_exists = os.path.exists("/etc/people-counter/config.yaml")
    failures += not check("Device config", config_exists,
                          "/etc/people-counter/config.yaml" if config_exists else "not found — copy from config.example.yaml")

    # --- Services ---
    print("\n[Services]")
    for svc in ["wifi-monitor", "people-counter", "people-counter-reset.timer"]:
        rc, out = run(["systemctl", "is-enabled", svc])
        failures += not check(f"{svc}", out.strip() == "enabled", out.strip())

    # --- Summary ---
    print(f"\n{'=' * 50}")
    if failures == 0:
        print("All checks passed. Device is ready.")
    else:
        print(f"{failures} check(s) failed. Review the items above.")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

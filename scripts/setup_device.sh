#!/bin/bash
# setup_device.sh — Automated setup for People Counter edge devices.
#
# Runs steps 4-9 of setup-guide.md after the first boot and apt upgrade.
# Prerequisites: RPi5 running Trixie, already updated (apt full-upgrade + reboot).
#
# Usage:
#   sudo bash /usr/src/people-counter/scripts/setup_device.sh
#
# Or remotely after cloning:
#   ssh pi@people-counter.local 'sudo bash /usr/src/people-counter/scripts/setup_device.sh'

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

if [ "$(id -u)" -ne 0 ]; then
    error "This script must be run as root (sudo)"
    exit 1
fi

REPO_DIR="/usr/src/people-counter"
if [ ! -d "$REPO_DIR" ]; then
    error "$REPO_DIR not found. Clone the repo first:"
    echo "  sudo git clone https://github.com/maurogasparri/people-counter.git $REPO_DIR"
    echo "  sudo chown -R pi:pi $REPO_DIR"
    exit 1
fi

# =========================================================================
# Step 4: System config (headless + watchdog + config.txt)
# =========================================================================
info "Step 4: Configuring system..."

info "  Disabling desktop (headless mode)"
raspi-config nonint do_boot_behaviour B1

info "  Installing and enabling watchdog"
apt install -y watchdog
sed -i 's/^#watchdog-device/watchdog-device/' /etc/watchdog.conf
sed -i 's/^#max-load-1/max-load-1/' /etc/watchdog.conf
systemctl enable watchdog
systemctl start watchdog

info "  Configuring config.txt (RTC, PCIe Gen 3, USB current, IMX708 cameras)"
CONFIG_TXT="/boot/firmware/config.txt"
# RTC charging: only for rechargeable ML2032 batteries.
# If using non-rechargeable CR2032, comment out or remove this line after setup.
grep -q "^dtparam=rtc_bbat_vchg" "$CONFIG_TXT" || echo "dtparam=rtc_bbat_vchg=3000000" >> "$CONFIG_TXT"
# PCIe Gen 3: required by AI HAT+
grep -q "^dtparam=pciex1_gen=3" "$CONFIG_TXT" || echo "dtparam=pciex1_gen=3" >> "$CONFIG_TXT"
# USB current: required by Waveshare PoE HAT (H) to avoid power-supply prompt
grep -q "^usb_max_current_enable=1" "$CONFIG_TXT" || echo "usb_max_current_enable=1" >> "$CONFIG_TXT"
# IMX708 cameras: disable autodetect and force overlay on both CSI ports
sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CONFIG_TXT"
grep -q "^dtoverlay=imx708" "$CONFIG_TXT" || sed -i '/^\[all\]/a dtoverlay=imx708' "$CONFIG_TXT"

# =========================================================================
# Step 5: Hailo
# =========================================================================
info "Step 5: Installing Hailo (minimal: runtime + PCIe driver + Python bindings)..."
apt install -y hailort hailort-pcie-driver python3-hailort

# =========================================================================
# Step 6: Nexmon (WiFi monitor mode)
# =========================================================================
info "Step 6: Installing nexmon..."
apt install -y dkms aircrack-ng tcpdump

NEXMON_FW="firmware-nexmon_0.2_all.deb"
NEXMON_DKMS="brcmfmac-nexmon-dkms_6.12.2_all.deb"

if [ ! -f "/tmp/$NEXMON_FW" ]; then
    wget -q -O "/tmp/$NEXMON_FW" "http://http.kali.org/pool/non-free-firmware/f/firmware-nexmon/$NEXMON_FW"
fi
if [ ! -f "/tmp/$NEXMON_DKMS" ]; then
    wget -q -O "/tmp/$NEXMON_DKMS" "http://http.kali.org/pool/contrib/b/brcmfmac-nexmon-dkms/$NEXMON_DKMS"
fi

dpkg -i --force-overwrite "/tmp/$NEXMON_FW"
dpkg -i "/tmp/$NEXMON_DKMS"

# =========================================================================
# Step 7: Project dependencies
# =========================================================================
info "Step 7: Installing project dependencies..."
apt install -y \
    python3-pip \
    python3-opencv python3-numpy \
    libopencv-dev \
    git

cd "$REPO_DIR"
pip install --break-system-packages --root-user-action=ignore -e ".[dev]"

info "  Downloading YOLOv8n model..."
PYTHONPATH="$REPO_DIR" python3 "$REPO_DIR/scripts/download_model.py" hef

# =========================================================================
# Step 8: Device config
# =========================================================================
info "Step 8: Creating directories and default config..."
mkdir -p /etc/people-counter/certs /var/lib/people-counter /var/log/people-counter
chown -R pi:pi /etc/people-counter /var/lib/people-counter /var/log/people-counter

if [ ! -f /etc/people-counter/config.yaml ]; then
    cp "$REPO_DIR/config/config.example.yaml" /etc/people-counter/config.yaml
    chown pi:pi /etc/people-counter/config.yaml
    warn "Default config copied to /etc/people-counter/config.yaml — edit before running pipeline"
else
    info "  Config already exists, skipping"
fi

# =========================================================================
# Step 9: Systemd services
# =========================================================================
info "Step 9: Installing systemd services..."
cp "$REPO_DIR/config/wifi-monitor.service" /etc/systemd/system/
cp "$REPO_DIR/config/people-counter.service" /etc/systemd/system/
cp "$REPO_DIR/config/people-counter-reset.service" /etc/systemd/system/
cp "$REPO_DIR/config/people-counter-reset.timer" /etc/systemd/system/
cp "$REPO_DIR/config/logrotate.conf" /etc/logrotate.d/people-counter

systemctl daemon-reload
systemctl enable wifi-monitor people-counter people-counter-reset.timer

# =========================================================================
# Done
# =========================================================================
echo ""
info "Setup complete. Reboot required to apply config.txt changes."
info ""
info "After reboot:"
info "  1. Edit /etc/people-counter/config.yaml with device-specific settings"
info "  2. Run: sudo PYTHONPATH=$REPO_DIR python3 $REPO_DIR/scripts/verify_hardware.py"
info "  3. Focus: PYTHONPATH=. python3 scripts/focus_assist.py --grid"
info "  4. Calibrate: PYTHONPATH=. python3 scripts/calibrate.py capture \\"
info "       --columns 11 --rows 7 --square-length 35 --marker-length 26 --count 30"
info ""
read -p "Reboot now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    reboot
fi

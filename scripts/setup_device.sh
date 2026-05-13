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

# IPv4 precedence cuando la red no tiene IPv6 routable.
# Sin esto, getaddrinfo devuelve la AAAA primero y paho-mqtt (y otros
# clientes) intenta IPv6 → SYN al vacío → cuelgue ~60-130s antes de
# caer en fallback a IPv4. Descomentar la línea de gai.conf invierte la
# preferencia: IPv4 primero, IPv6 si está disponible y es preferida.
info "  Preferring IPv4 in getaddrinfo (gai.conf) — avoids paho hang on IPv6-less networks"
sed -i 's|^#precedence ::ffff:0:0/96  100|precedence ::ffff:0:0/96  100|' /etc/gai.conf

info "  Installing and enabling watchdog"
apt install -y watchdog
sed -i 's/^#watchdog-device/watchdog-device/' /etc/watchdog.conf
sed -i 's/^#max-load-1/max-load-1/' /etc/watchdog.conf
systemctl enable watchdog
systemctl start watchdog

info "  Configuring config.txt (RTC, PCIe Gen 3, USB current, IMX708 cameras, low power)"
CONFIG_TXT="/boot/firmware/config.txt"
# Carga del RTC: solo para baterías ML2032 recargables.
# Si se usa una CR2032 no-recargable, comentar o sacar esta línea después del setup.
grep -q "^dtparam=rtc_bbat_vchg" "$CONFIG_TXT" || echo "dtparam=rtc_bbat_vchg=3000000" >> "$CONFIG_TXT"
# PCIe Gen 3: requerido por el AI HAT+
grep -q "^dtparam=pciex1_gen=3" "$CONFIG_TXT" || echo "dtparam=pciex1_gen=3" >> "$CONFIG_TXT"
# USB current: requerido por Waveshare PoE HAT (H) para evitar el prompt de power-supply
grep -q "^usb_max_current_enable=1" "$CONFIG_TXT" || echo "usb_max_current_enable=1" >> "$CONFIG_TXT"
# Cámaras IMX708: deshabilitar autodetect, forzar overlay por CSI port.
# Pi 5 requiere ,cam0/,cam1 explícitos — un "dtoverlay=imx708" pelado solo carga una cámara.
sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CONFIG_TXT"
grep -q "^dtoverlay=imx708,cam0" "$CONFIG_TXT" || sed -i '/^\[all\]/a dtoverlay=imx708,cam0' "$CONFIG_TXT"
grep -q "^dtoverlay=imx708,cam1" "$CONFIG_TXT" || sed -i '/^\[all\]/a dtoverlay=imx708,cam1' "$CONFIG_TXT"
# Apagar los LEDs onboard (ACT + power), los LEDs del jack Ethernet (link/activity),
# y el PWM de audio. El RGB externo es el indicador de status canónico.
#
# eth_led0 / eth_led1 usan los mode codes del PHY bcm54213; mode 4 = "off / always low".
grep -q "^dtparam=audio=" "$CONFIG_TXT" \
    && sed -i 's/^dtparam=audio=.*/dtparam=audio=off/' "$CONFIG_TXT" \
    || echo "dtparam=audio=off" >> "$CONFIG_TXT"
grep -q "^dtparam=act_led_trigger=" "$CONFIG_TXT" \
    || echo "dtparam=act_led_trigger=none" >> "$CONFIG_TXT"
grep -q "^dtparam=act_led_activelow=" "$CONFIG_TXT" \
    || echo "dtparam=act_led_activelow=off" >> "$CONFIG_TXT"
grep -q "^dtparam=power_led_trigger=" "$CONFIG_TXT" \
    || echo "dtparam=power_led_trigger=none" >> "$CONFIG_TXT"
grep -q "^dtparam=power_led_activelow=" "$CONFIG_TXT" \
    || echo "dtparam=power_led_activelow=off" >> "$CONFIG_TXT"
grep -q "^dtparam=eth_led0=" "$CONFIG_TXT" \
    || echo "dtparam=eth_led0=4" >> "$CONFIG_TXT"
grep -q "^dtparam=eth_led1=" "$CONFIG_TXT" \
    || echo "dtparam=eth_led1=4" >> "$CONFIG_TXT"

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

# rfkill: si el device booteó con WiFi disabled (default cuando se hace
# raspi-config nonint do_boot_behaviour B1 sin asociar a una red),
# phy0 queda soft-blocked y airmon-ng se cuelga al iniciar el pipeline.
# Desbloqueamos persistentemente — el ExecStartPre del wifi-monitor
# service también lo hace en runtime para máxima resiliencia.
apt install -y rfkill
rfkill unblock wifi

# =========================================================================
# Step 7: Project dependencies
# =========================================================================
info "Step 7: Installing project dependencies..."
apt install -y \
    python3-pip \
    python3-opencv python3-numpy python3-scipy \
    python3-yaml python3-paho-mqtt \
    libopencv-dev \
    python3-gpiozero python3-lgpio \
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

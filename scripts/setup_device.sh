#!/bin/bash
# setup_device.sh — Automated setup for People Counter edge devices.
#
# Runs steps 4-9 of setup_guide.md after the first boot and apt upgrade.
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

# Dominio regulatorio WiFi: habilita los canales permitidos del pais para el
# monitor mode. Sin esto, el regdomain 00 (world) restringe parte de 5 GHz y
# el channel hopping pierde esos canales. Default AR (PoC argentino);
# override con WIFI_COUNTRY=XX para otra jurisdiccion.
info "  Setting WiFi regulatory domain (${WIFI_COUNTRY:-AR})"
raspi-config nonint do_wifi_country "${WIFI_COUNTRY:-AR}" || true

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

# Memory cgroup controller. Raspbian Trixie lo deshabilita por default
# (overhead histórico de Pi1 con poca RAM, irrelevante en Pi5 8GB). Sin
# esto MemoryCurrent / MemoryPeak son [not set] en `systemctl show
# people-counter` y no podemos auditar crecimiento de RSS / leaks en
# producción sin shellar y correr ps. Habilitamos memory + cpuset.
CMDLINE_TXT="/boot/firmware/cmdline.txt"
if [ -f "$CMDLINE_TXT" ] && ! grep -q "cgroup_enable=memory" "$CMDLINE_TXT"; then
    info "  Enabling memory cgroup controller (requires reboot to take effect)"
    sed -i 's/$/ cgroup_enable=memory cgroup_memory=1/' "$CMDLINE_TXT"
fi

# Sysctl tuning (swappiness=10 + dirty page caps). Permite correr en Pi 5 2GB
# sin tocar swap bajo carga sostenida + bursts WiFi/BLE. Ver
# config/sysctl-people-counter.conf para el detalle.
info "  Deploying sysctl drop-in (vm.swappiness=10 + dirty page caps)"
cp "$REPO_DIR/config/sysctl-people-counter.conf" /etc/sysctl.d/99-people-counter.conf
sysctl --system 2>/dev/null | grep -E "swappiness|dirty" || true

# Regla polkit: autoriza a `pi` a reboot/poweroff vía logind (botones del
# viewer). El servicio corre con NoNewPrivileges → sudo no sirve; los botones
# usan `systemctl reboot|poweroff`, que polkit consulta contra esta regla.
info "  Deploying polkit rule (pi → reboot/poweroff via logind)"
cp "$REPO_DIR/config/polkit/10-people-counter-power.rules" /etc/polkit-1/rules.d/10-people-counter-power.rules
systemctl restart polkit 2>/dev/null || warn "    (no se pudo reiniciar polkit; toma efecto al próximo boot)"

# Audio stack innecesario. Pipewire + wireplumber + pipewire-pulse arrancan
# por default con la sesión del usuario `pi` (autostart como user units).
# El sistema NO usa audio (RGB LED es GPIO directo); ocupan ~30 MB sostenido
# de RAM por nada. Loginctl enable-linger + systemctl --user mask los frena
# desde el próximo login del user.
info "  Masking audio stack (no se usa, libera ~30 MB)"
sudo -u pi XDG_RUNTIME_DIR=/run/user/$(id -u pi) systemctl --user mask \
    pipewire.service pipewire.socket \
    wireplumber.service \
    pipewire-pulse.service pipewire-pulse.socket 2>/dev/null || \
    warn "    (mask falló — probablemente la sesión user no esta corriendo aún; correr manualmente como pi tras el primer login)"

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

# --- nexutil: activa monitor mode CON radiotap en el firmware ---
# Sin nexutil, `iw set type monitor` deja el interface en type=monitor pero el
# firmware entrega frames Ethernet (DLT EN10MB) y scapy no ve ningún 802.11 →
# CERO probes capturados (descubierto en el piloto). No hay paquete apt; se
# compila del repo nexmon (solo utilities/ + patches/include vía sparse
# checkout, sin bajar los blobs de firmware).
if ! command -v nexutil >/dev/null 2>&1; then
    info "  Building nexutil from nexmon (sparse checkout, utilities only)"
    apt install -y git build-essential libnl-3-dev
    NEXMON_SRC=/usr/src/nexmon-tools
    rm -rf "$NEXMON_SRC"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/seemoo-lab/nexmon.git "$NEXMON_SRC"
    git -C "$NEXMON_SRC" sparse-checkout set \
        utilities/nexutil utilities/libnexio utilities/libargp patches/include
    make -C "$NEXMON_SRC/utilities/nexutil"
    install -m 755 "$NEXMON_SRC/utilities/nexutil/nexutil" /usr/local/bin/nexutil
    info "  nexutil instalado en /usr/local/bin/nexutil"
fi

# --- Prerequisitos de monitor mode (rfkill / NM / systemd-rfkill) ---
# Descubiertos en el bring-up del piloto. El device bootea con WiFi
# soft-blocked por rfkill (default headless), y el monitor mode no arranca
# por una cadena de causas; las tres piezas de abajo lo resuelven de forma
# durable.
apt install -y rfkill network-manager

# 1. udev rule — EL FIX DE FONDO. El pipeline (people-counter.service) corre
#    como User=pi; `rfkill unblock` escribe el device node /dev/rfkill, que por
#    default es `crw-rw-r-- root root` → solo root escribe. Como pi falla con
#    "cannot open /dev/rfkill: Permission denied" y el monitor mode nunca
#    arranca (ip link up → RF-kill). La regla pone /dev/rfkill en grupo netdev
#    (pi ya pertenece) con escritura → el pipeline se desbloquea a sí mismo.
info "  Installing udev rule for /dev/rfkill (netdev write access)"
cp "$REPO_DIR/config/udev-rfkill.rules" /etc/udev/rules.d/90-rfkill.rules
udevadm control --reload && udevadm trigger --subsystem-match=misc || true

# 2. NetworkManager — wlan0 unmanaged. Interfaz dedicada a probing (la red va
#    por Ethernet); sin esto NM la administra y la re-bloquea al inicializarse.
info "  Marking wlan0 unmanaged in NetworkManager"
mkdir -p /etc/NetworkManager/conf.d
cp "$REPO_DIR/config/networkmanager-unmanage-wlan0.conf" \
    /etc/NetworkManager/conf.d/99-unmanage-wlan0.conf
nmcli general reload 2>/dev/null || true

# 3. systemd-rfkill enmascarado + saved-state de wlan limpiado. systemd-rfkill
#    restaura al boot el estado 'blocked' guardado Y, socket-activated, pelea
#    contra cada unblock → enmascararlo evita ese loop.
info "  Masking systemd-rfkill (avoids restoring/fighting the blocked state)"
systemctl mask systemd-rfkill.service systemd-rfkill.socket 2>/dev/null || true
rm -f /var/lib/systemd/rfkill/*wlan* 2>/dev/null || true

# Unblock inmediato (los archivos de arriba aplican al reboot final igual).
rfkill unblock all || true

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

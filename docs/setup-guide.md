# Guía de puesta en marcha — People Counter PoC

## 1. Preparar la MicroSD

Desde tu PC con Windows:

1. Descargar Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Insertar la MicroSD de 64GB
3. En Imager:
   - OS: Raspberry Pi OS (64-bit) — Trixie
   - Storage: tu MicroSD
   - Settings (engranaje):
     - Hostname: `people-counter`
     - Enable SSH: ✓ (password o key)
     - Set username: `pi`
     - Set password: [tu password]
     - Configure WiFi: tu red (solo para el setup inicial, después va por cable)
     - Locale: America/Argentina/Buenos_Aires
4. Write y esperar

## 2. Ensamblaje físico

Orden recomendado:

1. **Hailo-8L** → insertarlo en el slot M.2 del PoE M.2 HAT+
2. **HAT+** → montarlo sobre la Raspberry Pi 5 (conectores GPIO + FFC)
3. **Cámaras** → conectar los cables CSI a los puertos CAM0 y CAM1 de la Pi
   - Cámara derecha → CAM0
   - Cámara izquierda → CAM1
   - Orientar ambas igual (el conector flat tiene un lado con contactos expuestos)
4. **Disipador** → montar sobre el SoC de la Pi
5. **MicroSD** → insertar la tarjeta ya flasheada
6. **Batería RTC** → conectar al conector J5 de la Pi (entre los puertos USB y el GPIO).
   Usar una batería recargable LiMnO2 como la ML2032 (no confundir con CR2032 que no es recargable).
7. **NO conectar PoE todavía** — para el PoC usá la fuente USB-C estándar

## 3. Primer boot y actualización

1. Conectar por Ethernet + SSH (`ssh pi@people-counter.local`), o monitor HDMI + teclado
2. Esperar que termine el primer boot (puede tardar 2-3 min)

```bash
# Verificar que arranca
uname -a  # Debe decir aarch64

# Actualizar sistema
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

## 4. Configurar sistema (headless + config.txt)

```bash
# Deshabilitar entorno gráfico (libera ~200MB de RAM)
sudo raspi-config nonint do_boot_behaviour B1

# Habilitar watchdog (reinicio automático si se cuelga)
sudo apt install -y watchdog
sudo sed -i 's/^#watchdog-device/watchdog-device/' /etc/watchdog.conf
sudo sed -i 's/^#max-load-1/max-load-1/' /etc/watchdog.conf
sudo systemctl enable watchdog
sudo systemctl start watchdog

# Configurar GPU, RTC y PCIe Gen 3
sudo tee -a /boot/firmware/config.txt > /dev/null << 'CONF'
gpu_mem=16
dtparam=rtc_bbat_vchg=3000000
dtparam=pciex1_gen=3
CONF

sudo reboot
```

Si usás una pila no recargable (CR2032), **no agregar la línea de rtc_bbat_vchg**.

Referencias:
- RTC: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#add-a-backup-battery
- PCIe Gen 3: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#pcie-gen-3-0

## 5. Instalar Hailo

Referencia: https://www.raspberrypi.com/documentation/computers/ai.html#update

```bash
sudo apt install -y hailo-all
sudo reboot
```

## 6. Instalar nexmon (WiFi monitor mode)

El CYW43455 integrado no soporta monitor mode por defecto. Los paquetes de nexmon
(originalmente de Kali Linux) parchean el firmware y el driver para habilitarlo.

Referencia: https://www.kali.org/blog/raspberry-pi-wi-fi-glow-up/

```bash
sudo apt install -y dkms aircrack-ng tcpdump
wget http://http.kali.org/pool/non-free-firmware/f/firmware-nexmon/firmware-nexmon_0.2_all.deb
wget http://http.kali.org/pool/contrib/b/brcmfmac-nexmon-dkms/brcmfmac-nexmon-dkms_6.12.2_all.deb
sudo dpkg -i --force-overwrite firmware-nexmon_0.2_all.deb
sudo dpkg -i brcmfmac-nexmon-dkms_6.12.2_all.deb
sudo reboot
```

## 7. Instalar el proyecto

```bash
# Dependencias del sistema
sudo apt install -y \
  python3-pip \
  python3-opencv python3-numpy \
  libopencv-dev \
  git

# Clonar el repo
sudo git clone https://github.com/maurogasparri/people-counter.git /usr/src/people-counter
sudo chown -R pi:pi /usr/src/people-counter
cd /usr/src/people-counter

# Instalar dependencias del proyecto + captura WiFi/BLE
sudo pip install --break-system-packages --root-user-action=ignore -e ".[dev]"
sudo pip install --break-system-packages --root-user-action=ignore scapy bleak

# Descargar modelo YOLOv8n para Hailo-8L
PYTHONPATH=. python3 scripts/download_model.py hef

# Verificar tests
pytest -v
```

## 8. Configurar el dispositivo

```bash
# Crear directorios
sudo mkdir -p /etc/people-counter/certs /var/lib/people-counter /var/log/people-counter
sudo chown -R pi:pi /etc/people-counter /var/lib/people-counter /var/log/people-counter

# Copiar config de ejemplo y personalizarlo
sudo cp /usr/src/people-counter/config/config.example.yaml /etc/people-counter/config.yaml
sudo nano /etc/people-counter/config.yaml
```

Campos que hay que personalizar por dispositivo:
- `device.id` — identificador único (ej: `store-001-cam-01`)
- `device.store_id` — identificador del local (ej: `store-001`)
- `device.store_name` — nombre legible del local
- `mqtt.endpoint` — endpoint de AWS IoT Core
- `vision.calibration_file` — path al `.npz` de calibración (después de calibrar)

Alternativamente, usar `scripts/provision.py` que genera el config automáticamente.

## 9. Instalar servicios del sistema

```bash
# Copiar todos los servicios y configs
sudo cp /usr/src/people-counter/config/wifi-monitor.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter-reset.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter-reset.timer /etc/systemd/system/
sudo cp /usr/src/people-counter/config/logrotate.conf /etc/logrotate.d/people-counter

# Habilitar servicios
sudo systemctl daemon-reload
sudo systemctl enable wifi-monitor people-counter people-counter-reset.timer
```

## 10. Verificar todo

```bash
cd /usr/src/people-counter
sudo PYTHONPATH=. python3 scripts/verify_hardware.py
```

Este script verifica: kernel, config.txt, PCIe Gen 3, Hailo, cámaras, RTC, temperatura,
watchdog, nexmon, BLE, Python + dependencias, modelo HEF, config, y servicios systemd.

Para verificar las cámaras visualmente (headless):

```bash
rpicam-still -o /tmp/test_cam0.jpg --camera 0
rpicam-still -o /tmp/test_cam1.jpg --camera 1
# Desde tu PC:
scp pi@people-counter.local:/tmp/test_cam*.jpg .
```

## 11. Calibración estéreo

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/calibrate.py capture --count 30 --interval 5
```

Mover el patrón ChArUco entre capturas a distintas distancias (2.5-4m), ángulos y
posiciones. Buena iluminación, sin reflejos directos sobre el papel. Después calibrar:

```bash
PYTHONPATH=. python3 scripts/calibrate.py calibrate \
  --input-dir ./calibration/captures \
  --output /etc/people-counter/calibration.npz
```

## Troubleshooting

- **Cámaras no detectadas**: verificar que los cables CSI están bien
  insertados. El conector tiene un clip que se levanta, se inserta el
  flat y se baja el clip.
- **Hailo no detectado**: verificar que el M.2 está bien insertado en
  el HAT. Correr `lspci` y buscar Hailo.
- **Boot loop**: sacar el HAT y bootear solo con la Pi para descartar
  problemas de alimentación. La fuente USB-C debe ser de 5V/5A.
- **WiFi monitor mode no funciona**: verificar con `dmesg | grep nexmon`
  que el firmware nexmon está cargado. Si no aparece, reinstalar firmware-nexmon.
- **picamera2 no importa**: verificar que está instalado con
  `sudo apt install python3-picamera2`.
- **"Unknown error 524" en airmon-ng**: es esperado con nexmon en RPi5,
  no afecta la captura.

# Guía de puesta en marcha — People Counter PoC

## 1. Preparar la MicroSD

Desde tu PC con Windows:

1. Descargar Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Insertar la MicroSD de 64GB
3. En Imager:
   - OS: Raspberry Pi OS (64-bit) — Bookworm
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
6. **Batería RTC** → conectar al conector J5 de la Pi (entre los puertos USB y el GPIO)
7. **NO conectar PoE todavía** — para el PoC usá la fuente USB-C estándar

## 3. Primer boot

1. Conectar monitor HDMI + teclado, o acceder por SSH (`ssh pi@people-counter.local`)
2. Esperar que termine el primer boot (puede tardar 2-3 min)
3. Verificar que arranca:

```bash
uname -a
# Debe decir aarch64

df -h
# Verificar espacio en disco
```

## 4. Actualizar el sistema

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

Después del reinicio, verificar kernel:

```bash
uname -r
# Debe ser 6.6.31 o superior
```

## 5. Deshabilitar entorno gráfico

Los dispositivos corren headless. Deshabilitar el desktop libera ~200MB de RAM y reduce
el uso de CPU, importante con solo 4GB de RAM.

```bash
sudo raspi-config nonint do_boot_behaviour B1
sudo reboot
```

Esto configura el boot directo a consola (CLI). Para revertir temporalmente si necesitás
escritorio: `sudo raspi-config nonint do_boot_behaviour B4`.

## 6. Reducir memoria de GPU

Al correr headless, la GPU no necesita los 76MB por defecto. Reducirlo libera RAM
para OpenCV y el pipeline. Esto se configura en config.txt junto con los otros parámetros.

## 7. Habilitar watchdog

La RPi5 tiene un watchdog por hardware (BCM2712). Si el sistema se cuelga, el watchdog
lo reinicia automáticamente. Crítico para dispositivos que corren sin atención.

```bash
sudo apt install -y watchdog
```

Editar la configuración del watchdog:

```bash
sudo nano /etc/watchdog.conf
```

Descomentar/agregar estas líneas:

```
watchdog-device = /dev/watchdog
max-load-1 = 24
watchdog-timeout = 15
```

Habilitar el servicio:

```bash
sudo systemctl enable watchdog
sudo systemctl start watchdog
```

## 8. Configurar RTC y PCIe Gen 3

### 8.1. Batería RTC

La RPi5 tiene un RTC integrado con conector J5 para batería de respaldo. Esto permite
mantener la hora cuando el dispositivo está apagado o sin red (importante para timestamps
de los eventos de conteo).

Usar una batería recargable LiMnO2 como la ML2032 (no confundir con CR2032 que no es recargable).
Por defecto la RPi5 **no carga** la batería. Hay que habilitarlo en config.txt.

Referencia: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#add-a-backup-battery

### 8.2. PCIe Gen 3

El Hailo-8L requiere PCIe Gen 3 para alcanzar los 13 TOPS. Por defecto la RPi5 usa Gen 2.

Referencia: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#pcie-gen-3-0

### 8.3. Aplicar todos los cambios

```bash
sudo nano /boot/firmware/config.txt

# Agregar al final del archivo:
gpu_mem=16
dtparam=rtc_bbat_vchg=3000000
dtparam=pciex1_gen=3
```

Si usás una pila no recargable (CR2032), **no agregar la línea de rtc_bbat_vchg**.

Reiniciar:

```bash
sudo reboot
```

Verificar PCIe Gen 3:

```bash
# Debe decir "Speed 8GT/s" (Gen 3) y no "5GT/s" (Gen 2)
sudo lspci -vv | grep -i "lnksta"
```

## 9. Instalar y verificar Hailo

Referencia: https://www.raspberrypi.com/documentation/computers/ai.html#update

```bash
sudo apt install -y hailo-all
sudo reboot
```

Esto instala:
- Driver de kernel y firmware del Hailo
- HailoRT (runtime middleware)
- hailo_platform (Python SDK que usamos en detect.py)

Verificar:

```bash
# Verificar que el dispositivo se detecta por PCIe
lspci | grep Hailo
# Debe mostrar: "Co-processor: Hailo Technologies Ltd."

# Identificar el dispositivo y versión de firmware
hailortcli fw-control identify
# Debe mostrar: Hailo-8L, firmware version, etc.
# Anotar la versión de firmware para verificar que esté al día.
# Se actualiza automáticamente con: sudo apt update && sudo apt upgrade
```

**Importante**: el driver y el runtime de HailoRT deben tener la misma versión.
Si hay mismatch (ej. 4.20 vs 4.21), reinstalar con `sudo apt install hailo-all`.

## 10. Instalar nexmon (WiFi monitor mode)

El CYW43455 integrado no soporta monitor mode por defecto. Los paquetes de nexmon
(originalmente de Kali Linux) parchean el firmware y el driver para habilitarlo.

Referencia: https://www.kali.org/blog/raspberry-pi-wi-fi-glow-up/

```bash
# Instalar dependencias
sudo apt install -y dkms aircrack-ng

# Descargar los paquetes de nexmon (no agregar el repo de Kali)
wget http://http.kali.org/pool/non-free-firmware/f/firmware-nexmon/firmware-nexmon_0.2_all.deb
wget http://http.kali.org/pool/contrib/b/brcmfmac-nexmon-dkms/brcmfmac-nexmon-dkms_6.12.2_all.deb

# Instalar (--force-overwrite por conflicto con firmware-brcm80211)
sudo dpkg -i --force-overwrite firmware-nexmon_0.2_all.deb
sudo dpkg -i brcmfmac-nexmon-dkms_6.12.2_all.deb

sudo reboot
```

Verificar después del reinicio:

```bash
# Debe listar "monitor" como modo soportado
iw phy phy0 info | grep -i monitor

# Probar captura de probe requests (15 segundos)
sudo airmon-ng start wlan0
sudo timeout 15 tcpdump -i wlan0mon -e -c 10 'subtype probe-req'
sudo airmon-ng stop wlan0mon
```

**Nota**: el error "Unknown error 524" de airmon-ng es esperado y no afecta la captura.

## 11. Verificar cámaras

```bash
# Verificar que se detectan las dos cámaras
rpicam-hello --list-cameras
# Debe listar 2 cámaras

# Capturar una imagen de cada cámara
rpicam-still -o test_cam0.jpg --camera 0
rpicam-still -o test_cam1.jpg --camera 1
```

Como el dispositivo corre headless, bajar las imágenes a tu PC para verificarlas:

```bash
# Desde tu PC (PowerShell o terminal)
scp pi@people-counter.local:test_cam0.jpg .
scp pi@people-counter.local:test_cam1.jpg .
```

Verificar que ambas imágenes se ven bien, que el ángulo y la orientación son correctos,
y que las dos cámaras apuntan a la misma zona.

## 12. Verificación final

```bash
# Batería RTC: verificar que está cargando
cat /sys/devices/platform/soc*/soc*:rpi_rtc/rtc/rtc0/charging_voltage
# Debe mostrar 3000000 (3V)

# Temperatura CPU
vcgencmd measure_temp
```

La temperatura en idle debería estar por debajo de 60°C.
Si está muy alta, verificar que el disipador esté bien montado.

## 13. Instalar dependencias y el proyecto

```bash
# Instalar dependencias del sistema
sudo apt install -y \
  python3-pip python3-venv \
  python3-opencv python3-numpy \
  libopencv-dev \
  git

# Clonar el repo
sudo git clone https://github.com/maurogasparri/people-counter.git /usr/src/people-counter
sudo chown -R pi:pi /usr/src/people-counter
cd /usr/src/people-counter

# Crear virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e ".[dev]"

# Verificar tests
pytest -v
```

## 14. Instalar servicios del sistema

```bash
# Logrotate (rotación de logs)
sudo cp /usr/src/people-counter/config/logrotate.conf /etc/logrotate.d/people-counter

# WiFi monitor mode (arranca antes del pipeline, pone wlan0 en monitor)
sudo cp /usr/src/people-counter/config/wifi-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wifi-monitor

# Servicio principal (pipeline)
sudo cp /usr/src/people-counter/config/people-counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable people-counter

# Timer de reset diario (dedup counters a las 04:00)
sudo cp /usr/src/people-counter/config/people-counter-reset.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter-reset.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable people-counter-reset.timer

# Crear directorios de datos
sudo mkdir -p /etc/people-counter/certs /var/lib/people-counter /var/log/people-counter
sudo chown -R pi:pi /etc/people-counter /var/lib/people-counter /var/log/people-counter
```

## 15. PoC — Siguiente paso

Una vez que todo lo anterior funciona, abrí Claude Code en el directorio
del repo y pedile:

"Implementá capture.py para captura estéreo sincronizada
de las dos cámaras OV5647 vía libcamera/picamera2.
Leé CLAUDE.md para contexto."

A partir de ahí, sprint por sprint según el plan del CLAUDE.md.

## Troubleshooting

- **Cámaras no detectadas**: verificar que los cables CSI están bien
  insertados. El conector tiene un clip que se levanta, se inserta el
  flat y se baja el clip.
- **Hailo no detectado**: verificar que el M.2 está bien insertado en
  el HAT. Correr `lspci` y buscar Hailo.
- **Boot loop**: sacar el HAT y bootear solo con la Pi para descartar
  problemas de alimentación. La fuente USB-C debe ser de 5V/5A.

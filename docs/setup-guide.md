# Guía de puesta en marcha — People Counter PoC

## 1. Preparar la MicroSD

Desde tu PC con Windows:

1. Descargar Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Insertar la MicroSD de 32GB
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
   - Cámara izquierda → CAM0
   - Cámara derecha → CAM1
   - Orientar ambas igual (el conector flat tiene un lado con contactos expuestos)
   - **Usar OV5647 160° con filtro IR** (NO usar "NoIR" / "night vision" — producen tinte violeta y afectan la profundidad)
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

Los pasos 4 a 9 se pueden ejecutar automáticamente:

```bash
sudo git clone https://github.com/maurogasparri/people-counter.git /usr/src/people-counter
sudo chown -R pi:pi /usr/src/people-counter
sudo bash /usr/src/people-counter/scripts/setup_device.sh
```

O seguir el paso a paso manual a continuación.

## 4. Forzar Ethernet a 100 Mbps Full Duplex

Si el switch o inyector PoE tiene problemas de autonegociación, forzar la velocidad:

```bash
# Verificar nombre de la conexión
nmcli connection show

# Forzar 100 Mbps Full Duplex (persistente)
sudo nmcli connection modify "Wired connection 1" \
  802-3-ethernet.auto-negotiate no \
  802-3-ethernet.speed 100 \
  802-3-ethernet.duplex full
sudo nmcli connection up "Wired connection 1"

# Verificar
ethtool eth0 | grep -E 'Speed|Duplex|Auto'
# Speed: 100Mb/s / Duplex: Full / Auto-negotiation: off
```

> Reemplazar `"Wired connection 1"` por el nombre que muestre `nmcli connection show`.

## 5. Configurar sistema (headless + config.txt)

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

## 6. Instalar Hailo

Referencia: https://www.raspberrypi.com/documentation/computers/ai.html#update

```bash
sudo apt install -y hailo-all
sudo reboot
```

## 7. Instalar nexmon (WiFi monitor mode)

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

## 8. Instalar el proyecto

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

# Instalar el proyecto y todas las dependencias
sudo pip install --break-system-packages --root-user-action=ignore -e ".[dev]"

# Descargar modelo YOLOv8n para Hailo-8L
PYTHONPATH=. python3 scripts/download_model.py hef

# Verificar tests
pytest -v
```

## 9. Configurar el dispositivo

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

## 10. Instalar servicios del sistema

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

## 11. Verificar todo

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

## 12. Ajuste de foco y calibración estéreo

### 12.1. Ajustar foco

Antes de calibrar, ajustar el foco de ambas cámaras. Las OV5647 tienen un anillo
de foco manual que se gira con pinza de punta fina.

Poner un objeto con detalle (diario, patrón ChArUco) a la distancia de trabajo (~3m)
y correr el asistente de foco:

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/focus_assist.py
```

Abrir **http://people-counter.local:8080** para ver el preview en vivo de ambas
cámaras lado a lado. El script muestra un puntaje de foco en tiempo real (varianza
del Laplaciano). Girar el anillo hasta que el número sea lo más alto posible.

Opciones útiles:
- `--grid` — superpone una grilla 3×3 con puntaje de foco por celda (útil para lentes gran angular)
- `--no-zoom` — muestra el frame completo en vez del zoom al centro

Ctrl+C para salir y guardar los últimos frames.

Para verificar visualmente, bajar las imágenes a la PC:

```bash
scp pi@people-counter.local:/tmp/focus_left.jpg .
scp pi@people-counter.local:/tmp/focus_right.jpg .
```

### 12.2. Calibración estéreo

Los parámetros del board ChArUco deben coincidir exactamente con el patrón impreso.
El board de referencia (calib.io) es 7×5, checker 50mm, marker 37mm.

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/calibrate.py capture \
  --columns 7 --rows 5 --square-length 50 --marker-length 37 \
  --count 30
```

Abrir **http://people-counter.local:8080** para ver el preview en vivo con detección
de corners y grilla de cobertura. El script captura automáticamente cada ~1.5 segundos
cuando el board es detectado.

Mover el patrón ChArUco entre capturas a distintas posiciones (centro, bordes, esquinas),
ángulos (inclinado, rotado) y distancias (0.5-1.5m). Cubrir toda la grilla.
Buena iluminación, sin reflejos directos sobre el papel. Después calibrar:

```bash
PYTHONPATH=. python3 scripts/calibrate.py calibrate \
  --columns 7 --rows 5 --square-length 50 --marker-length 37 \
  --input-dir ./calibration/captures \
  --output /etc/people-counter/calibration.npz
```

El modo por defecto es `pinhole` (recorta el centro de la imagen para evitar distorsión
de lentes gran angular). Para lentes 160-170° que requieran FOV completo, agregar
`--mode fisheye`.

## 13. Habilitar overlayfs (protección de la SD)

**Hacer esto como último paso**, después de que todo funcione (calibración verificada,
servicios corriendo, config definitiva). Una vez activo, la partición root queda
read-only y los cambios fuera de los paths permitidos se pierden al reiniciar.

```bash
# Crear los paths read-write antes de activar
sudo mkdir -p /var/lib/people-counter /var/log/people-counter /tmp

# Activar overlay filesystem
sudo raspi-config nonint do_overlayroot 0
```

Esto monta `/` como read-only con una capa de escritura en RAM. Los directorios
que necesitan persistir entre reinicios ya están en paths separados:

- `/var/lib/people-counter/` — SQLite buffer, dedup DB
- `/var/log/people-counter/` — logs (rotados, 7 días)
- `/etc/people-counter/` — config y certificados
- `/tmp/` — capturas temporales

> **Para desactivar** (ej: actualizar software o reconfigurar):
> ```bash
> sudo raspi-config nonint do_overlayroot 1
> sudo reboot
> # ... hacer cambios ...
> sudo raspi-config nonint do_overlayroot 0
> sudo reboot
> ```

> **Nota**: `raspi-config` versión GUI también lo ofrece en Performance → Overlay File System.

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

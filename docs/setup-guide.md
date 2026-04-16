# Guía de puesta en marcha — People Counter PoC

## 1. Preparar la microSD

Desde tu PC con Windows:

1. Descargar Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Insertar la microSD (SanDisk Extreme 64GB)
3. En Imager:
   - OS: Raspberry Pi OS (64-bit) — Trixie
   - Storage: tu microSD
   - Settings (engranaje):
     - Hostname: `people-counter`
     - Enable SSH: ✓ (password o key)
     - Set username: `pi`
     - Set password: [tu password]
     - Locale: America/Argentina/Buenos_Aires

(No configurar WiFi — el dispositivo se conecta por Ethernet y el WiFi queda exclusivo para monitor mode/probe capture.)
4. Write y esperar

## 2. Ensamblaje físico

Orden recomendado:

1. **Raspberry Pi AI HAT+ 13 TOPS** → stackearlo sobre la Raspberry Pi 5 (GPIO + FFC al puerto PCIe)
2. **Waveshare PoE HAT (H)** → conectar por cables dupont (5V, GND, y los pines PoE del header) — no se stackea
3. **Cámaras** → conectar los cables CSI a los puertos CAM0 y CAM1 de la Pi
   - Cámara izquierda (mirando desde la cámara hacia la escena) → CAM1
   - Cámara derecha (mirando desde la cámara hacia la escena) → CAM0
   - Orientar ambas igual (el conector flat tiene un lado con contactos expuestos)
   - **Usar Arducam IMX708 120 HFOV con filtro IR** (modelo B0310)
4. **Raspberry Pi Active Cooler** → clip sobre el SoC de la Pi y conectar el cable PWM al header de 4 pines del fan
5. **microSD** → insertar la tarjeta ya flasheada
6. **Batería RTC** → conectar al conector J5 de la Pi (entre los puertos USB y el GPIO).
   Usar una batería recargable LiMnO2 como la ML2032 (no confundir con CR2032 que no es recargable).
7. **NO conectar PoE todavía** — para el PoC usá la fuente USB-C estándar

## 3. Primer boot y actualización

1. Conectar por Ethernet + SSH (`ssh pi@people-counter.local`), o monitor HDMI + teclado
2. Esperar que termine el primer boot (puede tardar 2-3 min)

```bash
uname -a
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

Los pasos 4 a 10 se pueden ejecutar automáticamente:

```bash
sudo git clone https://github.com/maurogasparri/people-counter.git /usr/src/people-counter
sudo chown -R pi:pi /usr/src/people-counter
sudo bash /usr/src/people-counter/scripts/setup_device.sh
```

O seguir el paso a paso manual a continuación.

## 4. Configurar sistema (headless + config.txt)

Deshabilita el entorno gráfico, instala watchdog y parchea `/boot/firmware/config.txt`
(RTC, PCIe Gen 3, USB current para el PoE HAT, overlays IMX708 por puerto). El bloque
de `config.txt` es idempotente.

```bash
sudo raspi-config nonint do_boot_behaviour B1

sudo apt install -y watchdog
sudo sed -i 's/^#watchdog-device/watchdog-device/' /etc/watchdog.conf
sudo sed -i 's/^#max-load-1/max-load-1/' /etc/watchdog.conf
sudo systemctl enable watchdog
sudo systemctl start watchdog

CFG=/boot/firmware/config.txt
sudo sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' $CFG
grep -q "^dtoverlay=imx708,cam0" $CFG || sudo sed -i '/^\[all\]/a dtoverlay=imx708,cam0' $CFG
grep -q "^dtoverlay=imx708,cam1" $CFG || sudo sed -i '/^\[all\]/a dtoverlay=imx708,cam1' $CFG
grep -q "^dtparam=rtc_bbat_vchg" $CFG    || echo "dtparam=rtc_bbat_vchg=3000000" | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=pciex1_gen=3" $CFG     || echo "dtparam=pciex1_gen=3"     | sudo tee -a $CFG > /dev/null
grep -q "^usb_max_current_enable=1" $CFG || echo "usb_max_current_enable=1" | sudo tee -a $CFG > /dev/null
```

Si usás una pila no recargable (CR2032), **no agregar la línea de rtc_bbat_vchg**.

Referencias:
- RTC: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#add-a-backup-battery
- PCIe Gen 3: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#pcie-gen-3-0

## 5. Instalar Hailo (mínimo necesario)

Solo instalamos los 3 paquetes que necesitamos para correr inferencia desde
Python (runtime + driver PCIe + bindings). Evitamos `hailo-all` que arrastra
TAPPAS, modelos de ejemplo y la integración con rpicam, que no usamos.

```bash
sudo apt install -y hailort hailort-pcie-driver python3-hailort
```

Referencia oficial (recomienda `hailo-all` como path estándar):
https://www.raspberrypi.com/documentation/computers/ai.html#update

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

Este reboot aplica también los cambios de los pasos 4 y 5.

## 7. Verificar hardware base

Tras el reboot, verificar que Hailo y las cámaras quedaron operativos antes de seguir:

```bash
hailortcli fw-control identify           # Hailo-8L, fw 4.23+
lspci | grep -i hailo                    # debe listar el chip
python3 -c "import hailo_platform; print(hailo_platform.__version__)"

rpicam-hello --list-cameras              # deben aparecer 2x imx708 (CAM0 y CAM1)

dmesg | grep nexmon                      # firmware nexmon cargado
```

Si alguno falla, revisar el Troubleshooting antes de avanzar.

## 8. Instalar el proyecto

```bash
sudo apt install -y \
  python3-pip \
  python3-opencv python3-numpy \
  libopencv-dev \
  git

sudo git clone https://github.com/maurogasparri/people-counter.git /usr/src/people-counter
sudo chown -R pi:pi /usr/src/people-counter
cd /usr/src/people-counter

sudo pip install --break-system-packages --root-user-action=ignore -e ".[dev]"

PYTHONPATH=. python3 scripts/download_model.py hef

pytest -v
```

## 9. Configurar el dispositivo

```bash
sudo mkdir -p /etc/people-counter/certs /var/lib/people-counter /var/log/people-counter
sudo chown -R pi:pi /etc/people-counter /var/lib/people-counter /var/log/people-counter

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
sudo cp /usr/src/people-counter/config/wifi-monitor.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter-reset.service /etc/systemd/system/
sudo cp /usr/src/people-counter/config/people-counter-reset.timer /etc/systemd/system/
sudo cp /usr/src/people-counter/config/logrotate.conf /etc/logrotate.d/people-counter

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
```

Y desde tu PC:

```bash
scp pi@people-counter.local:/tmp/test_cam*.jpg .
```

## 12. Ajuste de foco y calibración estéreo

### 12.1. Ajustar foco

**Crítico para estéreo**: ambas cámaras deben tener el foco lo más parejo posible.
Diferencias de foco entre L y R degradan la calidad del depth map más que cualquier
otro factor. Verificar también que el bracket mecánico no flexa — si el baseline
cambia entre calibración y operación, el depth deriva.

Las IMX708 tienen un anillo de foco manual M12 que se gira con pinza de punta fina.
Poner un objeto con detalle (diario, patrón ChArUco) a la **altura de las cabezas**
de las personas (típicamente ceiling_height - 1.7m, ej: techo a 3.5m → objeto a ~1.8m
de la cámara). Con M12 fija y 120° HFOV, la depth-of-field a 2m cubre ~1m–infinito.
Correr el asistente de foco:

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/focus_assist.py
```

Abrir **http://people-counter.local:8080** para ver el preview en vivo de ambas
cámaras lado a lado. El script muestra un puntaje de foco en tiempo real (varianza
del Laplaciano). Girar el anillo hasta que el número sea lo más alto posible.

Opciones útiles:
- `--grid` — superpone una grilla 3x3 con puntaje de foco por celda
- `--no-zoom` — muestra el frame completo en vez del zoom al centro

Ctrl+C para salir y guardar los últimos frames.

Para verificar visualmente, bajar las imágenes a la PC:

```bash
scp pi@people-counter.local:/tmp/focus_left.jpg .
scp pi@people-counter.local:/tmp/focus_right.jpg .
```

### 12.2. Calibración estéreo

Los parámetros del board ChArUco deben coincidir exactamente con el patrón impreso.
Board recomendado para IMX708: **11x7 squares, checker 35mm, marker 26mm, DICT_5X5_100, A3 landscape** (385x245mm impreso, 60 esquinas internas, 38 markers). Ya generado en `calibration/charuco_11x7_sq35mm_mk26mm_dict5X5_a3_calibio.pdf`. Imprimir desde Adobe Reader con "Actual size" (NO "Fit to page"), pegar sobre foam board rígido. Verificar el ancho total con calibre — debe medir 385mm. Si difiere, usar el valor medido en `--square-length`.

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/calibrate.py capture \
  --columns 11 --rows 7 --square-length 35 --marker-length 26 \
  --count 30
```

Abrir **http://people-counter.local:8080** para ver el preview en vivo con detección
de corners y grilla de cobertura. El script captura automáticamente cada ~1.5 segundos
cuando el board es detectado.

Mover el patrón ChArUco entre capturas a distintas posiciones (centro, bordes, esquinas),
ángulos (inclinado, rotado) y **distancias 0.5-3m** (cubrir todo el rango operativo,
no solo el de calibración — el modelo extrapola peor fuera del rango calibrado).
Cubrir toda la grilla. Buena iluminación, sin reflejos directos sobre el papel.
Después calibrar:

```bash
PYTHONPATH=. python3 scripts/calibrate.py calibrate \
  --columns 11 --rows 7 --square-length 35 --marker-length 26 \
  --input-dir ./calibration/captures \
  --output /etc/people-counter/calibration.npz
```

### 12.3. Validación post-calibración

El RMS de reproyección (que reporta `calibrate`) **no es suficiente** para validar
calidad para depth map. Hay que validar con métrica real:

1. **Error de profundidad a distancias conocidas**: poner un objeto a 1m, 2m y 3m
   medidos con cinta. Correr `scripts/diagnose_depth.py` y comparar el depth
   estimado con la distancia real. Error esperado: <5% a 2m, <10% a 3m.
2. **Error de altura estimada**: poner un objeto de altura conocida (ej: caja de 1m)
   a distintas distancias. Verificar que la altura estimada sea consistente.
3. **Consistencia centro vs bordes**: el error en bordes no debería ser >2x el del
   centro. Si lo es, recalibrar con más capturas en periferia.

Si la validación falla, recapturar (más poses, mejor cobertura, foco más parejo)
antes de pasar a producción.

## 13. Habilitar overlayfs (protección de la SD)

**Hacer esto como último paso**, después de que todo funcione (calibración verificada,
servicios corriendo, config definitiva). Una vez activo, la partición root queda
read-only y los cambios fuera de los paths permitidos se pierden al reiniciar.

```bash
sudo mkdir -p /var/lib/people-counter /var/log/people-counter /tmp
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
- **Hailo no detectado**: verificar que el AI HAT+ está bien stackeado
  sobre la Pi y que el cable FFC al puerto PCIe está firme.
  Correr `lspci` y buscar Hailo.
- **Boot loop**: sacar el HAT y bootear solo con la Pi para descartar
  problemas de alimentación. La fuente USB-C debe ser de 5V/5A.
- **WiFi monitor mode no funciona**: verificar con `dmesg | grep nexmon`
  que el firmware nexmon está cargado. Si no aparece, reinstalar firmware-nexmon.
- **picamera2 no importa**: verificar que está instalado con
  `sudo apt install python3-picamera2`.
- **"Unknown error 524" en airmon-ng**: es esperado con nexmon en RPi5,
  no afecta la captura.

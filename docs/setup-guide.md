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

1. **Raspberry Pi AI HAT+ 13 TOPS** → stackearlo sobre la Raspberry Pi 5 (GPIO + cable plano PCIe)
2. **Waveshare PoE HAT (H)** → conectar por dupont — no se stackea. Usar **2× 5V + 2× GND** para repartir corriente sin sobrecargar un solo contacto:

   | Header pin | RPi5 | PoE HAT |
   |------------|------|---------|
   | Pin 2      | 5V   | 5V      |
   | Pin 4      | 5V   | 5V      |
   | Pin 6      | GND  | GND     |
   | Pin 9      | GND  | GND     |

3. **Cámaras** → conectar los cables CSI a los puertos CAM0 y CAM1 de la Pi
   - Cámara izquierda (mirando desde la cámara hacia la escena) → CAM1
   - Cámara derecha (mirando desde la cámara hacia la escena) → CAM0
   - Orientar ambas igual (el conector flat tiene un lado con contactos expuestos)
   - **Usar Arducam IMX708 120 HFOV con filtro IR** (modelo B0310)
4. **Raspberry Pi Active Cooler** → clip sobre el SoC de la Pi y conectar el cable PWM al header de 4 pines del fan
5. **LED RGB de status** (3mm common cathode, 4 patas) → conectar por dupont 2x2 al bloque pin 11/12/13/14 con resistencias en serie con cada ánodo:

   | Header pin | GPIO | Pata LED | Resistor en serie |
   |------------|------|----------|-------------------|
   | Pin 11     | GPIO 17 | R    | 150 Ω             |
   | Pin 12     | GPIO 18 | G    | 100 Ω             |
   | Pin 13     | GPIO 27 | B    | 100 Ω             |
   | Pin 14     | GND  | cátodo (pata más larga) | — |

   Resistencias asimétricas porque G y B (InGaN, Vf≈3.1V) tienen apenas 0.2V de headroom contra el supply de 3.3V mientras que R (AlGaInP, Vf≈2.1V) tiene 1.2V. Los valores apuntan a brillo perceptualmente parejo entre canales, no a corrientes iguales — sin esa asimetría las mezclas tiran al verde por la mayor eficiencia luminosa del eye response.

6. **microSD** → insertar la tarjeta ya flasheada
7. **Batería RTC** → conectar al conector J5 de la Pi (entre los puertos USB y el GPIO).
   Usar una batería recargable LiMnO2 como la ML2032 (no confundir con CR2032 que no es recargable).
8. **NO conectar PoE todavía** — para el PoC usá la fuente USB-C estándar

## 3. Primer boot y actualización

1. Conectar por Ethernet + SSH (`ssh pi@people-counter.local`), o monitor HDMI + teclado
2. Esperar que termine el primer boot (puede tardar 2-3 min)

```bash
uname -a
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 3.1. (Opcional) Firewall corporativo con TLS interception

Si la red del lugar inspecciona TLS y reemplaza los certs por uno de la empresa,
hay que instalar el CA corporativo antes de seguir o `git clone` va a fallar
con `certificate verify failed`. Pedir el CA root a IT (formato `.cer` o `.pem`)
y copiarlo a `/tmp/empresa-ca.cer`.

```bash
# 1. Detectar formato (DER vs PEM) y convertir si hace falta
file /tmp/empresa-ca.cer
# - "Certificate, Version=3" o "data" → DER, convertir:
openssl x509 -inform der -in /tmp/empresa-ca.cer -out /tmp/empresa-ca.crt
# - "ASCII text" → PEM, solo renombrar:
# cp /tmp/empresa-ca.cer /tmp/empresa-ca.crt

# 2. Inspeccionar (subject + vigencia)
openssl x509 -in /tmp/empresa-ca.crt -noout -subject -issuer -dates

# 3. Instalar
sudo cp /tmp/empresa-ca.crt /usr/local/share/ca-certificates/empresa-ca.crt
sudo update-ca-certificates

# 4. Verificar
git ls-remote https://github.com/maurogasparri/people-counter.git | head -3
```

### 3.2. Setup automático

Los pasos 4 a 10 se pueden ejecutar automáticamente:

```bash
sudo git clone https://github.com/maurogasparri/people-counter.git /usr/src/people-counter
sudo chown -R pi:pi /usr/src/people-counter
sudo bash /usr/src/people-counter/scripts/setup_device.sh
```

O seguir el paso a paso manual a continuación.

## 4. Configurar sistema (headless + config.txt)

Deshabilita el entorno gráfico, instala watchdog y parchea `/boot/firmware/config.txt`
(RTC, PCIe Gen 3, USB current para el PoE HAT, overlays IMX708, low-power tweaks).
El bloque de `config.txt` es idempotente.

Las líneas de bajo consumo apagan el audio PWM, los LEDs onboard (ACT + power)
y los LEDs del jack Ethernet (link + activity) — el RGB externo es la única
fuente visual de estado.

```bash
sudo raspi-config nonint do_boot_behaviour B1

# Preferir IPv4 en getaddrinfo. Si la red del local no rutea IPv6
# (caso típico de ISPs residenciales/comerciales sin IPv6 nativo),
# getaddrinfo devuelve la AAAA primero y paho-mqtt intenta IPv6 →
# el SYN no llega a ningún lado y el cliente cuelga ~60-130s antes
# de fallback a IPv4. Descomentar esta línea invierte la preferencia.
sudo sed -i 's|^#precedence ::ffff:0:0/96  100|precedence ::ffff:0:0/96  100|' /etc/gai.conf

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
# Audio off + onboard LEDs off + Ethernet jack LEDs off (case semi-translúcido)
sudo sed -i 's/^dtparam=audio=.*/dtparam=audio=off/' $CFG \
    || echo "dtparam=audio=off" | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=act_led_trigger="     $CFG || echo "dtparam=act_led_trigger=none"     | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=act_led_activelow="   $CFG || echo "dtparam=act_led_activelow=off"    | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=power_led_trigger="   $CFG || echo "dtparam=power_led_trigger=none"   | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=power_led_activelow=" $CFG || echo "dtparam=power_led_activelow=off"  | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=eth_led0="            $CFG || echo "dtparam=eth_led0=4"               | sudo tee -a $CFG > /dev/null
grep -q "^dtparam=eth_led1="            $CFG || echo "dtparam=eth_led1=4"               | sudo tee -a $CFG > /dev/null
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
  python3-opencv python3-numpy python3-scipy \
  python3-yaml python3-paho-mqtt \
  libopencv-dev \
  python3-gpiozero python3-lgpio \
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

### 11.1. LED de status — código de colores

El LED RGB es la fuente visual de estado para diagnóstico en sitio sin SSH.
Cascada worst-first por capa (hardware > pipeline > internet > cloud > OK):

| LED | Patrón | Significado | Acción del operador |
|-----|--------|-------------|---------------------|
| Apagado | — | Sin power (PoE caído / cable desconectado) | Verificar PoE, switch, cable Ethernet |
| Rojo | Fijo | Boot failure (servicio no levanta) | Power cycle; si persiste, contactar soporte |
| Amarillo | Fijo | Hardware roto (cámara, Hailo, temp >80°C, disco lleno) | Power cycle; si persiste, reemplazo |
| Amarillo | Parpadeante | Pipeline stalled o software crasheó | Esperar 1 min al auto-restart; si persiste, power cycle |
| Verde | Parpadeante | Sin internet (ethernet up pero no llega afuera) | Verificar internet del local |
| Verde | Fijo | Internet OK, AWS IoT no responde | Soporte (problema cloud, no del dispositivo) |
| Azul | Fijo | **Operación normal** | Ninguna |
| Azul | Parpadeante | Sin provisioning (certs ausentes) | Re-provisionar (instalación) |

Cascada de prioridad (peor estado wins): Off > Rojo > Amarillo fijo > Amarillo
parpadeante > Verde parpadeante > Verde fijo > Azul parpadeante > Azul fijo.

Apagar el LED si molesta en el local (ej. brillo excesivo en una zona oscura,
restricciones estéticas del retail) en `config.yaml`:

```yaml
status_led:
  enabled: false
```

## 12. Ajuste de foco y calibración estéreo

### 12.1. Ajustar foco

**Crítico para estéreo**: ambas cámaras deben tener el foco lo más parejo posible.
Diferencias de foco entre L y R degradan la calidad del depth map más que cualquier
otro factor. Verificar también que el bracket mecánico no flexa — si el baseline
cambia entre calibración y operación, el depth deriva.

Las IMX708 tienen un anillo de foco manual M12 que se gira con pinza de punta fina.
Poner el **board ChArUco** (el mismo que vas a usar para calibración) a **1.5m**
de las cámaras. El asistente de foco lo detecta automáticamente y valida la
distancia. Con M12 fija y 120° HFOV, focar a 1.5m peakea el DoF sobre el rango
operativo del bbox de detección (cabeza+pie) en toda la flota mount 2.0–3.5m,
con DoF efectivo 0.59m–infinito.

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/focus_assist.py
```

Abrir **http://people-counter.local:8080**. La UI muestra:
- Barras de sharpness por cámara (centro + corners absoluto) con zona verde = target
- Distancia detectada del board (objetivo **1.30–1.70m** por default — lab protocol universal, override con `--target-distance-min-mm` / `--target-distance-max-mm`)
- Simetría L/R
- Warnings de iluminación/glare en vivo

Girar los anillos hasta que todas las barras estén verdes. Click **FINALIZAR**
cuando pase el check global — salva un reporte HTML en `/tmp/focus_report_*.html`
con todas las métricas y los frames finales embebidos.

**Escena compacta**: si el cuarto de test es chico y el board llena el frame
(bbox >25%), el tool auto-detecta la situación y omite el check de corners
porque los bordes ven paredes a distancia no relacionada con el board. Banner
azul en la UI lo indica. Forzá el modo con `--scene=compact|full` si hace falta.

### 12.2. Calibración estéreo (modo wizard guiado)

Board recomendado para IMX708: **9x6 squares, checker 45mm, marker 33mm, DICT_4X4_100, A3 landscape** (405x270mm impreso, 40 esquinas internas, 27 markers). Ya generado en `calibration/calib.io_charuco_420x297_6x9_45_33_DICT_4X4.pdf`. Imprimir desde Adobe Reader con "Actual size" (NO "Fit to page"), laminar OPP mate sobre PVC rígido 3mm (foam flexa demasiado en A3). Verificar con calibre — un square debe medir exactamente 45.0mm. Si difiere, pasar el valor medido con `--square-mm`.

**Comando de una sola pasada** — pre-flight + captura guiada + calibración + residuales + ground-truth + reporte HTML con QR:

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/calibrate.py wizard \
  --device-id DEV-001 \
  --output /etc/people-counter/calibration.npz
```

El wizard tiene 5 fases:

**Fase 0 — Pre-flight**: verifica que el puerto 8080 está libre, los directorios de salida son escribibles, hay >500MB de disco y hace backup de `calibration.npz` si ya existe. Aborta con `❌` si falta algo crítico.

**Fase 1 — Captura guiada**. Abrir **http://people-counter.local:8080**. La UI muestra una **silueta fantasma** de dónde poner el board para cada pose (20 poses cubriendo 3 distancias × 5 posiciones × tilts variados).
- Las **primeras 6 capturas** usan intrínsecos nominales + tolerancia suelta (25px, 12 esquinas). Al completarlas, el wizard ajusta los intrínsecos reales del sensor y pasa a **tolerancia estricta** (12px, 15 esquinas) — cada ejemplar de IMX708 tiene focal distinto ±3-5%, así la silueta se dibuja con el K real.
- Matching por **ID de esquina ChArUco** (no minAreaRect) — funciona bien incluso en poses con tilt fuerte.
- Auto-captura cuando: (a) alineado, (b) estable 1.5s, (c) L∩R ≥ 15 esquinas en común, (d) frames L y R sincronizados a ≤5ms.
- Warnings en vivo: reflejo, sub-exposición, asimetría L/R, **drift de iluminación** (si la luz cambia >25% desde el inicio).
- Botones UI: **Audio ON/OFF** (el teléfono habla — default OFF), **UNDO**, **Saltear pose**, **FINALIZAR**.

**Fase 2 — Calibración**: `calibrate_stereo` con modelo fisheye Kannala-Brandt (`cv2.fisheye.calibrate` + derivación de R/T per-pose), guarda `.npz`.

**Fase 3 — Verificación + residuales por par**: dibuja líneas epipolares en `verify_epipolar.png` y calcula el residual de reproyección por cada par capturado. En el reporte final, los pares con residual >2× la mediana se marcan como outlier (tile con borde rojo) — sabés cuáles recapturar si hace falta.

**Fase 4 — Ground-truth depth (opcional)**: el wizard te prompta en la terminal — "poné una superficie plana a X mm, escribí la distancia". Captura un frame, rectifica con la calibración recién hecha, corre SGBM + análisis de 5 zonas. El resultado (centro <5% a 2m / <10% a 3m, borde/centro <2×) queda en el HTML report. Enter vacío para saltear.

**Reporte + QR**: al terminar, el wizard levanta un HTTP server en `:8081` durante 10 min sirviendo el reporte, imprime un **QR code en la terminal** con la URL — escaneás con el teléfono y ves el reporte instantáneo, sin ssh/scp. Requiere `pip install qrcode` (opcional; si no está, muestra solo la URL). Ctrl+C para cerrar antes del timeout.

**Flags útiles**:
- `--resume` — continuar una sesión anterior interrumpida (lee `session.json` en el output_dir, salta las poses ya capturadas, re-fitea el bootstrap K si hay ≥6 capturas). Si hay una sesión previa y no pasás `--resume`, el wizard aborta para no sobreescribir.
- `--no-serve-report` — no levantar el server del reporte
- `--report-serve-sec 1800` — cambiar el tiempo de vida del server (default 600s)
- `--report-port 9090` — puerto alternativo para el report server
- `--dist-near-mm/--dist-mid-mm/--dist-far-mm` — distancias custom para la secuencia de poses (útil si el local tiene techo alto o pasillo ancho)

**Si preferís el flujo manual** (capture y calibrate por separado), los flags
del board tienen defaults canónicos — ya no hace falta pasarlos:

```bash
PYTHONPATH=. python3 scripts/calibrate.py capture --guided
PYTHONPATH=. python3 scripts/calibrate.py calibrate \
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

## 14. Backup y disaster recovery

Tres artefactos por dispositivo justifican backup. El resto (`buffer.db`,
`dedup.db`, logs) es volátil por diseño y se pierde al reiniciar sin impacto.

| Path en la Pi | Qué es | Estrategia |
|---------------|--------|------------|
| `/etc/people-counter/config.yaml` | device_id, store, ROI, endpoint MQTT | Queda en `provisioned/<id>/` durante `create` |
| `/etc/people-counter/calibration.npz` | calibración estéreo per-unidad | Pull al workstation con `harvest` post-calibración |
| `/etc/people-counter/certs/` | X.509 client cert per-device | **No se respalda** — se re-emite ante restore (cert rotation) |

### 14.1. Backup post-calibración

Después de validar la calibración, traer la `calibration.npz` al workstation:

```bash
# Desde el workstation (donde está provisioned/<id>/)
python scripts/provision.py harvest \
  --device-id store-001-cam-01 \
  --host people-counter.local
```

Eso deja `provisioned/<device-id>/calibration.npz` listo para un re-deploy.

### 14.2. Restore después de SD muerta

1. **Flashear nueva SD + setup steps 1-10** (o `setup_device.sh`).

2. **Reprovisionar el cert** desde el workstation. Revoca el viejo en
   AWS IoT Core y emite uno nuevo asociado a la misma thing:

   ```bash
   python scripts/provision.py reprovision --device-id store-001-cam-01
   ```

   El cert viejo queda detacheado de la thing, sin policies y `INACTIVE`
   en IoT Core, así si la SD original cae en manos ajenas no puede
   impersonar al device.

3. **Deploy de config + cert nuevo + calibration** (la `calibration.npz`
   se pushea automáticamente si existe en `provisioned/<id>/`):

   ```bash
   python scripts/provision.py deploy \
     --device-id store-001-cam-01 \
     --host people-counter.local
   ```

4. **Restart del servicio** en la Pi:

   ```bash
   ssh pi@people-counter.local 'sudo systemctl restart people-counter'
   ```

`provision.py reprovision` y `provision.py harvest` requieren AWS CLI
configurado en el workstation con permisos de admin de IoT Core. **Nunca
correrlos desde la Pi** — esas credenciales no deben vivir en una SD que
sale del entorno controlado.

## Troubleshooting

- **Cámaras no detectadas**: verificar que los cables CSI están bien
  insertados. El conector tiene un clip que se levanta, se inserta el
  flat y se baja el clip.
- **Hailo no detectado**: verificar que el AI HAT+ está bien stackeado
  sobre la Pi y que el cable plano del puerto PCIe está firme.
  Correr `lspci` y buscar Hailo.
- **Boot loop**: sacar el HAT y bootear solo con la Pi para descartar
  problemas de alimentación. La fuente USB-C debe ser de 5V/5A.
- **WiFi monitor mode no funciona**: verificar con `dmesg | grep nexmon`
  que el firmware nexmon está cargado. Si no aparece, reinstalar firmware-nexmon.
- **picamera2 no importa**: verificar que está instalado con
  `sudo apt install python3-picamera2`.
- **"Unknown error 524" en airmon-ng**: es esperado con nexmon en RPi5,
  no afecta la captura.
- **`airmon-ng start wlan0` se cuelga indefinidamente**: phy0 está
  soft-blocked por rfkill. Diagnóstico:
  `sudo rfkill list wifi` → `Soft blocked: yes`. Fix:
  `sudo rfkill unblock wifi`. El pipeline lo hace automáticamente en
  `setup_monitor_mode()` desde el commit que agregó rfkill unblock,
  pero si rebooteás con WiFi disabled en raspi-config, conviene tenerlo
  como step explícito antes del primer arranque del wifi-monitor service.
- **MQTT cuelga al arrancar el pipeline**: si los logs llegan hasta
  `MQTT client initialized` y se quedan ahí sin loguear `MQTT connecting
  to ...`, el cliente está intentando IPv6 sin route. Verificar con
  `ping -6 -c 2 <endpoint>` (típicamente "Network is unreachable") y
  con `getent ahosts <endpoint> | head -1` (devuelve la AAAA primero
  en vez de la A). Aplicar la fix de gai.conf del paso 4:
  `sudo sed -i 's|^#precedence ::ffff:0:0/96  100|precedence ::ffff:0:0/96  100|' /etc/gai.conf`
  y reiniciar el pipeline.

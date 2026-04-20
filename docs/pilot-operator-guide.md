# Guía de despliegue en campo — People Counter

Guía operativa para instalar un dispositivo **ya provisionado por ingeniería** en
un local de piloto. Cubre la visita: preparación, instalación física, activación
y verificación. No cubre setup del sistema ni compilación del modelo — eso está
en [`setup-guide.md`](setup-guide.md).

**Supuestos de partida (confirmados con ingeniería antes de salir):**

- El thing está registrado en AWS IoT Core.
- Los certificados X.509 están cargados en `/etc/people-counter/certs/`.
- Las unidades systemd están instaladas (`people-counter.service`,
  `people-counter-reset.service`, `people-counter-reset.timer`,
  `wifi-monitor.service`).
- La calibración estéreo y el HEF del modelo vienen precargados.
- El `device.id`, `device.store_id` y `mqtt.endpoint` vienen pre-escritos en
  `/etc/people-counter/config.yaml`.

Si algo de lo anterior **no** está, parar acá y escalar a ingeniería.

---

## 1. Preparación pre-visita

### 1.1. Checklist de hardware (caja de campo)

- [ ] Dispositivo ensamblado: RPi5 4GB + AI HAT+ (Hailo-8L) + 2x Arducam IMX708
      B0310 en bracket estéreo + PoE HAT (H) conectado por dupont + Active
      Cooler + microSD ya flasheada y provisionada.
- [ ] Cable UTP CAT6 (largo según local; llevar 10m y 20m).
- [ ] Soporte de montaje en techo (bracket o base adhesiva industrial según
      tipo de cielorraso).
- [ ] Tornillos + tarugos apropiados para el material del techo
      (yeso / durlock / losa).
- [ ] Destornillador Phillips + Torx T10.
- [ ] Nivel de burbuja (30 cm mínimo, ideal burbuja doble).
- [ ] Cinta métrica (mínimo 5m).
- [ ] Pinza de punta fina (para ajuste del anillo de foco M12 si hiciera falta).
- [ ] Escalera acorde a la altura (2.3–2.8m de montaje).
- [ ] Laptop con cliente SSH, lector de microSD, y acceso a internet (tethering
      celular si el local no tiene WiFi cliente disponible).
- [ ] Celular con linterna y cámara (para test de flicker de LEDs).
- [ ] Etiquetadora o cinta + fibra indeleble para rotular el dispositivo.

### 1.2. Confirmación con ingeniería

Antes de salir, pedir por escrito (mail o ticket):

- `device.id` asignado al dispositivo físico que se llevará.
- `device.store_id` y `device.store_name` del local.
- Endpoint de AWS IoT Core (`mqtt.endpoint`).
- Horario comercial del local (para `operating_hours`).
- Plano o foto de la puerta de acceso con ubicación tentativa de montaje.
- Confirmar que el thing figura como registrado en IoT Core.
- Confirmar que el HEF (`/usr/src/people-counter/models/yolov8n.hef`) y la
  calibración (`/etc/people-counter/calibration.npz`) fueron copiados.

### 1.3. Credenciales AWS (read-only)

Para verificar en campo que los mensajes llegan, ingeniería debe entregarte un
par de access key con política read-only sobre:

- `logs:FilterLogEvents` en el log group del Lambda de dedup.
- `iot:Publish` solamente en el topic de test (`store/<store_id>/debug`).
- `timestream:Select` sobre la DB del piloto (opcional, para queries manuales).

Configurar en la laptop:

```bash
aws configure --profile people-counter-field
# AWS Access Key ID: ...
# AWS Secret Access Key: ...
# Default region: us-east-1
```

Probar con:

```bash
aws --profile people-counter-field sts get-caller-identity
```

---

## 2. Instalación física en el local

### 2.1. Elegir la ubicación

Caminar el local con el referente del negocio y definir:

- Puerta principal de ingreso (si hay varias, priorizar la de mayor tráfico).
- Techo directamente sobre el umbral de la puerta o 30–60 cm hacia adentro.
- Evitar montaje sobre el umbral mismo si el marco sobresale — genera sombra
  y oclusión.

### 2.2. Altura de montaje

- **Rango válido: 2.3 m a 2.8 m** sobre el piso.
- Por debajo de 2.3 m: personas altas quedan fuera de frame.
- Por encima de 2.8 m: personas quedan chicas y la detección pierde recall.
- Si el techo está más alto, pedir a ingeniería extender con un caño
  descolgado (no improvisar en campo).

### 2.3. Orientación

- Cámaras **cenitales**, apuntando hacia abajo, perpendiculares al piso.
- **Baseline** (eje imaginario que une las dos cámaras) **paralelo al umbral**
  de la puerta y **perpendicular al flujo peatonal**.
- Si la puerta corre este-oeste, el baseline debe apuntar este-oeste; las
  personas cruzan el frame de norte a sur (o viceversa).

### 2.4. Nivelado

- Apoyar el nivel de burbuja sobre el bracket del dispositivo en dos ejes:
  - Eje del baseline (lateral): **±1° máximo**.
  - Eje perpendicular (frente–atrás): ±1°.
- Un desnivel de 3° a 2.5m de altura introduce ~13 cm de error en la
  proyección de la línea de conteo al piso. Aceptable, pero degrada la
  precisión del depth. **Insistir con ±1°**.
- Si el techo tiene pendiente, usar cuñas niveladoras — no compensar después
  por software.

### 2.5. Iluminación

Antes de fijar el bracket, parar frente a la ubicación elegida a la hora del
día de mayor tráfico esperado (o consultar al encargado):

- **Contraluz**: si hay vidriera o puerta vidriada detrás del cono de visión,
  el sol directo al amanecer o al atardecer (según orientación) genera
  siluetas negras. Re-ubicar 30–60 cm si es posible, o reportar a ingeniería.
- **LEDs con flicker**: filmar el techo con la cámara del celular en modo
  video. Si ves bandas o parpadeo, los LEDs tienen flicker significativo. El
  shutter de la IMX708 a 15 fps puede agarrar ciclos incompletos y meter
  ruido en el depth. Anotar y reportar — ingeniería puede ajustar exposure.
- **Sombras duras**: un spot cenital directo sobre la zona de conteo genera
  sombras negras debajo de las personas que la detección puede confundir con
  un segundo objeto. Preferir iluminación difusa.

### 2.6. Cableado y alimentación

- Tirar UTP CAT6 desde el switch PoE hasta el dispositivo. Dejar 50 cm de
  holgura en el extremo del dispositivo.
- **Confirmar con el encargado IT que el puerto del switch es 802.3at
  (PoE+, 25.5W), no 802.3af (PoE, 15W)**. 15W no alimenta al dispositivo
  bajo carga — va a hacer reboots térmicos.
- Si el switch es solo AF, pedir a ingeniería una fuente PoE+ inyectora
  externa antes de continuar.
- No conectar la alimentación todavía. Primero fijar mecánicamente.

### 2.7. Fijación mecánica

1. Marcar con lápiz los puntos de los tornillos sobre el techo.
2. Perforar con mecha apropiada al material.
3. Colocar tarugos.
4. Fijar el bracket.
5. Acoplar el dispositivo al bracket.
6. Verificar nivel **otra vez** después de atornillar (la torsión puede
   mover el bracket 1–2°).
7. Pasar el UTP por el bracket y conectar RJ45 al dispositivo.

### 2.8. Sellado y enclosure

El enclosure MVP es interior seco. Si el local tiene alguna de las
siguientes condiciones, **no energizar** y reportar a ingeniería:

- Humedad visible (manchas, goteo, condensación en cielorraso).
- Polvo industrial (local lindante con taller, obra, o cocina sin extracción).
- Temperatura ambiente >35 °C sostenida.

Ingeniería decidirá si corresponde enclosure IP54 o reubicar.

---

## 3. Activación del dispositivo

### 3.1. Conexión y SSH

Una vez energizado (LED verde de la Pi parpadeando + LEDs del switch activos):

```bash
# Desde la laptop, esperar 60–90s al primer boot
ssh pi@people-counter.local
```

Si mDNS no resuelve:

```bash
# Consultar al IT del local la IP asignada por DHCP (reserva por MAC
# es ideal; pedir que la configure antes de salir de la visita)
ssh pi@<ip-asignada>
```

Si no podés llegar por ninguno de los dos, revisar LEDs del RJ45 y del
switch. Si no hay link, probar con otro cable.

### 3.2. Preflight

El script `scripts/preflight.py` corre una batería de chequeos y te dice si
el dispositivo está listo.

```bash
cd /usr/src/people-counter
sudo PYTHONPATH=. python3 scripts/preflight.py
```

Salida esperada (todo OK):

```
People Counter — Pre-flight Check

[PASS] config_file: loaded (store-001-cam-01) from /etc/people-counter/config.yaml
[PASS] calibration_file: /etc/people-counter/calibration.npz (8 arrays)
[PASS] hef_model: /usr/src/people-counter/models/yolov8n.hef (6.2 MB)
[PASS] tls_certs: certs present; device cert valid for 364 day(s)
[PASS] cameras: 2 camera(s) enumerated
[PASS] wifi_monitor: wlan0mon present
[PASS] ble_adapter: hci0 UP RUNNING
[PASS] disk_space: /var/lib/people-counter: 52.34 GB free
[PASS] iot_endpoint: xxxxx.iot.us-east-1.amazonaws.com:8883 reachable
[PASS] systemd_units: both unit files present

==================================================
10/10 checks passed (0 skipped, 0 failed)
Device is ready.
```

### 3.3. Interpretar resultados

Cada chequeo devuelve **PASS**, **FAIL** o **SKIP**.

- **PASS**: todo bien, seguir.
- **FAIL**: **bloqueante**. Resolver antes de continuar. Ver sección 5
  (Troubleshooting) por cada FAIL específico.
- **SKIP**: el chequeo no aplica o no hay permisos suficientes. Es
  **esperado** en los siguientes casos:
  - `wifi_monitor: SKIP — not running as root`: correr preflight con
    `sudo` si querés validar monitor mode.
  - Cualquier chequeo reportando que una librería opcional no está
    instalada (ej: `cryptography not installed`): el dispositivo puede
    operar igual, pero avisar a ingeniería para que lo incluya en próximas
    imágenes.

Si hay FAILs que no podés resolver con la sección 5, escalar (sección 6).

### 3.4. Ajustes de configuración específicos del local

Editar `/etc/people-counter/config.yaml` **solo si** el preflight pasó y hay
algún campo que no quedó pre-configurado por ingeniería.

```bash
sudo nano /etc/people-counter/config.yaml
```

Campos a revisar:

**`device.store_id`** — verificar que coincida con el local.

**`counter.roi`** — rectángulo en píxeles de la imagen que define la zona de
conteo. La puerta del local debe estar dentro de este rectángulo en el
frame. Formato:

```yaml
counter:
  roi:
    x_min: 100
    x_max: 540
    y_min: 180
    y_max: 300
```

**`counter.line`** — línea virtual de cruce dentro del ROI. Cuando un track
cruza la línea, dispara un evento ingress o egress.

```yaml
counter:
  line:
    orientation: horizontal   # horizontal | vertical
    position: 240             # y-coord si horizontal, x-coord si vertical
```

Para ajustar ROI y línea **necesitás ver un frame de las cámaras**. Usar
`focus_assist.py` en modo preview:

```bash
cd /usr/src/people-counter
PYTHONPATH=. python3 scripts/focus_assist.py --grid --no-zoom
```

Abrir `http://people-counter.local:8080` en el navegador. Mirar dónde cae
la puerta en el frame y transcribir coordenadas al YAML. Ctrl+C para salir
del script cuando termines de medir.

**`operating_hours`** — horario del local. Vive en `cloud_defaults`:

```yaml
cloud_defaults:
  operating_hours:
    monday: "10:00-22:00"
    tuesday: "10:00-22:00"
    wednesday: "10:00-22:00"
    thursday: "10:00-22:00"
    friday: "10:00-22:00"
    saturday: "10:00-22:00"
    sunday: "10:00-21:00"
```

Formato: `"HH:MM-HH:MM"` (24h). Para día cerrado, usar `null`.

**`on_invalid_schedule`** — comportamiento si el horario está mal formateado.
Default `fail_open` (sigue contando). No cambiar en campo sin autorización
de ingeniería.

Validar el YAML antes de guardar:

```bash
python3 -c "import yaml; yaml.safe_load(open('/etc/people-counter/config.yaml'))"
```

Si hay error de sintaxis, nano no lo detecta. Corregirlo hasta que el
comando de arriba no imprima nada.

### 3.5. Iniciar servicios

```bash
sudo systemctl enable --now wifi-monitor
sudo systemctl enable --now people-counter
sudo systemctl enable --now people-counter-reset.timer
```

Confirmar que están activos:

```bash
systemctl is-active wifi-monitor people-counter people-counter-reset.timer
```

Las tres líneas deben responder `active`.

### 3.6. Tail de logs

```bash
sudo journalctl -u people-counter -f
```

En los primeros 30 segundos deberías ver:

- Log de arranque: `people-counter starting` o similar.
- Conexión MQTT: `connected to AWS IoT Core`.
- Primer frame capturado: `first stereo frame processed` o equivalente.
- `READY=1` enviado a systemd (notify).

Ctrl+C cuando lo veas estable. Si aparecen `ERROR` o `exception` repetidos,
ir a sección 5.

---

## 4. Verificación post-instalación (primeros 30 minutos)

### 4.1. Confirmar llegada de eventos a IoT Core

Desde la laptop, con las credenciales AWS del operador:

```bash
# Ver logs del Lambda de dedup (último minuto)
aws --profile people-counter-field logs tail /aws/lambda/people-counter-dedup \
  --since 1m --follow
```

Cada evento de conteo del dispositivo debería generar una invocación del
Lambda. Si ves invocations pero con errores de parsing, avisar a ingeniería.

Alternativa: pedir a ingeniería que te arme una query saved en Timestream
del estilo `SELECT * FROM counting WHERE device_id = '<id>' ORDER BY time
DESC LIMIT 20` y correrla desde la consola.

### 4.2. Caminata de test

Con el servicio corriendo y el tail de logs abierto en paralelo:

1. Pasar por la puerta **10 veces**: 5 ingresos + 5 egresos, alternando.
2. Esperar 5 segundos entre pasadas para que el tracker cierre cada track.
3. Anotar manualmente cuántos ingresos y cuántos egresos hiciste.
4. Abrir una segunda sesión SSH y mirar el tópico MQTT:

```bash
# En la segunda SSH, tail del log filtrando eventos de conteo
sudo journalctl -u people-counter -f | grep -E "ingress|egress"
```

Criterio de éxito del piloto día 1:

- 10/10 eventos publicados.
- Dirección correcta en ≥8/10 pasadas.
- Latencia <3s entre cruce físico y log.

Si sale <80%, revisar ROI y línea (sección 3.4) y repetir. Si sigue sin
andar, escalar.

### 4.3. Telemetría

Esperar 5 minutos y confirmar que llegan mensajes periódicos:

```bash
sudo journalctl -u people-counter --since '10 minutes ago' | grep telemetry
```

Cada mensaje debe incluir temp CPU, temp Hailo, RAM, disco, uptime. Rango
saludable:

- Temp CPU: 50–75 °C.
- Temp Hailo: 45–70 °C.
- RAM libre: >1 GB.
- Disco libre: >10 GB.

Si temp CPU >80 °C sostenida, el Active Cooler no está montado correctamente
o hay obstrucción de ventilación.

### 4.4. WiFi/BLE probes

El resumen se publica cada 15 min (`probe_interval_seconds: 900`). No vas a
tener datos confirmables hasta los 15–30 min de la visita.

Mientras tanto, confirmar que la captura está activa:

```bash
sudo journalctl -u wifi-monitor -f
```

Deberías ver probes entrando cada pocos segundos. Si el log está silencioso,
verificar que `wlan0mon` existe:

```bash
ip link show wlan0mon
```

A los 15 minutos, confirmar en logs del servicio principal:

```bash
sudo journalctl -u people-counter --since '20 minutes ago' | grep wifi_ble
```

Deberías ver un resumen con N hashes únicos.

---

## 5. Troubleshooting común

### 5.1. No conecta a MQTT (`iot_endpoint: FAIL`)

```bash
# 1. Certificados presentes
ls -l /etc/people-counter/certs/
# Deben existir: device.pem.crt, device.pem.key, AmazonRootCA1.pem

# 2. DNS resuelve
nslookup <endpoint-de-config.yaml>

# 3. Puerto 8883 alcanzable
nc -zv <endpoint> 8883
```

- Si `ls` falla: certs no instalados — escalar a ingeniería (no intentar
  copiar certs en campo).
- Si DNS falla: problema de red del local. Pedir a IT que el puerto tenga
  salida a internet TCP/8883 y DNS/53.
- Si `nc` falla pero DNS anda: firewall del local está bloqueando 8883.

### 5.2. Cámaras no detectadas (`cameras: FAIL — expected 2, found N`)

```bash
rpicam-still --list-cameras
```

Debe listar 2 cámaras `imx708`.

- Si lista 0 o 1: cables CSI sueltos. Desenergizar, revisar que el clip del
  conector CSI esté abajo sobre el flat en ambos extremos.
- Si lista otras cámaras: overlays mal. Verificar
  `/boot/firmware/config.txt`:

```bash
grep imx708 /boot/firmware/config.txt
```

Debe mostrar dos líneas `dtoverlay=imx708,cam0` y `dtoverlay=imx708,cam1`.
Si no están, escalar (no tocar config.txt en campo).

### 5.3. Cero detecciones o detecciones erróneas

Síntomas: personas pasan pero no se disparan eventos.

1. **Altura**: medir con cinta. Si >2.8m, las personas quedan chicas.
   Bajarla si el techo permite.
2. **HEF presente**:

   ```bash
   ls -l /usr/src/people-counter/models/yolov8n.hef
   ```

   Debe existir y pesar >5 MB. Si no, escalar.
3. **Iluminación**: si el local está muy oscuro (<100 lux), la YOLO pierde
   recall. Pedir al encargado subir iluminación o reubicar.
4. **Frame de preview**: correr `focus_assist.py` y mirar si las personas
   aparecen en el frame y en el ROI configurado. Muchas veces el ROI está
   desplazado respecto a la puerta real.

### 5.4. Conteo manual no coincide

Anotar para el reporte a ingeniería:

- Hora exacta del test.
- Condiciones de luz.
- Flujo peatonal aproximado (personas/minuto).
- Tipo de ropa (invierno con abrigos oscuros confunde más).
- Grupos vs. personas solas (grupos muy pegados pueden contarse como 1).
- Screenshot del preview:

```bash
PYTHONPATH=. python3 scripts/focus_assist.py --grid --no-zoom &
sleep 5
scp pi@people-counter.local:/tmp/focus_left.jpg .
kill %1
```

Comparar el ROI configurado con la ubicación real de la puerta en el frame.

### 5.5. El servicio reinicia solo

```bash
sudo journalctl -u people-counter --since '1 hour ago' | grep -Ei "error|exception|killed|oom"
```

- `Killed` o `OOM`: el dispositivo está quedándose sin RAM. Puede ser que
  el preview de `focus_assist.py` quedó abierto en paralelo. Matarlo y
  volver a probar.
- `TimeoutError` en Hailo: driver PCIe tiene algún problema. Reboot del
  dispositivo (`sudo reboot`) y volver a verificar. Si persiste, escalar.
- `ConnectionRefusedError` a MQTT: ver 5.1.

### 5.6. Conteo se detiene fuera de horario

Esperado. El dispositivo respeta `operating_hours`.

```bash
date
grep -A 10 "operating_hours" /etc/people-counter/config.yaml
```

- Si el reloj del dispositivo está mal (RTC sin pila o drift), el horario
  evalúa mal. `sudo timedatectl status` debe mostrar NTP sincronizado.
- Si el horario del local cambió, actualizar `operating_hours` en el YAML
  y `sudo systemctl restart people-counter`.
- Si estás en horario y no cuenta, verificar que `counting_enabled: true`
  en `cloud_defaults`.

### 5.7. `wlan0mon` no existe (`wifi_monitor: FAIL`)

```bash
sudo systemctl status wifi-monitor
```

Si el servicio falló, ver logs:

```bash
sudo journalctl -u wifi-monitor --since '10 minutes ago'
```

Causa común: firmware nexmon no cargado después de un apt upgrade. Reboot
del dispositivo suele resolverlo. Si no, escalar.

---

## 6. Escalación a ingeniería

### 6.1. Cuándo escalar

- Hardware visiblemente dañado al desempacar (lente rayado, cable CSI
  cortado, HAT desprendido).
- Cualquier FAIL del preflight que no está cubierto en la sección 5.
- El preflight pasa pero el servicio no levanta (reboot no resuelve).
- Precisión del conteo manual <85% después de ajustar ROI y repetir test
  en 24h.
- Temperaturas fuera de rango sostenidas (CPU >85 °C, Hailo >80 °C).
- Desconexiones MQTT frecuentes (>3 por hora) con buena conectividad de
  red.

### 6.2. Qué enviar

Armar un paquete con:

```bash
# En el dispositivo
sudo journalctl -u people-counter --since '2 hours ago' > /tmp/service.log
sudo journalctl -u wifi-monitor --since '2 hours ago' > /tmp/wifi-monitor.log
sudo PYTHONPATH=. python3 /usr/src/people-counter/scripts/preflight.py --json \
  > /tmp/preflight.json
cp /etc/people-counter/config.yaml /tmp/config-sanitized.yaml
# Asegurarse de NO incluir /etc/people-counter/certs/*
```

Desde la laptop:

```bash
scp pi@people-counter.local:/tmp/service.log .
scp pi@people-counter.local:/tmp/wifi-monitor.log .
scp pi@people-counter.local:/tmp/preflight.json .
scp pi@people-counter.local:/tmp/config-sanitized.yaml .
```

Adjuntar al ticket/mail:

1. Los 4 archivos de arriba.
2. **Foto del montaje**: cámaras + ambiente (puerta, iluminación general).
3. **Foto del frame** si aplica (`/tmp/focus_left.jpg` y `focus_right.jpg`).
4. **Descripción del problema**, en este formato:
   - **Cuándo empezó** (hora, fecha).
   - **Qué estaba pasando** cuando apareció (test de caminata,
     instalación inicial, llovió, etc.).
   - **Qué cambió** desde la última vez que anduvo (si aplica).
   - **Qué pasos de troubleshooting ya probaste** (sección 5).
   - **Resultado de cada paso**.
5. `device.id` del dispositivo y `store_id` del local.

### 6.3. Qué NO incluir nunca

- Contenido de `/etc/people-counter/certs/`. Los certificados son privados
  del dispositivo. Si ingeniería necesita renovarlos, se hace por IoT Core
  y redeploy, no por mail.
- Credenciales AWS (access keys, session tokens).
- Capturas de video o imágenes de personas identificables. El frame de
  preview de `focus_assist.py` es aceptable si es de un local vacío.

---

## Apéndice A: Comandos rápidos

```bash
# Estado general
systemctl is-active wifi-monitor people-counter people-counter-reset.timer

# Preflight
cd /usr/src/people-counter && sudo PYTHONPATH=. python3 scripts/preflight.py

# Logs en vivo
sudo journalctl -u people-counter -f

# Preview de cámaras (puerto 8080)
cd /usr/src/people-counter && PYTHONPATH=. python3 scripts/focus_assist.py --grid --no-zoom

# Reiniciar servicio tras editar config
sudo systemctl restart people-counter

# Temperaturas
vcgencmd measure_temp
hailortcli fw-control identify | grep -i temperature

# Verificar cámaras
rpicam-still --list-cameras

# Verificar monitor mode
ip link show wlan0mon
```

## Apéndice B: Referencias cruzadas

- Setup inicial del dispositivo (flasheo, instalación de paquetes,
  calibración): [`setup-guide.md`](setup-guide.md).
- Esquema de `config.yaml`: [`../config/config.example.yaml`](../config/config.example.yaml).
- Código del preflight: [`../scripts/preflight.py`](../scripts/preflight.py).
- Proyecto y convenciones: [`../CLAUDE.md`](../CLAUDE.md).

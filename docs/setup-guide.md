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
   - Cámara izquierda → CAM0
   - Cámara derecha → CAM1
   - Orientar ambas igual (el conector flat tiene un lado con contactos expuestos)
4. **Disipador** → montar sobre el SoC de la Pi
5. **MicroSD** → insertar la tarjeta ya flasheada
6. **Batería RTC** → conectar al conector de la Pi (mantiene la hora sin red)
7. **NO conectar PoE todavía** — para el PoC usá la fuente USB-C estándar

## 3. Primer boot

1. Conectar monitor HDMI + teclado, o acceder por SSH
2. Esperar que termine el primer boot (puede tardar 2-3 min)
3. Verificar que arranca:

```bash
# Verificar sistema
uname -a
# Debe decir aarch64

# Verificar cámaras
libcamera-hello --list-cameras
# Debe listar 2 cámaras

# Verificar espacio
df -h
```

## 4. Setup del entorno

Conectar por SSH desde tu PC y correr:

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y \
  python3-pip python3-venv \
  python3-opencv python3-numpy \
  libopencv-dev \
  git

# Instalar Hailo SDK (seguir guía oficial de Hailo para RPi5)
# https://github.com/hailo-ai/hailo-rpi5-examples
# Esto instala hailo_platform y los drivers

# Verificar Hailo
hailortcli fw-control identify
# Debe mostrar: Hailo-8L, firmware version, etc.

# Clonar el repo
git clone https://github.com/maurogasparri/people-counter.git
cd people-counter

# Crear virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e ".[dev]"

# Verificar tests
pytest -v
```

## 5. Test rápido de cámaras

```bash
# Capturar una imagen de cada cámara
libcamera-still -o test_cam0.jpg --camera 0
libcamera-still -o test_cam1.jpg --camera 1

# Verificar que ambas imágenes existen y se ven bien
ls -la test_cam*.jpg
```

## 6. Test rápido del Hailo

```bash
# Si instalaste hailo-rpi5-examples:
cd ~/hailo-rpi5-examples
python basic_pipelines/detection.py --input rpi
# Debe mostrar detección en vivo desde una cámara
```

## 7. PoC — Siguiente paso

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

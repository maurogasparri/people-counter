# Dependencias del sistema

Este documento registra **lo que efectivamente se instaló y validó**, para que el
entorno pueda reconstruirse. Se distingue en todo momento entre dos cosas que se
confunden con facilidad:

- **Restricciones de instalación** — los rangos de `pyproject.toml`. Definen qué
  versiones son admisibles. Un rango no reconstruye un entorno: describe una
  familia de entornos posibles.
- **Registro de lo validado** — las versiones exactas de este documento y de
  `requirements-lock.txt`. Son las que estaban corriendo cuando se tomaron las
  mediciones que el trabajo reporta.

Los rangos no se estrechan a las versiones observadas: son deliberados y su
justificación vive en `pyproject.toml`.

---

## 1. Dispositivo

Versiones **observadas** sobre la unidad del prototipo, tomadas el **2026-08-06**
con `pip freeze`, `dpkg -l` y `vcgencmd`. No son valores declarados ni inferidos.

### 1.1. Plataforma

| Componente | Versión observada |
|---|---|
| Equipo | Raspberry Pi 5 Model B Rev 1.0 |
| Sistema operativo | Debian GNU/Linux 13 (trixie) |
| Núcleo | 6.12.75+rpt-rpi-2712 · aarch64 |
| Firmware | Broadcom `9cd61c53`, build 2026-04-14 |
| Python | 3.13.5 · pip 25.1.1 |

### 1.2. Acelerador neuronal

Nada de esto llega por `pip`: son paquetes del sistema y un módulo del núcleo.

| Componente | Versión observada |
|---|---|
| `hailort` | 4.23.0 |
| `hailort-pcie-driver` | 4.23.0 |
| `python3-hailort` | 4.23.0-1 |
| `hailo_platform` (módulo Python) | 4.23.0 |

### 1.3. Pila de cámara

| Componente | Versión observada |
|---|---|
| `libcamera0.7` / `libcamera-ipa` / `libcamera-tools` | 0.7.0+rpt20260205-1 |
| `python3-libcamera` | 0.7.0+rpt20260205-1 |
| `python3-picamera2` | 0.3.34-1 |
| `rpicam-apps` | 1.11.1-1 |

### 1.4. Paquetes de Python del proyecto

Extracto de los que el código importa. El congelado completo —324 paquetes del
entorno de sistema— está en [`../requirements-lock.txt`](../requirements-lock.txt).

| Paquete | Observado | Rango declarado en `pyproject.toml` |
|---|---|---|
| `numpy` | 2.2.4 | `>=2.0` |
| `opencv-contrib-python` | 4.13.0.92 | `>=4.10,<5` |
| `scipy` | 1.15.3 | `>=1.13` |
| `paho-mqtt` | 2.1.0 | `>=2.1` |
| `PyYAML` | 6.0.2 | `>=6.0` |
| `scapy` | 2.7.0 | `>=2.6` |
| `bleak` | 3.0.1 | `>=0.22` |
| `picamera2` | 0.3.34 | — (paquete del sistema) |
| `pyserial` | 3.5 | — |
| `gpiozero` | 2.0.1 | — |
| `lgpio` | 0.2.2.0 | — |
| `simplejpeg` | 1.8.1 | — |
| `av` | 14.2.0 | — |

> La cota superior de OpenCV es deliberada: la versión 5.0 removió
> `cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC`, que usa el solve de calibración
> estéreo. Sin la cota, una instalación limpia resuelve a 5.x y la calibración
> falla en tiempo de ejecución.

---

## 2. Nube

### 2.1. Funciones

| Función | Entorno de ejecución | Memoria | Arquitectura |
|---|---|---|---|
| `people-counter-persist-event-dev` | python3.13 | 256 MB | x86_64 |
| `people-counter-ingest-pos-dev` | python3.13 | 256 MB | x86_64 |
| `people-counter-query-aggregates-dev` | python3.13 | 512 MB | x86_64 |

Sin capas. La plantilla de infraestructura declara `Runtime: python3.13` para las
tres.

### 2.2. Base de datos administrada

| | Versión |
|---|---|
| Declarada en la plantilla (`EngineVersion`) | PostgreSQL **16.6** |
| Corriendo efectivamente | PostgreSQL **16.13** |

**Las dos versiones difieren, y es comportamiento esperado.** La instancia tiene
habilitada la actualización automática de versiones menores
(`AutoMinorVersionUpgrade`), de modo que el proveedor aplica parches dentro de la
serie 16.x sin intervención. La plantilla fija el punto de partida; el servicio
administrado avanza desde ahí. Se declaran ambas porque reconstruir el entorno
desde la plantilla produce 16.6, mientras que las mediciones del trabajo se
tomaron sobre 16.13.

Clase de instancia: `db.t4g.micro`.

### 2.3. Contenedores

El servicio de visualización corre en ECS Fargate, plataforma `LATEST`, con la
revisión 5 de su definición de tarea.

| Contenedor | Etiqueta | Digest efectivo |
|---|---|---|
| Grafana (imagen propia en ECR) | `people-counter/grafana-dev:latest` | `sha256:135f4b96b7a54f904415dc51e71a1cd945b1f8400772b2feac2df5112b43f4ed` |
| Renderizador de imágenes | `grafana/grafana-image-renderer:latest` | `sha256:a30a68c2de11a1aad5733452536ac50fbc2f3958e6d0aa046ef9eb56db7c6a6d` |

**La etiqueta `latest` es móvil y no reconstruye nada**: mañana puede apuntar a
otra imagen. Lo reproducible es el digest. Ambos digests fueron leídos el
2026-08-06 de la tarea en ejecución, y el de Grafana coincide con el que la
etiqueta apunta hoy en el registro —imagen subida el 2026-05-20, 348 MB—.

Para reconstruir el entorno validado hay que referenciar el digest, no la
etiqueta.

---

## 3. Cómo reconstruir

En una máquina de desarrollo, para correr las pruebas alcanza con las
restricciones declaradas:

```bash
pip install -e ".[dev]"
```

Para reproducir el entorno **validado** del dispositivo, con las versiones
exactas:

```bash
pip install -r requirements-lock.txt
```

El congelado corresponde a un entorno de sistema completo sobre Debian 13, no a
un entorno virtual, e incluye paquetes del sistema operativo ajenos al proyecto.
Se excluyó una única línea: la instalación editable del propio repositorio, que
no es una dependencia.

Los componentes de las secciones 1.2 y 1.3 —acelerador y pila de cámara— **no se
instalan por `pip`**. El procedimiento completo está en
[`setup_guide.md`](setup_guide.md).

# Guía de foco + calibración en laboratorio

Esta guía cubre el proceso de **seteo de foco** y **calibración estéreo** de un
dispositivo (par de cámaras Arducam IMX708 + bracket rígido). Se hace una única
vez en laboratorio, idéntico para todos los dispositivos de la flota — los
parámetros por sitio (`mounting_height_m`, etc.) se definen después en
`config.yaml` al momento del deploy.

## Prerequisitos

### Hardware

| Item | Spec |
|------|------|
| Par de cámaras | 2× Arducam IMX708 B0310 (120° HFOV / 152° diag, M12 fisheye) |
| Bracket | Rígido, baseline 140mm, L/R fijas. **No desarmar entre calibración y deploy** |
| Trípode de cámara | Rango ~[70cm, 210cm], cabeza con 1/4" |
| Trípode del board | Rango ~[70cm, 210cm], cabeza con 1/4" |
| Board ChArUco | A3 landscape, 9×6 cuadrados, checker 45mm, marker 33mm, DICT_4X4_100 (PDF: `calibration/calib.io_charuco_420x297_6x9_45_33_DICT_4X4.pdf`) |
| Montaje del board | Sustrato rígido (3mm PVC o equivalente) + rosca 1/4" centrada |
| Loctite | 243 azul (threadlocker medio) para el lens post-foco |

### Espacio

- Mínimo **4m × 3m despejado** con buena iluminación uniforme (sin fuentes pulsátiles tipo LED barato a 100Hz).
- Ideal: showroom o similar, con fondo texturado (paredes con cuadros, percheros con ropa). Evitar paredes lisas blancas frente a la cámara — afectan el check de uniformidad de foco.

### Software

- Dispositivo RPi5 encendido, con el repo clonado y dependencias instaladas.
- Laptop o tablet con browser (Chrome/Firefox) en la misma red que el RPi, para la UI web del wizard.

## Paso 1 — Setup físico

### Alturas del trípode

Las alturas son del **stud 1/4" del trípode** (base de la rosca), no del punto óptico del objeto montado. Hay que sumar los offsets del mount:

- **Offset del bracket de cámara**: distancia desde el 1/4" al eje óptico del lente. Con el bracket de referencia, son **~40mm**. Medir el tuyo con regla.
- **Offset del soporte del board**: distancia desde el 1/4" al centro óptico del área impresa. Con el soporte de referencia (rosca al borde inferior), son **~148mm** (medio alto del A3 = 297/2). Si pusiste la rosca centrada en el dorso, son casi 0.

Ajustes si tus offsets difieren: si el offset de board es mayor a 148mm, el thread del board queda más bajo — podés colapsar contra el mínimo del trípode en D4/D5; compensá bajando far con `--dist-far-mm 2800` (ver Paso 3).

### Pasos

1. **Cámara en trípode, stud 1/4" a 1.50m del piso**, apuntando horizontal hacia el lado despejado. Con offset +40mm, el eje óptico queda a ~1.54m.
2. **Trípode del board cerca, con el stud 1/4" a 1.39m inicialmente** (corresponde al centro del board a 1.54m, matching la altura óptica de la cámara, para la primera pose frontal). El trípode del board se moverá durante el proceso.
3. **Nivelar el bracket** con burbuja. No es crítico para la matemática de calibración (los intrínsecos son invariantes a la orientación) pero facilita la lectura de los ghost overlays.
4. **Marcar con cinta en el piso** tres líneas perpendiculares a la cámara, a 1.0m, 2.0m y 3.0m medidas con cinta métrica.
5. **Limpiar los lens** con trapo de microfibra. Una huella digital baja el contraste y puede invalidar el check de nitidez.

### Alturas esperadas del thread del board a lo largo de las 20 poses

Con cámara a 1.50m y far=3.0m:

| Grupo | Pose | Thread del board (tripod) |
|---|---|---|
| A, D1 | Centro frontal / centro far | 1.39m |
| B1/B2 | Top mid | 1.96m |
| B3/B4 | Bottom mid | 0.82m |
| C3 | Top-center mid | 1.91m |
| C4 | Bottom-center mid | 0.87m |
| D2/D3 | Top far | **2.05m** (5cm margen al tope del trípode) |
| D4/D5 | Bottom far | **0.73m** (3cm margen al mínimo) |
| E | Centro near, tilts extremos | 1.39m |

D2/D3 y D4/D5 son los más apretados. Si tu trípode no los alcanza con margen de maniobra, bajar a far=2.8m (excursión ±62cm en lugar de ±66cm).

## Paso 2 — Foco

Objetivo: enfocar ambos lens a una distancia tal que el DoF cubra todo el rango operativo (1.15m a 3.30m). Focando a ~2.0m, el DoF va de ~1.2m a infinito con el M12 120° a f/~2.

### Correr el asistente

```bash
python scripts/focus_assist.py
```

Defaults aplicados: target range 1.80–2.20m (lab protocol), board definitivo,
compact-scene auto-detect.

1. Abrir `http://<rpi-hostname>:8080` en el browser.
2. Click "Comenzar" — desbloquea AudioContext y comienza el preview.
3. Posicionar el board a ~2.0m de la cámara. El status indica "cerca" o "lejos" si está fuera del target range.
4. Ajustar el lens izquierdo (rotándolo en su montaje M12) hasta que las barras de nitidez central y corners pasen (verde).
5. Repetir para el lens derecho. La barra de simetría L/R debe quedar por debajo del umbral.
6. Cuando ambos lens están en verde y el banner dice "LISTO", click "Finalizar".
7. El reporte HTML se abre automáticamente en pestaña nueva. Guardarlo.

### Fijar el foco con Loctite

**Solo después de que focus_assist dé PASS**:

1. Marcar con fibrita la posición del lens respecto al holder (referencia por si se mueve).
2. Aflojar el lens ~¼ de vuelta.
3. Aplicar una gota pequeña de Loctite 243 azul en la rosca expuesta. **Nunca cerca del front element**.
4. Volver a la marca.
5. Verificar en focus_assist que sigue en PASS (por si rotaste de más).
6. Dejar curar **24h antes de mover el dispositivo**.

## Paso 3 — Calibración estéreo

Objetivo: obtener los intrínsecos (K, D por cámara) y extrínsecos (R, T entre L/R) usando el modelo **fisheye Kannala-Brandt** (`cv2.fisheye.*`).

### Correr el wizard

```bash
python scripts/calibrate.py wizard --device-id DEV-XXX
```

Reemplazar `DEV-XXX` con el identificador del dispositivo (va al reporte).

Defaults aplicados: far=3.0m (cabe en tripod 70–210cm), 20 poses canónicas, tolerance "normal", pose-timeout 180s (tripod-friendly).

### Las 20 poses

El wizard guía al operador a través de 20 poses, agrupadas por zona del frame y nivel de inclinación. Las alturas del stud del tripod del board (para los offsets de mount de referencia) están en la tabla del Paso 1; acá está solo la distribución lógica:

- **A1-A4** (center near, 1.0m): frontal + pitch + yaw + mix.
- **B1-B4** (corners mid, 2.0m): con yaw/pitch.
- **C1-C4** (mid, roll y pitch extra): roll ±20°, top/bottom center.
- **D1-D5** (far, 3.0m): centro + 4 esquinas con diagonal tilt.
- **E1-E3** (extreme tilts at near): yaw ±25°, combinación 3-axis.

Para cada pose:

1. El wizard muestra el ghost overlay (silueta translúcida del board en la posición target) y dice por TTS "Pose N. [label]. A [X]cm de la cámara".
2. El operador **mueve el trípode del board** para que coincida con el ghost: distancia, posición lateral, altura. Ajusta tilts de la cabeza del trípode (pitch/yaw/roll) según indique el ghost.
3. El wizard detecta alineación + estabilidad (10 frames consecutivos con < 1.5px de jitter) → **captura automática**. Feedback audio confirma el shot.
4. Pasa a la siguiente pose. Si una pose tarda >180s, se auto-saltea.

### Checkpoints del wizard

- **Después de pose A1**: el wizard hace bootstrap de intrínsecos desde la primera pose centrada, y cambia la tolerancia de alineación de "loose" (25px) a "tight" (12px) para las restantes.
- **Después de 15 poses**: ya puede calibrar. Si alguna pose se salteó, se puede finalizar con menos de 20 — el reporte marca coverage gaps.
- **Al finalizar las 20**: corre `fisheye.calibrate` (por cámara) → deriva R, T del transform per-pose promediado → genera mapas de rectificación → valida residuales per-pair → abre reporte HTML.

### Ground-truth check

El wizard pide un último paso: **colocar el board frontal a una distancia conocida medida con cinta** (ej. 2000mm), y confirmar esa distancia por UI. El sistema computa la depth del centro del board usando el pipeline SGBM + calibración recién obtenida, y la compara contra la distancia real. Umbrales:

- **< 5% error @ 2m** → PASS
- **< 10% error @ 3m** → PASS
- **edge/center ratio < 2×** → PASS (chequea 5 zonas)

Si falla cualquiera, el reporte lo marca en rojo y sugiere causas (board poco diverso, bracket no rígido, foco perdido).

### Qué se guarda

En `--output-dir` (default `./calibration/captures`):

- `left_NNN_<pose_id>.png` / `right_NNN_<pose_id>.png` — frames crudos full-res (**no descartar**, sirven para re-procesar con otro modelo si hace falta).
- `session.json` — metadata: poses capturadas, skipped, ordinal mapping.
- `report.html` + `report/` — reporte auto-abierto en browser, con thumbnails, residuales, ground-truth, warnings.

En `--output` (default `./calibration.npz`):

- `camera_matrix_l`, `dist_coeffs_l`, `camera_matrix_r`, `dist_coeffs_r` — intrínsecos por cámara
- `R`, `T` — extrínsecos del par (R 3×3, T 3×1 en mm)
- `R1`, `R2`, `P1`, `P2`, `Q` — matrices de rectificación
- `map_l_x`, `map_l_y`, `map_r_x`, `map_r_y` — mapas precomputados para `cv2.remap`
- `image_size` — (w, h) de captura

## Paso 4 — Fin del ciclo

1. **No desarmar el bracket L/R**. La calibración viaja con el par físico, no con el dispositivo RPi. Si separás los lens, la extrínseca cambia y la calibración se invalida.
2. **Etiquetar el bracket** con el device-id del reporte.
3. **Guardar el `.npz`** en la estructura del provisioning (ver `scripts/provision.py`) asociado al device-id.
4. **Archivar las PNGs crudas** (no eliminarlas del RPi). Son oro si después hay que re-calibrar con otro modelo o validar con otra herramienta.

## Troubleshooting

### El ghost overlay no coincide con el board detectado

- **Bootstrap no terminó**: las primeras 4 poses usan K nominal (pinhole aproximado). Después del bootstrap, usa K fitted. Si el offset persiste después de las primeras 4-5 capturas, el bracket puede estar flexionado o la cámara mal centrada.

### RMS estéreo alto (> 0.5 px)

- **Board alabeado**: chequear planaridad (apoyar sobre vidrio, mirar contraluz).
- **Motion blur**: trípode inestable o cabeza con juego. Apretar locks, contrapeso en el trípode.
- **Poses degeneradas**: mirar `per_pair_residuals` en el reporte — si una pose tiene RMS muy por encima del resto, fue un bump. Re-correr solo esa con `--resume`.

### Baseline estimada lejos de 140mm

- **Capturas poco diversas**: RMS bajo pero baseline off. Solución: re-correr con más diversidad (asegurarse que las 20 poses se completen, no se salteen más de 2-3).

### Ground-truth falla solo en esquinas (edge/center > 2×)

- **Board no cubrió los bordes lo suficiente** durante captura. El solver fisheye necesita corner data. Re-correr con énfasis en B y D groups.

### Trípode no llega a 0.73m (D4/D5) o pasa 2.05m (D2/D3)

- Chequear que la cabeza esté al mínimo (o al máximo con el center column) y las patas bien abiertas.
- Plan A: pasar `--dist-far-mm 2800` al wizard — excursión vertical baja a ±62cm, el stud del board queda entre 0.77m y 2.01m (4-9cm de margen a cada extremo).
- Plan B: sacar el board del trípode, hand-hold solo las 2-4 poses problemáticas. El resto sigue en trípode.

## Referencias

- Modelo fisheye: `src/vision/calibration.py`, función `calibrate_stereo`
- Pose sequence: `src/vision/calibration.py`, función `default_pose_sequence`
- Wizard end-to-end: `scripts/calibrate.py`, función `cmd_wizard`
- Focus assistant: `scripts/focus_assist.py`
- Config runtime (para deploy): `config/config.example.yaml`, campo `mounting_height_m`

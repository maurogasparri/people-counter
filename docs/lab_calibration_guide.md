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
| Board ChArUco | A3 landscape, 9×6 cuadrados, checker 45mm, marker 33mm, DICT_4X4 legacy pattern (PDF: `calibration/calib.io_charuco_420x297_6x9_45_33_DICT_4X4.pdf`). Los scripts usan `--legacy-pattern` por default para matchear la enumeración de calib.io. |
| Montaje del board | Sustrato rígido (3mm PVC o equivalente) + rosca 1/4" (idealmente centrada; si está al borde inferior del board, contar 148mm de offset del thread al centro óptico) |
| Fijador del lens | **Esmalte de uñas transparente** aplicado al seam entre el barrel y el holder M12. Cura al aire en ~5-10 min (touch dry) y ~30-60 min (full). Mecánicamente débil comparado con un anaeróbico industrial, pero más que suficiente para fijar un lens M12 que no va a ser tocado más después del foco. Sin solvente fuerte = no migra a la óptica. |
| Llave de barrel | Llave diseñada para encastrar en el barrel del lens M12, permite girar el lens con dedos sobre un mango más grande — facilita ajustes finos durante el foco. Queda puesta durante calib y se retira después del curado total. |

### Espacio

- Mínimo **4m × 3m despejado** con buena iluminación uniforme (sin fuentes pulsátiles tipo LED barato a 100Hz).
- Ideal: showroom o similar, con fondo texturado (paredes con cuadros, percheros con ropa). Evitar paredes lisas blancas frente a la cámara — afectan el check de uniformidad de foco.

#### Fondo texturado: por qué importa

El check de **corner sharpness** del `focus_assist` mide varianza Laplaciana en las 4 esquinas del frame — necesita textura visible ahí para devolver un número representativo. Pared blanca = varianza ~0 = el check no puede distinguir "lente blando en los bordes" de "no hay contenido para medir".

La **validación ground-truth** del wizard al final muestrea profundidad en 5 zonas (centro + 4 esquinas). SGBM necesita textura para matchear L↔R; pared lisa devuelve fill-rate bajo (<50%) y std alto, y el reporte termina con números que parecen una calibración mala cuando en realidad es la escena la que no tiene info.

**Opciones de fondo, en orden de preferencia**:

1. **Jardín vertical / pared con plantas densas** a 3.5-4m del lente. Follaje cubre las 4 esquinas con textura de alta frecuencia. Caveat: hojas estáticas (AC off, sin viento) — si se mueven entre frames L y R (60-120ms desync) el SGBM mete ruido.
2. **Biblioteca con libros**, ladrillo visto, afiche con detalle denso, percheros con ropa. Cualquier superficie con detalle visible cubriendo el FOV completo.
3. **Pared lisa**: aceptable solo si **agregás objetos texturados en las 4 esquinas del frame** — caja con etiquetas en el piso, cuadros en la pared, plantas en macetas. Lo que importa es que cada esquina del frame de la cámara tenga textura, no que toda la pared sea uniforme.

Sin esto, el reporte va a tirar `FAIL` o `WARN` en uniformidad de corners y/o ground-truth aunque el lente esté bien enfocado.

### Software

- Dispositivo RPi5 encendido, con el repo clonado y dependencias instaladas.
- Laptop o tablet con browser (Chrome/Firefox) en la misma red que el RPi, para la UI web del wizard.

## Verificación de salud de calibración (post-calibración)

> El QC de bracket *pre*-calibración se retiró: medir la geometría de un
> fisheye de 120° con un modelo pinhole sin coeficientes de distorsión
> (que recién salen de la calibración) produce yaw/offsets fantasma. El
> case 3D-impreso fija la baseline a 140mm por construcción; la geometría
> real del bracket la valida el **reporte de calibración** (RMS estéreo +
> métricas de alineación computadas sobre los extrínsecos reales).

Una vez calibrado, verificá la salud con `diagnose_calibration.py`:
rectifica un board ChArUco con la calibración guardada y mide el error
epipolar residual L↔R (sub-píxel = sano; >1px = recalibrar). No necesita
ground-truth. Sirve como check de campo / periódico para detectar drift
(bracket movido, lente corrido, térmico):

```bash
PYTHONPATH=. python3 scripts/diagnose_calibration.py
```

## Paso 1 — Setup físico

### Alturas del trípode

Las alturas son del **stud 1/4" del trípode** (base de la rosca), no del punto óptico del objeto montado. Hay que sumar los offsets del mount:

- **Offset del bracket de cámara**: distancia desde el 1/4" al eje óptico del lente. Con el bracket de referencia, son **~40mm**. Medir el tuyo con regla.
- **Offset del soporte del board**: con la rosca centrada en el dorso del A3 (convención del soporte de referencia), el centro óptico del board coincide con el stud → offset **~0mm**. Si pusiste la rosca en otra posición, medirla.

### Pasos

1. **Cámara en trípode, stud 1/4" a 1.36m del piso**, apuntando horizontal hacia el lado despejado. Con offset +40mm, el eje óptico queda a 1.40m.
2. **Trípode del board cerca, con el stud 1/4" a 1.40m inicialmente** (con rosca centrada, el centro del board coincide con el stud → matches la altura óptica de la cámara para pose frontal). El trípode del board se moverá durante el proceso.
3. **Nivelar el bracket** con burbuja. No es crítico para la matemática de calibración (los intrínsecos son invariantes a la orientación) pero facilita la lectura de los ghost overlays.
4. **Marcar con cinta en el piso** tres líneas perpendiculares a la cámara, a 1.0m, 2.0m y 3.0m medidas con cinta métrica.
5. **Limpiar los lens** con trapo de microfibra. Una huella digital baja el contraste y puede invalidar el check de nitidez.

### Alturas esperadas del thread del board a lo largo de las 20 poses

Con cámara stud a 1.36m (eje óptico 1.40m), board con rosca centrada (offset 0mm), y far=3.0m:

| Grupo | Pose | Thread del board (tripod) |
|---|---|---|
| A, D1 | Centro frontal / centro far | 1.40m |
| B1/B2 | Top mid | 1.97m |
| B3/B4 | Bottom mid | 0.83m |
| C3 | Top-center mid | 1.92m |
| C4 | Bottom-center mid | 0.88m |
| D2/D3 | Top far | **2.06m** (4cm margen al tope) |
| D4/D5 | Bottom far | **0.74m** (4cm margen al piso) |
| E | Centro near, tilts extremos | 1.40m |

Todas las poses entran con 4cm de margen simétrico en los extremos. Si necesitás más holgura (trípode con stops imprecisos cerca de los límites), bajar a `--dist-far-mm 2800` para ganar ~4cm extra a cada extremo.

## Paso 2 — Foco + Lens locking con esmalte

Objetivo: enfocar ambos lens a una distancia tal que el DoF cubra todo el rango operativo del bbox de detección (cabeza + pie) del dispositivo.

**Target universal: foco a 1.5m ±20cm**. Sirve para todo el rango de mount 2.0–3.5m sin re-foco per-device.

Cálculo del DoF (M12 120° a f/2.0, IMX708 binned 2.8μm pitch, CoC=4.2μm):

| Caso | Mount | Cabeza (depth) | Pie/piso (depth) | Bbox depth range |
|------|-------|----------------|-------------------|-------------------|
| Adulto alto, mount mín | 2.0m | 1.0m (en counting line) | 2.0m | 1.0–2.0m |
| Niño chico, mount máx | 3.5m | 2.5m | 3.5m | 2.5–3.5m |
| **Unión flota** | 2.0–3.5m | 1.0–2.5m | 2.0–3.5m | **1.0–3.5m** |

Con foco a 1.5m: DoF = 0.59m a ∞. El blur en los extremos del rango operativo (1.0m y 3.5m) queda en ~1.4–1.6μm, bien por debajo del CoC threshold de 4.2μm. Foco simétricamente balanceado entre cabeza y pie.

**Por qué 1.5m y no 2.0m**: foco a 2.0m peakea la sharpness en el piso del mount máximo (3.5m), donde SGBM ya tiene textura de sobra. Penaliza la cabeza a 1.0m (extremo near del rango), donde el detector necesita más sharpness para preservar recall sobre adultos altos en mounts bajos.

El holder M12 del Arducam B0310 **no tiene set screw**, así que el lens se fija con **esmalte de uñas transparente** aplicado al seam entre el barrel y el holder una vez logrado el foco. Cura al aire (no necesita activador). Mecánicamente es débil comparado con un anaeróbico industrial (Trabasil AM3 o similar), pero suficiente para un lens M12 que no va a ser tocado más después del foco y que opera sin vibración constante en una cámara fija de techo.

El barrel del lens M12 se gira con una **llave dedicada** que encastra en sus ranuras y da más palanca que los dedos sobre el barrel pelado. La llave se usa durante el foco y se retira antes de aplicar el esmalte.

> **Para producción / flota / vibración alta**: evaluar un threadlocker
> anaeróbico industrial (Trabasil AM3 + activador anaeróbico, o
> equivalente Loctite) para torque de quiebre más alto y cure químico
> con PTFE filling de holguras. Esmalte fue suficiente para el PoC
> validado.

### Paso 2A — Hacer foco

Primero foco, después se fija. El esmalte se aplica al final, una vez confirmado que ambos lens quedaron en el target range.

```bash
sudo PYTHONPATH=. python3 scripts/focus_assist.py
```

Defaults aplicados: target range 1.30–1.70m (lab protocol universal), board definitivo, compact-scene auto-detect. **Default: modo MAPA** — en vez de un frame estático, paseás el board por todo el cuadro y se acumula la nitidez máxima por zona (grilla 3×3); al cubrir las 9 zonas (L y R) evalúa el mapa completo, así el check por zona / simetría tiene board real en cada celda (no fondo). `--static` vuelve al modo de un solo frame. Los pasos de abajo aplican a ambos — girás los rings mirando las barras del frame actual; en modo mapa además barrés el board para cubrir todas las zonas.

1. Abrir `http://<rpi-hostname>:8080` en el browser.
2. Click "Comenzar" — desbloquea AudioContext y comienza el preview.
3. Posicionar el board a ~1.5m de la cámara. El status indica "cerca" o "lejos" si está fuera del target range.
4. **Ajustar el lens izquierdo** girando la llave del barrel hasta que las barras de nitidez central y corners pasen (verde). La llave da palanca para movimientos finos — giros de 2-5° (1-3μm axial) permiten encontrar el peak con precisión.
5. **Repetir para el lens derecho**. La barra de simetría L/R debe quedar por debajo del umbral.
6. Cuando ambos lens están en verde y el banner dice "LISTO", click "Finalizar". Guardar el reporte HTML.
7. **Retirar la llave del barrel** ahora que el foco está logrado.

### Paso 2B — Aplicar esmalte y dejar curar

1. Con un pincelito fino (el del propio frasco de esmalte sirve si no chorrea, o un palillo de dientes para más control), **pintar el seam exterior** donde el barrel del lens se encuentra con la boca del holder. Una pasada delgada — el esmalte solo necesita "pegar" las dos piezas, no rellenar nada.
2. **Aplicar a ambos lens (izquierdo y derecho)** antes de que el primero termine de secar — el touch-dry tarda 5-10 min, hay tiempo de sobra.
3. Verificar que **no haya esmalte sobre la óptica**: si una gota cayó al cristal frontal del lens, limpiarla AHORA con hisopo + isopropílico antes de que cure.
4. **Esperar 15-20 min** desde la última pasada. En ese tiempo el touch-dry es completo: los lens quedan inmóviles bajo cargas vibratorias normales del manejo de calibración, aunque la cura full demore 30-60 min más.
5. **No rotar los lens después de este punto**. Cualquier giro rompe el sello del esmalte y desacomoda el foco logrado.

**Planning del cronograma**:

Todo el ciclo (foco + aplicar esmalte + 15min de espera + calibración + ground-truth) entra en una sola sesión de lab de ~1-1.5h. La cura full del esmalte continúa en background mientras se hace la calibración; al cabo de ~1h el set está completamente rígido.

## Paso 3 — Calibración estéreo

Objetivo: obtener los intrínsecos (K, D por cámara) y extrínsecos (R, T entre L/R) usando el modelo **fisheye Kannala-Brandt** (`cv2.fisheye.*`).

### Correr el wizard

```bash
python scripts/calibrate.py wizard --device-id DEV-XXX
```

Reemplazar `DEV-XXX` con el identificador del dispositivo (va al reporte).

**Modos de captura.** El default es **barrido libre (sweep)**: movés el board libremente por el cuadro y la herramienta auto-selecciona los frames diversos que necesita (gate de novedad + quietud + calidad), guiándote con un mapa de cobertura en vivo (grilla 3×3 × distancias × inclinaciones). Es mucho más fácil de operar en espacios chicos / luz difícil. Para **máxima precisión** con buen espacio + luz, agregá `--guided`: el modo clásico de 20 poses-silueta a 1/2/3m (`--manual` lo hace captura-por-botón). En luz baja: `--max-exposure-us 0` (o `50000`) destapa el shutter y, si hace falta, `--low-light` afloja los gates — aunque lo ideal es **sumar luz al board**. Ambos modos terminan igual: calibran, verifican y generan el **reporte HTML**.

Defaults del modo guiado: resolución 2304×1296 binned (4× más rápido que full-res en detect, mismo FOV), far=3.0m (cabe en tripod 70–210cm), 20 poses canónicas, tolerance "normal", pose-timeout 180s (tripod-friendly).

**Antes de arrancar**, verificá que las cámaras estén bien mapeadas con `focus_assist` — la pill verde "✓ L/R OK" en el panel del browser confirma. Si dice "L/R INVERTIDO", reiniciá pasando `--left/--right` swappeados (el wizard tiene los mismos flags). Calibrar contra un par invertido produce extrínsecos sign-flipped silenciosos.

**Flags compartidos con los otros setup tools** (`focus_assist`, `preview`, `diagnose_depth`, `diagnose_calibration`):

- `--max-exposure-us 16000` (default) — cap de shutter time a 16ms vía `FrameDurationLimits`. Mismo cap que el runtime para que la distribución de motion blur de la captura matchee la que vé el detector en producción. Pasar `0` para deshabilitar.
- `--lock-ae` — patrón canónico settle 2s → lock provisional → re-settle 1.5s al apretar Comenzar → re-lock final. Útil cuando la escena tiene luz variable (vidrieras, HVAC). Default OFF (AE auto, más simple). Los timings (`initial_settle_seconds`, `resettle_seconds`) se leen de `vision.ae_lock` del config per-device — en sites donde AE necesita más tiempo de convergencia, subir esos valores.
- `--meter matrix|centre|spot` — modo de AE metering. `matrix` (default) pondera todo el frame; `centre`/`spot` ignoran la periferia y exponen para el centro — usar cuando hay zonas brillantes alrededor del board (ventanas, paredes detrás de un backdrop texturado) que arrastran la exposición hacia abajo sobre el board.
- Parámetros del board (`--board-cols`, `--board-rows`, `--square-mm`, `--marker-mm`, `--dict`, `--legacy-pattern`) defaultean desde `vision.charuco` del config per-device. Para un board no canónico se edita el config; los CLI flags siguen funcionando como override per-corrida.

### Si necesitás restart limpio

Para arrancar la sesión desde cero (limpia capturas previas, `session.json` y `.npz`):

```bash
python scripts/calibrate.py reset --yes
```

Borra captures + session.json + .npz. Sin `--yes` lista qué borraría sin tocar nada.

### Salvaguardas del wizard

Si las capturas no son suficientemente diversas o detection falla en una de las dos cámaras, el wizard **bloquea antes de calibrar** para evitar producir un fit degenerado (RMS bueno pero ground-truth mal). Mensajes que vas a ver:

- **❌ Coverage crítico insuficiente**: falta una banda completa (near/mid/far) o un grupo entero (A/B/C/D). Re-capturá esas poses. Bypass: `--force-degenerate-coverage` (úsalo solo si entendés el riesgo).
- **❌ Pre-calibration sanity falló**: <70% de los pares sobreviven la re-detección estricta en ambas cámaras. Revisá lens limpio / foco / iluminación y recapturá. Para bajar el umbral: `--min-detect-rate 0.5` (otra vez, riesgo asumido).
- **⚠ Cámara R/L no detecta**: una cámara está en 0 corners por 20+ frames mientras la otra detecta. Limpiá lens, chequeá foco, asegurate que el board entre en su FOV. El wizard espera — no te traba.

### Las 20 poses

El wizard guía al operador a través de 20 poses, agrupadas por zona del frame y nivel de inclinación. Las alturas del stud del tripod del board (para los offsets de mount de referencia) están en la tabla del Paso 1; acá está solo la distribución lógica:

- **A1-A4** (center near, 1.0m): frontal + pitch + yaw + mix.
- **B1-B4** (corners mid, 2.0m): con yaw/pitch.
- **C1-C4** (mid, roll y pitch extra): roll ±20°, top/bottom center.
- **D1-D5** (far, 3.0m): centro + 4 esquinas con diagonal tilt.
- **E1-E3** (extreme tilts at near): yaw ±25°, combinación 3-axis.

Para cada pose:

1. El wizard muestra el ghost overlay (silueta translúcida del board en la posición target), emite un doble-tap agudo de notificación y muestra en el banner "Pose N. [label]. A [X]cm de la cámara".
2. El operador **mueve el trípode del board** para que coincida con el ghost: distancia, posición lateral, altura. Ajusta tilts de la cabeza del trípode (pitch/yaw/roll) según indique el ghost.
3. El wizard detecta alineación + estabilidad (10 frames consecutivos con < 1.5px de jitter) → **captura automática**. Beep cálido (660 Hz × 220ms) confirma el shot.
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

1. **Esperar la cura full del esmalte** (~30-60 min desde la aplicación). El dispositivo puede quedar en cualquier orientación durante este tiempo, pero sin manipular los lens.
2. **Verificar que los lens estén firmes**: intentar girarlos suavemente a mano (sin la llave). Deben resistir. Si se mueven, el esmalte no curó bien o la pasada fue demasiado fina — aplicar otra capa y esperar nuevamente.
3. **No desarmar el bracket L/R**. La calibración viaja con el par físico, no con el dispositivo RPi. Si separás los lens, la extrínseca cambia y la calibración se invalida.
4. **Etiquetar el bracket** con el device-id del reporte.
5. **Guardar el `.npz`** en la estructura del provisioning (ver `scripts/provision.py`) asociado al device-id.
6. **Archivar las PNGs crudas** (no eliminarlas del RPi). Son oro si después hay que re-calibrar con otro modelo o validar con otra herramienta.

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

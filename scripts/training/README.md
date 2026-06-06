# People Counter — Pipeline del detector

El detector cenital es **YOLOv8n fine-tuneado** sobre dataset propio
(geometría top-down, mounting 2.0–3.5m). Single-class (`person`),
input 640×640. Compilado a HEF para Hailo-8L con NMS on-chip.

End-to-end:

```
captures multi-site (motion-trigger)  →  sample_for_labeling.py (estratificado)
                                      →  X-AnyLabeling (local, bbox cabeza+hombros)
                                      →  labelme_to_yolo.py (dataset YOLO)
                                      →  Kaggle dataset privado (upload vía API)
                                      →  notebook Kaggle T4 (training, ~20 min)
                                      →  best.onnx (export desde ultralytics)
                                      →  hailomz compile (Docker x86 Linux)
                                      →  drop-in del HEF en la Pi
                                      →  postproceso runtime (tracking_zone +
                                         containment + static suppressor)
```

Esta carpeta contiene infraestructura **del lado del workstation**:
captura del pool, sampling para labeling, conversión a YOLO, active learning,
benchmarks, eval. El labeling se hace en **X-AnyLabeling local**. La
compilación corre en Docker x86 Linux; la Pi solo ejecuta el `.hef` final.

## Convención de labeling

**Antes de cada sesión de labeling**, leer [`label_guide.md`](label_guide.md).
Resumen ejecutivo:

- **Una sola clase: `person`** (no `head`).
- **Bbox = CABEZA + HOMBROS** desde vista cenital (forma de "T").
- **No** la silueta completa (rompe el v1, que generalizaba mal en bordes).
- **No** solo la cabeza (los hombros dan robustez a oclusión).
- Etiquetar también personas estáticas (cierra el gap del v1).
- No etiquetar reflejos, maniquíes, personas afuera de la vidriera.

## Herramientas

### Captura del pool de training

| Archivo | Para qué |
|---|---|
| `capture_mjpeg.py` | Capturador multi-site de streams MJPEG. Modos `--motion-trigger` (filtra absdiff, multiplica ×10–50 la fracción útil) o `--background-interval` (samples de fondo para FP rate). `--operating-hours` filtra por horario local. SBS estéreo + rectificación on-the-fly. |
| `record_clips.py` | Grabación continua de clips MP4 multi-site (un subprocess `ffmpeg` por site, `-c copy`). Para validation E2E con tracker — los snapshots de `capture_mjpeg.py` no preservan continuidad temporal. |
| `sample_for_calib.py` | Muestreo para calibration set del compile a HEF (200 imgs balanceadas por site, separadas de train/eval). |

### Labeling local (X-AnyLabeling)

| Archivo | Para qué |
|---|---|
| `sample_for_labeling.py` | **Muestreo estratificado** del pool `training_data/captures/` a una carpeta plana lista para X-AnyLabeling. Estratifica por site y por motion/bg. Renombra cada copia con prefijo `<site>__` para evitar colisiones, escribe `manifest.txt` con el mapeo. `--exclude-manifest` saltea imgs ya usadas en otro batch (anti leak train/val). `--exclude-window-seconds` extiende la exclusión a frames temporalmente cercanos del mismo site (los ramos del motion-trigger son casi-duplicados). |
| `mine_active_learning.py` | **Active learning** para v_next: minar el pool por frames informativos en vez de muestrear al azar. Combina dos señales: (1) disagreement entre el modelo actual y un oráculo high-recall (señal de recall — el oráculo detecta algo que el modelo actual perdió → probable persona); (2) uncertainty del modelo actual (detecciones con confidence 0.2-0.5). Selecciona top-N por score y los copia a carpeta para X-AnyLabeling con manifest que explica por qué se eligió cada uno. Excluye train+val+ventana via `--exclude-manifest`. |
| `labelme_to_yolo.py` | **Conversión** de la carpeta labeleada en X-AnyLabeling (formato labelme: `.json` con shapes `rectangle`) al formato canónico de Ultralytics: `images/`, `labels/<name>.txt` con `class cx cy w h` normalizado, `data.yaml`. Una imagen SIN `.json` se trata como **negativo revisado** (escribe `.txt` vacío → Ultralytics la usa como background, baja FPs). El caller garantiza que todas las imgs fueron miradas. |
| [`label_guide.md`](label_guide.md) | Convención de labeling + setup de X-AnyLabeling (qué/cómo etiquetar, casos dudosos, atajos). |

### Eval + comparación de modelos

| Archivo | Para qué |
|---|---|
| `bench_detector.py` | Bench de inferencia de modelos locales `.pt`/`.onnx`. Subcomandos `bench` (corre y dumpea JSON con métricas) y `diff` (compara dos reports). |
| `eval_yolo.py` | Corre un modelo YOLO sobre una carpeta de frames, dibuja bboxes y dumpea labels + summary. |
| `compare_detectors.py` | Side-by-side de dos detectores sobre el mismo set (visual + métricas), útil para validar que v_next no degradó casos que v_actual manejaba bien. |
| `analyze_eval_summary.py` | Análisis estadístico del output del eval — distribución de confidence, breakdown por site, etc. |
| `../capture_baseline_frames.py` | Captura frames rectificados de la Pi para usar como bench corpus (validación, **no** training). |

## Histórico de iteraciones del detector

Convención canónica desde el inicio: **single-class `person`, bbox
cabeza+hombros**, vista cenital, labeling local en X-AnyLabeling.

Métricas evaluadas contra el **mismo val set held-out** (245 imgs / 174
cajas) para todas las iteraciones — comparable apples-to-apples.

| Versión | Train imgs | Train cajas | Δ sobre anterior | mAP50 | mAP50-95 | Precision | Recall | FPS (Pi) | Estado |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| **v1** | 294 | 133 | (baseline) | 0.805 | 0.385 | 0.785 | 0.764 | — | Primera iter manual. Validó la convención cabeza+hombros y el flow X-AnyLabeling end-to-end. Recall bajo (gap en estáticos y personas marginales). |
| **v2** | 544 (+250 AL) | 438 (+305) | + 250 imgs informativas mineadas con active learning | **0.956** | **0.567** | **0.919** | **0.907** | **26.8** | **Deployado producción** desde 2026-05-20. Δ mAP50 +0.151 confirmó el ROI del AL. Confirmado "detecta de maravilla" en hardware. |
| v3 | 740 (+196 AL) | 450 (+12) | + 196 imgs vía 2da iter de AL — **186 fueron backgrounds, solo +12 cajas** | 0.939 | 0.538 | 0.884 | 0.902 | — | **Descartado**. Δ mAP50 −0.017, mAP50-95 −0.029, precision −0.035 vs v2 — peor en TODO salvo recall. **Causa real** (verificada contando el dataset `people-counter-yolo-v5`): la 2da ronda de AL surfaceó casi solo frames vacíos (bg 224→410) sin señal positiva nueva → diluyó el set con negativos. El AL ya había agotado los frames informativos en la 1ra ronda. |

**Total labelado**: 245 (val) + 294 (train v1) + 250 (AL v2) + 196 (AL
v3 descartado) = **985 imgs propias** con convención cabeza+hombros.

**Tamaño efectivo en producción** (v2 deployed): **544 imgs train + 245
val = 789 imgs**. Las 196 del v3 quedaron labeladas pero no aportaron
al modelo final — costo hundido pero confirmó empíricamente el sweet
spot con datos concretos (no intuición).

**Composición del dataset (background negatives)**: incluye a propósito
muchos frames **vacíos** (sin personas) como hard negatives para suprimir
falsos positivos sobre clutter fijo (perchero, mostrador, vidriera) — de ahí
que haya **menos cajas que imágenes**, normal en este dominio cenital de una
sola entrada (no es una escena de multitud). Vacías por versión: v1 190/294
(65%), v2 224/544 (41%), v3 410/740 (55%); las imágenes con personas
promedian ~1.3 cajas (máx 3). El salto v2→v3 fue +186 backgrounds y solo +12
cajas — la raíz de por qué la 2da ronda de AL no aportó. (Conteos exactos de
los datasets reales `people-counter-yolo-v{3,4,5}` en Kaggle.)

**Fuente de las métricas**: `results.csv` + eval final del notebook
Kaggle (`m.val(split="test")` sobre el val held-out). Logs originales
en `debug/kaggle_kernel/output/people-counter-v{3,4,5}-train.log`
(nombres legacy de los runs Kaggle, mapean a v1/v2/v3 nuevo).

### Lecciones del proceso (citables para TFG)

- **Val set held-out FIRST** — sin baseline fijo las métricas de mejora
  son ilusorias.
- **Convención escrita antes de empezar** (`label_guide.md`) — el drift
  entre sesiones de labeling es silencioso y caro.
- **No labelar todas las imgs del pool** — bajo retorno arriba de ~1500
  con muestreo random. Mejor **diversidad estratificada** (por site +
  motion/bg) + **active learning** para iteraciones.
- **Active learning > random sampling** para iteraciones — v1→v2
  agregó 250 imgs informativas (+305 cajas) y subió mAP50 de 0.80 → 0.96
  (+0.16). La 2da ronda (v3) agregó +196 imgs pero **186 eran backgrounds
  (solo +12 cajas)**: el AL ya había agotado los frames con señal positiva.
- **Detectar el sweet spot** — v3 regresó no por overfitting sino porque el
  incremento fue casi todo negativos (bg 224→410) que diluyeron el set.
  Saber cuándo parar de iterar es parte de la metodología, no desperdicio.

## Workflow de iteración (active learning)

Iteración típica para v(N+1) tras observar gaps del modelo actual v(N):

```bash
# 1. Identificar frames informativos del pool (active learning).
#    --v1 = oráculo high-recall (opcional, para señal de disagreement);
#    --v3 = modelo actual (current; el nombre del flag es histórico).
python scripts/training/mine_active_learning.py \
    --captures training_data/captures/ \
    --output training_data/label_v_next_01/ \
    --v3 models/training/people-counter-detector.pt \
    --v1 models/training/people-counter-detector.pt \
    --n-total 250 \
    --exclude-manifest training_data/label_train_v1.manifest \
    --exclude-manifest training_data/label_val_01.manifest

# 2. Labeling local en X-AnyLabeling (ver label_guide.md).
#    Abrir training_data/label_v_next_01/ en X-AnyLabeling.
#    Dibujar rectángulos cabeza+hombros. Save → .json + .txt YOLO local.

# 3. Convertir a YOLO + mergear con el train acumulado.
python scripts/training/labelme_to_yolo.py \
    --input training_data/label_v_next_01/ \
    --output training_data/dataset_v_next/

# 4. Subir a Kaggle dataset privado (workflow vía API en
#    memory: kaggle_automation_via_api).
kaggle datasets version -p training_data/dataset_v_next -m "v_next batch 01"

# 5. Notebook Kaggle T4: attach el dataset privado en el sidebar,
#    actualizar el name del run, Save & Run All (~20 min en T4).

# 6. Eval del nuevo modelo contra el val set held-out.
python scripts/training/eval_yolo.py \
    --model models/training/people-counter-detector-v_next.pt \
    --frames training_data/label_val_01/ \
    --report debug/eval_v_next.json

# 7. Comparar con el modelo actual (precisión, recall, donde falla cada uno).
python scripts/training/compare_detectors.py \
    --frames training_data/label_val_01/ \
    --models current=models/training/people-counter-detector.pt \
             v_next=models/training/people-counter-detector-v_next.pt \
    --output debug/compare_v_next.html

# 8. Si v_next no regresiona en casos que el actual maneja bien,
#    compilar a HEF + deployar (ver "Compilar a HEF y deployar" abajo).
#    Si NO mejora, descartar y documentarlo (lección citable).
```

**Histórico**: v1→v2 con active learning mejoró mAP50 de 0.80 → 0.96
(ver tabla de iteraciones abajo). La 2da ronda de AL (v3) no mejoró
sobre v2 — rendimientos decrecientes después de la primera iter.

## Validation set held-out

`capture_mjpeg.py` corre en el workstation contra los streams MJPEG de
los sites disponibles. Los sites se declaran en `training_data/sites.yaml`
(gitignored porque contiene IPs reales + matrices de calibración).
Template documentado en `training_data/sites.yaml.example`. Ejemplo:

```bash
python scripts/training/capture_mjpeg.py \
    --duration 3600 \
    --motion-trigger \
    --motion-threshold 5.0 \
    --min-interval 5 \
    --background-interval 600
```

(`--config` y `--output` default a `training_data/sites.yaml` y
`training_data/captures` respectivamente).

Los filenames distinguen el origen:
- `<ts>_motion_<rand>.jpg` — cambio detectado en absdiff
- `<ts>_bg_<rand>.jpg` — control de fondo periódico

Para sites con stream SBS (estéreo lado-a-lado en un único MJPEG) la
rectificación se aplica on-the-fly usando las matrices `K`, `D`, `R_rect`
y `P_rect` embebidas inline en cada site del YAML — el output queda
listo para training sin post-processing manual y sin dependencia de
dumps externos. Ver `training_data/README.md` para el flujo completo
y `training_data/sites.yaml.example` para el formato.

**El validation set se reserva** (no se entrena con él) para comparar
modelos cross-iteración. Los batches de training van en carpetas separadas
(`label_train_v1`, `label_v_next_01`, etc.) — el `--exclude-manifest` de
`sample_for_labeling.py` / `mine_active_learning.py` garantiza zero
overlap.

## Composición del dataset actual

- **Volumen v2 (deployed)**: 294 imgs base (`label_train_01`) + 250 imgs
  nuevas vía active learning (`label_al_01`).
- **Single-class** `person`, bbox cabeza+hombros.
- **Sites**: 5 en paralelo (workers concurrentes de `capture_mjpeg.py`).
- **Estratificación**: per-site uniforme (motion + bg balanceados), con
  override `--site-cap` para sites con sesgo conocido (vidriera, reflejos
  fuertes).
- **Hard negatives explícitos**: bg captures del `--background-interval`
  cubren clutter persistente (ropa colgada, sombras, estructura). En la
  revisión las bg con persona se promueven a positivo; las limpias entran
  como background (`.txt` vacío en formato YOLO).
- **Ratio target post-screening**: ~2:1 positivos:negativos. Cargado a
  positivos para favorecer recall.
- **Defense-in-depth runtime** (independiente del modelo): post-NMS el
  pipeline aplica containment filter (descarta bbox chico contenido >50%
  en otro de mayor confianza) + `StaticSuppressor` (cuadricula el frame
  en celdas de 30px y suprime detecciones sobre celdas con hit-rate ≥70%
  en una ventana rolling de 15s) + opcionalmente `tracking_zone` polygon
  filter pre-tracker (ver `docs/tracker_tuning.md` patrón 6).

## Workflow del bench

```bash
pip install ultralytics

# Capturar frames del Pi (en la Pi)
python3 scripts/capture_baseline_frames.py \
    --config /etc/people-counter/config.yaml \
    --num-frames 30 --interval 2.0 \
    --output /tmp/baseline_frames

# SCP al workstation
scp -r pi@<ip>:/tmp/baseline_frames debug/baseline_frames

# Bench de un modelo local
python scripts/training/bench_detector.py bench \
    --weights /path/to/best.pt \
    --frames debug/baseline_frames \
    --report debug/bench_local.json

# Diff entre dos reportes locales
python scripts/training/bench_detector.py diff \
    debug/bench_baseline.json debug/bench_local.json
```

Para que el bench sea informativo, los frames del Pi deben estar bien
curados:
- Buena iluminación (todas las luces prendidas, evitar sombras).
- Sin objetos cotidianos en escena que el modelo pueda confundir con
  cabezas (posavasos, latas, electrodomésticos circulares).
- Persona caminando despacio, pasos firmes y rectos, pausas de 2–3s en
  distintas posiciones del frame con cabeza centrada y nítida.
- Mezcla de ~10 frames vacíos (medir false positives) + ~20 con cabeza
  claramente visible.

## Compilar a HEF y deployar

La compilación corre solo en x86 Linux. La Pi es ARM y Windows nativo
no soporta el Hailo Dataflow Compiler. Usamos **Docker Desktop** con
backend WSL2 en Windows y la imagen `hailo8_ai_sw_suite_2025-10`.

### Setup one-time del Hailo SDK

1. Cuenta en <https://hailo.ai/developer-zone/> (gratis, verificación
   por email).
2. **Software Downloads** → bajar la imagen Hailo AI SW Suite (`.tar.gz`).
3. Cargar la imagen en Docker:
   ```bash
   docker load -i hailo_ai_sw_suite_2025-10.docker.tar.gz
   ```

### Export YOLOv8n a ONNX

Lo hace el notebook `train_head_detector.ipynb` automáticamente al
final del run (`m.export(format="onnx", imgsz=640, opset=11,
simplify=True)`). El `.onnx` queda disponible en el panel Output del
Kaggle como `<name>.onnx`.

Bajar el `.onnx` al workstation:

```bash
mkdir -p models/training/people-counter-detector/calib
mv ~/Downloads/people-counter-detector.onnx \
    models/training/people-counter-detector/best.onnx
```

### Calibration set

200 imágenes representativas del dominio de deployment, balanceadas
por site, **sin overlap con training/eval**. Pegarlas en
`models/training/people-counter-detector/calib/`. Generar el set con
`sample_for_calib.py` desde el pool de captures.

### Compile a HEF

```bash
docker run --rm \
    -v "C:/Users/MauroGasparri/source/repos/people-counter/models:/workspace/models" \
    --entrypoint bash hailo8_ai_sw_suite_2025-10:1 \
    -c "cd /workspace/models/training/people-counter-detector && hailomz compile yolov8n \
        --ckpt best.onnx \
        --hw-arch hailo8l \
        --calib-path calib/ \
        --classes 1 \
        --end-node-names \
            /model.22/cv2.0/cv2.0.2/Conv \
            /model.22/cv3.0/cv3.0.2/Conv \
            /model.22/cv2.1/cv2.1.2/Conv \
            /model.22/cv3.1/cv3.1.2/Conv \
            /model.22/cv2.2/cv2.2.2/Conv \
            /model.22/cv3.2/cv3.2.2/Conv"
```

Caveats:

- **`--end-node-names` es obligatorio**. El ONNX exportado por
  ultralytics termina en `Concat` + `Sigmoid` (post-NMS). El recipe
  `yolov8n.alls` del Hailo Model Zoo hace su propio NMS on-chip y
  necesita los 6 conv layers ANTES del concat (3 detection heads × 2
  outputs: `cv2` = bbox regression, `cv3` = classification). Sin esto
  el compile aborta con `AllocatorScriptParserException`.
- **`--classes 1`** es necesario para single-class. Sin esto asume 80
  (COCO) y el HEF queda con NMS configurado para 80 outputs.
- **Path Windows en `-v`**: Docker Desktop con WSL2 acepta forward
  slash (`C:/Users/...`). NO usar el formato git-bash `/c/Users/...`
  — Docker lo interpreta como path Linux inexistente y monta vacío.
- **3 contexts en hailo8l es normal**: el modelo no fitea en 1 context
  del entry-level. El compile splittea automáticamente. Performance
  validada: ~14 FPS en pipeline real.

10–20 min según CPU. Output: `<model>.hef`.

### Deploy a la Pi

```bash
scp models/training/people-counter-detector/yolov8n.hef \
    pi@<ip>:/tmp/people-counter-detector.hef

ssh pi@<ip> << 'EOF'
sudo mv /tmp/people-counter-detector.hef \
    /usr/src/people-counter/models/people-counter-detector.hef
sudo chown root:root /usr/src/people-counter/models/people-counter-detector.hef
sudo systemctl restart people-counter
sleep 3
sudo systemctl status people-counter --no-pager | head -20
EOF
```

Editar `/etc/people-counter/config.yaml` (o el shadow runtime-safe):

```yaml
detection:
  architecture: yolov8
  model_path: /usr/src/people-counter/models/people-counter-detector.hef
```

**Service systemd**: `ReadWritePaths` debe incluir
`/usr/src/people-counter` porque HailoRT escribe `pyhailort.log` y
`hailort.log` en `WorkingDirectory`. Sin ese path el activate() falla
con `errno=30` (read-only filesystem). Ya configurado en
`config/people-counter.service`.

Smoke test:

```bash
ssh pi@<ip> "sudo journalctl -u people-counter -n 50 --no-pager"
```

Buscar `Loading model: /usr/src/people-counter/models/people-counter-detector.hef`
sin error subsiguiente, y detecciones periódicas con personas en frame.
Pi dev sin certs MQTT: arrancar manual con `--no-mqtt` para smoke test.

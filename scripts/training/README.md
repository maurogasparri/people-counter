# People Counter — Pipeline del detector

El detector cenital es **YOLOv8n fine-tuneado** sobre dataset propio
(geometría top-down, mounting 2.0–3.5m). Single-class (`person`),
input 640×640. Compilado a HEF para Hailo-8L con NMS on-chip.

End-to-end:

```
dataset Roboflow (overhead heads)  →  notebook Kaggle T4 (training, ~20 min)
                                  →  best.onnx  (export desde ultralytics)
                                  →  hailomz compile (Docker x86 Linux)
                                  →  drop-in del HEF en la Pi
                                  →  postproceso runtime (containment + static suppressor)
```

Esta carpeta contiene infraestructura **del lado del workstation**:
descarga de datasets, captura de validation sets, benchmarks, sampling,
y el notebook de training. La compilación corre en Docker x86 Linux,
la Pi solo ejecuta el `.hef` final.

## Herramientas

| Archivo | Para qué |
|---|---|
| `train_head_detector.ipynb` | Notebook único Kaggle T4. Para iterar el modelo (v3, v4, ...) basta actualizar la URL de Roboflow en Cell 2 y el `name` del run en Cell 3. ~20 min en T4. |
| `download_roboflow.py` | Pull de un dataset Roboflow Universe a `dataset/` (formato YOLOv8). |
| `bench_detector.py` | Bench de inferencia de modelos locales `.pt`/`.onnx` (subcomandos `bench` y `diff`). |
| `bench_roboflow_api.py` | Triage de modelos publicados en Roboflow Universe vía REST sin descargar pesos. Útil para evaluar candidatos rápido. |
| `capture_mjpeg.py` | Capturador multi-site de streams MJPEG HTTP. Modos: random-interval o motion-trigger (`cv2.absdiff` entre frames consecutivos, multiplica × 10–50 la fracción útil). `--background-interval` agrega samples de fondo para medir FP rate. `--operating-hours` filtra por horario local. Soporta SBS estéreo + rectificación on-the-fly. |
| `record_clips.py` | Grabación continua de clips MP4 multi-site (un subprocess `ffmpeg` por site, `-c copy`). Para validation E2E con tracker — los snapshots de `capture_mjpeg.py` no preservan continuidad temporal. |
| `sample_for_roboflow.py` | Sampling estratificado de capturas para subir a Roboflow (balance positivos / hard negatives). |
| `polys_to_bboxes.py` | Conversión de polígonos SAM3 a bboxes YOLO. |
| `eval_yolo.py` | Corre un modelo YOLO sobre una carpeta de frames, dibuja bboxes y dumpea labels + summary. |
| `../capture_baseline_frames.py` | Captura frames rectificados de la Pi para usar como bench corpus (validación, **no** training). |
| `.env.example` | Convención del env-var `ROBOFLOW_API_KEY` (el `.env` real está gitignoreado). |

## Composición del dataset

Roboflow project: `people-counter-detector`, tipo Object Detection
(SAM3 polys se auto-convierten a bboxes en este tipo de project).

- **Volumen**: ~945 imgs sampleadas con `sample_for_roboflow.py` (490 motion + 455 bg) desde el pool multi-site `debug/mjpeg_capture/`. Balance de capacidad de pre-label: 11 credits Roboflow × 100 imgs/credit = 1100 ceiling, dejando ~155 de margen.
- **Stratificación por site**: 7 sites a 75 motion + 65 bg cada uno, salvo `site_54_21` capeado a 40 motion (vidriera con reflejo donde el detector ve poco — sobre-representarlo sesga el set).
- **Hard negatives explícitos**: bg captures del `--background-interval` cubren clutter persistente (ropa colgada, sombras, estructura). En la revisión, las bg que SAM3 detecta con persona se promueven a positivo; las que quedan limpias entran como "null examples" en Roboflow.
- **Ratio target post-screening**: ~2:1 positivos:negativos. Cargado a positivos para favorecer recall.
- **Defense-in-depth runtime** (independiente del modelo): post-NMS el pipeline aplica containment filter (descarta bbox chico contenido >50% en otro de mayor confianza) + `StaticSuppressor` (cuadricula el frame en celdas de 30px y suprime detecciones sobre celdas con hit-rate ≥70% en una ventana rolling de 3s).

> **Generate Version en Roboflow**: confirmá `Filter Null = Use / Include Null Images` antes de generar — por default Roboflow descarta las imágenes sin labels y se pierde todo el aporte de los hard negatives. Augmentations recomendadas: flip H, rotate ±10°, brightness ±20%, blur ligero (2× = ~1890 imgs finales). Evitar mosaic, shear/perspective fuerte y cutout-on-bbox: rompen el realismo cenital.

## Validation set

`capture_mjpeg.py` corre en el workstation contra los streams MJPEG de
los sites disponibles. Configurás los sites en un YAML (formato en el
docstring del script). Ejemplo:

```bash
python scripts/training/capture_mjpeg.py \
    --config debug/mjpeg_sites.yaml \
    --output debug/mjpeg_capture \
    --duration 3600 \
    --motion-trigger \
    --motion-threshold 5.0 \
    --min-interval 5 \
    --background-interval 600
```

Los filenames distinguen el origen:
- `<ts>_motion_<rand>.jpg` — cambio detectado en absdiff
- `<ts>_bg_<rand>.jpg` — control de fondo periódico

Para sites con stream SBS (estéreo lado-a-lado en un único MJPEG) la
rectificación se aplica on-the-fly usando `debug/calib_dumps/<ip>/calib.npz`
— el output queda listo para training sin post-processing manual. Ver
`debug/mjpeg_sites.yaml` para el formato.

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

# Bench de un modelo Roboflow Universe vía API (sin bajar pesos)
export ROBOFLOW_API_KEY=...
python scripts/training/bench_roboflow_api.py \
    --frames debug/baseline_frames \
    --report debug/bench_remote.json \
    --models "<workspace>/<project>/<version>"

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
final del run (Cell 4: `m.export(format="onnx", imgsz=640, opset=11,
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
`models/training/people-counter-detector/calib/`.

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

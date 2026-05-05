# People Counter — Pipeline de training

Fine-tune de `yolov8n` para detección de cabezas cenitales, para que el
Hailo-8L vea un modelo entrenado en la geometría que produce el bracket
(frames rectificados top-down) en lugar de caras frontales tipo
CrowdHuman. End-to-end:

```
preprocesar dataset  →  train en Kaggle (T4)  →  export ONNX
        →  hailomz compile (WSL2)  →  drop-in del HEF en la Pi
```

Esta carpeta tiene infraestructura **del lado del workstation**:
descargas, benchmarks, comparativas. El training corre en Kaggle, la
compilación en WSL2, la Pi solo ejecuta el `.hef` final.

## Herramientas

| Archivo | Para qué |
|---|---|
| `download_roboflow.py` | Pull de un dataset Roboflow Universe a `dataset/` (formato YOLOv8). |
| `bench_detector.py` | Bench de inferencia (`bench`) + diff entre reportes (`diff`). Compara modelos sobre un corpus de frames. |
| `../capture_baseline_frames.py` | Captura frames rectificados de la Pi para usar como bench corpus (validación, **no** training). |
| `.env.example` | Convención del env-var `ROBOFLOW_API_KEY` (el `.env` real está gitignoreado). |

## Estado del modelo

**TBD.** El modelo de producción todavía no está definido. Iteraciones
previas con datasets cenitales de Roboflow Universe (Coding Compass
overhead-head-detection, Person counter) fueron descartadas por bias o
mismatch de domain.

Próximo plan: **WEPDTOF** (Boston VIP COSSY) — 14k frames in-the-wild
cenitales, fisheye undistortados con su propio K/D, convertidos a
formato YOLO axis-aligned, entrenados desde `yolov8n.pt` (COCO
pretrained). Single source de training, sin combinar con otros datasets
para evitar contaminación de bias.

Cuando esté listo el preprocessing de WEPDTOF, este README se expande
con los pasos del pipeline (preprocesar → upload Kaggle Dataset →
notebook training → bench → compile → deploy).

## Workflow de Kaggle (cuando el modelo esté definido)

El método de corrida es **Save Version → Save & Run All (Commit)**.
Notas operativas que aprendimos en sangre:

1. Setup one-time:
   - Cuenta en <https://www.kaggle.com> + verificación de teléfono
     (Settings → Phone Verification, requerido para habilitar GPU).
   - API token en `~/.kaggle/access_token` (formato nuevo) o
     `~/.kaggle/kaggle.json` (viejo).

2. Cada run:
   - Sidebar derecho → Settings → Accelerator → **GPU T4 x2**.
   - Add-ons → Secrets → `ROBOFLOW_API_KEY` Attached → ON.
   - Esquina superior derecha → **Save Version → Save & Run All
     (Commit)** (NO Quick Save — Quick Save no captura
     `/kaggle/working/`).

3. Descarga de outputs:
   ```powershell
   $env:Path += ";$env:USERPROFILE\AppData\Local\Programs\Python\Python312\Scripts"
   kaggle kernels output <username>/<notebook-slug> -p <dest-folder>
   ```

   Funciona indefinidamente — la Version queda inmutable.

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

# Bench de baseline (modelo viejo)
python scripts/training/bench_detector.py bench \
    --weights yolov8n.pt \
    --frames debug/baseline_frames \
    --report debug/bench_baseline.json

# Bench del modelo nuevo
python scripts/training/bench_detector.py bench \
    --weights /path/to/best.pt \
    --frames debug/baseline_frames \
    --report debug/bench_finetuned.json

# Diff
python scripts/training/bench_detector.py diff \
    debug/bench_baseline.json debug/bench_finetuned.json
```

Para que el bench sea informativo, los frames del Pi deben estar bien
curados:
- Buena iluminación (todas las luces prendidas, evitar sombras).
- Sin objetos cotidianos en escena que el modelo pueda confundir con
  cabezas (posavasos, latas, electrodomésticos circulares).
- Persona caminando despacio, pasos firmes y rectos, pausas de 2-3s
  en distintas posiciones del frame con cabeza centrada y nítida.
- Mezcla de ~10 frames vacíos (medir false positives) + ~20 con
  cabeza claramente visible.

## Compilar a HEF y deployar (cuando haya un modelo aceptable)

La compilación corre solo en x86 Linux. La Pi es ARM y Windows nativo
no soporta el Hailo Dataflow Compiler. Usamos **WSL2 Ubuntu**.

### Setup one-time del Hailo SDK en WSL2

1. Cuenta en <https://hailo.ai/developer-zone/> (gratis, verificación
   por email).
2. **Software Downloads** → descargar:
   - `hailo_dataflow_compiler-X.X.X-py3-none-linux_x86_64.whl`
   - `hailo_model_zoo-X.X.X-py3-none-any.whl`
3. Adentro de WSL2 Ubuntu 22.04:

   ```bash
   sudo apt update
   sudo apt install -y python3.10 python3.10-venv python3-pip \
       libgraphviz-dev graphviz build-essential

   python3.10 -m venv ~/hailo-env
   source ~/hailo-env/bin/activate
   pip install --upgrade pip wheel setuptools

   pip install /mnt/c/Users/MauroGasparri/Downloads/hailo_dataflow_compiler-*.whl
   pip install /mnt/c/Users/MauroGasparri/Downloads/hailo_model_zoo-*.whl

   hailomz --version
   ```

### Compilar

```bash
# WSL2, venv activo
cd /mnt/c/Users/MauroGasparri/source/repos/people-counter/<path-to-onnx>/

hailomz compile yolov8n \
    --ckpt best.onnx \
    --hw-arch hailo8l \
    --calib-path calib/
```

10-20 min según CPU. Output: `yolov8n.hef`.

### Deploy a la Pi

```bash
scp yolov8n.hef pi@<ip>:/tmp/yolov8n.hef

ssh pi@<ip> << 'EOF'
sudo mv /tmp/yolov8n.hef /usr/src/people-counter/models/yolov8n.hef
sudo chown root:root /usr/src/people-counter/models/yolov8n.hef
sudo systemctl restart people-counter
sleep 3
sudo systemctl status people-counter --no-pager | head -20
EOF
```

`config.yaml` ya apunta a ese path vía `detection.model_path`.

Smoke test:

```bash
ssh pi@<ip> "sudo journalctl -u people-counter -n 50 --no-pager"
```

Buscar la línea `Loading model: /usr/src/people-counter/models/yolov8n.hef`
sin error subsiguiente, y detecciones periódicas si te parás delante de
la cámara.

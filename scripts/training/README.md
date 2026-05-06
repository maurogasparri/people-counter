# People Counter — Pipeline del detector

El detector cenital es **RAPiD pretrained** (Rotation-Aware People
Detection, Boston VIP COSSY). Backbone Darknet-53 con head custom de
rotated-bbox; output `[cx, cy, w, h, angle, conf]`. Funciona directo
sobre raw fisheye sin necesitar undistortion. License non-commercial:
uso autorizado en TFG; en eventual paso a producto comercial se
entrenaría un sustituto sobre data real recolectada en pilotos.

End-to-end:

```
RAPiD weights  →  export ONNX (script del repo de RAPiD)
              →  hailomz compile (WSL2/Docker x86 Linux)
              →  drop-in del HEF en la Pi
              →  postproceso (geom filter + ROI) en runtime
```

Esta carpeta tiene infraestructura **del lado del workstation**:
descargas de datasets, benchmarks, comparativas, captura de validation
sets. La compilación corre en WSL2/Docker, la Pi solo ejecuta el `.hef`
final.

## Herramientas

| Archivo | Para qué |
|---|---|
| `download_roboflow.py` | Pull de un dataset Roboflow Universe a `dataset/` (formato YOLOv8). |
| `bench_detector.py` | Bench de inferencia de modelos locales `.pt`/`.onnx` (subcomandos `bench` y `diff`). |
| `bench_roboflow_api.py` | Triage de modelos publicados en Roboflow Universe via REST sin descargar pesos. Útil para evaluar candidatos rápido. |
| `capture_mjpeg.py` | Capturador multi-site de streams MJPEG HTTP. Modos: random-interval o motion-trigger (`cv2.absdiff` entre frames consecutivos, multiplica × 10-50 la fracción útil para validation). `--background-interval` agrega samples de fondo para medir FP rate. `--operating-hours` filtra por horario local. |
| `../capture_baseline_frames.py` | Captura frames rectificados de la Pi para usar como bench corpus (validación, **no** training). |
| `.env.example` | Convención del env-var `ROBOFLOW_API_KEY` (el `.env` real está gitignoreado). |

## Modelo elegido: RAPiD MWHB1024

- **Weights**: `pL1_MWHB1024_Mar11_4000.ckpt` (entrenado en COCO + MW-R + HABBOF, input 1024×1024).
  Descarga: `https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB1024_Mar11_4000.ckpt`
- **Recall sobre baseline frames**: ~80% out-of-the-box, sin training.
- **Conf threshold operativo**: 0.30. Subir a 0.50 mejora precision al
  costo de perder TPs de baja confianza (motion blur, oclusión parcial).
- **Bbox semantics**: RAPiD detecta personas, no cabezas. En geometría
  cenital extrema (lente 120° HFOV centrado, mounting 2.5-4m) el
  centroide del bbox cae aproximadamente sobre la cabeza. Para depth
  lookup robusto, usar `argmax(disparity)` dentro del bbox en lugar del
  centroide (la cabeza es siempre la parte más cercana al techo, máxima
  disparity).

### Postprocesado runtime (cero costo, on-CPU post-NMS)

Filter geométrico para frames 1152×648:
- `area in [8000, 120000]` px²
- `min(w, h) >= 60` px (filtra slivers)
- `max(w,h)/min(w,h) <= 3` (rechaza shapes elongadas no humanas)

ROI mask per-deployment: máscara binaria que excluye zonas de clutter
conocidas (rack, escritorio, maniquíes). Aplicada antes de detect.

## Validation set

`capture_mjpeg.py` corre en el workstation contra los streams MJPEG
de los sites a los que tengamos acceso. Configurás los sites en un YAML
(formato en el docstring del script). Ejemplo de invocación:

```bash
python scripts/training/capture_mjpeg.py \
    --config debug/sites.yaml \
    --output debug/captures \
    --duration 3600 \
    --motion-trigger \
    --motion-threshold 5.0 \
    --min-interval 5 \
    --background-interval 600
```

Filenames distinguen tipo: `<ts>_motion_<rand>.jpg` (cambio detectado)
vs `<ts>_bg_<rand>.jpg` (control de fondo periódico).

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

# Bench de un modelo Roboflow Universe via API (sin bajar pesos)
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
- Persona caminando despacio, pasos firmes y rectos, pausas de 2-3s
  en distintas posiciones del frame con cabeza centrada y nítida.
- Mezcla de ~10 frames vacíos (medir false positives) + ~20 con
  cabeza claramente visible.

## Compilar a HEF y deployar

La compilación corre solo en x86 Linux. La Pi es ARM y Windows nativo
no soporta el Hailo Dataflow Compiler. Usamos **WSL2 Ubuntu** o
**Docker** dentro de Windows.

### Setup one-time del Hailo SDK

1. Cuenta en <https://hailo.ai/developer-zone/> (gratis, verificación
   por email).
2. **Software Downloads** → descargar:
   - `hailo_dataflow_compiler-X.X.X-py3-none-linux_x86_64.whl`
   - `hailo_model_zoo-X.X.X-py3-none-any.whl`
3. Adentro de WSL2 Ubuntu 22.04 (o un container Docker x86 Linux):

   ```bash
   sudo apt update
   sudo apt install -y python3.10 python3.10-venv python3-pip \
       libgraphviz-dev graphviz build-essential

   python3.10 -m venv ~/hailo-env
   source ~/hailo-env/bin/activate
   pip install --upgrade pip wheel setuptools

   pip install /mnt/c/<path>/hailo_dataflow_compiler-*.whl
   pip install /mnt/c/<path>/hailo_model_zoo-*.whl

   hailomz --version
   ```

### Export RAPiD a ONNX

El repo de RAPiD trae `export_onnx.py`. Apuntar al checkpoint:

```bash
git clone https://github.com/duanzhiihao/RAPiD.git
cd RAPiD
# Copiar el .ckpt elegido a weights/
python export_onnx.py --ckpt weights/pL1_MWHB1024_Mar11_4000.ckpt \
    --output rapid_mwhb1024.onnx
```

### Compile a HEF

```bash
# WSL2 con hailo-env activo
hailomz compile yolov3 \
    --ckpt rapid_mwhb1024.onnx \
    --hw-arch hailo8l \
    --calib-path calib/
```

(RAPiD usa Darknet-53 backbone tipo YOLOv3 — alias `yolov3` en el model
zoo es lo más cercano. Validar con tests de inferencia post-compile.)

10-20 min según CPU. Output: `<model>.hef`.

### Deploy a la Pi

```bash
scp <model>.hef pi@<ip>:/tmp/detector.hef

ssh pi@<ip> << 'EOF'
sudo mv /tmp/detector.hef /usr/src/people-counter/models/detector.hef
sudo chown root:root /usr/src/people-counter/models/detector.hef
sudo systemctl restart people-counter
sleep 3
sudo systemctl status people-counter --no-pager | head -20
EOF
```

Update `detection.model_path` en `config.yaml` para apuntar al nuevo HEF.

Smoke test:

```bash
ssh pi@<ip> "sudo journalctl -u people-counter -n 50 --no-pager"
```

Buscar `Loading model: /usr/src/people-counter/models/detector.hef`
sin error subsiguiente, y detecciones periódicas con personas en frame.

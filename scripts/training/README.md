# People Counter — Pipeline de training Phase A

Fine-tune de `yolov8n` para detección de cabezas/personas cenitales,
para que el Hailo-8L vea un modelo realmente entrenado en la geometría
que produce el bracket (frames rectificados top-down) en lugar de caras
frontales tipo CrowdHuman. End-to-end:

```
download_roboflow.py  ->  train en Colab (T4)  ->  export ONNX
        ->  hailomz compile (WSL2)  ->  drop-in del HEF en la Pi
```

Esta carpeta tiene las piezas **del lado del workstation**. Colab hace el
training. WSL2 hace la compilación. La Pi solamente ve el `.hef` final.

Phase A usa **únicamente Roboflow `overhead_person`** (ya viene en
proyección pinhole ≈ a lo que produce tu `rectify_pair`). Si después del
bench la recall no alcanza, Phase B agrega WEPDTOF (fisheye, undistort
con sus propios intrínsecos).

---

## 1. Elegir un dataset en Roboflow Universe

Andá a <https://universe.roboflow.com/> y buscá `overhead person`,
`cenital head` o `top-down people`. Pickeá un dataset donde:

- la cámara esté **montada en el techo, top-down** (NO eye-level — esa
  geometría es la que estamos tratando de evitar),
- 5k+ imágenes,
- una sola clase (`person` o `head`),
- exporte formato YOLOv8.

Abrí el dataset; la URL tiene la forma

```
https://universe.roboflow.com/<workspace>/<project>/<version>
```

Anotate esos tres slugs.

## 2. Descargar el dataset

```bash
# Una vez por workstation
pip install roboflow

# Cargá tu API key (sacala — free — en https://app.roboflow.com)
export ROBOFLOW_API_KEY=xxxxxxxxxxxxxx

# Bajá el dataset
python scripts/training/download_roboflow.py \
    --workspace <ws-slug> --project <proj-slug> --version <N>
```

El output queda en `dataset/roboflow_<ws>_<proj>_v<N>/` con el layout
estándar de YOLOv8 (`data.yaml` + `train/` + `valid/` + `test/`). El
script imprime el conteo de imágenes por split para chequeo de cordura.

## 3. Bench baseline (antes de entrenar)

Cuantificá qué tan mal anda el modelo off-the-shelf en TU escena antes
de re-entrenar. Necesitás ~30-50 frames de la Pi (cualquier carpeta de
`.jpg`s sirve):

```bash
pip install ultralytics
python scripts/training/bench_detector.py bench \
    --weights yolov8n.pt \
    --frames  /path/a/frames_capturados/ \
    --conf 0.25 \
    --report debug/bench_baseline.json
```

El reporte tiene detection-rate + counts por zona (centro / 4 esquinas)
+ stats de confidence. Guardalo para el diff posterior.

**Nota importante:** los frames del Pi se usan **sólo para validación
(inferencia)**, nunca para training. No salen del repo, no se anotan,
no fluyen gradientes hacia el modelo.

## 4. Entrenar en Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maurogasparri/people-counter/blob/main/notebooks/train_yolov8n_heads.ipynb)

Click en el badge de arriba (o abrí
`notebooks/train_yolov8n_heads.ipynb` directamente vía la pestaña
*File → Open notebook → GitHub* de Colab pegando la URL del repo).
Colab pulls el notebook live desde GitHub cada vez que se abre, así
que cualquier commit a `main` es lo que vas a correr.

Después:

1. `Runtime` → `Change runtime type` → `T4 GPU`.
2. Sidebar 🔑 → agregá secret `ROBOFLOW_API_KEY` (así no queda
   hardcoded en el notebook).
3. En cell 4, reemplazá `WORKSPACE` / `PROJECT` / `VERSION` con los
   slugs del paso 1.
4. `Runtime` → `Run all`.

Las ediciones que hagas dentro de Colab no se auto-commitean a GitHub
— si tweakeás el notebook y querés conservar el cambio, usá
*File → Save a copy in GitHub* o copiá el diff de vuelta al repo.

Wall-time en T4: ~2-4 h para ~5k imágenes × 50 epochs. Los
checkpoints se salvan a Drive cada 10 epochs en
`Drive/people-counter-training/runs/`. Si la sesión se desconecta,
re-correr la cell 5 retoma desde el último checkpoint.

Después del training, las cells 7-8 exportan ONNX + un calibration set
de 200 imágenes a `Drive/people-counter-training/export/<run-name>/`.

## 5. Bench post-train

Bajá `best.pt` de Drive al workstation, después:

```bash
python scripts/training/bench_detector.py bench \
    --weights /path/to/best.pt \
    --frames  /path/a/frames_capturados/ \
    --conf 0.25 \
    --report debug/bench_finetuned.json

python scripts/training/bench_detector.py diff \
    debug/bench_baseline.json debug/bench_finetuned.json
```

El diff te muestra detection-rate + deltas zona por zona. Regla de
decisión:

- mejora de `detection_rate` ≥ **+0.20** Y mean confidence sube por
  **≥ +0.10** → shippeamos Phase A. Compilamos a HEF (paso 6) y
  drop-in.
- mejoras menores → corremos Phase B (suma WEPDTOF) antes de compilar.

## 6. Compilar a HEF (WSL2)

La compilación corre solo sobre x86 Linux — la Pi es ARM y Windows no
puede correr el Hailo Dataflow Compiler en nativo. Adentro de WSL2
Ubuntu:

```bash
# One-time (~30 min)
pip install hailo-dataflow-compiler
# Verificación
hailomz --version

# Cada modelo
hailomz compile yolov8n \
    --ckpt /mnt/c/.../export/<run-name>/best.onnx \
    --hw-arch hailo8l \
    --calib-path /mnt/c/.../export/<run-name>/calib/
```

Output: `yolov8n.hef`. SCP a la Pi:

```bash
scp yolov8n.hef pi@<device>:/usr/src/people-counter/models/
sudo systemctl restart people-counter
```

`config.yaml` ya apunta a ese path vía `detection.model_path`.

---

## Mapa de archivos

| Archivo | Phase | Qué hace |
|---|---|---|
| `download_roboflow.py` | A | Pull de un dataset de Roboflow Universe a `dataset/`. |
| `bench_detector.py` | A + post | Benchmark de inferencia + diff de reportes. |
| `../../notebooks/train_yolov8n_heads.ipynb` | A | Notebook de Colab T4 para training. |
| `.env.example` | A | Convención del env-var de la API key (el `.env` real está gitignoreado). |

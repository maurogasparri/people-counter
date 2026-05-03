# People Counter — Pipeline de training Phase A

Fine-tune de `yolov8n` para detección de cabezas/personas cenitales,
para que el Hailo-8L vea un modelo realmente entrenado en la geometría
que produce el bracket (frames rectificados top-down) en lugar de caras
frontales tipo CrowdHuman. End-to-end:

```
download_roboflow.py  ->  train en Kaggle (T4)  ->  export ONNX
        ->  hailomz compile (WSL2)  ->  drop-in del HEF en la Pi
```

Esta carpeta tiene las piezas **del lado del workstation**. Kaggle hace
el training. WSL2 hace la compilación. La Pi solamente ve el `.hef` final.

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

## 4. Entrenar en Kaggle

Kaggle Notebooks Free ofrece T4 con 30 h/semana de GPU sin compute
units que se agoten, y los 16 GB efectivos del T4 sin sharing —
entrena este dataset (~15k imágenes overhead heads) estable a batch=8.

### Setup one-time (5 min)

1. Cuenta en <https://www.kaggle.com> + verificación de teléfono
   (Settings → Phone Verification, requerido para habilitar GPU).

### Cada vez que entrenes

1. `kaggle.com/code` → **+ New Notebook**.
2. Adentro: **File → Import Notebook → URL** y pegá:
   ```
   https://github.com/maurogasparri/people-counter/blob/main/notebooks/train_yolov8n_heads.ipynb
   ```
   (Si la pestaña URL no anda: bajá el `.ipynb` desde GitHub y usá
   **File → Import Notebook → Upload**.)
3. Sidebar derecho → **Settings** → **Accelerator** → **GPU T4 x2**
   (o P100 si T4 no está disponible).
4. Sidebar derecho → **Add-ons** → **Secrets** → **Add a new secret**:
   - Label: `ROBOFLOW_API_KEY`
   - Value: tu API key
   - Toggle **Attached** a este notebook → ON.
5. (Opcional) Editá `WORKSPACE` / `PROJECT` / `VERSION` en la cell 4
   si vas a entrenar otro dataset que no sea el de
   `coding-compass-nmjfb/overhead-head-detection-cwetj v2`.
6. **Run All**.

Wall-time esperado en T4: ~1-2 h para ~12k imágenes train × 50 epochs
con BATCH=8. Kaggle preserva `/kaggle/working/` así que checkpoints,
ONNX y calibration set quedan disponibles para descarga al final.

## 5. Bench post-train

Bajá los outputs desde Kaggle al workstation:

1. Sidebar derecho del notebook → **Output** → botón **Download All**.
2. Te baja un .zip con todo `/kaggle/working/`. Descomprimí y vas a
   tener `runs/<run-name>/weights/best.pt` + `export/<run-name>/best.onnx`
   + `export/<run-name>/calib/`.

Después:

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
| `../../notebooks/train_yolov8n_heads.ipynb` | A | Notebook de Kaggle T4 para training. |
| `../capture_baseline_frames.py` | A | Captura frames rectificados de la Pi para validation bench (no training). |
| `.env.example` | A | Convención del env-var de la API key (el `.env` real está gitignoreado). |

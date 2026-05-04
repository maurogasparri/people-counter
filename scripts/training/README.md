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
6. Esquina superior derecha → **Save Version** → elegí **Save & Run
   All (Commit)** → click **Save**.

Save & Run All corre el notebook en background en infra de Kaggle —
podés cerrar el browser, no requiere mantener la sesión activa, y al
terminar los outputs quedan inmutables como Output de la Version.

Wall-time esperado en T4: ~3 h para ~12k imágenes train × 50 epochs
con BATCH=8.

### Bajar los outputs

Setup one-time del Kaggle CLI:

```powershell
py -m pip install kaggle

# API token: kaggle.com/settings → API → Create New API Token
# Guardalo en %USERPROFILE%\.kaggle\access_token
mkdir $env:USERPROFILE\.kaggle -Force
"<TU-TOKEN>" | Out-File "$env:USERPROFILE\.kaggle\access_token" -Encoding ASCII -NoNewline

# Si el script no está en PATH (caso típico de Windows + py launcher):
$env:Path += ";$env:USERPROFILE\AppData\Local\Programs\Python\Python312\Scripts"
```

Descarga (cuando la Version 1 esté completed):

```powershell
mkdir C:\path\to\repo\debug\kaggle_output -Force
kaggle kernels output <username>/<notebook-slug> -p C:\path\to\repo\debug\kaggle_output
```

Te baja `best.pt`, `best.onnx`, `data.yaml`, `calib/` y el `.log` del
run. Funciona indefinidamente — la Version es inmutable.

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

## 6. Compilar a HEF y deployar

La compilación corre solo sobre x86 Linux — la Pi es ARM y Windows no
puede correr el Hailo Dataflow Compiler en nativo. Vamos a usar **WSL2
Ubuntu** (que ya tenés instalado por Docker Desktop) como entorno de
compilación.

### 6.1. Bajar el output de Kaggle al workstation

1. En la pestaña de Kaggle del notebook, sidebar derecho → **Output**.
2. Botón **Download All** (arriba a la derecha del panel) → te baja un
   `.zip` con todo `/kaggle/working/`.
3. Descomprimí en algún path del workstation, por ejemplo
   `C:\Users\MauroGasparri\source\repos\people-counter\debug\kaggle_output\`.
4. Verificá que adentro de `export/<run-name>/` están:
   - `best.pt` — pesos PyTorch (para bench local con `bench_detector.py`)
   - `best.onnx` — para compilar a HEF
   - `data.yaml` — referencia del dataset
   - `calib/` — ~200 jpgs random del train, calibration set para int8
     quantization.

### 6.2. Setup one-time del Hailo SDK en WSL2

Si todavía no tenés el SDK adentro de WSL2 (es lo más probable, Docker
Desktop no incluye Hailo), hacelo una sola vez:

1. Crear cuenta en <https://hailo.ai/developer-zone/> (gratis,
   verificación por email).
2. **Software Downloads** → descargar:
   - `hailo_dataflow_compiler-X.X.X-py3-none-linux_x86_64.whl`
   - `hailo_model_zoo-X.X.X-py3-none-any.whl`
3. Adentro de WSL2 Ubuntu 22.04 (`wsl -d Ubuntu-22.04`):

   ```bash
   # Sistema base
   sudo apt update
   sudo apt install -y python3.10 python3.10-venv python3-pip \
       libgraphviz-dev graphviz build-essential

   # Virtualenv aislado para evitar conflictos
   python3.10 -m venv ~/hailo-env
   source ~/hailo-env/bin/activate
   pip install --upgrade pip wheel setuptools

   # Instalá los .whl que bajaste (los path son al disco Windows
   # vía /mnt/c/ — ajustá si los moviste a otro lado)
   pip install /mnt/c/Users/MauroGasparri/Downloads/hailo_dataflow_compiler-*.whl
   pip install /mnt/c/Users/MauroGasparri/Downloads/hailo_model_zoo-*.whl

   # Verificación
   hailomz --version
   ```

   Si `hailomz --version` te imprime la versión sin error → SDK listo.
   El virtualenv queda en `~/hailo-env/`; cada vez que abrís WSL2 para
   compilar tenés que correr `source ~/hailo-env/bin/activate` primero.

### 6.3. Compilar el HEF

```bash
# WSL2, dentro del venv
cd /mnt/c/Users/MauroGasparri/source/repos/people-counter/debug/kaggle_output/export/<RUN_NAME>/

hailomz compile yolov8n \
    --ckpt best.onnx \
    --hw-arch hailo8l \
    --calib-path calib/
```

Esto dispara el flujo completo: parse del ONNX → optimización del grafo
→ quantization int8 usando las imágenes de `calib/` → compile final al
binario `.hef`. Tarda **10-20 min** según tu CPU.

Output esperado: `yolov8n.hef` en el cwd, ~5-10 MB.

### 6.4. SCP a la Pi y restart del servicio

```bash
# WSL2 sigue activo
scp yolov8n.hef pi@<ip-de-la-pi>:/tmp/yolov8n.hef

# SSH a la Pi para mover y reiniciar
ssh pi@<ip-de-la-pi> << 'EOF'
sudo mv /tmp/yolov8n.hef /usr/src/people-counter/models/yolov8n.hef
sudo chown root:root /usr/src/people-counter/models/yolov8n.hef
sudo systemctl restart people-counter
sleep 3
sudo systemctl status people-counter --no-pager | head -20
EOF
```

`config.yaml` ya apunta a ese path vía `detection.model_path`, no hace
falta tocar nada del lado de configuración.

### 6.5. Smoke test post-deploy

En la Pi, verificar que el modelo levantó y que detecta:

```bash
# Logs del servicio — buscá la línea de carga del modelo
ssh pi@<ip-de-la-pi> "sudo journalctl -u people-counter -n 50 --no-pager"
```

Lo que buscás:
- ✓ `Loading model: /usr/src/people-counter/models/yolov8n.hef`
  (sin error siguiente)
- ✓ Líneas de detección periódicas si pasás delante de la cámara
- ✗ Cualquier traceback de Hailo o ImportError → algo salió mal en la
  compilación, revisá el log del `hailomz compile`.

Si todo OK, ese HEF queda como production. Si querés volver al stock
yolov8n por cualquier razón (ej. el fine-tune resultó peor en bench),
revertís cambiando `detection.model_path` o restoreando el HEF anterior
desde tu workstation.

---

## Mapa de archivos

| Archivo | Phase | Qué hace |
|---|---|---|
| `download_roboflow.py` | A | Pull de un dataset de Roboflow Universe a `dataset/`. |
| `bench_detector.py` | A + post | Benchmark de inferencia + diff de reportes. |
| `../../notebooks/train_yolov8n_heads.ipynb` | A | Notebook de Kaggle T4 para training. |
| `../capture_baseline_frames.py` | A | Captura frames rectificados de la Pi para validation bench (no training). |
| `.env.example` | A | Convención del env-var de la API key (el `.env` real está gitignoreado). |

# training_data/

Workspace local para todo lo que alimenta el pipeline de training del
detector cenital. Todo el contenido (salvo este README y
`sites.yaml.example`) está gitignored — son datos voluminosos (~1.5GB)
y/o sensibles (IPs de customer + matrices de calibración).

## Layout

```
training_data/
  README.md                          (este archivo, tracked)
  sites.yaml.example                 (tracked — template documentado)
  sites.yaml                         (gitignored — sites reales + calib inline)
  captures/                          (gitignored — frames rectificados, ~1.4GB)
    <site_name>/*.jpg                (un subdir por site, frames sueltos adentro)
    capture.log
    roi_<site>.jpg                   (ROI snapshots per-site, opcional)
  roboflow_uploaded_manifest.txt     (gitignored — los 945 filenames subidos)
  clips/                             (gitignored — opcional, record_clips.py)
  roboflow_sample/                   (gitignored — opcional, sample_for_roboflow.py)
```

## Flujo end-to-end

1. **`sites.yaml`** declara los sites accesibles + sus matrices de
   calibración fisheye inline (K, D, R_rect, P_rect por lente).
   Formato en `sites.yaml.example`.
2. **`scripts/training/capture_mjpeg.py`** levanta un thread por site,
   lee el stream MJPEG HTTP, parte SBS, rectifica con las matrices
   inline y guarda frames a `captures/<site>/`. Defaults apuntan acá,
   no necesita `--config` ni `--output`.
3. **`scripts/training/sample_for_roboflow.py`** estratifica una
   selección de motion + bg per-site y la copia con prefijo numerado.
   Se sube a Roboflow para anotación.
4. **Anotación en Roboflow** con Smart Polygon (AI-Assisted Labeling,
   click-por-imagen) + revisión humana. La versión generada se descarga
   directamente en Kaggle desde `train_head_detector.ipynb` vía signed
   URL — no pasa por la PC local.
5. **`scripts/training/sample_for_calib.py`** arma el calib set para
   el QAT de Hailo. Usa
   `--exclude-manifest training_data/roboflow_uploaded_manifest.txt`
   para evitar leak train/eval.

## `roboflow_uploaded_manifest.txt`

Lista plana con los 945 filenames subidos al proyecto Roboflow
`people-counter-detector`. El formato es
`NNN_<site>__<original_filename>.jpg` que es el rename que aplica
`sample_for_roboflow.py`. El archivo real original está en
`captures/<site>/<original_filename>` — el manifest sirve sólo como
evidencia de qué frames fueron al training set, sin duplicar bytes.

## Migración desde el layout viejo

Si todavía tenés `debug/calib_dumps/<ip>/calib.npz` + `debug/mjpeg_sites.yaml`
de un setup previo, hay un helper one-shot para extraer las matrices
necesarias e inline-las:

```bash
python scripts/training/_embed_calib_into_sites.py \
    --in  debug/mjpeg_sites.yaml \
    --out training_data/sites.yaml
```

Sólo extrae el subset que `capture_mjpeg.py` realmente consume
(`scaled_*_intrinsic_4`, `*_distortion_4`, `scaled_*_R_rect_4`,
`scaled_*_intrinsic_rect_4`). Los `.npz` originales se pueden borrar
una vez que el inline YAML quedó verificado.

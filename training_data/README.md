# training_data/

Workspace local para todo lo que alimenta el pipeline de training del
detector cenital. Todo el contenido (salvo este README y
`sites.yaml.example`) está gitignored — son datos voluminosos (~2GB
con captures + datasets) y/o sensibles (IPs de customer + matrices de
calibración).

## Layout

```
training_data/
  README.md                          (este archivo, tracked)
  sites.yaml.example                 (tracked — template documentado)
  sites.yaml                         (gitignored — sites reales + calib inline)

  captures/                          (gitignored — frames rectificados, ~1.4GB)
    <site_name>/*.jpg                (un subdir por site, frames sueltos)
    capture.log
    counting_zone_<site>.jpg         (counting zone snapshots per-site, opcional)

  # Batches de labeling (X-AnyLabeling local, formato labelme .json)
  label_val_01/                      (gitignored — held-out, 245 imgs)
  label_train_01/                    (gitignored — train base v1, 294 imgs)
  label_al_01/                       (gitignored — active learning del v2, 250 imgs)
  label_bg_01/                       (gitignored — background revisado adicional)

  # Datasets YOLO consolidados (output de labelme_to_yolo.py, sube a Kaggle)
  val_set/                           (gitignored — val held-out en formato YOLO)
  train_set/                         (gitignored — train consolidado del v2 deployed)
  yolo_v1/                           (gitignored — dataset de la primera iter)
  yolo_v1.zip                        (gitignored — versión empaquetada para Kaggle)
  yolo_v2_dataset_deployed/          (gitignored — dataset exacto del modelo deployed)
  yolo_v2_dataset_deployed.zip      (gitignored — versión empaquetada para Kaggle)

  clips/                             (gitignored — opcional, record_clips.py)
```

## Flujo end-to-end

1. **`sites.yaml`** declara los sites accesibles + sus matrices de
   calibración fisheye inline (K, D, R_rect, P_rect por lente).
   Formato en `sites.yaml.example`.
2. **`scripts/training/capture_mjpeg.py`** levanta un thread por site,
   lee el stream MJPEG HTTP, parte SBS, rectifica con las matrices
   inline y guarda frames a `captures/<site>/`. Defaults apuntan acá,
   no necesita `--config` ni `--output`.
3. **`scripts/training/sample_for_labeling.py`** estratifica una
   selección de motion + bg per-site y la copia con prefijo
   `<site>__` a una carpeta plana (`label_<batch>/`). Genera
   `manifest.txt` con el mapeo origen→copia para reincorporación
   downstream. `--exclude-manifest` evita leak train/val.
4. **Labeling en X-AnyLabeling** (local, no SaaS): abrir la carpeta del
   batch, dibujar rectángulos cabeza+hombros (ver
   `scripts/training/label_guide.md` para la convención canónica),
   guardar (genera `.json` formato labelme + `.txt` YOLO local).
5. **`scripts/training/labelme_to_yolo.py`** convierte la carpeta
   labeleada a dataset YOLO con `images/`, `labels/`, `data.yaml`
   listo para Ultralytics. Imgs sin `.json` se tratan como background
   revisado (`.txt` vacío).
6. **Subir a Kaggle dataset privado** vía CLI:
   `kaggle datasets version -p training_data/yolo_v_next ...`.
7. **Notebook Kaggle T4** (`scripts/training/train_head_detector.ipynb`)
   consume el dataset attached y entrena en ~20 min.
8. **`scripts/training/sample_for_calib.py`** arma el calib set para el
   QAT de Hailo. Usa `--exclude-manifest` para evitar leak con el
   training set (los manifests de los `label_*/` cubren esto).

## Active learning para iteraciones (v_next)

Para mejorar el modelo deployed sin labeling al azar:

1. **`scripts/training/mine_active_learning.py`** minar el pool por
   frames informativos (combina disagreement vs un oráculo high-recall
   + uncertainty del modelo actual). Output: `label_v_next_NN/` con
   manifest explicando por qué se eligió cada imagen.
2. Repetir pasos 4-7 de arriba sobre ese batch.

**Histórico**: v1 (294 imgs) → v2 con active learning (+250 imgs,
mAP50 0.80→0.96, deployed) → v3 (2da iter AL +250 imgs, descartada,
rendimientos decrecientes). Tabla completa en
`scripts/training/README.md`.

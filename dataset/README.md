# Datasets de entrenamiento

Esta carpeta está gitignoreada. Acá se dropean los archivos de datasets:

- `wepdtof.zip` — dataset WEPDTOF de Boston VIP COSSY (se pide vía
  https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/).
- `roboflow_overhead_person/` — dataset Roboflow overhead_person ya
  descomprimido (lo baja `scripts/training/download_roboflow.py`).

El pipeline de training en `scripts/training/` lee desde acá. Los paths
son configurables vía flags CLI; los defaults asumen este layout.

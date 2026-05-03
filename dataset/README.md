# Training datasets

This folder is gitignored. Drop dataset archives here:

- `wepdtof.zip` — Boston VIP COSSY WEPDTOF dataset (request via
  https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/).
- `roboflow_overhead_person/` — extracted Roboflow overhead_person dataset
  (downloaded by `scripts/training/download_roboflow.py`).

The training pipeline scripts in `scripts/training/` read from this folder.
Paths are configurable via CLI flags; defaults assume this layout.

# People Counter — Phase A training pipeline

Fine-tune `yolov8n` for cenital head/person detection so the Hailo-8L
sees a model that was actually trained on the geometry your bracket
produces (top-down rectified frames), instead of CrowdHuman frontal
faces. End-to-end the path is:

```
download_roboflow.py  ->  train in Colab (T4)  ->  export ONNX
        ->  hailomz compile (WSL2)  ->  drop-in HEF on the Pi
```

This folder holds the **workstation-side** pieces. Colab does training.
WSL2 does compilation. The Pi only ever sees the final `.hef`.

Phase A uses **only Roboflow `overhead_person`** (already in pinhole
projection ≈ what your `rectify_pair` outputs). If post-bench the recall
isn't enough, Phase B adds WEPDTOF (fisheye, undistorted with their
intrinsics).

---

## 1. Find a dataset on Roboflow Universe

Go to <https://universe.roboflow.com/> and search for `overhead person`,
`cenital head`, or `top-down people`. Pick a dataset where:

- camera is **ceiling-mounted, top-down** (NOT eye-level — that's the
  wrong geometry for our use case),
- 5k+ images,
- single class (`person` or `head`),
- exports YOLOv8 format.

Open the dataset; the URL has the shape

```
https://universe.roboflow.com/<workspace>/<project>/<version>
```

Note those three slugs.

## 2. Download

```bash
# One-time
pip install roboflow

# Set your API key (get one at https://app.roboflow.com — free)
export ROBOFLOW_API_KEY=xxxxxxxxxxxxxx

# Pull the dataset
python scripts/training/download_roboflow.py \
    --workspace <ws-slug> --project <proj-slug> --version <N>
```

Output lands in `dataset/roboflow_<ws>_<proj>_v<N>/` with the standard
YOLOv8 layout (`data.yaml` + `train/` + `valid/` + `test/`). The script
prints a per-split image count so you can sanity-check.

## 3. Baseline bench (before training)

Quantify how badly the off-the-shelf model misses on YOUR scene before
retraining. Use ~50 frames captured from the Pi (any folder of `.jpg`s
will do):

```bash
pip install ultralytics
python scripts/training/bench_detector.py bench \
    --weights yolov8n.pt \
    --frames  /path/to/captured_frames/ \
    --conf 0.25 \
    --report debug/bench_baseline.json
```

The report has detection-rate + per-zone counts (center / 4 corners) +
confidence stats. Keep it for the diff later.

## 4. Train in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maurogasparri/people-counter/blob/main/notebooks/train_yolov8n_heads.ipynb)

Click the badge above (or open `notebooks/train_yolov8n_heads.ipynb`
directly via Colab's *File → Open notebook → GitHub* tab and paste the
repo URL). Colab pulls the notebook live from GitHub on every open, so
any commit to `main` is immediately what you'd run.

Then:

1. `Runtime` → `Change runtime type` → `T4 GPU`.
2. Sidebar 🔑 → add secret `ROBOFLOW_API_KEY` (so it isn't baked into
   the notebook).
3. In cell 4, replace `WORKSPACE` / `PROJECT` / `VERSION` with the
   slugs from step 1.
4. `Runtime` → `Run all`.

Edits you make inside Colab don't auto-commit back to GitHub — if you
tweak the notebook and want to keep the change, use Colab's
*File → Save a copy in GitHub* or copy the diff back into the repo.

Wall-time ~2-4 h on T4. Checkpoints save every 10 epochs to
`Drive/people-counter-training/runs/`. If the session disconnects,
re-running cell 5 picks up from the last checkpoint.

After training, cells 7-8 export ONNX + a 200-image calibration set
to `Drive/people-counter-training/export/<run-name>/`.

## 5. Post-train bench

Download `best.pt` from Drive to your workstation, then:

```bash
python scripts/training/bench_detector.py bench \
    --weights /path/to/best.pt \
    --frames  /path/to/captured_frames/ \
    --conf 0.25 \
    --report debug/bench_finetuned.json

python scripts/training/bench_detector.py diff \
    debug/bench_baseline.json debug/bench_finetuned.json
```

The diff shows detection-rate + zone-by-zone deltas. Decision rule:

- `detection_rate` improvement ≥ **+0.20** AND mean confidence up by
  **≥ +0.10** → ship Phase A. Compile to HEF (step 6) and drop in.
- improvements smaller → run Phase B (add WEPDTOF) before compiling.

## 6. Compile to HEF (WSL2)

Compilation runs on x86 Linux only — the Pi is ARM and Windows can't
run the Hailo Dataflow Compiler natively. Inside WSL2 Ubuntu:

```bash
# One-time (~30 min)
pip install hailo-dataflow-compiler
# Verify
hailomz --version

# Each model
hailomz compile yolov8n \
    --ckpt /mnt/c/.../export/<run-name>/best.onnx \
    --hw-arch hailo8l \
    --calib-path /mnt/c/.../export/<run-name>/calib/
```

Output: `yolov8n.hef`. SCP to the Pi:

```bash
scp yolov8n.hef pi@<device>:/usr/src/people-counter/models/
sudo systemctl restart people-counter
```

`config.yaml` already points at that path via `detection.model_path`.

---

## File map

| File | Phase | What it does |
|---|---|---|
| `download_roboflow.py` | A | Pull a Roboflow Universe dataset to `dataset/`. |
| `bench_detector.py` | A + post | Inference benchmark + report diff. |
| `../../notebooks/train_yolov8n_heads.ipynb` | A | Colab T4 training notebook. |
| `.env.example` | A | API-key env-var convention (gitignored at runtime). |

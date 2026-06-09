#!/usr/bin/env python3
"""Side-by-side visual comparison de dos detectores YOLO sobre nuestras captures.

Toma un sample estratificado de ``training_data/captures/`` (por site y por
motion/bg), corre dos modelos en cada imagen, y genera:

  output_dir/
    pairs/<site>__<basename>.jpg     ← side-by-side rendered (A | B)
    report.html                       ← grid navegable con stats arriba
    stats.json                        ← totales + per-site + conf distribution

Casos de uso típicos:

    # COCO off-the-shelf vs nuestro v1 fine-tuned (baseline para el TFG)
    python scripts/training/compare_detectors.py \\
        --captures training_data/captures \\
        --model-a yolov8n.pt --label-a "COCO YOLOv8n" \\
        --model-b models/training/people-counter-detector/people-counter-detector.pt \\
        --label-b "v1 fine-tuned" \\
        --coco-only-person --n-per-site 14 \\
        --output debug/compare_coco_vs_v1

    # v1 vs v2 (cuando termine el Kaggle nuevo)
    python scripts/training/compare_detectors.py \\
        --captures training_data/captures \\
        --model-a models/training/people-counter-detector/people-counter-detector.pt \\
        --label-a "v1" \\
        --model-b path/to/v2.pt --label-b "v2" \\
        --n-per-site 14 \\
        --output debug/compare_v1_vs_v2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def stratified_sample(
    captures_dir: Path,
    n_per_site: int,
    motion_ratio: float,
    seed: int,
) -> list[Path]:
    """Sample n_per_site JPGs por cada site_*/, con motion_ratio en _motion_."""
    rng = random.Random(seed)
    out: list[Path] = []
    for site_dir in sorted(captures_dir.glob("site_*")):
        jpgs = list(site_dir.glob("*.jpg"))
        motion = [p for p in jpgs if "_motion_" in p.name]
        bg = [p for p in jpgs if "_bg_" in p.name]
        n_motion = int(round(n_per_site * motion_ratio))
        n_bg = n_per_site - n_motion
        sample = rng.sample(motion, min(n_motion, len(motion)))
        sample += rng.sample(bg, min(n_bg, len(bg)))
        out.extend(sample)
    rng.shuffle(out)
    return out


def run_model(
    model: YOLO,
    img_path: Path,
    conf: float,
    only_class_id: int | None,
) -> list[tuple[tuple[float, float, float, float], float, int]]:
    """Devuelve lista de (xyxy_px, conf, cls_id) sobre img_path."""
    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    dets: list[tuple[tuple[float, float, float, float], float, int]] = []
    if res.boxes is None or len(res.boxes) == 0:
        return dets
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    for box, c, cls in zip(xyxy, confs, clss):
        if only_class_id is not None and int(cls) != only_class_id:
            continue
        dets.append(
            (
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                float(c),
                int(cls),
            )
        )
    return dets


def render_pair(
    img_path: Path,
    dets_a: list,
    dets_b: list,
    label_a: str,
    label_b: str,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    out_path: Path,
    max_side: int = 720,
) -> None:
    """Genera un JPG con dos copias de la imagen side-by-side, cajas pintadas."""
    base = Image.open(img_path).convert("RGB")
    w, h = base.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        base = base.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        w, h = base.size
    else:
        scale = 1.0

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_big = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_big = font

    def _paint(target: Image.Image, dets, color, label):
        draw = ImageDraw.Draw(target)
        for (x1, y1, x2, y2), c, _ in dets:
            draw.rectangle(
                [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                outline=color,
                width=3,
            )
            draw.text(
                (x1 * scale + 3, max(0, y1 * scale - 18)),
                f"{c:.2f}",
                fill=color,
                font=font,
            )
        # header
        draw.rectangle([0, 0, w, 32], fill=(0, 0, 0))
        draw.text((8, 6), f"{label} — {len(dets)} det", fill=color, font=font_big)

    left = base.copy()
    right = base.copy()
    _paint(left, dets_a, color_a, label_a)
    _paint(right, dets_b, color_b, label_b)

    combined = Image.new("RGB", (w * 2 + 4, h), (16, 16, 16))
    combined.paste(left, (0, 0))
    combined.paste(right, (w + 4, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path, quality=85)


def conf_buckets(confs: list[float]) -> dict[str, int]:
    buckets = {"<0.30": 0, "0.30-0.50": 0, "0.50-0.70": 0, ">=0.70": 0}
    for c in confs:
        if c < 0.30:
            buckets["<0.30"] += 1
        elif c < 0.50:
            buckets["0.30-0.50"] += 1
        elif c < 0.70:
            buckets["0.50-0.70"] += 1
        else:
            buckets[">=0.70"] += 1
    return buckets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--model-a", type=Path, required=True)
    p.add_argument("--model-b", type=Path, required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--conf-a", type=float, default=0.25)
    p.add_argument("--conf-b", type=float, default=0.25)
    p.add_argument(
        "--coco-only-person",
        action="store_true",
        help="Filtra solo class 0 en el modelo A (asume A = COCO).",
    )
    p.add_argument(
        "--n-per-site",
        type=int,
        default=14,
        help="N de imgs por site_* (~14 × 7 sites = 98).",
    )
    p.add_argument(
        "--motion-ratio",
        type=float,
        default=0.8,
        help="Fracción del sample que viene de frames _motion_ (resto _bg_).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    pairs_dir = args.output / "pairs"

    print(
        f"[sample] estratificando {args.n_per_site} imgs/site × motion {args.motion_ratio}"
    )
    sample = stratified_sample(
        args.captures, args.n_per_site, args.motion_ratio, args.seed
    )
    print(f"[sample] {len(sample)} imgs seleccionadas")

    print(f"[load] modelo A: {args.model_a}")
    model_a = YOLO(str(args.model_a))
    print(f"[load] modelo B: {args.model_b}")
    model_b = YOLO(str(args.model_b))

    only_a = 0 if args.coco_only_person else None

    stats: dict = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "conf_a": args.conf_a,
        "conf_b": args.conf_b,
        "n_samples": len(sample),
        "totals_a": {"dets": 0, "imgs_with_dets": 0, "imgs_zero_dets": 0},
        "totals_b": {"dets": 0, "imgs_with_dets": 0, "imgs_zero_dets": 0},
        "confs_a": [],
        "confs_b": [],
        "per_site_a": defaultdict(lambda: {"dets": 0, "imgs": 0, "imgs_with_dets": 0}),
        "per_site_b": defaultdict(lambda: {"dets": 0, "imgs": 0, "imgs_with_dets": 0}),
        "agreement": {
            "both_zero": 0,
            "a_only_has": 0,
            "b_only_has": 0,
            "both_have": 0,
        },
        "samples": [],
    }

    for i, img_path in enumerate(sample, 1):
        site = img_path.parent.name
        is_motion = "_motion_" in img_path.name
        dets_a = run_model(model_a, img_path, args.conf_a, only_a)
        dets_b = run_model(model_b, img_path, args.conf_b, None)

        stats["totals_a"]["dets"] += len(dets_a)
        stats["totals_b"]["dets"] += len(dets_b)
        if dets_a:
            stats["totals_a"]["imgs_with_dets"] += 1
        else:
            stats["totals_a"]["imgs_zero_dets"] += 1
        if dets_b:
            stats["totals_b"]["imgs_with_dets"] += 1
        else:
            stats["totals_b"]["imgs_zero_dets"] += 1

        stats["confs_a"].extend(c for _, c, _ in dets_a)
        stats["confs_b"].extend(c for _, c, _ in dets_b)

        stats["per_site_a"][site]["imgs"] += 1
        stats["per_site_b"][site]["imgs"] += 1
        stats["per_site_a"][site]["dets"] += len(dets_a)
        stats["per_site_b"][site]["dets"] += len(dets_b)
        if dets_a:
            stats["per_site_a"][site]["imgs_with_dets"] += 1
        if dets_b:
            stats["per_site_b"][site]["imgs_with_dets"] += 1

        if not dets_a and not dets_b:
            stats["agreement"]["both_zero"] += 1
        elif dets_a and not dets_b:
            stats["agreement"]["a_only_has"] += 1
        elif dets_b and not dets_a:
            stats["agreement"]["b_only_has"] += 1
        else:
            stats["agreement"]["both_have"] += 1

        pair_name = f"{site}__{img_path.stem}.jpg"
        pair_path = pairs_dir / pair_name
        render_pair(
            img_path,
            dets_a,
            dets_b,
            args.label_a,
            args.label_b,
            (40, 120, 255),
            (255, 50, 50),
            pair_path,
        )

        stats["samples"].append(
            {
                "site": site,
                "name": img_path.name,
                "is_motion": is_motion,
                "n_a": len(dets_a),
                "n_b": len(dets_b),
                "pair_path": f"pairs/{pair_name}",
            }
        )

        if i % 10 == 0 or i == len(sample):
            print(
                f"[run]   {i}/{len(sample)}  A={stats['totals_a']['dets']}  B={stats['totals_b']['dets']}"
            )

    # Bucketize confs antes de serializar (defaultdict → dict)
    stats["per_site_a"] = dict(stats["per_site_a"])
    stats["per_site_b"] = dict(stats["per_site_b"])
    stats["conf_buckets_a"] = conf_buckets(stats["confs_a"])
    stats["conf_buckets_b"] = conf_buckets(stats["confs_b"])
    stats["conf_mean_a"] = (
        round(mean(stats["confs_a"]), 3) if stats["confs_a"] else None
    )
    stats["conf_mean_b"] = (
        round(mean(stats["confs_b"]), 3) if stats["confs_b"] else None
    )
    stats["conf_median_a"] = (
        round(median(stats["confs_a"]), 3) if stats["confs_a"] else None
    )
    stats["conf_median_b"] = (
        round(median(stats["confs_b"]), 3) if stats["confs_b"] else None
    )

    # Pelar las raw conf lists del JSON pa que no pese tanto
    confs_a = stats.pop("confs_a")
    confs_b = stats.pop("confs_b")

    (args.output / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    write_report(args.output, stats, args.label_a, args.label_b)
    print(f"\n[done] reporte: {args.output / 'report.html'}")
    print_summary(stats, args.label_a, args.label_b)


def print_summary(stats: dict, label_a: str, label_b: str) -> None:
    a = stats["totals_a"]
    b = stats["totals_b"]
    ag = stats["agreement"]
    n = stats["n_samples"]
    print(f"\n=== RESUMEN ({n} imgs) ===")
    print(
        f"{label_a:25s}  dets={a['dets']:4d}  imgs_con_det={a['imgs_with_dets']}/{n}  "
        f"conf_avg={stats['conf_mean_a']}"
    )
    print(
        f"{label_b:25s}  dets={b['dets']:4d}  imgs_con_det={b['imgs_with_dets']}/{n}  "
        f"conf_avg={stats['conf_mean_b']}"
    )
    print(f"\nAgreement:")
    print(f"  ambos detectaron algo:       {ag['both_have']}/{n}")
    print(f"  ambos vacíos:                {ag['both_zero']}/{n}")
    print(f"  solo {label_a} detectó:       {ag['a_only_has']}/{n}")
    print(f"  solo {label_b} detectó:       {ag['b_only_has']}/{n}")


def write_report(output: Path, stats: dict, label_a: str, label_b: str) -> None:
    rows = "\n".join(
        f'<figure data-site="{s["site"]}"><img src="{s["pair_path"]}" loading="lazy"/>'
        f'<figcaption>{s["site"]} · {s["name"]} · '
        f'<b style="color:#4080ff">{label_a}:{s["n_a"]}</b> · '
        f'<b style="color:#ff3232">{label_b}:{s["n_b"]}</b>'
        f"</figcaption></figure>"
        for s in stats["samples"]
    )
    a = stats["totals_a"]
    b = stats["totals_b"]
    ag = stats["agreement"]
    n = stats["n_samples"]

    def _bucket_row(buckets: dict) -> str:
        return " · ".join(f"{k}:{v}" for k, v in buckets.items())

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>compare {label_a} vs {label_b}</title>
<style>
 body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
 h1, h2 {{ font-size: 18px; margin: 12px 0; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 td, th {{ padding: 6px 10px; border-bottom: 1px solid #333; text-align: left; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(540px, 1fr)); gap: 10px; margin-top: 16px; }}
 figure {{ margin: 0; background: #222; padding: 6px; border-radius: 8px; }}
 figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
 figcaption {{ font-size: 11px; color: #aaa; padding-top: 4px; word-break: break-all; }}
 .blue {{ color: #4080ff; }} .red {{ color: #ff3232; }}
</style>

<h1>Comparación: <span class="blue">{label_a}</span> vs <span class="red">{label_b}</span></h1>
<p>{n} imágenes, sample estratificado por site (motion-heavy).</p>

<table>
 <tr><th>Modelo</th><th>Total dets</th><th>Imgs con det</th><th>Conf media</th><th>Conf mediana</th><th>Buckets</th></tr>
 <tr><td class="blue">{label_a}</td><td>{a['dets']}</td><td>{a['imgs_with_dets']}/{n}</td>
     <td>{stats['conf_mean_a']}</td><td>{stats['conf_median_a']}</td>
     <td>{_bucket_row(stats['conf_buckets_a'])}</td></tr>
 <tr><td class="red">{label_b}</td><td>{b['dets']}</td><td>{b['imgs_with_dets']}/{n}</td>
     <td>{stats['conf_mean_b']}</td><td>{stats['conf_median_b']}</td>
     <td>{_bucket_row(stats['conf_buckets_b'])}</td></tr>
</table>

<h2>Agreement</h2>
<table>
 <tr><th>Caso</th><th>Cuenta</th></tr>
 <tr><td>Ambos detectaron algo</td><td>{ag['both_have']}/{n}</td></tr>
 <tr><td>Ambos vacíos</td><td>{ag['both_zero']}/{n}</td></tr>
 <tr><td>Solo {label_a}</td><td>{ag['a_only_has']}/{n}</td></tr>
 <tr><td>Solo {label_b}</td><td>{ag['b_only_has']}/{n}</td></tr>
</table>

<h2>Pares (azul = {label_a}, rojo = {label_b})</h2>
<div class="grid">{rows}</div>
"""
    (output / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()

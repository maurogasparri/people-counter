"""Bucketea las detecciones de eval_yolo por site y origin (motion/bg).

Lee el ``summary.txt`` que produce ``scripts/training/eval_yolo.py`` y
emite una tabla por (site, kind=motion|bg) con counts, % con detección,
detecciones totales y stats de confidence. Sirve para auditar bias por
site y estimar la proyección post-screening del próximo training set.

Uso:
    python scripts/training/analyze_eval_summary.py <path-al-summary.txt>
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("Uso: analyze_eval_summary.py <summary.txt>")
summary = Path(sys.argv[1]).read_text(encoding="utf-8")
line_re = re.compile(r"\s*(\S+\.jpg):\s+(\d+)\s+detects\s+\[([^\]]*)\]")
fn_re = re.compile(r"^\d+_(site_\d+_\d+)__.*?(_motion_|_bg_)")

stats = defaultdict(lambda: {"n": 0, "with_det": 0, "detects": 0, "confs": []})

for line in summary.splitlines():
    m = line_re.match(line)
    if not m:
        continue
    fn, n_det, confs_str = m.group(1), int(m.group(2)), m.group(3)
    fm = fn_re.match(fn)
    if not fm:
        continue
    site, kind = fm.group(1), fm.group(2).strip("_")
    key = (site, kind)
    s = stats[key]
    s["n"] += 1
    if n_det > 0:
        s["with_det"] += 1
    s["detects"] += n_det
    if confs_str.strip():
        s["confs"].extend(float(c) for c in confs_str.split(","))

hdr = f"{'site':<14} {'kind':<6} {'n':>4} {'with':>5} {'%det':>6} {'dets':>5} {'avg':>6} {'med_c':>7} {'max_c':>7}"
print(hdr)
print("-" * len(hdr))

agg = {"motion": [0, 0, 0, []], "bg": [0, 0, 0, []]}
for (site, kind), s in sorted(stats.items()):
    pct = 100.0 * s["with_det"] / s["n"] if s["n"] else 0
    avg = s["detects"] / s["n"] if s["n"] else 0
    med = sorted(s["confs"])[len(s["confs"]) // 2] if s["confs"] else 0
    mx = max(s["confs"]) if s["confs"] else 0
    print(
        f"{site:<14} {kind:<6} {s['n']:>4} {s['with_det']:>5} "
        f"{pct:>5.1f}% {s['detects']:>5} {avg:>6.2f} {med:>7.3f} {mx:>7.3f}"
    )
    a = agg[kind]
    a[0] += s["n"]
    a[1] += s["with_det"]
    a[2] += s["detects"]
    a[3].extend(s["confs"])

print("-" * len(hdr))
for kind in ("motion", "bg"):
    n, wd, dt, cf = agg[kind]
    pct = 100.0 * wd / n if n else 0
    avg = dt / n if n else 0
    med = sorted(cf)[len(cf) // 2] if cf else 0
    mx = max(cf) if cf else 0
    print(
        f"{'TOTAL':<14} {kind:<6} {n:>4} {wd:>5} {pct:>5.1f}% "
        f"{dt:>5} {avg:>6.2f} {med:>7.3f} {mx:>7.3f}"
    )

bg_confs = sorted(agg["bg"][3], reverse=True)
print("\nBg con detección — distribución de max-conf por imagen:")
print(f"  total bg con det: {agg['bg'][1]}")
buckets = [
    ("conf >= 0.80", sum(1 for c in bg_confs if c >= 0.80)),
    ("conf 0.50-0.80", sum(1 for c in bg_confs if 0.50 <= c < 0.80)),
    ("conf 0.25-0.50", sum(1 for c in bg_confs if 0.25 <= c < 0.50)),
]
for label, n in buckets:
    pct_of_bg = 100.0 * n / agg["bg"][0] if agg["bg"][0] else 0
    print(f"  {label}: {n}  ({pct_of_bg:.1f}% del bg total)")

# Estimación post-screening (asumiendo conf>=0.50 = persona real)
motion_pos = sum(1 for c in sorted(agg["motion"][3], reverse=True) if c >= 0.50)
# motion-positivos cuenta detecciones, no imgs — para pasar a imgs uso
# proxy: contar imgs de motion con AT LEAST UNA detección >=0.50.
motion_imgs_with_pos = 0
bg_imgs_with_pos = 0
for (site, kind), s in stats.items():
    high_confs = [c for c in s["confs"] if c >= 0.50]
    n_imgs_with_high = sum(
        1 for line in summary.splitlines()
        if (m := line_re.match(line))
        and (fm := fn_re.match(m.group(1)))
        and fm.group(1) == site
        and fm.group(2).strip("_") == kind
        and any(float(c) >= 0.50 for c in m.group(3).split(",") if c.strip())
    )
    if kind == "motion":
        motion_imgs_with_pos += n_imgs_with_high
    else:
        bg_imgs_with_pos += n_imgs_with_high

print("\nProyección post-screening (conf>=0.50 como cutoff de persona real):")
print(f"  Motion → positivos esperados: {motion_imgs_with_pos} de {agg['motion'][0]} ({100.0*motion_imgs_with_pos/agg['motion'][0]:.1f}%)")
print(f"  Bg con persona (promoción): {bg_imgs_with_pos} de {agg['bg'][0]} ({100.0*bg_imgs_with_pos/agg['bg'][0]:.1f}%)")
print(f"  Bg limpios (negativos finales): {agg['bg'][0] - bg_imgs_with_pos}")
print(f"  Total positivos v3 (motion + bg-promoted): {motion_imgs_with_pos + bg_imgs_with_pos}")
print(f"  Total negativos v3: {agg['bg'][0] - bg_imgs_with_pos}")
total_pos = motion_imgs_with_pos + bg_imgs_with_pos
total_neg = agg['bg'][0] - bg_imgs_with_pos
ratio = total_pos / total_neg if total_neg else 0
print(f"  Ratio positivos:negativos = {ratio:.2f}:1")

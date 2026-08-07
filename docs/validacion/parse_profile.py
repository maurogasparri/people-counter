#!/usr/bin/env python3
"""Parsea PROFILE lines del journal -> per-stage stats + CSV por frame.

Uso: py docs/validacion/parse_profile.py <profile.log> [label]
Segmenta por det=Y (con deteccion -> SGBM corre) vs det=N (escena vacia).
"""
import csv
import re
import statistics as st
import sys

PAT = re.compile(
    r"frame=(\d+) cap=(\d+)ms rect=(\d+)ms detect=(\d+)ms depth=(\d+)ms "
    r"\(det=(\w+) mode=(\w+)\) track=(\d+)ms n_tracks=(\d+) n_dets=(\d+) "
    r"TOTAL=(\d+)ms \(([\d.]+) FPS\)"
)
TS = re.compile(r'"time":"([\d\- :,]+)"')


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    rows = []
    t_first = t_last = None
    for line in open(path):
        m = PAT.search(line)
        if not m:
            continue
        tm = TS.search(line)
        if tm:
            t_last = tm.group(1)
            if t_first is None:
                t_first = tm.group(1)
        (frame, cap, rect, detect, depth, det, mode, track,
         ntr, ndet, total, fps) = m.groups()
        rows.append(dict(frame=int(frame), cap=int(cap), rect=int(rect),
                         detect=int(detect), depth=int(depth), det=det,
                         mode=mode, track=int(track), n_tracks=int(ntr),
                         n_dets=int(ndet), total=int(total), fps=float(fps)))

    out_csv = path.replace(".log", "_perframe.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def report(subset, title):
        if not subset:
            print(f"\n  [{title}] sin frames")
            return
        n = len(subset)
        print(f"\n  [{title}] n={n} frames")
        stages = ["cap", "rect", "detect", "depth", "track", "total"]
        tot_mean = st.mean([r["total"] for r in subset])
        print(f"    {'stage':<8} {'mean':>6} {'p50':>5} {'p95':>5} {'%frame':>7}")
        for sgi in stages:
            xs = [r[sgi] for r in subset]
            mean = st.mean(xs)
            share = 100 * mean / tot_mean if tot_mean and sgi != "total" else (
                100.0 if sgi == "total" else 0)
            print(f"    {sgi:<8} {mean:>6.1f} {pct(xs, .5):>5.0f} "
                  f"{pct(xs, .95):>5.0f} {share:>6.1f}%")
        fpsl = [r["fps"] for r in subset]
        print(f"    FPS (compute-bound 1000/TOTAL): mean={st.mean(fpsl):.1f} "
              f"p50={pct(fpsl, .5):.1f} min={min(fpsl):.1f}")

    print(f"=== A5 PROFILE [{label}] ===")
    print(f"  ventana: {t_first} -> {t_last}")
    # effective fps = frames / wall seconds
    if t_first and t_last:
        from datetime import datetime
        f = "%Y-%m-%d %H:%M:%S,%f"
        dt = (datetime.strptime(t_last, f) - datetime.strptime(t_first, f)).total_seconds()
        if dt > 0:
            print(f"  FPS efectivo (sostenido) = {len(rows)}/{dt:.0f}s = "
                  f"{len(rows) / dt:.1f} FPS")
    report([r for r in rows if r["det"] == "N"], "escena vacia (det=N, SGBM skip)")
    report([r for r in rows if r["det"] == "Y"], "con deteccion (det=Y, SGBM corre)")
    print(f"\n  CSV por frame -> {out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analiza los CSV crudos del banco (soak de sistema + power) -> stats del reporte.

Uso: py docs/validacion/analyze_bench.py
"""
import csv
import statistics as st

D = "docs/validacion/"


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(rows, col, cast=float):
    out = []
    for r in rows:
        v = r.get(col, "")
        try:
            out.append(cast(v))
        except (ValueError, TypeError):
            pass
    return out


# ---------- SOAK ----------
soak = load(D + "soak_system.csv")
print(f"=== A3/A4 SOAK ({len(soak)} muestras) ===")
t0, t1 = int(soak[0]["epoch"]), int(soak[-1]["epoch"])
print(f"  duracion: {(t1 - t0)} s  ({(t1 - t0) / 60:.1f} min)")

temp = num(soak, "temp_c")
print(f"  temp_c: min={min(temp):.1f} avg={st.mean(temp):.1f} "
      f"p95={pct(temp, .95):.1f} max={max(temp):.1f}")

thr = [r["throttled"] for r in soak]
bad = [x for x in thr if x not in ("0x0", "0X0")]
print(f"  throttled!=0x0: {len(bad)} muestras  {'(SANO)' if not bad else bad[:3]}")

arm = num(soak, "arm_hz")
print(f"  arm_mhz: min={min(arm) / 1e6:.0f} max={max(arm) / 1e6:.0f} "
      f"(downclock idle del governor)")

cur = [x / 2**20 for x in num(soak, "svc_mem_cur_b")]
peak = [x / 2**20 for x in num(soak, "svc_mem_peak_b")]
print(f"  svc mem.current MiB: min={min(cur):.0f} avg={st.mean(cur):.0f} "
      f"max={max(cur):.0f}")
print(f"  svc mem.peak  MiB: max={max(peak):.0f}")

for ev in ("cg_high", "cg_max", "cg_oom", "cg_oomkill"):
    vs = num(soak, ev, int)
    print(f"  {ev}: max={max(vs) if vs else 'n/a'}")

avail = [x / 2**20 for x in num(soak, "mem_avail_b")]
print(f"  sistema MemAvailable MiB: min={min(avail):.0f} (headroom)")
swfree = num(soak, "swap_free_b")
swtot = num(soak, "swap_total_b")
swused = [(t - f) / 2**20 for t, f in zip(swtot, swfree)]
print(f"  swap usado MiB: max={max(swused):.1f} (0 = sin swapping)")

# CPU busy% de deltas de jiffies
tot = num(soak, "cpu_total_j", int)
idl = num(soak, "cpu_idle_j", int)
busy = []
for i in range(1, len(tot)):
    dt = tot[i] - tot[i - 1]
    di = idl[i] - idl[i - 1]
    if dt > 0:
        busy.append(100 * (dt - di) / dt)
print(f"  CPU busy% (agregado 4 cores): avg={st.mean(busy):.1f} "
      f"p95={pct(busy, .95):.1f} max={max(busy):.1f}")
l1 = num(soak, "load1")
print(f"  loadavg(1m): avg={st.mean(l1):.2f} max={max(l1):.2f}")

# 2GB headroom: working set del servicio
print(f"  -> working set svc (peak {max(peak):.0f} MiB) vs target flota 2048 MiB"
      f" = {2048 - max(peak):.0f} MiB de margen ({max(peak) / 2048 * 100:.0f}% usado)")

# ---------- POWER ----------
pw = load(D + "power_session.csv")
w = num(pw, "total_w")
print(f"\n=== A1(b) POWER running, escena vacia/estatica ({len(w)} muestras) ===")
print(f"  total_w: avg={st.mean(w):.2f} p50={pct(w, .5):.2f} "
      f"p95={pct(w, .95):.2f} max={max(w):.2f} min={min(w):.2f}")
print(f"  draw pared estimado (+13%): avg={st.mean(w) * 1.13:.2f} W  "
      f"p95={pct(w, .95) * 1.13:.2f} W")
budget = 25.5
print(f"  % presupuesto PoE HAT (25.5W): avg={st.mean(w) / budget * 100:.1f}% "
      f"p95={pct(w, .95) / budget * 100:.1f}%")
# rieles dominantes (promedio)
for rail in ("VDD_CORE", "3V3_SYS", "1V8_SYS", "1V1_SYS", "0V8_SW", "3V7_WL_SW"):
    vs = num(pw, rail)
    if vs:
        print(f"    riel {rail:<10} avg={st.mean(vs):.3f} W")
print(f"\n  proyeccion energia: avg {st.mean(w):.2f}W x 12h x 363d = "
      f"{st.mean(w) * 12 * 363 / 1000:.1f} kWh/anio (output-side); "
      f"pared ~{st.mean(w) * 1.13 * 12 * 363 / 1000:.1f} kWh/anio")

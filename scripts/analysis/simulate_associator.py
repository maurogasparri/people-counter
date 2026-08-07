#!/usr/bin/env python3
"""Banco de simulación del asociador con trayectorias sintéticas.

Ejercita el par ``EuclideanTracker`` + ``Counter`` con trayectorias de verdad
conocida, sin cámara, sin Hailo y sin profundidad estéreo. El objetivo es
caracterizar el comportamiento del asociador ante cruces simultáneos en
sentidos opuestos — el modo de falla documentado como TC-03.

Responde dos preguntas:

**A. ¿El descarte por ambigüedad tiene sesgo de dirección?**  Se construyen
escenarios *exactamente* simétricos por reflexión puntual respecto del centro
de la counting zone. Bajo esa reflexión la trayectoria de ingreso se aplica
sobre la de egreso y viceversa, de modo que cualquier diferencia de resultado
entre sentidos sólo puede provenir del algoritmo. Se aplican dos pruebas:

  1. *Prueba determinista de simetría*: cada ensayo se corre junto con su
     reflejo exacto (misma realización de ruido, reflejada). Un algoritmo
     simétrico en dirección debe producir resultados espejados. Cualquier
     discrepancia es evidencia directa de asimetría, sin estadística de por
     medio.
  2. *Prueba estadística pareada*: sobre ensayos con ruido independiente se
     comparan las proporciones de pérdida por sentido con la prueba exacta de
     McNemar (los dos sentidos ocurren en el mismo ensayo → datos pareados) y
     un intervalo de confianza bootstrap para la diferencia.

**B. ¿Dónde empieza a rechazar el ratio-test?**  Barrido sobre separación
entre personas, velocidad y dirección relativa, reportando la fracción de
cruces de verdad conocida que no se contabilizan y el número de rechazos por
ambigüedad del asociador en cada punto.

---------------------------------------------------------------------------
MODELO DE RUIDO DE DETECCIÓN — declaración explícita
---------------------------------------------------------------------------

Las detecciones sintéticas se perturban con cuatro fuentes independientes:

  1. Pérdida de detección: Bernoulli(``--p-miss``) por persona y por frame,
     independiente entre frames y entre personas.
  2. Jitter de posición: gaussiano de desvío ``--pos-sigma-px`` sobre x e y,
     independiente.
  3. Ruido de profundidad: gaussiano de desvío ``--depth-sigma-mm`` sobre z.
  4. Confianza: gaussiana ``--conf-mu`` / ``--conf-sd`` recortada a [0, 1],
     enrutada a los mismos tramos que usa el runtime (alta confianza puede
     generar track nuevo; confianza baja sólo re-asocia; por debajo del piso
     se descarta).

**Este modelo NO está calibrado empíricamente.** No existe video ni secuencia
estéreo grabada del prototipo con la que ajustar sus parámetros, de modo que
las tasas y desvíos son supuestos del autor, no mediciones. En particular el
modelo omite, por construcción, la dependencia entre la separación de las
personas y la calidad de la detección: un detector real funde cajas y pierde
cabezas con más frecuencia cuando dos personas se solapan, y acá la pérdida
es independiente de la distancia. Esa omisión hace que los resultados de este
banco describan **la lógica de asociación**, no el desempeño del sistema
completo. Los valores absolutos no son extrapolables a una puerta real; lo
que sí es interpretable es la *comparación entre sentidos* bajo condiciones
idénticas, que es el objetivo A.

Uso:

    python scripts/analysis/simulate_associator.py --trials 400
    python scripts/analysis/simulate_associator.py --sweep
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tracking.counter import build_counter  # noqa: E402
from src.tracking.tracker import EuclideanTracker  # noqa: E402

# --- Geometría del dispositivo desplegado (config.yaml, 2026-06-27) ---------
DEFAULT_ZONE = {"x_min": 246.0, "x_max": 906.0, "y_min": 204.0, "y_max": 444.0}
DEFAULT_LINE = {
    "from": [246, 324],
    "to": [906, 324],
    "labels": {"top_to_bottom": "ingress", "bottom_to_top": "egress"},
}
DEFAULT_COUNTER_CFG = {
    "lines": [DEFAULT_LINE],
    "counting_zone": DEFAULT_ZONE,
    "min_crossing_movement_px": 0.0,
    "min_visit_range_for_death_emit": 80.0,
    "min_count_height_m": 1.0,
    "min_real_inside_frames": 2,
    "min_count_confidence": 0.6,
}

INGRESS = "ingress"
EGRESS = "egress"


@dataclass
class NoiseDraw:
    """Realización de ruido para una persona a lo largo de un ensayo."""

    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    conf: np.ndarray
    miss: np.ndarray

    def reflected(self) -> "NoiseDraw":
        """Refleja el ruido en el plano imagen (z, confianza y pérdidas intactas)."""
        return NoiseDraw(
            dx=-self.dx, dy=-self.dy, dz=self.dz, conf=self.conf, miss=self.miss
        )


@dataclass
class SimParams:
    separation_px: float = 60.0
    speed_px_frame: float = 12.0
    travel_px: float = 220.0
    depth_mm: float = 1800.0
    head_height_mm: float = 1700.0
    bbox_px: float = 70.0
    p_miss: float = 0.10
    pos_sigma_px: float = 6.0
    depth_sigma_mm: float = 60.0
    conf_mu: float = 0.75
    conf_sd: float = 0.12
    # Fusión por centroide previa al tracker (``detection.cluster_distance_px``
    # del runtime). Dos personas más próximas que este umbral colapsan en una
    # sola detección: es el mecanismo por el que un cruce simultáneo puede
    # perderse antes de que el asociador intervenga.
    cluster_distance_px: float = 150.0
    same_direction: bool = False
    settle_frames: int = 80


@dataclass
class TrialResult:
    ingress_counted: bool
    egress_counted: bool
    n_events: int
    labels: list[str] = field(default_factory=list)
    ambiguous_rejects: int = 0
    ghost_adoptions: int = 0

    def lost(self, direction: str) -> bool:
        return not (
            self.ingress_counted if direction == INGRESS else self.egress_counted
        )

    def counted(self, direction: str) -> int:
        """Cantidad de eventos emitidos con esa etiqueta (para el control de
        mismo sentido, donde la verdad conocida son DOS cruces del mismo
        signo y no uno de cada uno)."""
        return self.labels.count(direction)


def n_steps_for(p: SimParams) -> int:
    """Cantidad de frames del ensayo.

    La grilla se construye simétrica alrededor del cruce (offsets
    ``-m·v … 0 … +m·v``). Es un requisito de la prueba de simetría: si el
    conjunto de offsets no fuera invariante bajo negación, las dos personas
    quedarían muestreadas en fases distintas respecto de la línea y eso
    introduciría por sí solo una diferencia entre sentidos que nada tiene que
    ver con el algoritmo.
    """
    m = int(math.ceil(p.travel_px / p.speed_px_frame))
    return 2 * m + 1


def draw_noise(rng: np.random.Generator, n: int, p: SimParams) -> NoiseDraw:
    return NoiseDraw(
        dx=rng.normal(0.0, p.pos_sigma_px, n),
        dy=rng.normal(0.0, p.pos_sigma_px, n),
        dz=rng.normal(0.0, p.depth_sigma_mm, n),
        conf=np.clip(rng.normal(p.conf_mu, p.conf_sd, n), 0.0, 1.0),
        miss=rng.random(n) < p.p_miss,
    )


def build_stack(
    counter_cfg: dict[str, Any], tracking_cfg: dict[str, Any]
) -> tuple[EuclideanTracker, Any]:
    """Arma tracker + counter con el mismo cableado que ``src/main.py``."""
    sm = tracking_cfg["state_machine"]
    tracker = EuclideanTracker(
        max_disappeared=int(tracking_cfg["max_disappeared_frames"]),
        max_distance=float(tracking_cfg["max_distance_px"]),
        max_depth_delta=float(sm["depth_gate_m"]) * 1000.0,
        confirm_frames=int(sm["confirm_frames"]),
        pending_max_frames=int(sm["pending_max_frames"]),
        reid_gate_px=float(sm["reid_gate_px"]),
        pending_velocity_decay=float(sm.get("pending_velocity_decay", 0.5)),
        pending_grace_frames=int(sm.get("pending_grace_frames", 0)),
        ambiguous_match_ratio=float(sm.get("ambiguous_match_ratio", 1.0)),
        keepalive_max_frames=int(sm.get("keepalive_max_frames", 600)),
        adoption_window_frames=int(sm.get("adoption_window_frames", 30)),
        adoption_iou_min=float(sm.get("adoption_iou_min", 0.3)),
        adoption_max_dist_px=float(sm.get("adoption_max_dist_px", 100.0)),
        ghost_outside_invalidate_px=float(sm.get("ghost_outside_invalidate_px", 150.0)),
    )
    counter = build_counter({"counter": counter_cfg})
    # Mismo cableado que producción (src/main.py).
    tracker.keepalive_counting_zone = counter.counting_zone
    counter.death_emit_grace_frames = tracker.adoption_window_frames + 2
    return tracker, counter


def run_trial(
    counter_cfg: dict[str, Any],
    tracking_cfg: dict[str, Any],
    detection_cfg: dict[str, float],
    p: SimParams,
    noise_ingress: NoiseDraw,
    noise_egress: NoiseDraw,
) -> TrialResult:
    """Corre un ensayo de dos personas y devuelve qué sentidos se contabilizaron."""
    zone = counter_cfg["counting_zone"]
    cx = (float(zone["x_min"]) + float(zone["x_max"])) / 2.0
    cy = (float(zone["y_min"]) + float(zone["y_max"])) / 2.0

    tracker, counter = build_stack(counter_cfg, tracking_cfg)

    n_steps = n_steps_for(p)
    m_half = (n_steps - 1) // 2
    half = p.separation_px / 2.0
    new_thr = float(detection_cfg["new_track_threshold"])
    low_thr = float(detection_cfg["low_confidence_threshold"])

    labels: list[str] = []

    for k in range(n_steps):
        offset = (k - m_half) * p.speed_px_frame
        # Ingreso: arriba → abajo. Egreso: abajo → arriba (reflexión puntual).
        people = [
            (INGRESS, cx - half, cy + offset, noise_ingress),
            (
                EGRESS,
                cx + half,
                (cy + offset) if p.same_direction else (cy - offset),
                noise_egress,
            ),
        ]

        # 1. Materializar las detecciones del frame.
        dets: list[tuple[float, float, float, float]] = []  # x, y, z, conf
        for _role, bx, by, nz in people:
            if nz.miss[k]:
                continue
            conf = float(nz.conf[k])
            if conf < low_thr:
                continue
            dets.append((bx + nz.dx[k], by + nz.dy[k], p.depth_mm + nz.dz[k], conf))

        # 2. Clustering por centroide, igual que el runtime: se conserva la
        #    detección de mayor confianza de cada grupo y se absorben las que
        #    caen dentro del umbral. Es el mecanismo por el cual dos personas
        #    próximas colapsan en una sola detección antes de llegar al tracker.
        if p.cluster_distance_px > 0 and len(dets) > 1:
            thr_sq = p.cluster_distance_px**2
            kept: list[tuple[float, float, float, float]] = []
            for d in sorted(dets, key=lambda t: -t[3]):
                if any(
                    (d[0] - k2[0]) ** 2 + (d[1] - k2[1]) ** 2 < thr_sq for k2 in kept
                ):
                    continue
                kept.append(d)
            dets = kept

        # 3. Reparto por tramo de confianza, igual que el runtime.
        high_pos: list[np.ndarray] = []
        high_meta: list[dict[str, Any]] = []
        low_pos: list[np.ndarray] = []
        low_meta: list[dict[str, Any]] = []
        h = p.bbox_px / 2.0
        for x, y, zz, conf in dets:
            meta = {
                "confidence": conf,
                "near_depth_mm": zz,
                "head_height_mm": p.head_height_mm,
                "bbox": (int(x - h), int(y - h), int(x + h), int(y + h)),
            }
            if conf >= new_thr:
                high_pos.append(np.array([x, y, zz]))
                high_meta.append(meta)
            else:
                low_pos.append(np.array([x, y, zz]))
                low_meta.append(meta)

        tracks = tracker.update(
            high_pos,
            detection_metas=high_meta,
            candidate_positions=low_pos or None,
            candidate_metadata=low_meta or None,
        )
        labels += [e.direction for e in counter.check_all(tracks)]

    # Frames sin detección: dejan morir los tracks y disparan el death-emit
    # diferido (grace = adoption_window + 2).
    for _ in range(p.settle_frames):
        tracks = tracker.update([], detection_metas=[])
        labels += [e.direction for e in counter.check_all(tracks)]

    return TrialResult(
        ingress_counted=labels.count(INGRESS) >= 1,
        egress_counted=labels.count(EGRESS) >= 1,
        n_events=len(labels),
        labels=labels,
        ambiguous_rejects=tracker.ambiguous_reject_count,
        ghost_adoptions=tracker.adoption_count,
    )


# --- Estadística ------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor exacto bilateral de McNemar sobre los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb

    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def bootstrap_diff_ci(
    losses: list[tuple[bool, bool]], rng: np.random.Generator, reps: int = 5000
) -> tuple[float, float]:
    """IC percentil 95 % para (pérdida ingreso − pérdida egreso), pareado."""
    n = len(losses)
    if n == 0:
        return (float("nan"), float("nan"))
    arr = np.array(losses, dtype=float)
    idx = rng.integers(0, n, size=(reps, n))
    diffs = arr[idx, 0].mean(axis=1) - arr[idx, 1].mean(axis=1)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


# --- Programa ---------------------------------------------------------------


def load_cfgs(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    det = cfg.get("detection", {})
    return cfg["tracking"], {
        "new_track_threshold": float(det.get("new_track_threshold", 0.35)),
        "low_confidence_threshold": float(det.get("low_confidence_threshold", 0.10)),
        "cluster_distance_px": float(det.get("cluster_distance_px", 150.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=Path("config/config.example.yaml"))
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--symmetry-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--separation-px", type=float, default=60.0)
    ap.add_argument("--speed-px-frame", type=float, default=12.0)
    ap.add_argument("--p-miss", type=float, default=0.10)
    ap.add_argument("--pos-sigma-px", type=float, default=6.0)
    ap.add_argument("--depth-sigma-mm", type=float, default=60.0)
    ap.add_argument("--conf-mu", type=float, default=0.75)
    ap.add_argument("--conf-sd", type=float, default=0.12)
    ap.add_argument(
        "--cluster-distance-px",
        type=float,
        default=None,
        help="Umbral de fusión por centroide previo al tracker. Por defecto "
        "toma detection.cluster_distance_px del config. 0 lo desactiva.",
    )
    ap.add_argument("--sweep", action="store_true", help="Barrido de la superficie")
    ap.add_argument(
        "--sweep-trials", type=int, default=120, help="Ensayos por punto del barrido"
    )
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tracking_cfg, detection_cfg = load_cfgs(args.config)
    counter_cfg = DEFAULT_COUNTER_CFG

    cluster_px = (
        args.cluster_distance_px
        if args.cluster_distance_px is not None
        else detection_cfg["cluster_distance_px"]
    )
    base = SimParams(
        separation_px=args.separation_px,
        speed_px_frame=args.speed_px_frame,
        p_miss=args.p_miss,
        pos_sigma_px=args.pos_sigma_px,
        depth_sigma_mm=args.depth_sigma_mm,
        conf_mu=args.conf_mu,
        conf_sd=args.conf_sd,
        cluster_distance_px=cluster_px,
    )
    n_steps = n_steps_for(base)

    print("# Banco de simulación del asociador\n")
    print("## Configuración\n")
    print(f"- Parámetros de tracking: `{args.config}`")
    print(
        f"- `ambiguous_match_ratio` = "
        f"{tracking_cfg['state_machine']['ambiguous_match_ratio']} · "
        f"`max_distance_px` = {tracking_cfg['max_distance_px']} · "
        f"`depth_gate_m` = {tracking_cfg['state_machine']['depth_gate_m']}"
    )
    z = counter_cfg["counting_zone"]
    print(
        f"- Counting zone x {z['x_min']:.0f}–{z['x_max']:.0f}, "
        f"y {z['y_min']:.0f}–{z['y_max']:.0f}; línea y=324 "
        f"(ingreso = arriba→abajo)"
    )
    print(
        f"- Ruido: p_miss={base.p_miss}, σ_pos={base.pos_sigma_px} px, "
        f"σ_z={base.depth_sigma_mm} mm, conf~N({base.conf_mu},{base.conf_sd})"
    )
    print(
        f"- Fusión por centroide previa al tracker: "
        f"{base.cluster_distance_px:.0f} px · separación del par: "
        f"{base.separation_px:.0f} px · velocidad {base.speed_px_frame:.0f} px/frame"
    )
    print(f"- Semilla: {args.seed}\n")
    print(
        "> El modelo de ruido no está calibrado empíricamente (no hay video "
        "grabado del prototipo con el cual ajustarlo) y omite la dependencia "
        "entre proximidad y calidad de detección. Ver el encabezado del "
        "script.\n"
    )

    # --- Validación de simetría del montaje (sin ruido) ---------------------
    zero = NoiseDraw(
        dx=np.zeros(n_steps),
        dy=np.zeros(n_steps),
        dz=np.zeros(n_steps),
        conf=np.full(n_steps, base.conf_mu),
        miss=np.zeros(n_steps, dtype=bool),
    )
    clean = run_trial(counter_cfg, tracking_cfg, detection_cfg, base, zero, zero)
    print("## Control sin ruido\n")
    print(
        f"- Ingreso contabilizado: {'sí' if clean.ingress_counted else 'NO'} · "
        f"egreso contabilizado: {'sí' if clean.egress_counted else 'NO'} · "
        f"eventos emitidos: {clean.n_events}\n"
    )

    # --- A1. Prueba determinista de simetría --------------------------------
    rng = np.random.default_rng(args.seed)
    mismatches = 0
    checked = 0
    for _ in range(args.symmetry_trials):
        na = draw_noise(rng, n_steps, base)
        nb = draw_noise(rng, n_steps, base)
        orig = run_trial(counter_cfg, tracking_cfg, detection_cfg, base, na, nb)
        mirr = run_trial(
            counter_cfg,
            tracking_cfg,
            detection_cfg,
            base,
            nb.reflected(),
            na.reflected(),
        )
        checked += 1
        if (orig.ingress_counted != mirr.egress_counted) or (
            orig.egress_counted != mirr.ingress_counted
        ):
            mismatches += 1

    print("## A1 — Prueba determinista de simetría\n")
    print(
        "Cada ensayo se corre junto a su reflejo puntual exacto (misma "
        "realización de ruido, reflejada). Un algoritmo simétrico en "
        "dirección debe producir resultados espejados en todos los casos.\n"
    )
    print(f"- Pares evaluados: {checked}")
    print(f"- Discrepancias respecto del espejo: **{mismatches}**")
    print(
        "- Conclusión: "
        + (
            "no se detecta asimetría de dirección en las configuraciones " "evaluadas."
            if mismatches == 0
            else f"hay asimetría de dirección demostrada en {mismatches} pares."
        )
        + "\n"
    )

    # --- A2. Prueba estadística pareada -------------------------------------
    losses: list[tuple[bool, bool]] = []
    rejects: list[int] = []
    for _ in range(args.trials):
        na = draw_noise(rng, n_steps, base)
        nb = draw_noise(rng, n_steps, base)
        r = run_trial(counter_cfg, tracking_cfg, detection_cfg, base, na, nb)
        losses.append((r.lost(INGRESS), r.lost(EGRESS)))
        rejects.append(r.ambiguous_rejects)

    n = len(losses)
    li = sum(1 for a, _ in losses if a)
    le = sum(1 for _, b in losses if b)
    b = sum(1 for a, bb in losses if a and not bb)
    c = sum(1 for a, bb in losses if bb and not a)
    pval = mcnemar_exact(b, c)
    lo, hi = bootstrap_diff_ci(losses, rng)
    ci_i = wilson_ci(li, n)
    ci_e = wilson_ci(le, n)

    print("## A2 — Prueba estadística pareada\n")
    print(f"- Ensayos independientes: {n}")
    print(
        f"- Pérdida de ingreso: {li}/{n} = {li/n:.3f} "
        f"(IC 95 % Wilson {ci_i[0]:.3f}–{ci_i[1]:.3f})"
    )
    print(
        f"- Pérdida de egreso: {le}/{n} = {le/n:.3f} "
        f"(IC 95 % Wilson {ci_e[0]:.3f}–{ci_e[1]:.3f})"
    )
    print(f"- Pares discordantes: b={b} (sólo ingreso), c={c} (sólo egreso)")
    print(f"- McNemar exacto bilateral: p = {pval:.4f}")
    print(
        f"- Diferencia (ingreso − egreso) = {(li-le)/n:+.4f} "
        f"(IC 95 % bootstrap {lo:+.4f} — {hi:+.4f})"
    )
    distinguible = not (lo <= 0.0 <= hi)
    print(
        f"- La diferencia **{'SÍ' if distinguible else 'no'}** es "
        f"distinguible de cero al 95 %.\n"
    )
    print(
        f"- Rechazos por ambigüedad del asociador: total {sum(rejects)}, "
        f"media {statistics.mean(rejects):.2f} por ensayo, "
        f"máx {max(rejects)}\n"
    )

    # --- B. Superficie de decisión -----------------------------------------
    if args.sweep:
        print("## B — Superficie de decisión\n")
        print(
            "Fracción de cruces de verdad conocida no contabilizados, por "
            "separación entre personas y velocidad. `opuestos` = sentidos "
            "contrarios; `mismo` = control en el mismo sentido.\n"
        )
        seps = [10, 20, 40, 60, 100, 150, 250]
        speeds = [6, 12, 24]
        print(
            "| sep (px) | v (px/frame) | dirección | pérdida ingreso | "
            "pérdida egreso | pérdida total | rechazos/ensayo |"
        )
        print("|---:|---:|---|---:|---:|---:|---:|")
        for same in (False, True):
            for sep in seps:
                for v in speeds:
                    pp = SimParams(
                        separation_px=float(sep),
                        speed_px_frame=float(v),
                        p_miss=base.p_miss,
                        pos_sigma_px=base.pos_sigma_px,
                        depth_sigma_mm=base.depth_sigma_mm,
                        conf_mu=base.conf_mu,
                        conf_sd=base.conf_sd,
                        cluster_distance_px=base.cluster_distance_px,
                        same_direction=same,
                    )
                    ns = n_steps_for(pp)
                    li2 = le2 = rj = 0
                    same_lost = 0
                    for _ in range(args.sweep_trials):
                        r = run_trial(
                            counter_cfg,
                            tracking_cfg,
                            detection_cfg,
                            pp,
                            draw_noise(rng, ns, pp),
                            draw_noise(rng, ns, pp),
                        )
                        if same:
                            # Verdad conocida: DOS ingresos (ambas personas
                            # van en el mismo sentido).
                            same_lost += 2 - min(r.counted(INGRESS), 2)
                        else:
                            li2 += r.lost(INGRESS)
                            le2 += r.lost(EGRESS)
                        rj += r.ambiguous_rejects
                    t = args.sweep_trials
                    if same:
                        print(
                            f"| {sep} | {v} | mismo | — | — | "
                            f"{same_lost/(2*t):.3f} | {rj/t:.2f} |"
                        )
                    else:
                        print(
                            f"| {sep} | {v} | opuestos | "
                            f"{li2/t:.3f} | {le2/t:.3f} | "
                            f"{(li2+le2)/(2*t):.3f} | {rj/t:.2f} |"
                        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

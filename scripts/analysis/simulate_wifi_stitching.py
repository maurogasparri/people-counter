#!/usr/bin/env python3
"""Banco de simulación del agrupamiento de identidad inalámbrica.

Alimenta ``DedupEngine`` con secuencias de emisión sintéticas en las que cada
emisión tiene asignado, por construcción, el dispositivo verdadero que la
produjo. Sin radio, sin hardware y sin datos de operación: la verdad de
referencia es el generador.

Es una extensión de los casos de prueba de *stitching* ya reportados
(TC-09 y TC-10), no un experimento nuevo: los escenarios ejercitan las
cuatro reglas del esquema tal como están documentadas en
``src/wifi_ble/dedup.py``.

Escenarios
----------

Cada dispositivo verdadero se instancia con uno de cuatro comportamientos:

* ``wifi_seq``  — solo WiFi. Rota su dirección y **mantiene la continuidad
  del número de secuencia** entre rotaciones, porque el contador vive en el
  chip. Ejercita la regla 1.
* ``wifi_fp``   — solo WiFi. Rota su dirección y **reinicia el número de
  secuencia** en cada rotación (comportamiento de Apple H1 y posteriores),
  conservando la huella de capacidades. Ejercita la regla 4, que es la que
  cubre lo que la regla 1 no alcanza.
* ``dual``      — ambos protocolos activos de forma simultánea. La dirección
  WiFi rota a un ritmo y la privada BLE a otro, más lento. Ejercita la regla
  2 (ventana corta cross-protocolo) y la regla 3 (anclaje sobre un miembro
  BLE activo, unidireccional: solo una detección WiFi nueva se ancla).
* ``ble_only``  — un solo protocolo, con rotación de dirección privada.

Métricas
--------

1. **Falsos agrupamientos** — proporción de grupos resultantes que contienen
   emisiones de más de un dispositivo verdadero, y proporción de pares de
   dispositivos distintos incorrectamente unificados.
2. **Separaciones incorrectas** — proporción de dispositivos verdaderos cuyas
   emisiones quedan repartidas en más de un grupo, y cantidad media de grupos
   por dispositivo verdadero.
3. **Cociente grupos/dispositivos** — la medida directa del sesgo que el
   esquema introduce sobre el indicador de tráfico exterior. Por encima de 1
   sobreestima; por debajo, subestima.

Los intervalos de confianza son percentiles bootstrap **sobre repeticiones**,
no sobre dispositivos: dentro de una misma repetición los dispositivos
interactúan entre sí (compiten por los mismos grupos), de modo que no son
unidades independientes y tratarlas como tales estrecharía los intervalos de
forma indebida.

---------------------------------------------------------------------------
MODELO DE EMISIÓN — declaración explícita
---------------------------------------------------------------------------

Cada dispositivo emite a intervalo constante ``--emissions-per-min``, rota su
dirección WiFi cada ``--rotation-seconds`` y la privada BLE cada
``--ble-rotation-seconds``. La intensidad de señal nominal de cada dispositivo
se reparte uniformemente en el rango ``--rssi-min``…``--rssi-max`` y cada
emisión la perturba con ruido gaussiano de desvío ``--rssi-sigma``. Los
dispositivos de ambos protocolos emiten su trama BLE ``--dual-offset``
segundos después de la WiFi, dentro de la ventana corta cross-protocolo.

**Este modelo NO está calibrado empíricamente.** No se dispone de capturas de
tráfico real etiquetadas por dispositivo con las que ajustar sus parámetros:
cadencias, intervalos de rotación y dispersión de señal son supuestos del
autor. En particular el modelo asume emisión regular, rotación de período
fijo y ruido de señal independiente entre emisiones, mientras que un teléfono
real emite en ráfagas irregulares, rota con período variable y su intensidad
está correlacionada con una trayectoria física.

La sesión simulada dura por defecto 15 minutos, que es la ventana de
publicación del sistema: el cociente medido es entonces directamente el sesgo
sobre lo que el dispositivo reporta en cada ventana.

El barrido de densidad varía la cantidad de dispositivos **manteniendo fijo el
rango de intensidad de señal**. Es deliberado y refleja lo que ocurre en una
sucursal: más gente no ensancha el rango de señal del local, sino que mete más
dispositivos dentro del mismo rango, con lo que la separación media entre
ellos se reduce y el gate de ±5 dBm —el único criterio discriminante de las
cuatro reglas— pierde poder. Esa es la causa de la degradación que muestra el
barrido, y conviene enunciarla junto al resultado.

De ahí el alcance: el banco **caracteriza el comportamiento del algoritmo
ante patrones de rotación conocidos**, y no predice tasas sobre la población
real de dispositivos de una sucursal. La comparación entre densidades, que se
hace con todo lo demás fijo, es interpretable; los valores absolutos no son
extrapolables.

Uso:

    python scripts/analysis/simulate_wifi_stitching.py
    python scripts/analysis/simulate_wifi_stitching.py --devices 2,5,10,20,40
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wifi_ble import dedup as dedup_mod  # noqa: E402
from src.wifi_ble.dedup import DedupEngine  # noqa: E402

KINDS = ("wifi_seq", "wifi_fp", "dual", "ble_only")


class _RelojFijo:
    """Reloj controlado que sustituye a ``time`` dentro del módulo dedup.

    ``process_detection`` toma la marca temporal de ``time.time()``. Para
    simular media hora de emisiones sin esperarla, se reemplaza el módulo
    ``time`` que ve dedup por este objeto y se avanza a mano.
    """

    def __init__(self, t0: float = 1_800_000_000.0) -> None:
        self.t = t0

    def time(self) -> float:
        return self.t


@dataclass
class Emision:
    t: float
    dispositivo: int
    protocolo: str
    mac: str
    rssi: float
    seqnum: int | None
    fingerprint: str


@dataclass
class Parametros:
    n_dispositivos: int = 10
    session_seconds: float = 900.0
    emissions_per_min: float = 6.0
    rotation_seconds: float = 120.0
    ble_rotation_seconds: float = 900.0
    rssi_min: float = -75.0
    rssi_max: float = -45.0
    rssi_sigma: float = 2.0
    dual_offset: float = 0.5
    reparto: tuple[str, ...] = field(default_factory=lambda: KINDS)


def _mac_aleatoria(rng: np.random.Generator) -> str:
    """MAC localmente administrada (bit 0x02), que es lo que el pipeline cuenta."""
    octetos = [int(rng.integers(0, 256)) for _ in range(6)]
    octetos[0] = (octetos[0] | 0x02) & 0xFE
    return ":".join(f"{o:02X}" for o in octetos)


def generar_emisiones(p: Parametros, rng: np.random.Generator) -> list[Emision]:
    """Construye la secuencia de emisiones con su dispositivo verdadero."""
    emisiones: list[Emision] = []
    intervalo = 60.0 / p.emissions_per_min

    for idx in range(p.n_dispositivos):
        kind = p.reparto[idx % len(p.reparto)]
        # Señal nominal repartida en el rango; el gate de RSSI de las cuatro
        # reglas es +-5 dBm, así que este reparto es lo que decide cuántos
        # dispositivos son mutuamente confundibles.
        if p.n_dispositivos == 1:
            rssi_nom = (p.rssi_min + p.rssi_max) / 2.0
        else:
            rssi_nom = p.rssi_min + (p.rssi_max - p.rssi_min) * idx / (
                p.n_dispositivos - 1
            )
        fp_wifi = f"wifi-fp-{idx:04d}"
        fp_ble = f"ble-fp-{idx:04d}"

        mac_wifi = _mac_aleatoria(rng)
        mac_ble = _mac_aleatoria(rng)
        seq = int(rng.integers(0, 4096))
        ultima_rot_wifi = 0.0
        ultima_rot_ble = 0.0
        fase = float(rng.uniform(0, intervalo))

        t = fase
        while t < p.session_seconds:
            if kind in ("wifi_seq", "wifi_fp", "dual"):
                if t - ultima_rot_wifi >= p.rotation_seconds:
                    mac_wifi = _mac_aleatoria(rng)
                    ultima_rot_wifi = t
                    if kind == "wifi_fp":
                        # Reinicia el contador al rotar: la regla 1 no aplica.
                        seq = int(rng.integers(0, 4096))
                emisiones.append(
                    Emision(
                        t=t,
                        dispositivo=idx,
                        protocolo="wifi",
                        mac=mac_wifi,
                        rssi=float(rng.normal(rssi_nom, p.rssi_sigma)),
                        seqnum=seq % 4096,
                        fingerprint=fp_wifi,
                    )
                )
                seq = (seq + 1) % 4096

            if kind in ("dual", "ble_only"):
                if t - ultima_rot_ble >= p.ble_rotation_seconds:
                    mac_ble = _mac_aleatoria(rng)
                    ultima_rot_ble = t
                desfase = p.dual_offset if kind == "dual" else 0.0
                emisiones.append(
                    Emision(
                        t=t + desfase,
                        dispositivo=idx,
                        protocolo="ble",
                        mac=mac_ble,
                        rssi=float(rng.normal(rssi_nom, p.rssi_sigma)),
                        seqnum=None,
                        fingerprint=fp_ble,
                    )
                )
            t += intervalo

    emisiones.sort(key=lambda e: e.t)
    return emisiones


@dataclass
class Resultado:
    n_dispositivos: int
    n_grupos: int
    grupos_impuros: int
    pares_unificados: int
    pares_totales: int
    dispositivos_partidos: int
    grupos_por_dispositivo: float

    @property
    def prop_grupos_impuros(self) -> float:
        return self.grupos_impuros / self.n_grupos if self.n_grupos else 0.0

    @property
    def prop_pares_unificados(self) -> float:
        return self.pares_unificados / self.pares_totales if self.pares_totales else 0.0

    @property
    def prop_dispositivos_partidos(self) -> float:
        return self.dispositivos_partidos / self.n_dispositivos

    @property
    def cociente(self) -> float:
        return self.n_grupos / self.n_dispositivos


def correr_repeticion(p: Parametros, rng: np.random.Generator) -> Resultado:
    """Una corrida completa: genera, alimenta el motor y mide."""
    emisiones = generar_emisiones(p, rng)

    reloj = _RelojFijo()
    time_original = dedup_mod.time
    fd, ruta = tempfile.mkstemp(suffix=".sqlite", prefix="stitchbench_")
    os.close(fd)
    os.unlink(ruta)
    try:
        dedup_mod.time = reloj  # type: ignore[assignment]
        eng = DedupEngine(db_path=ruta)
        por_dispositivo: dict[int, set[str]] = {}
        por_grupo: dict[str, set[int]] = {}
        for e in emisiones:
            reloj.t = 1_800_000_000.0 + e.t
            r = eng.process_detection(
                e.mac, e.protocolo, e.rssi, seqnum=e.seqnum, fingerprint=e.fingerprint
            )
            gid = r["group_id"]
            por_dispositivo.setdefault(e.dispositivo, set()).add(gid)
            por_grupo.setdefault(gid, set()).add(e.dispositivo)
    finally:
        dedup_mod.time = time_original  # type: ignore[assignment]
        for suf in ("", "-wal", "-shm"):
            try:
                os.unlink(ruta + suf)
            except OSError:
                pass

    n_disp = len(por_dispositivo)
    n_grupos = len(por_grupo)
    impuros = sum(1 for d in por_grupo.values() if len(d) > 1)
    partidos = sum(1 for g in por_dispositivo.values() if len(g) > 1)

    # Pares de dispositivos distintos que comparten al menos un grupo.
    pares = set()
    for miembros in por_grupo.values():
        for a, b in itertools.combinations(sorted(miembros), 2):
            pares.add((a, b))
    pares_totales = n_disp * (n_disp - 1) // 2

    medio = statistics.mean(len(g) for g in por_dispositivo.values())

    return Resultado(
        n_dispositivos=n_disp,
        n_grupos=n_grupos,
        grupos_impuros=impuros,
        pares_unificados=len(pares),
        pares_totales=pares_totales,
        dispositivos_partidos=partidos,
        grupos_por_dispositivo=medio,
    )


def bootstrap_ic(
    valores: list[float], rng: np.random.Generator, reps: int = 5000
) -> tuple[float, float]:
    """IC percentil 95 % remuestreando REPETICIONES."""
    if not valores:
        return (math.nan, math.nan)
    arr = np.asarray(valores, dtype=float)
    idx = rng.integers(0, len(arr), size=(reps, len(arr)))
    medias = arr[idx].mean(axis=1)
    return (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5)))


def fmt(v: float, ic: tuple[float, float], dec: int = 3) -> str:
    return f"{v:.{dec}f} ({ic[0]:.{dec}f}–{ic[1]:.{dec}f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--devices", default="2,5,10,20,40")
    ap.add_argument("--repetitions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--session-seconds", type=float, default=900.0)
    ap.add_argument("--emissions-per-min", type=float, default=6.0)
    ap.add_argument("--rotation-seconds", type=float, default=120.0)
    ap.add_argument("--ble-rotation-seconds", type=float, default=900.0)
    ap.add_argument("--rssi-min", type=float, default=-75.0)
    ap.add_argument("--rssi-max", type=float, default=-45.0)
    ap.add_argument("--rssi-sigma", type=float, default=2.0)
    ap.add_argument("--dual-offset", type=float, default=0.5)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    densidades = [int(x) for x in args.devices.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    print("# Banco de simulación del agrupamiento de identidad inalámbrica\n")
    print("## Configuración\n")
    print(f"- Sesión simulada: {args.session_seconds/60:.0f} min por repetición")
    print(f"- Emisiones por dispositivo: {args.emissions_per_min:.0f}/min")
    print(
        f"- Rotación de dirección: WiFi {args.rotation_seconds:.0f} s · "
        f"BLE {args.ble_rotation_seconds:.0f} s"
    )
    print(
        f"- Señal nominal repartida en {args.rssi_min:.0f}…{args.rssi_max:.0f} dBm, "
        f"σ = {args.rssi_sigma:.1f} dBm por emisión"
    )
    print(f"- Repeticiones por punto: {args.repetitions} · semilla {args.seed}")
    print(
        "- Comportamientos, repartidos cíclicamente: `wifi_seq` (regla 1), "
        "`wifi_fp` (regla 4), `dual` (reglas 2 y 3), `ble_only`\n"
    )
    print(
        "> El modelo de emisión no está calibrado empíricamente. Caracteriza el "
        "comportamiento del algoritmo ante patrones de rotación conocidos; no "
        "predice tasas sobre la población real de dispositivos. Ver el "
        "encabezado del script.\n"
    )

    filas = []
    for n in densidades:
        p = Parametros(
            n_dispositivos=n,
            session_seconds=args.session_seconds,
            emissions_per_min=args.emissions_per_min,
            rotation_seconds=args.rotation_seconds,
            ble_rotation_seconds=args.ble_rotation_seconds,
            rssi_min=args.rssi_min,
            rssi_max=args.rssi_max,
            rssi_sigma=args.rssi_sigma,
            dual_offset=args.dual_offset,
        )
        res = [correr_repeticion(p, rng) for _ in range(args.repetitions)]
        filas.append((n, res))
        print(f"  … densidad {n} lista", file=sys.stderr)

    print("## 1. Falsos agrupamientos\n")
    print(
        "| dispositivos | grupos con más de un dispositivo | pares distintos unificados |"
    )
    print("|---:|---|---|")
    for n, res in filas:
        a = [r.prop_grupos_impuros for r in res]
        b = [r.prop_pares_unificados for r in res]
        print(
            f"| {n} | {fmt(statistics.mean(a), bootstrap_ic(a, rng))} "
            f"| {fmt(statistics.mean(b), bootstrap_ic(b, rng), 4)} |"
        )
    print()

    print("## 2. Separaciones incorrectas\n")
    print(
        "| dispositivos | dispositivos repartidos en más de un grupo | grupos por dispositivo |"
    )
    print("|---:|---|---|")
    for n, res in filas:
        a = [r.prop_dispositivos_partidos for r in res]
        b = [r.grupos_por_dispositivo for r in res]
        print(
            f"| {n} | {fmt(statistics.mean(a), bootstrap_ic(a, rng))} "
            f"| {fmt(statistics.mean(b), bootstrap_ic(b, rng), 2)} |"
        )
    print()

    print("## 3. Cociente grupos / dispositivos verdaderos\n")
    print("| dispositivos | cociente | sesgo sobre el indicador |")
    print("|---:|---|---|")
    for n, res in filas:
        c = [r.cociente for r in res]
        m = statistics.mean(c)
        ic = bootstrap_ic(c, rng)
        if ic[0] > 1.0:
            sesgo = f"sobreestima {(m-1)*100:+.0f} %"
        elif ic[1] < 1.0:
            sesgo = f"subestima {(m-1)*100:+.0f} %"
        else:
            sesgo = "no distinguible de 1"
        print(f"| {n} | {fmt(m, ic, 3)} | {sesgo} |")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

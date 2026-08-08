#!/usr/bin/env python3
"""Descompone el conjunto de trabajo del servicio en el dispositivo.

Lee ``memory.stat`` del cgroup y separa la parte irreducible —memoria anónima,
slab no recuperable y pilas del núcleo— de la recuperable bajo presión —caché
de archivos y slab recuperable—.

Importa para el dimensionamiento del hardware: ``memory.current`` incluye caché
oportunista que el núcleo retiene sólo porque hay RAM libre, de modo que en una
unidad de menor memoria ese valor sería más bajo. La cifra que un equipo de
2 GB debe poder sostener es la irreducible, no ``memory.current``.

Para medir bajo carga plena hay que sacar al pipeline del regulador de reposo:
abrir ``/stream`` del visor cuenta como suscriptor y lo fuerza a tasa plena.

Uso (en el dispositivo):  python3 memory_working_set.py
"""
import sys

CG = "/sys/fs/cgroup/system.slice/people-counter.service/"
M = 1048576

st = {}
for ln in open(CG + "memory.stat"):
    k, v = ln.split()
    st[k] = int(v)
cur = int(open(CG + "memory.current").read())
peak = int(open(CG + "memory.peak").read())

print(f"  memory.current : {cur/M:8.1f} MiB")
print(f"  memory.peak    : {peak/M:8.1f} MiB")
for k in ("anon", "file", "active_file", "inactive_file", "file_mapped",
          "file_dirty", "shmem", "slab_reclaimable", "slab_unreclaimable",
          "kernel_stack"):
    if k in st:
        print(f"  {k:<20}: {st[k]/M:8.1f} MiB")

rec = st.get("active_file", 0) + st.get("inactive_file", 0) + st.get("slab_reclaimable", 0)
irr = st.get("anon", 0) + st.get("slab_unreclaimable", 0) + st.get("kernel_stack", 0)
print(f"\n  RECUPERABLE : {rec/M:8.1f} MiB  ({100*rec/cur:.1f}%)")
print(f"  IRREDUCIBLE : {irr/M:8.1f} MiB  ({100*irr/cur:.1f}%)")

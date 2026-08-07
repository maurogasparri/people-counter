#!/usr/bin/env python3
"""TC-17 — store-and-forward con volumen equivalente a 72h (corte prolongado).

Simula el volumen (no el tiempo): inyecta 72h de eventos en un MessageBuffer
(DB temporal) offline, verifica persistencia sin pérdida, que NO desborde, y que
al "reconectar" drene íntegro (0 pérdida / 0 duplicado). Reproducible:
    py docs/validacion/tc17_buffer_72h.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from src.mqtt.buffer import MessageBuffer

# Volumen 72h a las cadencias reales del device:
TELEM = 12 * 72        # telemetría c/5min  = 864
WIFI = 4 * 72          # wifi_ble c/15min   = 288
COUNT = 150            # eventos de conteo (variable)
TOTAL = TELEM + WIFI + COUNT

tmp = tempfile.mkdtemp()
buf = MessageBuffer(str(Path(tmp) / "buffer.db"), max_backlog=50000)

print(f"=== TC-17: inyectando {TOTAL} eventos (72h) OFFLINE ===")
ids = []
for i in range(TELEM):
    ids.append(buf.enqueue("telemetry", {"seq": i, "cpu_temp_c": 50 + i % 10}))
for i in range(WIFI):
    # payload con hash opaco (sin MAC cruda) — coherente con privacidad
    ids.append(buf.enqueue("wifi_ble", {"devices": [{"visitor_hash": f"{i:032x}"}]}))
for i in range(COUNT):
    ids.append(buf.enqueue("counting", {"direction": "in", "track_id": i}))

unsent = buf.count_unsent()
print(f"  encolados={len(ids)} | count_unsent={unsent} | "
      f"persistidos sin pérdida: {unsent == TOTAL}")

dropped = buf.enforce_backlog_limit()
print(f"  enforce_backlog_limit dropeó={dropped} (cap 50000) -> "
      f"sin desborde: {dropped == 0 and buf.count_unsent() == TOTAL}")

# Reconexión: drenar en batches (igual que el replay), marcando PUBACK.
print("  reconectando -> drenando en batches de 100...")
sent_ids = []
after = 0
while True:
    batch = buf.get_pending(limit=100, after_id=after)
    if not batch:
        break
    for mid, topic, payload in batch:
        buf.mark_sent(mid)
        sent_ids.append(mid)
    after = batch[-1][0]

remaining = buf.count_unsent()
dup = len(sent_ids) != len(set(sent_ids))
lost = set(ids) - set(sent_ids)
print(f"  drenados={len(sent_ids)} | restantes={remaining} | "
      f"duplicados={dup} | perdidos={len(lost)}")
ok = remaining == 0 and not dup and not lost and len(sent_ids) == TOTAL
print(f"  VEREDICTO TC-17: {'CUMPLE' if ok else 'NO CUMPLE'} "
      f"(persistencia + drenaje íntegro, 0 pérdida/0 dup, sin desborde)")

# Control de no-desborde: cap chico + sobrecarga -> dropea los más viejos.
print("\n=== Control de cap (anti-desborde) ===")
buf2 = MessageBuffer(str(Path(tempfile.mkdtemp()) / "b.db"), max_backlog=1000)
for i in range(1500):
    buf2.enqueue("telemetry", {"seq": i})
d2 = buf2.enforce_backlog_limit()
print(f"  inyectados 1500, cap 1000 -> dropeados {d2}, "
      f"unsent final {buf2.count_unsent()} (acotado a 1000: "
      f"{buf2.count_unsent() <= 1000})")

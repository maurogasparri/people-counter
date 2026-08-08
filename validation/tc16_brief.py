import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, robusto al CWD
from src.mqtt.buffer import MessageBuffer
M = 30
buf = MessageBuffer(str(Path(tempfile.mkdtemp()) / "b.db"), max_backlog=50000)
ids = [buf.enqueue("counting", {"direction": "in", "track_id": i}) for i in range(M)]
print(f"  corte breve: {M} eventos encolados offline | unsent={buf.count_unsent()} | persistidos={buf.count_unsent()==M}")
# restablece -> drena en una pasada marcando PUBACK
sent = []
after = 0
while True:
    b = buf.get_pending(limit=100, after_id=after)
    if not b: break
    for mid, t, p in b: buf.mark_sent(mid); sent.append(mid)
    after = b[-1][0]
dup = len(sent) != len(set(sent)); lost = set(ids) - set(sent)
ok = buf.count_unsent()==0 and not dup and not lost and len(sent)==M
print(f"  restablecido: drenados={len(sent)} restantes={buf.count_unsent()} dup={dup} perdidos={len(lost)}")
print(f"  VEREDICTO TC-16: {'CUMPLE' if ok else 'NO CUMPLE'} (retransmisión íntegra, 0 pérdida/0 dup)")

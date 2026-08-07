#!/usr/bin/env python3
"""TC-09 / TC-10 — stitching WiFi y WiFi+BLE con detecciones sintéticas conocidas.

Inyecta un patrón de ground-truth en el DedupEngine real (DB temporal) y asierta
que las múltiples identidades de un mismo dispositivo colapsan a UN group_id.
Reproducible: py docs/validacion/tc09_10_stitching.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from src.wifi_ble.dedup import DedupEngine


def make_engine():
    tmp = tempfile.mkdtemp()
    return DedupEngine(
        str(Path(tmp) / "dedup.db"),
        cross_window_seconds=2.0,
        cross_rssi_delta=5.0,
        seqnum_stitch_enabled=True,
        seqnum_stitch_window_seconds=30.0,
        seqnum_max_delta=100,
        seqnum_rssi_delta=5.0,
        ble_anchor_enabled=True,
        ble_anchor_window_seconds=900.0,
    )


print("=" * 60)
print("TC-09 — Stitching WiFi (MAC randomizada rotando, mismo device)")
print("=" * 60)
eng = make_engine()
# Device A: 6 MACs randomizadas (bit locally-administered 0x02), seqnum continuo
# (Δ=40 ≤ 100), MISMO fingerprint, RSSI dentro de 5dBm → debe ser 1 solo grupo.
fp_a = "a1b2c3d4e5f6a7b8"
groups_a = set()
for i in range(6):
    mac = f"02:11:22:33:44:{i:02x}"
    r = eng.process_detection(mac, "wifi", -60.0 - (i % 3),
                              seqnum=100 + i * 40, fingerprint=fp_a)
    groups_a.add(r["group_id"])
# Device B: fingerprint distinto, seqnum lejano, RSSI distinta → grupo aparte.
fp_b = "ffffffff00000000"
rb = eng.process_detection("06:99:88:77:66:55", "wifi", -40.0,
                           seqnum=3000, fingerprint=fp_b)
print(f"  Device A: 6 MACs randomizadas -> {len(groups_a)} group_id "
      f"(esperado 1)")
print(f"  Device B (otro device) -> group distinto de A: "
      f"{rb['group_id'] not in groups_a}")
tc09_ok = len(groups_a) == 1 and rb["group_id"] not in groups_a
ratio_a = 1 / len(groups_a) if groups_a else 0
print(f"  Ratio de agrupamiento A: {len(groups_a)} grupo / 6 MACs "
      f"(ideal: 1 grupo)")
print(f"  VEREDICTO TC-09: {'CUMPLE' if tc09_ok else 'NO CUMPLE'}")

print("\n" + "=" * 60)
print("TC-10 — Stitching WiFi+BLE (mismo device, dos protocolos)")
print("=" * 60)
eng2 = make_engine()
# Device C: emite WiFi y BLE dentro de cross_window (2s) con RSSI compatible
# (Δ ≤ 5dBm) → regla cross-protocol L2 → mismo group_id.
rc_wifi = eng2.process_detection("02:AA:BB:CC:DD:EE", "wifi", -58.0)
rc_ble = eng2.process_detection("7a:bb:cc:dd:ee:ff", "ble", -60.0)
same = rc_wifi["group_id"] == rc_ble["group_id"]
# Control negativo: BLE de otro device con RSSI muy distinta → NO mergea.
rd_ble = eng2.process_detection("11:22:33:44:55:66", "ble", -20.0)
diff = rd_ble["group_id"] != rc_wifi["group_id"]
print(f"  WiFi+BLE del mismo device (dRSSI=2dBm, <2s) -> mismo group: {same}")
print(f"  BLE de otro device (dRSSI=38dBm) -> group distinto: {diff}")
tc10_ok = same and diff
print(f"  VEREDICTO TC-10: {'CUMPLE' if tc10_ok else 'NO CUMPLE'}")

print(f"\nRESUMEN: TC-09={'PASS' if tc09_ok else 'FAIL'} "
      f"TC-10={'PASS' if tc10_ok else 'FAIL'}")

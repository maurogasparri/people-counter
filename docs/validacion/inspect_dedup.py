#!/usr/bin/env python3
"""TC-14 — audita el SQLite del dedup WiFi/BLE: confirma que SOLO hay hashes
salteados (sin MACs crudas). Correr en la Pi: sudo python3 inspect_dedup.py
"""
import sqlite3

db = '/var/lib/people-counter/wifi_ble_dedup.sqlite'
c = sqlite3.connect(db)
cur = c.cursor()
print('=== tables / cols / rows ===')
tabs = [t for (t,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for t in tabs:
    cols = [col[1] for col in cur.execute(f'PRAGMA table_info({t})')]
    try:
        n = cur.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
    except Exception as e:
        n = f'err:{e}'
    print(f'  {t}: rows={n} cols={cols}')
print('=== sample row per table (bytes -> hex preview; busca MACs crudas) ===')
for t in tabs:
    cols = [col[1] for col in cur.execute(f'PRAGMA table_info({t})')]
    row = cur.execute(f'SELECT * FROM {t} LIMIT 1').fetchone()
    if row:
        out = {}
        for k, v in zip(cols, row):
            if isinstance(v, (bytes, bytearray)):
                out[k] = 'BYTES[' + str(len(v)) + ']:' + v.hex()[:20] + '...'
            else:
                out[k] = v
        print(f'  {t}: {out}')

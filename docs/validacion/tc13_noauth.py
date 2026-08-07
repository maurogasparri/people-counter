#!/usr/bin/env python3
"""TC-13 — rechazo sin autenticación. POST a la API sin firma SigV4/IAM →
debe responder 401/403. Reproducible (sin bash): py tc13_noauth.py"""
import json
import sys
import urllib.error
import urllib.request

URL = "https://api.<tu-dominio>/pos/transactions"
out_path = str(__import__("pathlib").Path(__file__).with_name("q_noauth_out.json"))
req = urllib.request.Request(
    URL,
    data=b"{}",
    method="POST",
    headers={"content-type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        code, body = r.status, r.read().decode("utf-8", "replace")
except urllib.error.HTTPError as e:
    code, body = e.code, e.read().decode("utf-8", "replace")
open(out_path, "w").write(json.dumps({"status": code, "body": body}))
print(f"TC-13 sin auth -> HTTP {code}  body={body[:120]}")
ok = code in (401, 403)
print(f"VEREDICTO TC-13 (no-auth): {'CUMPLE (rechazado)' if ok else 'NO CUMPLE'}")
sys.exit(0 if ok else 1)

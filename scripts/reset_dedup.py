#!/usr/bin/env python3
"""Reset diario del estado de dedup WiFi/BLE.

Lo invoca ``people-counter-reset.service``. Resuelve el path del SQLite de dedup
EXACTAMENTE como ``src.main.build_wifi_ble`` (config-aware), para no resetear el
DB equivocado: el ``.service`` viejo hardcodeaba ``/var/lib/people-counter/
dedup.db`` mientras el pipeline usa ``wifi_ble_dedup.sqlite`` (default) → el DB
activo nunca se reseteaba y los hashes/counts se acumulaban día tras día.

Borra ``hash_groups`` (los counts de passersby/shoppers son ``DISTINCT
group_id`` del día, así que esto los vuelve a cero). Idempotente y best-effort:
si el DB no existe todavía, ``reset_daily`` lo crea vacío.
"""

from __future__ import annotations

import os
import sys

from src.config.loader import load_device_config
from src.wifi_ble.dedup import DedupEngine


def _dedup_db_path(config: dict) -> str:
    """Mismo cálculo que build_wifi_ble: override explícito o el default
    junto al buffer.db."""
    wifi_cfg = config.get("wifi_ble", {}) or {}
    return wifi_cfg.get("dedup_db_path") or os.path.join(
        os.path.dirname(config["buffer"]["db_path"]),
        "wifi_ble_dedup.sqlite",
    )


def main() -> int:
    config = load_device_config()
    db_path = _dedup_db_path(config)
    DedupEngine(db_path=db_path).reset_daily()
    print(f"Dedup reset done: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

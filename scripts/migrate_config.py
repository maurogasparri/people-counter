#!/usr/bin/env python3
"""Migra ``/etc/people-counter/config.yaml`` a los nombres canónicos
actuales de las keys.

El loader runtime acepta los nombres viejos por compat y loggea warnings,
pero tener el config "limpio" elimina los warnings y deja el archivo
sincronizado con lo que documenta ``config/config.example.yaml``. Este
script hace el rename in-place con un backup ``.bak.<timestamp>`` y un
dry-run por default.

Uso:
    # Dry-run (default): muestra qué renombraría sin tocar el archivo.
    sudo PYTHONPATH=. python3 scripts/migrate_config.py

    # Aplicar los cambios.
    sudo PYTHONPATH=. python3 scripts/migrate_config.py --apply

    # Migrar otro path (testing / dev workstation).
    PYTHONPATH=. python3 scripts/migrate_config.py --config ./test.yaml --apply

Renames soportados:
    tracking.max_disappeared          → max_disappeared_frames
    tracking.max_distance             → max_distance_px
    vision.ae_lock.initial_settle_s   → initial_settle_seconds
    vision.ae_lock.resettle_s         → resettle_seconds

Keys obsoletas que se eliminan:
    vision.sensor_raw_size            (ahora se usa sensor.default_res)
    detection.static_suppressor.approx_fps  (supresor con ventana
                                              temporal real, ya no
                                              necesita estimación)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# Añadir la raíz del repo al path para importar el loader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import (
    _COMPAT_REMOVED,
    _COMPAT_RENAMES,
    _walk_nested_dict,
    DEFAULT_DEVICE_CONFIG_PATH,
)


def _path_str(path: tuple[str, ...], key: str) -> str:
    if path:
        return ".".join(path) + "." + key
    return key


def migrate(config: dict[str, Any]) -> list[str]:
    """Aplica los renames/removes a ``config`` in-place.

    Devuelve una lista de strings describiendo cada cambio aplicado.
    """
    changes: list[str] = []

    for path, old_key, new_key in _COMPAT_RENAMES:
        parent = _walk_nested_dict(config, path)
        if parent is None:
            continue
        if old_key in parent and new_key not in parent:
            parent[new_key] = parent.pop(old_key)
            changes.append(
                f"rename: {_path_str(path, old_key)} → {_path_str(path, new_key)}"
            )
        elif old_key in parent and new_key in parent:
            parent.pop(old_key)
            changes.append(
                f"remove obsolete (both present): {_path_str(path, old_key)}"
            )

    for path, old_key in _COMPAT_REMOVED:
        parent = _walk_nested_dict(config, path)
        if parent is None:
            continue
        if old_key in parent:
            parent.pop(old_key)
            changes.append(f"remove obsolete: {_path_str(path, old_key)}")

    return changes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migra el config per-device a los nombres canónicos.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_DEVICE_CONFIG_PATH,
        help=(f"Path al config. Default {DEFAULT_DEVICE_CONFIG_PATH}."),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplicar los cambios. Sin este flag el script es dry-run.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config no existe: {config_path}", file=sys.stderr)
        return 1

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    changes = migrate(config)

    if not changes:
        print(f"✓ {config_path} ya está en los nombres canónicos. Nada que migrar.")
        return 0

    print(f"Cambios a aplicar en {config_path}:")
    for change in changes:
        print(f"  - {change}")

    if not args.apply:
        print("\n(dry-run — pasar --apply para escribir el archivo)")
        return 0

    # Backup antes de escribir.
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(config_path.suffix + f".bak.{stamp}")
    shutil.copy2(config_path, backup_path)
    print(f"\nBackup: {backup_path}")

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"✓ {config_path} migrado")
    return 0


if __name__ == "__main__":
    sys.exit(main())

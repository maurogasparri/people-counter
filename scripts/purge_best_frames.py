"""Purga JPGs best-frame más viejos que un threshold de retención.

CLI standalone invocado diariamente por ``people-counter-purge-best-frames.timer``
(ver ``config/``). Camina ``--output-dir`` recursivamente y unlinkea cualquier
archivo cuyo mtime sea más viejo que ``--retention-days`` días. Idempotente y
seguro de correr desde cron.

Por qué un script separado (no el pipeline runtime):

  - La retención es una garantía *legal*. Tiene que seguir funcionando aún
    cuando el pipeline crashea, el dispositivo está inalcanzable, o el operador
    apaga el feature — es decir, independiente de ``src/main.py``.
  - El timer de systemd hace el schedule visible (`systemctl list-timers`) y
    audit-friendly.

Uso:

    python scripts/purge_best_frames.py \
        --output-dir /var/lib/people-counter/best_frames \
        --retention-days 7 \
        [--dry-run]

Exit codes:

  0 — success (sin errores; puede haber purgado 0 o más archivos).
  1 — output dir does not exist (treated as a hard error so cron loudly
      reports the misconfiguration instead of silently doing nothing).
  2 — invalid arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("purge_best_frames")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete best-frame JPGs older than --retention-days. "
            "Designed to be invoked by systemd timer."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory containing dated subfolders of JPGs.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        required=True,
        help=(
            "Files with mtime older than this many days are deleted. "
            "Must be > 0 — passing 0 would purge everything just-written."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be deleted without removing anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def purge(
    output_dir: Path,
    retention_days: int,
    *,
    dry_run: bool = False,
    now: float | None = None,
) -> tuple[int, int]:
    """Delete files older than ``retention_days``.

    Returns ``(deleted, kept)``. The ``now`` parameter is injected by
    tests so they can produce deterministic age comparisons; production
    callers leave it ``None`` to use ``time.time()``.
    """
    if not output_dir.exists():
        logger.error("output_dir does not exist: %s", output_dir)
        raise FileNotFoundError(str(output_dir))

    if retention_days <= 0:
        # Defensive: argparse already validates type, but a value of 0
        # would purge everything in the dir, including the JPG that the
        # pipeline wrote one second ago. The systemd unit ships with
        # retention_days=7; anyone passing 0 has a bug.
        raise ValueError("retention_days must be > 0")

    cutoff = (now if now is not None else time.time()) - (retention_days * 86400.0)
    deleted = 0
    kept = 0
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError as e:
            logger.warning("stat failed for %s: %s", path, e)
            continue
        if mtime < cutoff:
            if dry_run:
                logger.info(
                    "would delete: %s (age=%.1fd)",
                    path,
                    (cutoff - mtime + retention_days * 86400) / 86400,
                )
                deleted += 1
                continue
            try:
                path.unlink()
                deleted += 1
                logger.debug("deleted: %s", path)
            except OSError as e:
                logger.warning("unlink failed for %s: %s", path, e)
        else:
            kept += 1
    return deleted, kept


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.retention_days <= 0:
        print(
            "Error: --retention-days must be > 0 (got " f"{args.retention_days}).",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    try:
        deleted, kept = purge(
            output_dir,
            retention_days=args.retention_days,
            dry_run=args.dry_run,
        )
    except FileNotFoundError:
        return 1

    verb = "would delete" if args.dry_run else "deleted"
    logger.info(
        "%s %d file(s); kept %d file(s) (retention=%dd, dir=%s)",
        verb,
        deleted,
        kept,
        args.retention_days,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

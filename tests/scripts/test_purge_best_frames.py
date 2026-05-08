"""Tests for scripts/purge_best_frames.py.

Verifies retention enforcement: files older than --retention-days are
removed, fresh files are kept, and dry-run produces no side effects.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest


# Load the script as a module (it lives outside the src tree).
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "purge_best_frames.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "purge_best_frames",
        _SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


purge_mod = _load_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _touch(p: Path, age_days: float, content: bytes = b"x") -> None:
    """Create a file with mtime `age_days` days in the past."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    target = time.time() - age_days * 86400.0
    os.utime(p, (target, target))


# ---------------------------------------------------------------------------
# purge() core
# ---------------------------------------------------------------------------


def test_purge_deletes_old_keeps_new(tmp_path: Path):
    old = tmp_path / "2026-04-01" / "1_old.jpg"
    fresh = tmp_path / "2026-05-07" / "2_fresh.jpg"
    _touch(old, age_days=10)
    _touch(fresh, age_days=2)

    deleted, kept = purge_mod.purge(tmp_path, retention_days=7)
    assert deleted == 1
    assert kept == 1
    assert not old.exists()
    assert fresh.exists()


def test_purge_zero_files_returns_zero(tmp_path: Path):
    deleted, kept = purge_mod.purge(tmp_path, retention_days=7)
    assert deleted == 0
    assert kept == 0


def test_purge_dry_run_no_unlink(tmp_path: Path):
    old = tmp_path / "old.jpg"
    _touch(old, age_days=30)
    deleted, kept = purge_mod.purge(tmp_path, retention_days=7, dry_run=True)
    assert deleted == 1  # would-be count
    assert kept == 0
    assert old.exists()  # but still on disk


def test_purge_recursive(tmp_path: Path):
    """Files in nested date dirs must be visible to the walker."""
    a = tmp_path / "2026-01-01" / "1_a.jpg"
    b = tmp_path / "2026-01-01" / "subdir" / "2_b.jpg"
    c = tmp_path / "fresh" / "3_c.jpg"
    _touch(a, age_days=20)
    _touch(b, age_days=20)
    _touch(c, age_days=1)
    deleted, kept = purge_mod.purge(tmp_path, retention_days=7)
    assert deleted == 2
    assert kept == 1
    assert not a.exists()
    assert not b.exists()
    assert c.exists()


def test_purge_uses_injected_now(tmp_path: Path):
    """The `now` injection lets tests pin time deterministically."""
    f = tmp_path / "f.jpg"
    f.write_bytes(b"x")
    real_now = time.time()
    os.utime(f, (real_now, real_now))
    # Pretend it's 30 days in the future — file is now stale.
    deleted, _ = purge_mod.purge(
        tmp_path,
        retention_days=7,
        now=real_now + 30 * 86400,
    )
    assert deleted == 1
    assert not f.exists()


def test_purge_missing_dir_raises(tmp_path: Path):
    bogus = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        purge_mod.purge(bogus, retention_days=7)


def test_purge_negative_retention_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        purge_mod.purge(tmp_path, retention_days=0)
    with pytest.raises(ValueError):
        purge_mod.purge(tmp_path, retention_days=-3)


def test_purge_skips_directories(tmp_path: Path):
    """Walking should silently skip dir entries (rglob('*') returns
    them) — only files are unlinked. Otherwise an empty old dir would
    raise IsADirectoryError on unlink."""
    old_dir = tmp_path / "old_dir"
    old_dir.mkdir()
    target = time.time() - 30 * 86400.0
    os.utime(old_dir, (target, target))
    deleted, kept = purge_mod.purge(tmp_path, retention_days=7)
    assert deleted == 0
    assert old_dir.exists()


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


def test_main_returns_0_on_success(tmp_path: Path, capsys):
    f = tmp_path / "x.jpg"
    _touch(f, age_days=30)
    rc = purge_mod.main(
        [
            "--output-dir",
            str(tmp_path),
            "--retention-days",
            "7",
        ]
    )
    assert rc == 0
    assert not f.exists()


def test_main_returns_1_when_dir_missing(tmp_path: Path):
    rc = purge_mod.main(
        [
            "--output-dir",
            str(tmp_path / "nope"),
            "--retention-days",
            "7",
        ]
    )
    assert rc == 1


def test_main_returns_2_on_invalid_retention(tmp_path: Path):
    rc = purge_mod.main(
        [
            "--output-dir",
            str(tmp_path),
            "--retention-days",
            "0",
        ]
    )
    assert rc == 2

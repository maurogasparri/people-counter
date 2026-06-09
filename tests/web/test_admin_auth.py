"""Tests para src/web/admin_auth.py — hashing y storage de la contraseña admin."""

import os

from src.web.admin_auth import (
    hash_password,
    read_secret,
    verify_password,
    write_secret,
)


def test_hash_verify_roundtrip():
    h = hash_password("correct horse battery")
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)


def test_hash_uses_random_salt():
    assert hash_password("x") != hash_password("x")  # salt aleatorio por hash


def test_verify_rejects_garbage():
    assert not verify_password("x", None)
    assert not verify_password("x", "")
    assert not verify_password("x", "not-a-valid-hash")


def test_write_read_secret_roundtrip(tmp_path):
    p = str(tmp_path / "admin.secret")
    write_secret("hunter2pass", p)
    stored = read_secret(p)
    assert stored is not None and verify_password("hunter2pass", stored)
    if os.name == "posix":
        assert (os.stat(p).st_mode & 0o077) == 0  # sin acceso group/other


def test_read_secret_missing_returns_none(tmp_path):
    assert read_secret(str(tmp_path / "nope")) is None


# (session_token se removió de admin_auth: las sesiones ahora son tokens
# random per-login con expiry server-side en el viewer. La invalidación por
# cambio de contraseña la cubre test_change_password_invalidates_old_sessions
# en tests/web/test_viewer.py.)

"""Auth del panel admin del viewer: contraseña hasheada en un archivo dedicado.

Guarda ``pbkdf2_sha256(password, salt)`` en
``/etc/people-counter/admin.secret`` (modo 600). **Nunca plaintext.** Lo usan
el viewer (login / cambio de contraseña) y ``scripts/set_admin_password.py``
(seteo inicial en el provisioning).

Modelo: la presencia del archivo habilita los controles de power (Reiniciar /
Apagar). El token de cookie de sesión se deriva del hash almacenado, así un
cambio de contraseña invalida las sesiones viejas automáticamente. Todo stdlib
(hashlib/hmac/secrets) — sin dependencias.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

ADMIN_SECRET_PATH = "/etc/people-counter/admin.secret"
_FORMAT = "pbkdf2_sha256"
_ROUNDS = 200_000
MIN_PASSWORD_LEN = 8


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """``pbkdf2_sha256$rounds$salt_hex$dk_hex``."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ROUNDS)
    return f"{_FORMAT}${_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    """Compara ``password`` contra el hash almacenado (constant-time)."""
    if not stored:
        return False
    try:
        fmt, rounds_s, salt_hex, dk_hex = stored.strip().split("$")
        if fmt != _FORMAT:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
        rounds = int(rounds_s)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(dk, expected)


def read_secret(path: str = ADMIN_SECRET_PATH) -> str | None:
    """Hash almacenado, o None si el archivo no existe / está vacío / es ilegible."""
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def write_secret(password: str, path: str = ADMIN_SECRET_PATH) -> None:
    """Escribe el hash de ``password`` (escritura atómica, modo 600)."""
    data = (hash_password(password) + "\n").encode("utf-8")
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, path)  # atómico dentro del mismo dir (ReadWritePaths)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def session_token(stored_hash: str) -> str:
    """Token opaco de cookie derivado del hash almacenado (HMAC). Cambia cuando
    cambia la contraseña → las sesiones viejas dejan de validar solas."""
    return hmac.new(
        stored_hash.encode("utf-8"), b"pc-admin-session-v1", hashlib.sha256
    ).hexdigest()

#!/usr/bin/env python3
"""Setea la contraseña del panel admin del viewer (botones Reiniciar / Apagar).

Escribe el hash pbkdf2 en /etc/people-counter/admin.secret (modo 600). La
presencia de ese archivo habilita los controles de power en el viewer; el
cambio posterior se hace desde la propia UI.

Correr UNA vez en el provisioning del device, **como el usuario del servicio
(pi), NO con sudo** — así el archivo queda legible por el servicio:

    cd /usr/src/people-counter
    PYTHONPATH=. python3 scripts/set_admin_password.py

Pide la contraseña por stdin (no queda en el historial ni en `ps`).
"""

import getpass
import os
import sys

sys.path.insert(0, ".")
from src.web.admin_auth import ADMIN_SECRET_PATH, MIN_PASSWORD_LEN, write_secret


def main() -> None:
    if os.geteuid() == 0:
        print(
            "ADVERTENCIA: corriendo como root. El servicio corre como 'pi' y no "
            f"podrá leer un {ADMIN_SECRET_PATH} root:600. Correlo como pi (sin sudo).",
            file=sys.stderr,
        )
    pw1 = getpass.getpass("Nueva contraseña admin: ")
    if len(pw1) < MIN_PASSWORD_LEN:
        print(f"ERROR: mínimo {MIN_PASSWORD_LEN} caracteres.", file=sys.stderr)
        sys.exit(1)
    if pw1 != getpass.getpass("Repetir: "):
        print("ERROR: no coinciden.", file=sys.stderr)
        sys.exit(1)
    try:
        write_secret(pw1)
    except OSError as e:
        print(f"ERROR escribiendo {ADMIN_SECRET_PATH}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK — hash escrito en {ADMIN_SECRET_PATH} (modo 600).")
    print("Los botones Reiniciar / Apagar ya aparecen en el viewer (panel Admin).")


if __name__ == "__main__":
    main()

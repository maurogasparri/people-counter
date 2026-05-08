"""Hashing SHA-256 truncado para direcciones MAC."""
import hashlib


def hash_mac(mac: str, salt: str = "") -> str:
    """Hashea una dirección MAC con SHA-256 truncado a 16 bytes.

    Args:
        mac: String de la dirección MAC (cualquier formato).
        salt: Salt diario opcional para privacidad adicional.

    Returns:
        String hex del hash truncado (32 chars).
    """
    normalized = mac.upper().replace(":", "").replace("-", "")
    digest = hashlib.sha256(f"{normalized}{salt}".encode()).digest()[:16]
    return digest.hex()

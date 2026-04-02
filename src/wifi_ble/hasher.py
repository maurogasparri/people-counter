"""SHA-256 truncated hashing for MAC addresses."""
import hashlib


def hash_mac(mac: str, salt: str = "") -> str:
    """Hash a MAC address using SHA-256 truncated to 16 bytes.

    Args:
        mac: MAC address string (any format).
        salt: Optional daily salt for additional privacy.

    Returns:
        Hex string of truncated hash (32 chars).
    """
    normalized = mac.upper().replace(":", "").replace("-", "")
    digest = hashlib.sha256(f"{normalized}{salt}".encode()).digest()[:16]
    return digest.hex()

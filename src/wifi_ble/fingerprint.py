"""Fingerprinting de dispositivos para sobrevivir la rotación de MAC/dirección.

El fingerprint se construye con lo que es ESTABLE por hardware/driver/OS y
descarta lo volátil (identificador, seqnum, SSID sondeado, contadores). Dos
detecciones con fingerprint distinto son, con certeza, dispositivos distintos
(filtro duro); el mismo fingerprint + corroboración (RSSI/tiempo) sugiere el
mismo aparato — clave para re-unir rotaciones que el seqnum no cubre (iOS
resetea el seqnum al rotar la MAC).

Caveat: dispositivos idénticos (dos iPhone iguales, dos AirPods iguales) dan
fingerprints idénticos. Por eso el fingerprint es un *bucket* grueso: separa
con certeza, pero NO confirma identidad por sí solo.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Optional

# IEs WiFi cuyo VALOR es estable (capacidades de hw/driver). El resto solo
# aporta presencia + orden al fingerprint.
#   1   = supported rates           50  = extended supported rates
#   45  = HT capabilities (11n)     191 = VHT capabilities (11ac)
#   127 = extended capabilities     255 = element ID extension (HE/11ax, etc.)
_STABLE_VALUE_TAGS = frozenset({1, 45, 50, 127, 191, 255})
_SSID_TAG = 0
_VENDOR_TAG = 221
_APPLE_COMPANY_ID = 0x004C


def _digest(parts: list[str]) -> str:
    if not parts:
        return ""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def wifi_fingerprint(elements: Iterable[tuple[int, bytes]]) -> str:
    """Fingerprint estable de un probe request a partir de sus Information
    Elements ``(tag_id, value_bytes)`` en orden de aparición.

    Incluye: el ORDEN de tags presentes, el VALOR de los IEs de capacidades
    (rates/HT/VHT/ext/HE) y el OUI de los vendor-specific (tag 221). EXCLUYE:
    el valor del SSID (tag 0, cambia por red) y la data volátil del vendor.
    Devuelve "" si no hay IEs (probe sin fingerprintear)."""
    parts: list[str] = []
    for tag, value in elements:
        value = bytes(value or b"")
        if tag == _SSID_TAG:
            parts.append("0")  # presencia sí, valor no (volátil)
        elif tag == _VENDOR_TAG:
            parts.append("221:" + value[:3].hex())  # OUI del vendor, no la data
        elif tag in _STABLE_VALUE_TAGS:
            parts.append(f"{tag}:{value.hex()}")
        else:
            parts.append(str(tag))
    return _digest(parts)


def ble_fingerprint(
    manufacturer_data: Optional[Mapping[int, bytes]],
    service_uuids: Optional[Iterable[str]],
    tx_power: Optional[int],
) -> str:
    """Fingerprint estable de un advertisement BLE.

    Para Apple (company 0x004C) extrae los SUBTIPOS Continuity (los *type* de
    cada TLV: nearby, handoff, AirDrop, AirPods…), muy distintivos por
    modelo/OS, sin los bytes volátiles del payload. Para otros fabricantes usa
    company_id + longitud. Suma los service UUIDs anunciados y el TX power.
    Devuelve "" si no hay nada fingerprinteable."""
    parts: list[str] = []
    for cid in sorted(manufacturer_data or {}):
        data = bytes(manufacturer_data[cid] or b"")
        if cid == _APPLE_COMPANY_ID and data:
            types: list[str] = []
            i = 0
            while i + 1 < len(data):
                t = data[i]
                ln = data[i + 1]
                types.append(f"{t:02x}")
                i += 2 + ln
            parts.append("4c:" + ",".join(types))
        else:
            parts.append(f"{cid:04x}:{len(data)}")
    for uuid in sorted(str(u) for u in (service_uuids or [])):
        parts.append(uuid)
    if tx_power is not None:
        parts.append(f"tx{int(tx_power)}")
    return _digest(parts)

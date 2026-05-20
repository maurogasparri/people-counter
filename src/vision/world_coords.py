"""Helpers de world-space: geometría del montaje → mediciones físicas.

La calibración estéreo es independiente del install — medimos la profundidad
camera-to-scene. Para traducir esas profundidades a cantidades físicamente
significativas (altura de cabeza, coordenadas en el plano del piso, etc.)
combinamos la profundidad con la altura de montaje sobre el piso del install.

Actualmente se usa para clasificar detecciones como adulto vs niño en base a
la altura de la cabeza.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def head_height_above_floor(
    near_depth_mm: float,
    mounting_height_mm: float,
) -> Optional[float]:
    """Computa la altura del tope de un objeto trackeado sobre el piso.

    Para un par estéreo cenital, la menor profundidad válida dentro de un
    bbox de detección corresponde al punto más cercano — típicamente el tope
    de la cabeza de una persona. La altura sobre el piso es el complemento
    de esa profundidad:

        height = mounting_height - near_depth

    Args:
        near_depth_mm: Profundidad en el punto más cercano de la detección
            (ej: min_depth_at_bbox). Debe ser > 0 para ser válida.
        mounting_height_mm: Distancia camera-to-floor del install.

    Returns:
        Altura de la cabeza sobre el piso en milímetros, o None si las
        entradas son inválidas o la altura computada es negativa (indicaría
        que el objeto está debajo del piso, lo cual significa profundidad mala).
    """
    if near_depth_mm <= 0 or mounting_height_mm <= 0:
        return None
    height_mm = mounting_height_mm - near_depth_mm
    if height_mm < 0:
        return None
    return float(height_mm)


def classify_height(
    head_height_mm: Optional[float],
    adult_min_mm: float,
) -> str:
    """Clasifica una persona por altura de cabeza: adult / child / unknown.

    Args:
        head_height_mm: Salida de head_height_above_floor(). None → unknown.
        adult_min_mm: Threshold en milímetros. height >= threshold → adult.

    Returns:
        "adult" | "child" | "unknown".
    """
    if head_height_mm is None:
        return "unknown"
    return "adult" if head_height_mm >= adult_min_mm else "child"


def aggregate_height_class(samples: list[str]) -> str:
    """Estabiliza clasificaciones per-frame en un único verdict per-track.

    Usa voto por mayoría sobre las clasificaciones sampleadas del track,
    ignorando samples "unknown". Empates (counts iguales adult/child) se
    resuelven a la última observación no-unknown — sesgado hacia la
    profundidad más reciente, que suele ser la más limpia (track ya
    establecido, bbox estable).

    Args:
        samples: Lista de clasificaciones per-frame (valores de classify_height).

    Returns:
        Clasificación final del track: "adult", "child", o "unknown"
        cuando no hay samples no-unknown.
    """
    valid = [s for s in samples if s != "unknown"]
    if not valid:
        return "unknown"
    adult = sum(1 for s in valid if s == "adult")
    child = sum(1 for s in valid if s == "child")
    if adult > child:
        return "adult"
    if child > adult:
        return "child"
    return valid[-1]

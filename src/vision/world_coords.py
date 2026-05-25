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


# NOTA: las funciones `classify_height` y `aggregate_height_class` se
# eliminaron en la migración 2026-05-26-drop-height-class. La
# categorización adulto/niño ahora vive centralizada en la función SQL
# `height_class(height_m)` que se aplica server-side sobre
# `count_events.height_m` (mediana ya estabilizada del detection_history
# del track). El device solo persiste la medición cruda — ni el live
# preview categoriza, solo muestra el número en metros.

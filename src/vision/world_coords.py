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


def project_to_floor(
    head_pixel: tuple[float, float],
    head_height_mm: float,
    mounting_height_mm: float,
    cx_px: float,
    cy_px: float,
) -> tuple[float, float]:
    """Proyecta un pixel de cabeza a donde caerían los pies en image space.

    Background — parallax en zenith fisheye
    ---------------------------------------
    Con el bracket apuntando recto al piso, la línea desde el principal
    point a través del centro del lens es el único rayo con parallax
    cero: una persona parada justo debajo del nadir proyecta cabeza y
    pies en el mismo pixel. En todo otro lado, cabeza y pies caen en
    pixels *distintos* porque están a distintas distancias de la cámara
    — la cabeza está más cerca, así que proyecta más lejos del principal
    point que los pies.

    El bbox del detector NO contiene los pies (quedan escondidos bajo el
    torso en vista cenital). Entonces si usamos ``bbox.center`` como el
    punto de tracking y perseguimos una línea virtual dibujada a nivel
    del piso, disparamos la línea cuando la *proyección del
    hombro/torso* la cruza — lo que puede pasar decenas de centímetros
    antes que los pies en realidad, dependiendo de la excentricidad del
    FOV y la altura de montaje. A 60° de excentricidad (borde del frame)
    ese error es ~110 cm para una persona de 1.7 m bajo un mount de 3 m.

    Math — pinhole projection de una línea vertical
    -----------------------------------------------
    Cámara a altura ``H`` sobre piso, principal point ``(cx, cy)``.
    Un punto a altura ``h`` sobre el piso está a ``Z = H - h`` del
    centro óptico de la cámara. La pinhole projection de un punto 3D
    ``(X, Y, Z)`` es ``(cx + f*X/Z, cy + f*Y/Z)``. El mismo ``(X, Y)``
    físico proyectado a la profundidad de la cabeza ``Z_head = H -
    h_head`` versus la profundidad del piso ``Z_foot = H`` difiere
    solamente por el multiplicador ``Z_head / H`` — así que el pixel
    de los pies es el pixel de la cabeza escalado hacia el principal
    point por ese factor.

    Verificación:
    - Nadir (``head_pixel == (cx, cy)``) → ``foot_pixel == (cx, cy)``.
    - Periferia → shift hacia el principal point, magnitud = parallax.
    - Cuando ``h_head → 0``, ``Z_head → H``, scale → 1, no hay shift
      (la cabeza ya está en el piso).

    La función opera sobre el plano de imagen rectificado, así que
    ``(cx_px, cy_px)`` son las coordenadas del principal point de la
    cámara izquierda rectificada (es decir, ``P1[0,2]``, ``P1[1,2]``
    del ``.npz`` de calibración). La aproximación pinhole alcanza acá
    porque el rectification map ya removió la distorsión del lens — la
    imagen rectificada ES una pinhole projection del mundo.

    Args:
        head_pixel: ``(u, v)`` de la cabeza en pixels del frame rectificado.
        head_height_mm: Altura de la cabeza de la persona sobre el piso (mm).
            Viene de :func:`head_height_above_floor`.
        mounting_height_mm: Altura de la cámara sobre el piso (mm).
        cx_px, cy_px: Principal point de la cámara izquierda rectificada (px).

    Returns:
        ``(u_foot, v_foot)`` — el pixel donde proyectarían los pies de
        la persona si la cámara pudiese ver a través del torso. Con
        inputs degenerados (mount no positivo, altura de cabeza no
        positiva, cabeza igual o por encima de la cámara) devuelve el
        input sin cambios para que el caller pueda caer al comportamiento
        del centroide.
    """
    if mounting_height_mm <= 0 or head_height_mm <= 0:
        return (float(head_pixel[0]), float(head_pixel[1]))
    z_head = mounting_height_mm - head_height_mm
    if z_head <= 0:
        # Cabeza por encima del plano de la cámara (más alta que el
        # mount, o ruido del sensor). La proyección hacia el principal
        # point invertiría el signo; el comportamiento safe es caer al
        # pixel de input.
        return (float(head_pixel[0]), float(head_pixel[1]))
    scale = z_head / mounting_height_mm
    u_h, v_h = float(head_pixel[0]), float(head_pixel[1])
    u_foot = cx_px + (u_h - cx_px) * scale
    v_foot = cy_px + (v_h - cy_px) * scale
    return (u_foot, v_foot)


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

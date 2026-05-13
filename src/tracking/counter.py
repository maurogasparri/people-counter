"""Lógica de conteo para personas trackeadas.

Un único :class:`Counter` parametrizado por:

- Un ROI rectangular opcional (gate of interest — los tracks fuera del ROI
  se ignoran).
- Uno o más segmentos direccionales :class:`Line`. Cada línea lleva labels
  per-dirección: un cruce en una dirección configurada emite el label
  asociado; los cruces en direcciones no configuradas se ignoran sin ruido
  (gates one-way).

Un track se cuenta cuando:

  1. Entra al ROI desde afuera (o aparece en frame si no hay ROI).
  2. Cruza una de las líneas configuradas en una dirección con label
     mientras está dentro del ROI.
  3. Sale del ROI por el lado opuesto (o sale del frame si no hay ROI).

Cuando dispara (3), la meta del track se resetea así el mismo track puede
contar otro ciclo completo más tarde — importante cuando una persona entra
y sale enseguida sin dejar el frame de la cámara el tiempo suficiente como
para que el track muera.

``build_counter(config)`` construye un :class:`Counter` a partir de YAML.
Schema:

    counter:
      roi:                                # opcional
        x_min: 100
        x_max: 1050
        y_min: 150
        y_max: 500
      lines:
        - from: [200, 300]
          to:   [500, 300]
          labels:
            top_to_bottom: ingress
            bottom_to_top: egress
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.tracking.tracker import CONFIRMED, PENDING, Track
from src.vision.world_coords import project_to_floor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses públicos
# ---------------------------------------------------------------------------


@dataclass
class CountEvent:
    """Un evento de conteo."""

    track_id: int
    direction: str  # el label configurado en la línea para este cruce
    timestamp: float
    position_y: float
    # Atributos opcionales per-track que se populan cuando el classifier
    # está enabled. "unknown" cuando faltan datos de height (sin profundidad,
    # classifier disabled).
    height_class: str = "unknown"
    # Mediana de head height (m) y head depth (m) a lo largo del historial
    # de detecciones del track. None cuando no se sampleó profundidad
    # (classifier disabled o cada frame detectado cayó fuera del depth map).
    # Útil para analytics downstream — total_in/out solo no te dice el mix
    # demográfico.
    height_m: Optional[float] = None
    head_depth_m: Optional[float] = None
    # Mediana de confidence YOLO a lo largo del historial de detecciones del
    # track. Permite filtrar downstream eventos de baja confidence (probables
    # falsos positivos o poses marginales).
    confidence: Optional[float] = None


_HORIZONTAL_DIRECTIONS = ("top_to_bottom", "bottom_to_top")
_VERTICAL_DIRECTIONS = ("left_to_right", "right_to_left")


@dataclass
class Line:
    """Segmento de línea de conteo axis-aligned con labels per-dirección.

    ``from_xy`` y ``to_xy`` definen los endpoints del segmento. El
    segmento tiene que ser axis-aligned (puramente horizontal o
    vertical) — segmentos oblicuos no se soportan porque los nombres de
    dirección amigables al operador (``top_to_bottom`` /
    ``bottom_to_top`` para horizontal, ``left_to_right`` /
    ``right_to_left`` para vertical) solo tienen sentido sobre
    segmentos axis-aligned.

    ``labels`` mapea un string de dirección al label emitido cuando un
    track cruza el segmento en esa dirección. Las direcciones ausentes
    del mapa son *gates one-way*: un cruce en esa dirección se ignora
    silenciosamente. Esta es la forma natural de modelar dos puertas
    físicamente separadas (una IN, otra OUT) en el mismo frame.
    """

    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    labels: dict[str, str]
    orientation: str = field(init=False)
    _line_pos: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x1, y1 = float(self.from_xy[0]), float(self.from_xy[1])
        x2, y2 = float(self.to_xy[0]), float(self.to_xy[1])
        self.from_xy = (x1, y1)
        self.to_xy = (x2, y2)
        if y1 == y2 and x1 != x2:
            self.orientation = "horizontal"
            self._line_pos = y1
            valid = _HORIZONTAL_DIRECTIONS
        elif x1 == x2 and y1 != y2:
            self.orientation = "vertical"
            self._line_pos = x1
            valid = _VERTICAL_DIRECTIONS
        else:
            raise ValueError(
                f"Line segment {self.from_xy}->{self.to_xy} must be axis-aligned "
                "(purely horizontal or vertical)."
            )
        if not self.labels:
            raise ValueError(
                f"Line {self.from_xy}->{self.to_xy} has no direction labels — "
                "configure at least one of "
                f"{valid}."
            )
        for direction in self.labels:
            if direction not in valid:
                raise ValueError(
                    f"Direction {direction!r} invalid for {self.orientation} "
                    f"line. Valid: {valid}."
                )

    # ----------------------------------------------------------------- API
    def side_of(self, cx: float, cy: float) -> int:
        """Devuelve +1, -1, o 0 según el lado de la línea en que está el punto.

        Convención de signo: para una línea horizontal, ``-1`` significa
        "arriba" (menor ``y``), ``+1`` significa "abajo" (mayor ``y``);
        para una línea vertical, ``-1`` es "izquierda" (menor ``x``) y
        ``+1`` es "derecha" (mayor ``x``). Cero significa que el punto
        cae exactamente en la línea (edge case).
        """
        if self.orientation == "horizontal":
            if cy < self._line_pos:
                return -1
            if cy > self._line_pos:
                return 1
            return 0
        if cx < self._line_pos:
            return -1
        if cx > self._line_pos:
            return 1
        return 0

    def within_segment(self, cx: float, cy: float) -> bool:
        """True si ``(cx, cy)`` proyecta dentro del extent del segmento
        (entre los endpoints, no solo sobre la línea infinita). Se usa
        para ignorar cruces que pasan el plano de la línea afuera del
        gate real."""
        if self.orientation == "horizontal":
            lo, hi = sorted([self.from_xy[0], self.to_xy[0]])
            return lo <= cx <= hi
        lo, hi = sorted([self.from_xy[1], self.to_xy[1]])
        return lo <= cy <= hi

    def crossing_label(self, prev_side: int, new_side: int) -> Optional[str]:
        """Mapea una transición de lado al label configurado, o ``None``
        si la dirección no tiene label (gate one-way, dirección opuesta)."""
        if prev_side == 0 or new_side == 0 or prev_side == new_side:
            return None
        if self.orientation == "horizontal":
            direction = "top_to_bottom" if prev_side == -1 else "bottom_to_top"
        else:
            direction = "left_to_right" if prev_side == -1 else "right_to_left"
        return self.labels.get(direction)


# ---------------------------------------------------------------------------
# Helpers de agregación (metadata per-event extraída del track)
# ---------------------------------------------------------------------------


def _aggregate_height_class_from_track(track: Track) -> str:
    """Toma los samples per-frame de height_class de la metadata del
    track y elige un verdict por mayoría para el count event. Devuelve
    "unknown" si al tracker no se le pasó metadata de clasificación
    (feature disabled).
    """
    history = track.meta.get("detection_history", [])
    if not history:
        return "unknown"
    from src.vision.world_coords import aggregate_height_class

    samples = [rec.get("height_class", "unknown") for rec in history]
    return aggregate_height_class(samples)


def _aggregate_height_m_from_track(track: Track) -> Optional[float]:
    """Mediana de head height en metros sobre el detection history.
    None si ningún sample tiene head_height_mm medido (classifier
    disabled, sin depth)."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("head_height_mm")
        for rec in history
        if rec.get("head_height_mm") is not None
    ]
    if not samples:
        return None
    samples.sort()
    median_mm = samples[len(samples) // 2]
    return float(median_mm) / 1000.0


def _aggregate_head_depth_m_from_track(track: Track) -> Optional[float]:
    """Mediana de head depth (distancia del lens al tope de cabeza) en metros."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("near_depth_mm")
        for rec in history
        if rec.get("near_depth_mm") is not None and rec.get("near_depth_mm") > 0
    ]
    if not samples:
        return None
    samples.sort()
    median_mm = samples[len(samples) // 2]
    return float(median_mm) / 1000.0


def _latest_head_height_and_bbox(
    track: Track,
) -> tuple[Optional[float], Optional[tuple[float, float, float, float]]]:
    """(head_height_mm, bbox) válido más reciente del detection history
    del track. Lo usa :meth:`Counter._tracking_point` para manejar la
    proyección parallax-corrected del footpoint.

    Recorre la historia de atrás hacia adelante así un valor stale
    nunca desplaza a uno fresco, y devuelve ``head_height_mm`` solo
    cuando es positivo (que sea None es la señal de upstream de que el
    depth pick fue implausible — ver el height sanity gate en
    ``main.py``).
    """
    history = track.meta.get("detection_history", [])
    head_mm: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    for rec in reversed(history):
        if head_mm is None:
            v = rec.get("head_height_mm")
            if v is not None and float(v) > 0:
                head_mm = float(v)
        if bbox is None:
            b = rec.get("bbox")
            if b is not None and len(b) == 4:
                bbox = (
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(b[3]),
                )
        if head_mm is not None and bbox is not None:
            break
    return head_mm, bbox


def _aggregate_confidence_from_track(track: Track) -> Optional[float]:
    """Mediana de confidence YOLO sobre el detection history del track."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("confidence") for rec in history if rec.get("confidence") is not None
    ]
    if not samples:
        return None
    samples.sort()
    return float(samples[len(samples) // 2])


# ---------------------------------------------------------------------------
# ROI validation
# ---------------------------------------------------------------------------


def _validate_roi(
    roi: dict[str, float],
) -> tuple[float, float, float, float]:
    try:
        x_min = float(roi["x_min"])
        x_max = float(roi["x_max"])
        y_min = float(roi["y_min"])
        y_max = float(roi["y_max"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"counter.roi malformed: {e}") from e
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(f"counter.roi requires x_min<x_max and y_min<y_max, got {roi}")
    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class Counter:
    """Counter ROI + N líneas direccionales.

    La meta del track se keea bajo ``META_KEY``. El counter maneja:

    - ``inside``: si el track está actualmente dentro del ROI.
    - ``last_label``: el label más reciente seteado por un cruce de
      línea válido durante la visita actual al ROI. Se resetea en
      transiciones entry/exit.
    - ``line_sides``: cache per-line del "lado" del tracking point del
      frame previo. Se usa para detectar transiciones de lado que
      impliquen un cruce.
    - ``proj_active``: si los ``line_sides`` del frame previo se
      computaron usando el footpoint parallax-corrected (True) o el
      centroide bbox raw (False). Toggling entre convenciones se
      maneja resnapshoteando los lados en vez de compararlos cross.

    Tracking point — parallax-corrected cuando se puede
    ---------------------------------------------------
    El bbox del detector en zenith fisheye encierra cabeza + hombros +
    torso hasta aproximadamente la cintura; los pies no son visibles
    desde arriba. Usar el centroide (≈ hombros) como tracking point
    hace que el cruce de línea dispare cuando la *proyección del
    hombro* de la persona — no los pies — pasa la línea. Con un mount
    de 3 m eso son ~1 m de error en el borde del frame (60° de
    excentricidad).

    Cuando la metadata per-track lleva un ``head_height_mm`` válido de
    SGBM y el principal point de calibración + altura de montaje están
    disponibles, el counter usa :func:`project_to_floor` para escalar
    el head pixel hacia el principal point por ``Z_head / mount`` y
    cuenta el cruce del *foot* pixel proyectado. La altura 3D que SGBM
    nos provee permite recuperar el footpoint geométricamente, evitando
    el sesgo sistemático de contar por centroide/hombros bajo lentes
    cenitales con FOV ancho.

    Cuando la proyección no es viable para un track (sin head height
    todavía, sin calibración plumbed in, mount=0), el counter cae al
    comportamiento de centroide. Las dos convenciones nunca se mezclan
    dentro de la comparación de lados de un único frame:
    ``proj_active`` registra cuál se usó para los lados cacheados, y
    un flip de convención resnapshotea los lados sin emitir un cruce
    fantasma.
    """

    META_KEY = "counter"

    # Gate de mediana de confidence per-track. Bajo este threshold los
    # campos demográficos del CountEvent (height_class, height_m,
    # head_depth_m) se reportan como unknown/None — el bbox fue marginal
    # a lo largo del track y la altura derivada de SGBM no es confiable.
    # 0.5 separa razonablemente las pasadas tipo "v2 normal" (0.6-0.8)
    # de las degradadas por motion blur (0.3-0.45 que vimos en pasadas
    # rápidas). El conteo en sí (in/out) NO se afecta — solo la
    # clasificación adult/child se vuelve conservadora.
    HEIGHT_CONFIDENCE_GATE = 0.5

    def __init__(
        self,
        lines: list[Line],
        roi: Optional[dict[str, float]] = None,
        *,
        mounting_height_mm: Optional[float] = None,
        principal_point: Optional[tuple[float, float]] = None,
        min_crossing_movement_px: float = 0.0,
    ) -> None:
        """
        Args:
            lines: Al menos una línea de conteo axis-aligned.
            roi: ROI rectangular opcional gateando la región contada.
            mounting_height_mm: Altura de la cámara sobre el piso (mm).
                Setear esto junto con ``principal_point`` para habilitar
                el tracking de footpoint parallax-corrected. Cuando
                cualquiera de los dos es ``None`` el counter cae a
                tracking por centroide del bbox — preserva la semántica
                legacy para tests y runs pre-calibración.
            principal_point: ``(cx, cy)`` de la cámara izquierda
                rectificada en pixels. Viene de ``P1[0,2], P1[1,2]``
                del ``.npz`` de calibración estéreo. Tiene que estar
                en el mismo pixel space que las posiciones del
                tracking (es decir, la resolución runtime rectificada).
            min_crossing_movement_px: Threshold de debounce — si el
                track se movió menos que este valor (en pixels) entre
                frames consecutivos *dentro* del ROI, los flips de lado
                de línea de este frame se ignoran. Evita falsos cruces
                por jitter del tracker cerca de la línea (problema
                reconocido en sistemas de conteo cenital sin debounce).
                ``0.0`` (default) deshabilita el debounce — comportamiento
                idéntico al pre-feature. Valores típicos para 30fps a
                1m/s ≈ 10px/frame: ``2.0–3.0`` filtra jitter sin
                rechazar movimiento normal.
        """
        if not lines:
            raise ValueError("Counter requires at least one line.")
        self._lines: list[Line] = list(lines)
        self._roi: Optional[tuple[float, float, float, float]] = (
            _validate_roi(roi) if roi else None
        )
        # Params de proyección de footpoint. Ambos tienen que estar
        # populados para que la proyección se active; si no, el counter
        # cae al modo centroide (sin corrección de parallax).
        self._mounting_height_mm: Optional[float] = (
            float(mounting_height_mm)
            if mounting_height_mm is not None and float(mounting_height_mm) > 0
            else None
        )
        self._principal_point: Optional[tuple[float, float]] = (
            (float(principal_point[0]), float(principal_point[1]))
            if principal_point is not None
            else None
        )
        self._min_crossing_movement_px: float = max(0.0, float(min_crossing_movement_px))
        all_labels: set[str] = set()
        for line in self._lines:
            all_labels.update(line.labels.values())
        self._totals: dict[str, int] = {label: 0 for label in all_labels}

    # ----------------------------------------------------------------- API
    @property
    def lines(self) -> list[Line]:
        """Las líneas configuradas (copy read-only)."""
        return list(self._lines)

    @property
    def roi(self) -> Optional[tuple[float, float, float, float]]:
        """ROI como ``(x_min, x_max, y_min, y_max)`` o ``None`` si unset."""
        return self._roi

    @property
    def total_in(self) -> int:
        """Count de eventos ``ingress``. Alias de conveniencia que usa
        el resto del pipeline (telemetría, eventos MQTT)."""
        return self._totals.get("ingress", 0)

    @property
    def total_out(self) -> int:
        """Count de eventos ``egress``. Alias de conveniencia."""
        return self._totals.get("egress", 0)

    @property
    def totals(self) -> dict[str, int]:
        """Todos los totales keyed por label. Incluye cualquier label
        custom seteado en el mapa ``labels`` de la línea (no solo
        ``ingress``/``egress``)."""
        return dict(self._totals)

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks.values():
            ev = self._process_track(track)
            if ev is not None:
                events.append(ev)
        return events

    def reset_daily(self) -> None:
        for k in self._totals:
            self._totals[k] = 0

    # ------------------------------------------------------------- internal
    def _inside_roi(self, cx: float, cy: float) -> bool:
        if self._roi is None:
            return True
        x_min, x_max, y_min, y_max = self._roi
        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def _tracking_point(self, track: Track) -> tuple[float, float, bool]:
        """Elige el pixel usado para ROI + cruce de línea este frame.

        Devuelve ``(x, y, projected)`` donde ``projected`` es True
        cuando se usó el footpoint parallax-corrected. El caller guarda
        ``projected`` en la meta del track así el próximo frame puede
        detectar un flip de convención y re-snapshotear los line sides
        cacheados.

        Reglas de selección:

        1. Si el counter no se construyó con params de calibración
           (``mounting_height_mm`` + ``principal_point``), caer al
           centroide. Preserva el comportamiento legacy para tests y
           runs pre-calibración.
        2. Si el track tiene un ``head_height_mm`` válido en su
           metadata de detección más reciente, proyectar el *tope* del
           bbox (head pixel) al piso. El tope del bbox está más cerca
           de la cabeza real que el centroide en zenith fisheye — el
           centro del bbox cae aproximadamente sobre los hombros, así
           que proyectar el centroide subestimaría el shift de
           parallax por ~30%. Cae al X del centroide cuando falta bbox.
        3. Si no, centroide.

        Tomar head height del registro de detección MÁS RECIENTE (no
        la mediana) mantiene la proyección responsive a cambios de
        depth — una persona que se agacha tiene menos head height;
        deberíamos trackearla donde están sus pies ahora, no donde
        estaba su mediana. El speckle ya está filtrado upstream por
        ``head_depth_in_bbox`` (gates antropométricos) y el sanity
        check de altura en ``main.py`` (None-out de valores
        implausibles), así que latest es safe.
        """
        cx = float(track.positions[-1][0])
        cy = float(track.positions[-1][1])

        if self._mounting_height_mm is None or self._principal_point is None:
            return cx, cy, False

        head_mm, bbox = _latest_head_height_and_bbox(track)
        if head_mm is None or head_mm <= 0:
            return cx, cy, False

        # Head pixel = centro del tope del bbox. En zenith fisheye la
        # cabeza cae cerca del borde superior del bbox de persona (el
        # bbox se extiende hacia abajo a través de torso/cintura).
        # Cuando bbox no está disponible (meta vieja), caer al X del
        # centroide con el Y del centroide — mejor que nada, y la
        # proyección igual corrige la mayor parte del parallax porque
        # lo que importa es la dirección radial.
        if bbox is not None:
            x1, y1, x2, _y2 = bbox
            head_px = ((float(x1) + float(x2)) / 2.0, float(y1))
        else:
            head_px = (cx, cy)

        cx_pp, cy_pp = self._principal_point
        u_foot, v_foot = project_to_floor(
            head_px,
            head_mm,
            self._mounting_height_mm,
            cx_pp,
            cy_pp,
        )
        # Si project_to_floor rechazó los inputs (degenerados),
        # devuelve el head pixel sin cambios. No podemos llamar safely
        # a eso "footpoint" — caer al centroide así la flag de
        # convención refleja con precisión lo que usamos.
        if (u_foot, v_foot) == head_px and head_px != (cx, cy):
            return cx, cy, False
        return u_foot, v_foot, True

    def _process_track(self, track: Track) -> Optional[CountEvent]:
        # Los tracks CANDIDATE son demasiado inestables para contar.
        if track.state not in (CONFIRMED, PENDING):
            return None
        if not track.positions:
            return None

        # Tracking point — footpoint proyectado cuando depth +
        # calibración están disponibles, centroide raw si no.
        # ``projected`` flaggea la convención así un flip cross frames
        # dispara un resnapshot del side cache abajo.
        cx, cy, projected = self._tracking_point(track)
        meta = track.meta.setdefault(self.META_KEY, {})
        was_inside = bool(meta.get("inside", False))
        prev_projected = bool(meta.get("proj_active", False))
        is_inside = self._inside_roi(cx, cy)

        # Memoria de la última posición outside-ROI bajo la convención
        # actual. La entry-fresca abajo la usa para snapshotear sides[]
        # desde la trayectoria de approach (lado inequívoco de donde
        # viene el track) en lugar de la primera detección inside —
        # que con ROI chico + detector miss rate puede caer ya del otro
        # lado de la línea y dejar el cache corrupto, perdiendo el
        # cruce. Solo se actualiza cuando el track está outside; se
        # consume y opcionalmente persiste a través del cycle.
        if not is_inside:
            meta["last_outside_pos"] = (cx, cy)

        # Cache per-line del lado previo (un slot por línea). Se crea
        # lazy; se repara si cambió el count de líneas configurado
        # entre updates.
        sides = meta.get("line_sides")
        if not isinstance(sides, list) or len(sides) != len(self._lines):
            sides = [0] * len(self._lines)
            meta["line_sides"] = sides

        # Flip de convención: el frame previo cacheó sides bajo la
        # *otra* convención de tracking-point (centroide vs footpoint
        # proyectado). Comparar sides cross convenciones puede fabricar
        # un "cruce" — ej. el centroide estaba debajo de la línea, la
        # proyección pone los pies arriba, no pasó movimiento real.
        # Re-snapshotear bajo la nueva convención sin emitir un label
        # este frame; el próximo frame detectará cualquier cruce
        # subsecuente genuino limpiamente.
        convention_flipped = was_inside and (projected != prev_projected)
        if convention_flipped:
            for i, line in enumerate(self._lines):
                sides[i] = line.side_of(cx, cy) if is_inside else 0
            meta["proj_active"] = projected
            # La outside-pos memoria está expresada en la convención
            # vieja; descartarla así la próxima entry-fresca no
            # snapshotea con coordenadas heterogéneas.
            meta.pop("last_outside_pos", None)
            if not is_inside:
                # Tratarlo como un exit benigno (sin contexto de
                # cruce confiable).
                meta["inside"] = False
                meta["last_label"] = None
            return None
        meta["proj_active"] = projected

        if is_inside and not was_inside:
            # Entry fresca: resetear estado del ciclo y snapshotear
            # sides desde la última posición outside-ROI conocida (el
            # lado del cual el track se aproxima — inequívoco). Esto
            # protege contra el case patológico donde, con ROI chico
            # y detector miss rate, la primera detección inside cae
            # ya del otro lado de la línea y dejaría sides[] cacheado
            # del lado equivocado (resultando en cruces no detectados).
            # Fallback al (cx, cy) actual si el track nació inside
            # sin historia outside.
            meta["inside"] = True
            meta["last_label"] = None
            meta["last_track_pos"] = (cx, cy)
            outside_pos = meta.get("last_outside_pos")
            snap_x, snap_y = outside_pos if outside_pos is not None else (cx, cy)
            for i, line in enumerate(self._lines):
                sides[i] = line.side_of(snap_x, snap_y)
            return None

        if is_inside and was_inside:
            # Debounce: si el track se movió menos que el threshold
            # configurado desde el último frame "real", skipear la
            # detección de cruce. Evita falsos cruces por jitter
            # (típicamente <2px) cuando el track está parado cerca de
            # una línea. ``last_track_pos`` solo se actualiza en frames
            # no-debounced, así una secuencia larga de jitter no
            # acumula drift contra la última posición real.
            threshold = self._min_crossing_movement_px
            if threshold > 0:
                last_pos = meta.get("last_track_pos")
                if last_pos is not None:
                    dx = cx - float(last_pos[0])
                    dy = cy - float(last_pos[1])
                    if (dx * dx + dy * dy) < threshold * threshold:
                        # Movimiento sub-threshold — preservar sides[] y
                        # last_label como están, no evaluar cruces.
                        return None

            # Detectar transición de lado en cada línea. El track
            # puede cruzar múltiples líneas en una visita al ROI; el
            # cruce válido más reciente gana (decisión defensiva — en
            # deployments bien configurados las líneas cubren regiones
            # disjuntas, así que esto rara vez importa).
            for i, line in enumerate(self._lines):
                prev_side = sides[i]
                new_side = line.side_of(cx, cy)
                if (
                    prev_side != 0
                    and new_side != 0
                    and prev_side != new_side
                    and line.within_segment(cx, cy)
                ):
                    label = line.crossing_label(prev_side, new_side)
                    if label is not None:
                        meta["last_label"] = label
                sides[i] = new_side
            meta["last_track_pos"] = (cx, cy)
            return None

        # is_inside es False — si *estábamos* inside, este es el frame
        # de exit.
        if was_inside:
            # Detectar cruces sobre la propia transición de exit.
            # Importante si el track salta de inside-un-lado a
            # outside-el-otro en un único frame (low fps + movimiento
            # rápido, o gaps del detector).
            for i, line in enumerate(self._lines):
                prev_side = sides[i]
                new_side = line.side_of(cx, cy)
                if (
                    prev_side != 0
                    and new_side != 0
                    and prev_side != new_side
                    and line.within_segment(cx, cy)
                ):
                    label = line.crossing_label(prev_side, new_side)
                    if label is not None:
                        meta["last_label"] = label
            label = meta.get("last_label")
            # Resetear estado así el mismo track puede contar otro
            # ciclo completo más tarde. El invariante antiglitch
            # vale: contar de nuevo requiere re-entry desde afuera +
            # un cruce de línea con label + exit.
            meta["inside"] = False
            meta["last_label"] = None
            for i in range(len(sides)):
                sides[i] = 0
            if label:
                self._totals[label] = self._totals.get(label, 0) + 1
                logger.debug(
                    "count_event",
                    extra={"track_id": track.track_id, "label": label},
                )
                # Demographics gate: la altura se calcula desde SGBM dentro
                # del bbox, y SGBM degrada con motion blur (matching estéreo
                # falla en bordes desenfocados) y con bbox poco precisos
                # (centroide off-cabeza). Cuando la mediana de confidence
                # del track cae bajo HEIGHT_CONFIDENCE_GATE, el bbox fue
                # marginal a lo largo de la trayectoria y la altura
                # derivada no es confiable — el conteo en sí (dirección,
                # cruce) se mantiene porque solo depende del centroide.
                # Limpiar height_m/head_depth_m a None y height_class a
                # "unknown" así los dashboards no surfacean valores
                # demográficos espurios.
                conf_median = _aggregate_confidence_from_track(track)
                if (
                    conf_median is not None
                    and conf_median < self.HEIGHT_CONFIDENCE_GATE
                ):
                    height_class = "unknown"
                    height_m = None
                    head_depth_m = None
                else:
                    height_class = _aggregate_height_class_from_track(track)
                    height_m = _aggregate_height_m_from_track(track)
                    head_depth_m = _aggregate_head_depth_m_from_track(track)
                return CountEvent(
                    track_id=track.track_id,
                    direction=label,
                    timestamp=time.time(),
                    position_y=cy,
                    height_class=height_class,
                    height_m=height_m,
                    head_depth_m=head_depth_m,
                    confidence=conf_median,
                )
            logger.debug(
                "exit_without_crossing",
                extra={"track_id": track.track_id},
            )
        return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_counter(
    config: dict[str, Any],
    *,
    mounting_height_mm: Optional[float] = None,
    principal_point: Optional[tuple[float, float]] = None,
) -> Counter:
    """Construye el counter desde config YAML.

    Los opcionales ``mounting_height_mm`` y ``principal_point``
    habilitan el tracking de footpoint parallax-corrected (ver
    :class:`Counter` para el rationale). Hay que proveer ambos juntos;
    omitir cualquiera deja al counter en el path legacy de tracking
    por centroide. Se pasa desde ``main.py`` una vez que se cargó
    calibración.

    Schema (estricto):

        counter:
          roi:                              # opcional
            x_min: 100
            x_max: 1050
            y_min: 150
            y_max: 500
          lines:
            - from: [200, 300]
              to:   [500, 300]
              labels:
                top_to_bottom: ingress
                bottom_to_top: egress
    """
    counter_cfg = config.get("counter") or {}
    raw_lines = counter_cfg.get("lines")
    if not raw_lines:
        raise ValueError(
            "counter.lines is required and must not be empty. See the docstring "
            "of build_counter() for the expected schema."
        )
    lines: list[Line] = []
    for idx, raw in enumerate(raw_lines):
        try:
            from_xy = tuple(raw["from"])
            to_xy = tuple(raw["to"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"counter.lines[{idx}]: 'from' and 'to' required as [x, y] "
                f"pairs ({e})."
            ) from e
        if len(from_xy) != 2 or len(to_xy) != 2:
            raise ValueError(
                f"counter.lines[{idx}]: 'from' and 'to' must be [x, y] pairs."
            )
        labels = dict(raw.get("labels") or {})
        lines.append(Line(from_xy=from_xy, to_xy=to_xy, labels=labels))
    return Counter(
        lines=lines,
        roi=counter_cfg.get("roi"),
        mounting_height_mm=mounting_height_mm,
        principal_point=principal_point,
        min_crossing_movement_px=float(
            counter_cfg.get("min_crossing_movement_px", 0.0) or 0.0
        ),
    )

"""Lógica de conteo para personas trackeadas.

Un único :class:`Counter` parametrizado por:

- Un ROI rectangular opcional (gate of interest — los tracks fuera del ROI
  se ignoran).
- Uno o más segmentos direccionales :class:`Line`. Cada línea lleva labels
  per-dirección: un cruce en una dirección configurada emite el label
  asociado; los cruces en direcciones no configuradas se ignoran sin ruido
  (gates one-way).

Un track se cuenta cuando, EN ORDEN:

  1. Entra al ROI desde afuera.
  2. Cruza una de las líneas configuradas en una dirección con label
     mientras está dentro del ROI.
  3. **Sale del ROI** por el lado opuesto.

El conteo dispara recién en (3): la salida del ROI es obligatoria. NO hay
"salida sintética" por muerte del track dentro del ROI — una persona que
cruza la línea pero se queda en el ROI (parada/sentada/dudando en la puerta,
o que el detector pierde por pose fuera de distribución) NO se cuenta. Esto
evita falsos positivos de gente lingering en el umbral.

Sin ROI configurado no hay gate de salida (todo el frame es "inside"), así
que un cruce de línea solo no cuenta — en la práctica siempre se configura
un ROI.

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
    # Posición x del track al momento del cruce. Útil para downstream
    # analytics (clustering espacial de eventos) y para la cancelación
    # U-turn (un evento solo cancela contra otro si AMBOS caen dentro
    # del ROI — la x es necesaria para el test geométrico).
    position_x: float = 0.0
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

    Tracking point — centroide del bbox
    -----------------------------------
    El counter usa el centroide del bbox como tracking point para ROI +
    cruce de línea. El montaje es cenital sobre el umbral de la puerta a
    medir, así que el cruce ocurre cerca del nadir, donde el paralaje es
    ~cero (cabeza y pies proyectan al mismo pixel) — el centroide es el
    foot-point efectivo ahí. La corrección de paralaje image-space que
    existía antes (footpoint proyectado vía ``project_to_floor``) se
    retiró: comprimía la trayectoria hacia el principal point y rompía
    los INs en geometría de puerta central, y no aportaba en la zona del
    cruce. La altura 3D de SGBM se sigue usando, pero solo para
    clasificar adult/child (no para la posición del cruce).
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

    # Ventana temporal para cancelar pares opuestos IN/OUT dentro del
    # ROI (U-turn). Eventos opuestos del mismo ROI dentro de esta
    # ventana se cancelan mutuamente — captura "persona dudó en la
    # entrada, cruzó y volvió enseguida" o fragmentación de track cerca
    # de la línea (track A cruza IN, jitter parte el track, track B
    # cruza OUT). Tests patchean este attr de clase para validar
    # comportamiento fuera de ventana.
    UTURN_WINDOW_SECONDS = 5.0

    def __init__(
        self,
        lines: list[Line],
        roi: Optional[dict[str, float]] = None,
        *,
        min_crossing_movement_px: float = 0.0,
    ) -> None:
        """
        Args:
            lines: Al menos una línea de conteo axis-aligned.
            roi: ROI rectangular opcional gateando la región contada.
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
        self._min_crossing_movement_px: float = max(0.0, float(min_crossing_movement_px))
        all_labels: set[str] = set()
        for line in self._lines:
            all_labels.update(line.labels.values())
        self._totals: dict[str, int] = {label: 0 for label in all_labels}
        # Breakdown horario en memoria: hour (0-23) -> {label: count}. Para el
        # live preview. Reset diario (reset_daily). El histórico persistente
        # vive en RDS (count_events.bucket_15min), no en el borde.
        self._hourly: dict[int, dict[str, int]] = {}
        # U-turn cancellation: el ROI ACTÚA como zona de cancelación.
        # Eventos opuestos del mismo ROI dentro de ``UTURN_WINDOW_SECONDS``
        # se cancelan mutuamente — captura "persona dudó y volvió" así
        # como fragmentación de track cerca de la línea (track A cruza
        # IN, jitter parte el track, track B cruza OUT). Sin ROI
        # configurado, no hay cancelación (no podemos bound la zona).
        # Mapa label → label opuesto, derivado de las líneas.
        self._opposites: dict[str, str] = {}
        for line in self._lines:
            line_labels = list(line.labels.values())
            if len(line_labels) == 2:
                self._opposites[line_labels[0]] = line_labels[1]
                self._opposites[line_labels[1]] = line_labels[0]
        # Cache de eventos recientes para U-turn matching. Cada entrada:
        # ``(track_id, label, x, y, timestamp)``. Se purga en cada check
        # eliminando los eventos fuera de la ventana. Tamaño acotado
        # naturalmente por la ventana corta (típico 5s) + el flujo de
        # tráfico (decenas de eventos/min máximo por sucursal típica).
        self._recent_events: list[tuple[int, str, float, float, float]] = []

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

    def hourly_in_out(self) -> list[dict]:
        """Breakdown horario de HOY: lista de ``{hour, in, out}`` ordenada por
        hora, solo las horas con actividad. ``in`` = label ``ingress``,
        ``out`` = ``egress``. Tracking en memoria con reset diario; el borde
        NO persiste el histórico (eso vive en RDS, ``count_events``)."""
        rows = []
        for hour in sorted(self._hourly):
            n_in = self._hourly[hour].get("ingress", 0)
            n_out = self._hourly[hour].get("egress", 0)
            # Saltear horas que quedaron en cero (ej. un cruce cancelado por
            # U-turn) — no ensuciar la tabla con filas 0/0.
            if n_in or n_out:
                rows.append({"hour": hour, "in": n_in, "out": n_out})
        return rows

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        # Un track se cuenta SOLO cuando su centroide cruza la linea y luego
        # SALE del ROI (ver _process_track). NO hay "salida sintetica" por
        # muerte del track dentro del ROI: una persona parada/sentada/dudando
        # en el ROI (que el detector pierde por pose fuera de distribucion, o
        # que mueve la cabeza cruzando la linea sin trasladarse) NO debe
        # contar. Requerir la salida del ROI es la semantica canonica del gate
        # (entrar -> cruzar -> salir) y elimina ese falso positivo. Trade-off
        # aceptado: una pasada limpia ocluida JUSTO dentro del ROI antes de
        # salir no se cuenta (raro en montaje cenital sobre puerta).
        events: list[CountEvent] = []
        for track in tracks.values():
            ev = self._process_track(track)
            if ev is not None:
                events.append(ev)
        return events

    def reset_daily(self) -> None:
        for k in self._totals:
            self._totals[k] = 0
        self._hourly.clear()
        # Tirar el cache de eventos recientes — un reset es un boundary
        # semántico, no queremos que un IN del día anterior cancele un OUT
        # del nuevo día.
        self._recent_events.clear()

    # ------------------------------------------------------------- internal
    def _try_cancel_uturn(
        self,
        track_id: int,
        label: str,
        x: float,
        y: float,
        timestamp: float,
    ) -> bool:
        """Intenta cancelar el evento contra un opuesto reciente dentro
        del mismo ROI. Devuelve True si canceló (caller debe abortar la
        emisión).

        El ROI actúa como zona de cancelación: si el evento actual y un
        evento opuesto reciente ambos caen dentro del ROI y la ventana
        temporal no expiró, se cancelan mutuamente. Sin ROI configurado
        no hay cancelación (no podemos bound la zona).
        """
        if self._roi is None:
            return False
        if not self._inside_roi(x, y):
            return False
        opposite = self._opposites.get(label)
        if opposite is None:
            return False
        cutoff = timestamp - self.UTURN_WINDOW_SECONDS
        # Purgar eventos fuera de ventana.
        self._recent_events = [
            e for e in self._recent_events if e[4] >= cutoff
        ]
        for i, recent in enumerate(self._recent_events):
            r_tid, r_label, r_x, r_y, _r_ts = recent
            if r_label != opposite:
                continue
            if not self._inside_roi(r_x, r_y):
                continue
            # U-turn detectado: revertir el evento previo y abortar el
            # actual. No discriminamos por track_id — el escenario
            # típico es que el track muera entre el IN y el OUT y la
            # segunda mitad sea un track nuevo. Si fuera mismo track,
            # también cancela (idempotente).
            self._totals[opposite] = max(
                0, self._totals.get(opposite, 0) - 1
            )
            # Mantener el breakdown horario consistente con los totales: el
            # evento opuesto cancelado se contó en su hora; revertirlo ahí.
            _hb = self._hourly.get(time.localtime(_r_ts).tm_hour)
            if _hb and _hb.get(opposite, 0) > 0:
                _hb[opposite] -= 1
            del self._recent_events[i]
            logger.debug(
                "uturn_cancellation",
                extra={
                    "new_label": label,
                    "cancelled_label": opposite,
                    "new_track_id": track_id,
                    "cancelled_track_id": r_tid,
                },
            )
            return True
        return False

    def _record_event_for_uturn(
        self,
        track_id: int,
        label: str,
        x: float,
        y: float,
        timestamp: float,
    ) -> None:
        """Inserta el evento emitido en el cache de recent_events si el
        ROI está configurado. Eventos fuera del ROI no se cachean porque
        nunca podrían cancelarse contra nada (el lookup siempre requiere
        ambos dentro del ROI)."""
        if self._roi is None:
            return
        if not self._inside_roi(x, y):
            return
        self._recent_events.append((track_id, label, x, y, timestamp))

    def _inside_roi(self, cx: float, cy: float) -> bool:
        if self._roi is None:
            return True
        x_min, x_max, y_min, y_max = self._roi
        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def _tracking_point(self, track: Track) -> tuple[float, float]:
        """Pixel usado para ROI + cruce de línea: el centroide del bbox.

        El montaje es cenital sobre el umbral de la puerta a medir, así
        que el cruce ocurre cerca del nadir donde el paralaje es ~cero —
        el centroide es el foot-point efectivo. La corrección de paralaje
        image-space (footpoint proyectado) se retiró porque comprimía la
        trayectoria hacia el principal point y rompía los INs en puerta
        central, sin aportar en la zona del cruce.
        """
        return float(track.positions[-1][0]), float(track.positions[-1][1])

    def _process_track(self, track: Track) -> Optional[CountEvent]:
        # Los tracks CANDIDATE son demasiado inestables para contar.
        if track.state not in (CONFIRMED, PENDING):
            return None
        if not track.positions:
            return None

        # Tracking point — centroide del bbox (ver _tracking_point).
        cx, cy = self._tracking_point(track)
        meta = track.meta.setdefault(self.META_KEY, {})
        was_inside = bool(meta.get("inside", False))
        is_inside = self._inside_roi(cx, cy)

        # Memoria de la última posición outside-ROI. La entry-fresca
        # abajo la usa para snapshotear sides[] desde la trayectoria de
        # approach (lado inequívoco de donde viene el track) en lugar de
        # la primera detección inside — que con ROI chico + detector miss
        # rate puede caer ya del otro lado de la línea y dejar el cache
        # corrupto, perdiendo el cruce. Solo se actualiza cuando el track
        # está outside; se consume y opcionalmente persiste a través del
        # cycle.
        if not is_inside:
            meta["last_outside_pos"] = (cx, cy)

        # Cache per-line del lado previo (un slot por línea). Se crea
        # lazy; se repara si cambió el count de líneas configurado
        # entre updates.
        sides = meta.get("line_sides")
        if not isinstance(sides, list) or len(sides) != len(self._lines):
            sides = [0] * len(self._lines)
            meta["line_sides"] = sides

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
                        # Anclar la posición del cruce dentro del ROI
                        # para el U-turn lookup. El emit en exit usa esta
                        # posición (no la posición de salida, que está
                        # fuera del ROI por definición).
                        meta["last_crossing_pos"] = (cx, cy)
                sides[i] = new_side
            meta["last_track_pos"] = (cx, cy)
            return None

        # is_inside es False — si *estábamos* inside, este es el frame
        # de exit.
        if was_inside:
            # Detectar cruces sobre la propia transición de exit.
            # Importante si el track salta de inside-un-lado a
            # outside-el-otro en un único frame (low fps + movimiento
            # rápido, o gaps del detector). En este caso anclamos al
            # midpoint del segmento de exit — la posición real del
            # cruce cae entre los dos frames; el midpoint es la mejor
            # aproximación y suele caer dentro del ROI.
            last_inside = meta.get("last_track_pos")
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
                        if last_inside is not None:
                            meta["last_crossing_pos"] = (
                                (float(last_inside[0]) + cx) / 2.0,
                                (float(last_inside[1]) + cy) / 2.0,
                            )
                        else:
                            meta["last_crossing_pos"] = (cx, cy)
            label = meta.get("last_label")
            crossing_pos = meta.get("last_crossing_pos") or (cx, cy)
            # Resetear estado así el mismo track puede contar otro
            # ciclo completo más tarde. El invariante antiglitch
            # vale: contar de nuevo requiere re-entry desde afuera +
            # un cruce de línea con label + exit.
            meta["inside"] = False
            meta["last_label"] = None
            meta["last_crossing_pos"] = None
            for i in range(len(sides)):
                sides[i] = 0
            if label:
                now = time.time()
                cross_x = float(crossing_pos[0])
                cross_y = float(crossing_pos[1])
                # U-turn cancellation: si el evento cae dentro del ROI
                # (zona de cancelación) y matchea un evento opuesto
                # reciente, decrementa el opuesto y aborta este sin
                # emitir count. Captura "persona dudó y volvió" + track
                # fragmentation cerca de la línea sin inflar totales.
                if self._try_cancel_uturn(
                    track.track_id, label, cross_x, cross_y, now
                ):
                    return None
                self._totals[label] = self._totals.get(label, 0) + 1
                _hb = self._hourly.setdefault(time.localtime(now).tm_hour, {})
                _hb[label] = _hb.get(label, 0) + 1
                self._record_event_for_uturn(
                    track.track_id, label, cross_x, cross_y, now
                )
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
                    timestamp=now,
                    position_y=cross_y,
                    position_x=cross_x,
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


def build_counter(config: dict[str, Any]) -> Counter:
    """Construye el counter desde config YAML.

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
        min_crossing_movement_px=float(
            counter_cfg.get("min_crossing_movement_px", 0.0) or 0.0
        ),
    )

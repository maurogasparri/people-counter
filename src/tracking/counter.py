"""Lógica de conteo para personas trackeadas.

Un único :class:`Counter` parametrizado por:

- Una counting zone rectangular opcional (gate of interest — los tracks fuera de la counting zone
  se ignoran).
- Uno o más segmentos direccionales :class:`Line`. Cada línea lleva labels
  per-dirección: un cruce en una dirección configurada emite el label
  asociado; los cruces en direcciones no configuradas se ignoran sin ruido
  (gates one-way).

Un track se cuenta cuando, EN ORDEN:

  1. Entra a la counting zone desde afuera.
  2. Cruza una de las líneas configuradas en una dirección con label
     mientras está dentro de la counting zone.
  3. **Sale de la counting zone**.

El conteo dispara recién en (3): la salida de la counting zone es obligatoria. NO hay
"salida sintética" por muerte del track dentro de la counting zone — una persona que
cruza la línea pero se queda en la counting zone (parada/sentada/dudando en la puerta,
o que el detector pierde por pose fuera de distribución) NO se cuenta. Esto
evita falsos positivos de gente lingering en el umbral.

**Cancelación neta dentro de la counting zone (máquina de estados de la zona):**
durante una visita a la counting zone se acumula un balance NETO de cruces por línea —
cada cruce in-segment hacia un lado suma, hacia el opuesto resta. Al salir
se emite según el SIGNO del neto: si quedó ±1, ese es el sentido contado;
si quedó 0, la persona "fue y vino" (cruzó y re-cruzó en sentido opuesto
dentro de la counting zone) y NO se cuenta. Esto reemplaza el viejo "gana el último
cruce". Un gate one-way (línea con label en una sola dirección) es sticky:
el cruce de vuelta no tiene label, no resta, así que un IN no se cancela
por volver a salir por el lado sin label.

Sin counting zone configurado no hay gate de salida (todo el frame es "inside"), así
que un cruce de línea solo no cuenta — en la práctica siempre se configura
una counting zone.

Cuando dispara (3), la meta del track se resetea así el mismo track puede
contar otro ciclo completo más tarde — importante cuando una persona entra
y sale enseguida sin dejar el frame de la cámara el tiempo suficiente como
para que el track muera.

``build_counter(config)`` construye un :class:`Counter` a partir de YAML.
Schema:

    counter:
      counting_zone:                      # opcional
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
    # analytics (clustering espacial de eventos).
    position_x: float = 0.0
    # Mediana de head height (m) y head depth (m) a lo largo del historial
    # de detecciones del track. None cuando no se sampleó profundidad
    # (classifier disabled o cada frame detectado cayó fuera del depth map).
    # La categorización adulto/niño se aplica server-side vía la función SQL
    # height_class(height_m) sobre count_events. El device persiste solo la
    # medición cruda — el threshold vive centralizado en SQL.
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
# Counting zone validation
# ---------------------------------------------------------------------------


def _validate_counting_zone(
    zone: dict[str, float],
) -> tuple[float, float, float, float]:
    try:
        x_min = float(zone["x_min"])
        x_max = float(zone["x_max"])
        y_min = float(zone["y_min"])
        y_max = float(zone["y_max"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"counter.counting_zone malformed: {e}") from e
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(
            f"counter.counting_zone requires x_min<x_max and y_min<y_max, got {zone}"
        )
    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class Counter:
    """Counter counting_zone + N líneas direccionales.

    La meta del track se keea bajo ``META_KEY``. El counter maneja:

    - ``inside``: si el track está actualmente dentro de la counting zone.
    - ``crossing_net``: balance NETO de cruces por línea durante la
      visita actual a la counting zone. Cada cruce in-segment con label hacia el
      lado +1 suma 1, hacia -1 resta 1. Se resetea en entry; el signo
      al exit decide el evento (0 = fue y vino, no cuenta).
    - ``line_sides``: cache per-line del "lado" del tracking point del
      frame previo. Se usa para detectar transiciones de lado que
      impliquen un cruce.

    Tracking point — centroide del bbox
    -----------------------------------
    El counter usa el centroide del bbox como tracking point para la counting zone +
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
    DEFAULT_HEIGHT_CONFIDENCE_GATE = 0.5

    # Gate del cruce extrapolado por Kalman (track PENDING, sin match real este
    # frame). Sin esta relajación, un track con el detector dropeado en la
    # zona crítica de la línea no registra el cruce → conteo perdido. Con
    # ambos guardrails se distingue "walking → lost → Kalman cruza" del
    # "drift estacionario" (residual velocity, sin movimiento real):
    #   - MAX_KALMAN_CROSS_FRAMES: ≤ ~0.75s @ 20fps sin detección — matchea
    #     ByteTrackMaxTimeLost del incumbent (FFC).
    #   - MIN_KALMAN_CROSS_DISPLACEMENT_PX: desplazamiento decisivo (px) en la
    #     dirección del cruce desde la última posición REAL. El drift residual
    #     con velocity_decay alto no alcanza este umbral.
    MAX_KALMAN_CROSS_FRAMES = 15
    MIN_KALMAN_CROSS_DISPLACEMENT_PX = 30.0

    # Grace DEFAULT antes de disparar el death-emit. Cuando un track desaparece
    # de la dict (LOST), lo metemos en una cola pendiente: si dentro de esta
    # ventana el tracker lo RESUCITA via ghost pool (el mismo ID reaparece
    # adoptado por un track nuevo), CANCELAMOS el death-emit y dejamos que el
    # conteo emita naturalmente en el exit observado. Sin este grace,
    # death-emit y la eventual salida del track resucitado contarían DOS
    # veces. DEBE matchear (o superar levemente) ``adoption_window_frames``
    # del tracker — si la adopción es más larga que el grace, el counter
    # death-emit dispara antes de que el tracker pueda adoptar → doble cuenta.
    # El default 30 matchea el default del tracker; ``main.py`` lo sincroniza
    # explicitamente con ``tracker.adoption_window_frames + 2`` (buffer)
    # cuando construye el counter, así no quedan desfasados ante un tuneo.
    DEFAULT_DEATH_EMIT_GRACE_FRAMES = 30

    # Guards anti-falso-positivo en death-emit. Cubren el caso reportado en
    # producción: persona SENTADA dentro de la counting zone cuya cabeza jitterea
    # atravesando la línea (centroide cruza por ~3-10px sin que haya entrada
    # o salida real del edificio). Sin estos guards, el death-emit dispara
    # porque tecnicamente se registró un cross. Dos checks complementarios:
    #   1. ``had_outside_pos``: el track DEBE haber sido visto fuera de la counting zone
    #      en algún momento de su vida. Si spawneó adentro de la counting zone y nunca
    #      salió, no hay evidencia de que realmente entró/salió del edificio.
    #      Esto filtra sitters + clutter + re-id failures que spawnean tracks
    #      nuevos adentro de la counting zone.
    #   2. ``visit_range >= MIN``: el track debe haber recorrido al menos
    #      este desplazamiento (en px, max de x_range y y_range) durante la
    #      visita. Filtra el edge case donde un track salió brevemente del
    #      counting zone (had_outside_pos=True) y después jittereó adentro sin
    #      movimiento real. Default 80 px ≈ media zancada de adulto a la
    #      altura de montaje típica. **Tuneable per-instancia** via el
    #      kwarg ``min_visit_range_for_death_emit`` del constructor — bajarlo
    #      (ej. 50) en sites con detector flakey donde la mayoría del visit
    #      cae en frames Kalman (no actualizan range). Síntoma diagnóstico
    #      en telemetría: ``death_emit_count = 0`` mientras
    #      ``track_stitching_ratio > 1.3`` → relax este guard.
    DEFAULT_MIN_VISIT_RANGE_FOR_DEATH_EMIT = 80.0

    # Mínimo de frames con detección REAL inside la counting zone antes del
    # emit. Caso piloto 2026-05-24 15:23 (tid=67): track de campera flickeando
    # cuyo detector dispara UN solo frame al borde del counting zone (y=204
    # = y_min) y luego Kalman extrapola 247 px en 589 ms hasta outside abajo,
    # disparando IN espurio. Con un humano real caminando a velocidad normal
    # el detector dispara 3-5+ frames inside; 1 solo frame es señal de
    # flickering del detector sobre clutter (campera, sombra, objeto fijo).
    # Default 0 = filtro desactivado (back-compat). Piloto recomienda 2.
    # Sites con detector muy estable y FPs raros pueden subir a 3.
    DEFAULT_MIN_REAL_INSIDE_FRAMES = 0

    # Filtro anti-FP no-humanos por altura. Si la MEDIANA de head_height_mm del
    # track durante la visita es < este threshold (m), el counter NO emite el
    # conteo. Tracks sin medición de altura (median = None) NUNCA se filtran —
    # preferimos preservar recall en casos donde SGBM falla (motion blur,
    # oclusiones). Default 0.0 = filtro desactivado (back-compat). Tuneable
    # per-site: 1.0 filtra perros + objetos < 1m; 0.7 más conservador.
    # Caso real piloto 2026-05-24: perro caminando por el local cruzaba la
    # línea y disparaba IN; campera en silla con SGBM noise también.
    DEFAULT_MIN_COUNT_HEIGHT_M = 0.0

    def __init__(
        self,
        lines: list[Line],
        counting_zone: Optional[dict[str, float]] = None,
        *,
        min_crossing_movement_px: float = 0.0,
        death_emit_grace_frames: Optional[int] = None,
        min_visit_range_for_death_emit: Optional[float] = None,
        min_count_height_m: Optional[float] = None,
        height_confidence_gate: Optional[float] = None,
        min_real_inside_frames: Optional[int] = None,
    ) -> None:
        """
        Args:
            lines: Al menos una línea de conteo axis-aligned.
            counting_zone: Counting zone rectangular opcional gateando la región
                contada. Lo que el incumbent (FFC) llama TrackingZone — la
                región física del gate.
            min_crossing_movement_px: Threshold de debounce — si el
                track se movió menos que este valor (en pixels) entre
                frames consecutivos *dentro* de la counting zone, los flips de lado
                de línea de este frame se ignoran. Evita falsos cruces
                por jitter del tracker cerca de la línea (problema
                reconocido en sistemas de conteo cenital sin debounce).
                ``0.0`` (default) deshabilita el debounce — comportamiento
                idéntico al pre-feature. Valores típicos para 30fps a
                1m/s ≈ 10px/frame: ``2.0–3.0`` filtra jitter sin
                rechazar movimiento normal.
            death_emit_grace_frames: Frames a esperar antes de disparar
                el death-emit cuando un track desaparece de la dict —
                ventana de chance para que el tracker lo resucite via
                ghost adoption. ``None`` (default) usa
                ``DEFAULT_DEATH_EMIT_GRACE_FRAMES``. Production setup:
                pasar ``tracker.adoption_window_frames + 2`` (buffer) así
                el counter SIEMPRE le da chance a la adopción antes de
                emitir por muerte → evita doble-conteo si el tracker
                eventualmente adopta.
            min_visit_range_for_death_emit: Desplazamiento mínimo (px) que
                el track debe haber recorrido durante la visita a la counting zone para
                que death-emit fire. Solo se actualiza con detecciones
                REALES (no Kalman pushes), así un sitter con cabeza
                jitterosa cerca de la línea no dispara. ``None`` (default)
                usa ``DEFAULT_MIN_VISIT_RANGE_FOR_DEATH_EMIT = 80``. Bajar
                a 50 en sites con detector flakey donde la mayoría del
                visit cae en frames Kalman (síntoma diagnóstico:
                ``death_emit_count = 0`` con ``track_stitching_ratio > 1.3``
                sostenido en telemetría).
            min_count_height_m: Threshold de altura humana (mediana del track,
                en metros) para emitir el conteo. Si la mediana de
                head_height_mm cae por debajo Y la medición existe (no None),
                el counter rechaza el emit. ``None`` (default) usa
                ``DEFAULT_MIN_COUNT_HEIGHT_M = 0.0`` = filtro desactivado.
                Activar en sites con FPs no-humanos persistentes (perros,
                carritos altos, objetos sobre sillas). 1.0 filtra perros y
                objetos < 1m sin perder niños de 4+ años. 0.7 más conservador.
                Tracks sin medición SGBM (median=None) pasan — recall sobre
                precision en ambigüedad.
            height_confidence_gate: Threshold de mediana de YOLO confidence
                del track para que los campos demográficos (height_class,
                height_m, head_depth_m) se reporten en el CountEvent.
                Tracks con median(conf) < threshold reportan ``height_class
                = "unknown"`` y demás campos en None — el bbox fue marginal
                a lo largo de la trayectoria y la altura derivada de SGBM
                no es confiable. **El conteo en sí (in/out) NO se afecta**;
                solo la clasificación demográfica se vuelve conservadora.
                ``None`` (default) usa ``DEFAULT_HEIGHT_CONFIDENCE_GATE
                = 0.5``. Subir a 0.7 en sites con detector flaky donde
                pasadas marginales reportan altura espuria; bajar a 0.3
                solo si el detector está bien calibrado y querés
                demografía de pasadas rápidas también.
            min_real_inside_frames: Mínimo de frames con detección REAL
                inside la counting zone (entry-fresca + frames is_real=True
                durante la visita) antes de que el counter pueda emitir.
                Filtra tracks "fantasma" donde el detector dispara un solo
                frame al borde del counting zone y el Kalman extrapola toda
                la visita. Un humano real genera 3-5+ frames inside; 1 solo
                frame es señal de flickering del detector sobre clutter.
                ``None`` (default) usa ``DEFAULT_MIN_REAL_INSIDE_FRAMES = 0``
                (filtro desactivado, back-compat). 2 filtra single-frame
                entries sin perder caminantes normales; 3 más estricto.
        """
        if not lines:
            raise ValueError("Counter requires at least one line.")
        self._lines: list[Line] = list(lines)
        self._counting_zone: Optional[tuple[float, float, float, float]] = (
            _validate_counting_zone(counting_zone) if counting_zone else None
        )
        self.death_emit_grace_frames: int = (
            int(death_emit_grace_frames)
            if death_emit_grace_frames is not None
            else self.DEFAULT_DEATH_EMIT_GRACE_FRAMES
        )
        self.min_visit_range_for_death_emit: float = (
            float(min_visit_range_for_death_emit)
            if min_visit_range_for_death_emit is not None
            else self.DEFAULT_MIN_VISIT_RANGE_FOR_DEATH_EMIT
        )
        self._min_crossing_movement_px: float = max(
            0.0, float(min_crossing_movement_px)
        )
        self.min_count_height_m: float = (
            float(min_count_height_m)
            if min_count_height_m is not None
            else self.DEFAULT_MIN_COUNT_HEIGHT_M
        )
        self.height_confidence_gate: float = (
            float(height_confidence_gate)
            if height_confidence_gate is not None
            else self.DEFAULT_HEIGHT_CONFIDENCE_GATE
        )
        self.min_real_inside_frames: int = (
            max(0, int(min_real_inside_frames))
            if min_real_inside_frames is not None
            else self.DEFAULT_MIN_REAL_INSIDE_FRAMES
        )
        all_labels: set[str] = set()
        for line in self._lines:
            all_labels.update(line.labels.values())
        self._totals: dict[str, int] = {label: 0 for label in all_labels}
        # Breakdown horario en memoria: hour (0-23) -> {label: count}. Para el
        # live preview. Reset diario (reset_daily). El histórico persistente
        # vive en RDS (count_events.bucket_15min), no en el borde.
        self._hourly: dict[int, dict[str, int]] = {}
        # Snapshot per-track del estado relevante para el death-emit. Lo
        # populamos en check_all (después de _process_track) y lo consultamos
        # cuando un track desaparece de la dict (= muerte): si tenía un cruce
        # registrado y nunca salió de la counting zone, emitimos acá. Ver _emit_on_death.
        self._tracking_state: dict[int, dict] = {}
        # Cola de muertes pendientes con su edad. Cuando un track desaparece
        # de la dict, en vez de death-emit inmediato, lo movemos acá con
        # age=0 y damos ``death_emit_grace_frames`` de margen para que el
        # tracker lo resucite via ghost pool. Si reaparece dentro de la
        # ventana, el death-emit se cancela y el conteo emite naturalmente
        # en el exit. Si no reaparece, al expirar fire death-emit (fallback).
        self._potential_deaths: dict[int, dict] = {}
        # Métrica de stitching para la telemetría: cuántos track_ids distintos
        # vimos cruzar la counting zone hoy. ``stitching_ratio`` = unique_ids / conteos
        # emitidos; >1 indica fragmentación del tracker (un mismo sujeto se
        # spawnea con varios IDs). Reset diario.
        self._seen_track_ids: set[int] = set()
        # Contador de death-emits disparados hoy. Sirve de canary diferenciador:
        # ratio_stitching > 1 con ``death_emit_count`` alto = el detector se
        # fragmenta pero el fallback rescata (count probably OK). Mismo ratio
        # con death_emit_count bajo = fragmentación que NO se rescató (pérdida
        # real de cuentas). Reset diario.
        self._death_emit_count: int = 0

    # ----------------------------------------------------------------- API
    @property
    def lines(self) -> list[Line]:
        """Las líneas configuradas (copy read-only)."""
        return list(self._lines)

    @property
    def counting_zone(self) -> Optional[tuple[float, float, float, float]]:
        """Counting zone como ``(x_min, x_max, y_min, y_max)`` o ``None`` si unset.

        Es lo que el incumbent (FFC) llama TrackingZone — la región física del
        gate donde el counter mide entradas/salidas. En nuestro sistema se usa
        como gate semántico POST-tracker (no filtra detecciones pre-tracker)."""
        return self._counting_zone

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
        hora DESCENDENTE (la más reciente arriba), solo las horas con actividad.
        ``in`` = label ``ingress``, ``out`` = ``egress``. Tracking en memoria
        con reset diario; el borde NO persiste el histórico (eso vive en RDS,
        ``count_events``).

        Cap a las 10 horas más recientes para mantener el preview compacto
        y match al tamaño de la tabla de tráfico exterior (que también
        muestra recent ~10 rows). El histórico completo del día queda
        disponible en RDS / Grafana.
        """
        rows = []
        for hour in sorted(self._hourly, reverse=True):
            n_in = self._hourly[hour].get("ingress", 0)
            n_out = self._hourly[hour].get("egress", 0)
            # Saltear horas sin actividad — no ensuciar la tabla con filas 0/0.
            if n_in or n_out:
                rows.append({"hour": hour, "in": n_in, "out": n_out})
            if len(rows) >= 10:
                break
        return rows

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        # Camino normal: entrar a la counting zone → cruzar la línea (con detección real) →
        # salir de la counting zone → emite. Procesa también las MUERTES de tracks: si un
        # track desaparece de la dict habiendo CRUZADO (net != 0) y sin haber
        # SALIDO de la counting zone (inside aún True), emitimos el conteo igual en
        # ``_emit_on_death``. Cubre el caso que el detector pierde a la persona
        # después del cruce y el Kalman parka adentro de la counting zone sin alcanzar el
        # borde → sin death-emit ese conteo se perdía. El gate es selectivo:
        # un track que entró pero NUNCA cruzó (net=0) NO se cuenta en muerte —
        # así una persona que duda en la entrada y vuelve no genera un falso
        # positivo.
        events: list[CountEvent] = []
        alive_ids: set[int] = set(tracks.keys())

        # Resurrección: si un track que estaba en potential_deaths reapareció
        # (el tracker lo adoptó del ghost pool), cancelamos el death-emit. El
        # snapshot se descarta porque el track vivo es el portador autoritativo
        # del estado a partir de ahora.
        for tid in alive_ids & set(self._potential_deaths.keys()):
            del self._potential_deaths[tid]

        for track in tracks.values():
            ev = self._process_track(track)
            if ev is not None:
                events.append(ev)
            # Snapshot del estado relevante post-procesamiento. Guardamos los
            # agregados de altura/confidence ahora porque cuando detectemos la
            # muerte (en el próximo frame) el Track ya no existe.
            self._snapshot_track(track)

        # Detectar tracks que desaparecieron desde el último frame y meterlos
        # en la cola de muertes pendientes con grace window.
        new_dead = set(self._tracking_state.keys()) - alive_ids
        for tid in new_dead:
            snap = self._tracking_state.pop(tid)
            self._potential_deaths[tid] = {"snap": snap, "age": 0}

        # Aging + emit en expiry: solo después de ``death_emit_grace_frames``
        # sin que el tracker resucite el track, fire el death-emit como
        # fallback.
        expired_dead: list[int] = []
        for tid, info in self._potential_deaths.items():
            info["age"] += 1
            if info["age"] > self.death_emit_grace_frames:
                expired_dead.append(tid)
        for tid in expired_dead:
            info = self._potential_deaths.pop(tid)
            ev = self._emit_on_death(info["snap"])
            if ev is not None:
                events.append(ev)

        return events

    def _snapshot_track(self, track: Track) -> None:
        meta = track.meta.get(self.META_KEY) or {}
        cx, cy = (
            self._tracking_point(track) if track.positions else (0.0, 0.0)
        )
        # Visit-range para los guards anti-falso-positivo del death-emit.
        vx_min = float(meta.get("visit_x_min", cx))
        vx_max = float(meta.get("visit_x_max", cx))
        vy_min = float(meta.get("visit_y_min", cy))
        vy_max = float(meta.get("visit_y_max", cy))
        self._tracking_state[track.track_id] = {
            "track_id": track.track_id,
            "inside": bool(meta.get("inside", False)),
            "net": list(meta.get("crossing_net") or []),
            "last_crossing_pos": meta.get("last_crossing_pos"),
            "last_pos": (cx, cy),
            "had_outside_pos": bool(meta.get("had_outside_pos", False)),
            "visit_x_range": vx_max - vx_min,
            "visit_y_range": vy_max - vy_min,
            "real_inside_frames": int(meta.get("real_inside_frames", 0)),
            "height_m": _aggregate_height_m_from_track(track),
            "head_depth_m": _aggregate_head_depth_m_from_track(track),
            "confidence": _aggregate_confidence_from_track(track),
        }

    def _emit_on_death(self, snap: dict) -> Optional[CountEvent]:
        """Emite un CountEvent para un track que murió SIN haber salido de la
        counting zone, si tenía un cruce de línea registrado. Caso típico: el
        detector pierde a la persona después de cruzar y el Kalman parka adentro
        de la counting zone antes de alcanzar el borde — sin esto, el conteo se
        perdía."""
        if not snap.get("inside"):
            return None  # ya había salido (o nunca entró)
        net = snap.get("net") or []
        if not any(n != 0 for n in net):
            return None  # entró pero NUNCA cruzó — no contar

        # Guard 1: had_outside_pos. El track debe haber sido visto FUERA del
        # counting zone en algún momento de su vida. Filtra sitters + clutter + re-id
        # failures que spawnean adentro de la counting zone: si la cabeza de un sentado
        # jitterea atravesando la línea, el cross se registra pero el track
        # nunca tuvo evidencia de "entrar desde afuera" — no es una pasada
        # real, no emitir.
        if not snap.get("had_outside_pos"):
            logger.debug(  # TRACKDBG [REVERT]
                "TRACKDBG death_emit_skipped tid=%d reason=no_outside_history net=%s",
                snap.get("track_id"), net,
            )
            return None

        # Guard 2: visit_range mínimo. El track debe haber recorrido al menos
        # MIN_VISIT_RANGE_FOR_DEATH_EMIT px durante la visita (max de x_range
        # y y_range). Filtra el edge case donde un track tuvo outside_pos
        # pero después solo jittereó adentro de la counting zone sin movimiento real.
        x_range = float(snap.get("visit_x_range", 0.0))
        y_range = float(snap.get("visit_y_range", 0.0))
        if max(x_range, y_range) < self.min_visit_range_for_death_emit:
            logger.debug(  # TRACKDBG [REVERT]
                "TRACKDBG death_emit_skipped tid=%d reason=small_visit_range "
                "x_range=%.0f y_range=%.0f net=%s",
                snap.get("track_id"), x_range, y_range, net,
            )
            return None

        # Mismo veredicto que la rama de exit: signo del neto por línea, la
        # última con verdict no-nulo gana.
        label: Optional[str] = None
        for i, line in enumerate(self._lines):
            if i >= len(net):
                continue
            if net[i] > 0:
                verdict = line.crossing_label(-1, 1)
            elif net[i] < 0:
                verdict = line.crossing_label(1, -1)
            else:
                verdict = None
            if verdict is not None:
                label = verdict
        if label is None:
            return None

        # Guard 4: evidencia mínima — single-frame entries que mueren in-zone
        # también se rechazan (mismo razonamiento que en exit).
        if self.min_real_inside_frames > 0:
            real_frames = int(snap.get("real_inside_frames", 0))
            if real_frames < self.min_real_inside_frames:
                logger.debug(  # TRACKDBG [REVERT]
                    "TRACKDBG death_emit_skipped tid=%d reason=thin_evidence "
                    "real_inside_frames=%d threshold=%d net=%s",
                    snap.get("track_id"), real_frames,
                    self.min_real_inside_frames, net,
                )
                return None

        # Guard 3: altura humana. Mismo cap que en la rama de exit observado —
        # rechaza el death-emit si la mediana de altura del track es menor
        # que ``min_count_height_m`` y la medición existe (no None). FPs
        # no-humanos del detector (perro caminando, carrito) cuya muerte
        # llega a este punto se filtran acá.
        snap_height_m = snap.get("height_m")
        if (
            self.min_count_height_m > 0
            and snap_height_m is not None
            and snap_height_m < self.min_count_height_m
        ):
            logger.debug(  # TRACKDBG [REVERT]
                "TRACKDBG death_emit_skipped tid=%d reason=short_height "
                "height_m=%.2f threshold=%.2f net=%s",
                snap.get("track_id"), snap_height_m,
                self.min_count_height_m, net,
            )
            return None

        cross_pos = snap.get("last_crossing_pos") or snap.get("last_pos") or (0.0, 0.0)
        cross_x = float(cross_pos[0])
        cross_y = float(cross_pos[1])
        now = time.time()
        self._totals[label] = self._totals.get(label, 0) + 1
        _hb = self._hourly.setdefault(time.localtime(now).tm_hour, {})
        _hb[label] = _hb.get(label, 0) + 1
        self._death_emit_count += 1
        logger.info(
            "count_event_on_death tid=%d label=%s net=%s",
            snap["track_id"], label, net,
        )
        # Aplicar el mismo confidence gate que la rama de exit (altura no
        # confiable si el bbox fue marginal a lo largo del track).
        conf = snap.get("confidence")
        if conf is not None and conf < self.height_confidence_gate:
            height_m = None
            head_depth_m = None
        else:
            height_m = snap.get("height_m")
            head_depth_m = snap.get("head_depth_m")
        return CountEvent(
            track_id=snap["track_id"],
            direction=label,
            timestamp=now,
            position_x=cross_x,
            position_y=cross_y,
            height_m=height_m,
            head_depth_m=head_depth_m,
            confidence=conf,
        )

    def reset_daily(self) -> None:
        for k in self._totals:
            self._totals[k] = 0
        self._hourly.clear()
        self._tracking_state.clear()
        self._potential_deaths.clear()
        self._seen_track_ids.clear()
        self._death_emit_count = 0

    @property
    def stitching_ratio(self) -> float:
        """Canary de fragmentación del tracker: track_ids únicos vistos cruzar
        la counting zone hoy / conteos emitidos. Ideal ≈ 1.0 (cada persona = 1 ID = 1
        evento). >1.5 sugiere fragmentación (el detector pierde personas y el
        tracker spawnea IDs nuevos en vez de adoptar). 0 si no hubo conteo
        emitido todavía. Análogo a ``wifi_ble_stitching_ratio``.

        Combinar con ``death_emit_count`` para diagnosticar el sub-modo:
        ratio>1.3 + death_emit alto = fragmentación rescatada por el fallback
        (cuentas OK), ratio>1.3 + death_emit bajo = fragmentación sin rescue
        (cuentas perdidas, mirar recall del detector)."""
        total = sum(self._totals.values())
        if total <= 0:
            return 0.0
        return len(self._seen_track_ids) / float(total)

    @property
    def death_emit_count(self) -> int:
        """Cantidad de death-emits disparados hoy. Telemetría para
        diferenciar "fragmenta-y-rescata" de "fragmenta-y-pierde". Reset
        diario."""
        return self._death_emit_count

    # ------------------------------------------------------------- internal
    def _decisive_kalman_cross(
        self,
        track: Track,
        meta: dict,
        cx: float,
        cy: float,
        line: Line,
        new_side: int,
    ) -> bool:
        """¿Es decisivo un cruce extrapolado por Kalman (track PENDING) para
        contarlo como real?

        Rescata el caso "track caminando, detector la pierde en la zona crítica
        de la línea, Kalman extrapola atravesándola". Filtra el "drift
        estacionario" (track sin velocidad real que se desplaza poco por la
        inercia residual del Kalman).

        Decisivo si:
        1. ``disappeared <= MAX_KALMAN_CROSS_FRAMES`` (track con detección
           reciente — no es un track viejo que estuvo perdido mucho tiempo).
        2. Desplazamiento desde la última posición REAL es ``>=
           MIN_KALMAN_CROSS_DISPLACEMENT_PX`` en la dirección del cruce.
        """
        if int(getattr(track, "disappeared", 0) or 0) > self.MAX_KALMAN_CROSS_FRAMES:
            return False
        last_real = meta.get("last_track_pos")
        if last_real is None:
            return False
        dx = cx - float(last_real[0])
        dy = cy - float(last_real[1])
        # Línea horizontal (from.y == to.y) → componente vertical del
        # desplazamiento; vertical → componente horizontal. ``new_side``
        # multiplica el delta así un cruce hacia +1 requiere delta>0 (debajo
        # de horizontal, derecha de vertical), hacia -1 requiere delta<0.
        is_horizontal = abs(line.from_xy[1] - line.to_xy[1]) < 0.5
        if is_horizontal:
            decisive = dy * new_side
        else:
            decisive = dx * new_side
        return decisive >= self.MIN_KALMAN_CROSS_DISPLACEMENT_PX

    def _inside_counting_zone(self, cx: float, cy: float) -> bool:
        if self._counting_zone is None:
            return True
        x_min, x_max, y_min, y_max = self._counting_zone
        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def _tracking_point(self, track: Track) -> tuple[float, float]:
        """Pixel usado para la counting zone + cruce de línea: el centroide del bbox.

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
        is_inside = self._inside_counting_zone(cx, cy)

        # ¿Esta posición viene de una detección REAL este frame, o es una
        # predicción del Kalman (track PENDING sin match)? Solo registramos
        # cruces de línea con detección real: un track perdido que el Kalman
        # extrapola puede "cruzar" la línea por inercia y doble-contar. La
        # extrapolación SÍ se usa para las transiciones entry/exit de la counting zone —
        # así un track que cruzó (con detección real) y después se perdió
        # igual SALE de la counting zone y emite el cruce ya registrado (no se pierde el
        # conteo). El doble-conteo del drift se evita acá, no congelando el
        # track en el tracker.
        is_real = int(getattr(track, "disappeared", 0) or 0) == 0

        # Memoria de la última posición outside-counting-zone. La entry-fresca
        # abajo la usa para snapshotear sides[] desde la trayectoria de
        # approach (lado inequívoco de donde viene el track) en lugar de
        # la primera detección inside — que con counting zone chica + detector miss
        # rate puede caer ya del otro lado de la línea y dejar el cache
        # corrupto, perdiendo el cruce.
        #
        # Solo se actualiza con DETECCIONES REALES (is_real=True). Los
        # frames de pura extrapolación Kalman pueden alucinar posiciones
        # outside counting zone muy lejanas (ej. (888,109) cuando el sitter está a
        # (470,330)) durante el exit del track — si esa posición
        # alucinada se guarda y el track muere + es ghost-adopted, el
        # nuevo ciclo del track hereda un last_outside_pos espurio que
        # infunde had_outside_pos=True falso, bypaseando el guard de
        # capa 2 del rescue cascade. Filtrar por is_real=True corta ese
        # ciclo: las "evidencias de approach" se cuentan solo con
        # observaciones, no con alucinaciones del modelo de movimiento.
        # Coherente con cómo visit_x/y_range también se actualiza solo
        # con detecciones reales (capa 3 del rescue).
        if not is_inside and is_real:
            meta["last_outside_pos"] = (cx, cy)

        # Cache per-line del lado previo (un slot por línea). Se crea
        # lazy; se repara si cambió el count de líneas configurado
        # entre updates.
        sides = meta.get("line_sides")
        if not isinstance(sides, list) or len(sides) != len(self._lines):
            sides = [0] * len(self._lines)
            meta["line_sides"] = sides

        # Balance NETO de cruces por línea durante la visita actual al
        # counting zone (máquina de estados de la zona). Cada cruce in-segment
        # hacia el lado +1 suma 1, hacia el lado -1 resta 1 (solo cuando
        # la dirección tiene label — un gate one-way es "sticky": el
        # cruce de vuelta no tiene label y no cancela). Al salir de la counting zone
        # se emite según el SIGNO del neto: >0 = quedó del lado +1 (cruzó
        # neto en esa dirección), <0 = lado -1, 0 = fue y vino, NO cuenta.
        # Esto reemplaza el viejo "gana el último cruce" que contaba mal
        # el caso "entró, cruzó, volvió a cruzar y salió" (debía dar 0).
        net = meta.get("crossing_net")
        if not isinstance(net, list) or len(net) != len(self._lines):
            net = [0] * len(self._lines)
            meta["crossing_net"] = net

        if is_inside and not was_inside:
            # Guard anti-entry-Kalman: si el primer frame inside es pura
            # extrapolación Kalman (is_real=False), NO disparar la
            # entry-fresca. El track no fue REALMENTE observado adentro
            # — el Kalman lo proyectó a la counting zone desde una FP del detector
            # afuera, y la entry-fresca le snapshotearía sides[] +
            # configuraría was_inside=True habilitando el zigzag clásico
            # sobre la línea + EXIT por Kalman emitiendo count espurio.
            # Bug observado en piloto 2026-05-24 09:47-09:54 (tid=35):
            # entry is_real=False -> 9 crosses zigzag en 6 min -> exit
            # Kalman -> COUNT IN falso. Si la persona realmente está
            # adentro, va a venir un frame is_real=True en 1-3 ticks y
            # la entry-fresca se dispara ahí con misma información
            # geométrica (last_outside_pos sigue válido).
            if not is_real:
                logger.debug(  # TRACKDBG [REVERT]
                    "TRACKDBG entry_kalman_skipped tid=%d pos=(%.0f,%.0f)",
                    track.track_id, cx, cy,
                )
                return None
            # Entry fresca: resetear estado del ciclo y snapshotear
            # sides desde la última posición outside-counting-zone conocida (el
            # lado del cual el track se aproxima — inequívoco). Esto
            # protege contra el case patológico donde, con counting zone chica
            # y detector miss rate, la primera detección inside cae
            # ya del otro lado de la línea y dejaría sides[] cacheado
            # del lado equivocado (resultando en cruces no detectados).
            # Fallback al (cx, cy) actual si el track nació inside
            # sin historia outside.
            meta["inside"] = True
            meta["last_crossing_pos"] = None
            meta["last_track_pos"] = (cx, cy)
            # Tracking de rango de movimiento durante la visita (para los
            # guards del death-emit). Se inicializa en la posición de entrada
            # y se actualiza con cada detección REAL dentro de la counting zone.
            meta["visit_x_min"] = cx
            meta["visit_x_max"] = cx
            meta["visit_y_min"] = cy
            meta["visit_y_max"] = cy
            # Contador de frames con detección real dentro del counting zone.
            # La entry-fresca por definición es is_real=True (el guard
            # entry_kalman_skipped la rechazaría si fuera Kalman) → comienza en 1.
            # Filtra single-frame entries en el borde (campera flickeando).
            meta["real_inside_frames"] = 1
            # Resetear el balance neto de cruces para la nueva visita.
            for i in range(len(net)):
                net[i] = 0
            outside_pos = meta.get("last_outside_pos")
            # Flag: ¿este track tuvo posición OUTSIDE counting zone en algún momento?
            # Es evidencia de que entró desde afuera (legitimate walk-through)
            # vs spawneó adentro (sitter/clutter/re-id failure).
            meta["had_outside_pos"] = outside_pos is not None
            snap_x, snap_y = outside_pos if outside_pos is not None else (cx, cy)
            for i, line in enumerate(self._lines):
                sides[i] = line.side_of(snap_x, snap_y)
            # Registrar el track_id para la métrica de stitching. Set: cada
            # ID que entró a la counting zone hoy se registra una sola vez (idempotent).
            self._seen_track_ids.add(track.track_id)
            logger.debug(  # TRACKDBG [REVERT]
                "TRACKDBG entry tid=%d pos=(%.0f,%.0f) snap=(%.0f,%.0f) sides=%s is_real=%s",
                track.track_id, cx, cy, snap_x, snap_y, sides, is_real,
            )
            return None

        if is_inside and was_inside:
            # Frame de pura predicción (track PENDING, sin detección): NO
            # evaluar cruces — registrar uno acá sería especulativo y
            # doble-contaría el drift del Kalman. Preservamos sides/net/
            # last_track_pos como están (anclados a la última detección real).
            if not is_real:
                return None

            # Actualizar el rango de movimiento de la visita (solo con
            # detecciones REALES). Lo consulta el guard del death-emit
            # para distinguir walk-throughs de sitters con jitter.
            if cx < meta["visit_x_min"]:
                meta["visit_x_min"] = cx
            elif cx > meta["visit_x_max"]:
                meta["visit_x_max"] = cx
            if cy < meta["visit_y_min"]:
                meta["visit_y_min"] = cy
            elif cy > meta["visit_y_max"]:
                meta["visit_y_max"] = cy
            # Contador de frames inside con detección real (ver entry-fresca).
            meta["real_inside_frames"] = int(meta.get("real_inside_frames", 0)) + 1

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
                        # el balance neto como están, no evaluar cruces.
                        return None

            # Detectar transición de lado en cada línea y acumular el
            # balance neto de la visita. Un cruce in-segment hacia +1
            # suma 1 al neto, hacia -1 resta 1 (solo si la dirección
            # tiene label). El track puede cruzar múltiples líneas en
            # una visita; cada una lleva su propio neto.
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
                        net[i] += 1 if new_side == 1 else -1
                        # Anclar la posición del cruce dentro de la counting zone. El
                        # emit en exit la usa para el position_x/y del
                        # CountEvent (no la posición de salida, que está
                        # fuera de la counting zone por definición).
                        meta["last_crossing_pos"] = (cx, cy)
                        logger.debug(  # TRACKDBG [REVERT]
                            "TRACKDBG cross tid=%d line=%d new_side=%+d net=%+d pos=(%.0f,%.0f) label=%s",
                            track.track_id, i, new_side, net[i], cx, cy, label,
                        )
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
            # aproximación y suele caer dentro de la counting zone.
            # Registrar el cruce de la transición de exit si:
            #   - es una detección REAL, o
            #   - es una extrapolación Kalman DECISIVA (track caminando que el
            #     detector perdió en la zona crítica de la línea; ver
            #     _decisive_kalman_cross). El drift estacionario (residual
            #     velocity con decay) no pasa los guardrails y se descarta —
            #     anti doble-conteo del parado-cruzar.
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
                    if not is_real and not self._decisive_kalman_cross(
                        track, meta, cx, cy, line, new_side
                    ):
                        continue
                    cross_label = line.crossing_label(prev_side, new_side)
                    if cross_label is not None:
                        net[i] += 1 if new_side == 1 else -1
                        if last_inside is not None:
                            meta["last_crossing_pos"] = (
                                (float(last_inside[0]) + cx) / 2.0,
                                (float(last_inside[1]) + cy) / 2.0,
                            )
                        else:
                            meta["last_crossing_pos"] = (cx, cy)

            # Veredicto de la visita: según el SIGNO del balance neto por
            # línea. neto > 0 = cruzó neto hacia el lado +1; neto < 0 =
            # hacia -1; neto == 0 = fue y vino, no cuenta. Si hay varias
            # líneas, la última con veredicto no-nulo gana (defensiva; en
            # deploys bien configurados las líneas cubren zonas disjuntas).
            label = None
            for i, line in enumerate(self._lines):
                if net[i] > 0:
                    verdict = line.crossing_label(-1, 1)
                elif net[i] < 0:
                    verdict = line.crossing_label(1, -1)
                else:
                    verdict = None
                if verdict is not None:
                    label = verdict
            crossing_pos = meta.get("last_crossing_pos") or (cx, cy)
            logger.debug(  # TRACKDBG [REVERT]
                "TRACKDBG exit tid=%d is_real=%s net=%s verdict=%s exit_pos=(%.0f,%.0f) real_inside_frames=%d",
                track.track_id, is_real, net, label, cx, cy,
                int(meta.get("real_inside_frames", 0)),
            )
            # Snapshot del net antes del reset para logging del skip.
            net_snapshot = list(net)
            # Resetear estado así el mismo track puede contar otro
            # ciclo completo más tarde. El invariante antiglitch
            # vale: contar de nuevo requiere re-entry desde afuera +
            # un cruce de línea con label + exit.
            meta["inside"] = False
            meta["last_crossing_pos"] = None
            for i in range(len(sides)):
                sides[i] = 0
            for i in range(len(net)):
                net[i] = 0
            # Refrescar ``last_outside_pos`` a la posición de salida
            # cuando el track ya tiene evidencia legítima de approach
            # (``had_outside_pos=True``) O cuando el exit es real. Sin
            # esto, un exit Kalman tras un cruce real (caso piloto
            # 2026-05-24 14:57:55 tid=316) deja el meta con el
            # ``last_outside_pos`` del approach del ciclo anterior
            # (ej. arriba). Si el track re-entra desde abajo, la entry-
            # fresca snap-ea sides[] desde el outside_pos viejo (lado
            # incorrecto) → cualquier exit posterior dispara un cross
            # virtual y un emit espurio (doble ingress en 161 ms).
            #
            # El guard ``had_outside_pos OR is_real`` preserva el
            # invariante anterior (sitter alucinado por Kalman lejos
            # NO contamina): un track born-inside (had_outside_pos=
            # False) cuyo Kalman alucina hacia outside lateral NO
            # actualiza el meta, así el guard de capa 2 del rescue
            # cascade sigue agarrando ese FP. Ver test
            # ``test_last_outside_pos_only_updated_with_real_detections``.
            if is_real or bool(meta.get("had_outside_pos", False)):
                meta["last_outside_pos"] = (cx, cy)
            # Guard de exit-por-Kalman: tracks que nacieron adentro del
            # counting zone (had_outside_pos=False) no deben emitir counts vía
            # extrapolación Kalman. Filtra el caso del sitter parado
            # cerca de la línea: la persona se mueve un poco, cruza la
            # línea (cross real, net != 0), el detector la pierde, y el
            # Kalman extrapola la velocidad residual hacia afuera del
            # counting zone — sin este guard, el cruce intra-visita generaría un
            # count espurio aunque la persona nunca salió de la tienda.
            # Para exits con detección real (is_real=True) el guard NO
            # aplica: si el track salió observado, vale el count
            # aunque haya nacido inside (caso del primer-frame-perdido
            # por el detector en el borde de la counting zone). Coherente con el
            # mismo guard en _emit_on_death (capa 3 del rescue cascade).
            if (
                label
                and not is_real
                and not bool(meta.get("had_outside_pos", False))
            ):
                logger.debug(  # TRACKDBG [REVERT]
                    "TRACKDBG exit_kalman_skipped tid=%d "
                    "reason=no_outside_history net=%s",
                    track.track_id, net_snapshot,
                )
                label = None
            # Guard de evidencia mínima — defensa contra single-frame entries
            # en el borde del counting zone (detector flickea 1 vez sobre
            # clutter, Kalman extrapola el resto de la "visita"). Caso real
            # piloto 2026-05-24 15:23 tid=67: 1 frame al borde + 247 px de
            # extrapolación Kalman → ingress espurio.
            if label and self.min_real_inside_frames > 0:
                real_frames = int(meta.get("real_inside_frames", 0))
                if real_frames < self.min_real_inside_frames:
                    logger.debug(  # TRACKDBG [REVERT]
                        "TRACKDBG exit_thin_evidence_skipped tid=%d "
                        "real_inside_frames=%d threshold=%d net=%s",
                        track.track_id, real_frames,
                        self.min_real_inside_frames, net_snapshot,
                    )
                    label = None
            # Guard de altura humana — defensa contra FPs no-humanos del
            # detector (perros, carritos, objetos sobre sillas). Si la
            # MEDIANA de head_height_mm del track durante la visita es
            # menor que ``min_count_height_m`` y la medición existe (no
            # None), el counter rechaza el emit. None = sin medición SGBM
            # confiable -> pasa (recall sobre precisión). Caso piloto
            # 2026-05-24: perro caminando, campera en silla.
            if label and self.min_count_height_m > 0:
                track_height_m = _aggregate_height_m_from_track(track)
                if (
                    track_height_m is not None
                    and track_height_m < self.min_count_height_m
                ):
                    logger.debug(  # TRACKDBG [REVERT]
                        "TRACKDBG exit_short_height_skipped tid=%d "
                        "height_m=%.2f threshold=%.2f net=%s",
                        track.track_id, track_height_m,
                        self.min_count_height_m, net_snapshot,
                    )
                    label = None
            if label:
                now = time.time()
                cross_x = float(crossing_pos[0])
                cross_y = float(crossing_pos[1])
                self._totals[label] = self._totals.get(label, 0) + 1
                _hb = self._hourly.setdefault(time.localtime(now).tm_hour, {})
                _hb[label] = _hb.get(label, 0) + 1
                logger.debug(
                    "count_event",
                    extra={"track_id": track.track_id, "label": label},
                )
                # Demographics gate: la altura se calcula desde SGBM dentro
                # del bbox, y SGBM degrada con motion blur (matching estéreo
                # falla en bordes desenfocados) y con bbox poco precisos
                # (centroide off-cabeza). Cuando la mediana de confidence
                # del track cae bajo height_confidence_gate, el bbox fue
                # marginal a lo largo de la trayectoria y la altura
                # derivada no es confiable — el conteo en sí (dirección,
                # cruce) se mantiene porque solo depende del centroide.
                # Limpiar height_m/head_depth_m a None así los dashboards
                # no surfacean valores demográficos espurios (la
                # categorización adulto/niño se computa server-side desde
                # height_m, así NULL → 'unknown' en la vista).
                conf_median = _aggregate_confidence_from_track(track)
                if (
                    conf_median is not None
                    and conf_median < self.height_confidence_gate
                ):
                    height_m = None
                    head_depth_m = None
                else:
                    height_m = _aggregate_height_m_from_track(track)
                    head_depth_m = _aggregate_head_depth_m_from_track(track)
                return CountEvent(
                    track_id=track.track_id,
                    direction=label,
                    timestamp=now,
                    position_y=cross_y,
                    position_x=cross_x,
                    height_m=height_m,
                    head_depth_m=head_depth_m,
                    confidence=conf_median,
                )
            # Salió sin veredicto: o nunca cruzó la línea, o cruzó y
            # volvió a cruzar (balance neto 0 — "fue y vino" dentro del
            # counting zone). En ambos casos NO se cuenta.
            logger.debug(
                "exit_without_net_crossing",
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
          counting_zone:                    # opcional
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
                f"counter.lines[{idx}]: 'from' and 'to' required as [x, y] pairs ({e})."
            ) from e
        if len(from_xy) != 2 or len(to_xy) != 2:
            raise ValueError(
                f"counter.lines[{idx}]: 'from' and 'to' must be [x, y] pairs."
            )
        labels = dict(raw.get("labels") or {})
        lines.append(Line(from_xy=from_xy, to_xy=to_xy, labels=labels))

    # ``min_visit_range_for_death_emit`` per-site (tuneo del rescue cascade
    # capa 3). None → el Counter usa su DEFAULT (80px). Bajarlo a 50 en sites
    # con detector flakey cuando la telemetría muestra ``death_emit_count = 0``
    # con ``track_stitching_ratio > 1.3`` sostenido (rescate-de-crossers-con-
    # poca-observación rechazado por el guard). Ver docs/tracker_tuning.md.
    min_visit_range_cfg = counter_cfg.get("min_visit_range_for_death_emit")
    min_visit_range_kwarg = (
        float(min_visit_range_cfg) if min_visit_range_cfg is not None else None
    )

    # ``min_count_height_m`` per-site (filtro anti-FP no-humanos por altura).
    # None / ausente → DEFAULT 0.0 (filtro OFF, back-compat). Activar en sites
    # con perros, carritos, objetos que el detector toma como personas.
    # Ver docs/tracker_tuning.md patrón 5.
    min_count_height_cfg = counter_cfg.get("min_count_height_m")
    min_count_height_kwarg = (
        float(min_count_height_cfg) if min_count_height_cfg is not None else None
    )

    # ``min_real_inside_frames`` per-site (filtro anti-flickering del detector).
    # None / ausente → DEFAULT 0 (filtro OFF). Activar en sites con FPs
    # single-frame al borde del counting zone. Piloto: 2.
    min_real_frames_cfg = counter_cfg.get("min_real_inside_frames")
    min_real_frames_kwarg = (
        int(min_real_frames_cfg) if min_real_frames_cfg is not None else None
    )

    # ``height_confidence_gate`` per-site (umbral de median(conf) para
    # reportar demografía en el CountEvent). None / ausente → DEFAULT 0.5.
    # No afecta el conteo; solo la calidad del reporte de altura.
    height_conf_gate_cfg = counter_cfg.get("height_confidence_gate")
    height_conf_gate_kwarg = (
        float(height_conf_gate_cfg) if height_conf_gate_cfg is not None else None
    )

    return Counter(
        lines=lines,
        counting_zone=counter_cfg.get("counting_zone"),
        min_crossing_movement_px=float(
            counter_cfg.get("min_crossing_movement_px", 0.0) or 0.0
        ),
        min_visit_range_for_death_emit=min_visit_range_kwarg,
        min_count_height_m=min_count_height_kwarg,
        height_confidence_gate=height_conf_gate_kwarg,
        min_real_inside_frames=min_real_frames_kwarg,
    )

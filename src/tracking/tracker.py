"""Tracker 3D para trayectorias de personas.

Implementa asociación Hungarian-based con una state machine de corto plazo:

    CANDIDATE  -> CONFIRMED  (luego de N hits consecutivos)
    CONFIRMED  -> PENDING    (ante un frame perdido; predicho con Kalman)
    PENDING    -> CONFIRMED  (re-match dentro del re-id gate)
    PENDING    -> LOST       (luego de pending_max_frames misses consecutivos)

Keep-alive dentro del ROI: si se configura ``keepalive_roi`` (lo setea main
desde ``counter.roi``), un track PENDING cuya última posición predicha cae
dentro del ROI NO transiciona a LOST — se mantiene vivo indefinidamente
hasta que vuelve a matchear (PENDING->CONFIRMED) o su predicción sale del
ROI (ahí aplica el timeout normal). Cubre "cruzó y se quedó adentro mirando
algo" sin perder el cruce por muerte del track. Ver ``_record_miss``.

El tracker expone todos los tracks en cualquier estado; la capa counter
filtra a CONFIRMED/PENDING cuando toma decisiones de conteo. Los track IDs
se preservan a través de recoveries PENDING -> CONFIRMED, así el estado
per-track del counter (lado de entrada, línea cruzada) sobrevive oclusiones
cortas.

Las posiciones son [cx, cy, z] con cx/cy en pixels y z en mm. El costo de
matching es la distancia 2D en pixels; la profundidad se usa como gate
secundario (|dz| > max_depth_delta rechaza el match) porque pixels y
milímetros no son directamente comparables.

Modelo de movimiento
--------------------
Cada track posee un pequeño filtro Kalman de velocidad constante (ver
``src.tracking.kalman.TrackKalman``) con estado ``[cx, cy, vx, vy]``.
Lifecycle por frame:

1. Al inicio de ``update()``, el filtro de cada track avanza un paso vía
   ``predict()``. Esto setea ``Track.predicted_position`` (y la referencia
   de matching para tracks PENDING) a la expectativa del modelo para este
   frame, y crece la covarianza de posición — comportamiento natural para
   un track que estuvo varios frames sin match.
2. La asociación Hungarian corre contra ``predicted_position`` para tracks
   PENDING y ``last_position`` para los demás (la misma ref que usaba el
   tracker pre-Kalman; preserva la semántica del gate ajustado para tracks
   adyacentes).
3. Los tracks matcheados llaman ``kalman.update(measurement)`` para
   incorporar la nueva detección. Los no matcheados mantienen el estado
   post-predict — a lo largo de varios misses esto extrapola más lejos
   mientras ensancha la incertidumbre.

Nota: la profundidad NO está en el estado de Kalman. El gate de profundidad
corre sobre detección raw vs. última profundidad observada; las fluctuaciones
de profundidad frame-a-frame están dominadas por ruido SGBM antes que por
movimiento suave, así que un filtro 4D sobre (x, y, vx, vy) es el scope correcto.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman import TrackKalman

logger = logging.getLogger(__name__)

# Estados del track
CANDIDATE = "candidate"
CONFIRMED = "confirmed"
PENDING = "pending"
LOST = "lost"

# Defaults del Kalman — overrideables vía constructor de EuclideanTracker
# (que a su vez se alimenta de ``tracking.state_machine.kalman`` en el config
# mergeado).
DEFAULT_PROCESS_NOISE = 1.0
DEFAULT_MEASUREMENT_NOISE = 5.0
DEFAULT_INITIAL_VELOCITY_UNCERTAINTY = 100.0

# Cap del historial ``positions`` de un track. Normalmente los tracks son
# cortos (segundos) y nunca lo alcanzan, pero con keep-alive dentro del ROI
# un track puede vivir indefinidamente (persona parada mucho rato) — sin cap
# su lista de posiciones crecería sin límite a lo largo de un turno de 12h.
# 512 a ~20fps son ~25s de historia, de sobra para diagnósticos y para el
# chequeo de movimiento del stationary filter (que mira el rango reciente).
# El counter solo lee ``positions[-1]``, así que recortar el frente es inocuo.
_MAX_POSITIONS_HISTORY = 512


def _trim_positions(track: "Track") -> None:
    """Recorta el frente del historial de posiciones si excede el cap."""
    excess = len(track.positions) - _MAX_POSITIONS_HISTORY
    if excess > 0:
        del track.positions[:excess]


@dataclass
class Track:
    """Persona trackeada con historial de posiciones, estado y filtro de movimiento.

    ``positions`` mantiene los centroides observados raw por compatibilidad
    con consumers downstream (counter, viewer) que leen ``positions[-1]``.
    El filtro Kalman es la fuente de verdad para ``predicted_position``;
    la historia raw se preserva así los diagnósticos de trayectoria siguen
    funcionando.
    """

    track_id: int
    positions: list[np.ndarray] = field(default_factory=list)
    disappeared: int = 0
    state: str = CANDIDATE
    hits: int = 1
    # Motion filter — lo setea el tracker al registrar. ``Optional`` solo
    # porque el orden de init del dataclass requiere de otro modo un
    # default incómodo.
    kalman: Optional[TrackKalman] = None
    # Bookkeeping de la capa counter (lo setea el counter, no el tracker).
    # Guardado en el track así el estado sobrevive la re-identificación PENDING.
    meta: dict[str, Any] = field(default_factory=dict)
    # Última posición OBSERVADA (con detección real, no Kalman push). Se
    # actualiza solo en _record_hit. Usado por ghost-pool/clutter/diagnostics
    # que necesitan "dónde se vió por última vez al sujeto" — distinto de
    # ``positions[-1]`` que durante PENDING es la PREDICCIÓN del Kalman.
    last_observed_position: Optional[np.ndarray] = None

    @property
    def last_position(self) -> np.ndarray:
        """Centroide observado más reciente [cx, cy, z]. Compat para counter."""
        return self.positions[-1]

    @property
    def predicted_position(self) -> np.ndarray:
        """Posición predicha por Kalman para el PRÓXIMO frame como [cx, cy, z].

        No destructiva: usa ``kalman.peek_next()`` así el estado del
        filtro queda intacto. El caller obtiene la expectativa del
        modelo para un paso más allá del estado actual (es decir, dónde
        es más probable observar este track en el próximo
        ``tracker.update()``).

        El filtro Kalman solo modela movimiento 2D en pixels; el z
        devuelto carry-forwardea la profundidad observada más reciente
        (depth no tiene modelo de movimiento suave — ver docstring de módulo).
        """
        if self.kalman is None:
            # Fallback — solo pasa si un Track se construye sin filtro,
            # lo cual el propio tracker nunca hace. Defensive así
            # tests viejos / constructores externos no crashean.
            if len(self.positions) < 2:
                return self.positions[-1].copy()
            velocity = self.positions[-1] - self.positions[-2]
            return self.positions[-1] + velocity
        xy = self.kalman.peek_next()
        z = self.positions[-1][2] if len(self.positions[-1]) > 2 else 0.0
        return np.array([xy[0], xy[1], z], dtype=float)


@dataclass
class _GhostInfo:
    """Snapshot de un track recién muerto para la adoption window. Si una
    detección nueva aparece cerca dentro de la ventana, el track nuevo adopta
    este ID y meta — preserva la continuidad de identidad a través de un gap
    del detector que el re-id Kalman no pudo cerrar."""

    track_id: int
    last_observed_position: np.ndarray
    last_bbox: Optional[tuple[float, float, float, float]]
    meta_snapshot: dict
    age: int = 0


def _bbox_iou(
    a: Optional[tuple[float, float, float, float]],
    b: Optional[tuple[float, float, float, float]],
) -> float:
    """Intersection-over-Union de dos bboxes (x1, y1, x2, y2). 0 si alguno es
    None o si no hay intersección."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class EuclideanTracker:
    """Matching Hungarian 2D en pixels + depth gating, con state machine corta.

    Backward-compatible: el constructor sigue aceptando ``max_disappeared``
    y ``max_distance``; los kwargs opcionales nuevos habilitan la state
    machine y el comportamiento de re-identification. ``update(detections)``
    sigue devolviendo un ``dict[int, Track]`` con los tracks vivos
    actualmente (cualquier estado salvo LOST; los LOST se removieron).
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 50.0,
        max_depth_delta: float = 500.0,
        confirm_frames: int = 3,
        pending_max_frames: int = 5,
        reid_gate_px: float = 60.0,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        initial_velocity_uncertainty: float = DEFAULT_INITIAL_VELOCITY_UNCERTAINTY,
        pending_velocity_decay: float = 1.0,
        pending_grace_frames: int = 0,
        ambiguous_match_ratio: float = 1.0,
        max_track_id: int = 65536,
        keepalive_roi: tuple[float, float, float, float] | None = None,
        keepalive_max_frames: int = 600,
        adoption_window_frames: int = 30,
        adoption_iou_min: float = 0.3,
        adoption_max_dist_px: float = 100.0,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_depth_delta = max_depth_delta
        self.confirm_frames = max(1, confirm_frames)
        self.pending_max_frames = max(1, pending_max_frames)
        self.reid_gate_px = reid_gate_px
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_velocity_uncertainty = float(initial_velocity_uncertainty)
        # Factor de decay (0-1] aplicado a (vx, vy) del Kalman cada vez que
        # un track ENTRA al predict step ya en estado PENDING. Default 1.0
        # = sin decay (back-compat: el predict sigue extrapolando con la
        # última velocidad observada). Valores <1 simulan el prior
        # bayesiano "humano sin obs tiende a quedarse quieto" y previenen
        # el ghost-drift que dispara duplicados al re-identificar (track
        # nuevo en vez de re-asociar al original cuando reaparece >1.5s
        # después). Producción default 0.85 vía config.
        self.pending_velocity_decay = float(pending_velocity_decay)
        # Cantidad de frames iniciales de PENDING en los que SE PRESERVA
        # la velocidad completa antes de empezar a aplicar decay. Modela
        # oclusiones cortas (carrito que pasa, estructura overhead, motion
        # blur en ~3 frames) donde queremos que el Kalman siga
        # extrapolando a velocidad real así el predict cae cerca de la
        # detección cuando reaparece — no congelado en la posición de
        # entrada al PENDING. Default 0 = decay desde frame 1 (back-compat,
        # comportamiento previo). Producción default 3 vía config: a 12 fps
        # cubre ~250ms, suficiente para oclusión típica de carrito.
        self.pending_grace_frames = max(0, int(pending_grace_frames))
        # Ratio test post-Hungarian: rechaza matches donde el costo del
        # 2do-mejor alternativa (otra detección para el mismo track O
        # otro track para la misma detección) está dentro de este ratio
        # del costo del best. La intuición es Lowe (SIFT) aplicada a
        # tracking: si el match es ambiguo, mejor miss que ID swap. Con
        # ratio=0.8, se rechaza cualquier match donde best > 0.8 *
        # second_best — es decir, second_best está dentro del 25% del
        # best. Default 1.0 = off (back-compat). Producción 0.8 vía
        # config. Las detecciones de matches rechazados se CONSUMEN
        # (no spawnean tracks nuevos) para no inflar la cuenta con un
        # ID nuevo cuando el verdadero owner del bbox está ambiguo entre
        # dos tracks existentes.
        self.ambiguous_match_ratio = float(ambiguous_match_ratio)
        # Bound de los track IDs — los IDs ciclan módulo este valor.
        # Default 65536 (16-bit) es holgadísimo para un device: a 1000
        # tracks/día × 363 días sigue cabiendo en menos de 6 ciclos/año
        # y como tracks vivos simultáneos son ~10 max, las colisiones
        # son prácticamente imposibles. Sirve para (a) UX en el viewer
        # (IDs de 5 dígitos vs growth ilimitado de 7-8 dígitos en runs
        # largos), (b) safety si algún consumer downstream parsea el
        # ID como uint16/32. ``_register`` salta IDs en uso si chocara
        # en el rollover (defense in depth).
        self.max_track_id = max(1, int(max_track_id))
        # ROI de keep-alive: ``(x_min, x_max, y_min, y_max)`` o None. Un
        # track cuya última posición cae dentro de este rectángulo NO muere
        # por timeout (ni ``pending_max_frames`` ni ``max_disappeared``):
        # se mantiene PENDING indefinidamente mientras la predicción Kalman
        # quede dentro del ROI. Cubre el caso "persona cruzó la línea y se
        # quedó adentro mirando algo, el detector la pierde un rato, después
        # sale" — sin keep-alive el track moriría adentro y el cruce nunca
        # se contaría (no hay salida sintética). El static suppressor ya
        # exenta este mismo ROI a nivel detección, así que la persona quieta
        # sigue detectándose; el keep-alive es la red para los dropouts del
        # detector. Settable post-construcción (main lo setea desde
        # ``counter.roi``). None = sin keep-alive (back-compat).
        self.keepalive_roi = keepalive_roi
        # Cap del keep-alive en misses CONSECUTIVOS. "Eterno" haría que los
        # huérfanos (persona que se fue y el detector nunca vuelve a verla,
        # tras un re-id fallido que spawneó un track nuevo) se acumulen para
        # siempre — clutter en el preview + costo Hungarian + riesgo de que
        # un fantasma agarre por re-id a una persona nueva. Como
        # ``disappeared`` se resetea con CUALQUIER hit, una persona realmente
        # parada y detectada (aunque sea intermitente) nunca llega a este
        # cap; solo los huérfanos reales lo alcanzan y se garbage-collectean.
        # 600 frames ≈ 24s a 25fps — generoso para cualquier lingering real.
        self.keepalive_max_frames = max(1, int(keepalive_max_frames))
        # Ghost pool / ID adoption: cuando un track muere (LOST), su identidad
        # queda "disponible para adopción" durante ``adoption_window_frames``.
        # Si una detección nueva aparece dentro de ese tiempo con IoU >=
        # ``adoption_iou_min`` Y distancia <= ``adoption_max_dist_px`` respecto
        # del último bbox/pos del ghost, el track NUEVO adopta el ID y meta
        # del ghost → continuidad de identidad cross-gap-de-detección. Cubre
        # el caso "detector pierde N frames, persona reaparece, sin esto se
        # spawnea un track nuevo y se perdía la continuidad del cruce".
        # Defaults conservadores: IoU 0.3 (no booleano como FFC para evitar
        # ID-swap entre personas adyacentes), dist 100px (no teleports), 30
        # frames de ventana (~1.5s @ 20fps).
        self.adoption_window_frames = max(0, int(adoption_window_frames))
        self.adoption_iou_min = max(0.0, float(adoption_iou_min))
        self.adoption_max_dist_px = max(0.0, float(adoption_max_dist_px))
        self._next_id = 0
        self._tracks: OrderedDict[int, Track] = OrderedDict()
        # Pool de tracks recién muertos esperando posible adopción.
        # Key: track_id viejo. Value: _GhostInfo.
        self._ghosts: dict[int, "_GhostInfo"] = {}

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def count_by_state(self) -> dict[str, int]:
        """Devuelve un count de los tracks vivos agrupados por estado.

        Keys: ``candidate``, ``confirmed``, ``pending``, ``lost``. Los
        tracks LOST se purgan al final de ``update()`` así que el count
        normalmente es cero salvo que se llame mid-update.
        """
        counts = {CANDIDATE: 0, CONFIRMED: 0, PENDING: 0, LOST: 0}
        for track in self._tracks.values():
            counts[track.state] = counts.get(track.state, 0) + 1
        return counts

    # ------------------------------------------------------------------ API
    def update(
        self,
        detections: list[np.ndarray],
        detection_metas: Optional[list[dict]] = None,
        candidate_positions: Optional[list[np.ndarray]] = None,
        candidate_metadata: Optional[list[dict]] = None,
    ) -> dict[int, Track]:
        """Actualiza tracks a partir de las detecciones del frame actual.

        Args:
            detections: lista de posiciones (x, y, z) en coords pixel / mm.
                Estas son las detecciones *high-confidence* — pueden
                tanto actualizar tracks existentes COMO spawnear nuevos.
            detection_metas: lista paralela opcional de dicts de metadata
                per detección. Cuando una detección se matchea (o
                registra un track nuevo), su metadata se appendea a
                ``track.meta["detection_history"]`` (capped a
                ``DETECTION_HISTORY_MAX`` samples). Consumers downstream
                leen esa historia para labels agregados per-track
                (ej. clasificación adult/child).
                **Requerido para ghost adoption**: la key ``"bbox"``
                (lista/tupla ``[x1, y1, x2, y2]``) habilita la adopción
                desde el ghost pool al spawnear tracks nuevos. Sin
                ``bbox`` en la metadata, ghost adoption NO opera aunque
                el pool esté poblado (cae a spawn de track nuevo
                silenciosamente). Pasar siempre ``bbox`` en producción.
            candidate_positions: lista opcional de detecciones
                *low-confidence*. Solo se usa para re-asociar tracks
                existentes que no matchearon una detección
                high-confidence en el pass 1 (matching ByteTrack-style
                de dos etapas). Las detecciones low-confidence que NO
                matchean ningún track unmatched se descartan
                silenciosamente — nunca spawnean tracks nuevos. El
                default ``None`` desactiva este comportamiento por
                completo (el tracker degenera al comportamiento previo
                single-bucket).
            candidate_metadata: lista paralela opcional de metadata
                para ``candidate_positions``. Misma semántica que
                ``detection_metas`` pero solo se aplica cuando un
                candidato efectivamente re-asocia con un track existente.
        """
        det_arr = (
            np.asarray(detections, dtype=float)
            if len(detections) > 0
            else np.empty((0, 3))
        )

        if detection_metas is not None and len(detection_metas) != len(detections):
            raise ValueError(
                "detection_metas length must equal detections length "
                f"({len(detection_metas)} != {len(detections)})"
            )

        cand_arr = (
            np.asarray(candidate_positions, dtype=float)
            if candidate_positions is not None and len(candidate_positions) > 0
            else np.empty((0, 3))
        )
        if (
            candidate_metadata is not None
            and candidate_positions is not None
            and len(candidate_metadata) != len(candidate_positions)
        ):
            raise ValueError(
                "candidate_metadata length must equal candidate_positions "
                f"length ({len(candidate_metadata)} != "
                f"{len(candidate_positions)})"
            )

        def _meta_for(idx: int) -> Optional[dict]:
            return detection_metas[idx] if detection_metas is not None else None

        def _cand_meta_for(idx: int) -> Optional[dict]:
            return candidate_metadata[idx] if candidate_metadata is not None else None

        # Avanza el filtro Kalman de cada track un frame ANTES de la
        # asociación. Los tracks matcheados después llaman
        # kalman.update(z) para absorber la medición; los unmatched
        # mantienen el estado post-predict, que es la extrapolación
        # correcta para el frame faltante.
        for track in self._tracks.values():
            if track.kalman is not None:
                # Velocity decay para tracks que entran a este predict ya
                # en PENDING (= venían de uno o más misses sin obs).
                # Multiplicar (vx, vy) por pending_velocity_decay hace que
                # la extrapolación converja exponencialmente hacia "track
                # quieto" en vez de seguir empujando con la velocidad de
                # cuando había detecciones. Skip cuando decay >= 1.0 (off)
                # para evitar trabajo en el hot path del back-compat.
                #
                # Grace period: los primeros ``pending_grace_frames`` de
                # PENDING preservan velocidad completa (sin decay) para
                # modelar oclusiones cortas donde la persona sigue
                # caminando detrás del oclusor. Sin grace, decay desde el
                # primer miss puede congelar el predict atrás de la
                # detección cuando reaparece, rompiendo el reid. Con
                # grace, los primeros N frames extrapolan honest a la
                # velocidad observada (acepta drift transitorio) y solo
                # entonces empieza el decay si la pausa se extiende.
                # ``track.disappeared`` cuenta misses consecutivos, así
                # que > grace_frames significa "ya pasamos la ventana".
                if (
                    track.state == PENDING
                    and self.pending_velocity_decay < 1.0
                    and track.disappeared > self.pending_grace_frames
                ):
                    track.kalman.x[2:] *= self.pending_velocity_decay
                track.kalman.predict()

        if len(self._tracks) == 0:
            # Sin tracks vivos: las detecciones low-conf no pueden
            # re-asociar contra nada (y nunca spawnean). High-conf primero
            # intentan adoptar un ghost del pool; si no, spawnean track nuevo.
            if self._ghosts:
                still_unmatched: list[int] = []
                for i, det in enumerate(det_arr):
                    det_meta = _meta_for(i)
                    det_bbox: Optional[tuple[float, float, float, float]] = None
                    if isinstance(det_meta, dict):
                        bbox_val = det_meta.get("bbox")
                        if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
                            det_bbox = tuple(float(v) for v in bbox_val)  # type: ignore[assignment]
                    adopted_id = self._try_adopt_ghost(det, det_bbox)
                    if adopted_id is not None:
                        self._append_detection_meta(self._tracks[adopted_id], det_meta)
                    else:
                        still_unmatched.append(i)
                for i in still_unmatched:
                    tid = self._register(det_arr[i])
                    self._append_detection_meta(self._tracks[tid], _meta_for(i))
            else:
                for i, det in enumerate(det_arr):
                    tid = self._register(det)
                    self._append_detection_meta(self._tracks[tid], _meta_for(i))
            self._age_ghost_pool()
            return self.tracks

        track_ids = list(self._tracks.keys())
        # Para el matching, los tracks PENDING usan la posición actual
        # propagada por Kalman (que es la predicción para ESTE frame,
        # ya que predict se acaba de llamar arriba); CONFIRMED/CANDIDATE
        # usan el último centroide observado. Misma política de gating
        # que pre-Kalman — solo PENDING obtiene la referencia propagada,
        # así tracks CONFIRMED adyacentes no pueden driftear unos hacia
        # otros a lo largo de un único frame.
        track_refs = np.array(
            [
                (
                    self._pending_match_ref(self._tracks[t])
                    if self._tracks[t].state == PENDING
                    else self._tracks[t].last_position
                )
                for t in track_ids
            ],
            dtype=float,
        )

        matches, unmatched_t, unmatched_d = self._associate(
            track_refs, det_arr, track_ids
        )

        # Aplicar matches
        for t_idx, d_idx in matches:
            tid = track_ids[t_idx]
            self._record_hit(self._tracks[tid], det_arr[d_idx])
            self._append_detection_meta(self._tracks[tid], _meta_for(d_idx))

        # Pass 3 ByteTrack-style: intentar re-asociar los tracks aún
        # unmatched contra las detecciones low-confidence (candidatos)
        # con el reid gate ancho. Las detecciones candidatas que no
        # binden a ningún track acá se dropean — nunca spawnean tracks
        # nuevos. Se saltea entero cuando no se pasaron candidatos
        # (feature off).
        if len(cand_arr) > 0 and unmatched_t:
            unmatched_track_refs = track_refs[unmatched_t]
            unmatched_track_ids = [track_ids[i] for i in unmatched_t]
            cand_matches, _, _ = self._associate_candidates(
                unmatched_track_refs,
                cand_arr,
                unmatched_track_ids,
            )
            still_unmatched_t_local: set[int] = set(range(len(unmatched_t)))
            for local_t_idx, c_idx in cand_matches:
                tid = unmatched_track_ids[local_t_idx]
                self._record_hit(self._tracks[tid], cand_arr[c_idx])
                self._append_detection_meta(
                    self._tracks[tid],
                    _cand_meta_for(c_idx),
                )
                still_unmatched_t_local.discard(local_t_idx)
            unmatched_t = [unmatched_t[i] for i in sorted(still_unmatched_t_local)]

        # Marcar tracks unmatched como missed
        for t_idx in unmatched_t:
            tid = track_ids[t_idx]
            self._record_miss(self._tracks[tid])

        # Adopción del ghost pool ANTES de spawnear tracks nuevos: para cada
        # detección HIGH-conf unmatched, intentar adoptar un ghost reciente
        # (IoU + dist gates). Si se adopta, se preserva el ID + counter meta
        # del track muerto → continuidad cross-gap-de-detección sin perder el
        # cruce ya registrado.
        if self._ghosts and unmatched_d:
            still_unmatched_d: list[int] = []
            for d_idx in unmatched_d:
                det_meta = _meta_for(d_idx)
                det_bbox: Optional[tuple[float, float, float, float]] = None
                if isinstance(det_meta, dict):
                    bbox_val = det_meta.get("bbox")
                    if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
                        det_bbox = tuple(float(v) for v in bbox_val)  # type: ignore[assignment]
                adopted_id = self._try_adopt_ghost(det_arr[d_idx], det_bbox)
                if adopted_id is not None:
                    self._append_detection_meta(self._tracks[adopted_id], det_meta)
                else:
                    still_unmatched_d.append(d_idx)
            unmatched_d = still_unmatched_d

        # Registrar tracks nuevos solo para detecciones HIGH-confidence
        # unmatched. Las detecciones low-confidence (candidatas) se
        # dropean intencionalmente.
        for d_idx in unmatched_d:
            tid = self._register(det_arr[d_idx])
            self._append_detection_meta(self._tracks[tid], _meta_for(d_idx))

        # Purgar LOST
        to_remove = [tid for tid, t in self._tracks.items() if t.state == LOST]
        for tid in to_remove:
            del self._tracks[tid]

        self._age_ghost_pool()
        return self.tracks

    def _age_ghost_pool(self) -> None:
        """Envejece todos los ghosts del pool en 1 frame; descarta los que
        excedieron la adoption window (ya no tiene sentido esperar)."""
        if not self._ghosts:
            return
        expired = []
        for tid, ghost in self._ghosts.items():
            ghost.age += 1
            if ghost.age > self.adoption_window_frames:
                expired.append(tid)
        for tid in expired:
            del self._ghosts[tid]

    DETECTION_HISTORY_MAX = 60  # ~4s a 15 FPS — alcanza para mayoría estable

    def _append_detection_meta(self, track: Track, meta: Optional[dict]) -> None:
        if meta is None:
            return
        history = track.meta.setdefault("detection_history", [])
        history.append(dict(meta))
        if len(history) > self.DETECTION_HISTORY_MAX:
            del history[: len(history) - self.DETECTION_HISTORY_MAX]

    # -------------------------------------------------------- association
    def _associate(
        self,
        track_refs: np.ndarray,
        det_arr: np.ndarray,
        track_ids: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match Hungarian con gating, dos passes.

        El pass 1 usa el gate state-aware (``reid_gate_px`` para
        PENDING, ``max_distance`` para el resto). El pass 2 vuelve a
        correr Hungarian sobre lo que quedó unmatched con el
        ``reid_gate_px`` más ancho para todos — así un CONFIRMED que
        está a punto de pasar a PENDING tiene una oportunidad de
        re-matchear la detección huérfana antes de que esa detección se
        registre como track nuevo. Sin el pass 2, jitter de bbox o una
        única detección dropeada produce pares de tracks fantasma (el
        CONFIRMED original transiciona a PENDING mientras se crea un ID
        nuevo desde la misma persona física).

        Devuelve ``(matches, unmatched_t, unmatched_d)``.
        """
        n_t, n_d = len(track_refs), len(det_arr)
        if n_t == 0 or n_d == 0:
            return [], list(range(n_t)), list(range(n_d))

        dist_2d = np.linalg.norm(
            track_refs[:, np.newaxis, :2] - det_arr[np.newaxis, :, :2], axis=2
        )

        if track_refs.shape[1] > 2 and det_arr.shape[1] > 2:
            depth_delta = np.abs(
                track_refs[:, np.newaxis, 2] - det_arr[np.newaxis, :, 2]
            )
        else:
            depth_delta = np.zeros_like(dist_2d)

        # Pass 1: gate per-track. PENDING obtiene el re-id gate más
        # ancho para terminar de recuperarse; el resto obtiene el gate
        # ajustado para evitar que tracks adyacentes se swapeen.
        gate_pass1 = np.array(
            [
                (
                    self.reid_gate_px
                    if self._tracks[tid].state == PENDING
                    else self.max_distance
                )
                for tid in track_ids
            ],
            dtype=float,
        )[:, np.newaxis]

        matches, matched_t, matched_d = self._hungarian_with_gate(
            dist_2d,
            depth_delta,
            gate_pass1,
        )

        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [j for j in range(n_d) if j not in matched_d]

        # Pass 2: relajar el gate a reid_gate_px para los leftovers.
        # Esta es la safety net que previene la creación de tracks
        # duplicados cuando el bbox de un CONFIRMED jittereó apenas más
        # allá de max_distance — el depth gate no cambia, así que dos
        # personas distintas a profundidades distintas tampoco pueden
        # swapear IDs acá.
        if unmatched_t and unmatched_d:
            sub_dist = dist_2d[np.ix_(unmatched_t, unmatched_d)]
            sub_depth = depth_delta[np.ix_(unmatched_t, unmatched_d)]
            sub_gate = np.full(
                (len(unmatched_t), 1),
                self.reid_gate_px,
                dtype=float,
            )
            sub_matches, _, _ = self._hungarian_with_gate(
                sub_dist,
                sub_depth,
                sub_gate,
            )
            for r_local, c_local in sub_matches:
                t_idx = unmatched_t[r_local]
                d_idx = unmatched_d[c_local]
                matches.append((t_idx, d_idx))
                matched_t.add(t_idx)
                matched_d.add(d_idx)
            unmatched_t = [i for i in range(n_t) if i not in matched_t]
            unmatched_d = [j for j in range(n_d) if j not in matched_d]

        return matches, unmatched_t, unmatched_d

    def _associate_candidates(
        self,
        track_refs: np.ndarray,
        det_arr: np.ndarray,
        track_ids: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Re-asociación Hungarian de tracks leftover vs detecciones low-conf.

        Siempre usa el ``reid_gate_px`` ancho para cada track (intuición
        ByteTrack: los tracks leftover son los que el detector
        high-confidence falló en matchear este frame, así que aceptamos
        cualquier detección low-conf que caiga dentro del radio de
        re-id). El depth gate sigue aplicando como safety net
        secundaria, así dos personas físicamente distintas a
        profundidades muy distintas no pueden swapear IDs incluso
        cuando una falta este frame.

        Devuelve ``(matches, unmatched_t, unmatched_d)`` con índices
        locales a los inputs.
        """
        n_t, n_d = len(track_refs), len(det_arr)
        if n_t == 0 or n_d == 0:
            return [], list(range(n_t)), list(range(n_d))

        dist_2d = np.linalg.norm(
            track_refs[:, np.newaxis, :2] - det_arr[np.newaxis, :, :2], axis=2
        )

        if track_refs.shape[1] > 2 and det_arr.shape[1] > 2:
            depth_delta = np.abs(
                track_refs[:, np.newaxis, 2] - det_arr[np.newaxis, :, 2]
            )
        else:
            depth_delta = np.zeros_like(dist_2d)

        gate = np.full((n_t, 1), self.reid_gate_px, dtype=float)
        matches, matched_t, matched_d = self._hungarian_with_gate(
            dist_2d,
            depth_delta,
            gate,
        )
        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [j for j in range(n_d) if j not in matched_d]
        # track_ids aceptado por simetría con _associate pero sin uso —
        # no hay switch de comportamiento per-track (el gate es
        # uniformemente ancho acá).
        del track_ids
        return matches, unmatched_t, unmatched_d

    def _hungarian_with_gate(
        self,
        dist_2d: np.ndarray,
        depth_delta: np.ndarray,
        gate_per_track: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Corre Hungarian sobre una matriz de costos con gating de distancia + depth.

        ``gate_per_track`` se broadcastea contra ``dist_2d``; las depths
        sobre ``max_depth_delta`` siempre se rechazan independientemente
        del gate.

        Post-Hungarian: ratio test (Lowe-style) si
        ``ambiguous_match_ratio < 1.0``. Rechaza matches donde el costo
        del best no es significativamente menor que el del 2do-mejor
        alternativa (otra detección para el mismo track O otro track
        para la misma detección). Los tracks de matches rechazados se
        retornan UNMATCHED (caller los marcará como missed); las
        detecciones se retornan CONSUMED (incluidas en ``matched_d``)
        para evitar que spawneen tracks nuevos — preferimos perder una
        obs ambigua antes que duplicar el ID.

        Devuelve ``(matches, matched_track_indices, matched_det_indices)``.
        """
        INF = 1e9
        cost = dist_2d.copy()
        cost[dist_2d > gate_per_track] = INF
        cost[depth_delta > self.max_depth_delta] = INF

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        # Detecciones consumidas por matches rechazados — no son matched
        # en el sentido de generar un hit, pero el caller las trata como
        # matched para no spawnear tracks nuevos desde ellas.
        consumed_d: set[int] = set()
        ratio = self.ambiguous_match_ratio
        ratio_on = ratio < 1.0

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= INF:
                continue
            r = int(r)
            c = int(c)
            best = cost[r, c]

            if ratio_on:
                # 2nd-best detección para el track r (excluyendo c).
                row_copy = cost[r].copy()
                row_copy[c] = INF
                second_row = float(row_copy.min())
                # 2nd-best track para la detección c (excluyendo r).
                col_copy = cost[:, c].copy()
                col_copy[r] = INF
                second_col = float(col_copy.min())
                ambiguous = False
                # Si el 2do-mejor existe (no es INF) y está dentro del
                # ratio del best, el match es ambiguo.
                if second_row < INF / 2 and best > ratio * second_row:
                    ambiguous = True
                if not ambiguous and second_col < INF / 2 and best > ratio * second_col:
                    ambiguous = True
                if ambiguous:
                    consumed_d.add(c)
                    continue

            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)

        # Marcar detecciones consumidas como matched a nivel del caller
        # para que NO entren a unmatched_d y por ende no spawneen tracks.
        matched_d.update(consumed_d)
        return matches, matched_t, matched_d

    def _pending_match_ref(self, track: Track) -> np.ndarray:
        """Referencia de match para un track PENDING en el frame actual.

        Dentro de ``update()`` ya avanzamos el filtro de cada track,
        así que ``kalman.position`` es la expectativa del modelo para
        ESTE frame (la property `predicted_position` peekearía UN paso
        más adelante, lo que sería incorrecto para matching del frame
        actual).
        """
        if track.kalman is not None:
            xy = track.kalman.position
            z = track.positions[-1][2] if len(track.positions[-1]) > 2 else 0.0
            return np.array([xy[0], xy[1], z], dtype=float)
        # Fallback defensivo (Tracks creados sin filtro).
        if len(track.positions) < 2:
            return track.positions[-1].copy()
        velocity = track.positions[-1] - track.positions[-2]
        return track.positions[-1] + velocity

    # -------------------------------------------------- per-track updates
    def _make_kalman(self, centroid: np.ndarray) -> TrackKalman:
        return TrackKalman(
            initial_position=centroid[:2],
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            initial_velocity_uncertainty=self.initial_velocity_uncertainty,
        )

    def _register(self, centroid: np.ndarray) -> int:
        tid = self._next_id
        # Skip IDs ya en uso por algún track vivo — defense contra
        # colisión en el rollover. En condiciones normales el while
        # corre 0 veces (los tracks viejos hace rato murieron antes
        # de que el contador cicle 65536 veces); solo entra si el
        # max_track_id está bajado artificialmente (tests).
        while tid in self._tracks or tid in self._ghosts:
            tid = (tid + 1) % self.max_track_id
        centroid = np.asarray(centroid, dtype=float)
        self._tracks[tid] = Track(
            track_id=tid,
            positions=[centroid.copy()],
            state=CANDIDATE,
            hits=1,
            kalman=self._make_kalman(centroid),
            last_observed_position=centroid.copy(),
        )
        self._next_id = (tid + 1) % self.max_track_id
        return tid

    def _record_hit(self, track: Track, centroid: np.ndarray) -> None:
        centroid = np.asarray(centroid, dtype=float)
        track.positions.append(centroid.copy())
        track.last_observed_position = centroid.copy()
        _trim_positions(track)
        track.disappeared = 0
        track.hits += 1
        if track.kalman is not None:
            track.kalman.update(centroid[:2])
        if track.state == CANDIDATE and track.hits >= self.confirm_frames:
            track.state = CONFIRMED
        elif track.state == PENDING:
            # Recuperado; preservar ID y meta del counter.
            track.state = CONFIRMED

    def _record_miss(self, track: Track) -> None:
        track.disappeared += 1
        if track.state == CANDIDATE:
            # Misses no confirmados son baratos: dropear después de
            # max_disappeared, pero también dropear inmediatamente al
            # primer miss si confirm_frames requiere hits consecutivos.
            if track.disappeared > self.max_disappeared:
                track.state = LOST
            return

        if track.state == CONFIRMED:
            track.state = PENDING

        # Empujar la predicción Kalman a `positions` durante PENDING así
        # consumers downstream (counter line-cross, viewer) ven al track
        # moverse — SIGUE a la persona durante el gap del detector, así puede
        # SALIR del ROI y emitir el conteo (el conteo dispara en el exit). Sin
        # este push, una persona que cruza + se pierde mid-ROI quedaría clavada
        # adentro y su cruce nunca se contaría. La magnitud del paso se capea a
        # `max_distance` (anti-teleport con velocidad residual alta); el decay
        # (`pending_velocity_decay`) la va frenando. `z` carry-forwardea.
        #
        # El doble-conteo del drift (un track que extrapola CRUZANDO la línea
        # sin detección real) NO se previene acá: lo maneja el counter, que solo
        # registra un cruce cuando lo respalda una detección real
        # (``disappeared == 0``), no un frame de pura predicción.
        if track.kalman is not None and track.positions:
            last = track.positions[-1]
            xy_pred = track.kalman.position
            dx = float(xy_pred[0]) - float(last[0])
            dy = float(xy_pred[1]) - float(last[1])
            step_mag = (dx * dx + dy * dy) ** 0.5
            if step_mag > self.max_distance and step_mag > 0:
                scale = self.max_distance / step_mag
                dx *= scale
                dy *= scale
            z = float(last[2]) if len(last) > 2 else 0.0
            track.positions.append(
                np.array(
                    [float(last[0]) + dx, float(last[1]) + dy, z],
                    dtype=float,
                )
            )
            _trim_positions(track)

        # Keep-alive dentro del ROI: si la última posición (predicha) del
        # track cae dentro del ROI de conteo, NO lo matamos por timeout —
        # se mantiene PENDING. Modela la persona que cruzó y se quedó adentro
        # (el detector la pierde un rato). Cuando vuelve a moverse, o re-id
        # binde y vuelve a CONFIRMED, o la predicción Kalman la saca del ROI
        # y entonces sí aplica el timeout normal. Acotado a
        # ``keepalive_max_frames`` misses consecutivos para garbage-collectear
        # huérfanos (re-id falló, persona ya no está); como ``disappeared`` se
        # resetea con cualquier hit, el lingering real nunca llega al cap.
        if (
            self._inside_keepalive_roi(track)
            and track.disappeared <= self.keepalive_max_frames
        ):
            return

        inside_ka = self._inside_keepalive_roi(track)
        last = track.positions[-1] if track.positions else (0.0, 0.0, 0.0)
        if track.state == PENDING and track.disappeared > self.pending_max_frames:
            track.state = LOST
            self._enghost(track)
            logger.info(  # TRACKDBG [REVERT]
                "TRACKDBG death tid=%d reason=pending_max_frames disappeared=%d inside_keepalive=%s last_pos=(%.0f,%.0f)",
                track.track_id, track.disappeared, inside_ka, float(last[0]), float(last[1]),
            )
            return

        # Aplicar también el cap legacy max_disappeared como upper bound.
        if track.disappeared > self.max_disappeared:
            track.state = LOST
            self._enghost(track)
            logger.info(  # TRACKDBG [REVERT]
                "TRACKDBG death tid=%d reason=max_disappeared disappeared=%d inside_keepalive=%s last_pos=(%.0f,%.0f)",
                track.track_id, track.disappeared, inside_ka, float(last[0]), float(last[1]),
            )

    def _enghost(self, track: Track) -> None:
        """Snapshotea un track recién muerto al ghost pool para que pueda ser
        adoptado por un track nuevo dentro de ``adoption_window_frames``. Solo
        registra ghosts que tuvieron al menos una detección REAL (sin
        ``last_observed_position`` no podemos validar la adopción)."""
        if self.adoption_window_frames <= 0:
            return
        if track.last_observed_position is None:
            return
        # Buscar el último bbox real en el historial de detecciones.
        last_bbox: Optional[tuple[float, float, float, float]] = None
        history = track.meta.get("detection_history") if isinstance(track.meta, dict) else None
        if history:
            for rec in reversed(history):
                bbox = rec.get("bbox") if isinstance(rec, dict) else None
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    last_bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
                    break
        self._ghosts[track.track_id] = _GhostInfo(
            track_id=track.track_id,
            last_observed_position=track.last_observed_position.copy(),
            last_bbox=last_bbox,
            meta_snapshot=dict(track.meta) if isinstance(track.meta, dict) else {},
            age=0,
        )

    def _try_adopt_ghost(
        self,
        centroid: np.ndarray,
        bbox: Optional[tuple[float, float, float, float]],
    ) -> Optional[int]:
        """Intenta adoptar un ghost del pool. Aplica DOS gates geométricos
        (IoU + distancia) y elige el ghost VÁLIDO más cercano. Si adoptado,
        lo remueve del pool y resucita el Track con su ID + meta restaurado.
        Devuelve el track_id adoptado o ``None``."""
        if not self._ghosts or bbox is None:
            return None
        best_tid: Optional[int] = None
        best_dist: float = float("inf")
        best_iou: float = 0.0
        for tid, ghost in self._ghosts.items():
            if ghost.last_bbox is None:
                continue
            iou = _bbox_iou(bbox, ghost.last_bbox)
            if iou < self.adoption_iou_min:
                continue
            gpos = ghost.last_observed_position
            dx = float(centroid[0]) - float(gpos[0])
            dy = float(centroid[1]) - float(gpos[1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > self.adoption_max_dist_px:
                continue
            if dist < best_dist:
                best_dist = dist
                best_iou = iou
                best_tid = tid
        if best_tid is None:
            return None
        ghost = self._ghosts.pop(best_tid)
        self._resurrect_ghost(ghost, centroid)
        logger.info(  # TRACKDBG [REVERT]
            "TRACKDBG ghost_adopted tid=%d dist=%.1f iou=%.3f age=%d",
            best_tid, best_dist, best_iou, ghost.age,
        )
        return best_tid

    def _resurrect_ghost(self, ghost: _GhostInfo, centroid: np.ndarray) -> None:
        """Recrea el Track con el track_id + meta del ghost, anclado en la
        nueva detección. El Kalman se re-inicia desde la posición actual (el
        estado anterior es stale tras el gap). El track entra como CONFIRMED
        (no CANDIDATE) porque la adopción es evidencia de continuidad."""
        centroid = np.asarray(centroid, dtype=float)
        self._tracks[ghost.track_id] = Track(
            track_id=ghost.track_id,
            positions=[centroid.copy()],
            state=CONFIRMED,
            hits=max(2, self.confirm_frames),  # entra ya confirmado
            kalman=self._make_kalman(centroid),
            meta=ghost.meta_snapshot,
            last_observed_position=centroid.copy(),
        )

    def _inside_keepalive_roi(self, track: Track) -> bool:
        """True si la última posición del track cae dentro del ROI de
        keep-alive. Sin ROI configurado, siempre False (sin keep-alive)."""
        roi = self.keepalive_roi
        if roi is None or not track.positions:
            return False
        x_min, x_max, y_min, y_max = roi
        last = track.positions[-1]
        return x_min <= float(last[0]) <= x_max and y_min <= float(last[1]) <= y_max


def stationary_track_ids(
    tracks: dict[int, Track],
    roi: tuple[float, float, float, float] | None,
    min_frames: int = 20,
    max_movement_px: float = 50.0,
) -> set[int]:
    """IDs de tracks que son **clutter estático**: vivos hace >= ``min_frames``,
    que se movieron < ``max_movement_px`` en toda su historia, y cuyo centroide
    actual está FUERA del ROI.

    Son FPs del detector sobre estructura fija (sombra/borde/piso) que spawnean
    tracks fantasma — la gente real se mueve, el clutter no. Sin ROI devuelve
    set vacío (no filtramos sin zona de conteo). Pensado SOLO para display/stats
    (esconder el bbox fantasma + no inflar el contador de tracks): estos tracks
    están fuera del ROI y nunca cuentan, así que el conteo/tracking no se tocan.
    """
    if roi is None:
        return set()
    x_min, x_max, y_min, y_max = roi
    out: set[int] = set()
    for tid, t in tracks.items():
        positions = getattr(t, "positions", None)
        if not positions or len(positions) < min_frames:
            continue
        cx, cy = float(positions[-1][0]), float(positions[-1][1])
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            continue  # dentro del ROI: nunca lo tratamos como clutter
        xs = [float(p[0]) for p in positions]
        ys = [float(p[1]) for p in positions]
        movement = max(max(xs) - min(xs), max(ys) - min(ys))
        if movement < max_movement_px:
            out.add(tid)
    return out

"""Tracker 3D para trayectorias de personas.

Implementa asociación Hungarian-based con una state machine de corto plazo:

    CANDIDATE  -> CONFIRMED  (luego de N hits consecutivos)
    CONFIRMED  -> PENDING    (ante un frame perdido; predicho con Kalman)
    PENDING    -> CONFIRMED  (re-match dentro del re-id gate)
    PENDING    -> LOST       (luego de pending_max_frames misses consecutivos)

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
        self._next_id = 0
        self._tracks: OrderedDict[int, Track] = OrderedDict()

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
                track.kalman.predict()

        if len(self._tracks) == 0:
            # Sin tracks vivos: las detecciones low-conf no pueden
            # re-asociar contra nada (y nunca spawnean). Solo
            # high-conf spawnean.
            for i, det in enumerate(det_arr):
                tid = self._register(det)
                self._append_detection_meta(self._tracks[tid], _meta_for(i))
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

        return self.tracks

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
        del gate. Devuelve ``(matches, matched_track_indices, matched_det_indices)``.
        """
        INF = 1e9
        cost = dist_2d.copy()
        cost[dist_2d > gate_per_track] = INF
        cost[depth_delta > self.max_depth_delta] = INF

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= INF:
                continue
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))
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
        centroid = np.asarray(centroid, dtype=float)
        self._tracks[tid] = Track(
            track_id=tid,
            positions=[centroid.copy()],
            state=CANDIDATE,
            hits=1,
            kalman=self._make_kalman(centroid),
        )
        self._next_id += 1
        return tid

    def _record_hit(self, track: Track, centroid: np.ndarray) -> None:
        centroid = np.asarray(centroid, dtype=float)
        track.positions.append(centroid.copy())
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

        # Empujar la predicción Kalman a `positions` durante PENDING
        # así consumers downstream (counter line-cross, viewer) ven al
        # track moverse en lugar de quedar clavado en su última
        # observación. Sin este push, una persona que cruza la línea +
        # sale del FOV antes del próximo match nunca dispara el exit
        # branch del counter: la posición congelada queda IN-ROI, el
        # track muere por max_disappeared / pending_max_frames, y el
        # `last_label` se pierde sin emitir CountEvent. La magnitud del
        # paso se capea a `max_distance` para evitar teleportaciones
        # cuando Kalman extrapola con velocidad residual alta. `z`
        # carry-forwardea — Kalman 2D no modela depth.
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

        if track.state == PENDING and track.disappeared > self.pending_max_frames:
            track.state = LOST
            return

        # Aplicar también el cap legacy max_disappeared como upper bound.
        if track.disappeared > self.max_disappeared:
            track.state = LOST

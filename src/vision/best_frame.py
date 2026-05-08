"""Best-frame selector for tracked persons.

When the optional ``best_frame`` feature is enabled in ``hardware.yaml``,
the pipeline buffers a small rolling window of frames per active track
and picks one representative JPG when the track produces a count event
at the gate line. The chosen frame is written to local disk only —
**never** transmitted over MQTT, and **never** uploaded to the cloud —
and only its on-device path is attached to the count event metadata.

The privacy story is enforced as a stack of independent guarantees:

  1. Default OFF in ``hardware.yaml`` (zero storage, zero PII).
  2. Local-only writes (no MQTT publish of the image bytes — see
     ``src/main.py`` for the publish path).
  3. Short retention enforced by ``scripts/purge_best_frames.py`` +
     systemd timer.
  4. Optional anonymisation (``scripts/export_anonymized.py``) when the
     operator wants to ship samples for external labelling.
  5. Operator paperwork (DPIA, signage, privacy policy) gates flipping
     the toggle on — see ``docs/privacy.md``.

This module owns layers (1)–(2): the scoring math and the per-track
buffer plumbing. The other layers live in scripts + docs because they
need to keep working even when the pipeline is offline.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BufferedFrame:
    """One frame snapshot kept for a single live track.

    Holds the raw image bytes plus the per-frame scoring components so
    the picker can rank candidates without re-running detection. Image
    is stored as a numpy array; the buffer caps total entries (see
    ``buffer_size``) so RAM stays bounded.
    """

    frame_image: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    score: float
    score_components: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _bbox_area_norm(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> float:
    """Normalised bbox area in [0, 1]: area / frame_area. Saturates at 1.0
    so a degenerate bbox larger than the frame doesn't blow up the score.
    """
    x1, y1, x2, y2 = bbox
    w = max(0, int(x2) - int(x1))
    h = max(0, int(y2) - int(y1))
    h_frame, w_frame = frame_shape[:2]
    if h_frame <= 0 or w_frame <= 0:
        return 0.0
    area_norm = (w * h) / float(h_frame * w_frame)
    return float(min(1.0, max(0.0, area_norm)))


def _bbox_centrality(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> float:
    """Cercanía del centroide del bbox al centro del frame, en [0, 1].

    1.0 = centroide exactamente en el centro; 0.0 = en una esquina. La
    distancia se normaliza contra ``hypot(W/2, H/2)`` para que el rango
    sea estable a cualquier resolución. La preferencia por el centro
    refleja la geometría fisheye: el centro tiene la menor distorsión y
    los detalles ahí son los más útiles para active learning.
    """
    x1, y1, x2, y2 = bbox
    cx = (int(x1) + int(x2)) / 2.0
    cy = (int(y1) + int(y2)) / 2.0
    h_frame, w_frame = frame_shape[:2]
    if h_frame <= 0 or w_frame <= 0:
        return 0.0
    fx, fy = w_frame / 2.0, h_frame / 2.0
    max_dist = float(np.hypot(fx, fy)) or 1.0
    dist = float(np.hypot(cx - fx, cy - fy))
    return float(max(0.0, 1.0 - dist / max_dist))


def _bbox_sharpness(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Varianza Laplaciana del crop del bbox, normalizada a [0, 1].

    La varianza Laplaciana es el indicador clásico de motion blur (más
    alta = más bordes nítidos). Se normaliza dividiendo por una constante
    empírica ``REF`` y se satura en 1.0 para que la métrica conviva con
    los otros componentes en escala homogénea. Si el crop es vacío
    (bbox inválido fuera del frame) cae a 0.0.
    """
    if image is None or image.size == 0:
        return 0.0
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Empirical reference: a well-focused IMX708 crop at 1152x648 lands in
    # the 200-500 range. Cap at 1.0 so high-detail crops can't dominate the
    # weighted sum.
    REF = 500.0
    return float(min(1.0, var / REF))


def score_frame(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    confidence: float,
    frame_shape: tuple[int, int],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute the weighted score of a single (image, detection) pair.

    Returns ``(score, components)`` where ``components`` carries each
    metric in [0, 1] for diagnostics. The total is ``sum(w_i * c_i)``;
    callers are responsible for choosing weights that sum to ~1 (see the
    validator in ``src/config/hardware.py``).

    All four components saturate in [0, 1], so the maximum total under
    the documented default weights is 1.0.
    """
    conf = float(max(0.0, min(1.0, confidence)))
    area = _bbox_area_norm(bbox, frame_shape)
    centre = _bbox_centrality(bbox, frame_shape)
    sharp = _bbox_sharpness(image, bbox)

    components = {
        "confidence": conf,
        "bbox_area": area,
        "centrality": centre,
        "sharpness": sharp,
    }
    score = (
        weights.get("confidence_weight", 0.0) * conf
        + weights.get("bbox_area_weight", 0.0) * area
        + weights.get("centrality_weight", 0.0) * centre
        + weights.get("sharpness_weight", 0.0) * sharp
    )
    return float(score), components


def pick_best(buffer: list[BufferedFrame]) -> Optional[BufferedFrame]:
    """Return the single best-scoring frame from the buffer (or ``None``).

    Ties broken by ``timestamp`` (newer wins) so when the score saturates
    on a stationary subject the operator-facing JPG matches roughly the
    moment the count fires, not an arbitrary earlier sample.
    """
    if not buffer:
        return None
    return max(buffer, key=lambda bf: (bf.score, bf.timestamp))


# ---------------------------------------------------------------------------
# Per-track buffer manager
# ---------------------------------------------------------------------------


def _safe_makedirs(path: Path) -> bool:
    """``mkdir -p`` wrapper that swallows OSError and returns success.

    The pipeline must never crash because the JPG dir wasn't writable —
    we log and skip the save. Returns ``True`` on success.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("best_frame: cannot create %s: %s", path, e)
        return False


@dataclass
class _TrackBuffer:
    """Internal container — rolling list capped at ``max_size``."""

    max_size: int
    frames: list[BufferedFrame] = field(default_factory=list)

    def push(self, bf: BufferedFrame) -> None:
        self.frames.append(bf)
        # Drop the *oldest* element when over capacity. Keeping a strict
        # FIFO instead of "drop the lowest score" guarantees that a track
        # standing in the gate for many seconds still gets a recent
        # candidate even if an earlier high-score frame would otherwise
        # pin the buffer.
        if len(self.frames) > self.max_size:
            del self.frames[0 : len(self.frames) - self.max_size]


class BestFrameManager:
    """Per-track rolling buffer + JPG writer.

    Lifecycle:

      - ``observe(track_id, frame, detection)`` is called every pipeline
        frame for every active track. Cheap: just scores + buffers.
      - ``commit(track_id, event_timestamp)`` is called when a track
        emits a count event. Picks the best frame, writes it, returns
        the path. Returns ``None`` on any error so the publish path
        doesn't break.
      - ``forget(track_id)`` discards the buffer when a track ends
        without a count event (saves RAM in long sessions).

    Filename layout: ``<output_dir>/<YYYY-MM-DD>/<track_id>_<ts>.jpg``
    with ``ts`` formatted as ``%Y%m%dT%H%M%S_%f`` for sortability.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        buffer_size: int = 20,
        jpeg_quality: int = 85,
        weights: Optional[dict[str, float]] = None,
        clock: Optional[Any] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._buffer_size = max(1, int(buffer_size))
        self._jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self._weights = (
            dict(weights)
            if weights
            else {
                "confidence_weight": 0.4,
                "bbox_area_weight": 0.2,
                "centrality_weight": 0.2,
                "sharpness_weight": 0.2,
            }
        )
        self._buffers: dict[int, _TrackBuffer] = {}
        # Injectable for tests; defaults to time.time.
        self._clock = clock or time.time

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    # ---------------------------------------------------------- per-frame
    def observe(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        """Score the current frame and push it to the track's buffer.

        Copies the frame so subsequent in-place mutations by the
        pipeline (annotations, colour conversions) don't poison the
        buffered candidate.
        """
        if frame is None or frame.size == 0:
            return
        try:
            frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
            score, components = score_frame(
                frame,
                bbox,
                confidence,
                frame_shape,
                self._weights,
            )
            bf = BufferedFrame(
                frame_image=frame.copy(),
                bbox=tuple(int(v) for v in bbox),  # type: ignore[arg-type]
                confidence=float(confidence),
                score=score,
                score_components=components,
                timestamp=float(self._clock()),
            )
        except Exception:
            logger.exception(
                "best_frame.observe failed for track_id=%s",
                track_id,
            )
            return
        buf = self._buffers.get(track_id)
        if buf is None:
            buf = _TrackBuffer(max_size=self._buffer_size)
            self._buffers[track_id] = buf
        buf.push(bf)

    # --------------------------------------------------------- per-event
    def commit(
        self,
        track_id: int,
        event_timestamp: float,
    ) -> Optional[str]:
        """Pick best, write JPG, drop buffer. Returns the path or None.

        Errors (no buffer / write failure) all collapse to ``None`` and
        a warning log so the publish path can include / exclude the
        ``best_frame_path`` field cleanly.
        """
        buf = self._buffers.pop(track_id, None)
        if buf is None or not buf.frames:
            return None
        best = pick_best(buf.frames)
        if best is None:
            return None
        try:
            return self._write_jpg(track_id, event_timestamp, best)
        except Exception:
            logger.exception(
                "best_frame.commit write failed track_id=%s",
                track_id,
            )
            return None

    def forget(self, track_id: int) -> None:
        """Drop the buffer for a track that ended without an event."""
        self._buffers.pop(track_id, None)

    def gc(self, alive_track_ids: set[int]) -> int:
        """Drop buffers for tracks no longer in the live tracker dict.

        The pipeline doesn't always call ``forget`` (e.g. PENDING tracks
        timing out get garbage-collected by the tracker, not signalled
        explicitly here). Periodic gc bounds RAM in long-running sessions.
        Returns the number of buffers dropped.
        """
        stale = [tid for tid in self._buffers if tid not in alive_track_ids]
        for tid in stale:
            del self._buffers[tid]
        return len(stale)

    # --------------------------------------------------------- internal
    def _write_jpg(
        self,
        track_id: int,
        event_timestamp: float,
        bf: BufferedFrame,
    ) -> Optional[str]:
        # Date directory is in *local* time so operators can ls a day's
        # worth of frames quickly. The retention purge uses mtime which
        # is also local-tz friendly.
        dt = datetime.fromtimestamp(event_timestamp)
        day_dir = self._output_dir / dt.strftime("%Y-%m-%d")
        if not _safe_makedirs(day_dir):
            return None
        # Microsecond timestamp for collision-free filenames even when
        # multiple events fire on the same track in a single frame
        # (extremely rare, but cheap to defend against).
        ts_str = datetime.fromtimestamp(
            event_timestamp,
            tz=timezone.utc,
        ).strftime("%Y%m%dT%H%M%SZ_%f")
        fname = f"{int(track_id)}_{ts_str}.jpg"
        path = day_dir / fname
        try:
            ok = cv2.imwrite(
                str(path),
                bf.frame_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
        except cv2.error as e:
            logger.warning("best_frame: imwrite raised %s for %s", e, path)
            return None
        if not ok:
            logger.warning("best_frame: imwrite returned False for %s", path)
            return None
        return str(path)

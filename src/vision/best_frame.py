"""Best-frame selector para personas trackeadas.

Cuando el feature opcional ``best_frame`` está enabled en el config mergeado
(``best_frame.enabled: true``), el pipeline buffea una ventana rolling chica
de frames por track activo y elige un JPG representativo cuando el track
produce un evento de conteo en la línea del gate. El frame elegido se escribe
solo a disco local — **nunca** se transmite por MQTT, y **nunca** se sube
al cloud — y solo el path on-device se adjunta a la metadata del evento de
conteo.

El stack de privacidad se hace cumplir como capas independientes de garantías:

  1. Default OFF en ``config/config.example.yaml`` (zero storage, zero PII).
  2. Escrituras solo locales (sin publish MQTT de los bytes de imagen — ver
     ``src/main.py`` para el path de publish).
  3. Retención corta enforced por ``scripts/purge_best_frames.py`` + timer
     de systemd.
  4. Ofuscación visual opcional (``scripts/export_anonymized.py``) cuando el
     operador quiere shippear samples para labeling externo.
  5. Paperwork del operador (DPIA, signage, privacy policy) gatea poner
     el toggle en on — ver ``docs/privacy.md``.

Este módulo posee las capas (1)–(2): la matemática de scoring y el plumbing
del buffer per-track. Las otras capas viven en scripts + docs porque tienen
que seguir funcionando aún cuando el pipeline esté offline.
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
# Dataclasses públicos
# ---------------------------------------------------------------------------


@dataclass
class BufferedFrame:
    """Un snapshot guardado para un solo track vivo.

    ``frame_image`` es el CROP del bbox con margen de contexto (ver
    ``_crop_with_margin``), NO el frame completo, más los componentes de
    scoring per-frame así el picker puede rankear candidatos sin re-correr
    detección. ``bbox`` queda en coordenadas del frame original (metadata).
    El buffer capea las entries totales (``buffer_size``) y cada entry pesa
    ~0.05-0.2MB en vez de ~2.2MB — ver el racional en ``_CROP_MARGIN_FRAC``.
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
    """Área del bbox normalizada en [0, 1]: area / frame_area. Satura en 1.0
    así un bbox degenerado más grande que el frame no rompe el score.
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
    # Referencia empírica: un crop bien enfocado IMX708 a 1152x648 cae
    # en el rango 200-500. Capear a 1.0 así crops de alto detalle no
    # pueden dominar la suma ponderada.
    REF = 500.0
    return float(min(1.0, var / REF))


def score_frame(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    confidence: float,
    frame_shape: tuple[int, int],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Computa el score ponderado de un par (image, detection).

    Devuelve ``(score, components)`` donde ``components`` lleva cada
    métrica en [0, 1] para diagnóstico. El total es ``sum(w_i * c_i)``;
    los callers son responsables de elegir weights que sumen ~1 (ver
    el validator en ``src/config/loader.py``).

    Los cuatro componentes saturan en [0, 1], así que el total máximo
    bajo los weights default documentados es 1.0.
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


# Margen del crop alrededor del bbox (fracción del ancho/alto del bbox por
# lado). El buffer guarda SOLO el crop, no el frame completo: a 1152x648 cada
# frame copiado pesaba ~2.2MB × buffer_size 20 = ~45MB POR TRACK — con los
# ~10 tracks simultáneos de un site con tráfico, encender el feature rompía
# el presupuesto de RAM del Pi 5 2GB (working set ~270MB, MemoryMax=1500M).
# Un crop cabeza+hombros con 50% de contexto por lado pesa ~0.05-0.2MB y el
# JPG resultante sigue sirviendo para auditoría del operador y labeling.
_CROP_MARGIN_FRAC = 0.5


def _crop_with_margin(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_frac: float = _CROP_MARGIN_FRAC,
) -> Optional[np.ndarray]:
    """Crop del bbox con margen de contexto, clampeado al frame.

    Devuelve una COPIA (mutaciones in-place posteriores del pipeline no la
    tocan) o ``None`` si el bbox es degenerado o cae fuera del frame.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx, my = int(bw * margin_frac), int(bh * margin_frac)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


def pick_best(buffer: list[BufferedFrame]) -> Optional[BufferedFrame]:
    """Devuelve el único frame con el score más alto del buffer (o ``None``).

    Empates resueltos por ``timestamp`` (gana más nuevo) así cuando el
    score satura sobre un sujeto estacionario el JPG operator-facing
    matchea aproximadamente al momento en que dispara el conteo, no a
    un sample arbitrario anterior.
    """
    if not buffer:
        return None
    return max(buffer, key=lambda bf: (bf.score, bf.timestamp))


# ---------------------------------------------------------------------------
# Manager de buffer per-track
# ---------------------------------------------------------------------------


def _safe_makedirs(path: Path) -> bool:
    """Wrapper ``mkdir -p`` que se traga OSError y devuelve éxito.

    El pipeline nunca debe crashear porque el dir de JPGs no era
    writable — logueamos y saltamos el save. Devuelve ``True`` ante
    éxito.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("best_frame: cannot create %s: %s", path, e)
        return False


@dataclass
class _TrackBuffer:
    """Container interno — lista rolling capeada a ``max_size``."""

    max_size: int
    frames: list[BufferedFrame] = field(default_factory=list)

    def push(self, bf: BufferedFrame) -> None:
        self.frames.append(bf)
        # Dropear el elemento *más viejo* cuando se pasa de capacidad.
        # Mantener un FIFO estricto en vez de "dropear el de menor
        # score" garantiza que un track parado en el gate por muchos
        # segundos igual reciba un candidato reciente, incluso si un
        # frame anterior de alto score si no pinearía el buffer.
        if len(self.frames) > self.max_size:
            del self.frames[0 : len(self.frames) - self.max_size]


class BestFrameManager:
    """Buffer rolling per-track + writer de JPG.

    Lifecycle:

      - ``observe(track_id, frame, detection)`` se llama cada frame
        del pipeline para cada track activo. Barato: solo scores +
        buffers.
      - ``commit(track_id, event_timestamp)`` se llama cuando un
        track emite un count event. Elige el mejor frame, lo
        escribe, devuelve el path. Devuelve ``None`` ante cualquier
        error así el path de publish no se rompe.
      - ``forget(track_id)`` descarta el buffer cuando un track
        termina sin count event (ahorra RAM en sesiones largas).

    Layout de filename:
    ``<output_dir>/<YYYY-MM-DD>/<track_id>_<ts>.jpg`` con ``ts``
    formateado como ``%Y%m%dT%H%M%S_%f`` para que sea sortable.
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
        # Injectable para tests; default a time.time.
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
        """Scorea el frame actual y pushea el CROP del bbox al buffer.

        El scoring corre sobre el frame completo (centrality/area lo
        necesitan), pero al buffer va solo el crop con margen — copia
        independiente, así mutaciones in-place subsiguientes del pipeline
        (anotaciones, conversiones de color) no envenenan el candidato.
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
            crop = _crop_with_margin(frame, bbox)
            if crop is None:
                return
            bf = BufferedFrame(
                frame_image=crop,
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
        """Elige el mejor, escribe el JPG, dropea el buffer. Devuelve el path o None.

        Los errores (no buffer / write failure) colapsan a ``None``
        y un log de warning así el path de publish puede incluir /
        excluir el field ``best_frame_path`` limpiamente.
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
        """Dropea el buffer para un track que terminó sin evento."""
        self._buffers.pop(track_id, None)

    def gc(self, alive_track_ids: set[int]) -> int:
        """Dropea buffers de tracks que ya no están en el dict live del tracker.

        El pipeline no siempre llama a ``forget`` (ej. los tracks
        PENDING que timeoutean se garbage-collectean por el tracker,
        no se señalizan explícitamente acá). Un gc periódico bounda
        la RAM en sesiones largas. Devuelve la cantidad de buffers
        dropeados.
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
        # El dir por fecha está en hora *local* así los operadores
        # pueden listar los frames de un día rápido. La purga de
        # retención usa mtime que también es local-tz friendly.
        dt = datetime.fromtimestamp(event_timestamp)
        day_dir = self._output_dir / dt.strftime("%Y-%m-%d")
        if not _safe_makedirs(day_dir):
            return None
        # Timestamp con microsegundos para filenames libres de
        # colisión incluso cuando múltiples eventos disparan sobre
        # el mismo track en un único frame (extremadamente raro,
        # pero barato de defender).
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

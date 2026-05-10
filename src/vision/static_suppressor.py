"""Suppression de detecciones que aparecen consistentemente sobre clutter
estático del ambiente.

Patrón de FP que cubre: el detector dispara con confidence aceptable sobre
una zona específica del frame (sombra estable, blob oscuro estructural,
maniquí, ropa colgada) en la mayoría de los frames, generando tracks
fantasma que pueden colarse al counter pese a NMS, cluster por centroide
y containment filter.

Algoritmo: cuadriculado del frame en celdas de tamaño fijo. Para cada
detección, su centroide cae en una celda. Mantiene buffer rolling de qué
celdas tuvieron al menos una detección en cada frame de los últimos N
segundos. Las celdas con hit rate >= threshold se marcan "hot" y las
detecciones cuyos centroides caen en ellas se descartan.

Una persona que se queda parada cubre la celda por una fracción de la
ventana (típicamente <50% si pasa por el FOV en una visita normal); un
FP estructural está activo casi 100% del tiempo. El threshold default 0.7
separa esos casos sin sacrificar detección legítima de personas
ocasionalmente quietas.

Diseñado como defense-in-depth tras los filtros del detector. Inactivo
durante el "warm-up" inicial (buffer no lleno) — no filtra hasta tener
data suficiente para distinguir.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Iterable, Protocol


class _HasCentroid(Protocol):
    """Cualquier objeto que expone ``centroid`` como (x, y)."""

    @property
    def centroid(self) -> tuple[float, float]:
        ...


class StaticSuppressor:
    """Filtro rolling que suprime detecciones sobre celdas hot.

    Args:
        cell_size_px: Tamaño del cuadriculado en pixels. Define
            resolución espacial del filtro — más chico captura FPs muy
            localizados, más grande captura patrones esparcidos. 30 es
            razonable para frames 1152×648 con cabezas de ~50-80px de
            ancho (cada cabeza ocupa ~2-3 celdas).
        window_seconds: Ventana del análisis temporal. 3s es buen
            sweet spot — suficiente para que un FP estable se acumule,
            corto para que una persona parada por reasons legítimos
            (atención al mostrador) no quede suprimida.
        hit_rate_threshold: Fracción mínima de la ventana en que la
            celda debe estar activa para suprimirse. 0.7 distingue
            FP estructural (≈1.0) de presencia humana ocasional (~0.3-0.5).
        approx_fps: Estimación de cuántos frames procesa el pipeline
            por segundo, para convertir ``window_seconds`` a frames.
            No tiene que ser exacto — usar un conservative estimate
            (ej. 17 si el pipeline anda entre 16-26fps) es bien.
    """

    def __init__(
        self,
        cell_size_px: int = 30,
        window_seconds: float = 3.0,
        hit_rate_threshold: float = 0.7,
        approx_fps: int = 17,
    ) -> None:
        if cell_size_px <= 0:
            raise ValueError(f"cell_size_px must be > 0, got {cell_size_px}")
        if not 0.0 < hit_rate_threshold <= 1.0:
            raise ValueError(
                f"hit_rate_threshold must be in (0, 1], got {hit_rate_threshold}"
            )
        self.cell_size_px = int(cell_size_px)
        self.hit_rate_threshold = float(hit_rate_threshold)
        self._window_frames = max(1, int(window_seconds * approx_fps))
        self._buffer: deque[set[tuple[int, int]]] = deque(
            maxlen=self._window_frames
        )

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (int(x) // self.cell_size_px, int(y) // self.cell_size_px)

    def update_and_filter(
        self, detections: Iterable[_HasCentroid]
    ) -> list[_HasCentroid]:
        """Registra el frame actual y devuelve detections sin las que
        caen en celdas hot.

        Hasta que el buffer esté lleno (warm-up), nunca filtra — sin
        suficiente historia no se puede distinguir FP estable de
        presencia legítima.
        """
        det_list = list(detections)
        # Snapshot de celdas activas este frame (set para deduplicar
        # múltiples detecciones que caen en la misma celda).
        current_cells = {
            self._cell_of(d.centroid[0], d.centroid[1]) for d in det_list
        }
        self._buffer.append(current_cells)

        # Warm-up: hasta tener data suficiente, no filtramos.
        if len(self._buffer) < self._window_frames:
            return det_list

        # Acumular hits por celda a lo largo de la ventana.
        cell_hits: Counter[tuple[int, int]] = Counter()
        for frame_cells in self._buffer:
            cell_hits.update(frame_cells)

        threshold_count = int(self.hit_rate_threshold * len(self._buffer))
        hot_cells = {c for c, n in cell_hits.items() if n >= threshold_count}

        if not hot_cells:
            return det_list

        return [
            d
            for d in det_list
            if self._cell_of(d.centroid[0], d.centroid[1]) not in hot_cells
        ]

    @property
    def hot_cells(self) -> set[tuple[int, int]]:
        """Celdas hot actuales — útil para debugging / observabilidad.

        Se computa on-demand desde el buffer; no cachea para mantener
        el invariante: hot_cells siempre refleja el estado consistente
        con la última llamada a update_and_filter().
        """
        if len(self._buffer) < self._window_frames:
            return set()
        cell_hits: Counter[tuple[int, int]] = Counter()
        for frame_cells in self._buffer:
            cell_hits.update(frame_cells)
        threshold_count = int(self.hit_rate_threshold * len(self._buffer))
        return {c for c, n in cell_hits.items() if n >= threshold_count}

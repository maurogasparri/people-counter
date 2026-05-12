"""Suppression de detecciones que aparecen consistentemente sobre clutter
estático del ambiente.

Patrón de FP que cubre: el detector dispara con confidence aceptable sobre
una zona específica del frame (sombra estable, blob oscuro estructural,
maniquí, ropa colgada) en la mayoría de los frames, generando tracks
fantasma que pueden colarse al counter pese a NMS, cluster por centroide
y containment filter.

Algoritmo: cuadriculado del frame en celdas de tamaño fijo. Para cada
detección, su centroide cae en una celda. Mantiene buffer rolling de qué
celdas tuvieron al menos una detección en cada frame de los últimos
``window_seconds`` reales (no frames). Las celdas con hit rate >=
``hit_rate_threshold`` se marcan "hot" y las detecciones cuyos centroides
caen en ellas se descartan.

La ventana se mide en tiempo real (timestamps) y no depende del FPS
del pipeline — el FPS efectivo cambia con la carga, y un buffer fixed-
length lo traduciría a una ventana temporal que oscila con la
performance. Acá la semántica del operador ("3 segundos de historia")
se respeta exactamente sea cual sea el FPS instantáneo.

Una persona que se queda parada cubre la celda por una fracción de la
ventana (típicamente <50% si pasa por el FOV en una visita normal); un
FP estructural está activo casi 100% del tiempo. El threshold default 0.7
separa esos casos sin sacrificar detección legítima de personas
ocasionalmente quietas.

Diseñado como defense-in-depth tras los filtros del detector. Inactivo
durante el "warm-up" inicial — no filtra hasta que el buffer cubre la
ventana completa (al menos ``window_seconds`` reales de historia).
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Callable, Iterable, Protocol


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
        window_seconds: Ventana del análisis temporal en segundos reales.
            3s es buen sweet spot — suficiente para que un FP estable se
            acumule, corto para que una persona parada por reasons
            legítimos (atención al mostrador) no quede suprimida.
        hit_rate_threshold: Fracción mínima de la ventana en que la
            celda debe estar activa para suprimirse. 0.7 distingue
            FP estructural (≈1.0) de presencia humana ocasional (~0.3-0.5).
        min_samples_for_judgment: Mínimo de muestras requeridas antes de
            empezar a filtrar, ortogonal al gate de tiempo. Evita que un
            buffer chico pero con ventana ya cubierta filtre con
            estadística pobre (ej. 3 samples en 3s a 1 fps).
        time_fn: Callable que devuelve el tiempo en segundos (monotonic).
            Inyectable para tests determinísticos. Default
            ``time.monotonic`` para uso en runtime.
    """

    def __init__(
        self,
        cell_size_px: int = 30,
        window_seconds: float = 3.0,
        hit_rate_threshold: float = 0.7,
        min_samples_for_judgment: int = 8,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if cell_size_px <= 0:
            raise ValueError(f"cell_size_px must be > 0, got {cell_size_px}")
        if not 0.0 < hit_rate_threshold <= 1.0:
            raise ValueError(
                f"hit_rate_threshold must be in (0, 1], got {hit_rate_threshold}"
            )
        if window_seconds < 0:
            raise ValueError(
                f"window_seconds must be >= 0, got {window_seconds}"
            )
        self.cell_size_px = int(cell_size_px)
        self.window_seconds = float(window_seconds)
        self.hit_rate_threshold = float(hit_rate_threshold)
        self.min_samples_for_judgment = max(1, int(min_samples_for_judgment))
        self._time_fn = time_fn
        # Cada entry es (timestamp_seconds, set_of_active_cells).
        self._buffer: deque[tuple[float, set[tuple[int, int]]]] = deque()

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (int(x) // self.cell_size_px, int(y) // self.cell_size_px)

    def _evict_expired(self, now: float) -> None:
        """Saca del buffer todas las entries más viejas que la ventana."""
        while self._buffer and now - self._buffer[0][0] > self.window_seconds:
            self._buffer.popleft()

    def _is_warm(self, now: float) -> bool:
        """True si tenemos suficiente historia para juzgar.

        Necesitamos dos cosas:
        1. Mínimo de samples (anti-estadística-pobre).
        2. La muestra más vieja en el buffer cubre la ventana — i.e. el
           buffer abarca al menos ``window_seconds`` de tiempo real. Si
           el buffer todavía no llegó a llenar la ventana, cualquier
           hit rate es transitorio.
        """
        if len(self._buffer) < self.min_samples_for_judgment:
            return False
        if self.window_seconds <= 0:
            return True
        return now - self._buffer[0][0] >= self.window_seconds * 0.95

    def update_and_filter(
        self, detections: Iterable[_HasCentroid]
    ) -> list[_HasCentroid]:
        """Registra el frame actual y devuelve detections sin las que
        caen en celdas hot.

        Hasta que el buffer cubra la ventana temporal completa (warm-up),
        nunca filtra — sin historia suficiente no se puede distinguir
        FP estable de presencia legítima.
        """
        now = self._time_fn()
        self._evict_expired(now)

        det_list = list(detections)
        # Snapshot de celdas activas este frame (set para deduplicar
        # múltiples detecciones que caen en la misma celda).
        current_cells = {
            self._cell_of(d.centroid[0], d.centroid[1]) for d in det_list
        }
        self._buffer.append((now, current_cells))

        if not self._is_warm(now):
            return det_list

        # Acumular hits por celda a lo largo de la ventana.
        cell_hits: Counter[tuple[int, int]] = Counter()
        for _ts, frame_cells in self._buffer:
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

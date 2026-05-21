"""Tests para StaticSuppressor."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.vision.static_suppressor import StaticSuppressor


@dataclass
class _FakeDet:
    """Mínimo stand-in para Detection — solo expone centroid."""

    cx: float
    cy: float

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.cx, self.cy)


class _FakeClock:
    """Reloj monotonic determinístico para tests.

    Cada llamada al callable devuelve el tiempo actual; ``tick(dt)``
    avanza el tiempo. Inyectado vía el kwarg ``time_fn`` del supresor
    para que los tests no dependan del tiempo real ni del FPS observado.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def tick(self, dt: float) -> None:
        self.now += float(dt)


def _suppressor(
    clock: _FakeClock | None = None, **kw
) -> tuple[StaticSuppressor, _FakeClock]:
    """Factory con defaults convenientes para tests deterministas.

    Por default usa window_seconds=1.0 y un FPS sintético de 10 (dt=0.1
    por tick) — así 10 ticks llenan la ventana exacta de 1s.
    """
    if clock is None:
        clock = _FakeClock()
    sup = StaticSuppressor(
        cell_size_px=kw.pop("cell_size_px", 30),
        window_seconds=kw.pop("window_seconds", 1.0),
        hit_rate_threshold=kw.pop("hit_rate_threshold", 0.7),
        min_samples_for_judgment=kw.pop("min_samples_for_judgment", 8),
        time_fn=clock,
    )
    return sup, clock


def _tick_and_update(
    sup: StaticSuppressor,
    clock: _FakeClock,
    dets: list[_FakeDet],
    dt: float = 0.1,
) -> list:
    """Avanza el reloj `dt` segundos y llama update_and_filter."""
    clock.tick(dt)
    return sup.update_and_filter(dets)


class TestWarmup:
    def test_no_filter_during_warmup_low_samples(self) -> None:
        """Sin samples mínimos, no filtra aunque haya pasado tiempo."""
        sup, clock = _suppressor(min_samples_for_judgment=8)
        det = _FakeDet(100, 100)
        for _ in range(5):
            out = _tick_and_update(sup, clock, [det])
            assert out == [det], "Durante warm-up (samples<min) no debe filtrar"

    def test_no_filter_during_warmup_short_window(self) -> None:
        """Con samples mínimos pero antes de cubrir window_seconds, no filtra."""
        sup, clock = _suppressor(window_seconds=1.0, min_samples_for_judgment=2)
        det = _FakeDet(100, 100)
        # 5 ticks de 0.1s = buffer cubre 0.4s (newest-oldest), por debajo
        # de window_seconds*0.95 = 0.95s.
        for _ in range(5):
            out = _tick_and_update(sup, clock, [det], dt=0.1)
            assert out == [det], "Antes de cubrir la ventana no debe filtrar"


class TestHotCellSuppression:
    def test_persistent_detection_is_suppressed(self) -> None:
        """Detección que aparece en TODOS los frames de la ventana →
        100% hit rate, supera el threshold default 0.7 y se suprime."""
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        # 11 ticks de 0.1s = buffer cubre 1.0s (warm) con celda 100% hot.
        for _ in range(11):
            _tick_and_update(sup, clock, [det])
        out = _tick_and_update(sup, clock, [det])
        assert out == [], "FP estable debe suprimirse"

    def test_intermittent_detection_passes(self) -> None:
        """Detección que aparece en <=50% de los frames es persona
        ocasional, no FP — pasa el filtro."""
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        # Llenar la ventana con 50% de presencia.
        for i in range(11):
            _tick_and_update(sup, clock, [det] if i % 2 == 0 else [])
        out = _tick_and_update(sup, clock, [det])
        assert out == [det], "Hit rate 0.5 (<0.7) no debe suprimirse"


class TestExemptRoi:
    """Detecciones dentro de ``exempt_roi`` no se suprimen aunque su celda
    esté hot — exenta el ROI de conteo (gente parada en la puerta no debe
    morir y disparar counts fantasma)."""

    def _warm_hot_cell(self, sup, clock, det) -> None:
        for _ in range(11):
            _tick_and_update(sup, clock, [det])

    def test_hot_cell_inside_exempt_roi_is_kept(self) -> None:
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        self._warm_hot_cell(sup, clock, det)
        clock.tick(0.1)
        out = sup.update_and_filter([det], exempt_roi=(50, 200, 50, 200))
        assert out == [det], "celda hot dentro del ROI exento se conserva"

    def test_hot_cell_outside_exempt_roi_still_suppressed(self) -> None:
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        self._warm_hot_cell(sup, clock, det)
        clock.tick(0.1)
        out = sup.update_and_filter([det], exempt_roi=(400, 600, 400, 600))
        assert out == [], "celda hot fuera del ROI exento se suprime igual"

    def test_exempt_roi_none_is_prior_behavior(self) -> None:
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        self._warm_hot_cell(sup, clock, det)
        clock.tick(0.1)
        out = sup.update_and_filter([det])  # default exempt_roi=None
        assert out == [], "sin exempt_roi, la celda hot se suprime (comportamiento previo)"


class TestNonHotCellsPassFreely:
    def test_isolated_detection_in_cold_cell_passes(self) -> None:
        """Una detección en una celda nueva (sin historia) pasa
        siempre, aunque la ventana esté llena por otras celdas."""
        sup, clock = _suppressor()
        hot_det = _FakeDet(100, 100)
        for _ in range(11):
            _tick_and_update(sup, clock, [hot_det])

        new_det = _FakeDet(500, 500)  # celda completamente distinta
        out = _tick_and_update(sup, clock, [hot_det, new_det])
        assert hot_det not in out, "celda hot suprime"
        assert new_det in out, "celda fría pasa libre"

    def test_neighboring_cells_independent(self) -> None:
        """Detecciones en celdas vecinas no se contaminan entre sí."""
        sup, clock = _suppressor(cell_size_px=30)
        # (0,0) y (60,60) caen en celdas no adyacentes (0,0) y (2,2)
        for _ in range(11):
            _tick_and_update(sup, clock, [_FakeDet(0, 0)])
        new_det = _FakeDet(60, 60)
        out = _tick_and_update(sup, clock, [new_det])
        assert new_det in out


class TestMultiCellPerFrame:
    def test_two_persistent_cells_both_suppressed(self) -> None:
        """Dos zonas separadas con FP estable → ambas se suprimen."""
        sup, clock = _suppressor()
        a = _FakeDet(100, 100)
        b = _FakeDet(500, 500)
        for _ in range(11):
            _tick_and_update(sup, clock, [a, b])
        out = _tick_and_update(sup, clock, [a, b])
        assert out == [], "ambos FP estables deben suprimirse"

    def test_multiple_dets_same_cell_count_once(self) -> None:
        """Múltiples detecciones en la misma celda en un frame cuentan
        como UNA presencia (no inflan el hit rate artificialmente)."""
        sup, clock = _suppressor(hit_rate_threshold=0.9)
        a = _FakeDet(100, 100)
        b = _FakeDet(110, 110)  # misma celda
        c = _FakeDet(105, 95)   # misma celda
        # 50% de presencia con 3 dets por frame — si contara cada det
        # individualmente saltaría el 0.9; con set deduplica.
        for i in range(11):
            _tick_and_update(sup, clock, [a, b, c] if i % 2 == 0 else [])
        out = _tick_and_update(sup, clock, [a])
        assert out == [a], (
            "múltiples dets en la misma celda no deben inflar hit rate"
        )


class TestEmptyAndNone:
    def test_empty_detection_list_advances_buffer(self) -> None:
        """Frames sin detecciones avanzan el buffer (counted como
        no-hit) — necesario para que un FP que se calmó deje de ser
        hot eventualmente."""
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        for _ in range(11):
            _tick_and_update(sup, clock, [det])  # celda 100% hot
        # 11 ticks vacíos → la ventana ahora muestra cero presencia.
        for _ in range(11):
            _tick_and_update(sup, clock, [])
        out = _tick_and_update(sup, clock, [det])
        assert out == [det], "ventana rolling permite recovery"

    def test_filter_returns_list_not_iterable(self) -> None:
        """Confirmar que devolvemos list (no iterator) — caller hace
        len()/index sobre el resultado."""
        sup, clock = _suppressor()
        det = _FakeDet(100, 100)
        out = _tick_and_update(sup, clock, [det])
        assert isinstance(out, list)


class TestConfigValidation:
    def test_invalid_cell_size_raises(self) -> None:
        with pytest.raises(ValueError):
            StaticSuppressor(cell_size_px=0)
        with pytest.raises(ValueError):
            StaticSuppressor(cell_size_px=-5)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            StaticSuppressor(hit_rate_threshold=0.0)
        with pytest.raises(ValueError):
            StaticSuppressor(hit_rate_threshold=1.5)
        with pytest.raises(ValueError):
            StaticSuppressor(hit_rate_threshold=-0.1)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            StaticSuppressor(window_seconds=-1.0)

    def test_zero_window_does_not_crash(self) -> None:
        """window_seconds=0 es degenerate pero no debe crashear."""
        sup = StaticSuppressor(window_seconds=0.0)
        out = sup.update_and_filter([_FakeDet(0, 0)])
        assert isinstance(out, list)

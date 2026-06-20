"""Tests de la lógica pura del mapa de foco (scripts/focus_assist.py).

Cubren la acumulación max-por-zona + cobertura y el listado de zonas
faltantes. Sin hardware: solo arrays sintéticos. El loop + la UI se validan
on-hardware.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "focus_assist_script",
    _ROOT / "scripts" / "focus_assist.py",
)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def test_map_update_takes_max_where_valid():
    m = np.zeros((3, 3))
    cov = np.zeros((3, 3), dtype=bool)
    grid = np.array([[50.0, 100, 200], [0, 300, 0], [80, 0, 0]])
    fa.focus_map_update(m, cov, grid, np.ones((3, 3), dtype=bool), 60.0)
    np.testing.assert_allclose(m, grid)
    assert cov[0, 0] is np.False_ or cov[0, 0] == False  # 50 < 60  # noqa: E712
    assert cov[0, 1] == True  # noqa: E712
    assert cov[1, 1] == True  # noqa: E712
    assert cov[2, 0] == True  # noqa: E712


def test_map_update_keeps_max_across_calls():
    m = np.zeros((3, 3))
    cov = np.zeros((3, 3), dtype=bool)
    valid = np.ones((3, 3), dtype=bool)
    fa.focus_map_update(m, cov, np.full((3, 3), 100.0), valid, 60.0)
    fa.focus_map_update(m, cov, np.full((3, 3), 40.0), valid, 60.0)  # más bajo
    assert np.all(m == 100.0)  # se queda con el máximo


def test_map_update_skips_invalid_zone():
    m = np.zeros((3, 3))
    cov = np.zeros((3, 3), dtype=bool)
    grid = np.full((3, 3), 500.0)
    valid = np.ones((3, 3), dtype=bool)
    valid[0, 0] = False
    fa.focus_map_update(m, cov, grid, valid, 60.0)
    assert m[0, 0] == 0.0 and not cov[0, 0]
    assert m[1, 1] == 500.0 and cov[1, 1]


def test_map_missing_empty_when_all_covered():
    cl = np.ones((3, 3), dtype=bool)
    cr = np.ones((3, 3), dtype=bool)
    assert fa.focus_map_missing(cl, cr) == []


def test_map_missing_lists_uncovered_in_either_camera():
    cl = np.ones((3, 3), dtype=bool)
    cr = np.ones((3, 3), dtype=bool)
    cl[0, 0] = False  # falta en L
    cr[2, 2] = False  # falta en R
    miss = fa.focus_map_missing(cl, cr)
    assert "arriba-izq" in miss
    assert "abajo-der" in miss
    assert len(miss) == 2


def test_map_missing_uses_zone_names():
    cl = np.ones((3, 3), dtype=bool)
    cr = np.ones((3, 3), dtype=bool)
    cr[1, 1] = False
    assert fa.focus_map_missing(cl, cr) == ["centro"]

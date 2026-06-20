"""Tests de la lógica pura de scripts/diagnose_calibration.py.

Cubren matching de esquinas L↔R por ID, estadísticas epipolares y el
veredicto de salud de calibración. Sin hardware: solo arrays sintéticos.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "diagnose_calibration_script",
    _ROOT / "scripts" / "diagnose_calibration.py",
)
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)


def _corners(coords: list[tuple[float, float]]) -> np.ndarray:
    """(N,1,2) como devuelve detect_charuco_dual_pass."""
    return np.array(coords, dtype=np.float32).reshape(-1, 1, 2)


def _ids(values: list[int]) -> np.ndarray:
    """(N,1) como devuelve detect_charuco_dual_pass."""
    return np.array(values, dtype=np.int32).reshape(-1, 1)


# --------------------------------------------------------------------------
# match_corners
# --------------------------------------------------------------------------


def test_match_corners_shared_ids():
    cl = _corners([(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)])
    il = _ids([1, 2, 3])
    cr = _corners([(5.0, 20.0), (45.0, 60.0)])
    ir = _ids([1, 3])  # 2 no está en R
    pts_l, pts_r, ids = dc.match_corners(cl, il, cr, ir)
    assert ids == [1, 3]
    assert pts_l.shape == (2, 2)
    # Correspondencia correcta por ID, no por posición.
    np.testing.assert_allclose(pts_l[0], [10.0, 20.0])
    np.testing.assert_allclose(pts_r[0], [5.0, 20.0])
    np.testing.assert_allclose(pts_l[1], [50.0, 60.0])
    np.testing.assert_allclose(pts_r[1], [45.0, 60.0])


def test_match_corners_none_inputs():
    pts_l, pts_r, ids = dc.match_corners(None, None, None, None)
    assert pts_l.shape == (0, 2)
    assert pts_r.shape == (0, 2)
    assert ids == []


def test_match_corners_no_shared_ids():
    cl = _corners([(1.0, 1.0)])
    cr = _corners([(2.0, 2.0)])
    pts_l, pts_r, ids = dc.match_corners(cl, _ids([1]), cr, _ids([9]))
    assert ids == []
    assert pts_l.shape == (0, 2)


def test_match_corners_orders_by_id_independent_of_input_order():
    cl = _corners([(50.0, 60.0), (10.0, 20.0)])  # ids desordenados
    il = _ids([3, 1])
    cr = _corners([(45.0, 60.0), (5.0, 20.0)])
    ir = _ids([3, 1])
    pts_l, pts_r, ids = dc.match_corners(cl, il, cr, ir)
    assert ids == [1, 3]  # ordenado ascendente
    np.testing.assert_allclose(pts_l[0], [10.0, 20.0])


# --------------------------------------------------------------------------
# epipolar_stats
# --------------------------------------------------------------------------


def test_epipolar_stats_perfect_alignment():
    pts_l = np.array([[100.0, 50.0], [200.0, 80.0]])
    pts_r = np.array([[90.0, 50.0], [185.0, 80.0]])  # mismo y, x menor (disparidad)
    s = dc.epipolar_stats(pts_l, pts_r)
    assert s["n"] == 2
    assert s["rms_epipolar_px"] == pytest.approx(0.0)
    assert s["max_epipolar_px"] == pytest.approx(0.0)
    # disparidad dx = [10, 15] → mediana 12.5
    assert s["median_disparity_px"] == pytest.approx(12.5)


def test_epipolar_stats_known_residuals():
    pts_l = np.array([[100.0, 50.0], [200.0, 83.0]])
    pts_r = np.array([[90.0, 47.0], [185.0, 79.0]])  # dy = [3, 4]
    s = dc.epipolar_stats(pts_l, pts_r)
    # rms de [3,4] = sqrt((9+16)/2) = sqrt(12.5)
    assert s["rms_epipolar_px"] == pytest.approx(np.sqrt(12.5))
    assert s["max_epipolar_px"] == pytest.approx(4.0)


def test_epipolar_stats_empty():
    s = dc.epipolar_stats(np.empty((0, 2)), np.empty((0, 2)))
    assert s["n"] == 0
    assert np.isnan(s["rms_epipolar_px"])
    assert np.isnan(s["median_disparity_px"])


# --------------------------------------------------------------------------
# calibration_verdict
# --------------------------------------------------------------------------


def test_verdict_ok_below_threshold():
    v, detail = dc.calibration_verdict(
        0.4, n_matched=40, threshold_px=1.0, min_matched=20
    )
    assert v == "OK"
    assert "válida" in detail


def test_verdict_recalibrate_above_threshold():
    v, detail = dc.calibration_verdict(
        2.5, n_matched=40, threshold_px=1.0, min_matched=20
    )
    assert v == "RECALIBRAR"
    assert "cambió" in detail


def test_verdict_inconclusive_few_corners():
    v, detail = dc.calibration_verdict(
        0.3, n_matched=5, threshold_px=1.0, min_matched=20
    )
    assert v == "INCONCLUSO"
    assert "matcheadas" in detail


def test_verdict_inconclusive_nan_rms():
    v, _ = dc.calibration_verdict(
        float("nan"), n_matched=99, threshold_px=1.0, min_matched=20
    )
    assert v == "INCONCLUSO"


def test_verdict_exactly_at_threshold_is_ok():
    v, _ = dc.calibration_verdict(1.0, n_matched=30, threshold_px=1.0, min_matched=20)
    assert v == "OK"

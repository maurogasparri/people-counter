"""Tests de la lógica pura del modo barrido libre (--sweep) de calibrate.py.

Cubren la firma de pose, el gate de novedad y el resumen de cobertura — sin
hardware. El loop de captura + la UI se validan on-hardware.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "calibrate_sweep_script",
    _ROOT / "scripts" / "calibrate.py",
)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def _sig(cx, cy, size=0.25, tilt=0.0):
    return {"cx": cx, "cy": cy, "size": size, "tilt": tilt}


# --------------------------------------------------------------------------
# _sweep_signature
# --------------------------------------------------------------------------


def test_signature_centroid_and_size():
    # 4 esquinas formando un bbox de 200x200 centrado en (500,400) de un
    # frame 1000x1000.
    corners = np.array(
        [[400, 300], [600, 300], [600, 500], [400, 500]], dtype=np.float32
    ).reshape(-1, 1, 2)
    s = cal._sweep_signature(corners, (1000, 1000, 3))
    assert s["cx"] == 0.5
    assert s["cy"] == 0.4
    # bbox 200x200 / (1000*1000) = 0.04
    assert abs(s["size"] - 0.04) < 1e-9
    assert s["tilt"] == 0.0


def test_signature_tilt_from_rvec():
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32).reshape(
        -1, 1, 2
    )
    s = cal._sweep_signature(corners, (100, 100), rvec=np.array([0.3, 0.0, 0.0]))
    # degrees(0.3 rad) ≈ 17.19
    assert abs(s["tilt"] - 17.19) < 0.1


# --------------------------------------------------------------------------
# _sweep_novelty_distance
# --------------------------------------------------------------------------


def test_novelty_inf_when_empty():
    assert cal._sweep_novelty_distance(_sig(0.5, 0.5), []) == float("inf")


def test_novelty_zero_for_identical():
    a = _sig(0.5, 0.5)
    assert cal._sweep_novelty_distance(_sig(0.5, 0.5), [a]) == 0.0


def test_novelty_large_for_far_position():
    accepted = [_sig(0.1, 0.1)]
    d = cal._sweep_novelty_distance(_sig(0.9, 0.9), accepted)
    assert d > 0.5


def test_novelty_picks_minimum_distance():
    accepted = [_sig(0.1, 0.1), _sig(0.52, 0.5)]
    # el más cercano a (0.5,0.5) es (0.52,0.5) → distancia ~0.02
    d = cal._sweep_novelty_distance(_sig(0.5, 0.5), accepted)
    assert d < 0.05


# --------------------------------------------------------------------------
# _sweep_coverage
# --------------------------------------------------------------------------


def test_coverage_incomplete_when_sparse():
    accepted = [_sig(0.5, 0.5) for _ in range(3)]
    cov = cal._sweep_coverage(accepted)
    assert cov["complete"] is False
    assert cov["n"] == 3
    assert cov["cells_covered"] == 1  # todas en la celda central
    assert cov["missing"]  # hay hints de lo que falta


def test_coverage_complete_when_full():
    centers = [0.17, 0.5, 0.83]  # centros de las 3 celdas por eje
    accepted = []
    i = 0
    for cy in centers:
        for cx in centers:
            # 2 capturas por celda = 18 total; alterná size (2 bandas) y
            # poné tilt alto en las primeras de cada eje.
            accepted.append(_sig(cx, cy, size=0.04, tilt=20.0 if i < 5 else 0.0))
            accepted.append(_sig(cx, cy, size=0.36, tilt=0.0))
            i += 1
    cov = cal._sweep_coverage(accepted)
    assert cov["cells_covered"] == 9
    assert cov["bands_covered"] >= 2
    assert cov["n_tilted"] >= cal.SWEEP_MIN_TILTED
    assert cov["n"] >= cal.SWEEP_TARGET_CAPTURES
    assert cov["complete"] is True
    assert cov["missing"] == []


def test_coverage_html_renders():
    cov = cal._sweep_coverage([_sig(0.5, 0.5)])
    html = cal._sweep_coverage_html(cov)
    assert "Capturas: 1" in html
    assert "falta" in html.lower()


# --------------------------------------------------------------------------
# _sweep_is_still
# --------------------------------------------------------------------------


def test_is_still_false_when_too_few_frames():
    assert cal._sweep_is_still([(0.0, 0.0), (0.1, 0.1)], 5.0, 3) is False


def test_is_still_true_when_low_motion():
    pts = [(100.0, 100.0), (101.0, 100.5), (100.5, 101.0)]
    assert cal._sweep_is_still(pts, 5.0, 3) is True


def test_is_still_false_when_moving():
    pts = [(100.0, 100.0), (120.0, 100.0), (140.0, 100.0)]  # saltos de 20px
    assert cal._sweep_is_still(pts, 5.0, 3) is False


def test_is_still_uses_only_last_min_frames():
    # primer salto grande, pero los últimos 3 quietos → True
    pts = [(0.0, 0.0), (500.0, 500.0), (100.0, 100.0), (101.0, 100.0), (100.5, 100.5)]
    assert cal._sweep_is_still(pts, 5.0, 3) is True


# --------------------------------------------------------------------------
# _sweep_html — handler de prompt post-captura (regresión: el sweep no
# renderizaba el input de ground-truth porque su refresh() era recortado)
# --------------------------------------------------------------------------


def test_sweep_html_renders_ground_truth_prompt():
    html = cal._sweep_html()
    # Parsea data-phase y data-prompt-type (sin esto no detecta el prompt).
    assert 'data-phase="' in html
    assert "data-prompt-type" in html
    # Inyecta los controles del prompt y el input de ground-truth + botones.
    assert "prompt-ctrls" in html
    assert "ground_truth" in html
    assert "gt-dist" in html
    assert "submitGt" in html
    # Postea la respuesta del operador de vuelta al backend.
    assert "/wizard-input" in html


def test_sweep_html_handles_diversity_and_complete():
    html = cal._sweep_html()
    # Prompt de diversidad (Continuar / Cancelar) y auto-open del reporte.
    assert "diversity" in html
    assert "sendInput" in html
    assert "data-has-report" in html

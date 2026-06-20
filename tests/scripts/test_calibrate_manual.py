"""Tests del modo manual del wizard de calibración (scripts/calibrate.py).

Cubren la parte testeable sin hardware: el render condicional del botón
"Capturar" en la UI guiada y el estado mutable. El gating del loop de
captura (auto vs manual) se valida on-hardware.
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "calibrate_manual_script",
    _ROOT / "scripts" / "calibrate.py",
)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def test_guided_state_has_capture_requested():
    st = cal._GuidedState()
    assert st.capture_requested is False


def test_guided_html_shows_capture_button_when_manual():
    cal._guided_manual_enabled = True
    try:
        html = cal._guided_html()
    finally:
        cal._guided_manual_enabled = False
    assert "post('/capture')" in html  # el botón está cableado
    assert ">Capturar</button>" in html
    assert "__CAPTURE_BTN__" not in html  # placeholder reemplazado


def test_guided_html_hides_capture_button_when_auto():
    cal._guided_manual_enabled = False
    html = cal._guided_html()
    assert "post('/capture')" not in html
    assert "__CAPTURE_BTN__" not in html  # placeholder reemplazado a vacío

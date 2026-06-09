"""Tests para el generador del reporte HTML de calibración."""

import datetime
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.vision.report import (
    BASELINE_WARN_TOL_MM,
    DESIGN_BASELINE_MM,
    generate_html_report,
    save_report,
)


def _mock_calibration(baseline_mm: float = 140.0) -> dict:
    return {
        "camera_matrix_l": np.array([[1330.0, 0, 2304], [0, 1330.0, 1296], [0, 0, 1]]),
        "camera_matrix_r": np.array([[1332.0, 0, 2300], [0, 1332.0, 1290], [0, 0, 1]]),
        "T": np.array([[baseline_mm], [0.0], [0.0]]),
    }


class TestGenerateReport:
    def test_minimal_report_contains_device_id(self):
        html = generate_html_report(_mock_calibration(), device_id="DEV-001")
        assert "DEV-001" in html

    def test_reports_baseline_pass_at_design(self):
        html = generate_html_report(_mock_calibration(140.0), device_id="D")
        assert "tolerancia" in html
        assert "PASS" in html

    def test_reports_baseline_warning_when_off_design(self):
        off = DESIGN_BASELINE_MM + BASELINE_WARN_TOL_MM + 3
        html = generate_html_report(_mock_calibration(off), device_id="D")
        # Alert text mentions the baseline deviation (exact wording short).
        assert "baseline" in html.lower() and "difiere" in html.lower()
        assert 'class="alert"' in html

    def test_report_includes_depth_zones(self):
        zones = {
            "center": (2000.0, 10.0, 0.5, 95.0),
            "top-left": (2050.0, 20.0, 2.5, 80.0),
            "_pass": True,
            "_edge_ratio": 1.3,
            "_center_err": 0.5,
            "_distance_mm": 2000,
        }
        html = generate_html_report(
            _mock_calibration(), diagnose_zones=zones, device_id="D"
        )
        assert "center" in html
        assert "top-left" in html
        assert "Validación de profundidad" in html

    def test_report_embeds_capture_thumbnails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            lp = tmpdir / "left_000.png"
            rp = tmpdir / "right_000.png"
            img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(lp), img)
            cv2.imwrite(str(rp), img)
            html = generate_html_report(
                _mock_calibration(),
                capture_pairs=[(lp, rp, "A1", 40)],
                device_id="D",
            )
            assert "A1" in html
            assert "40 esquinas" in html
            assert "data:image/jpeg;base64," in html

    def test_html_is_well_formed(self):
        html = generate_html_report(_mock_calibration(), device_id="D")
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_save_report_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "report.html"
            html = generate_html_report(_mock_calibration(), device_id="D")
            result = save_report(html, out)
            assert result.exists()
            assert result.read_text(encoding="utf-8") == html

    def test_timestamp_used_in_header(self):
        ts = datetime.datetime(2025, 6, 15, 14, 30)
        html = generate_html_report(_mock_calibration(), device_id="D", timestamp=ts)
        assert "2025-06-15" in html

    def test_rms_shown_when_provided(self):
        html = generate_html_report(_mock_calibration(), device_id="D", rms_stereo=0.82)
        assert "0.820 px" in html or "0.82" in html

    def test_per_pair_residual_section_rendered(self):
        per_pair = [
            {"pair_idx": 0, "rms_l": 0.5, "rms_r": 0.6, "rms": 0.55, "n_corners": 40},
            {"pair_idx": 1, "rms_l": 0.7, "rms_r": 0.8, "rms": 0.75, "n_corners": 40},
            {"pair_idx": 2, "rms_l": 3.5, "rms_r": 3.6, "rms": 3.55, "n_corners": 40},
            {"pair_idx": 3, "rms_l": 0.6, "rms_r": 0.5, "rms": 0.55, "n_corners": 40},
        ]
        html = generate_html_report(
            _mock_calibration(),
            device_id="D",
            per_pair_residuals=per_pair,
        )
        assert "Residual por par" in html
        # Pair 2 should be flagged as outlier
        assert "OUTLIER" in html or "outlier" in html

    def test_residual_section_absent_when_no_data(self):
        html = generate_html_report(_mock_calibration(), device_id="D")
        assert "Residual por par" not in html

    def test_residual_nan_values_tolerated(self):
        per_pair = [
            {"pair_idx": 0, "rms_l": 0.5, "rms_r": 0.6, "rms": 0.55, "n_corners": 40},
            {
                "pair_idx": 1,
                "rms_l": float("nan"),
                "rms_r": float("nan"),
                "rms": float("nan"),
                "n_corners": 0,
            },
        ]
        html = generate_html_report(
            _mock_calibration(),
            device_id="D",
            per_pair_residuals=per_pair,
        )
        assert "Residual por par" in html
        # Should render without crashing and include the em-dash placeholder
        assert "—" in html

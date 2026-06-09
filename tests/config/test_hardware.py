"""Tests para src/config/hardware.py — HardwareParams + load_hardware_params."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.hardware import (
    FLEET_DEFAULTS,
    load_hardware_params,
)


class TestFleetDefaults:
    def test_fleet_defaults_match_canonical_shipping_setup(self) -> None:
        """Snapshot del setup que shippeamos: RPi5 + IMX708 B0310 + bracket
        140mm + ChArUco A3 canónico."""
        assert FLEET_DEFAULTS.full_res == (4608, 2592)
        assert FLEET_DEFAULTS.default_res == (2304, 1296)
        assert FLEET_DEFAULTS.nominal_focal_full_px == 2050.0
        assert FLEET_DEFAULTS.hfov_deg == 120.0
        assert FLEET_DEFAULTS.baseline_mm == 140.0
        assert FLEET_DEFAULTS.camera_left_csi == 0
        assert FLEET_DEFAULTS.camera_right_csi == 1
        assert FLEET_DEFAULTS.board_cols == 9
        assert FLEET_DEFAULTS.board_rows == 6
        assert FLEET_DEFAULTS.square_mm == 45.0
        assert FLEET_DEFAULTS.marker_mm == 33.0
        assert FLEET_DEFAULTS.aruco_dict == "DICT_4X4_100"
        assert FLEET_DEFAULTS.legacy_pattern is True
        assert FLEET_DEFAULTS.dual_pass_accept_threshold == 8
        assert FLEET_DEFAULTS.ae_initial_settle_seconds == 2.0
        assert FLEET_DEFAULTS.ae_resettle_seconds == 1.5

    def test_hardware_params_is_frozen(self) -> None:
        """HardwareParams es un dataclass inmutable — los setup tools lo
        comparten como singleton; mutarlo sería un bug."""
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            FLEET_DEFAULTS.full_res = (1920, 1080)  # type: ignore[misc]


class TestLoadHardwareParamsMissing:
    def test_returns_fleet_defaults_when_path_does_not_exist(
        self,
        tmp_path: Path,
    ) -> None:
        """Sin archivo de config, devuelve FLEET_DEFAULTS sin raisear.

        Pensado para tools que pueden correr antes del aprovisioning
        (factory QC, dev workstation sin /etc/people-counter/config.yaml).
        """
        missing = tmp_path / "no-such-file.yaml"
        hw = load_hardware_params(missing)
        assert hw == FLEET_DEFAULTS

    def test_returns_fleet_defaults_on_malformed_yaml(
        self,
        tmp_path: Path,
    ) -> None:
        """Si el YAML está corrupto, no rompe el tool — fallback a fleet
        defaults para que la tool pueda al menos arrancar y reportar."""
        broken = tmp_path / "broken.yaml"
        broken.write_text("not: valid: yaml: [unclosed", encoding="utf-8")
        hw = load_hardware_params(broken)
        assert hw == FLEET_DEFAULTS

    def test_empty_file_returns_fleet_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        hw = load_hardware_params(empty)
        assert hw == FLEET_DEFAULTS


class TestLoadHardwareParamsOverrides:
    def _write(self, path: Path, data: dict) -> None:
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def test_overrides_sensor_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        # Sensor distinto (ej. IMX477) — full_res + focal cambian.
        self._write(
            cfg,
            {
                "sensor": {
                    "full_res": [4056, 3040],
                    "default_res": [2028, 1520],
                    "nominal_focal_full_px": 2540.0,
                },
            },
        )
        hw = load_hardware_params(cfg)
        assert hw.full_res == (4056, 3040)
        assert hw.default_res == (2028, 1520)
        assert hw.nominal_focal_full_px == 2540.0
        # Las keys no overrideadas caen al default.
        assert hw.baseline_mm == FLEET_DEFAULTS.baseline_mm

    def test_overrides_bracket_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        # Bracket custom con baseline distinto + CSI swap.
        self._write(
            cfg,
            {
                "bracket": {
                    "baseline_mm": 200.0,
                    "camera_left_csi": 1,
                    "camera_right_csi": 0,
                },
            },
        )
        hw = load_hardware_params(cfg)
        assert hw.baseline_mm == 200.0
        assert hw.camera_left_csi == 1
        assert hw.camera_right_csi == 0

    def test_overrides_lens_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        self._write(
            cfg,
            {
                "lens": {"type": "m12_180deg", "hfov_deg": 180.0},
            },
        )
        hw = load_hardware_params(cfg)
        assert hw.hfov_deg == 180.0
        assert hw.lens_type == "m12_180deg"

    def test_overrides_charuco_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        # Board distinto (más chico, otro dict).
        self._write(
            cfg,
            {
                "vision": {
                    "charuco": {
                        "board_cols": 7,
                        "board_rows": 5,
                        "square_mm": 30.0,
                        "marker_mm": 22.0,
                        "dict": "DICT_5X5_100",
                        "legacy_pattern": False,
                        "dual_pass_accept_threshold": 12,
                    },
                },
            },
        )
        hw = load_hardware_params(cfg)
        assert hw.board_cols == 7
        assert hw.board_rows == 5
        assert hw.square_mm == 30.0
        assert hw.marker_mm == 22.0
        assert hw.aruco_dict == "DICT_5X5_100"
        assert hw.legacy_pattern is False
        assert hw.dual_pass_accept_threshold == 12

    def test_overrides_ae_lock_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        # Site con luz muy fluctuante necesita settle más largo.
        self._write(
            cfg,
            {
                "vision": {
                    "ae_lock": {
                        "initial_settle_seconds": 4.0,
                        "resettle_seconds": 2.5,
                    },
                },
            },
        )
        hw = load_hardware_params(cfg)
        assert hw.ae_initial_settle_seconds == 4.0
        assert hw.ae_resettle_seconds == 2.5

    def test_partial_override_keeps_defaults_for_missing_keys(
        self,
        tmp_path: Path,
    ) -> None:
        """Overrideando solo una key, el resto queda en FLEET_DEFAULTS."""
        cfg = tmp_path / "cfg.yaml"
        self._write(cfg, {"bracket": {"baseline_mm": 160.0}})
        hw = load_hardware_params(cfg)
        assert hw.baseline_mm == 160.0
        # El resto del bracket cae al default.
        assert hw.camera_left_csi == FLEET_DEFAULTS.camera_left_csi
        assert hw.camera_right_csi == FLEET_DEFAULTS.camera_right_csi
        # Y las otras secciones también.
        assert hw.full_res == FLEET_DEFAULTS.full_res
        assert hw.board_cols == FLEET_DEFAULTS.board_cols

    def test_full_override(self, tmp_path: Path) -> None:
        """Config completo overrideando todo — caso multi-hardware total
        (otro sensor + otro lens + otro bracket + otro board)."""
        cfg = tmp_path / "cfg.yaml"
        self._write(
            cfg,
            {
                "sensor": {
                    "full_res": [3280, 2464],
                    "default_res": [1640, 1232],
                    "nominal_focal_full_px": 1680.0,
                },
                "lens": {"type": "fisheye_220", "hfov_deg": 220.0},
                "bracket": {
                    "baseline_mm": 100.0,
                    "camera_left_csi": 2,
                    "camera_right_csi": 3,
                },
                "vision": {
                    "charuco": {
                        "board_cols": 11,
                        "board_rows": 8,
                        "square_mm": 25.0,
                        "marker_mm": 18.0,
                        "dict": "DICT_6X6_250",
                        "legacy_pattern": False,
                        "dual_pass_accept_threshold": 10,
                    },
                    "ae_lock": {
                        "initial_settle_seconds": 5.0,
                        "resettle_seconds": 3.0,
                    },
                },
            },
        )
        hw = load_hardware_params(cfg)
        assert hw.full_res == (3280, 2464)
        assert hw.default_res == (1640, 1232)
        assert hw.nominal_focal_full_px == 1680.0
        assert hw.hfov_deg == 220.0
        assert hw.lens_type == "fisheye_220"
        assert hw.baseline_mm == 100.0
        assert hw.camera_left_csi == 2
        assert hw.camera_right_csi == 3
        assert hw.board_cols == 11
        assert hw.board_rows == 8
        assert hw.aruco_dict == "DICT_6X6_250"
        assert hw.legacy_pattern is False
        assert hw.dual_pass_accept_threshold == 10
        assert hw.ae_initial_settle_seconds == 5.0
        assert hw.ae_resettle_seconds == 3.0


class TestLoadHardwareParamsEdgeCases:
    def test_malformed_full_res_falls_back(self, tmp_path: Path) -> None:
        """Si full_res viene mal (no es [W, H]), cae al default sin
        crashear — el resto del config sigue procesándose."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "sensor": {"full_res": "not a list"},
                    "bracket": {"baseline_mm": 150.0},
                }
            ),
            encoding="utf-8",
        )
        hw = load_hardware_params(cfg)
        assert hw.full_res == FLEET_DEFAULTS.full_res  # fallback
        assert hw.baseline_mm == 150.0  # el resto se respeta

    def test_charuco_missing_section_uses_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """Config válido pero sin la sección vision.charuco — todos los
        knobs del board caen al fleet default."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump({"vision": {"mounting_height_m": 3.0}}),
            encoding="utf-8",
        )
        hw = load_hardware_params(cfg)
        assert hw.board_cols == FLEET_DEFAULTS.board_cols
        assert hw.aruco_dict == FLEET_DEFAULTS.aruco_dict
        assert (
            hw.dual_pass_accept_threshold == FLEET_DEFAULTS.dual_pass_accept_threshold
        )

    def test_ae_lock_missing_section_uses_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump({"vision": {"resolution": [1152, 648]}}),
            encoding="utf-8",
        )
        hw = load_hardware_params(cfg)
        assert hw.ae_initial_settle_seconds == FLEET_DEFAULTS.ae_initial_settle_seconds
        assert hw.ae_resettle_seconds == FLEET_DEFAULTS.ae_resettle_seconds

    def test_non_numeric_scalar_falls_back_without_crash(
        self,
        tmp_path: Path,
    ) -> None:
        """Un scalar no-numérico (typo del operador, ej. ``"140mm"``) NO debe
        crashear el setup tool con un traceback — cae al fleet default. Antes
        el ``float(...)`` directo levantaba ValueError sin captura."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "bracket": {"baseline_mm": "140mm"},  # string con unidad
                    "lens": {"hfov_deg": "wide"},
                    "vision": {"charuco": {"board_cols": "nine"}},
                }
            ),
            encoding="utf-8",
        )
        hw = load_hardware_params(cfg)  # no debe levantar
        assert hw.baseline_mm == FLEET_DEFAULTS.baseline_mm
        assert hw.hfov_deg == FLEET_DEFAULTS.hfov_deg
        assert hw.board_cols == FLEET_DEFAULTS.board_cols

    def test_numeric_string_is_coerced(self, tmp_path: Path) -> None:
        """Un string numérico ('160.0') sí se coerce — tolerante con YAML que
        quoteó un número."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump({"bracket": {"baseline_mm": "160.0"}}),
            encoding="utf-8",
        )
        hw = load_hardware_params(cfg)
        assert hw.baseline_mm == 160.0

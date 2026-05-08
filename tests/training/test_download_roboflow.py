"""Tests para scripts/training/download_roboflow.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Load the script as a module (it lives outside the src/ package tree).
_SPEC = importlib.util.spec_from_file_location(
    "download_roboflow",
    Path(__file__).resolve().parents[2]
    / "scripts" / "training" / "download_roboflow.py",
)
download_roboflow = importlib.util.module_from_spec(_SPEC)  # type: ignore
sys.modules["download_roboflow"] = download_roboflow
_SPEC.loader.exec_module(download_roboflow)  # type: ignore


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_cli_wins(monkeypatch):
    """CLI flag must take precedence over the env var."""
    monkeypatch.setenv("ROBOFLOW_API_KEY", "from-env")
    assert download_roboflow._resolve_api_key("from-cli") == "from-cli"


def test_resolve_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "from-env")
    assert download_roboflow._resolve_api_key(None) == "from-env"


def test_resolve_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="API key not found"):
        download_roboflow._resolve_api_key(None)


# ---------------------------------------------------------------------------
# _output_dir
# ---------------------------------------------------------------------------


def test_output_dir_is_deterministic(tmp_path):
    """Same args → same path. Used so re-downloads don't make new folders."""
    a = download_roboflow._output_dir("ws", "proj", 3, tmp_path)
    b = download_roboflow._output_dir("ws", "proj", 3, tmp_path)
    assert a == b
    assert a.parent == tmp_path
    assert "ws" in a.name and "proj" in a.name and "v3" in a.name


def test_output_dir_sanitizes_slashes(tmp_path):
    """Slugs with slashes (rare but possible) shouldn't escape the base dir."""
    out = download_roboflow._output_dir("ws/etc", "p/p", 1, tmp_path)
    assert out.parent == tmp_path
    assert "/" not in out.name


# ---------------------------------------------------------------------------
# download (mocked Roboflow SDK)
# ---------------------------------------------------------------------------


def _make_fake_roboflow(extracted_location: str):
    """Build a fake Roboflow class chain that captures download() args."""
    fake_dataset = MagicMock()
    fake_dataset.location = extracted_location

    fake_version = MagicMock()
    fake_version.download.return_value = fake_dataset

    fake_project = MagicMock()
    fake_project.version.return_value = fake_version

    fake_workspace = MagicMock()
    fake_workspace.project.return_value = fake_project

    fake_rf_instance = MagicMock()
    fake_rf_instance.workspace.return_value = fake_workspace

    fake_rf_class = MagicMock(return_value=fake_rf_instance)
    return fake_rf_class, fake_version


def _seed_data_yaml(at: Path) -> None:
    """Place a stub data.yaml so download() doesn't raise the 'no data.yaml'
    SystemExit guard during tests."""
    at.mkdir(parents=True, exist_ok=True)
    (at / "data.yaml").write_text("nc: 1\nnames: ['head']\n")


def test_download_calls_sdk_with_expected_args(tmp_path):
    out = tmp_path / "out"
    _seed_data_yaml(out)
    fake_rf_class, fake_version = _make_fake_roboflow(str(out))

    fake_module = MagicMock()
    fake_module.Roboflow = fake_rf_class

    with patch.dict(sys.modules, {"roboflow": fake_module}):
        location = download_roboflow.download(
            workspace="myws",
            project="myproj",
            version=2,
            api_key="dummy",
            out_dir=out,
            fmt="yolov8",
        )

    fake_rf_class.assert_called_once_with(api_key="dummy")
    fake_version.download.assert_called_once_with(
        "yolov8", location=str(out), overwrite=True,
    )
    assert location == out


def test_download_creates_parent_dir(tmp_path):
    """Parent dir must exist; the SDK creates the leaf itself."""
    target = tmp_path / "nested" / "sub" / "out"
    _seed_data_yaml(target)
    fake_rf_class, _ = _make_fake_roboflow(str(target))
    fake_module = MagicMock()
    fake_module.Roboflow = fake_rf_class

    with patch.dict(sys.modules, {"roboflow": fake_module}):
        download_roboflow.download(
            workspace="ws", project="p", version=1,
            api_key="k", out_dir=target,
        )

    assert target.parent.is_dir()


def test_download_raises_when_no_data_yaml_after_call(tmp_path):
    """If the SDK no-ops silently (returns location with no files), we
    must SystemExit with a useful message rather than declare success."""
    out = tmp_path / "empty_out"
    out.mkdir()  # exists but no data.yaml inside
    fake_rf_class, _ = _make_fake_roboflow(str(out))
    fake_module = MagicMock()
    fake_module.Roboflow = fake_rf_class

    with patch.dict(sys.modules, {"roboflow": fake_module}):
        with pytest.raises(SystemExit, match="no data.yaml"):
            download_roboflow.download(
                workspace="ws", project="p", version=1,
                api_key="k", out_dir=out,
            )


def test_download_missing_sdk_raises(tmp_path, monkeypatch):
    """Friendly error if 'roboflow' isn't installed."""
    # Pre-poison sys.modules so the import statement re-evaluates and
    # then force the import to fail.
    monkeypatch.setitem(sys.modules, "roboflow", None)
    with pytest.raises(SystemExit, match="not installed"):
        download_roboflow.download(
            workspace="ws", project="p", version=1,
            api_key="k", out_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_reads_data_yaml_and_counts(tmp_path):
    """Builds a minimal YOLOv8-format dataset and checks the summary."""
    import yaml

    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"names": ["person"], "nc": 1}),
    )
    for split in ("train", "valid"):
        imgs = tmp_path / split / "images"
        imgs.mkdir(parents=True)
        for i in range(3 if split == "train" else 2):
            (imgs / f"f{i}.jpg").write_bytes(b"x")
        # A non-image file should be ignored
        (imgs / "readme.txt").write_text("ignore me")

    info = download_roboflow.summarize(tmp_path)
    assert info["classes"] == ["person"]
    assert info["num_classes"] == 1
    assert info["image_counts"]["train"] == 3
    assert info["image_counts"]["valid"] == 2


def test_summarize_no_data_yaml_returns_empty(tmp_path):
    info = download_roboflow.summarize(tmp_path)
    assert info == {}

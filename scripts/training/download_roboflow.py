#!/usr/bin/env python3
"""Download a Roboflow Universe dataset in YOLOv8 format.

Phase A of the cenital head-detection training pipeline. Wraps the
Roboflow Python SDK so you can pull a dataset deterministically without
clicking through the UI.

The dataset goes to ``dataset/roboflow_<workspace>_<project>_v<N>/`` with
the standard YOLOv8 layout (``data.yaml`` + ``train/`` + ``valid/`` +
``test/``). That folder is what bench_detector.py expects locally; the
Kaggle notebook re-pulls the same dataset directly from the Roboflow API
inside its own runtime.

Usage:
    # API key via env (preferred — avoids leaking it to shell history)
    export ROBOFLOW_API_KEY=xxxxxxxxxxxxxx
    python scripts/training/download_roboflow.py \\
        --workspace <workspace-slug> \\
        --project   <project-slug> \\
        --version   N

    # Or pass it inline (less safe, useful in CI)
    python scripts/training/download_roboflow.py \\
        --api-key xxxxxxxxxxxxxx \\
        --workspace <workspace-slug> \\
        --project   <project-slug> \\
        --version   N

How to find the slugs:
    Open the dataset on universe.roboflow.com. The URL has the shape
    https://universe.roboflow.com/<workspace>/<project>/<version>
    — copy those three pieces into the flags.

Candidate datasets for cenital head detection (audit before committing):
    - Search "overhead person" / "people overhead" / "cenital head"
      on universe.roboflow.com.
    - Prefer datasets where the camera is mounted ceiling-down (NOT
      eye-level), 5k+ images, with a single ``person`` or ``head`` class.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("download_roboflow")


def _resolve_api_key(cli_arg: str | None) -> str:
    """CLI flag wins; env var as fallback. Raise if neither is present."""
    if cli_arg:
        return cli_arg
    env = os.environ.get("ROBOFLOW_API_KEY")
    if env:
        return env
    raise SystemExit(
        "Roboflow API key not found. Pass --api-key or set ROBOFLOW_API_KEY "
        "in the environment. Get a key at https://app.roboflow.com (free)."
    )


def _output_dir(workspace: str, project: str, version: int,
                base: Path) -> Path:
    """Deterministic output path so re-downloads land in the same place."""
    safe_ws = workspace.replace("/", "_")
    safe_proj = project.replace("/", "_")
    return base / f"roboflow_{safe_ws}_{safe_proj}_v{version}"


def download(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    out_dir: Path,
    fmt: str = "yolov8",
    overwrite: bool = True,
) -> Path:
    """Download dataset to ``out_dir`` and return the path Roboflow wrote.

    The Roboflow SDK's ``Version.download(format, location=...)`` returns
    a ``Dataset`` object whose ``.location`` is the absolute path of the
    extracted folder — that's what we return.

    Important: the SDK silently no-ops if the target directory already
    exists and ``overwrite=False`` (its default). It still returns a
    non-None ``Dataset`` object pointing at the empty path, so the caller
    has no way to detect the no-op. We pass ``overwrite=True`` by default
    so a re-run with an existing (possibly partial) directory always
    re-downloads. Also: we DO NOT pre-create the directory, because that
    pre-creation alone was enough to trigger the silent no-op on some SDK
    versions even with a fresh-but-empty target.
    """
    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise SystemExit(
            "The 'roboflow' package is not installed. Run:\n"
            "    pip install roboflow\n"
            "(it's a workstation-only dep; not in pyproject.toml because "
            "the Pi never downloads training data)."
        ) from e

    # Make sure the parent exists; let the SDK create the leaf directory.
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading roboflow://%s/%s/v%d -> %s (format=%s, overwrite=%s)",
        workspace, project, version, out_dir, fmt, overwrite,
    )
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    dataset = ver.download(fmt, location=str(out_dir), overwrite=overwrite)

    location = Path(getattr(dataset, "location", out_dir))
    if not (location / "data.yaml").exists():
        raise SystemExit(
            f"Roboflow SDK returned location={location} but no data.yaml "
            "is there. The SDK likely no-op'd because of a stale state. "
            "Delete the target directory and re-run."
        )
    logger.info("Dataset extracted to: %s", location)
    return location


def summarize(dataset_dir: Path) -> dict:
    """Read data.yaml + count images per split. Used as a sanity check."""
    import yaml

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        logger.warning("No data.yaml found at %s — skipping summary", data_yaml)
        return {}

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    counts = {}
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split / "images"
        if split_dir.is_dir():
            counts[split] = sum(
                1 for _ in split_dir.iterdir()
                if _.suffix.lower() in (".jpg", ".jpeg", ".png")
            )

    return {
        "classes": data.get("names", []),
        "num_classes": data.get("nc", 0),
        "image_counts": counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace", required=True,
                        help="Roboflow workspace slug (from URL)")
    parser.add_argument("--project", required=True,
                        help="Roboflow project slug (from URL)")
    parser.add_argument("--version", type=int, required=True,
                        help="Roboflow dataset version number (from URL)")
    parser.add_argument("--api-key",
                        help="Roboflow API key (else $ROBOFLOW_API_KEY)")
    parser.add_argument(
        "--output-base", type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
        help="Base dir for dataset folder (default: <repo>/dataset/)")
    parser.add_argument("--format", default="yolov8",
                        choices=("yolov8", "yolov5pytorch", "coco"),
                        help="Roboflow export format (default: yolov8)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = _resolve_api_key(args.api_key)
    out_dir = _output_dir(args.workspace, args.project, args.version,
                          args.output_base)

    location = download(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        api_key=api_key,
        out_dir=out_dir,
        fmt=args.format,
    )

    info = summarize(location)
    if info:
        logger.info("Classes: %s (nc=%d)",
                    info["classes"], info["num_classes"])
        logger.info("Images per split: %s", info["image_counts"])

    print(f"\nDataset ready at: {location}")
    print(f"Use this path as the dataset root for bench_detector.py.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

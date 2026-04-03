#!/usr/bin/env python3
"""Download YOLOv8n model for People Counter.

Downloads the model in the appropriate format:
  - ONNX: For development/testing with the OpenCV backend.
  - HEF: For production on Hailo-8L (from Hailo Model Zoo).

Usage:
    # Download ONNX for development (works on any machine)
    python scripts/download_model.py onnx

    # Download pre-compiled HEF for Hailo-8L (run on RPi5)
    python scripts/download_model.py hef

    # Export ONNX from ultralytics (requires ultralytics pip package)
    python scripts/download_model.py export-onnx
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("download_model")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Hailo Model Zoo — pre-compiled HEFs for Hailo-8L
HAILO_MODEL_ZOO_REPO = "https://github.com/hailo-ai/hailo_model_zoo"
HAILO_HEF_URL = (
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
    "ModelZoo/Compiled/v2.14.0/hailo8l/yolov8n.hef"
)


def cmd_onnx(args: argparse.Namespace) -> None:
    """Export YOLOv8n to ONNX using ultralytics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "yolov8n.onnx"

    if output_path.exists() and not args.force:
        logger.info("ONNX model already exists at %s", output_path)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics not installed. Install with: pip install ultralytics"
        )
        sys.exit(1)

    logger.info("Loading YOLOv8n and exporting to ONNX...")
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, opset=12)

    # ultralytics saves next to the .pt file, move to our models dir
    exported = Path("yolov8n.onnx")
    if exported.exists():
        exported.rename(output_path)
    logger.info("ONNX model saved to %s", output_path)


def cmd_hef(args: argparse.Namespace) -> None:
    """Download pre-compiled HEF for Hailo-8L."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "yolov8n.hef"

    if output_path.exists() and not args.force:
        logger.info("HEF model already exists at %s", output_path)
        return

    logger.info("Downloading pre-compiled YOLOv8n HEF for Hailo-8L...")
    logger.info("Source: %s", HAILO_HEF_URL)

    try:
        subprocess.run(
            ["curl", "-L", "-o", str(output_path), HAILO_HEF_URL],
            check=True,
        )
    except FileNotFoundError:
        # Fallback to Python urllib
        import urllib.request

        urllib.request.urlretrieve(HAILO_HEF_URL, str(output_path))

    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("HEF model saved to %s (%.1f MB)", output_path, size_mb)
    else:
        logger.error(
            "Download failed. The pre-compiled HEF URL may have changed.\n"
            "Check the Hailo Model Zoo for the latest URL:\n"
            "  %s\n\n"
            "Alternatively, compile from ONNX using the Hailo Dataflow Compiler:\n"
            "  hailo optimize yolov8n.onnx --hw-arch hailo8l\n"
            "  hailo compile yolov8n.har --hw-arch hailo8l",
            HAILO_MODEL_ZOO_REPO,
        )
        output_path.unlink(missing_ok=True)
        sys.exit(1)


def cmd_export_onnx(args: argparse.Namespace) -> None:
    """Export YOLOv8n to ONNX (alias for onnx command)."""
    cmd_onnx(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLOv8n model")
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("onnx", help="Export YOLOv8n to ONNX (needs ultralytics)")
    sub.add_parser("hef", help="Download pre-compiled HEF for Hailo-8L")
    sub.add_parser("export-onnx", help="Alias for onnx")

    args = parser.parse_args()

    if args.command in ("onnx", "export-onnx"):
        cmd_onnx(args)
    elif args.command == "hef":
        cmd_hef(args)


if __name__ == "__main__":
    main()

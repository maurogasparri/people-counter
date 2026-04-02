"""Person detection using YOLOv8n on Hailo-8L accelerator.

Supports two backends:
  - hailo: Production. Runs HEF model on Hailo-8L via hailo_platform SDK.
  - opencv: Development/testing. Runs ONNX model via OpenCV DNN.

The post-processing (NMS, bbox extraction, confidence filtering) is
shared between both backends and fully testable without hardware.

COCO class 0 = "person". We filter for this class only.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO_PERSON_CLASS = 0
INPUT_SIZE = (640, 640)  # YOLOv8n standard input


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """A single person detection."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image coords
    confidence: float
    centroid: tuple[float, float]  # (cx, cy) center of bbox

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "centroid": list(self.centroid),
        }


# ---------------------------------------------------------------------------
# Model backend protocol
# ---------------------------------------------------------------------------


class DetectionBackend(Protocol):
    """Protocol for detection backends."""

    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed input.

        Args:
            preprocessed: (1, 3, 640, 640) float32 normalized [0, 1].

        Returns:
            Raw model output tensor. Shape depends on backend but is
            typically (1, 84, 8400) for YOLOv8n (transposed COCO).
        """
        ...


# ---------------------------------------------------------------------------
# Hailo backend (production — RPi5 + Hailo-8L)
# ---------------------------------------------------------------------------


class HailoBackend:
    """Hailo-8L inference backend using hailo_platform SDK."""

    def __init__(self, hef_path: str) -> None:
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError:
            raise ImportError(
                "hailo_platform SDK not installed. "
                "Install with: pip install hailo-platform. "
                "Only available on RPi5 with Hailo-8L."
            )

        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF model not found: {hef_path}")

        self._hef = HEF(hef_path)
        self._device = VDevice()
        self._network_group = self._device.configure(
            self._hef, ConfigureParams.create_from_hef(self._hef)
        )[0]

        self._input_params = InputVStreamParams.make_from_network_group(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self._output_params = OutputVStreamParams.make_from_network_group(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Store references for cleanup
        self._InferVStreams = InferVStreams

        logger.info("Hailo backend loaded: %s", hef_path)

    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run inference on Hailo-8L."""
        input_name = self._hef.get_input_vstream_infos()[0].name

        with self._InferVStreams(
            self._network_group, self._input_params, self._output_params
        ) as pipeline:
            input_data = {input_name: preprocessed}
            result = pipeline.infer(input_data)

        # Get first (and only) output
        output_name = list(result.keys())[0]
        return result[output_name]


# ---------------------------------------------------------------------------
# OpenCV DNN backend (development — any machine with ONNX model)
# ---------------------------------------------------------------------------


class OpenCVBackend:
    """OpenCV DNN inference backend for ONNX models (CPU/GPU)."""

    def __init__(self, onnx_path: str) -> None:
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        # Prefer CUDA if available, fall back to CPU
        try:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("OpenCV backend: CUDA")
        except Exception:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("OpenCV backend: CPU")

        logger.info("OpenCV DNN backend loaded: %s", onnx_path)

    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run inference via OpenCV DNN."""
        self._net.setInput(preprocessed)
        output = self._net.forward()
        return output


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(
    frame: np.ndarray,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[np.ndarray, float, int, int]:
    """Preprocess a frame for YOLOv8 inference.

    Applies letterbox resize maintaining aspect ratio, then normalizes.

    Args:
        frame: BGR image of any size.
        input_size: Target (width, height) for the model.

    Returns:
        (blob, scale, pad_x, pad_y) where:
            blob: (1, 3, H, W) float32 normalized [0, 1].
            scale: Scale factor applied during resize.
            pad_x, pad_y: Padding offsets in pixels.
    """
    target_w, target_h = input_size
    h, w = frame.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size (center padding)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # HWC BGR → CHW RGB, normalize to [0, 1]
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # Add batch dimension

    return blob, scale, pad_x, pad_y


# ---------------------------------------------------------------------------
# Post-processing (shared, hardware-independent)
# ---------------------------------------------------------------------------


def postprocess(
    raw_output: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_size: tuple[int, int],
) -> list[Detection]:
    """Post-process YOLOv8 raw output into person detections.

    YOLOv8 output shape: (1, 84, 8400) where:
        - 84 = 4 bbox coords (cx, cy, w, h) + 80 COCO class scores
        - 8400 = number of prediction anchors

    Args:
        raw_output: Raw model output tensor.
        confidence_threshold: Minimum confidence to keep.
        nms_threshold: IoU threshold for NMS.
        scale: Scale factor from preprocessing.
        pad_x, pad_y: Padding offsets from preprocessing.
        original_size: (width, height) of the original image.

    Returns:
        List of Detection objects for persons only.
    """
    # Squeeze batch dimension if present
    if raw_output.ndim == 3:
        output = raw_output[0]  # (84, 8400)
    else:
        output = raw_output

    # YOLOv8 output can be (84, 8400) or (8400, 84) depending on export
    if output.shape[0] == 84:
        output = output.T  # → (8400, 84)

    # Extract person class scores (class 0 in COCO)
    # Columns: [cx, cy, w, h, class0_score, class1_score, ..., class79_score]
    person_scores = output[:, 4 + COCO_PERSON_CLASS]

    # Filter by confidence
    mask = person_scores >= confidence_threshold
    if not np.any(mask):
        return []

    filtered = output[mask]
    scores = person_scores[mask]

    # Convert from cx, cy, w, h to x1, y1, x2, y2
    cx = filtered[:, 0]
    cy = filtered[:, 1]
    w = filtered[:, 2]
    h = filtered[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Remove padding and rescale to original image coordinates
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    orig_w, orig_h = original_size

    # Clip to image boundaries
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        confidence_threshold,
        nms_threshold,
    )

    if len(indices) == 0:
        return []

    # Build Detection objects
    detections = []
    for i in indices.flatten():
        bx1, by1, bx2, by2 = (
            int(boxes[i, 0]),
            int(boxes[i, 1]),
            int(boxes[i, 2]),
            int(boxes[i, 3]),
        )
        conf = float(scores[i])
        cx_det = (bx1 + bx2) / 2.0
        cy_det = (by1 + by2) / 2.0

        detections.append(
            Detection(
                bbox=(bx1, by1, bx2, by2),
                confidence=conf,
                centroid=(cx_det, cy_det),
            )
        )

    return detections


# ---------------------------------------------------------------------------
# Public API (matches original interface + adds backend flexibility)
# ---------------------------------------------------------------------------


def load_model(model_path: str, backend: str = "auto") -> dict[str, Any]:
    """Load detection model.

    Args:
        model_path: Path to HEF (Hailo) or ONNX (OpenCV) model file.
        backend: "hailo", "opencv", or "auto" (detect from file extension).

    Returns:
        Dict with "backend" instance and "type" string.
    """
    path = Path(model_path)

    if backend == "auto":
        if path.suffix == ".hef":
            backend = "hailo"
        elif path.suffix == ".onnx":
            backend = "opencv"
        else:
            raise ValueError(
                f"Cannot auto-detect backend for {path.suffix}. "
                "Use backend='hailo' or backend='opencv'."
            )

    if backend == "hailo":
        model_backend = HailoBackend(model_path)
    elif backend == "opencv":
        model_backend = OpenCVBackend(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {"backend": model_backend, "type": backend}


def detect_persons(
    frame: np.ndarray,
    model: dict[str, Any],
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    """Run person detection on a single frame.

    Args:
        frame: BGR image of any size.
        model: Dict from load_model().
        confidence_threshold: Minimum confidence for detections.
        nms_threshold: IoU threshold for NMS.

    Returns:
        List of Detection objects (person class only).
    """
    backend: DetectionBackend = model["backend"]

    blob, scale, pad_x, pad_y = preprocess(frame)
    raw_output = backend.infer(blob)

    orig_h, orig_w = frame.shape[:2]

    return postprocess(
        raw_output,
        confidence_threshold,
        nms_threshold,
        scale,
        pad_x,
        pad_y,
        (orig_w, orig_h),
    )

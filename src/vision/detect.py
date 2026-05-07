"""Person detection with pluggable model architectures.

Two backends share the preprocessing path (letterbox + uint8 NHWC for
Hailo, or float32 NCHW for OpenCV):

  - hailo: Production. Runs HEF on Hailo-8L via hailo_platform SDK.
  - opencv: Development/testing. Runs ONNX via OpenCV DNN.

The post-processing path is **architecture-specific** and lives in the
``ARCHITECTURES`` registry below. Each entry knows its input size and
its decoder. Today we ship two:

  - yolov8: COCO 80-class head. Auto-detects whether the HEF emitted a
    Hailo-NMS list-of-arrays or a raw ``(1, 84, N)`` tensor.
  - rapid:  Boston VIP rotated-bbox head, 1-class person. HEF emits
    three raw per-scale tensors (strides 8/16/32). Decode runs on CPU
    after Hailo since the rotated head is not supported by the on-chip
    NMS unit.

Adding a new architecture is one entry in ``ARCHITECTURES`` plus a
postprocess callable with the same signature.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO_PERSON_CLASS = 0
DEFAULT_ARCHITECTURE = "yolov8"


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

    def infer(self, preprocessed: np.ndarray) -> Any:
        """Run inference on preprocessed input.

        Args:
            preprocessed: ``(1, 3, H, W)`` float32 normalized [0, 1].

        Returns:
            Raw model output. Shape and structure depend on the
            architecture (see ``ARCHITECTURES``):

            - yolov8 + Hailo built-in NMS → list of 80 per-class arrays.
            - yolov8 raw → ndarray of shape ``(1, 84, N)``.
            - rapid raw → list of 3 per-scale ndarrays.
        """
        ...


# ---------------------------------------------------------------------------
# Hailo backend (production — RPi5 + Hailo-8L)
# ---------------------------------------------------------------------------


class HailoBackend:
    """Hailo-8L inference backend using hailo_platform SDK.

    Uses the VStream API with persistent activation — the network group
    and inference pipeline stay open for the lifetime of the backend,
    avoiding per-frame setup/teardown overhead.

    Handles both single-output HEFs (e.g. YOLOv8 with on-chip NMS, where
    the only output is the per-class detection list) and multi-output
    HEFs (e.g. RAPiD raw, three per-scale tensors).
    """

    def __init__(self, hef_path: str) -> None:
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoSchedulingAlgorithm,
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

        # Shared VDevice with round-robin scheduling (Hailo best practice)
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self._device = VDevice(params)

        self._network_group = self._device.configure(
            self._hef,
            ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            ),
        )[0]

        self._input_params = InputVStreamParams.make_from_network_group(
            self._network_group, quantized=True, format_type=FormatType.UINT8
        )
        self._output_params = OutputVStreamParams.make_from_network_group(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        self._input_name = self._hef.get_input_vstream_infos()[0].name
        # Capture HEF-declared output order so multi-output models decode
        # in a deterministic order (S, M, L for RAPiD).
        self._output_names = [
            info.name for info in self._hef.get_output_vstream_infos()
        ]

        # Activate network group and open inference pipeline persistently
        # instead of per-frame. Keeps the HW context warm.
        self._activation_ctx = self._network_group.activate()
        self._activation_ctx.__enter__()
        self._pipeline = InferVStreams(
            self._network_group, self._input_params, self._output_params
        )
        self._pipeline.__enter__()

        logger.info(
            "hailo_backend_loaded",
            extra={"path": hef_path, "outputs": len(self._output_names)},
        )

    def infer(self, preprocessed: np.ndarray) -> Any:
        """Run inference on Hailo-8L.

        The HEF model expects uint8 NHWC input. This method handles the
        conversion from the float32 NCHW blob produced by preprocess().

        Returns:
            For single-output HEFs, the unwrapped output (batch dim
            stripped). For multi-output HEFs, a list of per-output
            arrays in HEF declaration order.
        """
        # preprocess() outputs (1, 3, H, W) float32 [0,1]
        # Hailo expects (1, H, W, 3) uint8 [0,255]
        if preprocessed.ndim == 4 and preprocessed.shape[1] == 3:
            preprocessed = preprocessed.transpose(0, 2, 3, 1)
        img = (preprocessed * 255).clip(0, 255).astype(np.uint8)

        result = self._pipeline.infer(
            {self._input_name: np.ascontiguousarray(img)}
        )

        if len(self._output_names) == 1:
            # Strip batch dim; for YOLOv8-NMS this yields the list of 80
            # per-class arrays, for raw single-tensor heads it yields the
            # decoded ndarray.
            return result[self._output_names[0]][0]
        return [result[name][0] for name in self._output_names]

    def close(self) -> None:
        """Release Hailo resources."""
        try:
            self._pipeline.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._activation_ctx.__exit__(None, None, None)
        except Exception:
            pass
        logger.info("hailo_backend_closed")


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
            logger.info("opencv_backend_target", extra={"target": "cuda"})
        except Exception:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("opencv_backend_target", extra={"target": "cpu"})

        logger.info("opencv_backend_loaded", extra={"path": onnx_path})

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
    input_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, int, int]:
    """Preprocess a frame for letterboxed model inference.

    Applies letterbox resize maintaining aspect ratio, then normalizes.

    Args:
        frame: BGR image of any size.
        input_size: Target ``(width, height)`` for the model.

    Returns:
        ``(blob, scale, pad_x, pad_y)`` where:

        - ``blob``: ``(1, 3, H, W)`` float32 normalized [0, 1].
        - ``scale``: Scale factor applied during resize.
        - ``pad_x, pad_y``: Padding offsets in pixels.
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
# Post-processing: YOLOv8 raw output (ONNX via OpenCV DNN)
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

    YOLOv8 output shape: ``(1, 84, 8400)`` where:

    - 84 = 4 bbox coords (cx, cy, w, h) + 80 COCO class scores
    - 8400 = number of prediction anchors

    Args:
        raw_output: Raw model output tensor.
        confidence_threshold: Minimum confidence to keep.
        nms_threshold: IoU threshold for NMS.
        scale: Scale factor from preprocessing.
        pad_x, pad_y: Padding offsets from preprocessing.
        original_size: ``(width, height)`` of the original image.

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
# Post-processing: YOLOv8 with Hailo built-in NMS
# ---------------------------------------------------------------------------


def postprocess_hailo_nms(
    raw_output: list,
    confidence_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_size: tuple[int, int],
    input_size: tuple[int, int] = (640, 640),
) -> list[Detection]:
    """Post-process Hailo NMS output into person detections.

    Hailo NMS output is a list of 80 arrays (one per COCO class).
    Each array has shape ``(N, 5)`` where:

    - N: number of detections for that class (variable per class)
    - 5: ``[y_min, x_min, y_max, x_max, score]`` normalized [0, 1]

    Args:
        raw_output: List of 80 arrays from Hailo inference.
        confidence_threshold: Minimum confidence to keep.
        scale: Scale factor from preprocessing.
        pad_x, pad_y: Padding offsets from preprocessing.
        original_size: ``(width, height)`` of the original image.
        input_size: ``(width, height)`` model input — used to undo the
            normalized coords. YOLOv8 trained at 640.

    Returns:
        List of Detection objects for persons only.
    """
    # Extract person class (class 0) — shape (N, 5)
    person_data = np.array(raw_output[COCO_PERSON_CLASS])

    if person_data.ndim != 2 or person_data.shape[0] == 0:
        return []

    orig_w, orig_h = original_size
    input_w, input_h = input_size

    detections = []
    for i in range(person_data.shape[0]):
        y1_n, x1_n, y2_n, x2_n, score = person_data[i]

        if score < confidence_threshold:
            continue

        # Convert from normalized to input pixel coords
        x1 = x1_n * input_w
        y1 = y1_n * input_h
        x2 = x2_n * input_w
        y2 = y2_n * input_h

        # Remove padding and rescale to original image
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip to image boundaries
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        detections.append(
            Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(score),
                centroid=(cx, cy),
            )
        )

    return detections


# ---------------------------------------------------------------------------
# Post-processing: RAPiD raw 3-scale output
# ---------------------------------------------------------------------------

# RAPiD anchors for the MWHB1024 weights (Boston VIP). Per-scale 3 anchors,
# one row each = (anchor_w_px, anchor_h_px) at the model's native 1024×1024
# input. The HEF outputs raw per-scale tensors (no on-chip decode), so we
# reproduce the original PyTorch decode here on CPU.
RAPID_ANCHORS = {
    8: np.array(
        [[18.78, 33.47], [28.89, 61.75], [48.68, 68.39]], dtype=np.float32,
    ),
    16: np.array(
        [[45.07, 101.47], [63.10, 113.54], [81.39, 134.46]], dtype=np.float32,
    ),
    32: np.array(
        [[91.74, 144.99], [137.52, 178.48], [194.44, 250.80]], dtype=np.float32,
    ),
}
RAPID_STRIDES = (8, 16, 32)
RAPID_INPUT_SIZE = (1024, 1024)
# Angle range used by RAPiD's head: sigmoid output mapped to (-180°, 180°).
# The exact magnitude doesn't matter for our axis-aligned-bbox output (the
# tight enclosing rectangle is the same up to a sign-symmetric flip), but
# keeping the original parametrization avoids surprises if someone reuses
# this decoder for rotated NMS later.
_RAPID_ANGLE_RANGE_DEG = 360.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rapid_to_grid(
    arr: np.ndarray, expected_channels: int = 18,
) -> "np.ndarray | None":
    """Normalize a RAPiD per-scale tensor to ``(H, W, n_anchors, 6)``.

    Hailo VStream returns NHWC by default; the OpenCV path may return
    NCHW. We accept both and any leading batch dim.
    """
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        return None

    if a.shape[-1] == expected_channels:
        h, w, _ = a.shape
    elif a.shape[0] == expected_channels:
        a = np.transpose(a, (1, 2, 0))
        h, w, _ = a.shape
    else:
        return None

    return a.reshape(h, w, 3, 6)


def postprocess_rapid(
    raw_output: list,
    confidence_threshold: float,
    nms_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_size: tuple[int, int],
    input_size: tuple[int, int] = RAPID_INPUT_SIZE,
) -> list[Detection]:
    """Post-process RAPiD raw 3-scale output into person detections.

    The RAPiD HEF emits three per-scale tensors (S/M/L = strides 8/16/32),
    each with 18 channels = ``3 anchors × (cx, cy, w, h, angle, conf)``.
    We sigmoid + decode anchors on CPU, convert each rotated rectangle to
    its tight axis-aligned enclosing box, and run plain NMS on those
    enclosing boxes — the tracker downstream only consumes axis-aligned
    bboxes, so we collapse the rotation here.

    Args:
        raw_output: List of 3 per-scale ndarrays in HEF order (S, M, L).
        confidence_threshold: Min sigmoid confidence to keep.
        nms_threshold: IoU threshold for axis-aligned NMS.
        scale: Letterbox scale factor.
        pad_x, pad_y: Letterbox padding offsets.
        original_size: ``(width, height)`` of the original image.
        input_size: Model input ``(W, H)`` — defaults to 1024×1024.
    """
    if not isinstance(raw_output, (list, tuple)) or len(raw_output) != 3:
        return []

    all_boxes: list[np.ndarray] = []  # axis-aligned [x1, y1, x2, y2]
    all_scores: list[np.ndarray] = []

    for arr, stride in zip(raw_output, RAPID_STRIDES):
        grid = _rapid_to_grid(arr)
        if grid is None:
            continue

        h, w = grid.shape[:2]
        tx = grid[..., 0]
        ty = grid[..., 1]
        tw = grid[..., 2]
        th = grid[..., 3]
        ta = grid[..., 4]
        tc = grid[..., 5]

        conf = _sigmoid(tc)
        mask = conf >= confidence_threshold
        if not np.any(mask):
            continue

        # Build cell-grid offsets, broadcast across the 3 anchor slots.
        gy_, gx_ = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        gx = gx_[..., None].astype(np.float32)  # (h, w, 1)
        gy = gy_[..., None].astype(np.float32)

        bx = (_sigmoid(tx) + gx) * stride
        by = (_sigmoid(ty) + gy) * stride
        anchors = RAPID_ANCHORS[stride]  # (3, 2)
        bw = anchors[None, None, :, 0] * np.exp(tw)
        bh = anchors[None, None, :, 1] * np.exp(th)
        angle_deg = (_sigmoid(ta) - 0.5) * _RAPID_ANGLE_RANGE_DEG

        bx_f = bx[mask]
        by_f = by[mask]
        bw_f = bw[mask]
        bh_f = bh[mask]
        ang_f = angle_deg[mask]
        conf_f = conf[mask]

        # Tight axis-aligned bbox of the rotated rectangle: rotate the
        # 4 corner offsets and take the per-row min/max. Vectorized over
        # all surviving cells in this scale.
        cos_a = np.cos(np.deg2rad(ang_f))
        sin_a = np.sin(np.deg2rad(ang_f))
        dx = np.stack(
            [-bw_f / 2, bw_f / 2, bw_f / 2, -bw_f / 2], axis=1,
        )
        dy = np.stack(
            [-bh_f / 2, -bh_f / 2, bh_f / 2, bh_f / 2], axis=1,
        )
        rx = dx * cos_a[:, None] - dy * sin_a[:, None]
        ry = dx * sin_a[:, None] + dy * cos_a[:, None]
        cx = bx_f[:, None] + rx
        cy = by_f[:, None] + ry
        x1 = cx.min(axis=1)
        y1 = cy.min(axis=1)
        x2 = cx.max(axis=1)
        y2 = cy.max(axis=1)

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(conf_f)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)

    # Undo letterbox to get back to original-image coords.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    orig_w, orig_h = original_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        confidence_threshold,
        nms_threshold,
    )
    if len(indices) == 0:
        return []

    detections: list[Detection] = []
    for i in np.asarray(indices).flatten():
        bx1, by1, bx2, by2 = boxes[i]
        detections.append(
            Detection(
                bbox=(int(bx1), int(by1), int(bx2), int(by2)),
                confidence=float(scores[i]),
                centroid=((bx1 + bx2) / 2.0, (by1 + by2) / 2.0),
            )
        )
    return detections


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------


def _postprocess_yolov8(
    raw_output: Any,
    confidence_threshold: float,
    nms_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
) -> list[Detection]:
    """Dispatch yolov8 output to the right decoder.

    Hailo HEF with built-in NMS yields a per-class list; the OpenCV /
    raw-HEF path yields a single ndarray. We branch on type so the same
    arch entry covers both.
    """
    if isinstance(raw_output, list):
        return postprocess_hailo_nms(
            raw_output,
            confidence_threshold,
            scale,
            pad_x,
            pad_y,
            original_size,
            input_size=input_size,
        )
    return postprocess(
        raw_output,
        confidence_threshold,
        nms_threshold,
        scale,
        pad_x,
        pad_y,
        original_size,
    )


def _postprocess_rapid(
    raw_output: Any,
    confidence_threshold: float,
    nms_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
) -> list[Detection]:
    return postprocess_rapid(
        raw_output,
        confidence_threshold,
        nms_threshold,
        scale,
        pad_x,
        pad_y,
        original_size,
        input_size=input_size,
    )


# Each entry binds a name to its expected ``input_size`` (for letterbox)
# and its ``postprocess`` callable. Adding a new model = one entry here +
# a postprocess function with the same signature.
PostprocessFn = Callable[
    [Any, float, float, float, int, int, tuple[int, int], tuple[int, int]],
    list[Detection],
]
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "yolov8": {
        "input_size": (640, 640),
        "postprocess": _postprocess_yolov8,
    },
    "rapid": {
        "input_size": (1024, 1024),
        "postprocess": _postprocess_rapid,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(
    model_path: str,
    backend: str = "auto",
    architecture: str = DEFAULT_ARCHITECTURE,
) -> dict[str, Any]:
    """Load detection model.

    Args:
        model_path: Path to HEF (Hailo) or ONNX (OpenCV) model file.
        backend: ``"hailo"``, ``"opencv"``, or ``"auto"`` (file-extension
            based).
        architecture: Postprocess selector — must be a key in
            ``ARCHITECTURES``. Defaults to yolov8.

    Returns:
        Dict with ``backend`` instance, ``type`` string, ``architecture``
        name, and ``input_size`` tuple.
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

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Known: {sorted(ARCHITECTURES)}"
        )

    if backend == "hailo":
        model_backend: Any = HailoBackend(model_path)
    elif backend == "opencv":
        model_backend = OpenCVBackend(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    spec = ARCHITECTURES[architecture]
    return {
        "backend": model_backend,
        "type": backend,
        "architecture": architecture,
        "input_size": spec["input_size"],
    }


def detect_persons(
    frame: np.ndarray,
    model: dict[str, Any],
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    """Run person detection on a single frame.

    Args:
        frame: BGR image of any size.
        model: Dict from ``load_model()``.
        confidence_threshold: Minimum confidence for detections.
        nms_threshold: IoU threshold for NMS.

    Returns:
        List of Detection objects (person class only).
    """
    backend: DetectionBackend = model["backend"]
    architecture = model.get("architecture", DEFAULT_ARCHITECTURE)
    input_size: tuple[int, int] = model.get(
        "input_size", ARCHITECTURES[architecture]["input_size"],
    )
    postprocess_fn: PostprocessFn = ARCHITECTURES[architecture]["postprocess"]

    blob, scale, pad_x, pad_y = preprocess(frame, input_size=input_size)
    raw_output = backend.infer(blob)

    orig_h, orig_w = frame.shape[:2]
    return postprocess_fn(
        raw_output,
        confidence_threshold,
        nms_threshold,
        scale,
        pad_x,
        pad_y,
        (orig_w, orig_h),
        input_size,
    )

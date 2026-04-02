"""Tests for person detection module.

Tests the preprocessing, postprocessing, and Detection dataclass.
Does NOT test actual model inference (requires HEF or ONNX model file).
"""

import cv2
import numpy as np
import pytest

from src.vision.detect import (
    Detection,
    postprocess,
    preprocess,
)


class TestPreprocess:
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess(frame)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32

    def test_normalized_range(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob, _, _, _ = preprocess(frame)
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_scale_and_padding(self):
        # 640x480 → scale to fit 640x640 → scale = 1.0 on width
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess(frame)
        assert scale == 1.0
        assert pad_x == 0
        assert pad_y == 80  # (640-480)/2

    def test_wide_image_scaling(self):
        # 1280x480 → scale = 0.5, new size 640x240, pad_y = 200
        frame = np.zeros((480, 1280, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess(frame)
        assert scale == 0.5
        assert pad_x == 0
        assert pad_y == 200

    def test_tall_image_scaling(self):
        # 320x960 → scale = 320/960 ≈ 0.667 on height, new_h=640
        frame = np.zeros((960, 320, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess(frame)
        expected_scale = 640 / 960  # ~0.667
        assert abs(scale - expected_scale) < 0.01

    def test_square_image(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess(frame)
        assert scale == 1.0
        assert pad_x == 0
        assert pad_y == 0


class TestPostprocess:
    def _make_raw_output(
        self,
        detections: list[tuple[float, float, float, float, float]],
        num_classes: int = 80,
    ) -> np.ndarray:
        """Create a synthetic YOLOv8 raw output tensor.

        Args:
            detections: List of (cx, cy, w, h, person_confidence).
            num_classes: Number of COCO classes.

        Returns:
            (1, 84, N) tensor mimicking YOLOv8 output.
        """
        n = len(detections)
        if n == 0:
            return np.zeros((1, 4 + num_classes, 0), dtype=np.float32)

        output = np.zeros((4 + num_classes, n), dtype=np.float32)
        for i, (cx, cy, w, h, conf) in enumerate(detections):
            output[0, i] = cx
            output[1, i] = cy
            output[2, i] = w
            output[3, i] = h
            output[4, i] = conf  # person class = 0

        return np.expand_dims(output, 0)  # (1, 84, N)

    def test_single_detection(self):
        # Person at center of 640x640 input, 100x200 bbox
        raw = self._make_raw_output([(320, 320, 100, 200, 0.9)])
        dets = postprocess(
            raw,
            confidence_threshold=0.5,
            nms_threshold=0.45,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            original_size=(640, 640),
        )
        assert len(dets) == 1
        assert dets[0].confidence == pytest.approx(0.9, abs=0.01)
        assert dets[0].bbox[0] < dets[0].bbox[2]  # x1 < x2
        assert dets[0].bbox[1] < dets[0].bbox[3]  # y1 < y2

    def test_confidence_filtering(self):
        raw = self._make_raw_output([
            (100, 100, 50, 100, 0.9),  # above threshold
            (300, 300, 50, 100, 0.3),  # below threshold
        ])
        dets = postprocess(raw, 0.5, 0.45, 1.0, 0, 0, (640, 640))
        assert len(dets) == 1
        assert dets[0].confidence == pytest.approx(0.9, abs=0.01)

    def test_empty_output(self):
        raw = self._make_raw_output([])
        dets = postprocess(raw, 0.5, 0.45, 1.0, 0, 0, (640, 640))
        assert len(dets) == 0

    def test_all_below_threshold(self):
        raw = self._make_raw_output([
            (100, 100, 50, 100, 0.1),
            (300, 300, 50, 100, 0.2),
        ])
        dets = postprocess(raw, 0.5, 0.45, 1.0, 0, 0, (640, 640))
        assert len(dets) == 0

    def test_nms_suppression(self):
        # Two highly overlapping detections — NMS should keep only best
        raw = self._make_raw_output([
            (320, 320, 100, 200, 0.9),
            (325, 322, 100, 200, 0.8),  # nearly identical position
        ])
        dets = postprocess(raw, 0.5, 0.3, 1.0, 0, 0, (640, 640))
        assert len(dets) == 1  # NMS keeps only the best

    def test_scale_and_padding_undo(self):
        # Detection at (320, 320) in model space with scale=0.5, pad=(0, 80)
        # Real position: (320 - 0) / 0.5 = 640, (320 - 80) / 0.5 = 480
        raw = self._make_raw_output([(320, 400, 100, 100, 0.9)])
        dets = postprocess(
            raw,
            confidence_threshold=0.5,
            nms_threshold=0.45,
            scale=0.5,
            pad_x=0,
            pad_y=80,
            original_size=(1280, 960),
        )
        assert len(dets) == 1
        cx, cy = dets[0].centroid
        # Expected center: ((320-0)/0.5, (400-80)/0.5) = (640, 640)
        assert abs(cx - 640) < 10
        assert abs(cy - 640) < 10

    def test_bbox_clipping(self):
        # Detection near edge — bbox should be clipped
        raw = self._make_raw_output([(10, 10, 100, 100, 0.9)])
        dets = postprocess(raw, 0.5, 0.45, 1.0, 0, 0, (640, 480))
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0].bbox
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480

    def test_multiple_persons(self):
        raw = self._make_raw_output([
            (100, 200, 60, 120, 0.9),
            (400, 200, 60, 120, 0.85),
            (300, 500, 60, 120, 0.7),
        ])
        dets = postprocess(raw, 0.5, 0.45, 1.0, 0, 0, (640, 640))
        assert len(dets) == 3

    def test_transposed_output(self):
        """Should handle both (84, N) and (N, 84) orientations."""
        raw = self._make_raw_output([(320, 320, 100, 200, 0.9)])
        # Transpose to (1, 8400, 84)
        transposed = raw.transpose(0, 2, 1)
        dets = postprocess(transposed, 0.5, 0.45, 1.0, 0, 0, (640, 640))
        assert len(dets) == 1


class TestDetection:
    def test_to_dict(self):
        det = Detection(
            bbox=(10, 20, 110, 220),
            confidence=0.85,
            centroid=(60.0, 120.0),
        )
        d = det.to_dict()
        assert d["bbox"] == [10, 20, 110, 220]
        assert d["confidence"] == 0.85
        assert d["centroid"] == [60.0, 120.0]

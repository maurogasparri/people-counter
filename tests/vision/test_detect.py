"""Tests para el módulo de detección de personas.

Testea el preproceso, postproceso, y la dataclass Detection.
Does NOT test actual model inference (requires HEF or ONNX model file).
"""

import cv2
import numpy as np
import pytest

from src.vision.detect import (
    ARCHITECTURES,
    RAPID_ANCHORS,
    RAPID_STRIDES,
    Detection,
    cluster_detections,
    postprocess,
    postprocess_hailo_nms,
    postprocess_rapid,
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

    def test_rotated_default_none(self):
        """Axis-aligned detectors (yolov8) leave ``rotated`` unset; the
        field must default to None and stay out of ``to_dict`` so
        downstream consumers don't see a noisy null key."""
        det = Detection(
            bbox=(10, 20, 110, 220),
            confidence=0.85,
            centroid=(60.0, 120.0),
        )
        assert det.rotated is None
        assert "rotated" not in det.to_dict()

    def test_rotated_serialised_when_set(self):
        """Architectures that emit a rotated rectangle (RAPiD) populate
        the field; ``to_dict`` includes it so the JSON event preserves
        the body footprint for downstream consumers."""
        det = Detection(
            bbox=(10, 20, 110, 220),
            confidence=0.85,
            centroid=(60.0, 120.0),
            rotated=(60.0, 120.0, 80.0, 200.0, 30.0),
        )
        d = det.to_dict()
        assert d["rotated"] == [60.0, 120.0, 80.0, 200.0, 30.0]


class TestPostprocessRapid:
    """RAPiD raw 3-scale output (sigmoid + anchor decode + axis-aligned NMS)."""

    def _make_scale_tensor(self, h: int, w: int) -> np.ndarray:
        # NHWC layout, 18 channels = 3 anchors * 6 (cx,cy,w,h,angle,conf).
        # Confidence channels are 5/11/17. Default to large negative logits
        # so sigmoid → ~0 and nothing fires unless we set a cell explicitly.
        arr = np.zeros((h, w, 18), dtype=np.float32)
        for anchor_idx in range(3):
            arr[..., anchor_idx * 6 + 5] = -10.0
        return arr

    def _set_cell(
        self,
        arr: np.ndarray,
        gy: int,
        gx: int,
        anchor_idx: int,
        *,
        tx: float = 0.0,
        ty: float = 0.0,
        tw: float = 0.0,
        th: float = 0.0,
        ta: float = 0.0,
        tc: float = 4.0,
    ) -> None:
        base = anchor_idx * 6
        arr[gy, gx, base + 0] = tx
        arr[gy, gx, base + 1] = ty
        arr[gy, gx, base + 2] = tw
        arr[gy, gx, base + 3] = th
        arr[gy, gx, base + 4] = ta
        arr[gy, gx, base + 5] = tc

    def _empty_scales(self) -> list[np.ndarray]:
        return [
            self._make_scale_tensor(128, 128),
            self._make_scale_tensor(64, 64),
            self._make_scale_tensor(32, 32),
        ]

    def test_invalid_input_returns_empty(self):
        assert postprocess_rapid([], 0.3, 0.45, 1.0, 0, 0, (1024, 1024)) == []
        assert (
            postprocess_rapid(None, 0.3, 0.45, 1.0, 0, 0, (1024, 1024)) == []
        )

    def test_all_below_threshold(self):
        scales = self._empty_scales()
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert dets == []

    def test_single_detection_decoded_center(self):
        # One positive cell at M-scale grid (32, 32), anchor 0:
        # decoded center = ((sigmoid(0) + 32) * 16, (sigmoid(0) + 32) * 16)
        # = (32.5*16, 32.5*16) = (520, 520).
        # Anchor 0 size at stride 16 = (45.07, 101.47). With ta=0,
        # angle = (0.5 - 0.5) * 360 = 0° → axis-aligned bbox = anchor.
        scales = self._empty_scales()
        self._set_cell(scales[1], 32, 32, anchor_idx=0, tc=4.0)
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert len(dets) == 1
        cx, cy = dets[0].centroid
        assert abs(cx - 520) < 2
        assert abs(cy - 520) < 2
        ax, ay = RAPID_ANCHORS[16][0]
        x1, y1, x2, y2 = dets[0].bbox
        assert abs((x2 - x1) - ax) < 2
        assert abs((y2 - y1) - ay) < 2

    def test_confidence_filtering(self):
        scales = self._empty_scales()
        # cell A: tc=4 → sigmoid ≈ 0.982 → keep
        self._set_cell(scales[1], 10, 10, anchor_idx=0, tc=4.0)
        # cell B: tc=-2 → sigmoid ≈ 0.119 → drop (below 0.3)
        self._set_cell(scales[1], 20, 20, anchor_idx=1, tc=-2.0)
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert len(dets) == 1
        assert dets[0].confidence > 0.9

    def test_letterbox_undo(self):
        scales = self._empty_scales()
        self._set_cell(scales[1], 32, 32, anchor_idx=0, tc=4.0)
        # If the 1024×1024 input was a letterboxed crop of an original
        # 2048×1152 frame, scale=0.5, pad_y=64 (no x-pad), the decoded
        # center (520, 520) maps back to ((520-0)/0.5, (520-64)/0.5) =
        # (1040, 912) in original coords.
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 0.5, 0, 64, (2048, 1152),
        )
        assert len(dets) == 1
        cx, cy = dets[0].centroid
        assert abs(cx - 1040) < 4
        assert abs(cy - 912) < 4

    def test_nchw_layout_accepted(self):
        # Some backends emit NCHW. The decoder should detect 18 in axis-0
        # and transpose internally.
        s = np.full((18, 128, 128), -10.0, dtype=np.float32)
        m = np.zeros((18, 64, 64), dtype=np.float32)
        for anchor_idx in range(3):
            m[anchor_idx * 6 + 5] = -10.0
        m[5, 32, 32] = 4.0  # anchor 0 conf at grid (32, 32)
        large = np.full((18, 32, 32), -10.0, dtype=np.float32)
        dets = postprocess_rapid(
            [s, m, large], 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert len(dets) == 1
        cx, cy = dets[0].centroid
        assert abs(cx - 520) < 2
        assert abs(cy - 520) < 2

    def test_batch_dim_stripped(self):
        # If the backend leaves a leading batch=1 dim, the decoder should
        # strip it.
        scales = self._empty_scales()
        self._set_cell(scales[2], 16, 16, anchor_idx=0, tc=4.0)
        scales[2] = scales[2][np.newaxis, ...]  # (1, 32, 32, 18)
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert len(dets) == 1

    def test_rotated_field_populated(self):
        """The decoder must preserve the rotated rectangle (cx, cy, w,
        h, angle_deg) before collapsing to the axis-aligned envelope.
        Downstream consumers (head_depth_in_bbox polygon mask) need
        the body-aligned shape; without it the axis-aligned envelope
        sweeps in floor + adjacent structure at non-trivial angles.
        """
        # Cell at M-scale grid (32, 32), anchor 0, ta=0 → angle = 0°.
        # Decoded center = (520, 520); anchor size = (45.07, 101.47).
        scales = self._empty_scales()
        self._set_cell(scales[1], 32, 32, anchor_idx=0, tc=4.0)
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert len(dets) == 1
        rotated = dets[0].rotated
        assert rotated is not None
        rcx, rcy, rw, rh, rang = rotated
        assert abs(rcx - 520) < 2
        assert abs(rcy - 520) < 2
        ax, ay = RAPID_ANCHORS[16][0]
        assert abs(rw - ax) < 2
        assert abs(rh - ay) < 2
        assert abs(rang) < 1e-3  # ta=0 → sigmoid(0)=0.5 → angle=0

    def test_rotated_field_letterbox_undone(self):
        """Letterbox unscales centre + size on the rotated record the
        same way it does on the axis-aligned bbox. Angle is invariant
        under uniform scale + translation, so it stays untouched.
        """
        scales = self._empty_scales()
        self._set_cell(scales[1], 32, 32, anchor_idx=0, tc=4.0)
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 0.5, 0, 64, (2048, 1152),
        )
        assert len(dets) == 1
        rotated = dets[0].rotated
        assert rotated is not None
        rcx, rcy, rw, rh, rang = rotated
        # cx = (520 - 0)/0.5 = 1040; cy = (520 - 64)/0.5 = 912.
        assert abs(rcx - 1040) < 4
        assert abs(rcy - 912) < 4
        # Anchor (45.07, 101.47) at scale=0.5 → (90.14, 202.94).
        ax, ay = RAPID_ANCHORS[16][0]
        assert abs(rw - ax / 0.5) < 4
        assert abs(rh - ay / 0.5) < 4
        # Angle untouched by letterbox.
        assert abs(rang) < 1e-3

    def test_rotated_angle_preserved(self):
        """A non-zero angle logit decodes through the same sigmoid +
        range as the rest of the head; the rotated record must reflect
        the actual orientation, not 0.
        """
        scales = self._empty_scales()
        # ta = 0.5 → sigmoid(0.5) ≈ 0.622 → angle = (0.622 - 0.5) * 360
        # ≈ 44° (we just check it's clearly non-zero).
        self._set_cell(scales[1], 32, 32, anchor_idx=0, ta=0.5, tc=4.0)
        dets = postprocess_rapid(
            scales, 0.3, 0.45, 1.0, 0, 0, (1024, 1024),
        )
        assert len(dets) == 1
        assert dets[0].rotated is not None
        _, _, _, _, rang = dets[0].rotated
        assert abs(rang) > 30.0


class TestPostprocessHailoNmsInputSize:
    def test_input_size_parametric(self):
        # Hailo NMS output: list of 80 per-class arrays. Person class has
        # one detection in normalized coords [y_min, x_min, y_max, x_max,
        # score].
        raw: list = [np.zeros((0, 5)) for _ in range(80)]
        raw[0] = np.array([[0.4, 0.4, 0.6, 0.6, 0.9]], dtype=np.float32)
        # input_size=1024 → normalized 0.5 maps to 512 px.
        dets = postprocess_hailo_nms(
            raw, 0.5, 1.0, 0, 0, (1024, 1024), input_size=(1024, 1024),
        )
        assert len(dets) == 1
        cx, cy = dets[0].centroid
        assert abs(cx - 512) < 2
        assert abs(cy - 512) < 2

    def test_default_input_size_640(self):
        raw: list = [np.zeros((0, 5)) for _ in range(80)]
        raw[0] = np.array([[0.4, 0.4, 0.6, 0.6, 0.9]], dtype=np.float32)
        # Default input_size=(640, 640) → normalized 0.5 maps to 320 px.
        dets = postprocess_hailo_nms(raw, 0.5, 1.0, 0, 0, (640, 640))
        assert len(dets) == 1
        cx, cy = dets[0].centroid
        assert abs(cx - 320) < 2


class TestArchitectureRegistry:
    def test_yolov8_present(self):
        assert "yolov8" in ARCHITECTURES
        assert ARCHITECTURES["yolov8"]["input_size"] == (640, 640)
        assert callable(ARCHITECTURES["yolov8"]["postprocess"])

    def test_rapid_present(self):
        assert "rapid" in ARCHITECTURES
        assert ARCHITECTURES["rapid"]["input_size"] == (1024, 1024)
        assert callable(ARCHITECTURES["rapid"]["postprocess"])

    def test_rapid_strides_match_anchor_keys(self):
        assert set(RAPID_STRIDES) == set(RAPID_ANCHORS.keys())


class TestClusterDetections:
    """Centroid clustering safety net for stock-YOLOv8 multi-firing in
    cenital geometry (head + torso + limbs of one person produce distinct
    boxes that NMS can't collapse)."""

    @staticmethod
    def _det(cx: float, cy: float, conf: float) -> Detection:
        # 20×20 bbox centered on (cx, cy).
        return Detection(
            bbox=(int(cx) - 10, int(cy) - 10, int(cx) + 10, int(cy) + 10),
            confidence=conf,
            centroid=(cx, cy),
        )

    def test_disabled_returns_input(self):
        dets = [self._det(100, 100, 0.9), self._det(105, 105, 0.8)]
        out = cluster_detections(dets, max_centroid_distance_px=0)
        assert len(out) == 2

    def test_empty_input(self):
        assert cluster_detections([], max_centroid_distance_px=50) == []

    def test_singleton_passes_through(self):
        d = self._det(100, 100, 0.9)
        assert cluster_detections([d], max_centroid_distance_px=50) == [d]

    def test_close_pair_collapses_to_highest_conf(self):
        a = self._det(100, 100, 0.85)
        b = self._det(120, 110, 0.92)  # ~22 px from a, higher conf
        out = cluster_detections([a, b], max_centroid_distance_px=50)
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.92)

    def test_far_pair_kept_separate(self):
        a = self._det(100, 100, 0.85)
        b = self._det(400, 400, 0.85)
        out = cluster_detections([a, b], max_centroid_distance_px=50)
        assert len(out) == 2

    def test_three_on_one_person_collapse_to_one(self):
        # Stock YOLOv8 on cenital firing on head + torso + limb of the
        # same person at (100, 100). Even though pairwise IoU is < 0.45
        # (different body parts), centroids cluster within 50 px.
        head = self._det(100, 100, 0.65)
        torso = self._det(120, 130, 0.55)
        limb = self._det(80, 130, 0.42)
        out = cluster_detections(
            [head, torso, limb], max_centroid_distance_px=50,
        )
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.65)

    def test_two_persons_stay_two_clusters(self):
        # Two physically distinct people far apart, each with a multi-fire
        # cluster. We should end up with exactly two detections, one per
        # person, each the highest-confidence representative.
        person_a = [
            self._det(100, 100, 0.7),
            self._det(120, 110, 0.6),
        ]
        person_b = [
            self._det(500, 500, 0.8),
            self._det(515, 510, 0.55),
        ]
        out = cluster_detections(
            person_a + person_b, max_centroid_distance_px=50,
        )
        assert len(out) == 2
        confs = sorted(d.confidence for d in out)
        assert confs == pytest.approx([0.7, 0.8])

    def test_chain_does_not_collapse_distant_endpoints(self):
        # Three detections in a chain: A--B--C where A↔B and B↔C are
        # within the gate but A↔C are not. Greedy from highest conf
        # picks B first, absorbs A and C in turn.
        a = self._det(100, 100, 0.5)
        b = self._det(140, 100, 0.9)  # 40 px from a, 40 px from c
        c = self._det(180, 100, 0.6)
        out = cluster_detections([a, b, c], max_centroid_distance_px=50)
        # B (highest conf) absorbs A then C since both are within 50 px
        # of B's centroid.
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.9)

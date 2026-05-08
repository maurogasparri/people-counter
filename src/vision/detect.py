"""Detección de personas con arquitecturas de modelo intercambiables.

Dos backends comparten el path de preproceso (letterbox + uint8 NHWC para
Hailo, o float32 NCHW para OpenCV):

  - hailo: Producción. Corre HEF sobre Hailo-8L vía SDK hailo_platform.
  - opencv: Desarrollo/testing. Corre ONNX vía OpenCV DNN.

El path de postproceso es **architecture-specific** y vive en el registry
``ARCHITECTURES`` debajo. Cada entry conoce su input size y su decoder.
Actualmente shippeamos dos:

  - yolov8: head COCO 80-class. Auto-detecta si el HEF emitió una lista
    Hailo-NMS de arrays o un tensor raw ``(1, 84, N)``.
  - rapid:  head rotated-bbox de Boston VIP, 1-class person. El HEF emite
    tres tensores raw per-scale (strides 8/16/32). El decode corre en CPU
    después de Hailo ya que la head rotada no está soportada por la unidad
    NMS on-chip.

Agregar una arquitectura nueva es un entry en ``ARCHITECTURES`` más un
postprocess callable con la misma signature.
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
# Resultado de detección
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """Una detección individual de persona."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) en coords de imagen original
    confidence: float
    centroid: tuple[float, float]  # (cx, cy) centro del bbox
    # Rectángulo rotado ajustado de las arquitecturas que emiten uno (RAPiD).
    # ``(cx, cy, w, h, angle_deg)`` en coordenadas de imagen original — mismo
    # frame que ``bbox``. ``None`` para detectores axis-aligned (yolov8).
    # Los consumers a los que les importa el footprint real del cuerpo (head
    # depth, máscaras) deberían preferir este cuando esté presente; ``bbox``
    # es el envelope axis-aligned ajustado y sobrepasa cuando el ángulo no
    # es trivial.
    rotated: tuple[float, float, float, float, float] | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "centroid": list(self.centroid),
        }
        if self.rotated is not None:
            d["rotated"] = list(self.rotated)
        return d


# ---------------------------------------------------------------------------
# Protocolo del backend del modelo
# ---------------------------------------------------------------------------


class DetectionBackend(Protocol):
    """Protocolo para los backends de detección."""

    def infer(self, preprocessed: np.ndarray) -> Any:
        """Corre inferencia sobre la entrada preprocesada.

        Args:
            preprocessed: ``(1, 3, H, W)`` float32 normalizado [0, 1].

        Returns:
            Salida raw del modelo. Forma y estructura dependen de la
            arquitectura (ver ``ARCHITECTURES``):

            - yolov8 + NMS built-in de Hailo → lista de 80 arrays per-class.
            - yolov8 raw → ndarray de forma ``(1, 84, N)``.
            - rapid raw → lista de 3 ndarrays per-scale.
        """
        ...


# ---------------------------------------------------------------------------
# Backend Hailo (producción — RPi5 + Hailo-8L)
# ---------------------------------------------------------------------------


class HailoBackend:
    """Backend de inferencia Hailo-8L usando el SDK hailo_platform.

    Usa la API VStream con activación persistente — el network group
    y el pipeline de inferencia quedan abiertos durante toda la vida
    del backend, evitando el overhead de setup/teardown per frame.

    Maneja tanto HEFs single-output (ej. YOLOv8 con NMS on-chip, donde
    el único output es la lista de detección per-class) como HEFs
    multi-output (ej. RAPiD raw, tres tensores per-scale).
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

        # VDevice compartido con scheduling round-robin (best practice Hailo)
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
        # Capturar el orden de outputs declarado por el HEF así modelos
        # multi-output decodean en orden determinístico (S, M, L para RAPiD).
        self._output_names = [
            info.name for info in self._hef.get_output_vstream_infos()
        ]

        # Activar el network group y abrir el pipeline de inferencia
        # persistente en vez de per-frame. Mantiene el contexto HW caliente.
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
        """Corre inferencia en Hailo-8L.

        El modelo HEF espera input uint8 NHWC. Este método maneja la
        conversión desde el blob float32 NCHW producido por preprocess().

        Returns:
            Para HEFs single-output, el output unwrappeado (batch dim
            stripped). Para HEFs multi-output, una lista de arrays
            per-output en el orden declarado del HEF.
        """
        # preprocess() saca (1, 3, H, W) float32 [0,1]
        # Hailo espera (1, H, W, 3) uint8 [0,255]
        if preprocessed.ndim == 4 and preprocessed.shape[1] == 3:
            preprocessed = preprocessed.transpose(0, 2, 3, 1)
        img = (preprocessed * 255).clip(0, 255).astype(np.uint8)

        result = self._pipeline.infer(
            {self._input_name: np.ascontiguousarray(img)}
        )

        if len(self._output_names) == 1:
            # Stripear batch dim; para YOLOv8-NMS esto da la lista de
            # 80 arrays per-class, para heads single-tensor raw da el
            # ndarray decodificado.
            return result[self._output_names[0]][0]
        return [result[name][0] for name in self._output_names]

    def close(self) -> None:
        """Libera los recursos de Hailo."""
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
# Backend OpenCV DNN (desarrollo — cualquier máquina con modelo ONNX)
# ---------------------------------------------------------------------------


class OpenCVBackend:
    """Backend de inferencia OpenCV DNN para modelos ONNX (CPU/GPU)."""

    def __init__(self, onnx_path: str) -> None:
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        # Preferir CUDA si está disponible, caer a CPU
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
        """Corre inferencia vía OpenCV DNN."""
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
    """Preprocesa un frame para inferencia del modelo letterboxed.

    Aplica resize letterbox manteniendo aspect ratio, después normaliza.

    Args:
        frame: Imagen BGR de cualquier tamaño.
        input_size: ``(width, height)`` target para el modelo.

    Returns:
        ``(blob, scale, pad_x, pad_y)`` donde:

        - ``blob``: ``(1, 3, H, W)`` float32 normalizado [0, 1].
        - ``scale``: Factor de escala aplicado durante el resize.
        - ``pad_x, pad_y``: Offsets de padding en pixels.
    """
    target_w, target_h = input_size
    h, w = frame.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding hasta el target size (center padding)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # HWC BGR → CHW RGB, normalizar a [0, 1]
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # Agregar batch dimension

    return blob, scale, pad_x, pad_y


# ---------------------------------------------------------------------------
# Postprocess: output raw YOLOv8 (ONNX vía OpenCV DNN)
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
    """Postprocesa output raw de YOLOv8 a detecciones de personas.

    Shape de output YOLOv8: ``(1, 84, 8400)`` donde:

    - 84 = 4 coords bbox (cx, cy, w, h) + 80 scores de clase COCO
    - 8400 = cantidad de anchors de predicción

    Args:
        raw_output: Tensor raw de output del modelo.
        confidence_threshold: Mínima confidence a mantener.
        nms_threshold: Threshold de IoU para NMS.
        scale: Factor de escala del preprocessing.
        pad_x, pad_y: Offsets de padding del preprocessing.
        original_size: ``(width, height)`` de la imagen original.

    Returns:
        Lista de objetos Detection solo para personas.
    """
    # Squeeze batch dimension si está presente
    if raw_output.ndim == 3:
        output = raw_output[0]  # (84, 8400)
    else:
        output = raw_output

    # Output YOLOv8 puede ser (84, 8400) o (8400, 84) según el export
    if output.shape[0] == 84:
        output = output.T  # → (8400, 84)

    # Extraer scores de la clase person (clase 0 en COCO)
    # Columnas: [cx, cy, w, h, class0_score, class1_score, ..., class79_score]
    person_scores = output[:, 4 + COCO_PERSON_CLASS]

    # Filtrar por confidence
    mask = person_scores >= confidence_threshold
    if not np.any(mask):
        return []

    filtered = output[mask]
    scores = person_scores[mask]

    # Convertir de cx, cy, w, h a x1, y1, x2, y2
    cx = filtered[:, 0]
    cy = filtered[:, 1]
    w = filtered[:, 2]
    h = filtered[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Remover padding y reescalar a coords de imagen original
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    orig_w, orig_h = original_size

    # Clipear a los bordes de la imagen
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

    # Construir objetos Detection
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
# Postprocess: YOLOv8 con NMS built-in de Hailo
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
    """Postprocesa output de NMS de Hailo a detecciones de personas.

    El output de NMS Hailo es una lista de 80 arrays (uno por cada
    clase COCO). Cada array tiene shape ``(N, 5)`` donde:

    - N: cantidad de detecciones para esa clase (variable por clase)
    - 5: ``[y_min, x_min, y_max, x_max, score]`` normalizado [0, 1]

    Args:
        raw_output: Lista de 80 arrays de la inferencia Hailo.
        confidence_threshold: Mínima confidence a mantener.
        scale: Factor de escala del preprocessing.
        pad_x, pad_y: Offsets de padding del preprocessing.
        original_size: ``(width, height)`` de la imagen original.
        input_size: Input ``(width, height)`` del modelo — se usa para
            deshacer las coords normalizadas. YOLOv8 entrenado a 640.

    Returns:
        Lista de objetos Detection solo para personas.
    """
    # Extraer la clase person (clase 0) — shape (N, 5)
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

        # Convertir de coords normalizadas a coords pixel del input
        x1 = x1_n * input_w
        y1 = y1_n * input_h
        x2 = x2_n * input_w
        y2 = y2_n * input_h

        # Remover padding y reescalar a la imagen original
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clipear a los bordes de la imagen
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
# Postprocess: output raw 3-scale de RAPiD
# ---------------------------------------------------------------------------

# Anchors RAPiD para los weights MWHB1024 (Boston VIP). Per-scale 3
# anchors, una fila cada uno = (anchor_w_px, anchor_h_px) al input
# nativo 1024×1024 del modelo. El HEF saca tensores raw per-scale (sin
# decode on-chip), así que reproducimos el decode original de PyTorch
# acá en CPU.
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
# Rango de ángulo usado por la head de RAPiD: output sigmoid mapeado a
# (-180°, 180°). La magnitud exacta no importa para nuestro output de
# bbox axis-aligned (el rectángulo envolvente tight es el mismo hasta
# un flip sign-symmetric), pero mantener la parametrización original
# evita sorpresas si alguien reusa este decoder después para NMS rotado.
_RAPID_ANGLE_RANGE_DEG = 360.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rapid_to_grid(
    arr: np.ndarray, expected_channels: int = 18,
) -> "np.ndarray | None":
    """Normaliza un tensor RAPiD per-scale a ``(H, W, n_anchors, 6)``.

    Hailo VStream devuelve NHWC por default; el path OpenCV puede
    devolver NCHW. Aceptamos ambos y cualquier batch dim leading.
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
    """Postprocesa output raw 3-scale de RAPiD a detecciones de personas.

    El HEF de RAPiD emite tres tensores per-scale (S/M/L = strides
    8/16/32), cada uno con 18 channels = ``3 anchors × (cx, cy, w, h,
    angle, conf)``. Hacemos sigmoid + decode de anchors en CPU,
    convertimos cada rectángulo rotado a su tight axis-aligned
    enclosing box, y corremos NMS plano sobre esos enclosing boxes —
    el tracker downstream solo consume bboxes axis-aligned, así que
    colapsamos la rotación acá.

    Args:
        raw_output: Lista de 3 ndarrays per-scale en orden HEF (S, M, L).
        confidence_threshold: Mínima confidence sigmoid a mantener.
        nms_threshold: Threshold de IoU para NMS axis-aligned.
        scale: Factor de escala letterbox.
        pad_x, pad_y: Offsets de padding letterbox.
        original_size: ``(width, height)`` de la imagen original.
        input_size: Input ``(W, H)`` del modelo — default 1024×1024.
    """
    if not isinstance(raw_output, (list, tuple)) or len(raw_output) != 3:
        return []

    all_boxes: list[np.ndarray] = []  # axis-aligned [x1, y1, x2, y2]
    all_scores: list[np.ndarray] = []
    all_rotated: list[np.ndarray] = []  # [cx, cy, w, h, angle_deg]

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

        # Construir offsets de cell-grid, broadcast a los 3 slots de anchor.
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

        # Bbox axis-aligned tight del rectángulo rotado: rotar los 4
        # offsets de esquina y tomar el min/max per-row. Vectorizado
        # sobre todas las celdas que sobrevivieron en este scale.
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
        # Preservar el rectángulo rotado (cx, cy, w, h, angle_deg)
        # antes del collapse axis-aligned — los consumers downstream
        # (head_depth_in_bbox) lo usan como máscara polígono para
        # rechazar pixels de fondo que el envelope axis-aligned
        # arrastra cuando la rotación es no trivial.
        all_rotated.append(
            np.stack([bx_f, by_f, bw_f, bh_f, ang_f], axis=1).astype(np.float32)
        )

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    rotated = np.concatenate(all_rotated, axis=0).astype(np.float32)

    # Deshacer el letterbox para volver a coords de imagen original.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    # Lo mismo para el centro + size rotado. El ángulo es invariante
    # bajo scale + translation uniforme, así que no se toca.
    rotated[:, 0] = (rotated[:, 0] - pad_x) / scale
    rotated[:, 1] = (rotated[:, 1] - pad_y) / scale
    rotated[:, 2] = rotated[:, 2] / scale
    rotated[:, 3] = rotated[:, 3] / scale

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
        rcx, rcy, rw, rh, rang = rotated[i]
        detections.append(
            Detection(
                bbox=(int(bx1), int(by1), int(bx2), int(by2)),
                confidence=float(scores[i]),
                centroid=((bx1 + bx2) / 2.0, (by1 + by2) / 2.0),
                rotated=(
                    float(rcx),
                    float(rcy),
                    float(rw),
                    float(rh),
                    float(rang),
                ),
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
    """Dispatchea el output yolov8 al decoder correcto.

    El HEF Hailo con NMS built-in da una lista per-class; el path
    OpenCV / HEF-raw da un único ndarray. Branchamos por tipo así la
    misma entry de arch cubre ambos.
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


# Cada entry bindea un name a su ``input_size`` esperado (para
# letterbox) y su callable ``postprocess``. Agregar un modelo nuevo =
# una entry acá + una función postprocess con la misma signature.
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
# Clustering por centroides (safety net post-NMS)
# ---------------------------------------------------------------------------


def cluster_detections(
    detections: list[Detection],
    max_centroid_distance_px: float,
) -> list[Detection]:
    """Clustering greedy por centroides — mergea detecciones cercanas.

    Mantiene la detección de mayor confidence en cada cluster. Dos
    detecciones están en el mismo cluster sii sus centroides están
    dentro de ``max_centroid_distance_px`` una de otra.

    Por qué esto sobre NMS: el estándar ``cv2.dnn.NMSBoxes`` colapsa
    bboxes overlapping, pero el stock YOLOv8 entrenado en side-views
    de COCO dispara múltiples cajas distintas sobre distintas partes
    del cuerpo (cabeza, torso, miembros) de la misma persona en
    geometría cenital. Sus IoUs pairwise de bbox son <0.45 así que
    NMS las mantiene — pero sus centroides están a un bbox-width.
    Centroid distance agarra ese caso donde IoU no puede.

    Pasar ``max_centroid_distance_px <= 0`` para desactivar (devuelve
    las detecciones sin cambios).
    """
    if max_centroid_distance_px <= 0 or len(detections) <= 1:
        return list(detections)

    threshold_sq = max_centroid_distance_px ** 2
    # Highest-confidence primero así el representante de cada cluster
    # es la detección más confiable.
    sorted_dets = sorted(detections, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for det in sorted_dets:
        cx, cy = det.centroid
        absorbed = False
        for k in kept:
            kx, ky = k.centroid
            if (cx - kx) ** 2 + (cy - ky) ** 2 < threshold_sq:
                absorbed = True
                break
        if not absorbed:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(
    model_path: str,
    backend: str = "auto",
    architecture: str = DEFAULT_ARCHITECTURE,
) -> dict[str, Any]:
    """Carga el modelo de detección.

    Args:
        model_path: Path al archivo del modelo HEF (Hailo) o ONNX (OpenCV).
        backend: ``"hailo"``, ``"opencv"``, o ``"auto"`` (basado en
            extensión de archivo).
        architecture: Selector de postprocess — tiene que ser una key
            de ``ARCHITECTURES``. Default yolov8.

    Returns:
        Dict con la instancia ``backend``, string ``type``, nombre
        ``architecture``, y tuple ``input_size``.
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
    cluster_distance_px: float = 0.0,
) -> list[Detection]:
    """Corre detección de personas sobre un único frame.

    Args:
        frame: Imagen BGR de cualquier tamaño.
        model: Dict de ``load_model()``.
        confidence_threshold: Mínima confidence para detecciones.
        nms_threshold: Threshold de IoU para NMS.
        cluster_distance_px: Si > 0, después de NMS corre un pass de
            clustering por centroides (``cluster_detections``) que
            mergea detecciones cuyos centroides están dentro de esta
            cantidad de pixels. Se usa para absorber el multi-firing
            del YOLOv8 stock sobre geometría cenital donde la misma
            persona spawnea cajas sobre cabeza + torso + miembros y
            NMS no puede colapsarlas porque los IoUs de bbox son muy
            bajos. Default 0 desactiva.

    Returns:
        Lista de objetos Detection (solo clase person).
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
    detections = postprocess_fn(
        raw_output,
        confidence_threshold,
        nms_threshold,
        scale,
        pad_x,
        pad_y,
        (orig_w, orig_h),
        input_size,
    )
    if cluster_distance_px > 0:
        detections = cluster_detections(detections, cluster_distance_px)
    return detections

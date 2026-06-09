"""Detección de personas — backend hailo (HEF) o opencv (ONNX).

Dos backends comparten el path de preproceso (letterbox + uint8 NHWC para
Hailo, o float32 NCHW para OpenCV):

  - hailo: Producción. Corre HEF sobre Hailo-8L vía SDK hailo_platform.
  - opencv: Desarrollo/testing. Corre ONNX vía OpenCV DNN.

El postproceso vive en el registry ``ARCHITECTURES`` debajo. Hoy solo
shippeamos ``yolov8`` (head COCO 80-class; auto-detecta si el HEF emitió
una lista Hailo-NMS de arrays o un tensor raw ``(1, 84, N)``). Agregar
una arquitectura nueva es un entry en ``ARCHITECTURES`` más un
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

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "centroid": list(self.centroid),
        }


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
    multi-output (tensores raw per-scale para postproceso CPU).
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
        # multi-output decodean en orden determinístico per-scale.
        self._output_names = [
            info.name for info in self._hef.get_output_vstream_infos()
        ]

        # Pipeline de inferencia persistente (vs per-frame). Mantiene el
        # contexto HW caliente. NO llamar a network_group.activate() —
        # con HailoSchedulingAlgorithm.ROUND_ROBIN el scheduler maneja la
        # activación automáticamente, y un activate() manual queda
        # deprecated en SDKs >=4.23 (raisea en futuras versiones).
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

        ``preprocess()`` ya entrega ``(1, H, W, 3)`` uint8 RGB contiguo — el
        formato nativo del HEF. No hay conversión per-frame (antes había un
        round-trip float32→uint8 que dominaba el costo).

        Returns:
            Para HEFs single-output, el output unwrappeado (batch dim
            stripped). Para HEFs multi-output, una lista de arrays
            per-output en el orden declarado del HEF.
        """
        result = self._pipeline.infer({self._input_name: preprocessed})

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
        """Corre inferencia vía OpenCV DNN.

        ``preprocess()`` entrega uint8 NHWC (nativo de Hailo); cv2.dnn quiere
        float32 NCHW normalizado, así que la conversión vive acá (path de dev,
        donde el costo no importa).
        """
        blob = preprocessed.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        self._net.setInput(blob)
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
        ``(img, scale, pad_x, pad_y)`` donde:

        - ``img``: ``(1, H, W, 3)`` uint8 RGB [0, 255] — el formato NATIVO del
          HEF Hailo. Antes se devolvía float32 NCHW normalizado, pero el
          backend Hailo lo deshacía (×255 + uint8 + transpose) cada frame: un
          round-trip de ~5MB en float32 que dominaba el costo de "detect". El
          backend OpenCV (dev) convierte a float32 NCHW en su ``infer``.
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

    # HWC BGR → NHWC RGB uint8 (formato nativo del HEF; sin float ni transpose).
    img = np.ascontiguousarray(padded[:, :, ::-1][np.newaxis, ...])

    return img, scale, pad_x, pad_y


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
}


# ---------------------------------------------------------------------------
# Clustering por centroides (safety net post-NMS)
# ---------------------------------------------------------------------------


def suppress_contained(
    detections: list[Detection],
    containment_threshold: float = 0.5,
) -> list[Detection]:
    """Suprime detecciones cuyo bbox está mayormente contenido en otra
    detección con confidence más alta.

    Por qué esto sobre NMS y cluster por centroide: en geometría
    cenital el detector dispara una caja chica sobre una parte del
    cuerpo (hombro, brazo, mano) que cae dentro de la caja grande de
    la persona. NMS estándar usa IoU pairwise — IoU es bajo cuando
    una caja es chica y la otra es grande aunque haya overlap total,
    así que la chica sobrevive. El cluster por centroide tampoco la
    agarra cuando las cajas son lo suficientemente grandes para que
    sus centroides queden a >cluster_distance_px. Containment
    (intersection / area_chica) cierra ese gap.

    Pasar ``containment_threshold <= 0`` para desactivar.
    """
    if containment_threshold <= 0 or len(detections) <= 1:
        return list(detections)

    sorted_dets = sorted(detections, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for det in sorted_dets:
        x1, y1, x2, y2 = det.bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            continue
        absorbed = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k.bbox
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            iw = ix2 - ix1
            ih = iy2 - iy1
            if iw <= 0 or ih <= 0:
                continue
            if (iw * ih) / area >= containment_threshold:
                absorbed = True
                break
        if not absorbed:
            kept.append(det)
    return kept


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

    threshold_sq = max_centroid_distance_px**2
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
        "input_size",
        ARCHITECTURES[architecture]["input_size"],
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
    # Containment suppression: corre siempre con threshold conservador.
    # Captura el caso geométrico de un bbox chico (hombro / mano /
    # parte de cuerpo) mayormente contenido en otro grande (cabeza /
    # persona completa) que NMS y cluster-por-centroide dejan pasar.
    detections = suppress_contained(detections, 0.5)
    return detections

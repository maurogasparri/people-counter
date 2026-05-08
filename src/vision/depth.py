"""Computación del mapa de disparidad a partir de pares estéreo rectificados.

Usa Semi-Global Block Matching (SGBM) según Hirschmuller (2008), con parámetros
tuneados para el par estéreo Arducam IMX708 montado en techo a un rango de
~1.3-6m y baseline 14cm.

La profundidad (Z) en cada pixel es: Z = f * B / disparity
donde f = focal length en pixels, B = baseline en mm.

Estimación de focal length para IMX708 120° HFOV a resolución completa:
  f_px ≈ 4608 / (2 * tan(60°)) ≈ 1330 px
  disparity a 3m  = 1330 * 140 / 3000 ≈ 62 px
  disparity a 1.3m = 1330 * 140 / 1300 ≈ 143 px
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dump de debug de profundidad — diagnóstico para el triage del bug de height.
# Cuando está habilitado, ``head_depth_in_bbox`` guarda hasta DEPTH_DEBUG_MAX_DUMPS
# PNGs side-by-side (heatmap de profundidad + máscaras en capas) bajo
# DEPTH_DEBUG_DIR y luego se auto-detiene. Barato cuando está apagado (un solo
# check bool). Usarlo para confirmar qué hay en la slice de head-depth elegida
# cuando el walk del histograma cae en algo que no debería (speckle SGBM sobre
# piso de parquet, estructura vecina, etc.).
#
# Toggle vía ``enable_depth_debug()`` — main.py lo wirea al flag CLI
# ``--depth-debug``.
# ---------------------------------------------------------------------------
DEPTH_DEBUG_DIR = "/tmp"
DEPTH_DEBUG_MAX_DUMPS = 5
_depth_debug_enabled = False
_depth_debug_count = 0


def enable_depth_debug(enabled: bool = True) -> None:
    """Toggle del dump diagnóstico de head_depth_in_bbox.

    Cuando está habilitado, las próximas ``DEPTH_DEBUG_MAX_DUMPS`` llamadas a
    ``head_depth_in_bbox`` que produzcan resultado escriben un PNG a
    ``DEPTH_DEBUG_DIR`` y loguean un resumen de histograma. Las llamadas
    subsiguientes son no-op hasta que el counter se resetee (llamar de nuevo
    con True tras un restart de proceso, o llamar a ``reset_depth_debug()``
    para re-armar).
    """
    global _depth_debug_enabled
    _depth_debug_enabled = bool(enabled)


def reset_depth_debug() -> None:
    """Resetea el counter de dumps para que las próximas
    ``DEPTH_DEBUG_MAX_DUMPS`` llamadas vuelvan a disparar. Útil para tests y
    re-armar mid-process."""
    global _depth_debug_count
    _depth_debug_count = 0


# Parámetros SGBM para Arducam IMX708 120° HFOV, baseline 14cm, rango 1.3-6m.
DEFAULT_NUM_DISPARITIES = 192  # Cubre rango de disparidad hasta ~143px a 1.3m
DEFAULT_BLOCK_SIZE = 9  # Matching robusto en imágenes wide-angle
DEFAULT_P1_FACTOR = 12  # Penalidad de smoothness para cambio de disparidad ±1
DEFAULT_P2_FACTOR = 96  # Penalidad para discontinuidades grandes (8× P1)
DEFAULT_DISP12_MAX_DIFF = 2  # Permite mismatch left-right de ±2px
DEFAULT_UNIQUENESS_RATIO = 5  # Más bajo para cámaras con filtro IR y buen contraste
DEFAULT_SPECKLE_WINDOW_SIZE = 150  # Filtra blobs chicos de ruido
DEFAULT_SPECKLE_RANGE = 16  # Variación máxima de disparidad dentro de un speckle
DEFAULT_PRE_FILTER_CAP = 63
DEFAULT_MIN_DISPARITY = 0


def create_sgbm(
    num_disparities: int = DEFAULT_NUM_DISPARITIES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    p1_factor: int = DEFAULT_P1_FACTOR,
    p2_factor: int = DEFAULT_P2_FACTOR,
    disp12_max_diff: int = DEFAULT_DISP12_MAX_DIFF,
    uniqueness_ratio: int = DEFAULT_UNIQUENESS_RATIO,
    speckle_window_size: int = DEFAULT_SPECKLE_WINDOW_SIZE,
    speckle_range: int = DEFAULT_SPECKLE_RANGE,
    pre_filter_cap: int = DEFAULT_PRE_FILTER_CAP,
    min_disparity: int = DEFAULT_MIN_DISPARITY,
) -> cv2.StereoSGBM:
    """Crea un matcher SGBM configurado.

    Args:
        num_disparities: Disparidad máxima menos disparidad mínima.
            Debe ser divisible por 16.
        block_size: Tamaño del bloque de matching. Debe ser impar.
        p1_factor: Multiplicador de la penalidad de smoothness P1.
        p2_factor: Multiplicador de la penalidad de smoothness P2 (debe ser > p1_factor).
        disp12_max_diff: Diferencia máxima permitida en el check left-right de disparidad.
        uniqueness_ratio: Margen (%) por el cual el mejor match debe superar al segundo.
        speckle_window_size: Área máxima del componente conexo a filtrar.
        speckle_range: Variación máxima de disparidad dentro de un componente conexo.
        pre_filter_cap: Valor de truncation para los pixels pre-filtrados.
        min_disparity: Valor mínimo de disparity (usualmente 0).

    Returns:
        Instancia configurada de StereoSGBM.
    """
    channels = 1
    p1 = p1_factor * channels * block_size * block_size
    p2 = p2_factor * channels * block_size * block_size

    sgbm = cv2.StereoSGBM.create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm


def _to_gray(image: np.ndarray, use_green_channel: bool = False) -> np.ndarray:
    """Convierte una imagen a grayscale.

    Args:
        image: Imagen BGR o grayscale.
        use_green_channel: Si True, extrae solo el channel verde.
    """
    if len(image.shape) != 3:
        return image
    if use_green_channel:
        return image[:, :, 1]  # Verde en BGR
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_disparity(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    num_disparities: int = DEFAULT_NUM_DISPARITIES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sgbm: cv2.StereoSGBM | None = None,
    use_wls_filter: bool = False,
    use_green_channel: bool = False,
    use_clahe: bool = True,
    downscale: int = 1,
) -> np.ndarray:
    """Computa el disparity map usando SGBM.

    La calibración es resolución-independiente (K, D no cambian), así
    que podemos matchear a menor resolución para velocidad y reducción
    de ruido, después upscalear el disparity map. Los valores de
    disparity se reescalan para que correspondan a la resolución
    original.

    Args:
        left_rect: Imagen izquierda rectificada (BGR o grayscale).
        right_rect: Imagen derecha rectificada (BGR o grayscale).
        num_disparities: Rango máximo de disparity. Ignorado si se
            pasa sgbm.
        block_size: Tamaño de bloque para matching. Ignorado si se
            pasa sgbm.
        sgbm: Matcher SGBM pre-creado. Si None, se crea uno.
        use_wls_filter: Aplica filtro WLS para smoothing edge-preserving.
        use_green_channel: Usa solo el channel verde.
        use_clahe: Aplica enhancement de contraste CLAHE antes del matching.
        downscale: Factor para reducir resolución antes del matching
            (1=full, 2=half, 4=quarter). La disparity se upscalea de
            vuelta y los valores se multiplican por el factor así los
            cálculos de depth siguen siendo correctos. Más alto =
            más rápido pero menos detalle.

    Returns:
        Disparity map como float32 en pixels (a la resolución original).
        Los pixels inválidos son -1.0.
    """
    gray_l = _to_gray(left_rect, use_green_channel)
    gray_r = _to_gray(right_rect, use_green_channel)

    if downscale > 1:
        gray_l = cv2.resize(
            gray_l,
            (gray_l.shape[1] // downscale, gray_l.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )
        gray_r = cv2.resize(
            gray_r,
            (gray_r.shape[1] // downscale, gray_r.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    if sgbm is None:
        # Escalar numDisparities hacia abajo para resolución reducida
        nd = (
            max(16, (num_disparities // downscale // 16) * 16)
            if downscale > 1
            else num_disparities
        )
        sgbm = create_sgbm(num_disparities=nd, block_size=block_size)

    raw_disp_l = sgbm.compute(gray_l, gray_r)

    if use_wls_filter:
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        raw_disp_r = right_matcher.compute(gray_r, gray_l)

        wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)
        wls.setLambda(4000.0)
        wls.setSigmaColor(1.0)

        filtered = wls.filter(raw_disp_l, gray_l, disparity_map_right=raw_disp_r)
        disparity = filtered.astype(np.float32) / 16.0
    else:
        disparity = raw_disp_l.astype(np.float32) / 16.0

    disparity[disparity < 0] = -1.0

    if downscale > 1:
        # Upscalear disparity a la resolución original. Los valores
        # de disparity escalan linealmente con la resolución, así
        # que multiplicamos por el factor de downscale: Z = fx * B / d,
        # y tanto fx como d escalan juntos, pero necesitamos d a la
        # resolución original.
        orig_h, orig_w = left_rect.shape[:2]
        valid_mask = disparity > 0
        disparity = cv2.resize(
            disparity,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        upscaled_mask = cv2.resize(
            valid_mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        disparity[upscaled_mask] *= downscale
        disparity[~upscaled_mask] = -1.0

    return disparity


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_mm: float,
    min_depth_mm: float = 500.0,
    max_depth_mm: float = 10000.0,
) -> np.ndarray:
    """Convierte un disparity map a depth map en milímetros.

    Z = f * B / d

    Args:
        disparity: Disparity map de compute_disparity() (float32, pixels).
        focal_length_px: Focal length en pixels (de calibración P1[0,0]).
        baseline_mm: Baseline estéreo en mm.
        min_depth_mm: Profundidad mínima válida.
        max_depth_mm: Profundidad máxima válida.

    Returns:
        Depth map en mm como float32. Los pixels inválidos son 0.0.
    """
    depth = np.zeros_like(disparity)

    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_mm) / disparity[valid]

    out_of_range = (depth < min_depth_mm) | (depth > max_depth_mm)
    depth[out_of_range & valid] = 0.0
    depth[~valid] = 0.0

    return depth


def depth_at_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    percentile: float = 50.0,
) -> float:
    """Estima la profundidad de una persona detectada a partir de su bbox.

    Usa la mediana (o el percentile especificado) de los valores de
    depth válidos dentro del 50% central del bbox.

    Args:
        depth_map: Depth map en mm de disparity_to_depth().
        bbox: (x1, y1, x2, y2) bounding box en coordenadas pixel.
        percentile: Percentile de valores de depth a usar (50 = mediana).

    Returns:
        Profundidad estimada en mm. Devuelve 0.0 si no hay pixels
        válidos en el ROI.
    """
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    hw = (x2 - x1) // 4
    hh = (y2 - y1) // 4

    roi_x1 = max(0, cx - hw)
    roi_y1 = max(0, cy - hh)
    roi_x2 = min(depth_map.shape[1], cx + hw)
    roi_y2 = min(depth_map.shape[0], cy + hh)

    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    valid = roi[roi > 0]

    if len(valid) == 0:
        return 0.0

    return float(np.percentile(valid, percentile))


def head_depth_in_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    mounting_height_mm: float,
    fx_px: float,
    cx_px: float,
    cy_px: float,
    *,
    rotated_bbox: Optional[tuple[float, float, float, float, float]] = None,
    slice_thickness_mm: float = 100.0,
    min_head_area_px: int = 40,
    max_head_height_mm: float = 1800.0,
    min_head_above_floor_mm: float = 500.0,
    column_radius_mm: float = 250.0,
    debug_frame: Optional[np.ndarray] = None,
    debug_confidence: Optional[float] = None,
) -> Optional[float]:
    """Encuentra la depth de la cabeza dentro de un bbox usando histograma +
    connected components, pre-filtrado por una columna vertical anclada
    en el centroide 3D del cuerpo.

    El bbox 2D es el único anchor que nos da el detector, pero en
    geometría cenital el bbox a menudo recorta estructura overhead
    (cables, cornisas, bordes de estantes) que está a un (X, Y) métrico
    distinto al del cuerpo abajo. Cuando el cuerpo ocluye parcialmente
    esa estructura el gate funciona bien; cuando el cuerpo se mueve
    (se sienta, se agacha) y la estructura se vuelve visible, el
    histogram-walk-from-nearest alegremente elige la estructura como
    "head" porque realmente está más cerca de la cámara que la cabeza
    real.

    El fix es anclar en una estimación chica del centroide métrico
    del cuerpo (X_center, Y_center) y rechazar pixels cuyo (X, Y)
    métrico back-proyectado cae fuera de una columna vertical de
    radio ``column_radius_mm`` alrededor de él. Las cabezas reales
    siempre caen dentro de esa columna (la cabeza es parte del cuerpo
    que produjo el bbox). La estructura overhead offset en X o Y
    métrico queda excluida.

    Para estéreo zenith-mount con un bbox YOLO-COCO de persona el
    centroide cae sobre el torso, no la cabeza, así que ni la
    mediana del crop central (mide la superficie del torso) ni un
    percentile bajo del depth del bbox (gana single-pixel speckle)
    dan una lectura confiable de altura de cabeza. Después del
    filtro espacial de columna corremos el pick canónico de
    histogram walk + connected-components: caminar la distribución
    de depth desde lo más cercano a la cámara hacia afuera, devolver
    la mediana del cluster conexo de pixels más grande en el primer
    slice de 10 cm que tenga área suficiente como para ser una cabeza
    real.

    Referencia: Del Pizzo et al., "Counting people by RGB or depth
    overhead cameras", Pattern Recognition Letters 2016 (histogram +
    components). El gate espacial de columna es nuestra extensión —
    se necesita porque la estructura overhead produce depth slices
    que pasan el gate antropométrico pero no están cerca del cuerpo
    en el world frame.

    Args:
        depth_map: Depth map en mm de disparity_to_depth(). 0 = inválido.
        bbox: (x1, y1, x2, y2) bounding box en coordenadas pixel.
        mounting_height_mm: Distancia camera-to-floor.
        fx_px: Focal length rectificado en px (P1[0, 0]). Se usa para
            back-proyectar pixels a X, Y métrico. Usamos el mismo fx
            para X e Y porque el modelo rectificado es de pixel
            cuadrado y solo la escala horizontal entra en la
            conversión depth-to-world.
        cx_px: Principal point x rectificado en px (P1[0, 2]).
        cy_px: Principal point y rectificado en px (P1[1, 2]).
        rotated_bbox: Rectángulo rotado opcional ``(cx, cy, w, h,
            angle_deg)`` en coordenadas pixel de imagen. Cuando se
            provee, el sampling de depth se restringe a los pixels
            dentro de este polígono — estrictamente más tight que el
            ``bbox`` axis-aligned, cuyo envelope del rectángulo rotado
            infla el área hasta 2× a 45° de rotación. Crucial para
            geometría cenital donde RAPiD emite un rectángulo rotado
            body-aligned pero ``bbox`` se desborda al piso +
            estructura vecina que pasan el depth gate antropométrico.
            Pasar ``None`` (default) para detectores axis-aligned
            (yolov8) donde ``bbox`` ya es el envelope tight.
        slice_thickness_mm: Ancho de bin del histograma. 10 cm es
            más ancho que la cuantización de depth de SGBM en el
            rango operativo, así que una cabeza real llena al menos
            un bin sólidamente.
        min_head_area_px: Cantidad mínima de pixels para que un slice
            califique como cluster de cabeza. Tuneado para el
            disparity grid en el combo runtime resolution + downscale
            — chico suficiente para agarrar un niño pero grande
            suficiente para rechazar clusters de speckle.
        max_head_height_mm: Techo antropométrico de altura de cabeza;
            depths debajo de ``mount - max_head_height`` se rechazan
            como imposibles.
        min_head_above_floor_mm: Piso antropométrico (ignorar depths
            que corresponden a una cabeza esencialmente en el suelo —
            ese es el piso mismo, no una persona).
        column_radius_mm: Radio (mm) de la columna vertical alrededor
            del centroide 3D del cuerpo dentro del cual los pixels
            tienen que caer para ser considerados material candidato
            de cabeza. Default 250 mm: una cabeza humana cae bien
            dentro de una columna de 50 cm de ancho centrada en el
            torso, mientras la estructura overhead offset por >25 cm
            en X o Y métrico se rechaza. Setear muy grande para
            desactivar (efectivamente revierte al comportamiento
            pre-filtro espacial).

    Returns:
        Depth estimada de la cabeza en mm, o None si no se encuentra
        cluster plausible de cabeza. Usar ``head_height_above_floor()``
        para convertir a altura.
    """
    if depth_map.size == 0:
        return None
    if fx_px <= 0.0:
        return None
    h, w = depth_map.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_map[y1:y2, x1:x2]

    # Gate antropométrico: las cabezas viven en
    # [mount-max_head, mount-min_above_floor].
    near = mounting_height_mm - max_head_height_mm
    far = mounting_height_mm - min_head_above_floor_mm
    if far <= near:
        return None

    valid = (roi > 0) & (roi >= near) & (roi <= far)
    if int(valid.sum()) < min_head_area_px:
        return None

    # ---- Máscara de rectángulo rotado (detectores estilo RAPiD) ----
    # El ``bbox`` axis-aligned es el envelope tight de un rectángulo
    # rotado para arquitecturas como RAPiD. En ángulos no triviales
    # el envelope arrastra piso + estructura vecina; en geometría
    # cenital esa estructura puede caer a depths dentro del gate
    # antropométrico (borde de escritorio, respaldo de silla, gabinete
    # alto al margen del bbox) y el histogram-walk-from-nearest lo
    # elige como "cabeza". Filtrar ``valid`` contra el polígono rotado
    # le da a la función un ROI con forma de cuerpo incluso cuando el
    # bbox axis-aligned es 2× el área del cuerpo real.
    if rotated_bbox is not None:
        rcx, rcy, rw, rh, rang_deg = rotated_bbox
        rang = float(np.deg2rad(rang_deg))
        cos_a, sin_a = float(np.cos(rang)), float(np.sin(rang))
        # 4 esquinas del rectángulo rotado en coords de imagen original.
        dx = np.array(
            [-rw / 2.0, rw / 2.0, rw / 2.0, -rw / 2.0], dtype=np.float32
        )
        dy = np.array(
            [-rh / 2.0, -rh / 2.0, rh / 2.0, rh / 2.0], dtype=np.float32
        )
        corners_x = rcx + dx * cos_a - dy * sin_a
        corners_y = rcy + dx * sin_a + dy * cos_a
        # Translate a coords ROI-locales (origen en bbox top-left).
        # Redondear a int una sola vez — fillPoly toma int32.
        poly = np.empty((4, 2), dtype=np.int32)
        poly[:, 0] = np.round(corners_x - x1).astype(np.int32)
        poly[:, 1] = np.round(corners_y - y1).astype(np.int32)
        poly_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 1)
        valid = valid & poly_mask.astype(bool)
        if int(valid.sum()) < min_head_area_px:
            return None

    # ---- Back-projection 3D de cada pixel gateado en el ROI ----
    # Construir grids (u, v) de pixel-centre cubriendo el bbox en
    # coords full-frame así cx/cy referencian el mismo origen que
    # P1. Z es el depth gateado; X, Y vienen de la back-projection
    # pinhole estándar
    # X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fx
    # (mismo fx para ambos ejes — los pixels rectificados son
    # cuadrados).
    us = np.arange(x1, x2, dtype=np.float32) - float(cx_px)
    vs = np.arange(y1, y2, dtype=np.float32) - float(cy_px)
    u_grid, v_grid = np.meshgrid(us, vs)
    z_full = roi.astype(np.float32)
    inv_fx = np.float32(1.0 / fx_px)
    x_full = u_grid * z_full * inv_fx
    y_full = v_grid * z_full * inv_fx

    # ---- Centroide 3D del cuerpo dentro del bbox ----
    # La mediana sobre el crop central gateado es robusta a la
    # estructura overhead que motiva el column filter — la estructura
    # típicamente solo besa el bbox en el tope/bordes, mientras el
    # cuerpo llena el crop central. Usar el centro significa que
    # anclamos en el cuerpo incluso cuando la estructura domina el
    # count de área gateada a lo largo del slice nearest del
    # histograma.
    bw, bh = x2 - x1, y2 - y1
    cw = max(1, bw // 2)
    ch = max(1, bh // 2)
    cx_off = (bw - cw) // 2
    cy_off = (bh - ch) // 2
    central_valid = valid[cy_off : cy_off + ch, cx_off : cx_off + cw]
    if int(central_valid.sum()) >= 4:
        x_center = float(
            np.median(x_full[cy_off : cy_off + ch, cx_off : cx_off + cw][central_valid])
        )
        y_center = float(
            np.median(y_full[cy_off : cy_off + ch, cx_off : cx_off + cw][central_valid])
        )
    else:
        # El crop central tiene muy pocos pixels válidos — caer al
        # set gateado completo así la función igual devuelve algo
        # para bboxes finos o depth maps con muchos huecos.
        x_center = float(np.median(x_full[valid]))
        y_center = float(np.median(y_full[valid]))

    # ---- Filtro espacial de columna ----
    # Mantener solo pixels gateados cuyo (X, Y) está dentro de
    # column_radius_mm del centroide del cuerpo. Comparar distancia
    # al cuadrado evita el sqrt.
    dx = x_full - x_center
    dy = y_full - y_center
    column_mask = (dx * dx + dy * dy) <= (column_radius_mm * column_radius_mm)
    valid_col = valid & column_mask
    if int(valid_col.sum()) < min_head_area_px:
        return None

    # Despeckle: median blur sobre el ROI gateado por columna. Usar
    # NaN para los inválidos así el blur no smearea depth a través
    # de los huecos. Caer a mediana plana si medianBlur no es
    # aplicable al dtype/shape (paranoia).
    roi_f = np.where(valid_col, roi.astype(np.float32), np.nan)
    try:
        smooth = cv2.medianBlur(roi_f, 3)
    except cv2.error:
        smooth = roi_f
    valid_s = np.isfinite(smooth)
    if int(valid_s.sum()) < min_head_area_px:
        return None

    d = smooth[valid_s]
    bin_edges = np.arange(near, far + slice_thickness_mm, slice_thickness_mm)
    if len(bin_edges) < 2:
        return float(np.median(d))
    hist, edges = np.histogram(d, bins=bin_edges)

    # Caminar desde lo más cercano a la cámara hacia afuera; el primer
    # slice con >= min_head_area_px es la cabeza.
    candidates = np.where(hist >= min_head_area_px)[0]
    if len(candidates) == 0:
        return None
    bin_idx = int(candidates[0])
    d_lo = float(edges[bin_idx])
    d_hi = float(edges[bin_idx + 1])

    # Connected components dentro del slice de cabeza. El blob más
    # grande es la cabeza; blobs más chicos en el mismo depth slice
    # suelen ser ruido.
    head_mask = ((smooth >= d_lo) & (smooth < d_hi)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(head_mask, connectivity=4)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[biggest, cv2.CC_STAT_AREA]) < min_head_area_px:
        return None

    blob_pixels = smooth[lbl == biggest]
    if blob_pixels.size == 0:
        return None
    head_depth_mm = float(np.median(blob_pixels))

    # Dump diagnóstico — gateado por toggle, auto-stop después de MAX_DUMPS.
    if _depth_debug_enabled:
        _dump_depth_debug(
            roi=roi,
            valid=valid,
            valid_col=valid_col,
            head_blob_mask=(lbl == biggest),
            bbox=(x1, y1, x2, y2),
            rotated_bbox=rotated_bbox,
            near_mm=near,
            far_mm=far,
            d_lo=d_lo,
            d_hi=d_hi,
            head_depth_mm=head_depth_mm,
            mounting_height_mm=mounting_height_mm,
            confidence=debug_confidence,
            x_center=x_center,
            y_center=y_center,
            hist=hist,
            edges=edges,
            debug_frame=debug_frame,
        )

    return head_depth_mm


def _dump_depth_debug(
    *,
    roi: np.ndarray,
    valid: np.ndarray,
    valid_col: np.ndarray,
    head_blob_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    rotated_bbox: Optional[tuple[float, float, float, float, float]],
    near_mm: float,
    far_mm: float,
    d_lo: float,
    d_hi: float,
    head_depth_mm: float,
    mounting_height_mm: float,
    confidence: Optional[float],
    x_center: float,
    y_center: float,
    hist: np.ndarray,
    edges: np.ndarray,
    debug_frame: Optional[np.ndarray],
) -> None:
    """Guarda un PNG compuesto de 3 paneles (frame | depth heatmap | máscaras)
    para triage.

    Cuando se provee ``debug_frame`` (el frame izquierdo rectificado
    de main.py), el primer panel muestra el crop del bbox con
    polígono rotado + overlay de texto así la escena visual se
    matchea contra el análisis de depth side-by-side en una única
    imagen.

    Auto-desactiva después de ``DEPTH_DEBUG_MAX_DUMPS`` para mantener
    bounded el uso de disco. Una línea de log por dump resume el
    slice elegido + el histograma así el diagnóstico sigue siendo
    útil incluso sin el PNG.
    """
    global _depth_debug_count
    if _depth_debug_count >= DEPTH_DEBUG_MAX_DUMPS:
        return
    _depth_debug_count += 1
    idx = _depth_debug_count

    try:
        x1, y1, x2, y2 = bbox
        h, w = roi.shape[:2]

        # ---- Panel 2: heatmap de depth del ROI ----
        # Normalizado al rango antropométrico así la banda de cabeza
        # se destaca visualmente. Pixels fuera de [near, far] (piso,
        # cielo, inválidos) → negro.
        norm = np.zeros_like(roi, dtype=np.float32)
        in_range = (roi >= near_mm) & (roi <= far_mm)
        if far_mm > near_mm:
            norm[in_range] = (roi[in_range] - near_mm) / (far_mm - near_mm)
        norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        heatmap[roi == 0] = 0  # inválido → negro

        # ---- Panel 3: máscaras en capas ----
        # Azul = rango antropométrico; verde = después del column
        # filter; rojo = blob de cabeza elegido.
        masks = np.zeros((h, w, 3), dtype=np.uint8)
        masks[valid] = (180, 80, 0)         # B (antropométrico)
        masks[valid_col] = (0, 200, 0)      # G (column filter)
        masks[head_blob_mask] = (0, 0, 255) # R (blob de cabeza elegido)

        # ---- Panel 1: crop de frame con bbox + polígono rotado + texto ----
        # Construido solo cuando main.py nos pasó el frame rectificado.
        if debug_frame is not None:
            fh, fw = debug_frame.shape[:2]
            fx1 = max(0, int(x1))
            fy1 = max(0, int(y1))
            fx2 = min(fw, int(x2))
            fy2 = min(fh, int(y2))
            crop = debug_frame[fy1:fy2, fx1:fx2].copy()
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            elif crop.shape[2] == 1:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            # Resize para matchear el tamaño del panel del heatmap
            # si el cropping clipeó.
            if crop.shape[:2] != (h, w):
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

            # Dibujar el polígono rotado (el footprint del cuerpo
            # detectado por RAPiD) en verde; envelope axis-aligned
            # en blanco para contraste.
            cv2.rectangle(
                crop, (0, 0), (w - 1, h - 1), (255, 255, 255), 1,
            )
            if rotated_bbox is not None:
                rcx, rcy, rw, rh, rang_deg = rotated_bbox
                rang = float(np.deg2rad(rang_deg))
                cos_a, sin_a = float(np.cos(rang)), float(np.sin(rang))
                dx = np.array(
                    [-rw / 2.0, rw / 2.0, rw / 2.0, -rw / 2.0],
                    dtype=np.float32,
                )
                dy = np.array(
                    [-rh / 2.0, -rh / 2.0, rh / 2.0, rh / 2.0],
                    dtype=np.float32,
                )
                corners_x = rcx + dx * cos_a - dy * sin_a
                corners_y = rcy + dx * sin_a + dy * cos_a
                # Translate a coords crop-locales (origen en bbox top-left).
                poly = np.empty((4, 2), dtype=np.int32)
                poly[:, 0] = np.round(corners_x - x1).astype(np.int32)
                poly[:, 1] = np.round(corners_y - y1).astype(np.int32)
                cv2.polylines(crop, [poly], True, (0, 255, 0), 2)

            # Marcar el centroide del blob de cabeza elegido en el
            # crop con una cruz roja — pista visual principal:
            # matchea con lo que está en el blob en la escena real.
            ys, xs = np.where(head_blob_mask)
            if ys.size > 0:
                cv_crop_h, cv_crop_w = crop.shape[:2]
                blob_cx = int(np.round(xs.mean() * cv_crop_w / w))
                blob_cy = int(np.round(ys.mean() * cv_crop_h / h))
                cv2.drawMarker(
                    crop, (blob_cx, blob_cy), (0, 0, 255),
                    cv2.MARKER_CROSS, 20, 2,
                )

            # Overlay de texto (esquina top-left, dos líneas así
            # queda legible en bboxes chicos). Blanco sobre fondo
            # negro para contraste en cualquier imagen.
            height_m = (mounting_height_mm - head_depth_mm) / 1000.0
            line_a = f"d={head_depth_mm/1000.0:.2f}m h={height_m:.2f}m"
            line_b = f"bin=[{int(d_lo)}-{int(d_hi)}]"
            if confidence is not None:
                line_b = f"{line_b} conf={confidence:.2f}"
            for i, txt in enumerate((line_a, line_b)):
                y_off = 16 + i * 18
                cv2.putText(
                    crop, txt, (4, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    crop, txt, (4, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
            panel = np.concatenate([crop, heatmap, masks], axis=1)
        else:
            panel = np.concatenate([heatmap, masks], axis=1)

        path = os.path.join(
            DEPTH_DEBUG_DIR,
            f"depth_debug_{idx:02d}_d{int(head_depth_mm)}mm.png",
        )
        cv2.imwrite(path, panel)

        # Dump de histograma: bin (rango mm) → pixel count, solo para
        # bins que contienen pixels. Apunta a qué depth bin ganó.
        hist_summary = ", ".join(
            f"[{int(edges[i])}-{int(edges[i + 1])}]={int(hist[i])}"
            for i in range(len(hist))
            if hist[i] > 0
        )
        logger.info(
            "depth_debug dump=%d/%d path=%s bbox=%s "
            "head_depth_mm=%.0f picked_bin=[%.0f,%.0f] "
            "centroid_xy=(%.0f,%.0f) hist=%s",
            idx,
            DEPTH_DEBUG_MAX_DUMPS,
            path,
            tuple(int(v) for v in bbox),
            head_depth_mm,
            d_lo,
            d_hi,
            x_center,
            y_center,
            hist_summary,
        )
    except Exception as e:  # pragma: no cover — diagnóstico, nunca debe crashear runtime
        logger.warning("depth_debug_dump_failed err=%s", e)


def min_depth_at_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    low_percentile: float = 15.0,
) -> float:
    """Estima la profundidad del punto más cercano en un bbox (tope
    de la cabeza para estéreo ceiling-mounted).

    Sample del 50% central del bbox (matcheando ``depth_at_bbox``):
    en geometría cenital la cabeza cae cerca del centroide del bbox,
    así que el crop central excluye el ruido SGBM periférico de
    pixels de fondo que caen casualmente dentro del bbox de persona
    del YOLO stock. Después toma un percentile bajo (default 15%)
    para mantenerse robusto a outliers de speckle — el min raw o el
    percentile 5 son demasiado fáciles de tirar abajo de la depth
    real de cabeza con factores altos de downscale SGBM.

    Args:
        depth_map: Depth map en mm de disparity_to_depth().
        bbox: (x1, y1, x2, y2) bounding box en coordenadas pixel.
        low_percentile: Percentile para el valor "near". 15 es el
            piso empírico bajo el cual los clusters de speckle en
            el output SGBM empiezan a dominar a downscale=8.

    Returns:
        Near-depth estimada en mm. Devuelve 0.0 si no hay pixels
        válidos.
    """
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    hw = (x2 - x1) // 4
    hh = (y2 - y1) // 4

    roi_x1 = max(0, cx - hw)
    roi_y1 = max(0, cy - hh)
    roi_x2 = min(depth_map.shape[1], cx + hw)
    roi_y2 = min(depth_map.shape[0], cy + hh)

    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    valid = roi[roi > 0]

    if len(valid) == 0:
        return 0.0

    return float(np.percentile(valid, low_percentile))

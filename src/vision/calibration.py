"""Calibración estéreo usando patrones ChArUco.

Usa el modelo fisheye de OpenCV (cv2.fisheye.*, polinomio angular Kannala-Brandt
k1..k4). El lens M12 del Arducam IMX708 es fisheye (120° HFOV / 152° diagonal) —
pinhole + CALIB_RATIONAL_MODEL fitea el centro pero deja residuales medibles en
los bordes del frame, lo cual importa para conteo cenital de cobertura amplia
(~40% de excentricidad de frame para la zona de tracking).

El renderer del overlay ghost en project_pose() se mantiene pinhole — es una
guía visual gruesa para el operador, y a las tolerancias de alineación del
wizard la discrepancia pinhole/fisheye en los bordes del frame (~3-9px en
preview) es despreciable.

Compatible con OpenCV 4.8+ (contrib) que usa la API refactoreada de ArUco.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factory de board ChArUco
# ---------------------------------------------------------------------------

DEFAULT_BOARD_SIZE = (9, 6)  # (columnas, filas) de cuadros del chessboard — A3 landscape
DEFAULT_SQUARE_LENGTH = 45.0  # mm
DEFAULT_MARKER_LENGTH = 33.0  # mm (73% del cuadro — ratio default de calib.io)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_100

# Intrínsecos nominales para IMX708 + lens M12 del Arducam B0310.
# Derivados de la óptica física: f=2.87mm / pixel_pitch=1.4μm = 2050px a la res
# nativa 4608x2592. El lens es fisheye (152°D x 120°H x 66°V capturados), así que
# la f equivalente-pinhole aproxima solo la región central — los bordes tienen
# distorsión radial significativa que la calibración corrige. Se usa para el
# rendering del ghost + estimaciones de distancia (±10% precisión en el centro
# es suficiente).
NOMINAL_FULL_RES = (4608, 2592)
NOMINAL_FOCAL_PX = 2050.0  # f_mm / pixel_pitch (pinhole-equiv, región central)


def create_charuco_board(
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
    dict_id: int = ARUCO_DICT_ID,
    legacy_pattern: bool = True,
) -> cv2.aruco.CharucoBoard:
    """Arma un board ChArUco. legacy_pattern por default es True porque nuestros
    PDFs canónicos de calib.io usan la enumeración de marcadores pre-4.6 —
    OpenCV 4.6+ cambió el default y CharucoDetector.detectBoard devuelve cero
    esquinas ChArUco contra boards impresos con el legacy cuando corre bajo el
    pattern nuevo (los markers decodean, los IDs no caen donde el nuevo pattern
    espera). Ambos lados — detección y generateImage() — respetan este flag, así
    que toggling produce comportamiento producer + detector que matchea.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(board_size, square_length, marker_length, aruco_dict)
    board.setLegacyPattern(legacy_pattern)
    return board


def generate_board_image(
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int] = (4961, 3508),  # A3 landscape @ 300 DPI
    margin: int = 50,
) -> np.ndarray:
    img = board.generateImage(image_size, marginSize=margin)
    return img


# ---------------------------------------------------------------------------
# Detección de esquinas ChArUco
# ---------------------------------------------------------------------------


# Kernel de sharpen 3×3 (Laplacian unit-sum) usado por
# ``detect_charuco_dual_pass``. Compartido para mantener los outputs
# determinísticos entre tools.
_SHARPEN_KERNEL = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32,
)


def detect_charuco_dual_pass(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    min_corners: int = 4,
    lenient: bool = True,
    accept_threshold: int = 8,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detecta ChArUco intentando primero el frame original y, si quedó
    corto, reintentando con un sharpen 3×3.

    Resuelve el flicker de detección bajo foco marginal o iluminación
    pobre: el original suele detectar mejor cuando el board está bien
    enfocado, pero ante un copy de lens flojo el sharpen recupera
    decenas de markers que el detector descarta como border-soft. Si el
    sharpen amplifica ruido sobre el threshold del decoder y termina con
    *menos* corners que el original, se queda con el original.

    Args:
        accept_threshold: Cuando el original ya supera este número de
            corners, se devuelve sin probar sharpen. 8 es el piso típico
            para un PnP estable.

    Devuelve siempre ``(corners, ids)`` del mejor candidato — None solo si
    ninguno de los dos pasos alcanzó ``min_corners``.
    """
    c_orig, i_orig = detect_charuco_corners(
        image, board, min_corners=min_corners, lenient=lenient,
    )
    n_orig = 0 if i_orig is None else len(i_orig)
    if n_orig >= accept_threshold:
        return c_orig, i_orig
    sharp = cv2.filter2D(image, -1, _SHARPEN_KERNEL)
    c_sh, i_sh = detect_charuco_corners(
        sharp, board, min_corners=min_corners, lenient=lenient,
    )
    n_sh = 0 if i_sh is None else len(i_sh)
    if n_sh > n_orig:
        return c_sh, i_sh
    return c_orig, i_orig


def detect_charuco_corners(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    min_corners: int = 8,
    lenient: bool = False,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detecta esquinas internas ChArUco.

    min_corners — esquinas mínimas requeridas para devolver una detección válida.
        Calibración necesita >=8 para intrínsecos estables; tools de focus/setup
        pueden aceptar menos (lo justo para un PnP grueso).
    lenient — afloja los thresholds de detección de markers para levantar markers
        más chicos/lejanos. Tasa de falso-positivo levemente más alta; está bien
        para tools de setup, evitar para calibración.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if lenient:
        params = cv2.aruco.DetectorParameters()
        # Default es 0.03 (3% de la dim de imagen). Bajarlo nos permite encontrar
        # markers que abarquen apenas 1% del frame — es decir, boards chicos
        # lejanos donde cada marker ocupa solo ~10-15 px.
        params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 4
        # El refinement sub-pixel estabiliza las posiciones de las esquinas
        # (~0.1px vs 1px sin él), lo que reduce materialmente el flicker de PnP
        # cuando los markers son chicos o cerca de los bordes del envelope de
        # detección.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Tolera markers más borrosos (default 0.6 rechaza ~40% de bits con
        # error; 0.8 acepta una lectura más cruda). Pequeño aumento en false-
        # positive rate a cambio de detección más estable cerca del límite.
        params.errorCorrectionRate = 0.8
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
    else:
        detector = cv2.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        gray
    )

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None

    return charuco_corners, charuco_ids


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------


def _detect_all_pairs(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
    min_common: int = 8,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detecta esquinas ChArUco en todos los pares, devuelve obj/img points matcheados.

    Returns (all_obj, all_corners_l, all_corners_r, image_size).
    """
    board_corners_3d = board.getChessboardCorners()

    all_obj: list[np.ndarray] = []
    all_corners_l: list[np.ndarray] = []
    all_corners_r: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    for idx, (img_l, img_r) in enumerate(image_pairs):
        corners_l, ids_l = detect_charuco_corners(img_l, board, lenient=True)
        corners_r, ids_r = detect_charuco_corners(img_r, board, lenient=True)

        if corners_l is None or corners_r is None:
            continue

        common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
        if len(common_ids) < min_common:
            continue

        mask_l = np.isin(ids_l.flatten(), common_ids)
        mask_r = np.isin(ids_r.flatten(), common_ids)

        c_l = corners_l[mask_l]
        c_r = corners_r[mask_r]

        order_l = np.argsort(ids_l[mask_l].flatten())
        order_r = np.argsort(ids_r[mask_r].flatten())

        obj_pts = board_corners_3d[common_ids].astype(np.float32)

        all_obj.append(obj_pts)
        all_corners_l.append(c_l[order_l].reshape(-1, 1, 2).astype(np.float32))
        all_corners_r.append(c_r[order_r].reshape(-1, 1, 2).astype(np.float32))

        if image_size is None:
            h, w = img_l.shape[:2]
            image_size = (w, h)

    return all_obj, all_corners_l, all_corners_r, image_size


# ---------------------------------------------------------------------------
# Calibración estéreo
# ---------------------------------------------------------------------------


# Balance del fisheye stereoRectify: 0.0 recorta a cero pixeles negros en el
# output rectificado (FOV ajustado). Para SGBM preferimos no tener bordes negros
# antes que tener máximo campo — las regiones negras producen disparidad espuria
# en los bordes de la imagen. Subir a ~0.3 si el FOV de los bordes resulta más
# importante que la limpieza de la disparidad en testing.
FISHEYE_RECTIFY_BALANCE = 0.0


def _reshape_for_fisheye(
    obj_pts_list: list[np.ndarray],
    img_pts_list: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """cv2.fisheye.* es exigente: los object points tienen que ser
    (N,1,3) float64 y los image points (N,1,2) float64.
    _detect_all_pairs ya devuelve los image points como (N,1,2)
    float32 — solo upcasteamos el dtype y reshapeamos los object points.
    """
    obj_out = [np.asarray(o, dtype=np.float64).reshape(-1, 1, 3) for o in obj_pts_list]
    img_out = [np.asarray(p, dtype=np.float64).reshape(-1, 1, 2) for p in img_pts_list]
    return obj_out, img_out


def _derive_stereo_rt_from_per_pose(
    rvecs_l: list[np.ndarray],
    tvecs_l: list[np.ndarray],
    rvecs_r: list[np.ndarray],
    tvecs_r: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Promedia el transform relativo per-pose (left→right) sobre todas las poses.

    Para un bracket estéreo rígido,
    R_stereo @ R_board_to_left == R_board_to_right y T_stereo ==
    t_right - R_stereo @ t_left en cada pose. Computamos ambos
    per-pose y promediamos — translations con una mean simple,
    rotaciones via quaternion averaging (mean aritmético + renormalize
    está suficientemente cerca para esta aplicación y evita la
    dependencia pypkg scipy).
    """
    rel_rotations: list[np.ndarray] = []
    rel_translations: list[np.ndarray] = []
    for rv_l, tv_l, rv_r, tv_r in zip(rvecs_l, tvecs_l, rvecs_r, tvecs_r):
        R_l, _ = cv2.Rodrigues(np.asarray(rv_l, dtype=np.float64).reshape(3, 1))
        R_r, _ = cv2.Rodrigues(np.asarray(rv_r, dtype=np.float64).reshape(3, 1))
        t_l = np.asarray(tv_l, dtype=np.float64).reshape(3, 1)
        t_r = np.asarray(tv_r, dtype=np.float64).reshape(3, 1)
        R_rel = R_r @ R_l.T
        t_rel = t_r - R_rel @ t_l
        rel_rotations.append(R_rel)
        rel_translations.append(t_rel)

    # Promediar las translations directamente.
    T_mean = np.mean(np.stack(rel_translations, axis=0), axis=0).reshape(3, 1)

    # Promediar rotaciones: mean aritmético de las matrices R,
    # después SVD-project a la rotación válida más cercana
    # (ortogonal con det=+1).
    R_sum = np.sum(np.stack(rel_rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        # Reflexión espejo — flipear la última columna singular.
        U[:, -1] *= -1
        R_mean = U @ Vt

    return R_mean.astype(np.float64), T_mean.astype(np.float64)


def _compute_stereo_rms(
    obj_list: list[np.ndarray],
    corners_l: list[np.ndarray],
    corners_r: list[np.ndarray],
    K_l: np.ndarray,
    D_l: np.ndarray,
    K_r: np.ndarray,
    D_r: np.ndarray,
    rvecs_l: list[np.ndarray],
    tvecs_l: list[np.ndarray],
    R_stereo: np.ndarray,
    T_stereo: np.ndarray,
) -> float:
    """RMS de reprojection del fit estéreo con intrínsecos fijos /
    extrínsecos promediados.

    Para cada pose, los extrínsecos de la izquierda vienen del solve
    per-pose de fisheye.calibrate; los extrínsecos de la derecha se
    derivan aplicando R_stereo, T_stereo. Proyectamos los object
    points 3D a través de ambas cámaras y comparamos contra las
    esquinas detectadas. Es la misma cantidad que stereoCalibrate
    minimiza, solo que computada después del hecho.
    """
    errs: list[float] = []
    for obj_pts, pts_l, pts_r, rv_l, tv_l in zip(
        obj_list,
        corners_l,
        corners_r,
        rvecs_l,
        tvecs_l,
    ):
        rv_l_arr = np.asarray(rv_l, dtype=np.float64).reshape(3, 1)
        tv_l_arr = np.asarray(tv_l, dtype=np.float64).reshape(3, 1)
        R_l, _ = cv2.Rodrigues(rv_l_arr)

        proj_l, _ = cv2.fisheye.projectPoints(
            obj_pts,
            rv_l_arr,
            tv_l_arr,
            K_l,
            D_l,
        )
        err_l = np.linalg.norm(
            proj_l.reshape(-1, 2) - pts_l.reshape(-1, 2),
            axis=1,
        )

        # Extrínsecos derecha: encadenar la izquierda con el transform estéreo.
        R_r = R_stereo @ R_l
        t_r = R_stereo @ tv_l_arr + T_stereo
        rv_r_arr, _ = cv2.Rodrigues(R_r)
        proj_r, _ = cv2.fisheye.projectPoints(
            obj_pts,
            rv_r_arr,
            t_r,
            K_r,
            D_r,
        )
        err_r = np.linalg.norm(
            proj_r.reshape(-1, 2) - pts_r.reshape(-1, 2),
            axis=1,
        )

        errs.extend(err_l.tolist())
        errs.extend(err_r.tolist())

    if not errs:
        return float("nan")
    errs_arr = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt((errs_arr**2).mean()))


def calibrate_stereo(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH,
    dict_id: int = ARUCO_DICT_ID,
    min_pairs: int = 15,
    legacy_pattern: bool = True,
) -> dict[str, np.ndarray]:
    """Calibración estéreo usando el modelo fisheye (Kannala-Brandt)."""
    board = create_charuco_board(
        board_size,
        square_length,
        marker_length,
        dict_id,
        legacy_pattern,
    )
    all_obj, all_corners_l, all_corners_r, image_size = _detect_all_pairs(
        image_pairs,
        board,
    )

    valid_pairs = len(all_obj)
    logger.info(
        "stereo_pairs_validated",
        extra={"valid": valid_pairs, "total": len(image_pairs)},
    )

    if valid_pairs < min_pairs:
        raise ValueError(f"Need at least {min_pairs} valid pairs, got {valid_pairs}.")

    obj_f, corners_l_f = _reshape_for_fisheye(all_obj, all_corners_l)
    _, corners_r_f = _reshape_for_fisheye(all_obj, all_corners_r)

    # CHECK_COND raisea si una pose es muy degenerada. Lo omitimos —
    # con capturas de lab reales las 20 poses canónicas están bien
    # distribuidas, y mantenerlo off evita fallas crípticas en casos
    # borderline raros.
    calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # Pre-allocar K/D con las shapes que espera el binding fisheye,
    # pero NO pre-allocar rvecs/tvecs — el binding Python las construye
    # él mismo a partir del count de poses, y pasar placeholders de
    # shape fijo dispara una assertion _OutputArray (OpenCV 4.13+).
    K_l = np.zeros((3, 3), dtype=np.float64)
    D_l = np.zeros((4, 1), dtype=np.float64)
    K_r = np.zeros((3, 3), dtype=np.float64)
    D_r = np.zeros((4, 1), dtype=np.float64)

    rms_l, K_l, D_l, rvecs_l, tvecs_l = cv2.fisheye.calibrate(
        obj_f,
        corners_l_f,
        image_size,
        K_l,
        D_l,
        flags=calib_flags,
        criteria=criteria,
    )
    logger.info(
        "camera_calibrated",
        extra={"camera": "left", "rms": rms_l, "pairs": valid_pairs},
    )

    rms_r, K_r, D_r, rvecs_r, tvecs_r = cv2.fisheye.calibrate(
        obj_f,
        corners_r_f,
        image_size,
        K_r,
        D_r,
        flags=calib_flags,
        criteria=criteria,
    )
    logger.info(
        "camera_calibrated",
        extra={"camera": "right", "rms": rms_r, "pairs": valid_pairs},
    )

    D_l = np.asarray(D_l, dtype=np.float64).reshape(4, 1)
    D_r = np.asarray(D_r, dtype=np.float64).reshape(4, 1)

    # fisheye.stereoCalibrate requiere counts de puntos idénticos
    # entre poses (limitación de OpenCV 4.13). Nuestras capturas de
    # lab tienen counts per-pose variables (las distancias lejanas
    # pierden esquinas), así que derivamos el transform rígido L→R
    # directamente de los extrínsecos per-pose que devuelve
    # fisheye.calibrate: para cada pose i, R_stereo = R_ri @ R_li.T
    # y T_stereo = t_ri - R_stereo @ t_li. Promediar sobre las poses
    # (con proyección SO(3) en las rotaciones) da el mismo R, T al
    # cual stereoCalibrate joint convergería, dentro de una fracción
    # de pixel de RMS para un bracket rígido.
    R, T = _derive_stereo_rt_from_per_pose(rvecs_l, tvecs_l, rvecs_r, tvecs_r)
    rms_stereo = _compute_stereo_rms(
        obj_f,
        corners_l_f,
        corners_r_f,
        K_l,
        D_l,
        K_r,
        D_r,
        rvecs_l,
        tvecs_l,
        R,
        T,
    )
    logger.info(
        "stereo_calibrated",
        extra={"rms": rms_stereo, "pairs": valid_pairs},
    )

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l,
        D_l,
        K_r,
        D_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
        balance=FISHEYE_RECTIFY_BALANCE,
    )
    map_l_x, map_l_y = cv2.fisheye.initUndistortRectifyMap(
        K_l,
        D_l,
        R1,
        P1,
        image_size,
        cv2.CV_32FC1,
    )
    map_r_x, map_r_y = cv2.fisheye.initUndistortRectifyMap(
        K_r,
        D_r,
        R2,
        P2,
        image_size,
        cv2.CV_32FC1,
    )

    logger.info(
        "stereo_geometry_computed",
        extra={
            "fx": float(K_l[0, 0]),
            "fy": float(K_l[1, 1]),
            "baseline_mm": float(abs(T[0, 0])),
        },
    )

    result: dict[str, np.ndarray] = {
        "camera_matrix_l": K_l,
        "dist_coeffs_l": D_l,
        "camera_matrix_r": K_r,
        "dist_coeffs_r": D_r,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map_l_x": map_l_x,
        "map_l_y": map_l_y,
        "map_r_x": map_r_x,
        "map_r_y": map_r_y,
        "image_size": np.array(list(image_size)),
    }
    _ensure_rectified_intrinsics(result)
    return result


# ---------------------------------------------------------------------------
# Diagnóstico de lens alignment
# ---------------------------------------------------------------------------


def lens_alignment_metrics(
    R: np.ndarray,
    T: np.ndarray,
) -> dict[str, float]:
    """Descompone los extrínsecos estéreo en métricas de alineamiento
    operator-friendly.

    Solo diagnóstico — estos números describen qué tan limpiamente el
    bracket armó un par estéreo, no si la calibración es usable. El
    pipeline runtime ya tiene en cuenta cualquier misalignment vía
    R/T (la rectificación absorbe la geometría), así que un grado de
    yaw o un milímetro de offset vertical se corrige internamente y
    no rompe el depth. Los exponemos así QA puede comparar unidades
    y flaggear outliers de fabricación.

    Returns:
        Dict con:
        - ``offset_x_mm`` / ``offset_y_mm`` / ``offset_z_mm``:
          componentes de T (mm). Offset X es el baseline; Y y Z
          deberían ser chicos en un bracket limpio.
        - ``rotation_x_deg`` / ``rotation_y_deg`` / ``rotation_z_deg``:
          ángulos de Euler (convención XYZ) de R, en grados. Pitch /
          yaw / roll entre los lentes. Bracket perfecto = todos cero.
    """
    T_arr = np.asarray(T, dtype=np.float64).reshape(3)

    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R_arr.shape}")

    # Descomposición Euler convención XYZ estándar. Maneja la
    # singularidad gimbal-lock cuando |R[2,0]| ~ 1 (cos(pitch) ~ 0):
    # en ese caso el roll colapsa en el pitch y reportamos el valor
    # combinado con roll=0 (la convención estándar — no hay
    # descomposición única ahí, pero para un bracket estéreo real
    # nunca nos acercamos).
    sy = math.sqrt(R_arr[0, 0] ** 2 + R_arr[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R_arr[2, 1], R_arr[2, 2])
        ry = math.atan2(-R_arr[2, 0], sy)
        rz = math.atan2(R_arr[1, 0], R_arr[0, 0])
    else:
        rx = math.atan2(-R_arr[1, 2], R_arr[1, 1])
        ry = math.atan2(-R_arr[2, 0], sy)
        rz = 0.0

    return {
        "offset_x_mm": float(T_arr[0]),
        "offset_y_mm": float(T_arr[1]),
        "offset_z_mm": float(T_arr[2]),
        "rotation_x_deg": math.degrees(rx),
        "rotation_y_deg": math.degrees(ry),
        "rotation_z_deg": math.degrees(rz),
    }


# ---------------------------------------------------------------------------
# Rectification helpers
# ---------------------------------------------------------------------------


def rectify_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rect_l = cv2.remap(
        img_l,
        calibration["map_l_x"],
        calibration["map_l_y"],
        cv2.INTER_LINEAR,
    )
    rect_r = cv2.remap(
        img_r,
        calibration["map_r_x"],
        calibration["map_r_y"],
        cv2.INTER_LINEAR,
    )
    return rect_l, rect_r


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _ensure_rectified_intrinsics(data: dict[str, np.ndarray]) -> None:
    """Inyecta intrínsecos rectificados escalares derivados de la proyección P1.

    Después de ``cv2.fisheye.stereoRectify`` el frame rectificado
    izquierdo tiene una proyección pinhole de la forma

        P1 = [[fx, 0,  cx, 0],
              [0,  fy, cy, 0],
              [0,  0,  1,  0]]

    Estos tres escalares son necesarios cada vez que el código
    runtime quiere back-proyectar un pixel (u, v, Z) a un punto
    métrico (X, Y, Z) en el frame de cámara izquierda — prácticamente
    toda la lógica 3D que consume un depth map (column filter de
    ``head_depth_in_bbox``, futuros helpers de world-space). Son
    una función solo de P1, así que los derivamos una vez acá en
    vez de pedirle a cada caller que se meta dentro de la matriz.

    El .npz en disco solo guarda P1 (y P2). Calibraciones viejas sin
    P1 quedan intactas — los callers todavía tienen que tolerar la
    ausencia.
    """
    p1 = data.get("P1")
    if p1 is None:
        return
    p1_arr = np.asarray(p1, dtype=np.float64)
    if p1_arr.shape != (3, 4) and p1_arr.shape != (3, 3):
        return
    data["fx_rect"] = np.float64(p1_arr[0, 0])
    data["fy_rect"] = np.float64(p1_arr[1, 1])
    data["cx_rect"] = np.float64(p1_arr[0, 2])
    data["cy_rect"] = np.float64(p1_arr[1, 2])


def load_calibration(path: str) -> dict[str, np.ndarray]:
    cal_path = Path(path)
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = dict(np.load(cal_path))
    # Derivar intrínsecos rectificados escalares (fx, fy, cx, cy) de
    # P1 así el pipeline runtime no tiene que crackear la projection
    # matrix en cada conversión depth-to-world.
    _ensure_rectified_intrinsics(data)
    # Convertir los rectify maps float32 a forma fixed-point INT16 +
    # UINT16 (cv2 CV_16SC2 / CV_16UC1) una vez al cargar. cv2.remap
    # con estos mapas es ~2x más rápido que con mapas float32 porque
    # el lookup es integer + bilinear-table lookup en lugar de
    # interpolación float. El .npz en disco guarda los mapas float32
    # por compatibilidad — re-derivamos la forma rápida en cada load.
    if "map_l_x" in data and "map_l_y" in data:
        try:
            data["map_l_x"], data["map_l_y"] = cv2.convertMaps(
                data["map_l_x"],
                data["map_l_y"],
                cv2.CV_16SC2,
            )
            data["map_r_x"], data["map_r_y"] = cv2.convertMaps(
                data["map_r_x"],
                data["map_r_y"],
                cv2.CV_16SC2,
            )
            logger.info(
                "rectify_maps_optimized",
                extra={"format": "CV_16SC2 + CV_16UC1"},
            )
        except cv2.error as e:
            logger.warning(
                "rectify_maps_optimization_failed",
                extra={"error": str(e)},
            )
    logger.info("Calibration loaded", extra={"path": path, "keys": list(data.keys())})
    return data


def save_calibration(params: dict[str, np.ndarray], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)
    logger.info("Calibration saved", extra={"path": path})


# ---------------------------------------------------------------------------
# Guided calibration: pose targets, ghost projection, quality/matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseTarget:
    """Posición + orientación target para el board ChArUco durante captura guiada.

    Translation en milímetros en camera frame (+X derecha, +Y abajo,
    +Z adelante). Rotaciones en grados aplicadas como rotZ(roll) *
    rotY(yaw) * rotX(pitch).
    """

    id: str
    label: str
    tvec_mm: tuple[float, float, float]
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    def rvec(self) -> np.ndarray:
        """Vector de rotación Rodrigues a partir de los ángulos de Euler."""
        rx = math.radians(self.pitch_deg)
        ry = math.radians(self.yaw_deg)
        rz = math.radians(self.roll_deg)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(rx), -math.sin(rx)],
                [0, math.sin(rx), math.cos(rx)],
            ]
        )
        Ry = np.array(
            [
                [math.cos(ry), 0, math.sin(ry)],
                [0, 1, 0],
                [-math.sin(ry), 0, math.cos(ry)],
            ]
        )
        Rz = np.array(
            [
                [math.cos(rz), -math.sin(rz), 0],
                [math.sin(rz), math.cos(rz), 0],
                [0, 0, 1],
            ]
        )
        R = Rz @ Ry @ Rx
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()


DEFAULT_DIST_NEAR_MM = 1000.0
DEFAULT_DIST_MID_MM = 2000.0
DEFAULT_DIST_FAR_MM = 3000.0


def default_pose_sequence(
    near_mm: float = DEFAULT_DIST_NEAR_MM,
    mid_mm: float = DEFAULT_DIST_MID_MM,
    far_mm: float = DEFAULT_DIST_FAR_MM,
) -> list[PoseTarget]:
    """Devuelve una secuencia canónica de 20 poses cubriendo
    near/mid/far × posiciones × tilts.

    Las distancias default (1000/2000/3000mm) forman el protocolo
    universal de fábrica — la misma calibración se shippea con cada
    device y sirve a todo el rango de mount de la flota 2.0–3.5m sin
    adaptación per-device. El polinomio angular K-B es independiente
    de la profundidad: lo que importa es samplear la distorsión a lo
    largo de todo el plano de imagen (centro + periferia), no
    matchear punto-por-punto la profundidad operativa. El spread
    1/2/3m más los tilts en los grupos A–E llevan las esquinas a
    través de la periferia de alta curvatura en múltiples ángulos
    de incidencia, que es lo que fittea k1–k4 robustamente.

    Límites prácticos a 3m con f≈1025px (binned 2304×1296): un
    marker de 33mm proyecta a ~11 px — al borde de detección
    confiable de ChArUco. Empujar far a 3.5m deja los markers en
    ~10 px y empieza a perder esquinas; el polinomio extrapola el
    0.5m extra cómodamente (<20%) así el piso al mount más alto
    queda well-behaved.

    Check de cobertura operativa: cabezas a depth 1.0m (persona de
    1.85m a mount 2.0m, borde near de la línea de conteo) hasta 2.5m
    (niño de 1.0m a mount 3.5m); piso 2.0m–3.5m. Rango completo de
    bbox depth 1.0–3.5m cae bien dentro de las distancias sampleadas
    1–3m más la extrapolación tolerada de ±0.5m.
    """
    near, mid, far = near_mm, mid_mm, far_mm

    # Offsets laterales: valores X/Y de tvec elegidos así el board
    # proyectado cae cerca de la posición target en pantalla en cada
    # distancia.
    def off(frac_x: float, frac_y: float, z: float) -> tuple[float, float]:
        # Proyectar un offset en pixels del centro de imagen de vuelta
        # a mm en world a depth z.
        half_w_px = NOMINAL_FULL_RES[0] / 2
        half_h_px = NOMINAL_FULL_RES[1] / 2
        return (
            frac_x * half_w_px * z / NOMINAL_FOCAL_PX,
            frac_y * half_h_px * z / NOMINAL_FOCAL_PX,
        )

    poses: list[PoseTarget] = []

    # Grupo A — Center en near, tilts sistemáticos
    poses.append(PoseTarget("A1", "Center near, frontal", (0, 0, near)))
    poses.append(PoseTarget("A2", "Center near, pitch up", (0, 0, near), pitch_deg=-20))
    poses.append(PoseTarget("A3", "Center near, yaw left", (0, 0, near), yaw_deg=-20))
    poses.append(
        PoseTarget(
            "A4", "Center near, pitch/yaw mix", (0, 0, near), pitch_deg=15, yaw_deg=15
        )
    )

    # Grupo B — Esquinas en mid, edge constraints
    dx, dy = off(-0.55, -0.45, mid)
    poses.append(PoseTarget("B1", "Top-left mid, yaw left", (dx, dy, mid), yaw_deg=-15))
    dx, dy = off(0.55, -0.45, mid)
    poses.append(
        PoseTarget("B2", "Top-right mid, yaw right", (dx, dy, mid), yaw_deg=15)
    )
    dx, dy = off(-0.55, 0.45, mid)
    poses.append(
        PoseTarget("B3", "Bottom-left mid, pitch up", (dx, dy, mid), pitch_deg=15)
    )
    dx, dy = off(0.55, 0.45, mid)
    poses.append(
        PoseTarget("B4", "Bottom-right mid, pitch down", (dx, dy, mid), pitch_deg=-15)
    )

    # Grupo C — Distancia mid, diversidad de roll + pitch
    poses.append(PoseTarget("C1", "Center mid, roll +20", (0, 0, mid), roll_deg=20))
    poses.append(PoseTarget("C2", "Center mid, roll -20", (0, 0, mid), roll_deg=-20))
    dx, dy = off(0.0, -0.4, mid)
    poses.append(
        PoseTarget("C3", "Top-center mid, pitch up", (dx, dy, mid), pitch_deg=-20)
    )
    dx, dy = off(0.0, 0.4, mid)
    poses.append(
        PoseTarget("C4", "Bottom-center mid, pitch down", (dx, dy, mid), pitch_deg=20)
    )

    # Grupo D — Distancia far, cobertura
    poses.append(PoseTarget("D1", "Center far, frontal", (0, 0, far)))
    dx, dy = off(-0.45, -0.35, far)
    poses.append(
        PoseTarget(
            "D2", "Top-left far, diag tilt", (dx, dy, far), yaw_deg=-10, pitch_deg=-10
        )
    )
    dx, dy = off(0.45, -0.35, far)
    poses.append(
        PoseTarget(
            "D3", "Top-right far, diag tilt", (dx, dy, far), yaw_deg=10, pitch_deg=-10
        )
    )
    dx, dy = off(-0.45, 0.35, far)
    poses.append(
        PoseTarget(
            "D4", "Bottom-left far, diag tilt", (dx, dy, far), yaw_deg=-10, pitch_deg=10
        )
    )
    dx, dy = off(0.45, 0.35, far)
    poses.append(
        PoseTarget(
            "D5", "Bottom-right far, diag tilt", (dx, dy, far), yaw_deg=10, pitch_deg=10
        )
    )

    # Grupo E — Tilts extremos en near
    dx, dy = off(-0.5, 0, near)
    poses.append(
        PoseTarget("E1", "Left-mid near, strong yaw", (dx, dy, near), yaw_deg=-25)
    )
    dx, dy = off(0.5, 0, near)
    poses.append(
        PoseTarget("E2", "Right-mid near, strong yaw", (dx, dy, near), yaw_deg=25)
    )
    poses.append(
        PoseTarget(
            "E3",
            "Center mid, combined 3-axis",
            (0, 0, mid),
            pitch_deg=-12,
            yaw_deg=-12,
            roll_deg=15,
        )
    )

    return poses


def project_pose(
    pose: PoseTarget,
    board_size: tuple[int, int],
    square_length: float,
    preview_size: tuple[int, int],
    focal_full_px: float = NOMINAL_FOCAL_PX,
    full_res: tuple[int, int] = NOMINAL_FULL_RES,
    fitted_K: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Proyecta un pose target a coordenadas pixel del preview.

    Args:
        fitted_K: Si se pasa, usa esta matriz 3x3 de intrínsecos
            (a full_res) escalada a preview_size. Override de
            focal_full_px + principal point centrado. Se usa durante
            el handoff bootstrap → tight-tolerance.

    Devuelve un dict con:
        outer_corners: (4, 2) — outline del board para el ghost visual,
            sentido horario TL,TR,BR,BL
        inner_corners: (N, 2) — esquinas internas del chessboard
            (para matching vs detección)
        center: (2,) — centro proyectado del board
    """
    cols, rows = board_size
    board_w_mm = cols * square_length
    board_h_mm = rows * square_length

    preview_w, preview_h = preview_size
    full_w, full_h = full_res
    scale_x = preview_w / full_w
    scale_y = preview_h / full_h

    if fitted_K is not None:
        f_preview_x = float(fitted_K[0, 0]) * scale_x
        f_preview_y = float(fitted_K[1, 1]) * scale_y
        cx_preview = float(fitted_K[0, 2]) * scale_x
        cy_preview = float(fitted_K[1, 2]) * scale_y
    else:
        f_preview_x = focal_full_px * scale_x
        f_preview_y = focal_full_px * scale_y
        cx_preview = preview_w / 2
        cy_preview = preview_h / 2

    K = np.array(
        [[f_preview_x, 0, cx_preview], [0, f_preview_y, cy_preview], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros(5)

    # Puntos del board expresados en un frame centrado en el centro
    # del board (así tvec coloca el centro del board, no su esquina
    # superior izquierda).
    half_w = board_w_mm / 2
    half_h = board_h_mm / 2

    outer_3d = np.array(
        [
            [-half_w, -half_h, 0],
            [half_w, -half_h, 0],
            [half_w, half_h, 0],
            [-half_w, half_h, 0],
        ],
        dtype=np.float32,
    )

    inner_pts = []
    for r in range(1, rows):
        for c in range(1, cols):
            x = c * square_length - half_w
            y = r * square_length - half_h
            inner_pts.append([x, y, 0])
    inner_3d = np.array(inner_pts, dtype=np.float32)

    rvec = pose.rvec().astype(np.float64)
    tvec = np.array(pose.tvec_mm, dtype=np.float64)

    outer_2d, _ = cv2.projectPoints(outer_3d, rvec, tvec, K, dist)
    inner_2d, _ = cv2.projectPoints(inner_3d, rvec, tvec, K, dist)
    center_2d, _ = cv2.projectPoints(
        np.array([[0, 0, 0]], dtype=np.float32),
        rvec,
        tvec,
        K,
        dist,
    )

    return {
        "outer_corners": outer_2d.reshape(-1, 2),
        "inner_corners": inner_2d.reshape(-1, 2),
        "center": center_2d.reshape(2),
    }


def _min_area_rect(
    points: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Wrapper sobre cv2.minAreaRect devolviendo (center, size, angle)."""
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.minAreaRect(pts)


def compute_alignment_error(
    detected_preview: np.ndarray,
    ghost_inner: np.ndarray,
) -> dict[str, float]:
    """Compara esquinas ChArUco detectadas (en preview px) contra
    las esquinas internas del ghost proyectado.

    Devuelve center_px, scale_ratio (detected/ghost), rotation_deg
    (wrapped a [-45, 45]). Todas las comparaciones están en
    coordenadas pixel de preview.
    """
    det_center, det_size, det_angle = _min_area_rect(detected_preview)
    ghost_center, ghost_size, ghost_angle = _min_area_rect(ghost_inner)

    cdx = det_center[0] - ghost_center[0]
    cdy = det_center[1] - ghost_center[1]
    center_px = float(math.hypot(cdx, cdy))

    det_diag = math.hypot(det_size[0], det_size[1])
    ghost_diag = math.hypot(ghost_size[0], ghost_size[1])
    scale_ratio = det_diag / ghost_diag if ghost_diag > 1 else 0.0

    # Los ángulos de minAreaRect wrapean a 90°, entonces wrapeamos
    # la diferencia a [-45, 45]
    raw_diff = det_angle - ghost_angle
    while raw_diff > 45:
        raw_diff -= 90
    while raw_diff < -45:
        raw_diff += 90

    return {
        "center_px": center_px,
        "offset_x": float(cdx),
        "offset_y": float(cdy),
        "scale_ratio": float(scale_ratio),
        "rotation_deg": float(raw_diff),
    }


ALIGN_CENTER_TOL_PX = 20.0
ALIGN_SCALE_TOL = 0.05  # ±5%
ALIGN_ROTATION_TOL_DEG = 3.0


def is_aligned(error: dict[str, float]) -> bool:
    """Gate de tolerancia tight para auto-capture (legacy basado en minAreaRect)."""
    return (
        error["center_px"] <= ALIGN_CENTER_TOL_PX
        and abs(error["scale_ratio"] - 1.0) <= ALIGN_SCALE_TOL
        and abs(error["rotation_deg"]) <= ALIGN_ROTATION_TOL_DEG
    )


# ---------------------------------------------------------------------------
# Matching basado en IDs de esquina (preferido sobre minAreaRect —
# maneja poses tilteadas)
# ---------------------------------------------------------------------------


ALIGN_MEAN_ERR_TOL_PX_TIGHT = 12.0
ALIGN_MEAN_ERR_TOL_PX_LOOSE = 25.0
ALIGN_MATCHED_MIN_TIGHT = 15
ALIGN_MATCHED_MIN_LOOSE = 12


def compute_alignment_by_corners(
    detected_preview: np.ndarray,
    detected_ids: np.ndarray,
    ghost_inner_all: np.ndarray,
) -> dict[str, float]:
    """Matchea esquinas ChArUco detectadas contra el ghost por ID,
    computa error per-corner.

    Args:
        detected_preview: (N, 2) esquinas internas detectadas en preview px
        detected_ids: (N,) IDs ChArUco de cada esquina detectada
        ghost_inner_all: (M, 2) todas las esquinas internas proyectadas
            del ghost, indexadas por ID

    Devuelve dict con count matched, mean_error_px, max_error_px,
    vector offset, y centroid_offset_px. Poses tilteadas se manejan
    naturalmente porque comparamos per-corner, no via un fit de
    similarity 2D.
    """
    pairs_det: list[np.ndarray] = []
    pairs_ghost: list[np.ndarray] = []
    ids_flat = np.asarray(detected_ids).flatten()
    for i, cid in enumerate(ids_flat):
        cid_int = int(cid)
        if 0 <= cid_int < len(ghost_inner_all):
            pairs_det.append(detected_preview[i])
            pairs_ghost.append(ghost_inner_all[cid_int])

    n = len(pairs_det)
    empty = {
        "matched": n,
        "mean_error_px": float("inf"),
        "max_error_px": float("inf"),
        "offset_x": 0.0,
        "offset_y": 0.0,
        "centroid_offset_px": float("inf"),
    }
    if n < 4:
        return empty

    det = np.asarray(pairs_det, dtype=np.float32).reshape(-1, 2)
    ghost = np.asarray(pairs_ghost, dtype=np.float32).reshape(-1, 2)
    errors = np.linalg.norm(det - ghost, axis=1)

    offset = det.mean(axis=0) - ghost.mean(axis=0)
    return {
        "matched": n,
        "mean_error_px": float(errors.mean()),
        "max_error_px": float(errors.max()),
        "offset_x": float(offset[0]),
        "offset_y": float(offset[1]),
        "centroid_offset_px": float(math.hypot(offset[0], offset[1])),
    }


def is_aligned_by_corners(
    err: dict[str, float],
    mean_err_tol_px: float = ALIGN_MEAN_ERR_TOL_PX_TIGHT,
    min_matched: int = ALIGN_MATCHED_MIN_TIGHT,
) -> bool:
    """Gate de alignment para el matcher por corner-ID. Tolerancias
    configurables así la fase bootstrap puede usar límites más
    sueltos mientras la fase fitted-K es estricta.
    """
    return err["matched"] >= min_matched and err["mean_error_px"] <= mean_err_tol_px


def alignment_hint_by_corners(
    err: dict[str, float],
    mm_per_px: float | None = None,
) -> str:
    """Indicación human-readable para el operador. Usa el centroid
    offset como el vector de error dominante; expone el
    residual-after-translation cuando es chico.

    Si se provee `mm_per_px`, el hint reporta offsets en cm en lugar
    de pixels — más significativo para un operador sosteniendo un
    board físico. Esta ratio es `target_distance_mm /
    focal_length_px` evaluada a la profundidad esperada de la pose.
    """

    def _fmt_offset(px: float) -> str:
        if mm_per_px is None:
            return f"{px:.0f}px"
        cm = abs(px) * mm_per_px / 10.0
        return f"{cm:.0f}cm"

    if err["matched"] < 8:
        return "Board apenas visible — movelo al centro"
    if err["centroid_offset_px"] < 10:
        if err["mean_error_px"] > ALIGN_MEAN_ERR_TOL_PX_TIGHT:
            residual = _fmt_offset(err["mean_error_px"])
            return f"Ajustá tilt/escala (residual {residual})"
        return "Alineado"
    parts = []
    if abs(err["offset_x"]) > abs(err["offset_y"]):
        direction = "izquierda" if err["offset_x"] > 0 else "derecha"
        parts.append(f"Movelo {direction} {_fmt_offset(err['offset_x'])}")
    else:
        direction = "arriba" if err["offset_y"] > 0 else "abajo"
        parts.append(f"Movelo {direction} {_fmt_offset(err['offset_y'])}")
    if err["mean_error_px"] > ALIGN_MEAN_ERR_TOL_PX_TIGHT + err["centroid_offset_px"]:
        parts.append("y ajustá tilt")
    return " · ".join(parts)


def count_common_corners(
    ids_l: Optional[np.ndarray], ids_r: Optional[np.ndarray]
) -> int:
    """Cuenta los IDs ChArUco presentes en ambas detecciones (izq y der)."""
    if ids_l is None or ids_r is None:
        return 0
    return int(
        np.intersect1d(np.asarray(ids_l).flatten(), np.asarray(ids_r).flatten()).size
    )


# ---------------------------------------------------------------------------
# Check de diversidad de cobertura de poses (cuida contra sets degenerados)
# ---------------------------------------------------------------------------


def analyze_pose_coverage(
    captured_pose_ids: list[str],
    all_poses: Optional[list[PoseTarget]] = None,
    near_mm: float = DEFAULT_DIST_NEAR_MM,
    mid_mm: float = DEFAULT_DIST_MID_MM,
    far_mm: float = DEFAULT_DIST_FAR_MM,
) -> dict[str, object]:
    """Reporta qué tan diverso es un set de poses capturadas.

    La secuencia canónica agrupa poses por distancia + patrón de tilt
    via la letra prefijo (A=center near, B=corners mid, C=roll/pitch
    mid, D=far, E=extreme). Una calibración saludable captura al
    menos 2 poses de cada A..D y cubre tilts de pitch, yaw y roll.

    near_mm/mid_mm/far_mm: thresholds para la clasificación de
    bandas, derivados de los flags reales --dist del wizard.
    Default al canónico 1/2/3m. Sin estos parámetros en scope,
    setups de distancia custom (ej. PoC a 0.5/0.9/1.3m) tenían todo
    clasificado como near/mid porque los thresholds hardcodeados
    1200/2000 no matcheaban.

    Returns:
        {
          "by_distance": {"near": n, "mid": n, "far": n},
          "by_tilt_axis": {"frontal": n, "pitch": n, "yaw": n, "roll": n},
          "by_group": {"A": n, "B": n, "C": n, "D": n, "E": n},
          "warnings": [str, ...],
          "ok": bool,
        }
    """
    if all_poses is None:
        all_poses = default_pose_sequence(near_mm=near_mm, mid_mm=mid_mm, far_mm=far_mm)
    pose_by_id = {p.id: p for p in all_poses}

    # Bordes de banda: midpoint entre distancias target adyacentes.
    # Con el canónico 1000/2000/3000mm da 1500 y 2500 (cerca de los
    # 1200/2000 hardcodeados anteriores). Con distancias PoC
    # (ej. 500/900/1300) da 700 y 1100, clasificando correctamente
    # cada banda.
    near_mid_threshold = (near_mm + mid_mm) / 2
    mid_far_threshold = (mid_mm + far_mm) / 2

    by_distance = {"near": 0, "mid": 0, "far": 0}
    by_tilt = {"frontal": 0, "pitch": 0, "yaw": 0, "roll": 0}
    by_group: dict[str, int] = {}

    for pid in captured_pose_ids:
        pose = pose_by_id.get(pid)
        if pose is None:
            continue
        z = pose.tvec_mm[2]
        if z < near_mid_threshold:
            by_distance["near"] += 1
        elif z < mid_far_threshold:
            by_distance["mid"] += 1
        else:
            by_distance["far"] += 1

        tilts = (pose.pitch_deg, pose.yaw_deg, pose.roll_deg)
        if max(abs(t) for t in tilts) < 5:
            by_tilt["frontal"] += 1
        if abs(pose.pitch_deg) >= 10:
            by_tilt["pitch"] += 1
        if abs(pose.yaw_deg) >= 10:
            by_tilt["yaw"] += 1
        if abs(pose.roll_deg) >= 10:
            by_tilt["roll"] += 1

        group = pose.id[0] if pose.id else "?"
        by_group[group] = by_group.get(group, 0) + 1

    warnings: list[str] = []
    critical: list[str] = []  # Gaps de cobertura que producen fits degenerados
    for band, count in by_distance.items():
        if count == 0:
            critical.append(f"Banda de distancia '{band}' sin capturas")
        elif count < 2:
            warnings.append(f"Solo {count} captura en banda de distancia '{band}'")
    for axis, count in by_tilt.items():
        if axis == "frontal":
            continue  # frontal es nice-to-have, no requerido
        if count < 2:
            warnings.append(f"Poca diversidad de tilt '{axis}' ({count} captura/s)")
    for group in ["A", "B", "C", "D"]:  # E es extras, no requerido
        if by_group.get(group, 0) < 1:
            critical.append(f"Grupo de poses {group} sin capturas")

    return {
        "by_distance": by_distance,
        "by_tilt_axis": by_tilt,
        "by_group": by_group,
        "warnings": warnings,
        "critical": critical,
        "ok": len(warnings) == 0 and len(critical) == 0,
    }


# ---------------------------------------------------------------------------
# Check de readiness para early-stop (le ahorra tiempo al operador en
# sesiones smooth)
# ---------------------------------------------------------------------------

# Thresholds conservadores. Sesiones de calidad lab caen en RMS <0.5
# px fácilmente; si estás debajo de eso con cobertura completa, más
# poses no van a mejorar materialmente la calibración (rendimientos
# decrecientes). El mín de poses previene la trampa de "capturé 8
# frames fáciles a las apuradas" donde la cobertura se ve fine pero
# el solver no vio suficiente variedad. El wizard agrega un confirm
# manual encima — el operador siempre puede seguir si quiere.
EARLY_STOP_MIN_POSES = 12
EARLY_STOP_MAX_RMS_PX = 0.50
EARLY_STOP_MIN_PER_BAND = 2
EARLY_STOP_MIN_PER_GROUP = 1


def is_calibration_ready_for_early_stop(
    coverage: dict[str, object],
    per_pair_rms_px: float,
    captured_count: int,
    *,
    min_poses: int = EARLY_STOP_MIN_POSES,
    max_rms_px: float = EARLY_STOP_MAX_RMS_PX,
    min_per_band: int = EARLY_STOP_MIN_PER_BAND,
    min_per_group: int = EARLY_STOP_MIN_PER_GROUP,
) -> tuple[bool, str]:
    """Decide si el wizard podría finalizar antes de alcanzar todas las poses.

    Todas las condiciones tienen que valer simultáneamente:
      - Total de capturas >= ``min_poses`` (default 12, bien por
        debajo del canónico 20 — deja margen para que el solver vea
        variedad real).
      - La cobertura tiene cero gaps ``critical`` (sin band/group
        faltante).
      - Cada banda (near/mid/far) tiene >= ``min_per_band`` capturas.
      - Cada group requerido (A..D) tiene >= ``min_per_group`` capturas.
      - La estimación actual de RMS per-pair <= ``max_rms_px``
        (finalizar solo cuando el fit ya es lab-grade).

    Devuelve ``(ready, reason)``. ``reason`` queda vacío cuando está
    ready, o un string corto operator-friendly explicando qué
    todavía falta.
    """
    if captured_count < min_poses:
        return False, (f"faltan capturas ({captured_count}/{min_poses} mínimo)")

    critical = coverage.get("critical") or []
    if critical:
        return False, "cobertura incompleta: " + "; ".join(str(c) for c in critical)

    by_band = coverage.get("by_distance") or {}
    for band in ("near", "mid", "far"):
        if int(by_band.get(band, 0)) < min_per_band:
            return False, (
                f"banda '{band}' con {by_band.get(band, 0)} capturas "
                f"(mínimo {min_per_band})"
            )

    by_group = coverage.get("by_group") or {}
    for group in ("A", "B", "C", "D"):
        if int(by_group.get(group, 0)) < min_per_group:
            return False, (
                f"grupo {group} sin capturas suficientes " f"(mínimo {min_per_group})"
            )

    if per_pair_rms_px != per_pair_rms_px:  # NaN
        return False, "RMS no calculable todavía"
    if per_pair_rms_px > max_rms_px:
        return False, (
            f"RMS={per_pair_rms_px:.2f}px sobre el umbral "
            f"({max_rms_px:.2f}px) — seguí capturando"
        )

    return True, ""


# ---------------------------------------------------------------------------
# Fit bootstrap de intrínsecos — cambia la proyección del ghost de K
# nominal a K medido per-sensor después de las primeras capturas
# ---------------------------------------------------------------------------


def fit_single_camera_intrinsics(
    images: list[np.ndarray],
    board: cv2.aruco.CharucoBoard,
) -> Optional[np.ndarray]:
    """Calibra intrínsecos para una sola cámara a partir de un puñado
    de imágenes ChArUco.

    Devuelve la matriz 3×3 de cámara, o None si la detección falla
    en demasiados frames. Se usa para el bootstrap de captura guiada
    — después de unas pocas capturas reemplazamos el K nominal usado
    para renderizar los ghost targets con el valor medido per-sensor.

    Se mantiene intencionalmente en el modelo pinhole
    (cv2.calibrateCamera) aunque el pipeline estéreo principal es
    fisheye. El único consumer es project_pose() que también
    renderiza el ghost con proyección pinhole — mantener ambas
    mitades en el mismo modelo evita un offset visible entre el
    ghost y el board detectado en los bordes del frame durante el
    bootstrap. La calibración final que efectivamente va al .npz
    usa cv2.fisheye.* en calibrate_stereo.
    """
    all_obj: list[np.ndarray] = []
    all_corners: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    board_corners_3d = board.getChessboardCorners()
    for img in images:
        corners, ids = detect_charuco_corners(img, board)
        if corners is None or ids is None or len(ids) < 8:
            continue
        obj_pts = board_corners_3d[ids.flatten()].astype(np.float32)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
        all_obj.append(obj_pts)
        all_corners.append(img_pts)
        if image_size is None:
            h, w = img.shape[:2]
            image_size = (w, h)

    if len(all_obj) < 4 or image_size is None:
        return None

    try:
        _rms, K, _dist, _rvecs, _tvecs = cv2.calibrateCamera(
            all_obj,
            all_corners,
            image_size,
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL,
        )
    except cv2.error:
        return None
    return K


# ---------------------------------------------------------------------------
# Análisis de residuales per-par (flaggea outliers en el reporte HTML)
# ---------------------------------------------------------------------------


def _fisheye_reprojection_rms(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    """Residual PnP + reproyección para una vista bajo el modelo fisheye.

    OpenCV no shippea un solvePnP fisheye, así que undistortamos las
    observaciones a coords pinhole normalizadas, resolvemos PnP
    contra una cámara identidad, después reproyectamos a través del
    modelo fisheye y medimos el residual en pixels contra las
    observaciones originales. Devuelve NaN ante cualquier falla
    de OpenCV.
    """
    try:
        # Normalizar las observaciones: fisheye.undistortPoints
        # devuelve (N,1,2) en coords de imagen normalizadas (lo que
        # vería una cámara pinhole con K=I).
        undistorted = cv2.fisheye.undistortPoints(
            img_pts.reshape(-1, 1, 2).astype(np.float64),
            K,
            D,
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts.astype(np.float64),
            undistorted,
            np.eye(3, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return float("nan")
        proj, _ = cv2.fisheye.projectPoints(
            obj_pts.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            K,
            D,
        )
        err = np.linalg.norm(
            proj.reshape(-1, 2) - img_pts.reshape(-1, 2),
            axis=1,
        )
        return float(np.sqrt((err**2).mean()))
    except cv2.error:
        return float("nan")


def compute_per_pair_residuals(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
    calibration: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    """Computa el residual de reproyección para cada par tras la
    calibración.

    Devuelve un dict por par: {pair_idx, rms_l, rms_r, rms,
    n_corners}. Se usa para flaggear capturas outlier en el reporte
    de calibración así el operador puede ver qué poses arrastraron
    el RMS global hacia arriba.
    """
    K_l = calibration["camera_matrix_l"]
    D_l = calibration["dist_coeffs_l"]
    K_r = calibration["camera_matrix_r"]
    D_r = calibration["dist_coeffs_r"]
    board_corners_3d = board.getChessboardCorners()

    results: list[dict[str, float]] = []
    for idx, (img_l, img_r) in enumerate(image_pairs):
        corners_l, ids_l = detect_charuco_corners(img_l, board, lenient=True)
        corners_r, ids_r = detect_charuco_corners(img_r, board, lenient=True)
        if corners_l is None or corners_r is None or ids_l is None or ids_r is None:
            results.append(
                {
                    "pair_idx": idx,
                    "rms_l": float("nan"),
                    "rms_r": float("nan"),
                    "rms": float("nan"),
                    "n_corners": 0,
                }
            )
            continue

        common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
        if len(common_ids) < 6:
            results.append(
                {
                    "pair_idx": idx,
                    "rms_l": float("nan"),
                    "rms_r": float("nan"),
                    "rms": float("nan"),
                    "n_corners": int(len(common_ids)),
                }
            )
            continue

        mask_l = np.isin(ids_l.flatten(), common_ids)
        mask_r = np.isin(ids_r.flatten(), common_ids)
        order_l = np.argsort(ids_l[mask_l].flatten())
        order_r = np.argsort(ids_r[mask_r].flatten())

        obj_pts = board_corners_3d[common_ids].astype(np.float32)
        pts_l = corners_l[mask_l][order_l].reshape(-1, 1, 2).astype(np.float32)
        pts_r = corners_r[mask_r][order_r].reshape(-1, 1, 2).astype(np.float32)

        rms_l = _fisheye_reprojection_rms(obj_pts, pts_l, K_l, D_l)
        rms_r = _fisheye_reprojection_rms(obj_pts, pts_r, K_r, D_r)

        combined = float("nan")
        vals = [v for v in (rms_l, rms_r) if not math.isnan(v)]
        if vals:
            combined = sum(vals) / len(vals)

        results.append(
            {
                "pair_idx": idx,
                "rms_l": rms_l,
                "rms_r": rms_r,
                "rms": combined,
                "n_corners": int(len(common_ids)),
            }
        )
    return results


def alignment_hint(error: dict[str, float]) -> str:
    """Indicación human-readable para que el operador mueva el board."""
    parts = []
    if error["center_px"] > ALIGN_CENTER_TOL_PX:
        if abs(error["offset_x"]) > abs(error["offset_y"]):
            direction = "derecha" if error["offset_x"] < 0 else "izquierda"
            parts.append(f"Movelo {direction} {abs(error['offset_x']):.0f}px")
        else:
            direction = "abajo" if error["offset_y"] < 0 else "arriba"
            parts.append(f"Movelo {direction} {abs(error['offset_y']):.0f}px")
    scale_off = error["scale_ratio"] - 1.0
    if abs(scale_off) > ALIGN_SCALE_TOL:
        parts.append("Alejalo" if scale_off > 0 else "Acercalo")
    if abs(error["rotation_deg"]) > ALIGN_ROTATION_TOL_DEG:
        direction = "horario" if error["rotation_deg"] < 0 else "antihorario"
        parts.append(f"Rotá {direction} {abs(error['rotation_deg']):.0f}°")
    return " · ".join(parts) if parts else "Alineado"


# ---------------------------------------------------------------------------
# Estabilidad: buffer rolling de posiciones de esquina, chequea
# movimiento entre frames
# ---------------------------------------------------------------------------


STABILITY_MAX_DISP_PX = 1.5
STABILITY_WINDOW_FRAMES = 10


class StabilityTracker:
    """Buffer rolling de posiciones de esquinas detectadas para
    detectar cuando el board está quieto.

    Llamar push() cada frame con el array de esquinas detectadas
    (preview px). Llamar is_stable() para chequear si los últimos
    `window` frames tienen un displacement inter-frame máximo por
    debajo del threshold.

    Cuando se proveen IDs, el displacement se mide sobre la
    INTERSECCIÓN de IDs de esquinas entre frames. Iluminación
    marginal a menudo hace que los counts de detección fluctúen
    (23 ↔ 35 esquinas) aunque el board esté quieto — el
    comportamiento previo de "reset al cambio de shape" hacía la
    estabilidad inalcanzable en esas condiciones, aunque el board
    no se estuviese moviendo.
    """

    def __init__(
        self,
        window: int = STABILITY_WINDOW_FRAMES,
        max_disp_px: float = STABILITY_MAX_DISP_PX,
    ) -> None:
        self.window = window
        self.max_disp_px = max_disp_px
        # Cada entry es (posiciones Nx2, ids array 1D o None cuando
        # el caller no pasó IDs). Cuando ids es None caemos al
        # comportamiento legacy de requerir shapes idénticas entre
        # frames.
        self._buffer: list[tuple[np.ndarray, Optional[np.ndarray]]] = []

    def reset(self) -> None:
        self._buffer.clear()

    def push(
        self,
        corners_preview: Optional[np.ndarray],
        ids: Optional[np.ndarray] = None,
    ) -> None:
        """Agrega las posiciones de esquinas de un frame (e IDs
        opcionales). Pasar None si la detección falló."""
        if corners_preview is None or len(corners_preview) == 0:
            self._buffer.clear()
            return
        pts = np.asarray(corners_preview).reshape(-1, 2).astype(np.float32)
        ids_arr: Optional[np.ndarray]
        if ids is not None:
            ids_arr = np.asarray(ids).reshape(-1).astype(int)
        else:
            ids_arr = None
        # Fallback legacy: cuando no se pasan IDs no podemos
        # intersectar, así que mantenemos la semántica vieja
        # "restart al cambio de count".
        if self._buffer and ids_arr is None and self._buffer[-1][0].shape != pts.shape:
            self._buffer.clear()
        self._buffer.append((pts, ids_arr))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def max_displacement(self) -> float:
        """Displacement máximo per-corner entre frames consecutivos en
        el buffer.

        Con IDs disponibles, compara solo esquinas que aparecen en
        ambos frames de cada par consecutivo — tolerante a
        drop-in/drop-out de detección sin perder precisión en las
        esquinas que SÍ persisten."""
        if len(self._buffer) < 2:
            return float("inf")
        worst = 0.0
        for (prev_pts, prev_ids), (curr_pts, curr_ids) in zip(
            self._buffer[:-1],
            self._buffer[1:],
        ):
            if prev_ids is not None and curr_ids is not None:
                common, prev_idx, curr_idx = np.intersect1d(
                    prev_ids,
                    curr_ids,
                    return_indices=True,
                )
                if common.size == 0:
                    return float("inf")
                disp = np.linalg.norm(
                    curr_pts[curr_idx] - prev_pts[prev_idx],
                    axis=1,
                ).max()
            else:
                if prev_pts.shape != curr_pts.shape:
                    return float("inf")
                disp = np.linalg.norm(curr_pts - prev_pts, axis=1).max()
            worst = max(worst, float(disp))
        return worst

    def is_stable(self) -> bool:
        if len(self._buffer) < self.window:
            return False
        return self.max_displacement() <= self.max_disp_px


# ---------------------------------------------------------------------------
# Gate de calidad de frame (blur, count de esquinas, saturación,
# exposición)
# ---------------------------------------------------------------------------


QUALITY_MIN_CORNERS = 15
QUALITY_MIN_BLUR = (
    10.0  # Varianza Laplaciana sobre grayscale (capturas full-res 4608x2592).
)
# A 12MP cada pixel es más fino así que los gradients per-pixel son
# más chicos; 10 es el piso empírico para un frame bien enfocado.
# La sharpness local de esquinas (check separado abajo) maneja el
# caso "se ve sharp globalmente pero las esquinas están blandas".
QUALITY_MAX_SATURATION_PCT = 8.0  # % de pixels >250
QUALITY_MIN_EXPOSURE = 35.0  # valor mediano de grayscale
QUALITY_MAX_LR_BRIGHTNESS_PCT = 25.0  # % de asimetría entre L y R
QUALITY_MIN_CORNER_SHARPNESS = 40.0  # var Laplaciana media en ventana 15x15 por esquina
QUALITY_CORNER_WINDOW = 15


def corner_local_sharpness(gray: np.ndarray, corners: Optional[np.ndarray]) -> float:
    """Sharpness media de varianza Laplaciana en una ventana 15×15
    alrededor de cada esquina.

    Agarra capturas con motion smearing donde el Laplaciano del
    frame entero pasa pero las regiones localizadas de esquinas
    están blandas (común con boards en mano). Devuelve NaN si no se
    pasan esquinas.
    """
    if corners is None or len(corners) == 0:
        return float("nan")
    h, w = gray.shape[:2]
    half = QUALITY_CORNER_WINDOW // 2
    pts = corners.reshape(-1, 2)
    scores: list[float] = []
    for x, y in pts:
        xi, yi = int(round(float(x))), int(round(float(y)))
        x1, x2 = max(0, xi - half), min(w, xi + half + 1)
        y1, y2 = max(0, yi - half), min(h, yi + half + 1)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        scores.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def assess_frame_quality(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    n_corners: int,
    corners_l: Optional[np.ndarray] = None,
    corners_r: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Evalúa un par de frames capturados. Devuelve checks + pass/fail
    + reasons.

    corners_l/corners_r (opcional) en coords pixel full-res. Si se
    proveen, corre un check de sharpness local per-corner encima del
    Laplaciano del frame entero — rechaza pares que pasan el test
    global de blur pero tienen esquinas blandas.
    """
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY) if frame_l.ndim == 3 else frame_l
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if frame_r.ndim == 3 else frame_r

    blur_l = float(cv2.Laplacian(gray_l, cv2.CV_64F).var())
    blur_r = float(cv2.Laplacian(gray_r, cv2.CV_64F).var())
    sat_l = float((gray_l > 250).mean() * 100)
    sat_r = float((gray_r > 250).mean() * 100)
    exp_l = float(np.median(gray_l))
    exp_r = float(np.median(gray_r))
    lr_bright_diff = abs(exp_l - exp_r) / max(exp_l, exp_r, 1) * 100

    corner_sharp_l = corner_local_sharpness(gray_l, corners_l)
    corner_sharp_r = corner_local_sharpness(gray_r, corners_r)
    corner_sharp_ok_l = (
        math.isnan(corner_sharp_l) or corner_sharp_l >= QUALITY_MIN_CORNER_SHARPNESS
    )
    corner_sharp_ok_r = (
        math.isnan(corner_sharp_r) or corner_sharp_r >= QUALITY_MIN_CORNER_SHARPNESS
    )

    checks = {
        "corners": n_corners >= QUALITY_MIN_CORNERS,
        "blur_l": blur_l >= QUALITY_MIN_BLUR,
        "blur_r": blur_r >= QUALITY_MIN_BLUR,
        "corner_sharp_l": corner_sharp_ok_l,
        "corner_sharp_r": corner_sharp_ok_r,
        "saturation_l": sat_l <= QUALITY_MAX_SATURATION_PCT,
        "saturation_r": sat_r <= QUALITY_MAX_SATURATION_PCT,
        "exposure_l": exp_l >= QUALITY_MIN_EXPOSURE,
        "exposure_r": exp_r >= QUALITY_MIN_EXPOSURE,
        "lr_balance": lr_bright_diff <= QUALITY_MAX_LR_BRIGHTNESS_PCT,
    }
    reasons: list[str] = []
    if not checks["corners"]:
        reasons.append(f"Pocas esquinas ({n_corners}<{QUALITY_MIN_CORNERS})")
    if not checks["blur_l"]:
        reasons.append(
            f"Imagen izquierda borrosa ({blur_l:.0f}<{QUALITY_MIN_BLUR:.0f})"
        )
    if not checks["blur_r"]:
        reasons.append(f"Imagen derecha borrosa ({blur_r:.0f}<{QUALITY_MIN_BLUR:.0f})")
    if not checks["corner_sharp_l"]:
        reasons.append(
            f"Esquinas izquierda blandas ({corner_sharp_l:.0f}<{QUALITY_MIN_CORNER_SHARPNESS:.0f})"
        )
    if not checks["corner_sharp_r"]:
        reasons.append(
            f"Esquinas derecha blandas ({corner_sharp_r:.0f}<{QUALITY_MIN_CORNER_SHARPNESS:.0f})"
        )
    if not checks["saturation_l"]:
        reasons.append(f"Brillo/reflejo en izquierda ({sat_l:.1f}%)")
    if not checks["saturation_r"]:
        reasons.append(f"Brillo/reflejo en derecha ({sat_r:.1f}%)")
    if not checks["exposure_l"]:
        reasons.append(f"Izquierda muy oscura (mediana {exp_l:.0f})")
    if not checks["exposure_r"]:
        reasons.append(f"Derecha muy oscura (mediana {exp_r:.0f})")
    if not checks["lr_balance"]:
        weaker = "Izquierda" if exp_l < exp_r else "Derecha"
        reasons.append(f"{weaker} más oscura ({lr_bright_diff:.0f}% diff)")

    return {
        "checks": checks,
        "blur_l": blur_l,
        "blur_r": blur_r,
        "corner_sharp_l": corner_sharp_l,
        "corner_sharp_r": corner_sharp_r,
        "sat_l": sat_l,
        "sat_r": sat_r,
        "exp_l": exp_l,
        "exp_r": exp_r,
        "lr_bright_diff": lr_bright_diff,
        "n_corners": n_corners,
        "all_pass": all(checks.values()),
        "reasons": reasons,
    }


# Advisories de iluminación en tiempo real (subset de calidad —
# warnings durante el aiming, no bloquean la captura).


def live_lighting_warnings(frame_l: np.ndarray, frame_r: np.ndarray) -> list[str]:
    """Warnings no bloqueantes mostrados mientras el operador todavía
    está posicionando."""
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY) if frame_l.ndim == 3 else frame_l
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if frame_r.ndim == 3 else frame_r
    sat_l = float((gray_l > 250).mean() * 100)
    sat_r = float((gray_r > 250).mean() * 100)
    exp_l = float(np.median(gray_l))
    exp_r = float(np.median(gray_r))
    lr_diff = abs(exp_l - exp_r) / max(exp_l, exp_r, 1) * 100

    warnings: list[str] = []
    if sat_l > QUALITY_MAX_SATURATION_PCT or sat_r > QUALITY_MAX_SATURATION_PCT:
        warnings.append("Reflejo/brillo detectado — inclinalo o movelo")
    if exp_l < QUALITY_MIN_EXPOSURE or exp_r < QUALITY_MIN_EXPOSURE:
        warnings.append("Escena muy oscura — aumentá iluminación")
    if lr_diff > QUALITY_MAX_LR_BRIGHTNESS_PCT:
        weaker = "izquierda" if exp_l < exp_r else "derecha"
        warnings.append(f"Cámara {weaker} más oscura — revisá obstrucción")
    return warnings

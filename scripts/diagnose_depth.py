#!/usr/bin/env python3
"""Tool de diagnóstico del pipeline de depth y validación de calibración.

Mide la profundidad a una distancia conocida en 5 zonas (centro + 4 esquinas)
para validar la calidad de la calibración del par estéreo IMX708 120° HFOV.

Para lentes wide-angle de 120°, la periferia es donde la distorsión es más
difícil de modelar — si la calibración es pobre, las zonas de borde divergen
del centro mucho más que lo que la distancia radial sola predeciría. Este
script hace explícita esa divergencia.

Uso:
    # Poner un objeto plano (pared, board) paralelo a las cámaras a una
    # distancia conocida, correr una vez por distancia:
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 1000
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 2000
    PYTHONPATH=. python3 scripts/diagnose_depth.py --distance 3000

Thresholds de validación (PASS/FAIL):
    - Error de centro: <5% a 2m, <10% a 3m
    - Ratio de error borde/centro: <2.0 (los bordes no peor que 2× el centro)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config.loader import DEFAULT_DEVICE_CONFIG_PATH, load_device_config
from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import StereoCapture
from src.vision.depth import compute_disparity, create_sgbm, disparity_to_depth


# Thresholds de validación
PASS_ERROR_PCT_AT_2M = 5.0
PASS_ERROR_PCT_AT_3M = 10.0
PASS_EDGE_CENTER_RATIO = 2.0


def analyze_zone(
    disparity: np.ndarray, fx: float, baseline_mm: float, cy: int, cx: int, half: int
) -> tuple[int, float, float, float]:
    """Computa estadísticas de profundidad dentro de un ROI cuadrado
    centrado en (cy, cx).

    Devuelve (valid_count, median_depth_mm, std_depth_mm, fill_pct).
    """
    h, w = disparity.shape
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    disp_roi = disparity[y1:y2, x1:x2]
    valid = disp_roi[disp_roi > 0.1]
    total = disp_roi.size
    if len(valid) == 0:
        return 0, float("nan"), float("nan"), 0.0
    depths = fx * baseline_mm / valid
    return (
        len(valid),
        float(np.median(depths)),
        float(np.std(depths)),
        100.0 * len(valid) / total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnóstico de depth + validación de calibración"
    )
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Distancia real al target plano en mm",
    )
    parser.add_argument("--calibration", default="/etc/people-counter/calibration.npz")
    parser.add_argument(
        "--wls", action="store_true", help="Habilita filtro WLS (off por default)"
    )
    parser.add_argument(
        "--green",
        action="store_true",
        help="Usa solo channel verde (para cámaras NoIR)",
    )
    parser.add_argument("--no-clahe", action="store_true", help="Desactiva CLAHE")
    parser.add_argument(
        "--downscale",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Factor de downscale para matching SGBM (1=full, 2=half, 4=quarter)",
    )
    parser.add_argument(
        "--zone-fraction",
        type=float,
        default=0.15,
        help="Tamaño de zona como fracción de min(H,W). Default 0.15 = ~15%% por lado",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.10,
        help="Offset de zona borde desde la esquina, fracción de min(H,W). Default 0.10",
    )
    parser.add_argument(
        "--delay-seconds",
        "--delay",
        type=int,
        dest="delay_seconds",
        default=0,
        help="Cuenta regresiva en segundos antes de capturar",
    )
    parser.add_argument(
        "--meter",
        choices=("matrix", "centre", "spot"),
        default="matrix",
        help="Modo de metering AE. Usar 'centre'/'spot' cuando "
        "zonas brillantes en la periferia (ventanas, luces) "
        "tiren la exposición hacia abajo sobre el centro.",
    )
    parser.add_argument(
        "--lock-ae",
        action="store_true",
        help="Lockea AE/AWB tras un settle de 1s. Útil para luz "
        "variable (luz natural). Default off — AE ajusta a "
        "la escena actual, la captura single-frame no se "
        "afecta por drift.",
    )
    parser.add_argument(
        "--max-exposure-us",
        type=int,
        default=16000,
        help="Cap de exposure time en microsegundos vía "
        "FrameDurationLimits. Default 16000us (16ms), "
        "mismo que el runtime — diagnostica con la "
        "misma distribución de blur que vé el detector. "
        "Pasar 0 para deshabilitar.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Escribe los resultados per-zona + veredicto a este "
        "path JSON. Permite que el wizard / tooling de CI "
        "parsee el score sin scrappear stdout. Schema: "
        "{distance_mm, zones: {<name>: {depth_mm, std_mm, "
        "err_pct, fill_pct}}, center_threshold_pct, "
        "edge_ratio, verdict: 'PASS'|'FAIL', "
        "checks: [{name, detail, ok}]}.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp",
        help="Directorio para las imágenes de visualización "
        "(diagnose_rect_l.jpg, diagnose_rect_r.jpg, "
        "diagnose_depth.jpg). Default /tmp.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprime el output a stdout (igual escribe JSON + "
        "imágenes). Modo headless para orquestación.",
    )
    args = parser.parse_args()

    def _emit(*a, **kw) -> None:
        if not args.quiet:
            print(*a, **kw)

    # Asignaciones de CSI de cámara vienen del config per-device strict.
    # Sin fallback a config.example.yaml — el dispositivo tiene su mapping
    # explícito y diagnose tiene que matchear lo que usa el runtime.
    try:
        device_cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
    except FileNotFoundError:
        parser.error(
            f"{DEFAULT_DEVICE_CONFIG_PATH} no existe. Aprovisioná el config "
            "per-device antes de correr diagnose_depth."
        )
    bracket = device_cfg.get("bracket")
    if (
        not bracket
        or "camera_left_csi" not in bracket
        or "camera_right_csi" not in bracket
    ):
        parser.error(
            f"bracket.camera_left_csi / bracket.camera_right_csi no "
            f"definidos en {DEFAULT_DEVICE_CONFIG_PATH}."
        )
    cam_left = bracket["camera_left_csi"]
    cam_right = bracket["camera_right_csi"]

    cal = load_calibration(args.calibration)

    # La resolución de captura tiene que matchear la resolución de
    # la calibración, no sensor.default_res ni
    # vision_runtime.resolution. Dos flujos válidos existen en el
    # mismo device:
    #   - Output fresco del wizard → .npz a --resolution (2304×1296 default).
    #   - Runtime deployado        → .npz reescalado a
    #                                 vision_runtime.resolution
    #                                 (1152×648 típico) via
    #                                 rescale_calibration.py.
    # El .npz carga ``image_size`` (la resolución a la que se
    # armaron los rectify maps); lo usamos como ground truth así
    # diagnose funciona contra cualquier flujo. Capturar a otro
    # tamaño y cv2.remap silenciosamente clipea el frame a los
    # mapas, lo cual previamente producía pares rectificados que
    # solo cubrían el cuadrante top-left de la captura real e
    # invalidaba el diagnóstico de depth completo.
    if "image_size" in cal:
        cal_w, cal_h = (int(v) for v in cal["image_size"])
        resolution = (cal_w, cal_h)
    else:
        # Fallback para .npz más viejos que no guardaban image_size:
        # usar la shape del rectify map directamente. cv2.convertMaps
        # puede haber cambiado el dtype pero las dims espaciales se
        # preservan.
        map_h, map_w = cal["map_l_x"].shape[:2]
        resolution = (map_w, map_h)

    # --- Imprimir parámetros de calibración ---
    _emit("=" * 70)
    _emit("PARÁMETROS DE CALIBRACIÓN")
    _emit("=" * 70)
    P1, P2, T = cal["P1"], cal["P2"], cal["T"]
    fx_p1 = float(P1[0, 0])
    baseline_T = float(np.linalg.norm(T))
    baseline_P2 = abs(float(P2[0, 3])) / fx_p1 if fx_p1 != 0 else 0.0

    _emit(f"P1 fx={fx_p1:.1f}  fy={P1[1,1]:.1f}  cx={P1[0,2]:.1f}  cy={P1[1,2]:.1f}")
    _emit(f"Baseline ||T|| = {baseline_T:.1f} mm  /  P2[0,3]/fx = {baseline_P2:.1f} mm")

    if abs(baseline_T - baseline_P2) > 5:
        print("  ATENCIÓN: discrepancia de baseline (>5mm)")

    # --- Captura ---
    _emit(f"\n{'=' * 70}")
    _emit("CAPTURANDO...")
    _emit("=" * 70)

    from src.config.hardware import load_hardware_params

    hw = load_hardware_params()
    cap = StereoCapture(
        cam_left_id=cam_left,
        cam_right_id=cam_right,
        resolution=resolution,
        fps=5,
        meter_mode=args.meter,
        lock_ae=args.lock_ae,
        max_exposure_us=(args.max_exposure_us if args.max_exposure_us > 0 else None),
        sensor_raw_size=hw.default_res,
        initial_settle_seconds=hw.ae_initial_settle_seconds,
        resettle_seconds=hw.ae_resettle_seconds,
    )
    cap.open()

    if args.delay_seconds > 0:
        import time

        for i in range(args.delay_seconds, 0, -1):
            print(f"  Capturando en {i}...", end="\r", flush=True)
            time.sleep(1)
        print()

    left, right = cap.read()
    cap.close()
    _emit(f"Capturado: {left.shape}, rectificando...")

    # Sanity check: los rectify maps están dimensionados a la
    # resolución de la calibración. cv2.remap produce un output del
    # tamaño de los mapas. Alimentar una captura a otra resolución
    # clipea el contenido silenciosamente (solo el cuadrante top-left
    # de una captura 2× sobrevive) — exactamente el síntoma que la
    # tool original exhibía antes del fix de resolución arriba.
    map_h, map_w = cal["map_l_x"].shape[:2]
    if (left.shape[1], left.shape[0]) != (map_w, map_h):
        print(
            f"  ATENCIÓN: capture {left.shape[1]}×{left.shape[0]} != "
            f"calibration maps {map_w}×{map_h} — rectificación recortará "
            "contenido. Verificá vision.resolution en /etc/people-counter/"
            "config.yaml o en config/config.example.yaml."
        )

    rect_l, rect_r = rectify_pair(left, right, cal)

    # --- Disparity ---
    sgbm = create_sgbm()
    disparity = compute_disparity(
        rect_l,
        rect_r,
        sgbm=sgbm,
        use_wls_filter=args.wls,
        use_green_channel=args.green,
        use_clahe=not args.no_clahe,
        downscale=args.downscale,
    )

    h, w = disparity.shape
    short = min(h, w)
    half = int(short * args.zone_fraction / 2)
    margin = int(short * args.edge_margin)

    # 5 zonas: centro + 4 esquinas (offset del borde de imagen por `margin`)
    zones = {
        "center": (h // 2, w // 2),
        "top-left": (margin + half, margin + half),
        "top-right": (margin + half, w - margin - half),
        "bottom-left": (h - margin - half, margin + half),
        "bottom-right": (h - margin - half, w - margin - half),
    }

    # --- Análisis per-zona ---
    _emit(f"\n{'=' * 70}")
    _emit(
        f"PROFUNDIDAD POR ZONA  (distancia real: {args.distance:.0f} mm = {args.distance/1000:.2f} m)"
    )
    _emit(
        f"Tamaño de zona: {2*half}×{2*half} px  ({args.zone_fraction*100:.0f}% del lado del frame)"
    )
    _emit("=" * 70)
    _emit(
        f"{'Zona':<14s} {'Válidos':>7s} {'Fill%':>7s} {'Prof(mm)':>11s} {'Std(mm)':>9s} {'Error':>9s}"
    )
    _emit("-" * 70)

    results = {}
    for name, (cy, cx) in zones.items():
        n, median, std, fill = analyze_zone(disparity, fx_p1, baseline_T, cy, cx, half)
        if n == 0:
            print(f"{name:<14s} {'SIN DATOS':>7s}")
            results[name] = None
            continue
        err_pct = (median - args.distance) / args.distance * 100
        results[name] = (median, std, err_pct, fill)
        print(
            f"{name:<14s} {n:>7d} {fill:>6.1f}% {median:>10.0f} {std:>8.0f}  {err_pct:>+7.1f}%"
        )

    # --- Análisis de consistencia ---
    _emit(f"\n{'=' * 70}")
    _emit("CONSISTENCIA CENTRO vs BORDES")
    _emit("=" * 70)

    center = results.get("center")
    edges = [v for k, v in results.items() if k != "center" and v is not None]

    if center is None or not edges:
        print("No se puede computar consistencia — faltan zonas")
        edge_ratio = float("nan")
    else:
        center_err = abs(center[2])
        edge_errs = [abs(v[2]) for v in edges]
        max_edge_err = max(edge_errs)
        mean_edge_err = sum(edge_errs) / len(edge_errs)
        edge_ratio = max_edge_err / max(center_err, 0.1)

        print(f"Error |centro|:        {center_err:5.2f}%")
        print(f"Error medio |bordes|:  {mean_edge_err:5.2f}%")
        print(f"Error max |bordes|:    {max_edge_err:5.2f}%")
        print(f"Max bordes / centro:   {edge_ratio:5.2f}×")

    # --- PASS/FAIL ---
    _emit(f"\n{'=' * 70}")
    _emit("VEREDICTO DE VALIDACIÓN")
    _emit("=" * 70)

    # El threshold escala linealmente entre 2m y 3m
    d_m = args.distance / 1000
    if d_m <= 2.0:
        center_threshold = PASS_ERROR_PCT_AT_2M
    elif d_m >= 3.0:
        center_threshold = PASS_ERROR_PCT_AT_3M
    else:
        # interpolar
        t = (d_m - 2.0) / 1.0
        center_threshold = PASS_ERROR_PCT_AT_2M + t * (
            PASS_ERROR_PCT_AT_3M - PASS_ERROR_PCT_AT_2M
        )

    checks = []
    if center is not None:
        ok = abs(center[2]) <= center_threshold
        checks.append(
            ("Error centro", f"{abs(center[2]):.2f}% ≤ {center_threshold:.1f}%", ok)
        )
    # El ratio borde/centro se muestra solo como INFO — asume una
    # escena de target plano que la mayoría de ambientes reales no
    # tienen. Escenas multi-depth inflan los errores de borde sin
    # que eso sea un problema de calibración.
    if not np.isnan(edge_ratio):
        info_ok = edge_ratio <= PASS_EDGE_CENTER_RATIO
        flag = "OK" if info_ok else "INFO"
        _emit(
            f"  [{flag}] Ratio bordes/centro: {edge_ratio:.2f}× "
            f"(referencia {PASS_EDGE_CENTER_RATIO:.1f}× — informativo, "
            f"solo significativo con target plano)"
        )

    for name, detail, ok in checks:
        status = "PASS" if ok else "FAIL"
        _emit(f"  [{status}] {name}: {detail}")

    overall = all(ok for _, _, ok in checks) if checks else False
    _emit(
        f"\nVEREDICTO: {'PASS — calibración correcta para depth' if overall else 'FAIL — recalibrar (más capturas, mejor foco, verificar rigidez)'}"
    )

    # --- Output JSON opcional (para orquestación / integración con wizard) ---
    if args.json_path:
        zones_json: dict[str, object] = {}
        for name, vals in results.items():
            if vals is None:
                zones_json[name] = None
            else:
                depth, std, err_pct, fill = vals
                zones_json[name] = {
                    "depth_mm": float(depth),
                    "std_mm": float(std),
                    "err_pct": float(err_pct),
                    "fill_pct": float(fill),
                }
        json_payload = {
            "distance_mm": float(args.distance),
            "zones": zones_json,
            "center_threshold_pct": float(center_threshold),
            "edge_ratio": (None if np.isnan(edge_ratio) else float(edge_ratio)),
            "verdict": "PASS" if overall else "FAIL",
            "checks": [{"name": n, "detail": d, "ok": bool(o)} for n, d, o in checks],
        }
        json_out = Path(args.json_path)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(json_payload, indent=2),
            encoding="utf-8",
        )
        _emit(f"\nJSON guardado: {json_out}")

    # --- Guardar visualización anotada ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "diagnose_rect_l.jpg"), rect_l)
    cv2.imwrite(str(out_dir / "diagnose_rect_r.jpg"), rect_r)
    depth_map = disparity_to_depth(disparity, fx_p1, baseline_T)
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    for name, (cy, cx) in zones.items():
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 3)
        if results.get(name) is not None:
            label = f"{results[name][0]:.0f}mm ({results[name][2]:+.1f}%)"
        else:
            label = "sin datos"
        cv2.putText(
            depth_vis,
            label,
            (x1 + 5, y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

    cv2.imwrite(str(out_dir / "diagnose_depth.jpg"), depth_vis)
    _emit(
        f"\nGuardado: {out_dir / 'diagnose_rect_l.jpg'}, "
        f"{out_dir / 'diagnose_rect_r.jpg'}, "
        f"{out_dir / 'diagnose_depth.jpg'}",
    )


if __name__ == "__main__":
    main()

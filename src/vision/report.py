"""Generador de reporte HTML de calibración.

Produce un único archivo HTML self-contained por dispositivo con:
- RMS de calibración + focal + baseline vs diseño (140mm) con alerta de tolerancia
- Validación de profundidad en 5 zonas PASS/FAIL
- Thumbnails embedded en base64 de todos los pares capturados con corner count
- Timestamp + device id para audit trail
"""

from __future__ import annotations

import base64
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DESIGN_BASELINE_MM = 140.0
BASELINE_WARN_TOL_MM = 5.0


def _img_to_base64_jpeg(img: np.ndarray, max_w: int = 320, quality: int = 60) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _pill(text: str, ok: bool) -> str:
    color = "#2ecc71" if ok else "#e74c3c"
    label = "PASS" if ok else "FAIL"
    return f'<span class="pill" style="background:{color}">{label}</span> {text}'


def generate_html_report(
    calibration: dict,
    diagnose_zones: Optional[dict] = None,
    capture_pairs: Optional[list[tuple[Path, Path, str, int, int]]] = None,
    device_id: str = "unknown",
    rms_stereo: Optional[float] = None,
    timestamp: Optional[_dt.datetime] = None,
    per_pair_residuals: Optional[list[dict[str, float]]] = None,
    epipolar_image: Optional[Path] = None,
    ground_truth_image: Optional[Path] = None,
) -> str:
    """Construye un string de reporte HTML self-contained.

    Args:
        calibration: dict con camera_matrix_l/r, T, P1, info de baseline
        diagnose_zones: dict opcional de diagnose_depth: {zone_name: (depth, std, err_pct, fill)}
                        más bool "_pass", float "_edge_ratio", float "_center_err",
                        y "_distance_mm" ground truth.
        capture_pairs: lista de (left_path, right_path, pose_id, n_corners_l, n_corners_r)
        device_id: identificador del dispositivo para el header del reporte
        rms_stereo: RMS general (px) de la calibración estéreo si se conoce
        timestamp: override del timestamp; default a ahora
    """
    ts = timestamp or _dt.datetime.now()

    K_l = calibration["camera_matrix_l"]
    K_r = calibration.get("camera_matrix_r", K_l)
    T = calibration["T"]
    fx_l = float(K_l[0, 0])
    fy_l = float(K_l[1, 1])
    baseline_mm = float(abs(T[0, 0]))
    baseline_delta = baseline_mm - DESIGN_BASELINE_MM
    baseline_ok = abs(baseline_delta) <= BASELINE_WARN_TOL_MM

    rms_text = f"{rms_stereo:.3f} px" if rms_stereo is not None else "—"

    # Métricas de lens alignment — descomposición operator-friendly de
    # R+T en offset_{x,y,z}_mm + rotation_{x,y,z}_deg. Solo diagnóstico,
    # no gate el reporte. El pipeline runtime rectifica cualquier
    # misalignment; los exponemos para QA cross-unit (detectar
    # outliers de fabricación del bracket comparando varios devices
    # lado a lado).
    alignment_html = ""
    R = calibration.get("R")
    if R is not None:
        try:
            from src.vision.calibration import lens_alignment_metrics
            am = lens_alignment_metrics(R, T)
            alignment_html = f"""
<h2>Alineamiento entre lentes</h2>
<p style="color:#555;font-size:13px">Geometría relativa entre el lente izquierdo y el derecho (descomposición de
R+T). <b>Diagnóstico de fabricación del bracket</b> — el runtime corrige cualquier desviación vía rectificación, así
que estos números no gate el resultado. Útil para comparar unidades entre sí y detectar outliers de fabricación.
Bracket bien armado: pitch/roll &lt;0.5°, yaw &lt;1°, offsets Y/Z &lt;2 mm.</p>
<div class="kv">
<b>Offset X (baseline)</b><span>{am['offset_x_mm']:+.2f} mm</span>
<b>Offset Y (vertical)</b><span>{am['offset_y_mm']:+.2f} mm</span>
<b>Offset Z (fore-aft)</b><span>{am['offset_z_mm']:+.2f} mm</span>
<b>Pitch (rotación X)</b><span>{am['rotation_x_deg']:+.3f}°</span>
<b>Yaw (rotación Y)</b><span>{am['rotation_y_deg']:+.3f}°</span>
<b>Roll (rotación Z)</b><span>{am['rotation_z_deg']:+.3f}°</span>
</div>
"""
        except Exception:
            logger.exception("lens_alignment_metrics failed; skipping section")

    # Sección de validación de depth
    depth_html = ""
    depth_pass = None
    if diagnose_zones:
        depth_pass = diagnose_zones.get("_pass", False)
        distance_gt = diagnose_zones.get("_distance_mm", 0)
        edge_ratio = diagnose_zones.get("_edge_ratio", float("nan"))
        center_err = diagnose_zones.get("_center_err", float("nan"))
        center_threshold = diagnose_zones.get("_center_threshold", 5.0)

        center_class = "good" if center_err <= center_threshold else "bad"
        verdict_pill = _pill("PASS Depth validation", True) if depth_pass \
            else _pill("FAIL Depth validation", False)

        # Clasificación de confiabilidad per-zona: la lectura de
        # depth de una zona es confiable cuando std es bajo (SGBM
        # matcheó limpio) y el fill es alto (la mayoría de los
        # pixels matchearon). Lecturas no confiables (std enorme,
        # fill bajo) significan que SGBM falló en esa zona — usualmente
        # superficie smooth/specular (monitor, vidrio) o ruido de luz
        # baja. Ocultamos los números de depth/error de zonas no
        # confiables porque no son significativos.
        def _reliability(std_mm: float, fill_pct: float) -> str:
            std_factor = max(0.0, 1.0 - std_mm / 2000.0)
            score = std_factor * (fill_pct / 100.0)
            if score >= 0.40:
                return "reliable"
            if score >= 0.15:
                return "partial"
            return "unreliable"

        center_row_html = ""
        edge_rows_html: list[str] = []
        for name, vals in diagnose_zones.items():
            if name.startswith("_"):
                continue

            is_center = name == "center"
            row_class = "row-center" if is_center else ""

            if vals is None:
                edge_rows_html.append(
                    f'<tr class="row-unreliable"><td>{name}</td>'
                    f'<td colspan=4 class=dim>NO DATA</td>'
                    f'<td><span class="confidence-tag confidence-unreliable">'
                    f'⚠ Sin datos</span></td></tr>'
                )
                continue

            depth, std, err_pct, fill = vals
            abs_err = abs(err_pct)

            if is_center:
                # Zona del verdict — numbers completos, badge grande VERDICT
                err_color = "good" if abs_err <= center_threshold else "bad"
                row_class = "row-center"
                status_html = ('<span class="badge-verdict">VERDICT</span>')
                depth_cell = f"{depth:.0f} mm"
                err_cell = (
                    f'<span class="{err_color}">{err_pct:+.2f}%</span>'
                )
            else:
                rel = _reliability(std, fill)
                if rel == "unreliable":
                    row_class = "row-unreliable"
                    status_html = ('<span class="confidence-tag '
                                   'confidence-unreliable">⚠ SGBM falló</span>')
                    depth_cell = '<span class=dim>—</span>'
                    err_cell = '<span class=dim>—</span>'
                elif abs_err <= center_threshold:
                    # Confiable Y cerca del target
                    row_class = "row-reliable"
                    status_html = ('<span class="confidence-tag '
                                   'confidence-reliable">✓ Coincide</span>')
                    depth_cell = f"{depth:.0f} mm"
                    err_cell = f'<span class="good">{err_pct:+.2f}%</span>'
                else:
                    # Confiable (o parcial) pero midiendo otro plano
                    row_class = "row-different"
                    label = ("● Otro plano (precisión limitada)"
                             if rel == "partial"
                             else "● Otro plano")
                    status_html = (f'<span class="confidence-tag '
                                   f'confidence-different">{label}</span>')
                    depth_cell = f"{depth:.0f} mm"
                    err_cell = f'<span class=dim>{err_pct:+.2f}%</span>'

            row_html = (
                f'<tr class="{row_class}"><td>{name}</td>'
                f'<td>{depth_cell}</td>'
                f'<td>{std:.0f} mm</td>'
                f'<td>{err_cell}</td>'
                f'<td>{fill:.1f}%</td>'
                f'<td>{status_html}</td></tr>'
            )
            if is_center:
                center_row_html = row_html
            else:
                edge_rows_html.append(row_html)

        all_rows = center_row_html + "".join(edge_rows_html)

        depth_html = f"""
<h2>Validación de profundidad</h2>

<div class="depth-verdict">
  <div class="verdict-grid">
    <div class="verdict-cell">
      <div class="verdict-label">Distancia real (cinta/láser)</div>
      <div class="verdict-value">{distance_gt:.0f} <span class="unit">mm</span></div>
    </div>
    <div class="verdict-cell">
      <div class="verdict-label">Error en el centro</div>
      <div class="verdict-value {center_class}">{center_err:.2f}<span class="unit">%</span></div>
      <div class="verdict-sub">umbral PASS ≤ {center_threshold:.1f}%</div>
    </div>
    <div class="verdict-cell verdict-pill-cell">
      {verdict_pill}
      <div class="verdict-sub">basado en error del centro</div>
    </div>
  </div>
</div>

<h3 class="zones-h3">Mediciones por zona</h3>
<table class="zones-table">
  <thead><tr>
    <th>Zona</th><th>Profundidad</th><th>Std (SGBM)</th>
    <th>Δ vs target</th><th>Fill</th><th>Estado</th>
  </tr></thead>
  <tbody>{all_rows}</tbody>
</table>

<p class="depth-note">
  <b>Cómo leer este cuadro</b>. La fila <b>centro</b> es el verdict — es la
  única zona cuya distancia se mide con cinta/láser. Las 4 zonas perimetrales
  miden lo que haya en esa parte del frame; no se garantiza que estén al
  mismo plano que el centro.
  <br>
  <b>✓ Coincide</b>: SGBM matcheó limpio y la profundidad cae dentro del
  threshold del centro — la zona ve el mismo target.
  <b>● Otro plano</b>: SGBM matcheó limpio pero a otra distancia — la zona
  está midiendo un objeto/pared distinto, no es error de calibración.
  <b>⚠ SGBM falló</b>: alta varianza o bajo fill — la medición no es
  significativa (escena oscura, superficie reflectante, sin textura).
  <br>
  Ratio borde/centro: <b>{edge_ratio:.2f}×</b> — solo significativo en escenas
  con target plano cubriendo todo el FOV. No gate del verdict.
</p>
"""

    # Análisis de residual per-par
    residuals_html = ""
    outlier_indices: set[int] = set()
    if per_pair_residuals:
        valid = [r for r in per_pair_residuals
                 if r["rms"] == r["rms"]  # no es NaN
                 and r["rms"] != float("inf")]
        if valid:
            rms_values = sorted(r["rms"] for r in valid)
            median_rms = rms_values[len(rms_values) // 2]
            outlier_threshold = max(2 * median_rms, 1.0)
            rows_html = []
            for r in per_pair_residuals:
                rms_v = r["rms"]
                is_nan = not (rms_v == rms_v) or rms_v == float("inf")
                is_outlier = (not is_nan) and rms_v > outlier_threshold
                if is_outlier:
                    outlier_indices.add(int(r["pair_idx"]))
                cls = "bad" if is_outlier else ("warn" if (not is_nan and rms_v > median_rms * 1.5) else "good")
                rms_text = f"{rms_v:.2f}" if not is_nan else "—"
                rms_l_text = f"{r.get('rms_l', float('nan')):.2f}" if r.get('rms_l') == r.get('rms_l') else "—"
                rms_r_text = f"{r.get('rms_r', float('nan')):.2f}" if r.get('rms_r') == r.get('rms_r') else "—"
                rows_html.append(
                    f"<tr><td>{int(r['pair_idx'])}</td><td class='{cls}'>{rms_text}</td>"
                    f"<td>{rms_l_text}</td><td>{rms_r_text}</td><td>{int(r.get('n_corners', 0))}</td></tr>"
                )
            residuals_html = f"""
<h2>Residual por par (reproyección, px)</h2>
<p>Mediana: <b>{median_rms:.2f} px</b> · Umbral outlier (2× mediana): <b>{outlier_threshold:.2f} px</b> ·
<b>{len(outlier_indices)}</b> outliers resaltados</p>
<table><thead><tr><th>#</th><th>RMS</th><th>RMS L</th><th>RMS R</th><th>Esquinas</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>
"""

    # Grid de capturas
    captures_html = ""
    if capture_pairs:
        tiles = []
        for idx, pair in enumerate(capture_pairs):
            # Backwards-compat: viejo (path, path, id, n_corners) y
            # nuevo (path, path, id, n_l, n_r). Tratar 4-tuple como
            # simétrico.
            if len(pair) == 5:
                lp, rp, pose_id, n_l, n_r = pair
            else:
                lp, rp, pose_id, n_l = pair
                n_r = n_l
            try:
                img_l = cv2.imread(str(lp))
                img_r = cv2.imread(str(rp))
                combined = np.hstack([img_l, img_r]) if (img_l is not None and img_r is not None) else None
            except Exception:
                combined = None
            b64 = _img_to_base64_jpeg(combined, max_w=480, quality=50) if combined is not None else ""
            img_tag = f'<img src="data:image/jpeg;base64,{b64}"/>' if b64 else '<div class="miss">missing</div>'
            is_outlier = idx in outlier_indices
            badge = '<span class="outlier-badge">OUTLIER</span>' if is_outlier else ''
            tile_cls = "tile outlier" if is_outlier else "tile"
            corners_text = (
                f"L={n_l}/R={n_r}" if n_l != n_r else f"{n_l}"
            )
            tiles.append(
                f'<div class="{tile_cls}">'
                f'<div class="tile-label">#{idx} · {pose_id} — {corners_text} esquinas {badge}</div>'
                f'{img_tag}</div>'
            )
        captures_html = f"""
<h2>Capturas ({len(capture_pairs)})</h2>
<div class="grid">{''.join(tiles)}</div>
"""

    baseline_note = ""
    if not baseline_ok:
        baseline_note = (
            f'<div class="alert">⚠ Baseline estimada difiere {baseline_delta:+.1f} mm '
            f'del diseño ({DESIGN_BASELINE_MM:.0f}mm, tolerancia ±{BASELINE_WARN_TOL_MM:.0f}mm).</div>'
        )

    # Imagen de verificación de rectificación (epipolar), embedded.
    epipolar_html = ""
    if epipolar_image is not None and Path(epipolar_image).exists():
        img = cv2.imread(str(epipolar_image))
        if img is not None:
            b64 = _img_to_base64_jpeg(img, max_w=960, quality=70)
            epipolar_html = f"""
<h2>Rectificación epipolar</h2>
<p style="color:#555;font-size:13px">Pares rectificados con líneas horizontales cada 30px. Las esquinas
correspondientes entre L y R deberían caer sobre la misma línea si la rectificación es correcta.</p>
<img src="data:image/jpeg;base64,{b64}" style="width:100%;max-width:960px;border-radius:6px;border:1px solid #ddd">
"""

    # Visualización de depth ground-truth, embedded.
    ground_truth_html = ""
    if ground_truth_image is not None and Path(ground_truth_image).exists():
        img = cv2.imread(str(ground_truth_image))
        if img is not None:
            b64 = _img_to_base64_jpeg(img, max_w=960, quality=70)
            ground_truth_html = f"""
<h2>Validación ground-truth — imagen</h2>
<p style="color:#555;font-size:13px">Mapa de profundidad de la escena usada para el check de distancia conocida.</p>
<img src="data:image/jpeg;base64,{b64}" style="width:100%;max-width:960px;border-radius:6px;border:1px solid #ddd">
"""

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Calibración {device_id} — {ts:%Y-%m-%d %H:%M}</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:24px auto;color:#222;padding:0 16px}}
h1{{border-bottom:3px solid #2c3e50;padding-bottom:8px}}
h2{{color:#2c3e50;margin-top:32px}}
.kv{{display:grid;grid-template-columns:240px 1fr;gap:8px 16px;margin:12px 0}}
.kv b{{color:#555}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#f5f5f5}}
.pill{{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-weight:600;font-size:12px}}
.alert{{background:#fff5e0;border-left:4px solid #e67e22;padding:10px 14px;margin:10px 0;border-radius:4px}}
.good{{color:#2ecc71;font-weight:600}}
.warn{{color:#f39c12;font-weight:600}}
.bad{{color:#e74c3c;font-weight:600}}
.dim{{color:#aaa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}}
.tile{{border:1px solid #ddd;border-radius:6px;padding:6px;background:#fafafa}}
.tile.outlier{{border:2px solid #e74c3c;background:#fff5f5}}
.tile img{{width:100%;display:block;border-radius:4px}}
.tile-label{{font-size:12px;color:#555;margin-bottom:4px}}
.outlier-badge{{background:#e74c3c;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px}}
.miss{{background:#eee;padding:30px;text-align:center;color:#999}}
/* Panel de validación de depth */
.depth-verdict{{background:#fafafa;border:1px solid #e3e3e3;border-radius:8px;padding:18px 22px;margin:14px 0 22px}}
.verdict-grid{{display:grid;grid-template-columns:1fr 1fr auto;gap:28px;align-items:center}}
.verdict-cell{{}}
.verdict-pill-cell{{text-align:right}}
.verdict-label{{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.verdict-value{{font-size:26px;font-weight:600;color:#2c3e50;line-height:1.1}}
.verdict-value .unit{{font-size:14px;font-weight:400;color:#888;margin-left:2px}}
.verdict-value.good{{color:#27ae60}}
.verdict-value.bad{{color:#e74c3c}}
.verdict-sub{{font-size:11px;color:#aaa;margin-top:4px}}
.verdict-pill-cell .pill{{font-size:13px;padding:5px 14px}}
.zones-h3{{font-size:14px;color:#555;margin-top:14px;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}}
.zones-table tr.row-center{{background:#fff8e1}}
.zones-table tr.row-center td{{font-weight:500;border-top:2px solid #f39c12;border-bottom:2px solid #f39c12}}
.zones-table tr.row-center td:first-child{{border-left:2px solid #f39c12}}
.zones-table tr.row-center td:last-child{{border-right:2px solid #f39c12}}
.zones-table tr.row-unreliable td{{color:#999}}
.badge-verdict{{background:#f39c12;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;letter-spacing:.5px}}
.confidence-tag{{font-size:11px;padding:2px 8px;border-radius:10px;display:inline-block;font-weight:500;white-space:nowrap}}
.confidence-reliable{{background:#d4edda;color:#155724}}
.confidence-different{{background:#e2e3e5;color:#383d41}}
.confidence-unreliable{{background:#fff3cd;color:#856404}}
.depth-note{{font-size:.85em;color:#666;background:#f8f8f8;padding:12px 16px;border-radius:6px;border-left:3px solid #2c3e50;margin-top:16px;line-height:1.55}}
</style></head>
<body>
<h1>Reporte de calibración — {device_id}</h1>
<p>Timestamp: <b>{ts:%Y-%m-%d %H:%M:%S}</b></p>

{baseline_note}

<h2>Parámetros</h2>
<div class="kv">
<b>RMS estéreo</b><span>{rms_text}</span>
<b>Focal izquierda (fx, fy)</b><span>{fx_l:.1f} px, {fy_l:.1f} px</span>
<b>Baseline medido</b><span>{baseline_mm:.2f} mm (diseño {DESIGN_BASELINE_MM:.0f} mm, Δ {baseline_delta:+.2f}mm) {_pill('tolerancia ±' + f'{BASELINE_WARN_TOL_MM:.0f}mm', baseline_ok)}</span>
<b>Principal point L</b><span>cx={K_l[0,2]:.1f}, cy={K_l[1,2]:.1f}</span>
<b>Principal point R</b><span>cx={K_r[0,2]:.1f}, cy={K_r[1,2]:.1f}</span>
</div>

{alignment_html}

{depth_html}

{ground_truth_html}

{residuals_html}

{epipolar_html}

{captures_html}

<p style="color:#888;font-size:12px;margin-top:40px">
Generado por people-counter · Device {device_id} · {ts.isoformat()}
</p>
</body></html>"""


def save_report(html: str, output_path: Path) -> Path:
    """Guarda un reporte HTML a disco. Crea los dirs padre si es necesario."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved", extra={"path": str(output_path)})
    return output_path

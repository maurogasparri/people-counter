"""Anonimiza JPGs best-frame antes de exportarlos fuera del dispositivo.

Cuando samples necesitan salir del dispositivo — ej: shippeados a un proveedor
externo de labeling para loops de active-learning — primero tienen que ser
anonimizados. Este script implementa dos modos:

  - **Blur targeted (preferido)**: lee el bbox de un JSON sidecar
    (``<jpg>.json`` con ``{"bbox": [x1, y1, x2, y2], ...}``) y blurrea solo
    esa área. El resto del frame mantiene el contexto de escena que hace al
    label útil.
  - **Fallback agresivo**: si no existe el sidecar, blurrea todo el frame
    y overlay los edges de Canny así el labeller igual puede ver siluetas /
    poses sin identificar a nadie.

Uso:

    python scripts/export_anonymized.py \
        --input-dir /var/lib/people-counter/best_frames \
        --output-dir ./export_anon \
        [--blur-kernel 31] \
        [--canny-low 50 --canny-high 150] \
        [--quality 90] \
        [--dry-run]

El directorio de output se crea si falta. El árbol relativo bajo el
input directory is preserved.

Failure modes:

  - Unreadable input JPG → logged + skipped (no abort).
  - Malformed sidecar JSON → fall back to aggressive blur.
  - Unwriteable output → logged, exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("export_anonymized")


# ---------------------------------------------------------------------------
# Anonymisation primitives
# ---------------------------------------------------------------------------


def _ensure_odd(k: int) -> int:
    """El kernel size del Gaussian de OpenCV tiene que ser impar. Redondea hacia arriba."""
    k = max(3, int(k))
    return k if k % 2 == 1 else k + 1


def blur_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    kernel: int = 31,
) -> np.ndarray:
    """Aplica Gaussian blur a la región del bbox; preserva el resto del frame.

    El padding queda clampeado al frame así que un bbox off-screen no
    rompe el slicing. Devuelve un array *nuevo* — el input no se muta.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        # Bbox degenerado a nada — caer a blur full-frame para errar
        # del lado seguro (esperábamos un bbox y no llegó ninguno).
        return blur_full_with_edges(image, kernel=kernel)
    out = image.copy()
    k = _ensure_odd(kernel)
    out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


def blur_full_with_edges(
    image: np.ndarray,
    kernel: int = 31,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """Blur fuerte del frame completo + edges Canny overlaid en blanco.

    Fallback agresivo para frames sin sidecar bbox: ofusca completamente
    los valores de pixel pero preserva las siluetas para labelling
    downstream de pose / shape. Devuelve una imagen BGR de 3 canales.
    """
    h, w = image.shape[:2]
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    k = _ensure_odd(kernel)
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    if blurred.ndim == 2:
        blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    # Pintamos los edges arriba del frame blureado en blanco. Single-channel
    # broadcasteado sobre BGR.
    mask = edges > 0
    out = blurred.copy()
    out[mask] = (255, 255, 255)
    # Las operaciones de arriba nunca cambian el shape, pero asertamos
    # el invariante por si algún refactor lo rompiera.
    assert out.shape[:2] == (h, w)
    return out


# ---------------------------------------------------------------------------
# Sidecar parsing
# ---------------------------------------------------------------------------


def load_sidecar(jpg_path: Path) -> Optional[dict[str, Any]]:
    """Busca ``<jpg>.json`` al lado de la imagen. Devuelve el dict parseado
    o ``None`` si no existe o está corrupto.
    """
    sidecar = jpg_path.with_suffix(jpg_path.suffix + ".json")
    if not sidecar.exists():
        # También aceptar ``<stem>.json`` (sin double-suffix). Filenames
        # menos verbosos son comunes en algunos pipelines de labelling.
        alt = jpg_path.with_suffix(".json")
        if alt.exists():
            sidecar = alt
        else:
            return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("sidecar unreadable for %s: %s", jpg_path, e)
        return None


def extract_bbox(meta: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    """Pull a bbox tuple out of a sidecar dict, or return None.

    Accepted shapes:
      - ``{"bbox": [x1, y1, x2, y2]}``
      - ``{"bbox": {"x1":..., "y1":..., "x2":..., "y2":...}}``
    """
    raw = meta.get("bbox")
    if raw is None:
        return None
    try:
        if isinstance(raw, dict):
            return (
                int(raw["x1"]),
                int(raw["y1"]),
                int(raw["x2"]),
                int(raw["y2"]),
            )
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return tuple(int(v) for v in raw)  # type: ignore[return-value]
    except (KeyError, ValueError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def anonymise_one(
    in_path: Path,
    out_path: Path,
    *,
    blur_kernel: int,
    canny_low: int,
    canny_high: int,
    quality: int,
    dry_run: bool,
) -> str:
    """Anonymise one JPG. Returns a short status: ``targeted`` /
    ``fallback`` / ``skipped`` / ``error``."""
    image = cv2.imread(str(in_path))
    if image is None:
        logger.warning("cannot read %s — skipping", in_path)
        return "error"

    meta = load_sidecar(in_path)
    bbox = extract_bbox(meta) if meta else None

    if bbox is not None:
        out_img = blur_bbox(image, bbox, kernel=blur_kernel)
        status = "targeted"
    else:
        out_img = blur_full_with_edges(
            image,
            kernel=blur_kernel,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        status = "fallback"

    if dry_run:
        logger.info("would write %s (%s)", out_path, status)
        return status

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(out_path),
        out_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        logger.warning("imwrite failed for %s", out_path)
        return "error"
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Anonymise best-frame JPGs before exporting them off-device. "
            "Targeted blur when a sidecar JSON describes the bbox; full "
            "blur + Canny edges otherwise."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        logger.error("input-dir does not exist: %s", in_dir)
        return 1

    counts = {"targeted": 0, "fallback": 0, "error": 0}
    for jpg in sorted(in_dir.rglob("*.jpg")):
        rel = jpg.relative_to(in_dir)
        out_path = out_dir / rel
        status = anonymise_one(
            jpg,
            out_path,
            blur_kernel=args.blur_kernel,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            quality=args.quality,
            dry_run=args.dry_run,
        )
        counts[status] = counts.get(status, 0) + 1

    logger.info(
        "Done. targeted=%d fallback=%d error=%d (dry_run=%s)",
        counts["targeted"],
        counts["fallback"],
        counts["error"],
        args.dry_run,
    )
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Capturador concurrente de N streams MJPEG HTTP.

Pensado para acumular validation/bootstrap data desde cámaras en sitio
(IP cameras, NVRs, embedded vendors que exponen un endpoint MJPEG). Cada
site corre en su propio thread, lee el stream MJPEG continuamente, y
guarda frames según el modo elegido:

- **random-interval** (default): un frame cada ``uniform(min,max)``
  segundos, cobertura temporal uniforme.
- **motion-trigger** (``--motion-trigger``): solo guarda cuando hay
  cambio significativo entre frames consecutivos (cv2.absdiff). Multiplica
  la fracción de frames con eventos reales.

Streams estéreo SBS (side-by-side, ambos lentes lado a lado en un solo
frame) se soportan declarando ``sbs: true`` en el site. El frame se
parte por la mitad horizontal y cada lado se procesa por separado.
Si además se da un bloque ``calibration:`` con las matrices inline se
rectifica on-the-fly usando los mapas precalculados — el output queda
listo para training sin requerir post-processing.

Resiliente por diseño:
- Si el stream se cae, reconecta solo
- Si un frame está corrupto, lo skipea
- Ctrl+C cierra todos los workers limpio

Usage:
    python scripts/training/capture_mjpeg.py \\
        --config training_data/sites.yaml \\
        --output training_data/captures \\
        --duration 3600 \\
        --motion-trigger \\
        --min-interval 5 \\
        --background-interval 600 \\
        --save-side left

Config YAML format:

    sites:
      # Mono stream — frames guardados tal cual.
      - name: site_a
        url: http://192.168.1.10/mjpg/video.mjpg

      # SBS estéreo — partido por la mitad, rectificado por lente.
      # Las matrices van embebidas para no depender de archivos externos.
      - name: site_b
        url: http://192.168.2.10/mjpg/raw.mjpg
        sbs: true
        calibration:
          ref_size: [160, 120]    # resolución a la que fueron escritas K, P_rect
          left:
            K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            D: [k1, k2, k3, k4]   # fisheye distortion
            R_rect: [[...3x3...]]
            P_rect: [[...3x4...]]
          right: { K: [...], D: [...], R_rect: [...], P_rect: [...] }

      # SBS sin calibración — partido pero sin rectificar.
      - name: site_c
        url: http://192.168.2.11/mjpg/raw.mjpg
        sbs: true

Layout del output:
    <output>/
      site_a/
        20260506_103752_motion_22fda5.jpg
        20260506_104646_bg_db7865.jpg
        ...
      site_b/
        20260507_120015_motion_L_a3f1c2.jpg   <- rectificado L (o ambos)
        ...
      capture.log     <- stats per-site appended cada 60s
"""

from __future__ import annotations

import argparse
import logging
import random
import secrets
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger("capture_mjpeg")

# Las matrices vienen embebidas en el sites.yaml por site. Cada bloque
# `calibration` declara la resolución de referencia a la que están escritas
# (típicamente 160×120 en los dumps que usamos) y nosotros escalamos K y
# P_rect linealmente al half_size real del stream. La distorsión es
# adimensional así que no se escala.


def _load_site_calibration(
    cal_block: dict[str, Any], half_size: tuple[int, int],
) -> dict[str, tuple[Any, Any]]:
    """Precomputa los maps de rectificación fisheye para un par de lentes.

    ``cal_block`` es el bloque ``calibration`` inline del sites.yaml — un
    dict con ``ref_size: [W, H]`` y ``{left, right}: {K, D, R_rect, P_rect}``.
    Escala K y P_rect de ``ref_size`` a ``half_size`` y devuelve un dict con
    ``"left"`` y ``"right"`` mapeando a ``(map1, map2)`` listos para
    ``cv2.remap``. Produce la imagen rectificada (geometría epipolar
    alineada), no solo la imagen sin distorsión.
    """
    ref_w, ref_h = cal_block["ref_size"]
    out_w, out_h = half_size
    sx = out_w / ref_w
    sy = out_h / ref_h

    def _scale_K(K_raw: list) -> np.ndarray:
        K2 = np.array(K_raw, dtype=np.float64)
        K2[0, 0] *= sx
        K2[0, 2] *= sx
        K2[1, 1] *= sy
        K2[1, 2] *= sy
        return K2

    maps: dict[str, tuple[Any, Any]] = {}
    for side in ("left", "right"):
        side_block = cal_block[side]
        K = _scale_K(side_block["K"])
        D = np.array(side_block["D"], dtype=np.float64).reshape(-1, 1)
        R = np.array(side_block["R_rect"], dtype=np.float64)
        P = _scale_K(side_block["P_rect"])
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, P, (out_w, out_h), cv2.CV_16SC2,
        )
        maps[side] = (m1, m2)
    return maps


def _split_sbs(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parte un frame estéreo side-by-side por la mitad horizontal."""
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


def _process_frame(
    frame: np.ndarray,
    sbs: bool,
    rect_maps: Optional[dict[str, tuple[Any, Any]]],
    save_side: str,
) -> list[tuple[str, np.ndarray]]:
    """Aplica split SBS + rectificación opcional, devuelve imágenes guardables.

    Devuelve una lista de tuplas ``(side_label, image)``. ``side_label`` es
    ``""`` para streams mono, ``"L"`` o ``"R"`` para estéreo. El label queda
    en el filename para que el dataset sea autodescriptivo.
    """
    if sbs:
        left, right = _split_sbs(frame)
    else:
        left, right = frame, None

    out: list[tuple[str, np.ndarray]] = []
    want_left = save_side in ("left", "both")
    want_right = sbs and save_side in ("right", "both")

    if want_left:
        if rect_maps is not None:
            m1, m2 = rect_maps["left"]
            left = cv2.remap(left, m1, m2, cv2.INTER_LINEAR)
        out.append(("L" if sbs else "", left))

    if want_right and right is not None:
        if rect_maps is not None:
            m1, m2 = rect_maps["right"]
            right = cv2.remap(right, m1, m2, cv2.INTER_LINEAR)
        out.append(("R", right))

    return out


def _parse_hours(s: str | None) -> tuple[int, int] | None:
    """'10:21' -> (10, 21). None/empty -> None (no filter, capture 24/7)."""
    if not s:
        return None
    try:
        a, b = s.split(":")
        ai, bi = int(a), int(b)
    except ValueError as e:
        raise SystemExit(f"--operating-hours formato 'START:END' (ej. '10:21'): {e}")
    if not (0 <= ai <= 24 and 0 <= bi <= 24):
        raise SystemExit("--operating-hours: horas deben estar en [0, 24]")
    if ai == bi:
        raise SystemExit("--operating-hours: START == END (rango vacío)")
    return ai, bi


def _is_in_window(window: tuple[int, int] | None) -> bool:
    """True si la hora local actual cae dentro del [start, end). Si end <
    start, asume rango que cruza medianoche (ej. 22:6 = 22h-6h)."""
    if window is None:
        return True
    start, end = window
    h = datetime.now().hour
    if start <= end:
        return start <= h < end
    return h >= start or h < end


def _seconds_until_next_window(window: tuple[int, int]) -> float:
    """Segundos hasta el próximo START de la ventana operativa."""
    start, _ = window
    now = datetime.now()
    target = now.replace(hour=start, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(1.0, (target - now).total_seconds())


@dataclass
class SiteStats:
    """Contadores running per-site."""

    name: str
    frames_saved: int = 0
    frames_read: int = 0
    reconnects: int = 0
    bytes_written: int = 0
    last_error: str = ""
    last_save_ts: float = 0.0
    started_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        elapsed = time.time() - self.started_at
        rate = self.frames_saved / elapsed * 3600 if elapsed > 0 else 0
        return (
            f"{self.name:<25} saved={self.frames_saved:<5} "
            f"read={self.frames_read:<6} reconnects={self.reconnects:<3} "
            f"rate={rate:.0f}/h size={self.bytes_written / 1e6:.1f}MB"
        )


def _capture_site(
    site: dict[str, Any],
    output_dir: Path,
    stop_event: threading.Event,
    min_interval: float,
    max_interval: float,
    stats: SiteStats,
    operating_hours: tuple[int, int] | None = None,
    motion_trigger: bool = False,
    motion_threshold: float = 5.0,
    background_interval: float = 0.0,
    save_side: str = "left",
) -> None:
    """Loop de captura para un site hasta que ``stop_event`` se setee.

    Dos modos de muestreo:
    - **Random interval** (default): guarda un frame cada
      `uniform(min_interval, max_interval)` segundos, independientemente
      del contenido. Cobertura temporal uniforme pero baja chance de
      pegar a una persona puntual.
    - **Motion trigger** (``motion_trigger=True``): guarda un frame
      cuando ``mean(absdiff(prev_gray, gray)) > motion_threshold`` Y
      pasaron al menos ``min_interval`` segundos del último save
      (cooldown). Multiplica por 10-50× la fracción de frames con
      personas/movimiento.

    Reconecta solo si el stream se cae. Si ``operating_hours`` está
    seteado, duerme fuera de esa ventana sin abrir el stream.
    """
    name = site["name"]
    url = site["url"]
    site_dir = output_dir / name
    site_dir.mkdir(parents=True, exist_ok=True)

    sbs = bool(site.get("sbs", False))
    rect_maps: Optional[dict[str, tuple[Any, Any]]] = None
    # La calibración se carga lazy en el primer frame exitoso porque
    # necesitamos el half-frame size para computar el scale factor; los
    # streams SBS están siempre split horizontal, así half_size = (W//2, H).
    calib_block = site.get("calibration")
    calib_pending = bool(calib_block) and sbs

    while not stop_event.is_set():
        # Operating-hours gate antes de abrir el stream — evita conexiones
        # innecesarias fuera del horario de tránsito y libera el endpoint.
        if not _is_in_window(operating_hours):
            secs = _seconds_until_next_window(operating_hours)  # type: ignore[arg-type]
            wake_at = datetime.now() + timedelta(seconds=secs)
            logger.info(
                "[%s] fuera de horario (now %02dh, ventana %02d-%02d), "
                "dormido hasta %s (%.0fs)",
                name, datetime.now().hour, operating_hours[0],  # type: ignore[index]
                operating_hours[1], wake_at.strftime("%H:%M"), secs,  # type: ignore[index]
            )
            # stop_event.wait(secs) permite Ctrl+C inmediato
            if stop_event.wait(secs):
                return
            continue

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            stats.last_error = "VideoCapture failed to open"
            logger.warning("[%s] no pude abrir el stream — reintento en 5s", name)
            stats.reconnects += 1
            for _ in range(5):
                if stop_event.wait(1):
                    return
            continue

        # Estado del trigger
        next_save_at = time.time() + random.uniform(min_interval, max_interval)
        last_motion_save_t = 0.0
        last_bg_save_t = time.time()  # arranca contando desde apertura
        prev_gray = None
        consecutive_failures = 0

        while not stop_event.is_set():
            # Salir del read loop si caímos fuera del horario operativo.
            # El outer loop volverá a esperar hasta el next window.
            if not _is_in_window(operating_hours):
                logger.info("[%s] cierre de horario operativo", name)
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    logger.warning(
                        "[%s] %d fallos consecutivos de read — reconectando",
                        name, consecutive_failures,
                    )
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            stats.frames_read += 1
            now = time.time()

            # Carga lazy de los maps de rectificación una vez que conocemos
            # el tamaño del frame — half_size depende del split SBS, que
            # depende de la resolución real del stream.
            if calib_pending:
                fh, fw = frame.shape[:2]
                half_size = (fw // 2, fh)
                try:
                    rect_maps = _load_site_calibration(
                        calib_block, half_size,  # type: ignore[arg-type]
                    )
                    logger.info(
                        "[%s] calibración cargada (inline) para half=%dx%d",
                        name, half_size[0], half_size[1],
                    )
                except Exception as e:
                    stats.last_error = f"calibration load failed: {e!r}"
                    logger.exception(
                        "[%s] no pude cargar calibración inline — guardo sin rectificar",
                        name,
                    )
                    rect_maps = None
                calib_pending = False

            # Procesamos el frame una sola vez (split + rectify) y reusamos
            # el output tanto para la detección de movimiento como para el
            # save efectivo.
            processed = _process_frame(frame, sbs, rect_maps, save_side)

            def _save_processed(kind: str = "") -> None:
                for side, img in processed:
                    _save_frame(img, site_dir, name, stats,
                                kind=kind, side=side)

            # La detección de movimiento corre sobre la imagen LEFT (o única)
            # ya procesada porque es la vista canónica que alimenta al detector.
            motion_view = processed[0][1] if processed else None

            if motion_trigger:
                if motion_view is not None:
                    gray = cv2.cvtColor(motion_view, cv2.COLOR_BGR2GRAY)
                else:
                    gray = None
                if gray is not None and prev_gray is not None \
                        and gray.shape == prev_gray.shape:
                    diff_mean = float(cv2.absdiff(prev_gray, gray).mean())
                    if (diff_mean > motion_threshold
                            and now - last_motion_save_t > min_interval):
                        _save_processed(kind="motion")
                        last_motion_save_t = now
                prev_gray = gray
                # Background sample independiente del motion — captura el
                # estado de la escena en momentos sin gente, útil para
                # medir FP rate del detector después
                if (background_interval > 0
                        and now - last_bg_save_t >= background_interval):
                    _save_processed(kind="bg")
                    last_bg_save_t = now
            else:
                if now >= next_save_at:
                    _save_processed()
                    next_save_at = now + random.uniform(min_interval, max_interval)

        cap.release()
        if not stop_event.is_set():
            stats.reconnects += 1
            logger.info("[%s] reconectando en 2s", name)
            stop_event.wait(2)


def _save_frame(
    frame: Any, site_dir: Path, name: str, stats: SiteStats,
    kind: str = "", side: str = "",
) -> None:
    """Codifica y escribe un JPEG. Filename: ``<ts>[_<kind>][_<side>]_<rand>.jpg``.

    ``kind`` (motion/bg/empty) marca el origen del save; ``side`` (L/R/
    empty) identifica el lente cuando el stream era estéreo. Ambos
    quedan en el filename para que el dataset sea autodescriptivo.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    label = f"_{kind}" if kind else ""
    side_label = f"_{side}" if side else ""
    out_path = site_dir / f"{ts}{label}{side_label}_{suffix}.jpg"
    try:
        # cv2.imwrite usa el JPEG quality default (95). Ajustable con
        # [int(cv2.IMWRITE_JPEG_QUALITY), 92] en los params si hace falta.
        ok = cv2.imwrite(str(out_path), frame)
        if ok:
            stats.frames_saved += 1
            stats.last_save_ts = time.time()
            stats.bytes_written += out_path.stat().st_size
        else:
            stats.last_error = "imwrite returned False"
            logger.warning("[%s] imwrite falló para %s", name, out_path)
    except Exception as e:
        stats.last_error = repr(e)
        logger.exception("[%s] error guardando el frame", name)


def _stats_reporter(
    all_stats: list[SiteStats],
    stop_event: threading.Event,
    log_path: Path,
) -> None:
    """Vuelca stats per-site al log + archivo periódicamente."""
    while not stop_event.wait(60):
        line = f"\n=== stats @ {datetime.now():%H:%M:%S} ===\n"
        for s in all_stats:
            line += "  " + s.summary() + "\n"
        logger.info(line.rstrip())
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("training_data/sites.yaml"),
        help="YAML con la lista de sites (ver docstring para el formato). "
             "Default training_data/sites.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("training_data/captures"),
        help="Carpeta raíz donde se crean los subfolders por site. "
             "Default training_data/captures",
    )
    parser.add_argument(
        "--duration", type=int, default=3600,
        help="Duración total en segundos (default 3600 = 1h). 0 = infinito",
    )
    parser.add_argument(
        "--min-interval", type=float, default=8.0,
        help="Mínimo segundos entre frames guardados por site (default 8)",
    )
    parser.add_argument(
        "--max-interval", type=float, default=25.0,
        help="Máximo segundos entre frames por site (default 25)",
    )
    parser.add_argument(
        "--operating-hours", type=str, default=None,
        help="Restringir captura a la ventana 'START:END' en hora local. "
             "Ej. '10:21' = capturar entre 10h y 21h (cerrado a las 21). "
             "Fuera de esa ventana cada worker duerme hasta el próximo "
             "START. Default: sin filtro (24/7).",
    )
    parser.add_argument(
        "--motion-trigger", action="store_true",
        help="Guarda frames solo cuando hay cambio significativo entre "
             "frames consecutivos (cv2.absdiff grayscale). En este modo "
             "--max-interval se ignora y --min-interval funciona como "
             "cooldown post-save. Multiplica la chance de capturar gente.",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=5.0,
        help="mean(absdiff) sobre 0-255 que dispara save en motion mode. "
             "Default 5.0 — escena estática típica está en 0.5-1.5, "
             "persona caminando 8-20. Subí si hay falsos por iluminación.",
    )
    parser.add_argument(
        "--background-interval", type=float, default=0.0,
        help="En motion mode, guarda también un frame 'bg' cada N "
             "segundos independiente del motion (control para medir FP "
             "rate del detector después). Default 0 = off. Sugerido: "
             "300-600 (cada 5-10 min). Filename incluye '_bg_'.",
    )
    parser.add_argument(
        "--save-side", choices=("left", "right", "both"), default="left",
        help="Para sites con sbs: cuál de los dos lentes se guarda. "
             "Default 'left' — el detector en producción corre sobre "
             "rect_l, así que entrenar con left maximiza match de "
             "dominio. 'both' duplica volumen pero los pares L/R son "
             "casi idénticos a 14 cm de baseline; preferí más sites a "
             "más ángulos por site. Sites mono ignoran este flag.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Config no encontrado: {args.config}")
    if args.min_interval >= args.max_interval:
        sys.exit("--min-interval debe ser menor a --max-interval")
    operating_hours = _parse_hours(args.operating_hours)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sites = config.get("sites", [])
    if not sites:
        sys.exit("Config no tiene sites definidos")

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "capture.log"

    op_str = (
        f"{operating_hours[0]:02d}:00-{operating_hours[1]:02d}:00 hora local"
        if operating_hours else "24/7"
    )
    if args.motion_trigger:
        bg_str = (f", bg cada {args.background_interval:.0f}s"
                  if args.background_interval > 0 else "")
        mode_str = (
            f"motion-trigger (umbral diff>{args.motion_threshold:.1f}, "
            f"cooldown {args.min_interval:.0f}s{bg_str})"
        )
    else:
        mode_str = (
            f"random-interval {args.min_interval:.0f}-{args.max_interval:.0f}s"
        )
    logger.info(
        "Iniciando captura: %d sites, %s, duración %s, horario %s",
        len(sites), mode_str,
        f"{args.duration}s" if args.duration else "infinita", op_str,
    )
    for s in sites:
        flags = []
        if s.get("sbs"):
            flags.append("sbs")
        if s.get("calibration"):
            flags.append("rectify=inline")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        logger.info("  - %s -> %s%s", s["name"], s["url"], flag_str)
    logger.info("save-side: %s", args.save_side)

    stop_event = threading.Event()

    def _sigint(_sig: int, _frame: Any) -> None:
        logger.info("Ctrl+C recibido — cerrando workers...")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    all_stats = [SiteStats(name=s["name"]) for s in sites]
    workers = []
    for site, stats in zip(sites, all_stats):
        t = threading.Thread(
            target=_capture_site,
            args=(
                site, args.output, stop_event,
                args.min_interval, args.max_interval, stats,
                operating_hours,
                args.motion_trigger, args.motion_threshold,
                args.background_interval,
                args.save_side,
            ),
            daemon=True,
            name=f"capture-{site['name']}",
        )
        t.start()
        workers.append(t)

    reporter = threading.Thread(
        target=_stats_reporter,
        args=(all_stats, stop_event, log_path),
        daemon=True,
        name="stats-reporter",
    )
    reporter.start()

    try:
        if args.duration > 0:
            stop_event.wait(args.duration)
        else:
            while not stop_event.wait(3600):
                pass
    except KeyboardInterrupt:
        pass

    stop_event.set()
    for t in workers:
        t.join(timeout=15)

    # Resumen final
    logger.info("\n=== FINAL ===")
    total_saved = sum(s.frames_saved for s in all_stats)
    for s in all_stats:
        logger.info("  " + s.summary())
    logger.info("Total frames guardados: %d", total_saved)
    logger.info("Output en: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Capture frames from N MJPEG HTTP streams concurrently.

Pensado para acumular validation/bootstrap data desde cámaras en sitio
(IP cameras, NVRs, embedded vendors que exponen un endpoint MJPEG). Cada
site corre en su propio thread, lee el stream MJPEG continuamente, y
guarda frames según el modo elegido:

- **random-interval** (default): un frame cada ``uniform(min,max)``
  segundos, cobertura temporal uniforme.
- **motion-trigger** (``--motion-trigger``): solo guarda cuando hay
  cambio significativo entre frames consecutivos (cv2.absdiff). Multiplica
  la fracción de frames con eventos reales.

Resilient by design:
- Si el stream se cae, reconecta solo
- Si un frame está corrupto, lo skipea
- Ctrl+C cierra todos los workers limpio

Usage:
    python scripts/training/capture_mjpeg.py \\
        --config debug/mjpeg_sites.yaml \\
        --output debug/mjpeg_capture \\
        --duration 3600 \\
        --motion-trigger \\
        --min-interval 5 \\
        --background-interval 600

Config YAML format:

    sites:
      - name: site_a
        url: http://192.168.1.10/mjpg/video.mjpg
      - name: site_b
        url: http://192.168.2.10/mjpg/video.mjpg

Output layout:
    <output>/
      site_a/
        20260506_103752_motion_22fda5.jpg
        20260506_104646_bg_db7865.jpg
        ...
      site_b/
        ...
      capture.log     <- per-site stats appended every 60s
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import secrets
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import yaml

logger = logging.getLogger("capture_mjpeg")


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
    """Per-site running counters."""

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
) -> None:
    """Run capture loop for one site until stop_event is set.

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
            logger.warning("[%s] could not open stream — retry in 5s", name)
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
                        "[%s] %d consecutive read failures — reconnecting",
                        name, consecutive_failures,
                    )
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            stats.frames_read += 1
            now = time.time()

            if motion_trigger:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff_mean = float(cv2.absdiff(prev_gray, gray).mean())
                    if (diff_mean > motion_threshold
                            and now - last_motion_save_t > min_interval):
                        _save_frame(frame, site_dir, name, stats,
                                    kind="motion")
                        last_motion_save_t = now
                prev_gray = gray
                # Background sample independiente del motion — captura el
                # estado de la escena en momentos sin gente, útil para
                # medir FP rate del detector después
                if (background_interval > 0
                        and now - last_bg_save_t >= background_interval):
                    _save_frame(frame, site_dir, name, stats, kind="bg")
                    last_bg_save_t = now
            else:
                if now >= next_save_at:
                    _save_frame(frame, site_dir, name, stats)
                    next_save_at = now + random.uniform(min_interval, max_interval)

        cap.release()
        if not stop_event.is_set():
            stats.reconnects += 1
            logger.info("[%s] reconnecting in 2s", name)
            stop_event.wait(2)


def _save_frame(
    frame: Any, site_dir: Path, name: str, stats: SiteStats,
    kind: str = "",
) -> None:
    """Encode + write a single JPEG. Filename: <ts>[_<kind>]_<rand>.jpg.

    Si ``kind`` se da (ej. 'motion', 'bg'), se incluye en el filename
    para distinguir el tipo de save al analizar el dataset después.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    label = f"_{kind}" if kind else ""
    out_path = site_dir / f"{ts}{label}_{suffix}.jpg"
    try:
        # cv2.imwrite uses default JPEG quality (95). Tweak with
        # [int(cv2.IMWRITE_JPEG_QUALITY), 92] in the params if needed.
        ok = cv2.imwrite(str(out_path), frame)
        if ok:
            stats.frames_saved += 1
            stats.last_save_ts = time.time()
            stats.bytes_written += out_path.stat().st_size
        else:
            stats.last_error = "imwrite returned False"
            logger.warning("[%s] imwrite failed for %s", name, out_path)
    except Exception as e:
        stats.last_error = repr(e)
        logger.exception("[%s] error saving frame", name)


def _stats_reporter(
    all_stats: list[SiteStats],
    stop_event: threading.Event,
    log_path: Path,
) -> None:
    """Periodically dump per-site stats to log + file."""
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
        "--config", type=Path, required=True,
        help="YAML con la lista de sites (ver docstring para el formato)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Carpeta raíz donde se crean los subfolders por site",
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
        logger.info("  - %s -> %s", s["name"], s["url"])

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

    # Final summary
    logger.info("\n=== FINAL ===")
    total_saved = sum(s.frames_saved for s in all_stats)
    for s in all_stats:
        logger.info("  " + s.summary())
    logger.info("Total frames guardados: %d", total_saved)
    logger.info("Output en: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

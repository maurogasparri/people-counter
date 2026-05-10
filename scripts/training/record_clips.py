#!/usr/bin/env python3
"""Grabador concurrente de clips de video de N streams MJPEG HTTP.

Pensado para armar dataset de validación E2E del pipeline de conteo
(detect → tracking → line crossing). A diferencia del capture de
snapshots, acá se graban los frames consecutivos del stream en MP4 a
fps nativo del feed — necesario para que un tracker tipo ByteTrack
mantenga IDs entre cruces.

Implementación: un subprocess de ffmpeg por site, ``-c copy`` para
stream-copy sin re-encoding (rápido + lossless). Corre en paralelo
con ``capture_mjpeg.py`` sin conflicto (cada site soporta varias
conexiones concurrentes en MJPEG HTTP).

Uso:
    python scripts/training/record_clips.py \\
        --config debug/mjpeg_sites.yaml \\
        --output debug/mjpeg_clips \\
        --duration 900           # 15 minutos por site

Salida:
    <output>/<site_name>/<YYYYmmdd_HHMMSS>.mp4
"""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("record_clips")


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg no encontrado en PATH. Instalalo (winget install Gyan.FFmpeg).")


def _record_site_realtime(
    site: dict[str, Any],
    output_dir: Path,
    duration: int,
    stop_event: threading.Event,
) -> None:
    """Grabación realtime usando OpenCV: mide el fps REAL del stream en
    los primeros segundos y graba a ese fps. El MP4 resultante reproduce
    a tiempo real (sin aceleración) y cv2.VideoCapture luego retorna
    el fps correcto, así ByteTrack u otros consumidores no necesitan
    overrides.

    Flujo:
    1. Conectar al stream
    2. Probar 50 frames o 10s, lo que llegue primero — calcular fps real
    3. Inicializar VideoWriter con ese fps + dimensiones del frame
    4. Escribir las probe frames + continuar grabando hasta `duration`
    """
    import cv2  # local import — solo en realtime mode

    name = site["name"]
    url = site["url"]
    site_dir = output_dir / name
    site_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = site_dir / f"{ts}.mp4"

    logger.info("[%s] conectando → %s", name, url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error("[%s] no pude abrir el stream", name)
        return

    # Probe del fps real
    probe_frames: list[tuple[float, Any]] = []
    t_start = time.time()
    while not stop_event.is_set():
        ok, frame = cap.read()
        if ok and frame is not None:
            probe_frames.append((time.time(), frame))
        elapsed = time.time() - t_start
        if len(probe_frames) >= 50 or elapsed >= 10:
            break

    if len(probe_frames) < 5:
        logger.error("[%s] probe falló: %d frames en 10s",
                     name, len(probe_frames))
        cap.release()
        return

    span = probe_frames[-1][0] - probe_frames[0][0]
    actual_fps = (len(probe_frames) - 1) / max(span, 0.1)
    h, w = probe_frames[0][1].shape[:2]
    logger.info("[%s] fps real=%.1f, %dx%d → grabando %ds",
                name, actual_fps, w, h, duration)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (w, h))
    if not out.isOpened():
        logger.error("[%s] VideoWriter init falló", name)
        cap.release()
        return

    n_written = 0
    for _, f in probe_frames:
        out.write(f)
        n_written += 1

    # Continuar hasta cumplir duration (contado desde t_start)
    t_end = t_start + duration
    while time.time() < t_end and not stop_event.is_set():
        ok, frame = cap.read()
        if ok and frame is not None:
            out.write(frame)
            n_written += 1

    cap.release()
    out.release()
    elapsed = time.time() - t_start
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    logger.info(
        "[%s] listo: %d frames en %.1fs (%.1f fps) → %s (%.1f MB)",
        name, n_written, elapsed, n_written / max(elapsed, 1),
        out_path.name, size_mb,
    )


def _record_site(
    site: dict[str, Any],
    output_dir: Path,
    duration: int,
    encode: bool,
    stop_event: threading.Event,
) -> None:
    """Spawn ffmpeg para un site, grabar duration segundos."""
    name = site["name"]
    url = site["url"]
    site_dir = output_dir / name
    site_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = site_dir / f"{ts}.mp4"

    if encode:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    else:
        codec_args = ["-c", "copy"]

    # frag_keyframe+empty_moov hace MP4 fragmentado: cada keyframe es
    # un fragment self-contained, así si ffmpeg se interrumpe (SIGINT,
    # crash de red, etc.) el archivo SIGUE siendo playable.
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "warning",
        "-i", url,
        "-t", str(duration),
        *codec_args,
        "-movflags", "+frag_keyframe+empty_moov",
        "-f", "mp4",
        str(out_path),
    ]
    logger.info("[%s] arrancando → %s", name, out_path.name)
    # stdin=PIPE permite mandar 'q' a ffmpeg para shutdown limpio.
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )

    # Watchdog en otro thread por si stop_event sucede
    def _watchdog() -> None:
        while proc.poll() is None:
            if stop_event.wait(1):
                # 'q' por stdin → ffmpeg flushea + cierra moov limpio,
                # mucho mejor que SIGTERM que corta el output sin moov.
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.write("q")
                        proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                return

    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    proc.wait()
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.returncode != 0 and not stop_event.is_set():
        logger.warning("[%s] ffmpeg exit %d: %s",
                       name, proc.returncode, stderr.strip()[:300])
    else:
        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
        logger.info("[%s] listo → %s (%.1f MB)", name, out_path.name, size_mb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--duration", type=int, default=900,
        help="Segundos a grabar por site (default 900 = 15 min).",
    )
    parser.add_argument(
        "--encode", action="store_true",
        help="Re-encode con libx264 (más chico en disco). Default: "
             "stream-copy (rápido + lossless, archivos más grandes).",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Modo recomendado: usa cv2 + auto-detección de fps real "
             "del stream. El MP4 resultante reproduce a real-time (sin "
             "aceleración) y cv2.VideoCapture devuelve el fps correcto. "
             "Default: ffmpeg stream-copy (rápido pero puede grabar "
             "acelerado si el container declara fps != real fps).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _ensure_ffmpeg()
    if not args.config.exists():
        sys.exit(f"Config no encontrado: {args.config}")

    with open(args.config, encoding="utf-8") as f:
        sites = yaml.safe_load(f).get("sites", [])
    if not sites:
        sys.exit("No hay sites en el config")

    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Grabando %d sites por %ds, codec=%s",
        len(sites), args.duration, "x264" if args.encode else "copy",
    )

    stop_event = threading.Event()

    def _sigint(_sig: int, _frame: Any) -> None:
        logger.info("Ctrl+C — abortando grabaciones")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    workers = []
    for site in sites:
        if args.realtime:
            t = threading.Thread(
                target=_record_site_realtime,
                args=(site, args.output, args.duration, stop_event),
                daemon=False,
                name=f"rec-{site['name']}",
            )
        else:
            t = threading.Thread(
                target=_record_site,
                args=(site, args.output, args.duration, args.encode, stop_event),
                daemon=False,
                name=f"rec-{site['name']}",
            )
        t.start()
        workers.append(t)

    try:
        for t in workers:
            t.join()
    except KeyboardInterrupt:
        stop_event.set()
        for t in workers:
            t.join(timeout=10)

    logger.info("=== FINAL ===")
    for site in sites:
        site_dir = args.output / site["name"]
        clips = sorted(site_dir.glob("*.mp4")) if site_dir.exists() else []
        for c in clips:
            size_mb = c.stat().st_size / 1e6
            logger.info("  %s/%s  (%.1f MB)", site["name"], c.name, size_mb)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Helpers de anotación para el viewer web del runtime.

Funciones puras que dibujan sobre frames BGR. Sin estado, sin I/O. Se
mantienen fuera de ``viewer.py`` así el módulo de streaming queda chico y
fácil de testear aislado.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Paleta de colores BGR
_COLOR_CONFIRMED = (0, 255, 0)
_COLOR_PENDING = (0, 165, 255)
_COLOR_CANDIDATE = (180, 180, 180)
_COLOR_COUNTING_ZONE = (0, 0, 255)         # rojo

# Paleta de la counting-line por label de dirección. Cualquier otra cosa cae
# a blanco así que un label exótico igual renderiza visiblemente. Verdes para
# eventos IN-side, azules para OUT-side — matchea el modelo mental del operador.
_LINE_COLOR_BY_LABEL = {
    "ingress": (0, 255, 0),       # verde: IN
    "egress": (255, 0, 0),        # azul: OUT
    "in": (0, 255, 0),
    "out": (255, 0, 0),
    "enter": (0, 255, 0),
    "leave": (255, 0, 0),
}
_LINE_COLOR_FALLBACK = (255, 255, 255)

# Hysteresis sobre el flip CONFIRMED -> PENDING en el display:
# antes de mostrar al track como naranja necesitamos verlo en
# PENDING por al menos este número de misses consecutivos. Sin
# hysteresis, un solo frame de miss del detector hace que el
# color cambie y vuelva instantáneamente al frame siguiente,
# dando la sensación al operador de que el tracking se está
# rompiendo cuando en realidad es ruido normal del detector.
# 5 frames @ 12 fps = ~400ms de pérdida sostenida antes del flip,
# tiempo suficiente para que oclusiones cortas no se vean como
# fallas visuales.
_PENDING_DISPLAY_HYSTERESIS_FRAMES = 5

# Ventana de la mediana móvil de head_height_mm que se muestra en el
# label. La altura derivada de SGBM tiene jitter natural de ~1-2cm
# frame-a-frame por ruido del matching estéreo; mostrar el último
# sample crudo amplifica ese ruido visualmente. La mediana sobre los
# últimos N samples lo achata a <0.5cm de fluctuación. 10 samples ~=
# 0.8s a 12 fps, suficiente para responder a cambios reales de altura
# (persona sentándose, agachándose) sin transmitir el ruido per-frame.
_HEIGHT_DISPLAY_MEDIAN_WINDOW = 10

# Tamaño FIJO (lado en px) de la caja que se dibuja en el preview, IGUAL para
# todos los tracks. El detector emite tamaños distintos por frame y por sujeto
# (el bbox "respira" y varía entre personas); para el preview eso distrae. Una
# caja cuadrada constante centrada en el sujeto da una pantalla limpia y
# uniforme — solo la POSICIÓN varía (sigue a cada track), no el tamaño. 80px en
# el frame canónico 1152×648 es un marcador legible sin tapar la escena.
_BBOX_DISPLAY_SIZE_PX = 80

# Largo de la trayectoria (track trail): cantidad de posiciones recientes del
# track que se dibujan como polilínea. ~30 puntos ≈ 1-2s de recorrido a
# 15-25 fps — suficiente para ver de dónde viene la persona sin ensuciar el
# frame.
_TRAIL_LENGTH = 30

# Flash visual +IN/+OUT al contar: vida útil del overlay en segundos. ~1.5s
# es suficiente para que el operador asocie visualmente el evento con el
# cruce sin que el overlay se vuelva permanente en la escena. Fade lineal
# del alpha durante la vida (1.0 al disparar → 0.0 al expirar).
_COUNT_FLASH_DURATION_S = 1.5
# Tamaño del texto del flash (HERSHEY_SIMPLEX). 2.0 es grande pero no tapa
# multiples tracks en escena llena.
_COUNT_FLASH_FONT_SCALE = 1.8
_COUNT_FLASH_FONT_THICKNESS = 4
# Offset vertical del flash respecto del centroide del track (px hacia
# arriba). Suficiente para no pisar el label #ID + altura.
_COUNT_FLASH_Y_OFFSET = 60
# Kernel del blur del fondo del flash. Reemplaza un rectángulo negro
# opaco — el blur del contexto deja que se note "hay algo abajo" sin
# tapar al sujeto y mejora el contraste de los colores oscuros del texto
# (especialmente el azul del +OUT, que sobre negro pierde mucho). Kernel
# 21 da diffusion fuerte sin emborronar el texto encima (que va sin blur
# en la misma operación de overlay).
_COUNT_FLASH_BLUR_KERNEL = (21, 21)


def annotate_left(
    frame: np.ndarray,
    tracks: dict,
    counter: Optional[Any],
    tracking_zone_polygon: Optional[list[tuple[float, float]]] = None,
    recent_counts: Optional[dict[int, tuple[str, float, float, float]]] = None,
    now_mono: Optional[float] = None,
) -> np.ndarray:
    """Dibuja la counting zone, línea y tracks sobre una copia de ``frame``.

    NO dibuja totales/FPS — esos van en el dashboard del viewer (evita el
    overlay duplicado). Acá solo la geometría (counting zone + línea) + las
    cajas/centroides/track-ids de los tracks.

    Deliberadamente NO dibuja las detecciones raw del detector: las cajas
    grises por-frame mostraban FPs de clutter estructural (sombras, bordes)
    que el filtro de clutter estático YA esconde a nivel de track. Dibujar
    las detecciones crudas hacía reaparecer esos "ghosts" parpadeando en el
    preview (sobre todo del lado del approach de IN), confundiendo al
    operador. Mostrar solo los tracks (ya filtrados) deja el preview limpio.

    Args:
        frame: Frame BGR de la cámara izquierda (rectificado).
        tracks: dict[int, Track] de EuclideanTracker (idealmente ya
            filtrado de clutter por el caller).
        counter: instancia del Counter (se lee para el overlay de geometría
            counting zone + línea). Puede ser None.
        tracking_zone_polygon: si se provee, la zona FUERA de este polígono
            se rendere blureada. Da feedback visual al operador de qué
            área del frame está siendo procesada por el pipeline (el resto
            es ignorado por el ``tracking_zone`` del tracker). None = sin
            blur (back-compat, sites sin tracking_zone activo).
        recent_counts: dict ``{track_id: (label, mono_ts, pos_x, pos_y)}``
            con los eventos de conteo recientes que disparó el counter. Para
            cada uno se dibuja un flash visual "+IN" o "+OUT" durante
            ``_COUNT_FLASH_DURATION_S`` con fade lineal del alpha. La
            posición usada es la registrada al momento del cruce (sobrevive
            al death-emit del track) — no la posición actual del tracker.
        now_mono: timestamp monotónico actual (``time.monotonic()``). Si se
            omite cae a ``time.monotonic()`` interno — solo se inyecta en
            tests para controlar el fade. None en producción.
    """
    out = frame.copy()

    # Blur de la zona fuera del tracking_zone polygon ANTES de la geometría +
    # los tracks, así el overlay y los bboxes quedan crisp encima.
    if tracking_zone_polygon is not None and len(tracking_zone_polygon) >= 3:
        out = _blur_outside_polygon(out, tracking_zone_polygon)

    # Geometría primero así los tracks quedan arriba.
    _draw_counter_geometry(out, counter)

    # Diferir el import así un import circular en init parcial no
    # vuela el módulo viewer.
    from src.tracking.tracker import CONFIRMED, PENDING
    base_state_colour = {
        CONFIRMED: _COLOR_CONFIRMED,
        PENDING: _COLOR_PENDING,
        # CANDIDATE: deliberadamente fuera del mapping — los tracks
        # candidate son tentativos (1 sola detección, sin confirmar)
        # y aparecen/desaparecen rápido cuando el detector flickerea
        # sobre clutter. Mostrarlos como "ghosts" grises ruido visual
        # al operador. El operador solo ve tracks ya comprometidos
        # (CONFIRMED o re-id en PENDING desde CONFIRMED).
    }
    # Fuentes más grandes así el operador puede leer el overlay
    # mientras camina bajo las cámaras durante un check de piloto (el
    # live viewer está pensado para debug on-site, no para archivar).
    for tid, track in tracks.items():
        state = getattr(track, "state", None)
        base_colour = base_state_colour.get(state)
        if base_colour is None:
            continue
        positions = getattr(track, "positions", None)
        if not positions:
            continue

        # Hysteresis sobre el flip CONFIRMED -> PENDING: si el track
        # está recién entrando a PENDING (poco disappeared), seguimos
        # mostrando verde para que el operador no vea flickeo
        # naranja-verde en cada miss aislado del detector. Solo
        # después de sustained PENDING (>= threshold) flipea a
        # naranja.
        disappeared = int(getattr(track, "disappeared", 0) or 0)
        if state == PENDING and disappeared < _PENDING_DISPLAY_HYSTERESIS_FRAMES:
            display_colour = _COLOR_CONFIRMED
        else:
            display_colour = base_colour

        # Trayectoria del track: polilínea por las últimas N posiciones (el
        # recorrido reciente de la persona). Se dibuja primero para que la
        # caja/círculo/label queden por encima.
        trail_pts = positions[-_TRAIL_LENGTH:]
        if len(trail_pts) >= 2:
            trail = np.array(
                [[int(p[0]), int(p[1])] for p in trail_pts], dtype=np.int32
            )
            cv2.polylines(
                out, [trail], isClosed=False, color=display_colour,
                thickness=1, lineType=cv2.LINE_AA,
            )

        # Caja de tamaño FIJO centrada en la última detección. El detector
        # emite tamaños ligeramente distintos cada frame (el bbox "respira")
        # incluso con el sujeto quieto; para el preview eso distrae. La
        # POSICIÓN sigue al sujeto (centro de la última detección, sin lag),
        # pero el TAMAÑO es constante (_BBOX_DISPLAY_SIZE_PX) → rectángulo
        # estable. Persiste durante PENDING usando el último bbox conocido.
        meta = getattr(track, "meta", None)
        history = meta.get("detection_history") if isinstance(meta, dict) else None
        bbox = None
        if history:
            latest = next(
                (
                    rec.get("bbox")
                    for rec in reversed(history)
                    if isinstance(rec.get("bbox"), (list, tuple))
                    and len(rec["bbox"]) == 4
                ),
                None,
            )
            if latest is not None:
                cx_bbox = (float(latest[0]) + float(latest[2])) / 2.0
                cy_bbox = (float(latest[1]) + float(latest[3])) / 2.0
                half = _BBOX_DISPLAY_SIZE_PX / 2.0
                bbox = (
                    cx_bbox - half,
                    cy_bbox - half,
                    cx_bbox + half,
                    cy_bbox + half,
                )
        # Determinar la posición de display del círculo (#NN label
        # + height label). Si hay un bbox cacheado, anclamos el
        # marker al centro del bbox displayed — durante PENDING esto
        # mantiene círculo + bbox visualmente juntos en la última
        # posición observada en vez de "dispararse" siguiendo la
        # extrapolación del Kalman (que es para el matching interno,
        # no para mostrar al operador). Si no hay bbox aún (track
        # nuevo sin metadata), cae al positions[-1].
        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), display_colour, 2)
            cx_disp = (x1 + x2) // 2
            cy_disp = (y1 + y2) // 2
        else:
            cx_disp = int(positions[-1][0])
            cy_disp = int(positions[-1][1])

        cv2.circle(out, (cx_disp, cy_disp), 10, display_colour, -1)
        cv2.putText(out, f"#{tid}", (cx_disp + 14, cy_disp - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_colour, 2,
                    cv2.LINE_AA)
        cx, cy = cx_disp, cy_disp  # alias para el label de altura abajo
        # Label de altura — mediana sobre los últimos N samples para
        # suavizar el jitter de ~1-2cm del SGBM frame-a-frame. Si la
        # historia es corta (<2 samples), cae al último valor crudo
        # — sin suficiente ventana la mediana no aporta nada.
        if history:
            recent = history[-_HEIGHT_DISPLAY_MEDIAN_WINDOW:]
            head_samples = [
                float(rec.get("head_height_mm"))
                for rec in recent
                if isinstance(rec.get("head_height_mm"), (int, float))
                and rec.get("head_height_mm") > 0
            ]
            cls_samples = [
                rec.get("height_class")
                for rec in recent
                if rec.get("height_class")
            ]
            if head_samples:
                head_samples.sort()
                head_mm = head_samples[len(head_samples) // 2]
            else:
                head_mm = None
            # Para height_class: voto mayoritario sobre la misma ventana.
            # Estabiliza el label adult/child/unknown contra flips espurios
            # cuando la altura mediana ronda el threshold 1.55m.
            if cls_samples:
                cls = max(set(cls_samples), key=cls_samples.count)
            else:
                cls = "unknown"
            if head_mm is not None:
                label = f"{head_mm/1000:.2f}m {cls}"
            else:
                label = cls
            cv2.putText(out, label, (cx + 14, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, display_colour, 2,
                        cv2.LINE_AA)

    # Flash +IN/+OUT sobre los tracks que el counter acaba de contar. Se
    # dibuja AL FINAL para quedar arriba del bbox/label/trail.
    if recent_counts:
        if now_mono is None:
            import time
            now_mono = time.monotonic()
        _draw_count_flashes(out, recent_counts, now_mono)

    return out


def _draw_count_flashes(
    frame: np.ndarray,
    recent_counts: dict[int, tuple[str, float, float, float]],
    now_mono: float,
) -> None:
    """Dibuja "+IN" o "+OUT" sobre las posiciones de los eventos recientes
    con fade-out lineal del alpha. ``recent_counts`` viene del caller y
    contiene SOLO los eventos vivos (≤ _COUNT_FLASH_DURATION_S de edad);
    el pruning lo hace el caller. Aquí asumimos que todo lo que llega se
    dibuja, pero igual sanity-checkeamos la edad para evitar artifacts si
    el caller prunee con TTL distinto.
    """
    for _tid, (label, mono_ts, pos_x, pos_y) in recent_counts.items():
        elapsed = now_mono - mono_ts
        if elapsed < 0 or elapsed > _COUNT_FLASH_DURATION_S:
            continue
        # Alpha lineal: 1.0 al disparar → 0.0 al expirar. Cv2 no tiene alpha
        # blending nativo en putText, así que usamos addWeighted sobre una
        # capa temp del overlay.
        alpha = max(0.0, 1.0 - elapsed / _COUNT_FLASH_DURATION_S)

        # Color por label. Matchea la paleta de la counting line:
        # in/ingress = verde, out/egress = azul.
        text_color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
        # Texto canónico — usamos +IN/+OUT para que el operador lea "+1"
        # mentalmente independientemente del label interno
        # (ingress/egress/in/out).
        if label in ("in", "ingress", "enter"):
            text = "+IN"
        elif label in ("out", "egress", "leave"):
            text = "+OUT"
        else:
            text = f"+{label.upper()}"

        # Posición: arriba del centroide del cruce, centrado en X. Si queda
        # cortado por el borde superior, lo movemos hacia abajo del
        # centroide (mejor que recortarlo).
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX,
            _COUNT_FLASH_FONT_SCALE, _COUNT_FLASH_FONT_THICKNESS,
        )
        anchor_x = int(pos_x) - text_w // 2
        anchor_y = int(pos_y) - _COUNT_FLASH_Y_OFFSET
        if anchor_y - text_h < 0:
            anchor_y = int(pos_y) + _COUNT_FLASH_Y_OFFSET + text_h

        # Capa overlay para alpha blending. Dibujamos en una copia del
        # frame y luego mezclamos con el original según alpha. Es chico
        # (solo la región del texto) así no es costoso.
        x0 = max(0, anchor_x - 8)
        y0 = max(0, anchor_y - text_h - 8)
        x1 = min(frame.shape[1], anchor_x + text_w + 8)
        y1 = min(frame.shape[0], anchor_y + baseline + 8)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = frame[y0:y1, x0:x1]
        # Overlay = blur del ROI original. Reemplaza el rectángulo negro
        # opaco para que el contexto detrás del texto siga siendo visible
        # (no tapa cabezas/cuerpos al sujeto contado) y los colores
        # oscuros del texto (azul del +OUT) ganen contraste contra un
        # fondo difuminado en lugar de negro absoluto.
        overlay = cv2.GaussianBlur(roi, _COUNT_FLASH_BLUR_KERNEL, 0)
        # Texto sobre el overlay (coords relativas al ROI).
        cv2.putText(
            overlay,
            text,
            (anchor_x - x0, anchor_y - y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            _COUNT_FLASH_FONT_SCALE,
            text_color,
            _COUNT_FLASH_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def depth_to_colormap(
    depth_map: Optional[np.ndarray],
    vmin_mm: float = 500.0,
    vmax_mm: float = 5000.0,
) -> np.ndarray:
    """Renderiza un mapa depth-en-mm como un colormap BGR JET.

    Cero (= disparity inválido) renderiza en negro. Input None /
    vacío devuelve un frame chico gris oscuro así el panel es
    visualmente distinto de un depth map real y el caller no necesita
    chequear None.
    """
    if depth_map is None or depth_map.size == 0:
        return np.full((100, 100, 3), 30, dtype=np.uint8)
    d = depth_map.astype(np.float32)
    invalid = d <= 0
    d = np.clip(d, vmin_mm, vmax_mm)
    norm = ((d - vmin_mm) / (vmax_mm - vmin_mm) * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    if np.any(invalid):
        coloured[invalid] = (0, 0, 0)
    return coloured


def compose_3panel(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    target_height: int = 320,
) -> np.ndarray:
    """Composite de 3 paneles lado a lado a la misma altura: L | R |
    disparidad.

    Cada panel se escala POR ALTURA preservando su aspect ratio, así la
    disparidad (que tiene la misma proporción que las cámaras) NO se
    deforma — el bug previo la estiraba al ancho de la fila L|R. Los tres
    paneles quedan del mismo tamaño (fácil comparar). Paneles faltantes /
    vacíos se llenan con un placeholder gris oscuro así el composite nunca
    crashea ante input parcial. ``target_height`` chico = stream liviano.
    """
    def _panel(img: Optional[np.ndarray], h_target: int) -> np.ndarray:
        if img is None or img.size == 0:
            # Placeholder con el mismo aspect ~16:9 que los paneles reales.
            return np.full((h_target, max(1, h_target * 16 // 9), 3), 30, dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if h != h_target:
            img = cv2.resize(
                img, (max(1, round(w * h_target / h)), h_target),
                interpolation=cv2.INTER_AREA,
            )
        return img

    return cv2.hconcat(
        [
            _panel(left, target_height),
            _panel(right, target_height),
            _panel(depth, target_height),
        ]
    )


# Kernel del blur del tracking_zone overlay. 31×31 con sigma auto da un blur
# fuerte pero no opaco (todavía se distingue la silueta de objetos para
# que el operador pueda ubicar visualmente qué quedó afuera del polígono).
# Costo ~5-10ms en 1152×648 — solo se aplica cuando hay subscribers del
# preview, no impacta el hot path del counter.
_TRACKING_ZONE_BLUR_KERNEL = (31, 31)
# Oscurecimiento adicional de la zona blureada — multiplicador BGR. 0.55
# ≈ 45% más oscuro, refuerza visualmente que es "zona ignorada" sin
# llegar a negro absoluto (que tapa toda referencia de contexto).
_TRACKING_ZONE_DARKEN_FACTOR = 0.55


def _blur_outside_polygon(
    frame: np.ndarray,
    polygon: list[tuple[float, float]],
) -> np.ndarray:
    """Devuelve ``frame`` con la zona FUERA del polígono blureada y
    oscurecida. La zona adentro queda intacta. Sirve para feedback visual
    del ``tracking_zone`` — el operador ve exactamente qué área del frame
    está siendo procesada por el pipeline.

    Implementación: máscara binaria del polígono → blur+darken al frame
    entero → composición ``np.where(mask, original, blurred)``.
    """
    h, w = frame.shape[:2]
    contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    # Máscara: 255 dentro del polígono, 0 afuera.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    # Blur + darken aplicado al frame entero (la composición selecciona
    # qué pixels son los blureados).
    blurred = cv2.GaussianBlur(frame, _TRACKING_ZONE_BLUR_KERNEL, 0)
    blurred = cv2.convertScaleAbs(blurred, alpha=_TRACKING_ZONE_DARKEN_FACTOR, beta=0)
    # Composición: donde mask=255 va el original, donde mask=0 va el blureado.
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask_3 == 255, frame, blurred)


# ----------------------------------------------------------------- internals
def _draw_counter_geometry(frame: np.ndarray, counter: Optional[Any]) -> None:
    """Overlay del rectángulo de la counting zone (si hay) y cada segmento de
    línea con una flecha perpendicular mostrando la dirección contada.

    Cada línea se dibuja en el color que matchea su label dominante
    (``ingress`` -> verde, ``egress`` -> azul, cualquier otra cosa en
    blanco). Una flecha por dirección con label, perpendicular al
    segmento, apuntando hacia el lado en el que un cruce tiene que
    terminar para que dispare ese label.
    """
    if counter is None:
        return
    zone = getattr(counter, "counting_zone", None)
    if zone is not None:
        try:
            x_min, x_max, y_min, y_max = zone
            cv2.rectangle(frame, (int(x_min), int(y_min)),
                          (int(x_max), int(y_max)), _COLOR_COUNTING_ZONE, 4)
        except Exception:
            logger.debug("counting zone overlay failed", exc_info=True)

    lines = getattr(counter, "lines", None) or []
    for line in lines:
        try:
            _draw_line_with_arrows(frame, line)
        except Exception:
            logger.debug("Line overlay failed", exc_info=True)


def _draw_line_with_arrows(frame: np.ndarray, line: Any) -> None:
    """Dibuja una línea de conteo + flechas per-dirección.

    El largo de la flecha escala con el largo del segmento así sigue
    siendo visible en ROIs chicos sin desbordar en grandes. Si ambas
    direcciones tienen label el segmento renderiza en blanco (neutral)
    y el color per-flecha codifica qué lado de la línea es qué evento.
    """
    x1, y1 = int(line.from_xy[0]), int(line.from_xy[1])
    x2, y2 = int(line.to_xy[0]), int(line.to_xy[1])
    orientation = line.orientation
    labels: dict[str, str] = line.labels

    seg_len = max(1, abs(x2 - x1) + abs(y2 - y1))
    arrow_len = max(20, min(60, seg_len // 6))
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2

    if len(labels) >= 2:
        seg_color = _LINE_COLOR_FALLBACK
    else:
        only_label = next(iter(labels.values()), None)
        seg_color = _LINE_COLOR_BY_LABEL.get(
            only_label or "", _LINE_COLOR_FALLBACK,
        )
    cv2.line(frame, (x1, y1), (x2, y2), seg_color, 4)

    # La cola de la flecha se ancla sobre la línea y la punta se
    # extiende un ``arrow_len`` hacia el lado donde el cruce tiene
    # que terminar. Esto mantiene la flecha estrictamente de un lado
    # del segmento en vez de cruzarlo, lo cual hace que la lectura
    # "andá para acá" sea inmediata.
    if orientation == "horizontal":
        for direction, label in labels.items():
            color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
            if direction == "top_to_bottom":
                tail = (mx, my)
                tip = (mx, my + arrow_len)
            else:  # bottom_to_top
                tail = (mx, my)
                tip = (mx, my - arrow_len)
            cv2.arrowedLine(frame, tail, tip, color, 4, tipLength=0.35)
    else:
        for direction, label in labels.items():
            color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
            if direction == "left_to_right":
                tail = (mx, my)
                tip = (mx + arrow_len, my)
            else:  # right_to_left
                tail = (mx, my)
                tip = (mx - arrow_len, my)
            cv2.arrowedLine(frame, tail, tip, color, 4, tipLength=0.35)

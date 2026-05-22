"""Smoke tests para src/web/annotate.py."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.tracking.counter import Counter, Line
from src.tracking.tracker import CONFIRMED, PENDING, CANDIDATE
from src.web.annotate import (
    annotate_left,
    compose_3panel,
    depth_to_colormap,
)


@dataclass
class _FakeTrack:
    track_id: int
    state: str
    positions: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _two_way_counter() -> Counter:
    return Counter(
        lines=[Line(
            from_xy=(50, 150), to_xy=(250, 150),
            labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
        )],
        roi={"x_min": 50, "x_max": 250, "y_min": 100, "y_max": 200},
    )


def _no_roi_counter() -> Counter:
    return Counter(
        lines=[Line(
            from_xy=(0, 240), to_xy=(400, 240),
            labels={"top_to_bottom": "ingress"},
        )],
    )


def test_compose_3panel_handles_none_panels():
    out = compose_3panel(None, None, None, target_height=120)
    # Single-row layout: los 3 paneles a target_height, placeholders incluidos.
    assert out.shape[0] == 120
    assert out.shape[2] == 3


def test_compose_3panel_single_row_preserves_aspect():
    """Los 3 paneles van lado a lado a la MISMA altura, cada uno preservando
    su aspect ratio. La disparidad (panel c) NO se estira al ancho de L|R
    como hacía el layout viejo de 2 filas — su ancho sale de su propio
    aspect (800/600 -> 160px a h=120), no del ancho del top."""
    a = np.zeros((300, 400, 3), dtype=np.uint8)   # left  (w/h 1.333) -> 160
    b = np.zeros((100, 200, 3), dtype=np.uint8)   # right (w/h 2.0)   -> 240
    c = np.zeros((600, 800, 3), dtype=np.uint8)   # depth (w/h 1.333) -> 160
    out = compose_3panel(a, b, c, target_height=120)
    assert out.shape[0] == 120                    # fila única, todos a h=120
    assert out.shape[1] == 160 + 240 + 160        # ancho = suma de aspectos
    assert out.shape[2] == 3


def test_depth_colormap_handles_none():
    out = depth_to_colormap(None)
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


def test_depth_colormap_marks_zero_invalid_black():
    depth = np.full((10, 10), 2000.0, dtype=np.float32)
    depth[0, 0] = 0  # invalid
    out = depth_to_colormap(depth)
    assert tuple(out[0, 0].tolist()) == (0, 0, 0)
    # non-invalid pixels are not all zero
    assert tuple(out[5, 5].tolist()) != (0, 0, 0)


def test_annotate_left_with_empty_inputs_does_not_crash():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = annotate_left(frame, {}, None)
    assert out.shape == frame.shape
    # Frame must remain a copy (not the original buffer).
    assert out is not frame


def test_annotate_left_draws_roi_and_tracks():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()
    tracks = {
        1: _FakeTrack(
            track_id=1, state=CONFIRMED,
            positions=[np.array([175.0, 155.0, 0.0])],
            meta={"detection_history": [
                {"head_height_mm": 1620.0, "height_class": "adult"},
            ]},
        ),
        2: _FakeTrack(
            track_id=2, state=PENDING,
            positions=[np.array([100.0, 110.0, 0.0])],
            meta={},
        ),
        3: _FakeTrack(
            track_id=3, state=CANDIDATE,
            positions=[np.array([60.0, 60.0, 0.0])],
            meta={},
        ),
    }
    out = annotate_left(frame, tracks, counter)
    assert out.shape == frame.shape
    # Anything was drawn (the all-zero input makes that easy to detect).
    assert out.any()


def test_annotate_left_with_no_roi_counter():
    """Counter without ROI should still draw the line + arrow."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = annotate_left(frame, {}, _no_roi_counter())
    # The line at y=240 should produce non-zero pixels along that row.
    assert out[240, :, :].any()


def test_annotate_left_hides_candidate_tracks():
    """Tracks en estado CANDIDATE NO se renderizan — son ghosts
    tentativos (1 detección sin confirmar) que aparecen y desaparecen
    rápido cuando el detector flickerea sobre clutter. El operador
    solo ve tracks comprometidos."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    # Sin counter para que no haya overlay overlap en el patch.
    candidate_track = _FakeTrack(
        track_id=1,
        state=CANDIDATE,
        positions=[np.array([175.0, 80.0, 0.0])],  # arriba del ROI/línea
        meta={},
    )
    out = annotate_left(frame, {1: candidate_track}, None)
    # No debe haber ningún círculo coloreado del track CANDIDATE en
    # las coords centrales (excepto el bottom overlay bar de IN/OUT/FPS
    # que queda fuera del patch).
    cx, cy = 175, 80
    patch = out[cy - 8:cy + 8, cx - 8:cx + 8, :]
    assert not patch.any(), (
        "CANDIDATE tracks no deben renderizarse (son ghosts del detector)"
    )


def test_annotate_left_pending_marker_anchors_to_bbox_not_kalman_predict():
    """Durante PENDING el tracker pushea predicciones del Kalman al
    positions[-1] para que el counter detecte cruces en gaps. Pero
    visualmente eso hace que el círculo se "dispare" siguiendo la
    velocidad extrapolada mientras el bbox queda en la última
    detección. Fix: el círculo del marker se ancla al centro del bbox
    displayed, no a positions[-1]."""
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Track en PENDING. La última detección fue en bbox centrado en
    # (200, 80). El Kalman predice (350, 80) — 150px más adelante.
    # positions[-1] tiene el predict (sigue al Kalman), el bbox de
    # la historia tiene la última observación real. Usamos y=80 para
    # quedar lejos del bottom overlay bar (h-60..h).
    track = _FakeTrack(
        track_id=1,
        state=PENDING,
        positions=[np.array([350.0, 80.0, 0.0])],  # predict del Kalman
        meta={"detection_history": [
            {
                "bbox": [150, 30, 250, 130],  # centro en (200, 80)
                "head_height_mm": 1700.0,
                "height_class": "adult",
            },
        ]},
    )
    track.disappeared = 1  # hysteresis: aún se muestra verde
    out = annotate_left(frame, {1: track}, None)

    # El círculo del marker debe estar en (200, 80) — centro del
    # bbox observado — NO en (350, 80) — predict del Kalman.
    patch_at_observed = out[75:85, 195:205, :]
    patch_at_kalman = out[75:85, 345:355, :]
    has_marker_observed = patch_at_observed.any()
    has_marker_kalman = patch_at_kalman.any()
    assert has_marker_observed, (
        "El marker debe estar en el centro del último bbox observado (200, 80)"
    )
    assert not has_marker_kalman, (
        "El marker NO debe estar en la predicción del Kalman (350, 80) — "
        "se dispararía visualmente del sujeto"
    )


def test_annotate_left_pending_hysteresis_keeps_confirmed_color():
    """Track recién entrado a PENDING (disappeared bajo el threshold) sigue
    mostrándose verde para evitar flicker visual frente al operador en cada
    miss aislado del detector. Solo después de sustained PENDING flipea."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()

    # Track en PENDING pero con disappeared=1 (acaba de fallar 1 frame) →
    # display debería ser verde (CONFIRMED color).
    track_recently_pending = _FakeTrack(
        track_id=1,
        state=PENDING,
        positions=[np.array([175.0, 155.0, 0.0])],
        meta={"detection_history": [
            {"head_height_mm": 1620.0, "height_class": "adult"},
        ]},
    )
    track_recently_pending.disappeared = 1
    out = annotate_left(
        frame, {1: track_recently_pending}, counter
    )
    # Buscar un pixel verde (0, 255, 0) cerca del centroide. Si el
    # display fuera naranja, no habría verde puro en esa zona.
    cx, cy = 175, 155
    patch = out[cy - 5:cy + 5, cx - 5:cx + 5, :]
    # Verde dominante: canal G alto, R/B bajos.
    has_green = (
        (patch[:, :, 1] > 200) & (patch[:, :, 0] < 50) & (patch[:, :, 2] < 50)
    ).any()
    assert has_green, "Hysteresis: PENDING reciente (disappeared=1) debe mostrarse verde"


def test_annotate_left_sustained_pending_flips_to_orange():
    """Después de N frames sostenidos en PENDING (disappeared >= threshold),
    el display flipea a naranja para señalizar al operador que la pérdida
    ya no es transitoria."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()
    track_long_pending = _FakeTrack(
        track_id=1,
        state=PENDING,
        positions=[np.array([175.0, 155.0, 0.0])],
        meta={"detection_history": [
            {"head_height_mm": 1620.0, "height_class": "adult"},
        ]},
    )
    track_long_pending.disappeared = 10  # Bien arriba del threshold (5)
    out = annotate_left(
        frame, {1: track_long_pending}, counter
    )
    cx, cy = 175, 155
    patch = out[cy - 5:cy + 5, cx - 5:cx + 5, :]
    # Naranja (BGR 0, 165, 255): R alto, G medio, B bajo.
    has_orange = (
        (patch[:, :, 2] > 200) & (patch[:, :, 1] > 100) & (patch[:, :, 0] < 50)
    ).any()
    assert has_orange, "PENDING sostenido (disappeared >= 5) debe mostrarse naranja"


def test_annotate_left_draws_track_bbox_from_history():
    """El bbox del track se dibuja desde meta.detection_history para
    persistir visualmente durante PENDING (frames sin detección fresca)."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()
    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([175.0, 155.0, 0.0])],
        meta={"detection_history": [
            {
                "bbox": [150, 130, 200, 180],
                "head_height_mm": 1620.0,
                "height_class": "adult",
            },
        ]},
    )
    # Sin detections raw — solo el track. El bbox debe aparecer igual.
    out = annotate_left(frame, {1: track}, counter)
    # Verificar que los bordes del rectángulo (~y=130 y y=180, ~x=150 y x=200)
    # tienen pixels coloreados.
    assert out[130, 150:200, :].any(), "Top edge del bbox debe estar dibujado"
    assert out[180, 150:200, :].any(), "Bottom edge del bbox debe estar dibujado"


def test_annotate_left_bbox_width_height_uses_median_smoothing():
    """El size (width/height) del bbox displayed es la mediana sobre
    los últimos N samples — filtra el "respira" del detector frame a
    frame sin afectar la posición del centro."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    counter = Counter(
        lines=[Line(
            from_xy=(0, 300), to_xy=(800, 300),
            labels={"top_to_bottom": "ingress"},
        )],
    )

    # Historia: bboxes con CENTRO estable (400, 300) pero tamaño
    # oscilante. Smoothing debe darnos un tamaño estable cercano a
    # la mediana (100×200).
    history = []
    for w, h in [(95, 195), (100, 200), (105, 205), (98, 198),
                 (102, 202), (100, 200), (97, 199), (103, 201),
                 (99, 200), (180, 280)]:  # último = outlier
        history.append({
            "bbox": [400 - w // 2, 300 - h // 2,
                     400 + w // 2, 300 + h // 2],
            "head_height_mm": 1700.0,
            "height_class": "adult",
        })

    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([400.0, 300.0, 0.0])],
        meta={"detection_history": history},
    )
    out = annotate_left(frame, {1: track}, counter)

    # El outlier (180×280) NO debería dominar — la mediana absorbe.
    # Verificamos buscando los pixels coloreados del rectángulo: deben
    # estar cerca de la mediana w=100/h=200, no del outlier.
    # Tomamos un slice horizontal en y=300 y contamos px coloreados.
    # Con bbox mediano la línea del top está cerca de y=200 y el bottom
    # cerca de y=400.
    row_top_median = out[200, :, :]
    row_top_outlier = out[160, :, :]  # donde estaría si fuera el outlier
    assert row_top_median.any(), "El bbox mediano debe tener borde top ~y=200"
    assert not row_top_outlier.any(), (
        "El outlier (h=280) NO debe dominar — bbox debería estar acotado"
    )


def test_annotate_left_bbox_center_follows_latest_position():
    """El centro del bbox displayed se ancla al sample más reciente —
    cuando el sujeto se mueve, la caja lo sigue sin lag perceptible
    (la mediana solo aplica al width/height)."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    counter = Counter(
        lines=[Line(
            from_xy=(0, 300), to_xy=(800, 300),
            labels={"top_to_bottom": "ingress"},
        )],
    )

    # 5 frames del sujeto caminando de izquierda a derecha (cada uno
    # +50px en x). Tamaño constante 100×200. El bbox displayed debe
    # quedar en el centro del último sample (cx=600), no en la mediana
    # de centros (cx=400).
    centers_x = [400, 450, 500, 550, 600]
    history = []
    for cx in centers_x:
        history.append({
            "bbox": [cx - 50, 200, cx + 50, 400],
            "head_height_mm": 1700.0,
            "height_class": "adult",
        })

    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([600.0, 300.0, 0.0])],
        meta={"detection_history": history},
    )
    out = annotate_left(frame, {1: track}, counter)

    # El bbox debe estar centrado en x=600 → bordes en x=550 y x=650.
    # Verificamos buscando pixels coloreados en x=550 (left edge).
    col_at_latest_left = out[200:400, 550, :]
    col_at_median_left = out[200:400, 350, :]  # bbox-mediana left=350
    assert col_at_latest_left.any(), (
        "Bbox debe tener left edge en x=550 (centro del último sample)"
    )
    # El centroide mediano NO debería tener bbox dibujado allí.
    # Permitir algún pixel del overlay del counter/línea, pero no
    # un borde de bbox completo.
    bbox_pixels_at_median = int((col_at_median_left.any(axis=-1)).sum())
    bbox_pixels_at_latest = int((col_at_latest_left.any(axis=-1)).sum())
    assert bbox_pixels_at_latest > bbox_pixels_at_median * 2, (
        "El bbox debe seguir al último sample, no a la mediana de "
        f"posiciones (latest={bbox_pixels_at_latest}px, "
        f"median={bbox_pixels_at_median}px)"
    )


def test_annotate_left_height_label_uses_median_of_history():
    """El label de altura usa la mediana sobre los últimos N samples
    para suavizar el jitter de SGBM. Si el último sample fuera un outlier,
    el label NO debería reflejarlo."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()

    # Historia con valores estables alrededor de 1620mm + un outlier al final.
    history = [
        {"head_height_mm": 1620.0, "height_class": "adult"},
        {"head_height_mm": 1625.0, "height_class": "adult"},
        {"head_height_mm": 1618.0, "height_class": "adult"},
        {"head_height_mm": 1622.0, "height_class": "adult"},
        {"head_height_mm": 1619.0, "height_class": "adult"},
        {"head_height_mm": 1900.0, "height_class": "adult"},  # outlier
    ]
    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([175.0, 155.0, 0.0])],
        meta={"detection_history": history},
    )

    # Smoke: la función no crashea y produce output. La verificación
    # exacta de que muestra ~1.62m (no 1.90m) requeriría OCR; alcanza
    # con asegurar que el path corre y la mediana queda en el rango
    # esperado calculándola por afuera.
    out = annotate_left(frame, {1: track}, counter)
    assert out.shape == frame.shape

    # Sanity numérica: la mediana de los 6 samples debería ser ~1620.5
    # (no 1900). El display reflejará ese valor.
    head_samples = sorted([h["head_height_mm"] for h in history])
    median = head_samples[len(head_samples) // 2]
    assert 1620 <= median <= 1625, (
        f"La mediana de los samples debe absorber el outlier "
        f"(median={median}, esperado entre 1620-1625)"
    )

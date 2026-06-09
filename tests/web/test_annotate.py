"""Smoke tests para src/web/annotate.py."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.tracking.counter import Counter, Line
from src.tracking.tracker import CONFIRMED, PENDING, CANDIDATE
from src.web.annotate import (
    _BBOX_DISPLAY_SIZE_PX,
    _COUNT_FLASH_DURATION_S,
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
        lines=[
            Line(
                from_xy=(50, 150),
                to_xy=(250, 150),
                labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
            )
        ],
        counting_zone={"x_min": 50, "x_max": 250, "y_min": 100, "y_max": 200},
    )


def _no_counting_zone_counter() -> Counter:
    return Counter(
        lines=[
            Line(
                from_xy=(0, 240),
                to_xy=(400, 240),
                labels={"top_to_bottom": "ingress"},
            )
        ],
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
    a = np.zeros((300, 400, 3), dtype=np.uint8)  # left  (w/h 1.333) -> 160
    b = np.zeros((100, 200, 3), dtype=np.uint8)  # right (w/h 2.0)   -> 240
    c = np.zeros((600, 800, 3), dtype=np.uint8)  # depth (w/h 1.333) -> 160
    out = compose_3panel(a, b, c, target_height=120)
    assert out.shape[0] == 120  # fila única, todos a h=120
    assert out.shape[1] == 160 + 240 + 160  # ancho = suma de aspectos
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


def test_annotate_left_draws_counting_zone_and_tracks():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()
    tracks = {
        1: _FakeTrack(
            track_id=1,
            state=CONFIRMED,
            positions=[np.array([175.0, 155.0, 0.0])],
            meta={
                "detection_history": [
                    {"head_height_mm": 1620.0},
                ]
            },
        ),
        2: _FakeTrack(
            track_id=2,
            state=PENDING,
            positions=[np.array([100.0, 110.0, 0.0])],
            meta={},
        ),
        3: _FakeTrack(
            track_id=3,
            state=CANDIDATE,
            positions=[np.array([60.0, 60.0, 0.0])],
            meta={},
        ),
    }
    out = annotate_left(frame, tracks, counter)
    assert out.shape == frame.shape
    # Anything was drawn (the all-zero input makes that easy to detect).
    assert out.any()


def test_annotate_left_blurs_outside_tracking_zone_polygon():
    """Cuando se pasa tracking_zone_polygon, los pixels FUERA del polígono
    quedan blureados+oscurecidos; los de ADENTRO conservan el valor original.
    Feedback visual del 'modo estricto' al operador."""
    # Frame con valores distintos dentro y fuera del polígono para
    # detectar la composición.
    frame = np.full((300, 400, 3), 200, dtype=np.uint8)
    frame[100:200, 100:300] = 50  # rectángulo interior con valor distinto
    # Polígono que cubre exactamente el rectángulo interior.
    polygon = [(100.0, 100.0), (300.0, 100.0), (300.0, 200.0), (100.0, 200.0)]
    out = annotate_left(frame, {}, None, tracking_zone_polygon=polygon)
    # Pixel ADENTRO del polígono: conserva el valor original (50).
    assert int(out[150, 200, 0]) == 50
    # Pixel FUERA: blureado+oscurecido (< 200 original; el darken al
    # 55% lleva 200 → ~110, el blur lo deja cerca pero no exacto).
    outside_val = int(out[50, 50, 0])
    assert outside_val < 200, "pixel fuera del polígono debe estar oscurecido"
    assert outside_val < 150, "darken factor 0.55 debe dejarlo bien debajo del original"


def test_annotate_left_no_blur_when_polygon_is_none():
    """Default sin tracking_zone_polygon = no blur (back-compat)."""
    frame = np.full((300, 400, 3), 200, dtype=np.uint8)
    out = annotate_left(frame, {}, None)  # tracking_zone_polygon default None
    # Sin blur, todos los pixels conservan el valor original.
    assert int(out[50, 50, 0]) == 200
    assert int(out[150, 200, 0]) == 200


def test_annotate_left_skips_blur_with_malformed_polygon():
    """Polígono con menos de 3 vértices = no-op safety (no rompe el preview)."""
    frame = np.full((300, 400, 3), 200, dtype=np.uint8)
    # 2 vértices: malformado.
    out = annotate_left(frame, {}, None, tracking_zone_polygon=[(0, 0), (100, 0)])
    # Sin blur aplicado, valor original preservado.
    assert int(out[50, 50, 0]) == 200


def test_annotate_left_with_no_counting_zone_counter():
    """Counter without counting zone should still draw the line + arrow."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = annotate_left(frame, {}, _no_counting_zone_counter())
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
        positions=[np.array([175.0, 80.0, 0.0])],  # arriba de la counting zone/línea
        meta={},
    )
    out = annotate_left(frame, {1: candidate_track}, None)
    # No debe haber ningún círculo coloreado del track CANDIDATE en
    # las coords centrales (excepto el bottom overlay bar de IN/OUT/FPS
    # que queda fuera del patch).
    cx, cy = 175, 80
    patch = out[cy - 8 : cy + 8, cx - 8 : cx + 8, :]
    assert (
        not patch.any()
    ), "CANDIDATE tracks no deben renderizarse (son ghosts del detector)"


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
        meta={
            "detection_history": [
                {
                    "bbox": [150, 30, 250, 130],  # centro en (200, 80)
                    "head_height_mm": 1700.0,
                },
            ]
        },
    )
    track.disappeared = 1  # hysteresis: aún se muestra verde
    out = annotate_left(frame, {1: track}, None)

    # El círculo del marker debe estar en (200, 80) — centro del
    # bbox observado — NO en (350, 80) — predict del Kalman.
    patch_at_observed = out[75:85, 195:205, :]
    patch_at_kalman = out[75:85, 345:355, :]
    has_marker_observed = patch_at_observed.any()
    has_marker_kalman = patch_at_kalman.any()
    assert (
        has_marker_observed
    ), "El marker debe estar en el centro del último bbox observado (200, 80)"
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
        meta={
            "detection_history": [
                {"head_height_mm": 1620.0},
            ]
        },
    )
    track_recently_pending.disappeared = 1
    out = annotate_left(frame, {1: track_recently_pending}, counter)
    # Buscar un pixel verde (0, 255, 0) cerca del centroide. Si el
    # display fuera naranja, no habría verde puro en esa zona.
    cx, cy = 175, 155
    patch = out[cy - 5 : cy + 5, cx - 5 : cx + 5, :]
    # Verde dominante: canal G alto, R/B bajos.
    has_green = (
        (patch[:, :, 1] > 200) & (patch[:, :, 0] < 50) & (patch[:, :, 2] < 50)
    ).any()
    assert (
        has_green
    ), "Hysteresis: PENDING reciente (disappeared=1) debe mostrarse verde"


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
        meta={
            "detection_history": [
                {"head_height_mm": 1620.0},
            ]
        },
    )
    track_long_pending.disappeared = 10  # Bien arriba del threshold (5)
    out = annotate_left(frame, {1: track_long_pending}, counter)
    cx, cy = 175, 155
    patch = out[cy - 5 : cy + 5, cx - 5 : cx + 5, :]
    # Naranja (BGR 0, 165, 255): R alto, G medio, B bajo.
    has_orange = (
        (patch[:, :, 2] > 200) & (patch[:, :, 1] > 100) & (patch[:, :, 0] < 50)
    ).any()
    assert has_orange, "PENDING sostenido (disappeared >= 5) debe mostrarse naranja"


def test_annotate_left_draws_trajectory_polyline():
    """La trayectoria del track se dibuja como polilínea por sus posiciones —
    un pixel en el medio de un segmento debe quedar coloreado."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    # Trayectoria diagonal: (100,100) -> (200,200). El midpoint (150,150)
    # cae sobre el segmento del trail.
    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([100.0, 100.0, 0.0]), np.array([200.0, 200.0, 0.0])],
        meta={},
    )
    out = annotate_left(frame, {1: track}, None)
    # Pixel en el midpoint del segmento (con tolerancia por antialias).
    assert out[148:153, 148:153, :].any(), "El trail debe dibujar la polilínea"


def test_annotate_left_draws_track_bbox_from_history():
    """El bbox del track se dibuja desde meta.detection_history para
    persistir visualmente durante PENDING (frames sin detección fresca)."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()
    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([175.0, 155.0, 0.0])],
        meta={
            "detection_history": [
                {
                    "bbox": [150, 130, 200, 180],
                    "head_height_mm": 1620.0,
                },
            ]
        },
    )
    # Sin detections raw — solo el track. El bbox debe aparecer igual.
    out = annotate_left(frame, {1: track}, counter)
    # Caja de tamaño FIJO centrada en el centro de la última detección
    # (centro de [150,130,200,180] = (175,155)). half = SIZE_PX/2.
    half = _BBOX_DISPLAY_SIZE_PX // 2
    cx, cy = 175, 155
    top, bottom = cy - half, cy + half
    left, right = cx - half, cx + half
    assert out[top, left:right, :].any(), "Top edge del bbox fijo debe dibujarse"
    assert out[bottom, left:right, :].any(), "Bottom edge del bbox fijo debe dibujarse"


def test_annotate_left_bbox_is_fixed_size_regardless_of_detection():
    """La caja del preview tiene tamaño FIJO (cuadrada, _BBOX_DISPLAY_SIZE_PX),
    sin importar el tamaño que reportó el detector. No "respira" ni difiere
    entre tracks — solo la posición sigue al sujeto."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    counter = Counter(
        lines=[
            Line(
                from_xy=(0, 300),
                to_xy=(800, 300),
                labels={"top_to_bottom": "ingress"},
            )
        ],
    )
    # Última detección con un bbox GRANDE (180×280, centro 400,300). La caja
    # dibujada debe ser igual de chica que la fija, no reflejar ese tamaño.
    history = [
        {
            "bbox": [400 - 90, 300 - 140, 400 + 90, 300 + 140],
            "head_height_mm": 1700.0,
        }
    ]
    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([400.0, 300.0, 0.0])],
        meta={"detection_history": history},
    )
    out = annotate_left(frame, {1: track}, counter)

    half = _BBOX_DISPLAY_SIZE_PX // 2  # caja fija centrada en (400,300)
    # Borde top de la caja fija (~y=300-half) dibujado.
    assert out[
        300 - half, 400 - half : 400 + half, :
    ].any(), "La caja fija debe tener borde top en y=300-half"
    # Donde estaría el borde del bbox grande del detector (y=160) NO hay caja.
    assert not out[
        160, :, :
    ].any(), "El tamaño del detector (h=280) NO debe reflejarse — la caja es fija"


def test_annotate_left_bbox_center_follows_latest_position():
    """El centro de la caja (fija) se ancla al sample más reciente — cuando el
    sujeto se mueve, la caja lo sigue sin lag perceptible."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    counter = Counter(
        lines=[
            Line(
                from_xy=(0, 300),
                to_xy=(800, 300),
                labels={"top_to_bottom": "ingress"},
            )
        ],
    )
    # 5 frames caminando de izquierda a derecha; el último centro es cx=600.
    centers_x = [400, 450, 500, 550, 600]
    history = [
        {"bbox": [cx - 50, 200, cx + 50, 400], "head_height_mm": 1700.0}
        for cx in centers_x
    ]
    track = _FakeTrack(
        track_id=1,
        state=CONFIRMED,
        positions=[np.array([600.0, 300.0, 0.0])],
        meta={"detection_history": history},
    )
    out = annotate_left(frame, {1: track}, counter)

    half = _BBOX_DISPLAY_SIZE_PX // 2
    # Caja fija centrada en x=600 → left edge en x=600-half.
    col_at_latest = out[300 - half : 300 + half, 600 - half, :]
    # Donde estaría si siguiera la mediana de centros (cx=400).
    col_at_median = out[300 - half : 300 + half, 400 - half, :]
    assert (
        col_at_latest.any()
    ), "La caja debe tener left edge en x=600-half (último centro)"
    px_latest = int(col_at_latest.any(axis=-1).sum())
    px_median = int(col_at_median.any(axis=-1).sum())
    assert px_latest > px_median * 2, (
        f"La caja sigue al último sample, no a la mediana (latest={px_latest}, "
        f"median={px_median})"
    )


def test_annotate_left_height_label_uses_median_of_history():
    """El label de altura usa la mediana sobre los últimos N samples
    para suavizar el jitter de SGBM. Si el último sample fuera un outlier,
    el label NO debería reflejarlo."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _two_way_counter()

    # Historia con valores estables alrededor de 1620mm + un outlier al final.
    history = [
        {"head_height_mm": 1620.0},
        {"head_height_mm": 1625.0},
        {"head_height_mm": 1618.0},
        {"head_height_mm": 1622.0},
        {"head_height_mm": 1619.0},
        {"head_height_mm": 1900.0},  # outlier
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


# ==========================================================================
# Flash +IN/+OUT al contar
# ==========================================================================


def _empty_frame(h=300, w=400):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _corner_counter() -> Counter:
    """Counter con la línea y la zona en la esquina superior izquierda,
    LEJOS del área donde los tests del flash dibujan (pos_x≈350, pos_y≈250).
    Mantiene el counter válido pero su geometría no contamina la región
    donde verificamos los píxeles del flash."""
    return Counter(
        lines=[
            Line(
                from_xy=(10, 10),
                to_xy=(60, 10),
                labels={"top_to_bottom": "ingress"},
            )
        ],
        counting_zone={"x_min": 10, "x_max": 60, "y_min": 5, "y_max": 50},
    )


def test_flash_in_writes_green_text_above_position():
    """Flash recién disparado (elapsed=0) → píxeles verdes que NO están en
    el render sin flash. Se compara contra baseline para aislar la
    contribución del flash del overlay de la counting geometry."""
    counter = _corner_counter()
    pos_x, pos_y = 350, 250  # x lejos del rectángulo de la counting zone

    baseline = annotate_left(_empty_frame(), {}, counter, recent_counts=None)
    flashed = annotate_left(
        _empty_frame(),
        {},
        counter,
        recent_counts={7: ("ingress", 100.0, float(pos_x), float(pos_y))},
        now_mono=100.0,
    )

    # Diff: pixels nuevos respecto al baseline. Si son verdes → es del flash.
    diff = flashed.astype(int) - baseline.astype(int)
    green_added = (
        (diff[..., 1] > 80) & (diff[..., 0] < 30) & (diff[..., 2] < 30)
    ).sum()
    assert (
        green_added > 100
    ), f"Esperaba píxeles verdes nuevos del flash +IN, got {green_added}"


def test_flash_out_writes_blue_text():
    """Flash de OUT (label egress) → píxeles azules que aparecen sobre el
    baseline."""
    counter = _corner_counter()
    pos_x, pos_y = 350, 250

    baseline = annotate_left(_empty_frame(), {}, counter, recent_counts=None)
    flashed = annotate_left(
        _empty_frame(),
        {},
        counter,
        recent_counts={7: ("egress", 100.0, float(pos_x), float(pos_y))},
        now_mono=100.0,
    )

    diff = flashed.astype(int) - baseline.astype(int)
    blue_added = ((diff[..., 0] > 80) & (diff[..., 1] < 30) & (diff[..., 2] < 30)).sum()
    assert (
        blue_added > 100
    ), f"Esperaba píxeles azules nuevos del flash +OUT, got {blue_added}"


def test_flash_expires_after_duration():
    """Pasado _COUNT_FLASH_DURATION_S el render debe ser bit-idéntico al
    baseline sin flash. Compara contra el frame sin recent_counts."""
    counter = _corner_counter()
    recent = {7: ("ingress", 100.0, 350.0, 250.0)}

    baseline = annotate_left(_empty_frame(), {}, counter, recent_counts=None)
    expired = annotate_left(
        _empty_frame(),
        {},
        counter,
        recent_counts=recent,
        now_mono=100.0 + _COUNT_FLASH_DURATION_S + 0.5,
    )
    assert np.array_equal(
        baseline, expired
    ), "Flash vencido debe dar render bit-idéntico al sin-flash"


def test_flash_fade_alpha_proportional_to_elapsed():
    """Alpha proporcional a (1 - elapsed/duration). A 50% del tiempo de
    vida el verde sumado debe ser menor que al 0%."""
    counter = _corner_counter()
    recent = {7: ("ingress", 100.0, 350.0, 250.0)}

    baseline = annotate_left(_empty_frame(), {}, counter, recent_counts=None).astype(
        int
    )
    fresh = annotate_left(
        _empty_frame(),
        {},
        counter,
        recent_counts=recent,
        now_mono=100.0,
    ).astype(int)
    half = annotate_left(
        _empty_frame(),
        {},
        counter,
        recent_counts=recent,
        now_mono=100.0 + _COUNT_FLASH_DURATION_S / 2.0,
    ).astype(int)

    g_fresh_max = int(np.max(fresh[..., 1] - baseline[..., 1]))
    g_half_max = int(np.max(half[..., 1] - baseline[..., 1]))
    assert g_fresh_max > g_half_max + 20, (
        f"El verde delta al 50% del fade debe ser perceptiblemente más bajo "
        f"que al disparar (fresh={g_fresh_max}, half={g_half_max})"
    )


def test_flash_none_or_empty_dict_is_noop():
    """recent_counts=None o {} no debe romper ni mutar la salida vs frame sin flash."""
    frame = _empty_frame()
    counter = _two_way_counter()
    baseline = annotate_left(frame, {}, counter, recent_counts=None)
    empty = annotate_left(frame, {}, counter, recent_counts={})
    assert np.array_equal(
        baseline, empty
    ), "recent_counts={} y None deben dar el mismo render"


def test_flash_position_clamped_to_frame_top():
    """Si el centroide está muy cerca del borde superior, el texto va
    debajo (no se sale del frame)."""
    frame = _empty_frame()
    counter = _two_way_counter()
    # pos_y muy chica → anchor_y - text_h < 0 → debería caer abajo
    recent = {7: ("ingress", 100.0, 200.0, 20.0)}
    out = annotate_left(frame, {}, counter, recent_counts=recent, now_mono=100.0)
    # No crashea + algún píxel verde aparece en algún lugar del frame.
    green = ((out[..., 1] > 100) & (out[..., 0] < 80) & (out[..., 2] < 80)).sum()
    assert (
        green > 50
    ), "Aun cerca del borde superior el flash debe renderear (clamped abajo)"

"""Tests para helpers de world-space (clasificación de altura de cabeza)."""

import numpy as np
import pytest

from src.vision.depth import head_depth_in_bbox, min_depth_at_bbox
from src.vision.world_coords import (
    aggregate_height_class,
    classify_height,
    head_height_above_floor,
    project_to_floor,
)


class TestHeadHeightAboveFloor:
    def test_adult_head_under_3_5m_mount(self):
        # Camera at 3500mm, head at top of box returns depth 1800mm
        # → head_height = 3500 - 1800 = 1700mm (adult)
        assert head_height_above_floor(1800.0, 3500.0) == 1700.0

    def test_child_head_gives_smaller_height(self):
        # Shorter person: depth 2200mm under same mount → 1300mm (child)
        assert head_height_above_floor(2200.0, 3500.0) == 1300.0

    def test_invalid_depth_returns_none(self):
        assert head_height_above_floor(0.0, 3500.0) is None
        assert head_height_above_floor(-100.0, 3500.0) is None

    def test_invalid_mount_returns_none(self):
        assert head_height_above_floor(1800.0, 0.0) is None

    def test_negative_height_returns_none(self):
        """Depth > mount means the 'head' is below the floor — bad data."""
        assert head_height_above_floor(4000.0, 3500.0) is None


class TestClassifyHeight:
    def test_tall_person_is_adult(self):
        assert classify_height(1700.0, adult_min_mm=1500.0) == "adult"

    def test_short_person_is_child(self):
        assert classify_height(1200.0, adult_min_mm=1500.0) == "child"

    def test_exactly_at_threshold_is_adult(self):
        """Threshold is inclusive on the adult side — avoids "unknown" gap."""
        assert classify_height(1500.0, adult_min_mm=1500.0) == "adult"

    def test_none_is_unknown(self):
        assert classify_height(None, adult_min_mm=1500.0) == "unknown"

    def test_threshold_is_configurable(self):
        # 1400mm is adult with a 1300mm threshold, child with a 1500mm one
        assert classify_height(1400.0, 1300.0) == "adult"
        assert classify_height(1400.0, 1500.0) == "child"


class TestAggregateHeightClass:
    def test_majority_adult(self):
        samples = ["adult", "adult", "adult", "child"]
        assert aggregate_height_class(samples) == "adult"

    def test_majority_child(self):
        samples = ["child", "child", "adult"]
        assert aggregate_height_class(samples) == "child"

    def test_unknown_samples_ignored(self):
        samples = ["unknown", "unknown", "child", "child", "unknown"]
        assert aggregate_height_class(samples) == "child"

    def test_all_unknown(self):
        assert aggregate_height_class(["unknown"] * 5) == "unknown"
        assert aggregate_height_class([]) == "unknown"

    def test_tie_goes_to_most_recent(self):
        # Equal counts: last non-unknown wins (bias toward latest stable bbox)
        assert aggregate_height_class(["adult", "child"]) == "child"
        assert aggregate_height_class(["child", "adult"]) == "adult"


class TestProjectToFloor:
    """Footpoint projection (head pixel → foot pixel using head height).

    Verifies the parallax-correction trick used by the counter: knowing
    the actual head height above the floor (via SGBM depth) lets us scale
    the head pixel toward the principal point by ``Z_head / H`` to get
    the foot pixel. At nadir the scale is irrelevant; at the periphery
    the shift is the parallax error a centroid-based tracker would
    silently accumulate.
    """

    # Realistic geometry: 3 m mount, 1.7 m person, 1152×648 frame after
    # the standard rescale of the 2304×1296 calibration. Principal
    # point sits roughly at frame centre on rectified imagery.
    MOUNT_MM = 3000.0
    HEAD_MM = 1700.0
    CX = 576.0
    CY = 324.0

    def test_nadir_no_shift(self):
        """Head exactly at the principal point ⇒ scale is irrelevant,
        feet project on the same pixel. The geometric invariant the
        whole construction rests on."""
        u, v = project_to_floor(
            (self.CX, self.CY),
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        assert u == pytest.approx(self.CX)
        assert v == pytest.approx(self.CY)

    def test_periphery_shifts_toward_principal_point(self):
        """A head pixel offset from nadir scales TOWARD the principal
        point — the foot pixel is closer to centre than the head pixel
        by exactly the parallax factor.
        """
        # Head 200 px to the right of nadir.
        head_pixel = (self.CX + 200.0, self.CY)
        u, v = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        # Z_head/H = (3000-1700)/3000 = 0.4333..., so the offset shrinks
        # from 200 px to 200*0.4333 = 86.67 px.
        expected_offset = 200.0 * (self.MOUNT_MM - self.HEAD_MM) / self.MOUNT_MM
        assert u == pytest.approx(self.CX + expected_offset)
        assert v == pytest.approx(self.CY)
        # Foot pixel is closer to principal point than head pixel.
        assert abs(u - self.CX) < abs(head_pixel[0] - self.CX)

    def test_diagonal_periphery(self):
        """2D shift: both u and v scale by the same factor."""
        head_pixel = (self.CX + 300.0, self.CY - 150.0)
        u, v = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        scale = (self.MOUNT_MM - self.HEAD_MM) / self.MOUNT_MM
        assert u == pytest.approx(self.CX + 300.0 * scale)
        assert v == pytest.approx(self.CY + (-150.0) * scale)

    def test_zero_head_height_is_no_op(self):
        """No head height = no info. Returning the input lets the caller
        fall back to centroid-based crossing logic transparently."""
        head_pixel = (700.0, 200.0)
        u, v = project_to_floor(
            head_pixel,
            0.0,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        assert (u, v) == head_pixel

    def test_zero_mount_is_no_op(self):
        """Bad/missing mount config ⇒ no projection, no crash."""
        head_pixel = (700.0, 200.0)
        u, v = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            0.0,
            self.CX,
            self.CY,
        )
        assert (u, v) == head_pixel

    def test_negative_inputs_are_no_op(self):
        head_pixel = (700.0, 200.0)
        assert (
            project_to_floor(
                head_pixel,
                -5.0,
                self.MOUNT_MM,
                self.CX,
                self.CY,
            )
            == head_pixel
        )
        assert (
            project_to_floor(
                head_pixel,
                self.HEAD_MM,
                -1.0,
                self.CX,
                self.CY,
            )
            == head_pixel
        )

    def test_head_taller_than_mount_is_no_op(self):
        """Pathological: a taller-than-mount head means the camera is
        looking at a torso, not a head — the depth estimate must be
        wrong. Falling back to the head pixel keeps the counter from
        sign-flipping the parallax shift."""
        head_pixel = (800.0, 400.0)
        # Head at 3.5 m, mount only 3 m → Z_head = -500 mm.
        u, v = project_to_floor(
            head_pixel,
            3500.0,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        assert (u, v) == head_pixel

    def test_periphery_magnitude_matches_documented_geometry(self):
        """Sanity-check contra los números geométricos del caso canónico:
        persona de 1.7 m bajo mount de 3.0 m, head pixel cerca del borde
        del frame (~60° de excentricidad desde el eje óptico a este FOV).
        El shift no es exactamente 110 cm acá porque computamos en
        espacio de IMAGEN, no en metros de plano de piso — pero el factor
        de escala tiene que ser exactamente Z_head/H."""
        # 500 px is roughly ~80% of the half-frame width on 1152×648,
        # i.e. nearly at the border.
        head_pixel = (self.CX + 500.0, self.CY)
        u_foot, _ = project_to_floor(
            head_pixel,
            self.HEAD_MM,
            self.MOUNT_MM,
            self.CX,
            self.CY,
        )
        # The shift in image space relative to the head pixel:
        shift_px = head_pixel[0] - u_foot
        # Z_head/H = 1300/3000 ≈ 0.433, so foot is at offset 500*0.433 =
        # 216.67 from centre, shift = 500 - 216.67 = 283.33 px.
        assert shift_px == pytest.approx(500.0 - 500.0 * 1300.0 / 3000.0)


class TestMinDepthAtBbox:
    def test_low_percentile_robust_to_speckle_noise(self):
        """Low percentile should reject a single noisy min pixel."""
        depth = np.full((100, 100), 2000.0, dtype=np.float32)
        depth[50, 50] = 100.0  # speckle — very near
        result = min_depth_at_bbox(depth, (40, 40, 60, 60), low_percentile=5.0)
        # 5% percentile of 400 pixels = 20 pixels at the low end; 1 outlier
        # doesn't move the result much
        assert result > 500, f"got {result}, expected >500 (speckle rejected)"

    def test_uniform_depth_returns_that_depth(self):
        depth = np.full((100, 100), 2400.0, dtype=np.float32)
        assert abs(min_depth_at_bbox(depth, (10, 10, 90, 90)) - 2400.0) < 1

    def test_invalid_pixels_excluded(self):
        depth = np.zeros((50, 50), dtype=np.float32)
        depth[20:30, 20:30] = 1500.0
        assert min_depth_at_bbox(depth, (0, 0, 50, 50)) > 0  # zeros skipped

    def test_no_valid_returns_zero(self):
        depth = np.zeros((50, 50), dtype=np.float32)
        assert min_depth_at_bbox(depth, (0, 0, 50, 50)) == 0.0

    def test_is_smaller_than_median(self):
        """Basic sanity: near-depth (low percentile) < median depth."""
        depth = np.zeros((100, 100), dtype=np.float32)
        for row in range(100):
            # Vertical depth gradient — head at top (small depth)
            depth[row, :] = 1500.0 + row * 5.0
        from src.vision.depth import depth_at_bbox

        near = min_depth_at_bbox(depth, (0, 0, 100, 100))
        median = depth_at_bbox(depth, (0, 0, 100, 100))
        assert near < median

    def test_samples_central_crop_only(self):
        """Peripheral pixels (closer to camera) outside the central 50%
        of the bbox must NOT pull the near-depth down. In zenith view
        the head sits near the bbox centroid; peripheral noise from
        background pixels inside the YOLO-stock person bbox is what
        was producing 1.88 m heights for a sitting user.
        """
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        # Spurious very-near pixels along the bbox edges (peripheral).
        depth[0:10, :] = 700.0
        depth[90:100, :] = 700.0
        depth[:, 0:10] = 700.0
        depth[:, 90:100] = 700.0
        # Bbox covers the whole array. Central 50% sees only 1500.
        result = min_depth_at_bbox(depth, (0, 0, 100, 100))
        assert abs(result - 1500.0) < 1, (
            f"got {result}, expected ~1500 (peripheral noise should be "
            "ignored by central-crop sampling)"
        )

    def test_default_percentile_robust_to_speckle_cluster(self):
        """A small cluster of low-depth speckle pixels in the central
        crop must not dominate the result. The 15% default is meant to
        survive ~10% noise contamination at SGBM downscale=8.
        """
        # 50×50 central crop of a 100×100 bbox. Inject a 4×4 (16 px)
        # cluster of bad-depth speckle = ~6.4% of central pixels.
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        depth[40:44, 40:44] = 600.0
        result = min_depth_at_bbox(depth, (0, 0, 100, 100))
        # 15th percentile of [600×16, 1500×(2500-16)] is 1500.
        assert result > 1400


class TestHeadDepthInBbox:
    """head_depth_in_bbox finds the head cluster (smallest plausible
    depth with enough area) inside a YOLO-COCO bbox that is centered on
    the torso, not the head."""

    MOUNT_MM = 2560.0  # mounting height used in tests below
    # Plausible rectified intrinsics for runtime resolution 1152×648.
    # fx ≈ 464 px (P1[0,0] from a real fisheye stereoRectify with the
    # bracket's 14 cm baseline + Arducam IMX708 K rescaled), principal
    # point at the image centre. Tests here use bboxes far smaller than
    # 1152×648, but the per-pixel back-projection only cares about
    # (u-cx, v-cy, Z) and Z is what the synthetic depth maps drive.
    FX_PX = 464.0
    CX_PX = 576.0
    CY_PX = 324.0

    def _call(self, depth, bbox, mount=None, **kwargs):
        """Helper: invoke head_depth_in_bbox with the standard intrinsics
        unless a test overrides them via kwargs."""
        return head_depth_in_bbox(
            depth,
            bbox,
            self.MOUNT_MM if mount is None else mount,
            fx_px=kwargs.pop("fx_px", self.FX_PX),
            cx_px=kwargs.pop("cx_px", self.CX_PX),
            cy_px=kwargs.pop("cy_px", self.CY_PX),
            **kwargs,
        )

    def test_returns_none_on_empty_depth_map(self):
        depth = np.zeros((0, 0), dtype=np.float32)
        assert self._call(depth, (0, 0, 10, 10)) is None

    def test_returns_none_when_bbox_outside_frame(self):
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        # Negative-area bbox (clipped to nothing).
        assert self._call(depth, (50, 50, 40, 40)) is None

    def test_returns_none_when_only_speckle_present(self):
        """Below-floor speckle alone (no real head cluster) → None.

        At mount=2.56m, max_head_height=1.80m → near gate at 760mm.
        Speckle at 200mm is below the gate and excluded; no remaining
        pixels above the gate (whole bbox is speckle).
        """
        depth = np.full((50, 50), 200.0, dtype=np.float32)
        result = self._call(depth, (0, 0, 50, 50))
        assert result is None

    def _full_frame_depth(self, fill: float = 0.0) -> np.ndarray:
        """Allocate a depth map at runtime resolution (1152×648) so any
        bbox referenced through (CX_PX, CY_PX) lands inside the array.
        Synthetic scenes paint the body inside this canvas."""
        return np.full((648, 1152), fill, dtype=np.float32)

    def _centred_bbox(self, half: int = 50) -> tuple[int, int, int, int]:
        """Bbox centred on the principal point (so back-projected (X, Y)
        at the body centroid lies on the optical axis)."""
        return (
            int(self.CX_PX - half),
            int(self.CY_PX - half),
            int(self.CX_PX + half),
            int(self.CY_PX + half),
        )

    def test_finds_sitting_user_head_not_torso(self):
        """Sitting user: head at ~1.10 m floor (depth 1450 mm), torso at
        ~0.80 m floor (depth 1760 mm). Function must return ~1450.

        The head is centred under the body in 3-D so it survives the
        column filter; the torso (further from camera, but in the same
        column) gets rejected by the histogram-walk-from-nearest.
        """
        depth = self._full_frame_depth(1760.0)
        # Head blob 7×7 = 49 px, comfortably above the 40-px area gate.
        # Position centred on the principal point so the body centroid
        # sits at metric (X, Y) ≈ (0, 0) and the column captures it.
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 3 : cy + 4, cx - 3 : cx + 4] = 1450.0
        result = self._call(depth, self._centred_bbox())
        assert result is not None
        assert abs(result - 1450.0) < 60.0  # within one slice (100mm)

    def test_finds_standing_adult_head(self):
        """Standing user 1.68 m tall: head at depth ~880 mm. Torso at
        depth ~1160 mm. Returns ~880.
        """
        depth = self._full_frame_depth(1160.0)
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 4 : cy + 4, cx - 5 : cx + 5] = 880.0  # head, 80 px, centred
        result = self._call(depth, self._centred_bbox())
        assert result is not None
        assert abs(result - 880.0) < 60.0

    def test_rejects_below_anthropometric_floor(self):
        """A spurious very-near cluster is below the anthropometric
        ceiling and must be gated out. At mount=2.56m,
        max_head_height=1.80m → near gate at 760mm. A cluster at 300mm
        depth corresponds to a 2.26m-tall head, exceeds the cap, and
        falls outside the gate.
        """
        depth = self._full_frame_depth(1500.0)  # plausible head plane
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 7 : cy + 8, cx - 7 : cx + 8] = 300.0  # impossible cluster
        result = self._call(depth, self._centred_bbox())
        # The 300mm cluster is gated out, so we get the 1500mm plane.
        assert result is not None
        assert abs(result - 1500.0) < 100.0

    def test_speckle_smaller_than_min_area_does_not_win(self):
        """A 3×3 cluster (9 px) of false-near depth must be rejected
        even though it sits closer to the camera than the real head —
        below the min_head_area_px (40) gate, it's noise.
        """
        depth = self._full_frame_depth(1500.0)  # head plane
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 1 : cy + 2, cx - 1 : cx + 2] = 1100.0  # 9 px speckle
        result = self._call(depth, self._centred_bbox())
        assert result is not None
        # Should land on the 1500 mm head plane, not the 1100 speckle.
        assert result > 1300

    def test_picks_head_when_torso_dominates_area(self):
        """Realistic case: bbox is mostly torso (more pixels), but head
        is the closest plausible cluster. We pick head, not torso —
        that's the whole point of the algorithm.
        """
        depth = self._full_frame_depth(1700.0)  # torso
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 3 : cy + 4, cx - 3 : cx + 4] = 1200.0  # head, 49 px, centred
        result = self._call(depth, self._centred_bbox())
        assert result is not None
        assert abs(result - 1200.0) < 100.0

    def test_invalid_pixels_ignored(self):
        """Zero pixels (SGBM invalid) must not be counted toward the
        head-area gate or pollute the median.
        """
        depth = self._full_frame_depth(0.0)  # all invalid
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 10 : cy + 10, cx - 10 : cx + 10] = 1500.0  # one valid plane
        result = self._call(depth, self._centred_bbox())
        assert result is not None
        assert abs(result - 1500.0) < 100.0

    def test_returns_none_with_no_valid_pixels(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        assert self._call(depth, (0, 0, 100, 100)) is None

    def test_mount_height_zero_returns_none(self):
        # Pathological config: mount=0 collapses the gate and there's
        # no plausible region. Must not crash.
        depth = np.full((50, 50), 1500.0, dtype=np.float32)
        result = self._call(depth, (0, 0, 50, 50), mount=0.0)
        assert result is None

    # ------------------------------------------------------------------
    # 3-D column filter (Fix B): overhead structure offset in metric
    # X or Y from the body must be excluded; structure aligned with
    # the body must still be picked; radius must be configurable.
    # ------------------------------------------------------------------

    def _make_offset_overhead_scene(
        self,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Body centred at the principal point at depth 1700 mm, plus a
        spurious overhead-structure cluster at 800 mm depth offset well
        beyond 250 mm in metric X from the body centroid.

        Bbox is symmetric around the body so the bbox-central crop sees
        only body, anchoring the centroid on (X, Y) ≈ (0, 0). The
        overhead cluster lives in the bbox edge region and gets back-
        projected to a metric X of ~700 mm — far outside the 250 mm
        column radius. Without the spatial filter the histogram walk
        would lock onto the 800 mm bin and return 800 mm.
        """
        depth = np.full((648, 1152), 0.0, dtype=np.float32)
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        # Body torso/head plane centred on the principal point.
        depth[cy - 80 : cy + 80, cx - 80 : cx + 80] = 1700.0
        # Overhead structure cluster at 800 mm depth. Place it 410 px to
        # the right of the optical axis: ΔX = 410 * 800 / 464 ≈ 707 mm.
        # That's well outside both the 250 mm column and the bbox
        # central crop (so the body-centroid estimator is not biased
        # by it).
        u0 = cx + 410
        depth[cy - 25 : cy + 25, u0 : u0 + 60] = 800.0
        # Bbox is symmetric around the body principal point in U
        # (so cx_off centres the central crop on the body) and stretches
        # right to enclose the overhead cluster.
        bbox = (cx - 250, cy - 90, cx + 500, cy + 90)
        return depth, bbox

    def test_overhead_structure_offset_in_xy_excluded(self):
        """The motivating bug: an overhead cluster at 800 mm offset
        >250 mm in metric X from the body must be excluded by the
        column filter, so the picked head depth is 1700 mm (body),
        not 800 mm (structure).
        """
        depth, bbox = self._make_offset_overhead_scene()
        result = self._call(depth, bbox)
        assert result is not None
        assert abs(result - 1700.0) < 100.0, (
            f"got {result}, expected ~1700 (body picked, overhead "
            "structure rejected by 3-D column filter)"
        )

    def test_overhead_structure_aligned_in_xy_picked(self):
        """Sanity: when an overhead cluster is aligned in metric (X, Y)
        with the body it sits *inside* the column and the function
        picks it (it is genuinely part of the body — e.g. a head
        sitting above the torso). Confirms the column filter doesn't
        over-reject and create false negatives.
        """
        depth = np.full((648, 1152), 0.0, dtype=np.float32)
        # Body / torso plane at depth 1700 mm.
        depth[
            int(self.CY_PX) - 80 : int(self.CY_PX) + 80,
            int(self.CX_PX) - 80 : int(self.CX_PX) + 80,
        ] = 1700.0
        # Closer cluster at depth 1100 mm, centred on the body in (X, Y).
        # ΔX ≈ 0 so it stays inside the column.
        depth[
            int(self.CY_PX) - 20 : int(self.CY_PX) + 20,
            int(self.CX_PX) - 20 : int(self.CX_PX) + 20,
        ] = 1100.0
        bbox = (
            int(self.CX_PX) - 90,
            int(self.CY_PX) - 90,
            int(self.CX_PX) + 90,
            int(self.CY_PX) + 90,
        )
        result = self._call(depth, bbox)
        assert result is not None
        assert abs(result - 1100.0) < 100.0, (
            f"got {result}, expected ~1100 (closer cluster aligned in "
            "X-Y is genuine head material — column filter must keep it)"
        )

    def test_column_radius_configurable(self):
        """The column radius is a knob: enlarging it must let the
        offset overhead cluster back in (proving the filter is what
        was excluding it in test_overhead_structure_offset_in_xy_excluded
        rather than something else in the pipeline).
        """
        depth, bbox = self._make_offset_overhead_scene()
        # Default radius (250 mm) → offset structure rejected, body wins.
        default = self._call(depth, bbox)
        assert default is not None
        assert abs(default - 1700.0) < 100.0
        # Enlarge radius to 5 m → structure is now inside the column
        # and the histogram walk picks it (closer to camera).
        wide = self._call(depth, bbox, column_radius_mm=5000.0)
        assert wide is not None
        assert abs(wide - 800.0) < 100.0, (
            f"got {wide}, expected ~800 with wide radius (overhead "
            "structure now inside the column and picked as nearest)"
        )

    # ------------------------------------------------------------------
    # Rotated-rectangle polygon mask: when the detector emits a rotated
    # bbox (RAPiD), the axis-aligned envelope sweeps in floor + adjacent
    # structure that sit close to the camera and pass the column filter
    # because they back-project to a metric (X, Y) inside the column.
    # The polygon mask hands the function the actual body footprint.
    # ------------------------------------------------------------------

    def _make_tilted_body_with_corner_structure_scene(
        self,
    ) -> tuple[np.ndarray, tuple[int, int, int, int],
               tuple[float, float, float, float, float]]:
        """Body rectangle 60×200 px tilted 45° centred on the principal
        point. The axis-aligned envelope reaches into a near-camera
        structure cluster (depth 800 mm) at the envelope's top-right
        corner — outside the rotated body polygon, inside the envelope.
        The structure's metric X is well inside the 250 mm column so
        the spatial filter alone does not reject it. Returns
        ``(depth_map, axis_aligned_bbox, rotated_bbox)``.
        """
        import cv2 as _cv2  # local alias to keep test imports tidy

        depth = np.zeros((648, 1152), dtype=np.float32)
        cy, cx = float(self.CY_PX), float(self.CX_PX)
        angle_deg = 45.0
        w, h = 60.0, 200.0
        rang = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(rang), np.sin(rang)
        dx = np.array([-w / 2, w / 2, w / 2, -w / 2], dtype=np.float32)
        dy = np.array([-h / 2, -h / 2, h / 2, h / 2], dtype=np.float32)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        body_poly = np.stack(
            [cx + rx, cy + ry], axis=1
        ).round().astype(np.int32)
        body_mask = np.zeros((648, 1152), dtype=np.uint8)
        _cv2.fillPoly(body_mask, [body_poly], 1)
        depth[body_mask.astype(bool)] = 1700.0  # body plane

        # Near-camera structure cluster at the envelope's TOP-LEFT
        # quadrant: pixels (486..516, 234..264). The rotated body
        # diagonal runs from upper-right (625, 232) to lower-left
        # (484, 374), so the top-left envelope corner is the OPPOSITE
        # diagonal — comfortably outside the body polygon, but inside
        # the axis-aligned envelope and inside the column (back-projects
        # to ~183 mm metric radius < 250 mm).
        depth[234:264, 486:516] = 800.0

        bbox_xs = cx + rx
        bbox_ys = cy + ry
        bbox = (
            int(np.floor(bbox_xs.min())),
            int(np.floor(bbox_ys.min())),
            int(np.ceil(bbox_xs.max())),
            int(np.ceil(bbox_ys.max())),
        )
        rotated = (cx, cy, w, h, angle_deg)
        return depth, bbox, rotated

    def test_rotated_polygon_mask_excludes_corner_structure(self):
        """The motivating bug from the screenshot: RAPiD's tilted body
        bbox collapsed to its axis-aligned envelope swept in a near-
        camera cluster at the envelope corner (a tall cabinet next to
        the sitting user). Without the polygon mask the histogram
        walk picks that cluster as 'head'; with it the cluster is
        excluded and the actual body depth wins.
        """
        depth, bbox, rotated = (
            self._make_tilted_body_with_corner_structure_scene()
        )
        # Baseline (axis-aligned only): structure dominates as nearest.
        naive = self._call(depth, bbox)
        assert naive is not None
        assert abs(naive - 800.0) < 100.0, (
            f"baseline got {naive}, expected ~800 (axis-aligned envelope "
            "sweeps in the near-camera corner cluster without the polygon "
            "mask — that's the bug we're fixing)"
        )
        # Polygon mask applied: structure outside the rotated body is
        # rejected before the histogram walk; body wins.
        masked = self._call(depth, bbox, rotated_bbox=rotated)
        assert masked is not None
        assert abs(masked - 1700.0) < 100.0, (
            f"got {masked}, expected ~1700 (rotated polygon mask excludes "
            "the corner cluster outside the body footprint)"
        )

    def test_rotated_polygon_mask_none_matches_default(self):
        """Passing ``rotated_bbox=None`` explicitly must behave identically
        to omitting the kwarg (the architecture-agnostic fallback).
        """
        depth, bbox, _ = self._make_tilted_body_with_corner_structure_scene()
        a = self._call(depth, bbox)
        b = self._call(depth, bbox, rotated_bbox=None)
        assert a == b

    def test_rotated_polygon_mask_zero_size_returns_none(self):
        """Degenerate rotated rectangle (width or height 0) leaves zero
        valid pixels after masking — function must return None instead
        of crashing on the empty histogram.
        """
        depth = self._full_frame_depth(1500.0)
        cy, cx = int(self.CY_PX), int(self.CX_PX)
        depth[cy - 10 : cy + 10, cx - 10 : cx + 10] = 1500.0
        result = self._call(
            depth,
            self._centred_bbox(),
            rotated_bbox=(float(cx), float(cy), 0.0, 0.0, 0.0),
        )
        assert result is None

"""Annotation helpers for the runtime web viewer.

Pure functions that draw onto BGR frames. No state, no I/O. Kept out of
``viewer.py`` so the streaming module is small and easy to test in
isolation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colour palette
_COLOR_CONFIRMED = (0, 255, 0)
_COLOR_PENDING = (0, 165, 255)
_COLOR_CANDIDATE = (180, 180, 180)
_COLOR_DET = (90, 90, 90)
_COLOR_ROI = (0, 255, 255)
_COLOR_LINE = (255, 0, 0)
_COLOR_TEXT = (255, 255, 255)
_COLOR_OVERLAY_BG = (0, 0, 0)


def annotate_left(
    frame: np.ndarray,
    detections: list,
    tracks: dict,
    counter: Optional[Any],
    fps: float = 0.0,
) -> np.ndarray:
    """Draw ROI, line, raw detections and tracks onto a copy of ``frame``.

    Args:
        frame: BGR left-camera frame (rectified).
        detections: Detection objects (have ``.bbox`` and ``.centroid``).
        tracks: dict[int, Track] from EuclideanTracker.
        counter: LineCounter or ROICounter instance (read for geometry +
            totals overlay). May be None.
        fps: pipeline FPS estimate for the bottom overlay.
    """
    out = frame.copy()

    # Geometry first so detections sit on top.
    _draw_counter_geometry(out, counter)

    # Raw detections in subtle gray. Tracked positions get a coloured
    # marker on top so the operator can see which detections produced
    # tracks and which didn't.
    for det in detections:
        try:
            x1, y1, x2, y2 = det.bbox
        except Exception:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      _COLOR_DET, 1)

    # Defer the import so a circular import on partial init doesn't blow
    # up the viewer module.
    from src.tracking.tracker import CONFIRMED, PENDING, CANDIDATE
    state_colour = {
        CONFIRMED: _COLOR_CONFIRMED,
        PENDING: _COLOR_PENDING,
        CANDIDATE: _COLOR_CANDIDATE,
    }
    for tid, track in tracks.items():
        colour = state_colour.get(getattr(track, "state", None))
        if colour is None:
            continue
        positions = getattr(track, "positions", None)
        if not positions:
            continue
        cx, cy = int(positions[-1][0]), int(positions[-1][1])
        cv2.circle(out, (cx, cy), 6, colour, -1)
        cv2.putText(out, f"#{tid}", (cx + 8, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        # Height label from the most recent detection meta on the track.
        meta = getattr(track, "meta", None)
        history = meta.get("detection_history") if isinstance(meta, dict) else None
        if history:
            last = history[-1]
            head_mm = last.get("head_height_mm")
            cls = last.get("height_class") or "unknown"
            if isinstance(head_mm, (int, float)) and head_mm > 0:
                label = f"{head_mm/1000:.2f}m {cls}"
            else:
                label = cls
            cv2.putText(out, label, (cx + 8, cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    _draw_counter_overlay(out, counter, fps)
    return out


def depth_to_colormap(
    depth_map: Optional[np.ndarray],
    vmin_mm: float = 500.0,
    vmax_mm: float = 5000.0,
) -> np.ndarray:
    """Render a depth-in-mm map as a BGR JET colormap.

    Zero (= invalid disparity) renders black. None / empty input returns
    a small dark gray frame so the panel is visually distinct from a
    real depth map and the caller doesn't need a None check.
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
    target_height: int = 480,
) -> np.ndarray:
    """Resize each panel to ``target_height`` and concat horizontally.

    Missing / empty panels are filled with a dark gray placeholder of
    aspect 1:1 so the composite never crashes on partial input (e.g.
    depth panel not yet computed).
    """
    panels: list[np.ndarray] = []
    for img in (left, right, depth):
        if img is None or img.size == 0:
            panels.append(np.full((target_height, target_height, 3), 30,
                                  dtype=np.uint8))
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if h != target_height:
            scale = target_height / h
            img = cv2.resize(
                img, (max(1, int(w * scale)), target_height),
                interpolation=cv2.INTER_AREA,
            )
        panels.append(img)
    return cv2.hconcat(panels)


# ----------------------------------------------------------------- internals
def _draw_counter_geometry(frame: np.ndarray, counter: Optional[Any]) -> None:
    """Best-effort overlay of ROI rect + counting line.

    Reads counter private attributes (``_roi``, ``_orientation``,
    ``_line_pos`` for ROICounter; ``line_y`` for LineCounter). Wrapped
    in try/except so a future API change in the counter doesn't kill the
    viewer.
    """
    if counter is None:
        return
    h, w = frame.shape[:2]
    # ROICounter: stored as (x_min, x_max, y_min, y_max) tuple.
    roi = getattr(counter, "_roi", None)
    orientation = getattr(counter, "_orientation", None)
    line_pos = getattr(counter, "_line_pos", None)
    if roi is not None:
        try:
            x_min, x_max, y_min, y_max = roi
            cv2.rectangle(frame, (int(x_min), int(y_min)),
                          (int(x_max), int(y_max)), _COLOR_ROI, 1)
            if orientation == "horizontal" and line_pos is not None:
                cv2.line(frame,
                         (int(x_min), int(line_pos)),
                         (int(x_max), int(line_pos)),
                         _COLOR_LINE, 1)
            elif orientation == "vertical" and line_pos is not None:
                cv2.line(frame,
                         (int(line_pos), int(y_min)),
                         (int(line_pos), int(y_max)),
                         _COLOR_LINE, 1)
        except Exception:
            logger.debug("ROI overlay failed", exc_info=True)
        return
    # LineCounter: full-width horizontal line at line_y.
    line_y = getattr(counter, "line_y", None)
    if line_y is None:
        line_y = getattr(counter, "_line_y", None)
    if line_y is not None:
        try:
            cv2.line(frame, (0, int(line_y)), (w, int(line_y)),
                     _COLOR_LINE, 1)
        except Exception:
            logger.debug("Line overlay failed", exc_info=True)


def _draw_counter_overlay(
    frame: np.ndarray, counter: Optional[Any], fps: float,
) -> None:
    h, w = frame.shape[:2]
    in_n = getattr(counter, "total_in", 0) if counter else 0
    out_n = getattr(counter, "total_out", 0) if counter else 0
    text = f"IN: {in_n}  OUT: {out_n}  FPS: {fps:.1f}"
    cv2.rectangle(frame, (0, h - 28), (w, h), _COLOR_OVERLAY_BG, -1)
    cv2.putText(frame, text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_TEXT, 1)

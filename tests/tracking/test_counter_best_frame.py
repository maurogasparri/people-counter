"""Tests de integración para el flow buffer best-frame + evento del counter.

El wiring real vive en ``src/main.py`` (el counter en sí mismo es
intentionally untouched — see scope rules). These tests validate the
contract from the counter's perspective:

  - When the BestFrameManager is None (feature OFF, the default),
    counting must produce identical events to before — no side effects,
    no path attached, no buffer maintenance.
  - When a manager is provided, ``commit`` is called with the count
    event's track_id and timestamp, and the resulting path can be
    spliced into the MQTT payload.

We exercise the same per-frame loop pattern that ``main.py`` uses
(observe each frame, then check counter, then commit on event) but in a
deterministic, hardware-free harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tracking.counter import Counter, Line
from src.tracking.tracker import CONFIRMED, Track
from src.vision.best_frame import BestFrameManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ROI = {"x_min": 100, "x_max": 500, "y_min": 200, "y_max": 400}


def _line_h() -> Line:
    return Line(
        from_xy=(100, 300),
        to_xy=(500, 300),
        labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
    )


def _track(positions: list[list[float]]) -> Track:
    return Track(
        track_id=1,
        positions=[np.array(p, dtype=float) for p in positions],
        state=CONFIRMED,
    )


def _frame() -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, size=(100, 200, 3), dtype=np.uint8)


def _step(counter, track, position):
    track.positions.append(np.array(position, dtype=float))
    return counter._process_track(track)


# ---------------------------------------------------------------------------
# Feature OFF (manager is None) — counter behaviour is identical
# ---------------------------------------------------------------------------


def test_counter_unchanged_when_manager_none():
    """Without a BestFrameManager, the counter must produce events with
    no awareness of the feature — same payload as before this feature
    landed. (The wiring layer in main.py handles the path attachment.)
    """
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _track([[300, 210, 3000]])
    counter._process_track(track)
    _step(counter, track, [300, 280, 3000])
    _step(counter, track, [300, 320, 3000])
    ev = _step(counter, track, [300, 420, 3000])
    assert ev is not None
    assert ev.direction == "ingress"
    # CountEvent has no best_frame_path field — it's added at the
    # publish step in main.py — so the dataclass shape is unaffected.
    assert not hasattr(ev, "best_frame_path")


# ---------------------------------------------------------------------------
# Feature ON — observe + commit produces a valid path attached to event
# ---------------------------------------------------------------------------


def test_observe_then_commit_produces_path(tmp_path: Path):
    """Simulate the per-frame loop: observe each frame; on event,
    commit. The returned path must exist on disk and the buffer must
    drain so a second event for the same track produces no duplicate
    write from a stale buffer.
    """
    mgr = BestFrameManager(tmp_path, buffer_size=8)
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _track([[300, 210, 3000]])

    # Per-frame observe — bbox covers a sensible chunk of the frame.
    bbox = (140, 40, 200, 80)

    def observe_and_step(pos, conf):
        mgr.observe(track.track_id, _frame(), bbox, conf)
        return _step(counter, track, pos)

    counter._process_track(track)
    mgr.observe(track.track_id, _frame(), bbox, 0.4)
    observe_and_step([300, 280, 3000], 0.5)
    observe_and_step([300, 320, 3000], 0.9)  # highest conf
    ev = observe_and_step([300, 420, 3000], 0.6)
    assert ev is not None

    path = mgr.commit(ev.track_id, ev.timestamp)
    assert path is not None
    assert Path(path).exists()
    # Buffer drained — a stale commit with the same track_id is a no-op.
    assert mgr.commit(ev.track_id, ev.timestamp) is None


def test_commit_with_no_observations_returns_none(tmp_path: Path):
    """If a track somehow emits an event without any observed frames
    (e.g. all observes were skipped due to gating), commit must return
    None gracefully. The publish path then omits best_frame_path.
    """
    mgr = BestFrameManager(tmp_path)
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _track([[300, 210, 3000]])
    counter._process_track(track)
    _step(counter, track, [300, 320, 3000])
    ev = _step(counter, track, [300, 420, 3000])
    assert ev is not None
    assert mgr.commit(ev.track_id, ev.timestamp) is None


def test_two_full_cycles_produce_two_paths(tmp_path: Path, monkeypatch):
    """A single track that completes two full ingress/egress cycles must
    produce two distinct JPGs (one per event), without state leaking
    between cycles. Patcheamos U-turn cancellation OFF — el ROI por
    default cancelaría el segundo cycle (ver test_uturn_zone_* en
    test_counter.py). Este test cubre el manager, no la cancelación."""
    monkeypatch.setattr(Counter, "_try_cancel_uturn", lambda *a, **kw: False)
    mgr = BestFrameManager(tmp_path, buffer_size=10)
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _track([[300, 210, 3000]])
    bbox = (140, 40, 200, 80)

    counter._process_track(track)
    mgr.observe(track.track_id, _frame(), bbox, 0.7)
    _step(counter, track, [300, 280, 3000])
    mgr.observe(track.track_id, _frame(), bbox, 0.7)
    _step(counter, track, [300, 320, 3000])
    mgr.observe(track.track_id, _frame(), bbox, 0.7)
    ev1 = _step(counter, track, [300, 420, 3000])
    assert ev1 is not None
    p1 = mgr.commit(ev1.track_id, ev1.timestamp)
    assert p1 is not None

    # Cycle 2 — observe must repopulate the buffer; commit produces a
    # fresh path.
    mgr.observe(track.track_id, _frame(), bbox, 0.8)
    _step(counter, track, [300, 380, 3000])
    mgr.observe(track.track_id, _frame(), bbox, 0.8)
    _step(counter, track, [300, 290, 3000])
    mgr.observe(track.track_id, _frame(), bbox, 0.8)
    ev2 = _step(counter, track, [300, 180, 3000])
    assert ev2 is not None
    p2 = mgr.commit(ev2.track_id, ev2.timestamp + 1.0)
    assert p2 is not None
    assert p1 != p2


# ---------------------------------------------------------------------------
# Privacy invariant — manager never writes when not used
# ---------------------------------------------------------------------------


def test_manager_does_not_write_until_commit(tmp_path: Path):
    """observe() must keep frames in RAM only — nothing on disk until
    commit() fires. Walks the output dir and asserts it stays empty."""
    mgr = BestFrameManager(tmp_path)
    img = _frame()
    for _ in range(20):
        mgr.observe(1, img, (10, 10, 90, 90), 0.5)
    # Output dir was given to the manager but never written to.
    files = list(Path(tmp_path).rglob("*"))
    assert files == []


def test_forget_skips_write_for_aborted_track(tmp_path: Path):
    """A track that disappears without firing an event (operator walks
    in but track dies before crossing) must not produce a JPG when the
    manager is GC'd."""
    mgr = BestFrameManager(tmp_path)
    for _ in range(5):
        mgr.observe(1, _frame(), (10, 10, 90, 90), 0.5)
    mgr.forget(1)
    assert list(Path(tmp_path).rglob("*.jpg")) == []

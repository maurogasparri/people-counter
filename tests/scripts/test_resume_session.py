"""Tests para el sidecar --resume session.json en scripts/calibrate.py."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "calibrate_script", _ROOT / "scripts" / "calibrate.py",
)
calibrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate)


def _args(tmp: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        columns=9, rows=6, square_length=45.0, marker_length=33.0,
        dist_near_mm=1000.0, dist_mid_mm=2000.0, dist_far_mm=3000.0,
        output_dir=str(tmp),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _StateStub:
    def __init__(self, n_poses):
        self.pose_status = ["pending"] * n_poses
        self.captured_pairs = []


class TestSessionRoundtrip:
    def test_save_then_load(self, tmp_path):
        poses = calibrate.default_pose_sequence()
        state = _StateStub(len(poses))
        state.pose_status[0] = "captured"
        lp = tmp_path / "left_000_A1.png"
        rp = tmp_path / "right_000_A1.png"
        lp.write_bytes(b"")
        rp.write_bytes(b"")
        state.captured_pairs.append((lp, rp, "A1"))

        args = _args(tmp_path)
        calibrate._save_session(tmp_path, state, poses, args)

        data = calibrate._load_session(tmp_path, args)
        assert data is not None
        assert len(data["captures"]) == 1
        assert data["captures"][0]["pose_id"] == "A1"
        assert data["pose_status"]["A1"] == "captured"
        assert data["params"]["board_size"] == [9, 6]

    def test_load_returns_none_when_missing(self, tmp_path):
        assert calibrate._load_session(tmp_path, _args(tmp_path)) is None

    def test_load_rejects_mismatched_board(self, tmp_path):
        poses = calibrate.default_pose_sequence()
        state = _StateStub(len(poses))
        args_orig = _args(tmp_path)
        calibrate._save_session(tmp_path, state, poses, args_orig)

        args_diff = _args(tmp_path, square_length=50.0)
        assert calibrate._load_session(tmp_path, args_diff) is None

    def test_load_rejects_mismatched_distance(self, tmp_path):
        poses = calibrate.default_pose_sequence()
        state = _StateStub(len(poses))
        calibrate._save_session(tmp_path, state, poses, _args(tmp_path))

        args_diff = _args(tmp_path, dist_far_mm=3500.0)
        assert calibrate._load_session(tmp_path, args_diff) is None

    def test_load_tolerates_garbage(self, tmp_path):
        (tmp_path / "session.json").write_text("not json at all")
        assert calibrate._load_session(tmp_path, _args(tmp_path)) is None

    def test_save_is_atomic(self, tmp_path):
        """Writing uses a .tmp then replace — no half-written file on disk."""
        poses = calibrate.default_pose_sequence()
        state = _StateStub(len(poses))
        calibrate._save_session(tmp_path, state, poses, _args(tmp_path))
        # After a save the .tmp file should not linger
        assert not (tmp_path / "session.json.tmp").exists()
        # Session file should be parseable
        payload = json.loads((tmp_path / "session.json").read_text())
        assert payload["version"] == calibrate.SESSION_VERSION

    def test_session_tracks_skipped_poses(self, tmp_path):
        poses = calibrate.default_pose_sequence()
        state = _StateStub(len(poses))
        state.pose_status[0] = "captured"
        state.pose_status[3] = "skipped"
        calibrate._save_session(tmp_path, state, poses, _args(tmp_path))

        data = json.loads((tmp_path / "session.json").read_text())
        skipped = [pid for pid, status in data["pose_status"].items()
                   if status == "skipped"]
        assert skipped == [poses[3].id]

"""Smoke tests para src/web/viewer.py."""
from __future__ import annotations

import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from src.web.viewer import WebViewer


def _free_port() -> int:
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_push_before_start_is_noop():
    """push() before start() must not crash and must not raise."""
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # No start() called.
    viewer.push(frame, {"total_in": 0})  # must not raise
    assert viewer.running is False


def test_bind_failure_returns_false():
    """Binding to a port already in use returns False, not raise."""
    a = WebViewer(port=_free_port(), host="127.0.0.1")
    assert a.start() is True
    try:
        b = WebViewer(port=a.port, host="127.0.0.1")
        # SO_REUSEADDR may make this succeed in some envs; either way it
        # must not raise. If it succeeds, we still close it cleanly.
        ok = b.start()
        if ok:
            b.stop()
    finally:
        a.stop()


def test_index_and_stats_endpoints():
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        url = f"http://127.0.0.1:{viewer.port}"
        with urllib.request.urlopen(f"{url}/", timeout=2) as r:
            body = r.read()
            assert r.status == 200
            assert b"<html" in body.lower() or b"<!DOCTYPE" in body
        with urllib.request.urlopen(f"{url}/stats", timeout=2) as r:
            assert r.status == 200
            import json
            payload = json.loads(r.read())
            assert "total_in" in payload
            assert "total_out" in payload
            assert "fps" in payload

        # 404 path
        try:
            urllib.request.urlopen(f"{url}/does-not-exist", timeout=2)
        except urllib.error.HTTPError as e:
            assert e.code == 404
        else:
            pytest.fail("Expected 404 on unknown path")
    finally:
        viewer.stop()


def test_push_updates_stats():
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        viewer.push(frame, {
            "total_in": 5, "total_out": 3, "fps": 12.5,
            "tracks": 2, "dets": 4,
        })
        # Give the encoder a moment.
        time.sleep(0.2)
        url = f"http://127.0.0.1:{viewer.port}/stats"
        import json
        with urllib.request.urlopen(url, timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["total_in"] == 5
        assert payload["total_out"] == 3
        assert payload["fps"] == 12.5
        assert payload["tracks"] == 2
        assert payload["dets"] == 4
    finally:
        viewer.stop()


def test_stop_idempotent():
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    viewer.stop()
    viewer.stop()  # second stop must not raise

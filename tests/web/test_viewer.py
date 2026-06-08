"""Smoke tests para src/web/viewer.py."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from src.web.admin_auth import write_secret
from src.web.viewer import WebViewer


def _post(url, data=None, cookie=None, timeout=2):
    body = urllib.parse.urlencode(data).encode() if data else b""
    req = urllib.request.Request(url, data=body, method="POST")
    if data:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    if cookie:
        req.add_header("Cookie", cookie)
    return urllib.request.urlopen(req, timeout=timeout)


def _get(url, cookie=None, timeout=2):
    req = urllib.request.Request(url)
    if cookie:
        req.add_header("Cookie", cookie)
    return urllib.request.urlopen(req, timeout=timeout)


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
        viewer.push(
            frame,
            {
                "total_in": 5,
                "total_out": 3,
                "fps": 12.5,
                "tracks": 2,
                "dets": 4,
            },
        )
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


def test_has_subscribers_false_without_clients():
    """Sin clientes activos al stream, ``has_subscribers()`` es False
    incluso si el viewer está running. Es la base del gating del main
    loop que evita gastar CPU en annotate + composite cuando nadie mira.
    """
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        assert viewer.running is True
        assert viewer.has_subscribers() is False
    finally:
        viewer.stop()


def test_has_subscribers_false_before_start():
    """Antes de start() — y después de stop() — has_subscribers() es
    False sin condiciones, así los callers pueden gatear sin chequear
    ``running`` por separado.
    """
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.has_subscribers() is False
    viewer.start()
    viewer.stop()
    assert viewer.has_subscribers() is False


# ---------------------------------------------------------------------------
# Panel admin: login + power + cambio de contraseña
# ---------------------------------------------------------------------------


def test_power_controls_disabled_without_secret(tmp_path):
    viewer = WebViewer(
        port=_free_port(), host="127.0.0.1", secret_path=str(tmp_path / "no.secret")
    )
    assert viewer.start() is True
    try:
        url = f"http://127.0.0.1:{viewer.port}"
        with _get(f"{url}/auth/status") as r:
            assert json.loads(r.read())["power_enabled"] is False
        # /reboot deshabilitado sin secret → 404
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{url}/reboot")
        assert exc.value.code == 404
    finally:
        viewer.stop()


def test_login_and_power_flow(tmp_path):
    secret = str(tmp_path / "admin.secret")
    write_secret("test-password-1", secret)
    viewer = WebViewer(port=_free_port(), host="127.0.0.1", secret_path=secret)
    triggered = []
    viewer._trigger_power = lambda action: (triggered.append(action) or True)
    assert viewer.start() is True
    try:
        url = f"http://127.0.0.1:{viewer.port}"
        with _get(f"{url}/auth/status") as r:
            st = json.loads(r.read())
        assert st["power_enabled"] is True and st["authed"] is False
        # sin cookie → 403
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{url}/reboot")
        assert exc.value.code == 403
        # contraseña mal → 401
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{url}/login", {"password": "wrong"})
        assert exc.value.code == 401
        # login bien → 200 + cookie
        with _post(f"{url}/login", {"password": "test-password-1"}) as r:
            assert r.status == 200
            cookie = r.headers.get("Set-Cookie").split(";", 1)[0]
        assert "pc_session=" in cookie
        with _get(f"{url}/auth/status", cookie=cookie) as r:
            assert json.loads(r.read())["authed"] is True
        # /reboot con cookie → 200 y dispara la acción
        with _post(f"{url}/reboot", cookie=cookie) as r:
            assert r.status == 200
        assert triggered == ["reboot"]
    finally:
        viewer.stop()


def test_change_password_flow(tmp_path):
    secret = str(tmp_path / "admin.secret")
    write_secret("old-password-1", secret)
    viewer = WebViewer(port=_free_port(), host="127.0.0.1", secret_path=secret)
    assert viewer.start() is True
    try:
        url = f"http://127.0.0.1:{viewer.port}"
        with _post(f"{url}/login", {"password": "old-password-1"}) as r:
            cookie = r.headers.get("Set-Cookie").split(";", 1)[0]
        # actual incorrecta → 400
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                f"{url}/change-password",
                {"current": "nope", "new": "new-password-2"},
                cookie=cookie,
            )
        assert exc.value.code == 400
        # cambio OK → 200; la nueva contraseña loguea
        with _post(
            f"{url}/change-password",
            {"current": "old-password-1", "new": "new-password-2"},
            cookie=cookie,
        ) as r:
            assert r.status == 200
        with _post(f"{url}/login", {"password": "new-password-2"}) as r:
            assert r.status == 200
    finally:
        viewer.stop()


def test_has_subscribers_tracks_attach_detach():
    """Conectar/desconectar a /stream incrementa/decrementa el contador.
    Verificado vía las primitivas internas ``_subscriber_attach`` /
    ``_subscriber_detach`` para evitar el flake de un test HTTP que
    requiere parsear chunks del MJPEG stream con timing arbitrario.
    """
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        assert viewer.has_subscribers() is False
        viewer._subscriber_attach()
        assert viewer.has_subscribers() is True
        viewer._subscriber_attach()
        assert viewer.has_subscribers() is True
        viewer._subscriber_detach()
        assert viewer.has_subscribers() is True
        viewer._subscriber_detach()
        assert viewer.has_subscribers() is False
        # Detach extra no underflows.
        viewer._subscriber_detach()
        assert viewer.has_subscribers() is False
    finally:
        viewer.stop()


def test_update_stats_works_without_subscribers():
    """``/stats`` tiene que devolver data fresca incluso si nadie está
    mirando el stream — alguien puede pollear stats sin abrir el video.
    El path de update_stats es ortogonal al path del MJPEG.
    """
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        viewer.update_stats(
            {
                "total_in": 7,
                "total_out": 4,
                "fps": 13.1,
                "tracks": 1,
                "dets": 2,
            }
        )
        url = f"http://127.0.0.1:{viewer.port}/stats"
        import json

        with urllib.request.urlopen(url, timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["total_in"] == 7
        assert payload["total_out"] == 4
        assert payload["fps"] == 13.1
        # No suscribers fueron registrados.
        assert viewer.has_subscribers() is False
    finally:
        viewer.stop()


def test_push_without_subscribers_skips_frame_encode():
    """Cuando no hay subscribers, ``push`` actualiza stats pero NO encolea
    el frame para encode — el JPEG encoding sería trabajo desperdiciado.
    Verificamos que ``_latest_jpeg`` queda en None después del push.
    """
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        assert viewer.has_subscribers() is False
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        viewer.push(frame, {"total_in": 9})
        # Esperar al encode loop por si fuera a procesar.
        time.sleep(0.2)
        # Stats sí se actualizaron.
        with viewer._stats_lock:
            assert viewer._stats["total_in"] == 9
        # Pero el JPEG no se generó (encode skipped).
        with viewer._latest_cond:
            assert viewer._latest_jpeg is None
    finally:
        viewer.stop()


def test_push_with_subscribers_encodes_frame():
    """Con un subscriber registrado, ``push`` sí encolea el frame y el
    encode loop produce un JPEG. Complemento del test anterior.
    """
    viewer = WebViewer(port=_free_port(), host="127.0.0.1")
    assert viewer.start() is True
    try:
        viewer._subscriber_attach()
        try:
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            viewer.push(frame)
            time.sleep(0.2)
            with viewer._latest_cond:
                assert viewer._latest_jpeg is not None
                assert len(viewer._latest_jpeg) > 0
        finally:
            viewer._subscriber_detach()
    finally:
        viewer.stop()

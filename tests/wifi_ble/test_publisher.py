"""Tests del bridge WiFi/BLE → MQTT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.wifi_ble.publisher import WifiBlePublisher


@dataclass
class _FakeMQTT:
    """Captura cada publish_event para asserts."""

    sent: list[dict[str, Any]] = field(default_factory=list)

    def publish_event(self, event_type: str, data: dict, qos: int = 1) -> int:
        self.sent.append({"event_type": event_type, "data": data, "qos": qos})
        return len(self.sent)


@dataclass
class _FakeDedup:
    """DedupEngine de juguete con summary pre-seteado."""

    passersby: int = 0
    shoppers: int = 0
    queries: list[dict[str, Any]] = field(default_factory=list)

    def get_window_summary(
        self,
        since_ts: float,
        until_ts: float | None = None,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
    ) -> dict[str, int]:
        self.queries.append(
            {
                "since": since_ts,
                "until": until_ts,
                "rssi_passerby": rssi_passerby,
                "rssi_shopper": rssi_shopper,
            }
        )
        return {"passersby": self.passersby, "shoppers": self.shoppers}


def _clock(values):
    """Stub de time.time() que devuelve `values` uno a uno (último persiste)."""
    state = {"i": 0}

    def _now() -> float:
        i = min(state["i"], len(values) - 1)
        v = values[i]
        state["i"] += 1
        return v

    return _now


def test_no_publish_before_period_elapses():
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(passersby=10, shoppers=2)
    # t=0 init, t=100 query → menos que period_seconds=900.
    pub = WifiBlePublisher(
        mqtt_client=mqtt, dedup=dedup, period_seconds=900.0, now_fn=_clock([0, 100])
    )

    sent = pub.maybe_publish()

    assert sent == 0
    assert mqtt.sent == []
    # Tampoco se consultó a dedup — no hubo motivo.
    assert dedup.queries == []


def test_publishes_summary_when_period_elapsed():
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(passersby=160, shoppers=27)
    pub = WifiBlePublisher(
        mqtt_client=mqtt, dedup=dedup, period_seconds=900.0, now_fn=_clock([0, 900])
    )

    sent = pub.maybe_publish()

    assert sent == 1
    assert len(mqtt.sent) == 1
    payload = mqtt.sent[0]["data"]
    assert payload["passersby"] == 160
    assert payload["shoppers"] == 27
    assert payload["period_start"] == 0
    assert payload["period_end"] == 900
    # Topic logical name correcto.
    assert mqtt.sent[0]["event_type"] == "wifi_ble"


def test_empty_window_does_not_publish_but_advances():
    """passersby=0 → no publica, pero last_period_end avanza."""
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(passersby=0, shoppers=0)
    pub = WifiBlePublisher(
        mqtt_client=mqtt, dedup=dedup, period_seconds=10.0, now_fn=_clock([0, 11])
    )

    sent = pub.maybe_publish()

    assert sent == 0
    assert mqtt.sent == []
    # Importante: avanzó la ventana así la próxima consulta empieza desde acá.
    assert pub.last_period_end == 11


def test_thresholds_propagados_a_dedup():
    """Los RSSI thresholds del config llegan al dedup query."""
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(passersby=5, shoppers=1)
    pub = WifiBlePublisher(
        mqtt_client=mqtt,
        dedup=dedup,
        period_seconds=10.0,
        rssi_passerby=-80.0,
        rssi_shopper=-60.0,
        now_fn=_clock([0, 11]),
    )

    pub.maybe_publish()

    assert dedup.queries[0]["rssi_passerby"] == -80.0
    assert dedup.queries[0]["rssi_shopper"] == -60.0


def test_period_boundaries_are_disjoint():
    """Después de publicar, el próximo `last_period_end` arranca donde terminó."""
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(passersby=1, shoppers=0)
    # t=0 init, t=10 primer tick (publica), t=11 segundo tick demasiado cerca,
    # t=21 ok (publica con period_start=10).
    pub = WifiBlePublisher(
        mqtt_client=mqtt,
        dedup=dedup,
        period_seconds=10.0,
        now_fn=_clock([0, 10, 11, 21]),
    )

    pub.maybe_publish()  # publica @ t=10
    assert pub.last_period_end == 10
    pub.maybe_publish()  # @ t=11 — todavía no pasó la ventana
    assert pub.last_period_end == 10
    pub.maybe_publish()  # @ t=21 — publica con period_start=10
    assert pub.last_period_end == 21

    # La segunda publicación debe tener period_start=10 (sin overlap con la primera).
    second_publish = [m for m in mqtt.sent if m["data"]["period_start"] == 10]
    assert len(second_publish) == 1
    assert second_publish[0]["data"]["period_end"] == 21


def test_dedup_query_failure_advances_window():
    """Si get_window_summary raisea, loguea pero la ventana avanza igual.

    Si no avanzara la ventana, la próxima consulta cubriría un período cada
    vez más largo — eventualmente la query timeout y nunca publicamos nada.
    """

    class _RaisingDedup:
        def get_window_summary(self, **kwargs):
            raise RuntimeError("boom")

    mqtt = _FakeMQTT()
    pub = WifiBlePublisher(
        mqtt_client=mqtt,
        dedup=_RaisingDedup(),
        period_seconds=10.0,
        now_fn=_clock([0, 11]),
    )

    sent = pub.maybe_publish()

    assert sent == 0
    assert mqtt.sent == []
    assert pub.last_period_end == 11


def test_mqtt_publish_failure_advances_window():
    """Si publish_event raisea, loguea pero la ventana avanza igual."""

    class _RaisingMQTT:
        def publish_event(self, *_a, **_kw):
            raise RuntimeError("mqtt down")

    dedup = _FakeDedup(passersby=10, shoppers=2)
    pub = WifiBlePublisher(
        mqtt_client=_RaisingMQTT(),
        dedup=dedup,
        period_seconds=10.0,
        now_fn=_clock([0, 11]),
    )

    sent = pub.maybe_publish()

    assert sent == 0
    # La ventana avanza igual — el outbox de MQTTClient cubre la resiliencia.
    assert pub.last_period_end == 11

"""Tests del bridge WiFi/BLE → MQTT (per-device events, post PR 2)."""

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
    """DedupEngine de juguete con records pre-seteados."""

    records: list[dict[str, Any]] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)

    def get_window_records(
        self,
        since_ts: float,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append({"since": since_ts, "until": until_ts})
        return list(self.records)


def _clock(values):
    """Stub de time.time() que devuelve `values` uno a uno (último persiste)."""
    state = {"i": 0}

    def _now() -> float:
        i = min(state["i"], len(values) - 1)
        v = values[i]
        state["i"] += 1
        return v

    return _now


def _make_record(
    visitor_hash: str = "aa" * 16,
    protocol: str = "wifi",
    rssi_max: int = -55,
    first: float = 0.0,
    last: float = 10.0,
) -> dict:
    return {
        "visitor_hash": visitor_hash,
        "protocol": protocol,
        "rssi_max": rssi_max,
        "first_seen_ts": first,
        "last_seen_ts": last,
    }


def test_no_publish_before_period_elapses():
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(records=[_make_record()])
    # t=0 init, t=100 query → menos que period_seconds=900.
    pub = WifiBlePublisher(
        mqtt_client=mqtt, dedup=dedup, period_seconds=900.0, now_fn=_clock([0, 100])
    )

    sent = pub.maybe_publish()

    assert sent == 0
    assert mqtt.sent == []
    # Tampoco se consultó a dedup — no hubo motivo.
    assert dedup.queries == []


def test_publishes_devices_array_when_period_elapsed():
    mqtt = _FakeMQTT()
    records = [
        _make_record(
            visitor_hash="aa" * 16,
            protocol="wifi",
            rssi_max=-55,
            first=10.0,
            last=120.0,
        ),
        _make_record(
            visitor_hash="bb" * 16, protocol="ble", rssi_max=-70, first=20.0, last=200.0
        ),
    ]
    dedup = _FakeDedup(records=records)
    pub = WifiBlePublisher(
        mqtt_client=mqtt, dedup=dedup, period_seconds=900.0, now_fn=_clock([0, 900])
    )

    sent = pub.maybe_publish()

    assert sent == 1
    assert len(mqtt.sent) == 1
    payload = mqtt.sent[0]["data"]
    assert payload["period_start"] == 0
    assert payload["period_end"] == 900
    assert payload["devices"] == records
    # Topic logical name correcto.
    assert mqtt.sent[0]["event_type"] == "wifi_ble"


def test_empty_window_does_not_publish_but_advances():
    """Sin records → no publica, pero last_period_end avanza al siguiente boundary."""
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(records=[])
    pub = WifiBlePublisher(
        mqtt_client=mqtt, dedup=dedup, period_seconds=10.0, now_fn=_clock([0, 11])
    )

    sent = pub.maybe_publish()

    assert sent == 0
    assert mqtt.sent == []
    # Importante: avanzó al boundary 10 (no a now=11) — el alineamiento al
    # epoch garantiza que la próxima ventana empiece exactamente en t=10.
    assert pub.last_period_end == 10


def test_period_boundaries_are_disjoint():
    """Las ventanas son contiguas y alineadas a múltiplos de period_seconds."""
    mqtt = _FakeMQTT()
    dedup = _FakeDedup(records=[_make_record()])
    # t=0 init (last_period_end=0), t=10 primer tick (publica 0→10),
    # t=11 segundo tick (todavía dentro del bucket 10-20),
    # t=21 ok (publica con period 10→20).
    pub = WifiBlePublisher(
        mqtt_client=mqtt,
        dedup=dedup,
        period_seconds=10.0,
        now_fn=_clock([0, 10, 11, 21]),
    )

    pub.maybe_publish()  # publica @ t=10 con period 0→10
    assert pub.last_period_end == 10
    pub.maybe_publish()  # @ t=11 — antes del siguiente boundary (20)
    assert pub.last_period_end == 10
    pub.maybe_publish()  # @ t=21 — publica 10→20, last_period_end pasa a 20
    assert pub.last_period_end == 20

    # La segunda publicación tiene period 10→20 (alineado al boundary, no a now=21).
    second_publish = [m for m in mqtt.sent if m["data"]["period_start"] == 10]
    assert len(second_publish) == 1
    assert second_publish[0]["data"]["period_end"] == 20


def test_dedup_query_failure_does_not_advance_window_immediately():
    """Un fallo transitorio de get_window_records NO debe perder la ventana.

    A diferencia del counting (que tiene outbox), un fallo acá perdería ~15min
    de tráfico irrecuperable. No avanzamos al primer fallo: reintentamos la
    misma ventana en el próximo tick. Recién tras ``max_query_failures`` fallos
    consecutivos la damos por perdida y avanzamos (para no quedar pegados ni
    crecer la ventana sin límite si el DB está realmente roto).
    """

    class _RaisingDedup:
        def get_window_records(self, **kwargs):
            raise RuntimeError("boom")

    mqtt = _FakeMQTT()
    pub = WifiBlePublisher(
        mqtt_client=mqtt,
        dedup=_RaisingDedup(),
        period_seconds=10.0,
        now_fn=_clock([0, 11]),
        max_query_failures=3,
    )

    # Primeros 2 fallos: la ventana NO avanza (retry de la misma ventana).
    for _ in range(2):
        assert pub.maybe_publish() == 0
        assert pub.last_period_end == 0
    # Tercer fallo consecutivo: se da por perdida y se avanza.
    assert pub.maybe_publish() == 0
    assert pub.last_period_end == 10
    assert mqtt.sent == []


def test_dedup_query_recovers_after_transient_failure():
    """Fallo transitorio seguido de éxito publica la MISMA ventana (no perdida)."""

    calls = {"n": 0}

    class _FlakyDedup:
        def get_window_records(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient lock")
            return [
                {
                    "visitor_hash": "abc",
                    "protocol": "wifi",
                    "rssi_max": -50,
                    "first_seen_ts": 1,
                    "last_seen_ts": 2,
                }
            ]

    mqtt = _FakeMQTT()
    pub = WifiBlePublisher(
        mqtt_client=mqtt,
        dedup=_FlakyDedup(),
        period_seconds=10.0,
        now_fn=_clock([0, 11]),
    )

    # Fallo transitorio: no avanza.
    assert pub.maybe_publish() == 0
    assert pub.last_period_end == 0
    # Retry exitoso: publica la ventana original [0, 10).
    assert pub.maybe_publish() == 1
    assert pub.last_period_end == 10
    assert len(mqtt.sent) == 1
    assert mqtt.sent[0]["data"]["period_start"] == 0
    assert mqtt.sent[0]["data"]["period_end"] == 10


def test_mqtt_publish_failure_advances_window():
    """Si publish_event raisea, loguea pero la ventana avanza igual."""

    class _RaisingMQTT:
        def publish_event(self, *_a, **_kw):
            raise RuntimeError("mqtt down")

    dedup = _FakeDedup(records=[_make_record()])
    pub = WifiBlePublisher(
        mqtt_client=_RaisingMQTT(),
        dedup=dedup,
        period_seconds=10.0,
        now_fn=_clock([0, 11]),
    )

    sent = pub.maybe_publish()

    assert sent == 0
    # La ventana avanza igual — el outbox de MQTTClient cubre la resiliencia.
    assert pub.last_period_end == 10

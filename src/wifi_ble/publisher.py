"""Bridge entre DedupEngine y MQTT.

Cada ``probe_interval_seconds`` (default 900s = 15 min) el publisher consulta
los agregados de la ventana cerrada y publica un único payload reducido al
topic ``wifi_ble``. Los hashes nunca salen del device — solo counts.

Payload publicado:

    {
        "device_id": ...,
        "timestamp": ...,
        "type": "wifi_ble",
        "data": {
            "period_start": <epoch>,
            "period_end":   <epoch>,
            "passersby":    160,        # RSSI >= rssi_passerby_threshold
            "shoppers":      27         # RSSI >= rssi_shopper_threshold
        }
    }

``store_id`` lo infiere la Lambda persist_event desde el ``device_id``.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class _MQTTPublisher(Protocol):
    """Subset de :class:`MQTTClient` que necesita el publisher.

    Mantenido como Protocol para que los tests puedan inyectar un fake
    sin importar paho.
    """

    def publish_event(
        self,
        event_type: str,
        data: dict,
        qos: int = ...,
    ) -> int | None: ...


class _DedupSummary(Protocol):
    """Subset de :class:`DedupEngine` que consume el publisher."""

    def get_window_summary(
        self,
        since_ts: float,
        until_ts: float | None = ...,
        rssi_passerby: float = ...,
        rssi_shopper: float = ...,
    ) -> dict[str, int]: ...


class WifiBlePublisher:
    """Publica resúmenes WiFi/BLE periódicos al cloud.

    Se invoca ``maybe_publish()`` desde el main loop. El publisher decide
    por sí solo si tocaba publicar — no hace falta scheduling externo.

    Notes:
        - Cuando la ventana no produjo detecciones (passersby == 0), no se
          publica nada — evita ruido en la BD.
        - WiFi y BLE van en el MISMO mensaje, post-L2 dedup local. Un
          dispositivo detectado por ambos cuenta como 1 visitante.
        - Las ventanas son disjuntas: ``last_period_end`` marca el inicio
          de la siguiente.
    """

    def __init__(
        self,
        mqtt_client: _MQTTPublisher,
        dedup: _DedupSummary,
        period_seconds: float = 900.0,
        rssi_passerby: float = -75.0,
        rssi_shopper: float = -55.0,
        now_fn=time.time,
    ) -> None:
        self._mqtt = mqtt_client
        self._dedup = dedup
        self._period = float(period_seconds)
        self._rssi_passerby = float(rssi_passerby)
        self._rssi_shopper = float(rssi_shopper)
        self._now = now_fn
        self._last_period_end: float = self._now()

    @property
    def last_period_end(self) -> float:
        return self._last_period_end

    def maybe_publish(self) -> int:
        """Si ya pasó una ventana completa desde el último publish, emite.

        Returns:
            1 si se publicó, 0 si todavía no tocaba o si la ventana fue vacía.
        """
        now = self._now()
        if now - self._last_period_end < self._period:
            return 0

        period_start = self._last_period_end
        period_end = now

        try:
            summary = self._dedup.get_window_summary(
                since_ts=period_start,
                until_ts=period_end,
                rssi_passerby=self._rssi_passerby,
                rssi_shopper=self._rssi_shopper,
            )
        except Exception:
            logger.exception(
                "wifi_ble_publisher_dedup_query_failed",
                extra={"period_start": period_start, "period_end": period_end},
            )
            # Igual avanzamos la ventana — sino quedamos pegados en un período
            # roto y la próxima query incluiría una ventana cada vez más larga.
            self._last_period_end = period_end
            return 0

        if summary["passersby"] == 0:
            logger.debug(
                "wifi_ble_publisher_empty_window",
                extra={
                    "period_start": period_start,
                    "period_end": period_end,
                },
            )
            self._last_period_end = period_end
            return 0

        try:
            self._mqtt.publish_event(
                "wifi_ble",
                {
                    "period_start": int(period_start),
                    "period_end": int(period_end),
                    "passersby": summary["passersby"],
                    "shoppers": summary["shoppers"],
                },
            )
            logger.info(
                "wifi_ble_summary_published",
                extra={
                    "passersby": summary["passersby"],
                    "shoppers": summary["shoppers"],
                    "period_seconds": int(period_end - period_start),
                },
            )
            self._last_period_end = period_end
            return 1
        except Exception:
            logger.exception(
                "wifi_ble_publisher_mqtt_publish_failed",
                extra={"passersby": summary["passersby"]},
            )
            # MQTT failure: avanzamos la ventana igual. El MQTTClient tiene
            # outbox SQLite local que cubre la resiliencia — este publisher no
            # debería intentar retransmitir.
            self._last_period_end = period_end
            return 0

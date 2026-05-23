"""Bridge entre DedupEngine y MQTT.

Cada ``wifi_ble.summary_interval_seconds`` (default 900s = 15 min, acotado a
[30, 900]) el publisher consulta los agregados de la ventana cerrada y publica
un único payload reducido al topic ``wifi_ble``. Los hashes nunca salen del
device — solo counts.

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
        # Alineamos las ventanas a múltiplos de ``period_seconds`` desde el
        # epoch (Unix 0). Con period=900s (15min) los boundaries quedan
        # exactamente en :00, :15, :30, :45 UTC. Esto sincroniza el
        # period_bucket del wifi_ble con el event_bucket de counting_events
        # en Postgres, así turn_in_rate y conversion_rate joinean limpio.
        #
        # El primer ``maybe_publish`` emite el período del boundary que
        # antecede al arranque hasta el siguiente boundary. Es parcial
        # (solo tiene datos desde el arranque, no del boundary completo),
        # pero etiqueta una ventana completa de 15min — aceptable trade-off
        # contra perder los primeros 1-15min de data por skip.
        self._last_period_end: float = (
            self._now() // self._period
        ) * self._period

    @property
    def last_period_end(self) -> float:
        return self._last_period_end

    def maybe_publish(self) -> int:
        """Si ya pasó el siguiente boundary, emite la ventana cerrada.

        Las ventanas son ``[N*period, (N+1)*period)`` desde el epoch — no
        relativas al arranque. Cuando ``now`` cruzó ``last_period_end +
        period``, la ventana ``[last_period_end, last_period_end + period)``
        ya está cerrada y se emite con esos límites exactos (no ``now``).

        Si el pipeline se atrasa y cruzó varios boundaries en una sola call,
        emitimos solo el siguiente; el resto se cubre en las próximas calls
        (el loop principal llama esto cada frame, catch-up es rápido).

        Returns:
            1 si se publicó, 0 si todavía no tocaba o si la ventana fue vacía.
        """
        now = self._now()
        next_boundary = self._last_period_end + self._period
        if now < next_boundary:
            return 0

        period_start = self._last_period_end
        period_end = next_boundary

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
            # El device manda period_start (cuándo arrancó la ventana del
            # summary, alineado al múltiplo de _period) + last_seen_ts
            # (última detección real dentro de la ventana, informativa).
            # NO manda bucket_15min — eso lo deriva Postgres server-side
            # vía GENERATED ALWAYS AS (date_bin de period_start) STORED.
            # Desacopla el device del bucket size del schema (migrar a
            # bucket de 5min en RDS = ALTER TABLE sin tocar device).
            payload = {
                "period_start": int(period_start),
                "period_end": int(period_end),
                "passersby": summary["passersby"],
                "shoppers": summary["shoppers"],
            }
            last_seen = summary.get("last_seen_ts")
            if last_seen is not None:
                payload["last_seen_ts"] = float(last_seen)
            self._mqtt.publish_event("wifi_ble", payload)
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

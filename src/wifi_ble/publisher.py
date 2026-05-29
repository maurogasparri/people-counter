"""Bridge entre DedupEngine y MQTT — per-device events.

Cada ``wifi_ble.summary_interval_seconds`` (default 900s = 15 min) el publisher
consulta los registros per-device de la ventana cerrada y publica un único
payload con un array ``devices`` al topic ``wifi_ble``.

Payload publicado:

    {
        "device_id":  ...,
        "timestamp":  ...,
        "type":       "wifi_ble",
        "data": {
            "period_start": <epoch>,
            "period_end":   <epoch>,
            "devices": [
                {
                    "visitor_hash":  "<32 hex chars>",  # group_id post-stitching
                    "protocol":      "wifi" | "ble",
                    "rssi_max":      -53,
                    "first_seen_ts": <epoch float>,
                    "last_seen_ts":  <epoch float>,
                },
                ...
            ]
        }
    }

La categorización passerby/shopper la aplica server-side la función SQL
``rssi_class(rssi_max)`` — el device solo emite el RSSI crudo. Privacy:
``visitor_hash`` es el ``group_id`` opaco post-stitching local (16 bytes,
derivados de SHA-256 + salt diaria local). MACs crudas nunca salen del Pi.

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


class _DedupRecords(Protocol):
    """Subset de :class:`DedupEngine` que consume el publisher."""

    def get_window_records(
        self,
        since_ts: float,
        until_ts: float | None = ...,
    ) -> list[dict]: ...


class WifiBlePublisher:
    """Publica eventos per-device WiFi/BLE periódicos al cloud.

    Se invoca ``maybe_publish()`` desde el main loop. El publisher decide
    por sí solo si tocaba publicar — no hace falta scheduling externo.

    Notes:
        - Cuando la ventana no produjo detecciones, no se publica nada.
        - WiFi y BLE van en el MISMO mensaje, post-stitching local. Un
          dispositivo detectado por ambos cuenta como 1 entry en ``devices``.
        - Las ventanas son disjuntas: ``last_period_end`` marca el inicio
          de la siguiente.
    """

    def __init__(
        self,
        mqtt_client: _MQTTPublisher,
        dedup: _DedupRecords,
        period_seconds: float = 900.0,
        now_fn=time.time,
        max_query_failures: int = 20,
    ) -> None:
        self._mqtt = mqtt_client
        self._dedup = dedup
        self._period = float(period_seconds)
        self._now = now_fn
        # Fallos consecutivos de get_window_records sobre la MISMA ventana.
        # No avanzamos la ventana al primer fallo (un lock transitorio de
        # SQLite haría perder 15min de tráfico sin recuperación, a diferencia
        # del counting que tiene outbox). Reintentamos en los próximos ticks;
        # recién después de ``max_query_failures`` fallos seguidos damos la
        # ventana por perdida y avanzamos, así un DB realmente roto no clava
        # el publisher para siempre ni hace crecer la ventana sin límite.
        self._max_query_failures = int(max_query_failures)
        self._consecutive_query_failures = 0
        # Alineamos las ventanas a múltiplos de ``period_seconds`` desde el
        # epoch (Unix 0). Con period=900s (15min) los boundaries quedan
        # exactamente en :00, :15, :30, :45 UTC. Esto sincroniza el bucket
        # del wifi_ble con el de counting_events en Postgres, así
        # turn_in_rate y conversion_rate joinean limpio.
        #
        # El primer ``maybe_publish`` emite el período del boundary que
        # antecede al arranque hasta el siguiente boundary. Es parcial
        # (solo tiene datos desde el arranque), pero etiqueta una ventana
        # completa — aceptable trade-off contra perder los primeros 1-15min.
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
        emitimos solo el siguiente; el resto se cubre en las próximas calls.

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
            devices = self._dedup.get_window_records(
                since_ts=period_start,
                until_ts=period_end,
            )
        except Exception:
            self._consecutive_query_failures += 1
            give_up = self._consecutive_query_failures >= self._max_query_failures
            logger.exception(
                "wifi_ble_publisher_dedup_query_failed",
                extra={
                    "period_start": period_start,
                    "period_end": period_end,
                    "consecutive_failures": self._consecutive_query_failures,
                    "giving_up": give_up,
                },
            )
            if give_up:
                # DB roto de forma persistente: damos la ventana por perdida
                # y avanzamos para no quedar pegados ni crecer sin límite.
                self._last_period_end = period_end
                self._consecutive_query_failures = 0
            # Si no, NO avanzamos: reintentamos esta misma ventana en el
            # próximo tick (cubre locks transitorios de SQLite).
            return 0

        # Query exitosa — reseteamos el contador de fallos.
        self._consecutive_query_failures = 0

        if not devices:
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
            # El device manda period_start (cuándo arrancó la ventana,
            # alineado al múltiplo de _period) + un array per-device.
            # NO manda bucket_15min — eso lo deriva Postgres server-side
            # vía GENERATED ALWAYS AS STORED desde last_seen_ts.
            payload = {
                "period_start": int(period_start),
                "period_end": int(period_end),
                "devices": devices,
            }
            self._mqtt.publish_event("wifi_ble", payload)
            logger.info(
                "wifi_ble_events_published",
                extra={
                    "n_devices": len(devices),
                    "period_seconds": int(period_end - period_start),
                },
            )
            self._last_period_end = period_end
            return 1
        except Exception:
            logger.exception(
                "wifi_ble_publisher_mqtt_publish_failed",
                extra={"n_devices": len(devices)},
            )
            # MQTT failure: avanzamos la ventana igual. El MQTTClient tiene
            # outbox SQLite local que cubre la resiliencia — este publisher no
            # debería intentar retransmitir.
            self._last_period_end = period_end
            return 0

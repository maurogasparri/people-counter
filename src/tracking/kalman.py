"""Filtro Kalman de velocidad constante para posiciones de tracks 2D.

Vector de estado: ``[cx, cy, vx, vy]^T`` — posición en pixels más velocidad
en pixels por frame. Las mediciones son solo posiciones 2D (el detector ve
posición, no velocidad).

Esta es una implementación pequeña hand-rolled deliberadamente. El repo ya
depende de numpy y no queremos un paquete `filterpy` solo para 30 líneas de
álgebra matricial. Se usa un único paso ``dt = 1`` por frame; el tracker
llama a :meth:`predict` una vez al inicio de cada frame y a :meth:`update`
solo cuando el detector matcheó el track.

Convenciones de estado:

* ``predict()`` avanza ``x`` y crece ``P`` por el process noise. Llamarlo
  dos veces seguidas simula dos frames perdidos — exactamente lo que
  queremos para un track que sigue PENDING pero sin match.
* ``update(z)`` hace la corrección Kalman estándar usando la medición 2D
  ``z = [cx, cy]``.
* ``position`` devuelve la mejor estimación actual ``[cx, cy]`` sin avanzar
  el estado — seguro de llamar repetidamente.

Los defaults se eligen para que el filtro se comporte parecido a la
extrapolación lineal ``last + (last - prev)`` cuando llegan mediciones cada
frame, pero con un crecimiento elegante de incertidumbre a lo largo de los
misses.
"""

from __future__ import annotations

import numpy as np


class TrackKalman:
    """Filtro Kalman 2D de velocidad constante, estado 4D, medición 2D.

    Parameters
    ----------
    initial_position:
        ``[cx, cy]`` de la primera detección. La velocidad se inicializa en 0.
    process_noise:
        Valor de la diagonal de ``Q``. Más alto = menos confianza en el
        modelo, dejar que la velocidad flexione más entre frames.
    measurement_noise:
        Valor de la diagonal de ``R``. El ruido del detector a 1-sigma en pixels.
    initial_velocity_uncertainty:
        Valor de la diagonal de ``P`` para los componentes de velocidad al
        momento del init. Alto por default porque un track nuevo todavía
        no tiene dirección observada — la primera medición lo va a colapsar
        rápido.
    """

    __slots__ = ("x", "P", "F", "H", "Q", "R", "_I")

    def __init__(
        self,
        initial_position: np.ndarray,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
        initial_velocity_uncertainty: float = 100.0,
    ) -> None:
        cx, cy = float(initial_position[0]), float(initial_position[1])
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)

        # Covarianza inicial: la incertidumbre de posición matchea el
        # measurement noise (lo acabamos de observar), la incertidumbre de
        # velocidad es alta.
        self.P = np.diag(
            [
                measurement_noise,
                measurement_noise,
                initial_velocity_uncertainty,
                initial_velocity_uncertainty,
            ]
        ).astype(float)

        # Transición de velocidad constante con dt=1.
        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Solo medimos posición.
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        # Process / measurement noise diagonales.
        self.Q = np.eye(4) * float(process_noise)
        self.R = np.eye(2) * float(measurement_noise)
        self._I = np.eye(4)

    # ------------------------------------------------------------------ API
    @property
    def position(self) -> np.ndarray:
        """Devuelve la estimación 2D actual de posición ``[cx, cy]``.

        Idempotente — no avanza el estado.
        """
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Devuelve la estimación 2D actual de velocidad ``[vx, vy]``."""
        return self.x[2:].copy()

    def predict(self) -> np.ndarray:
        """Avanza un paso bajo el modelo de velocidad constante.

        Muta ``x`` y crece ``P`` por ``Q``. Devuelve la nueva estimación
        2D de posición.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def peek_next(self) -> np.ndarray:
        """Lookahead 1-step no destructivo.

        Devuelve ``H @ F @ x`` — lo que daría ``predict()``, sin mutar
        ``x`` ni ``P``. Útil como getter externo para "¿dónde esperamos
        a este track el próximo frame?" sin acoplar al caller con el
        ciclo de predict per-frame.
        """
        return (self.F @ self.x)[:2]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Aplica una corrección por medición.

        Parameters
        ----------
        measurement:
            Observación 2D ``[cx, cy]``.

        Returns
        -------
        La estimación 2D de posición post-update.
        """
        z = np.asarray(measurement, dtype=float).reshape(2)
        # Innovation
        y = z - self.H @ self.x
        # Covarianza de innovation
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update del estado
        self.x = self.x + K @ y
        # Update de covarianza (la forma de Joseph sería más estable
        # numéricamente pero la forma estándar alcanza para este
        # problema 4x4 y matchea las referencias canónicas).
        self.P = (self._I - K @ self.H) @ self.P
        return self.x[:2].copy()

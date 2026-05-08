"""Constant-velocity Kalman filter for 2D track positions.

State vector: ``[cx, cy, vx, vy]^T`` — pixel position plus pixel velocity
per frame. Measurements are 2D positions only (the detector sees position,
not velocity).

This is a deliberately small hand-rolled implementation. The repo already
depends on numpy and we don't want a `filterpy` package just to get 30
lines of matrix algebra. A single ``dt = 1`` step is used per frame; the
tracker calls :meth:`predict` once at the start of each frame and
:meth:`update` only when the detector matched the track.

State conventions:

* ``predict()`` advances ``x`` and grows ``P`` by the process noise. Calling
  it twice in a row simulates two missed frames — exactly what we want
  for a track that's still PENDING but unmatched.
* ``update(z)`` does the standard Kalman correction using the 2D
  measurement ``z = [cx, cy]``.
* ``position`` returns the current best estimate ``[cx, cy]`` without
  advancing state — safe to call repeatedly.

Defaults are chosen so the filter behaves close to ``last + (last - prev)``
linear extrapolation when measurements arrive every frame, but with
graceful uncertainty growth across misses.
"""

from __future__ import annotations

import numpy as np


class TrackKalman:
    """Constant-velocity 2D Kalman filter, 4D state, 2D measurement.

    Parameters
    ----------
    initial_position:
        ``[cx, cy]`` of the first detection. Velocity is initialised to 0.
    process_noise:
        Diagonal value of ``Q``. Higher = trust the model less, let
        velocity flex more between frames.
    measurement_noise:
        Diagonal value of ``R``. The 1-sigma detector noise in pixels.
    initial_velocity_uncertainty:
        Diagonal value of ``P`` for the velocity components at init time.
        High by default because a new track has no observed direction
        yet — the first measurement update will collapse it quickly.
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

        # Initial covariance: position uncertainty matches measurement
        # noise (we just observed it), velocity uncertainty is high.
        self.P = np.diag(
            [
                measurement_noise,
                measurement_noise,
                initial_velocity_uncertainty,
                initial_velocity_uncertainty,
            ]
        ).astype(float)

        # Constant-velocity transition with dt=1.
        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # We measure position only.
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        # Diagonal process / measurement noise.
        self.Q = np.eye(4) * float(process_noise)
        self.R = np.eye(2) * float(measurement_noise)
        self._I = np.eye(4)

    # ------------------------------------------------------------------ API
    @property
    def position(self) -> np.ndarray:
        """Return the current 2D position estimate ``[cx, cy]``.

        Idempotent — does not advance the state.
        """
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Return the current 2D velocity estimate ``[vx, vy]``."""
        return self.x[2:].copy()

    def predict(self) -> np.ndarray:
        """Advance one step under the constant-velocity model.

        Mutates ``x`` and grows ``P`` by ``Q``. Returns the new 2D
        position estimate.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def peek_next(self) -> np.ndarray:
        """Non-destructive 1-step lookahead.

        Returns ``H @ F @ x`` — what ``predict()`` would yield, without
        mutating ``x`` or ``P``. Useful as an external getter for
        "where do we expect this track next frame?" without coupling
        the caller to the per-frame predict cycle.
        """
        return (self.F @ self.x)[:2]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Apply a measurement correction.

        Parameters
        ----------
        measurement:
            2D ``[cx, cy]`` observation.

        Returns
        -------
        The post-update 2D position estimate.
        """
        z = np.asarray(measurement, dtype=float).reshape(2)
        # Innovation
        y = z - self.H @ self.x
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + K @ y
        # Covariance update (Joseph form would be more numerically stable
        # but the standard form is fine for this 4x4 problem and matches
        # canonical references).
        self.P = (self._I - K @ self.H) @ self.P
        return self.x[:2].copy()

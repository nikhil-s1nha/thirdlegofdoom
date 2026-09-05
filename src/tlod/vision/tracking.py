"""Temporal tracking and motion prediction.

This module is the answer to the central problem of the whole robot.

Measured end to end, the loop from a hand moving to the arm moving is
roughly 250-450 ms: camera pipeline, ~20 ms of landmark inference, then
100-300 ms of servo travel. A human's visual reaction time is about
200-250 ms. A robot that steers toward where it last *saw* a hand is
therefore permanently a quarter of a second behind, and will lose every
race it enters.

The fix is not to make the pipeline faster -- there is a floor, and we are
near it. The fix is to stop aiming at the past. A Kalman filter estimates
where the hand is *and how fast it is going*, so the controller can aim at
where it will be when the arm arrives. Prediction converts a latency
problem into an accuracy problem, and accuracy is something we can trade
against, because a slap does not need millimetres.

Prediction horizon should be set to the measured total loop latency; see
`tlod bench loop`, which reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ConstantVelocityFilter:
    """6-state Kalman filter: position and velocity in 3D.

    Constant velocity, not constant acceleration. A hand in a slap game
    accelerates hard and briefly, and a CA model responds to that by
    over-extrapolating wildly the moment the acceleration stops -- which
    is exactly at the interesting moment. CV with generous process noise
    is more honest about what it does not know and degrades better.
    """

    def __init__(
        self,
        position: np.ndarray,
        stamp: float,
        process_noise: float = 4.0,      # accel stddev, m/s^2
        measurement_noise: float = 0.012,  # position stddev, metres
        initial_velocity_var: float = 1.0,
    ) -> None:
        self.x = np.zeros(6)
        self.x[:3] = np.asarray(position, float)
        self.P = np.eye(6)
        self.P[:3, :3] *= measurement_noise**2
        self.P[3:, 3:] *= initial_velocity_var
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.stamp = float(stamp)

    def predict_to(self, stamp: float) -> None:
        dt = stamp - self.stamp
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt

        # Piecewise-constant white acceleration noise.
        q = self.q**2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt**4 / 4) * q
        Q[:3, 3:] = np.eye(3) * (dt**3 / 2) * q
        Q[3:, :3] = np.eye(3) * (dt**3 / 2) * q
        Q[3:, 3:] = np.eye(3) * (dt**2) * q

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.stamp = stamp

    def update(self, position: np.ndarray, stamp: float) -> None:
        self.predict_to(stamp)
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = np.eye(3) * self.r**2
        y = np.asarray(position, float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[3:]))

    def predict(self, horizon: float) -> np.ndarray:
        """Where the target will be `horizon` seconds after its last update."""
        return self.x[:3] + self.x[3:] * horizon

    def position_uncertainty(self, horizon: float = 0.0) -> float:
        """1-sigma position uncertainty, metres, optionally extrapolated.

        Worth surfacing to the game layer: when this grows past a few
        centimetres the prediction is not trustworthy and the right move is
        to hold rather than swing at a guess.
        """
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * horizon
        P = F @ self.P @ F.T
        return float(np.sqrt(np.trace(P[:3, :3]) / 3.0))


@dataclass(slots=True)
class Track:
    id: int
    filter: ConstantVelocityFilter
    label: str = "hand"
    hits: int = 1
    misses: int = 0
    first_seen: float = 0.0

    @property
    def confirmed(self) -> bool:
        """Seen enough times to be believed. Stops a single false detection
        from making the arm lunge."""
        return self.hits >= 3

    @property
    def age(self) -> float:
        return self.filter.stamp - self.first_seen


class MultiTracker:
    """Greedy nearest-neighbour association over a small number of targets.

    Greedy rather than Hungarian: this robot tracks at most a few hands on
    a table, where targets are far apart relative to their per-frame
    motion, and the optimal assignment is almost always the greedy one. Not
    worth the complexity until proven otherwise.
    """

    def __init__(
        self,
        max_distance: float = 0.18,   # metres, gate for association
        max_misses: int = 8,
        process_noise: float = 4.0,
        measurement_noise: float = 0.012,
    ) -> None:
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: list[np.ndarray], stamp: float, label: str = "hand") -> list[Track]:
        for t in self.tracks:
            t.filter.predict_to(stamp)

        unmatched = list(range(len(detections)))
        pairs: list[tuple[int, int]] = []

        if self.tracks and unmatched:
            cost = np.full((len(self.tracks), len(detections)), np.inf)
            for i, t in enumerate(self.tracks):
                for j in unmatched:
                    d = float(np.linalg.norm(t.filter.position - detections[j]))
                    if d <= self.max_distance:
                        cost[i, j] = d
            while True:
                i, j = np.unravel_index(np.argmin(cost), cost.shape)
                if not np.isfinite(cost[i, j]):
                    break
                pairs.append((int(i), int(j)))
                cost[i, :] = np.inf
                cost[:, j] = np.inf

        matched_dets = {j for _, j in pairs}
        matched_tracks = {i for i, _ in pairs}

        for i, j in pairs:
            t = self.tracks[i]
            t.filter.update(detections[j], stamp)
            t.hits += 1
            t.misses = 0

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.misses += 1

        for j, d in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(
                    Track(
                        id=self._next_id,
                        filter=ConstantVelocityFilter(
                            d, stamp, self.process_noise, self.measurement_noise
                        ),
                        label=label,
                        first_seen=stamp,
                    )
                )
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def best(self) -> Track | None:
        """The most established confirmed track. What a single-target game
        should aim at."""
        confirmed = [t for t in self.tracks if t.confirmed]
        if not confirmed:
            return None
        return max(confirmed, key=lambda t: t.hits)

    def reset(self) -> None:
        self.tracks.clear()

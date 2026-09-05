"""Simulated arm.

This is not a stub that echoes back whatever you write. It models the
first-order tracking behaviour of a position-controlled servo with a finite
slew rate, because the whole point of the simulator is to answer questions
like "can the arm get there before the hand moves?" -- and a backend that
teleports gives the reassuring, useless answer "always".

The default `max_speed` is taken from the STS3215 at 12 V under the
SO-101's 1:345 gearing. Treat it as an optimistic estimate until measured
on real hardware with `tlod bench servo`; it exists so that timing
conclusions drawn in simulation are in the right order of magnitude, not
so they can be quoted as fact.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from tlod.arm.backend import ArmBackend
from tlod.arm.model import GRIPPER_LIMITS, JOINT_LIMITS
from tlod.types import NUM_JOINTS, JointState


class MockArm(ArmBackend):
    def __init__(
        self,
        q0: np.ndarray | None = None,
        max_speed: float = 3.5,       # rad/s at the output shaft
        accel: float = 25.0,          # rad/s^2
        latency: float = 0.004,       # command -> servo begins moving, seconds
        noise: float = 0.0,           # encoder noise stddev, radians
    ) -> None:
        self._q = np.zeros(NUM_JOINTS) if q0 is None else np.asarray(q0, float).copy()
        self._dq = np.zeros(NUM_JOINTS)
        self._goal = self._q.copy()
        self.max_speed = max_speed
        self.accel = accel
        self.latency = latency
        self.noise = noise
        self._torque = True
        self._torque_limit = 800
        self._connected = False
        self._t = time.perf_counter()
        self._pending: list[tuple[float, np.ndarray]] = []
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        self._connected = True
        self._t = time.perf_counter()

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def set_torque(self, enabled: bool) -> None:
        self._torque = enabled

    def set_torque_limit(self, value: int) -> None:
        self._torque_limit = int(value)

    # -- simulation --------------------------------------------------------
    def _integrate(self, now: float) -> None:
        """Advance the sim to `now`. Called lazily on read/write so the
        simulator costs nothing when idle and stays in step with wall clock,
        which is what makes latency measurements in sim comparable to real.

        The whole body holds the lock. It is called from every thread that
        touches the arm -- the control loop through write(), and the
        perception loop and viewer through read() -- and it is not
        reentrant: two threads both read `_t`, both compute the same dt,
        and both integrate, so the simulated arm advances twice per tick
        and moves faster than its configured slew rate. That failure is
        invisible except as timing conclusions that are quietly wrong,
        and it only appears when something else is watching, which is the
        worst possible property for a simulator to have.
        """
        with self._lock:
            self._integrate_locked(now)

    def _integrate_locked(self, now: float) -> None:
        dt = now - self._t
        if dt <= 0:
            return
        self._t = now

        while self._pending and self._pending[0][0] <= now:
            _, goal = self._pending.pop(0)
            self._goal = goal

        if not self._torque:
            self._dq[:] = 0.0
            return

        # Trapezoidal approach: accelerate toward the goal, but never
        # command a speed that cannot be braked before arrival.
        err = self._goal - self._q
        v_max = np.minimum(self.max_speed, np.sqrt(2.0 * self.accel * np.abs(err) + 1e-12))
        v_target = np.sign(err) * v_max
        dv = np.clip(v_target - self._dq, -self.accel * dt, self.accel * dt)
        self._dq += dv
        step = self._dq * dt
        # Do not overshoot the goal within a tick.
        step = np.where(np.abs(step) > np.abs(err), err, step)
        self._q += step

        lim = np.vstack([JOINT_LIMITS, np.array([GRIPPER_LIMITS])])
        self._q = np.clip(self._q, lim[:, 0], lim[:, 1])

    def read(self) -> JointState:
        now = time.perf_counter()
        with self._lock:
            self._integrate_locked(now)
            q = self._q.copy()
            dq = self._dq.copy()
        if self.noise:
            q = q + np.random.normal(0.0, self.noise, NUM_JOINTS)
        return JointState(q=q, stamp=now, dq=dq)

    def write(self, q: np.ndarray) -> None:
        now = time.perf_counter()
        goal = np.asarray(q, float).copy()
        lim = np.vstack([JOINT_LIMITS, np.array([GRIPPER_LIMITS])])
        goal = np.clip(goal, lim[:, 0], lim[:, 1])
        with self._lock:
            self._integrate_locked(now)
            self._pending.append((now + self.latency, goal))

    def diagnostics(self) -> dict[str, object]:
        return {"sim": True, "torque": self._torque, "torque_limit": self._torque_limit,
                "speed": float(np.abs(self._dq).max())}

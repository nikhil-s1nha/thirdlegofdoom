"""Motion profiling: how a commanded setpoint is allowed to change.

Why this exists
---------------
The controller used to shape commands with a single first-order rate cap::

    step_cap = max_speed * dt
    cmd = prev + clip(target - prev, -step_cap, step_cap)

That bounds velocity and nothing else. The moment a new target appears the
commanded velocity goes from 0 to `max_speed` inside one tick and back to 0
just as abruptly on arrival, which is an unbounded acceleration and an
unbounded jerk in the setpoint stream. A position-controlled servo answers a
step in position error with a step in PWM duty, and a step in duty is a
current spike. Six of those at once is a brownout on a marginal supply.

Min-jerk time scaling (`controller.minimum_jerk`) already smooths the
*planned* primitives, but it only covers motions that know their duration in
advance. `servo_pose`, the streaming path the games actually use, has no
such plan -- and even a min-jerk plan is re-clipped by the rate cap, which
reintroduces the corner it was trying to avoid.

So the shaping belongs here, below every path, applied to the setpoint
stream itself.

What it does
------------
Three limits instead of one, applied per joint:

    |v| <= max_speed        rad/s     as before
    |a| <= max_accel        rad/s^2   bounds motor torque, hence current
    |j| <= max_jerk         rad/s^3   bounds the *rate of change* of current

Acceleration is the one that matters for the power supply. Motor torque is
proportional to current, and for a geared arm torque is dominated by
`I*a + gravity`, so capping `a` caps the draw directly. Jerk matters for a
different reason: it is what stops the current *stepping*, and a supply with
finite output impedance sags on the step, not on the average.

Synchronisation
---------------
The limits are also applied *jointly*, not independently. With per-joint
caps, a move where the shoulder travels 1.0 rad and the wrist 0.1 rad runs
both at full speed until the wrist arrives, so the joint-space path bends
and every joint draws its peak at the same moment. Scaling each joint's
limits by its share of the largest displacement makes them start, peak and
arrive together, which keeps the path straight and means only one joint is
ever at the full limit.

This is standard synchronised point-to-point behaviour in industrial motion
controllers. The equivalent library for the general case is `ruckig` (online
jerk-limited trajectory generation, time-optimal, handles a target that
moves mid-motion). It is a better solver than this one and worth reaching
for if the requirements grow; it is not used here because it is a compiled
dependency on a board where the install is not free, and because a tracking
filter for a setpoint that changes every tick only needs the reachable-state
part of what it offers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProfileLimits:
    """Kinematic bounds on the commanded setpoint stream.

    Defaults are the safe-motion figures, not the strike figures; anything
    that wants to move faster passes its own limits per call, the same way
    `max_speed` was already overridden.

    `max_accel` is the current knob. The relationship is roughly linear:
    halving it halves the dynamic component of motor torque and so halves
    the accelerating current, at the cost of a longer move. `max_jerk` is
    conventionally set to a few times `max_accel`; too low and the arm feels
    mushy and lags, too high and the jerk limit stops doing anything.
    """

    max_speed: float = 2.0        # rad/s
    max_accel: float = 8.0        # rad/s^2
    max_jerk: float = 80.0        # rad/s^3
    synchronise: bool = True

    def scaled(self, factor: float) -> ProfileLimits:
        """Time-scale the limits by `factor` in (0, 1].

        Slowing a trajectory by a factor s in time scales velocity by s,
        acceleration by s^2 and jerk by s^3 -- the classical time-scaling
        result, and the reason derating is so effective against a current
        limit: a 30% slower move draws roughly half the accelerating
        current. Callers pass the *velocity* factor and get the consistent
        set, rather than each site inventing its own scaling.
        """
        f = float(np.clip(factor, 1e-3, 1.0))
        return ProfileLimits(
            max_speed=self.max_speed * f,
            max_accel=self.max_accel * f * f,
            max_jerk=self.max_jerk * f * f * f,
            synchronise=self.synchronise,
        )


class MotionProfile:
    """Jerk-limited, synchronised shaper for a stream of joint setpoints.

    Stateful and incremental: it holds the commanded position, velocity and
    acceleration, and `step` advances them one tick toward whatever target
    is current. That is what makes it usable for both a planned motion and a
    visual-servoing stream -- the target is allowed to move, jump, or
    reverse between ticks and the output stays within the limits regardless.

    The control law is the standard proximate-time-optimal one: aim for the
    fastest velocity from which the joint can still stop on the target, then
    rate-limit the acceleration needed to get there.
    """

    __slots__ = ("n", "limits", "q", "v", "a", "rest_time")

    def __init__(self, q0: np.ndarray, limits: ProfileLimits | None = None) -> None:
        self.q = np.asarray(q0, float).copy()
        self.n = self.q.shape[0]
        self.limits = limits or ProfileLimits()
        self.v = np.zeros(self.n)
        self.a = np.zeros(self.n)
        # How long the setpoint has been stationary. A caller asking "have
        # we arrived?" means the *arm*, not the command, and the arm is
        # always a few milliseconds of servo lag behind the command it was
        # last given. Measuring how long the command has held still is the
        # cheapest honest proxy, and unlike reading the encoder it costs no
        # bus traffic on the control thread.
        self.rest_time = 0.0

    def reset(self, q: np.ndarray) -> None:
        """Re-seed at a known configuration, at rest.

        Used on start and after an e-stop: resuming with stale velocity and
        acceleration state would let the profile continue a motion the
        operator just stopped.
        """
        self.q = np.asarray(q, float).copy()
        self.v[:] = 0.0
        self.a[:] = 0.0
        self.rest_time = 0.0

    @property
    def speed(self) -> float:
        return float(np.abs(self.v).max()) if self.n else 0.0

    @property
    def accel(self) -> float:
        return float(np.abs(self.a).max()) if self.n else 0.0

    def _per_joint_limits(self, err: np.ndarray, limits: ProfileLimits):
        """Split the limits across joints so they arrive together."""
        v_max = np.full(self.n, limits.max_speed)
        a_max = np.full(self.n, limits.max_accel)
        j_max = np.full(self.n, limits.max_jerk)
        if not limits.synchronise:
            return v_max, a_max, j_max

        span = float(np.abs(err).max())
        if span < 1e-9:
            return v_max, a_max, j_max
        # Each joint gets the fraction of full speed that its own travel
        # bears to the longest travel, so all of them finish at once. The
        # floor keeps a joint with a tiny residual from being scaled to a
        # standstill and never converging.
        share = np.clip(np.abs(err) / span, 0.05, 1.0)
        return v_max * share, a_max * share, j_max * share

    def step(
        self,
        target: np.ndarray,
        dt: float,
        limits: ProfileLimits | None = None,
    ) -> np.ndarray:
        """Advance one tick toward `target`. Returns the new setpoint."""
        if dt <= 0.0:
            return self.q.copy()
        lim = limits or self.limits
        target = np.asarray(target, float)
        err = target - self.q
        v_max, a_max, j_max = self._per_joint_limits(err, lim)

        # Velocity the joint will actually be carrying once the current
        # acceleration has been ramped out at the jerk limit. Braking has to
        # be judged against that, not against the present velocity: the
        # acceleration cannot be dropped instantly, so some further speed is
        # already committed. Leaving this term out is what makes a naive
        # jerk-limited tracker overshoot and then hunt around the target.
        ramp_out = self.a * np.abs(self.a) / (2.0 * np.maximum(j_max, 1e-9))
        v_effective = self.v + ramp_out

        # Fastest velocity from which the remaining distance is still enough
        # to stop in, under the acceleration limit.
        v_stop = np.sign(err) * np.minimum(v_max, np.sqrt(2.0 * a_max * np.abs(err)))

        a_desired = np.clip((v_stop - v_effective) / dt, -a_max, a_max)
        self.a = np.clip(a_desired, self.a - j_max * dt, self.a + j_max * dt)
        self.v = np.clip(self.v + self.a * dt, -v_max, v_max)
        step = self.v * dt

        # Do not step past the target inside a tick. Without this the
        # discrete integration can jump the setpoint over the goal on the
        # last tick of a move and leave a small permanent oscillation.
        overshooting = np.abs(step) > np.abs(err)
        step = np.where(overshooting, err, step)
        self.v = np.where(overshooting, 0.0, self.v)
        self.a = np.where(overshooting, 0.0, self.a)

        self.q = self.q + step

        moving = float(np.abs(self.v).max()) > 1e-4 or float(np.abs(step).max()) > 1e-7
        self.rest_time = 0.0 if moving else self.rest_time + dt
        return self.q.copy()

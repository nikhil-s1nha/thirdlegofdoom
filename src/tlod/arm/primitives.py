"""Motion primitives.

Every primitive is *steppable*, not blocking:

    motion.start(controller)
    while not motion.step(controller, dt):
        ...

This shape is deliberate. A game is a state machine running on the
control thread at a fixed rate, and it must be able to abandon a motion
mid-flight -- because the hand moved, because a safety gate tripped,
because the round ended. A blocking `strike()` that returns when it is
done cannot be interrupted, and on a machine that swings at people that
is not an acceptable property.

The strike primitives encode the finding from docs/slap-analysis.md:
short strikes are better on both axes at once, faster to land *and*
softer on impact, so the safe strike and the effective strike are the
same strike. `StrikeLimits` enforces that rather than trusting callers.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass

import numpy as np

from tlod.arm import model
from tlod.arm.controller import ArmController, minimum_jerk
from tlod.types import Pose


@dataclass(slots=True)
class StrikeLimits:
    """Bounds on anything that moves fast toward a person.

    Defaults come from the measured strike table: an 8 cm drop lands in
    ~210 ms at ~0.7 m/s, which beats a 230-400 ms human escape budget
    while tapping more softly than a casual high-five (1-3 m/s).
    """

    max_drop: float = 0.08              # metres; the single most important cap
    hover_height: float = 0.08          # resting height above the target plane
    strike_speed: float = 3.5           # rad/s during a strike
    retract_speed: float = 4.0          # rad/s returning; faster is fine, it moves away
    plane_margin: float = 0.005         # never command below target plane minus this
    torque_limit: int = 350             # of 1000, while striking; yields on contact
    normal_torque_limit: int = 800
    min_strike_interval: float = 0.35   # seconds between strikes; thermal and safety

    def clamp_drop(self, drop: float) -> float:
        return float(np.clip(drop, 0.0, self.max_drop))


class Motion(abc.ABC):
    """A steppable movement. `step` returns True when finished."""

    name: str = "motion"

    def __init__(self) -> None:
        self.started_at: float = 0.0
        self.finished: bool = False

    def start(self, controller: ArmController) -> None:
        self.started_at = time.perf_counter()
        self.finished = False
        self._on_start(controller)

    def _on_start(self, controller: ArmController) -> None:
        pass

    @abc.abstractmethod
    def step(self, controller: ArmController, dt: float) -> bool: ...

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def abort(self) -> None:
        self.finished = True


class GoTo(Motion):
    """Min-jerk interpolation to a joint configuration."""

    name = "goto"

    def __init__(self, q_target: np.ndarray, duration: float = 0.6, speed: float | None = None) -> None:
        super().__init__()
        self.q_target = np.asarray(q_target, float)
        self.duration = max(duration, 1e-3)
        self.speed = speed
        self._q0: np.ndarray | None = None

    def _on_start(self, controller) -> None:
        self._q0 = controller.commanded.copy()
        target = self.q_target
        if target.shape[0] == 5:
            target = np.concatenate([target, [self._q0[5]]])
        self.q_target = target

    def step(self, controller, dt) -> bool:
        if self.finished:
            return True
        s = minimum_jerk(self.elapsed / self.duration)
        controller._write(self._q0 + (self.q_target - self._q0) * s,
                          max_speed=self.speed or controller.limits.strike_speed, dt=dt)
        if self.elapsed >= self.duration:
            self.finished = True
        return self.finished


class GoToPose(GoTo):
    """Min-jerk to a Cartesian pose. IK is solved once, at start.

    Solving once rather than per tick is intentional: re-solving against a
    moving warm start can drift between IK branches mid-motion, which on a
    fast move looks like the arm suddenly reconfiguring. For a strike that
    would be both alarming and dangerous.
    """

    name = "goto_pose"

    def __init__(self, pose: Pose, duration: float = 0.6, speed: float | None = None) -> None:
        super().__init__(np.zeros(5), duration, speed)
        self.pose = pose
        self.ok = False

    def _on_start(self, controller) -> None:
        result, _, _ = controller.solve(self.pose, position_only=True)
        self.ok = result.ok
        self.q_target = result.q
        super()._on_start(controller)

    def step(self, controller, dt) -> bool:
        if not self.ok:
            self.finished = True
            return True
        return super().step(controller, dt)


class Hover(GoToPose):
    """Sit above a target point, ready to strike."""

    name = "hover"

    def __init__(self, target_xyz, limits: StrikeLimits, duration: float = 0.5) -> None:
        t = np.asarray(target_xyz, float)
        super().__init__(Pose(float(t[0]), float(t[1]), float(t[2]) + limits.hover_height),
                         duration, speed=limits.retract_speed)


class Strike(Motion):
    """Drive straight down onto a target plane, then stop.

    Safety is structural rather than advisory:

      * the drop is clamped to `StrikeLimits.max_drop`
      * the commanded depth never goes below the target plane, so the
        worst case on a misjudged height is a gentle stall, not a press
      * torque limit is lowered for the duration, so the servo yields on
        unexpected contact instead of pushing through it
      * IK is solved once at start, so the path cannot switch branches
        halfway down
    """

    name = "strike"

    def __init__(
        self,
        target_xyz,
        limits: StrikeLimits,
        duration: float = 0.21,
        depth: float | None = None,
    ) -> None:
        super().__init__()
        self.target = np.asarray(target_xyz, float)
        self.limits = limits
        self.duration = max(duration, 1e-3)
        self.depth = depth
        self.ok = False
        self._q0: np.ndarray | None = None
        self._q1: np.ndarray | None = None
        self._restored = False
        self._controller = None

    def _on_start(self, controller) -> None:
        self._q0 = controller.commanded.copy()
        start_z = model.tool_pose(self._q0[:5]).z
        drop = self.limits.clamp_drop(
            start_z - self.target[2] if self.depth is None else self.depth
        )
        # Never below the plane. This is the guard that makes a wrong
        # height estimate harmless rather than injurious.
        end_z = max(self.target[2] + self.limits.plane_margin, start_z - drop)
        goal = Pose(float(self.target[0]), float(self.target[1]), float(end_z))
        result, _, _ = controller.solve(goal, position_only=True)
        self.ok = result.ok
        self._q1 = np.concatenate([result.q, [self._q0[5]]])

        self._controller = controller
        set_limit = getattr(controller.backend, "set_torque_limit", None)
        if callable(set_limit):
            set_limit(self.limits.torque_limit)
        self._restored = False

    def _restore(self, controller) -> None:
        if self._restored:
            return
        set_limit = getattr(controller.backend, "set_torque_limit", None)
        if callable(set_limit):
            set_limit(self.limits.normal_torque_limit)
        self._restored = True

    def step(self, controller, dt) -> bool:
        if self.finished:
            return True
        if not self.ok:
            self.finished = True
            self._restore(controller)
            return True
        s = minimum_jerk(self.elapsed / self.duration)
        controller._write(self._q0 + (self._q1 - self._q0) * s,
                          max_speed=self.limits.strike_speed, dt=dt)
        if self.elapsed >= self.duration:
            self.finished = True
            self._restore(controller)
        return self.finished

    def abort(self) -> None:
        """Restore torque on the way out.

        An aborted strike used to leave the servos capped at the strike
        limit permanently. That is not hypothetical: run_motion() aborts
        whatever motion it replaces, and the feint handler aborts
        explicitly, so any interrupted strike silently left the arm weak
        for the rest of the session.
        """
        super().abort()
        if self._controller is not None:
            self._restore(self._controller)


class Retract(GoTo):
    """Return to the pre-strike configuration, quickly."""

    name = "retract"

    def __init__(self, q_home: np.ndarray, limits: StrikeLimits, duration: float = 0.25) -> None:
        super().__init__(q_home, duration, speed=limits.retract_speed)


class Feint(Motion):
    """Commit part-way, then pull back.

    Only possible because the robot owns the clock. A feint costs the
    human a reaction -- they flinch, withdraw, and then have to come back
    -- and the recovery is the opening. `fraction` is how much of a real
    strike to show; too little is unconvincing, too much is just a slow
    strike that loses.
    """

    name = "feint"

    def __init__(self, target_xyz, limits: StrikeLimits, fraction: float = 0.45,
                 out: float = 0.10, back: float = 0.18) -> None:
        super().__init__()
        self.target = np.asarray(target_xyz, float)
        self.limits = limits
        self.fraction = float(np.clip(fraction, 0.05, 0.8))
        self.out = out
        self.back = back
        self._q0: np.ndarray | None = None
        self._q1: np.ndarray | None = None
        self.ok = False

    def _on_start(self, controller) -> None:
        self._q0 = controller.commanded.copy()
        start_z = model.tool_pose(self._q0[:5]).z
        drop = self.limits.clamp_drop(start_z - self.target[2]) * self.fraction
        goal = Pose(float(self.target[0]), float(self.target[1]), float(start_z - drop))
        result, _, _ = controller.solve(goal, position_only=True)
        self.ok = result.ok
        self._q1 = np.concatenate([result.q, [self._q0[5]]])

    def step(self, controller, dt) -> bool:
        if self.finished or not self.ok:
            self.finished = True
            return True
        total = self.out + self.back
        e = self.elapsed
        if e < self.out:
            s = minimum_jerk(e / self.out)
        else:
            s = 1.0 - minimum_jerk((e - self.out) / self.back)
        controller._write(self._q0 + (self._q1 - self._q0) * s,
                          max_speed=self.limits.strike_speed, dt=dt)
        if e >= total:
            self.finished = True
        return self.finished


class Hold(Motion):
    """Do nothing for a while, without blocking the control thread."""

    name = "hold"

    def __init__(self, duration: float) -> None:
        super().__init__()
        self.duration = duration

    def step(self, controller, dt) -> bool:
        if self.elapsed >= self.duration:
            self.finished = True
        return self.finished


class Sequence(Motion):
    """Run motions back to back. Aborting aborts the whole sequence."""

    name = "sequence"

    def __init__(self, motions: list[Motion]) -> None:
        super().__init__()
        self.motions = motions
        self._i = 0

    def _on_start(self, controller) -> None:
        self._i = 0
        if self.motions:
            self.motions[0].start(controller)

    def step(self, controller, dt) -> bool:
        if self.finished or not self.motions:
            self.finished = True
            return True
        if self.motions[self._i].step(controller, dt):
            self._i += 1
            if self._i >= len(self.motions):
                self.finished = True
                return True
            self.motions[self._i].start(controller)
        return False

    @property
    def current(self) -> Motion | None:
        return self.motions[self._i] if self._i < len(self.motions) else None

    def abort(self) -> None:
        for m in self.motions:
            m.abort()
        super().abort()

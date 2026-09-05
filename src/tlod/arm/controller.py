"""Motion and safety layer.

Everything that can hurt the robot, the table, or a person's hand is
enforced here, in one place, on every command -- rather than being the
responsibility of each game to remember. A game asks for a pose; this
decides whether that is allowed and how fast to get there.

The guards are deliberately conservative by default. This machine is
designed to move quickly toward a human hand, which is a sentence worth
re-reading. `SafetyLimits.max_speed` and `strike_speed` are the two knobs
that change how hard it can hit; they are separated so that raising the
speed for a game is an explicit, visible decision.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from tlod.arm import model
from tlod.arm.backend import ArmBackend
from tlod.types import ARM_JOINTS, GRIPPER, NUM_JOINTS, JointState, Pose

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SafetyLimits:
    """Hard bounds applied to every command."""

    # Joint space
    max_speed: float = 2.0            # rad/s, normal motion
    strike_speed: float = 5.0         # rad/s, allowed only in an explicit strike
    joint_margin: float = 0.05        # rad to stay clear of the URDF limits

    # Cartesian keep-out. The arm sits on a table; the table is at z=0 in
    # base coordinates unless the mount says otherwise.
    table_z: float = 0.0
    min_height: float = 0.015         # never drive the TCP below this
    max_radius: float = 0.33          # horizontal reach cap, metres
    min_radius: float = 0.08          # do not fold back into the base
    max_height: float = 0.45

    # Watchdog: if nobody sends a command for this long, hold position.
    command_timeout: float = 0.5

    def clamp_pose(self, p: Pose) -> tuple[Pose, list[str]]:
        """Project a requested pose into the allowed workspace.

        Returns the safe pose and a list of which guards fired, so callers
        can log or surface "I could not fully reach there" rather than
        silently doing something different from what was asked.
        """
        violations: list[str] = []
        x, y, z = p.x, p.y, p.z

        if z < self.table_z + self.min_height:
            z = self.table_z + self.min_height
            violations.append("min_height")
        if z > self.max_height:
            z = self.max_height
            violations.append("max_height")

        r = float(np.hypot(x, y))
        if r > self.max_radius:
            s = self.max_radius / r
            x, y = x * s, y * s
            violations.append("max_radius")
        elif r < self.min_radius:
            if r < 1e-6:
                x, y = self.min_radius, 0.0
            else:
                s = self.min_radius / r
                x, y = x * s, y * s
            violations.append("min_radius")

        return Pose(x, y, z, p.pitch, p.roll), violations


@dataclass(slots=True)
class ControllerStats:
    commands: int = 0
    ik_failures: int = 0
    guard_hits: int = 0
    last_ik_ms: float = 0.0
    last_violations: list[str] = field(default_factory=list)


def minimum_jerk(s: float) -> float:
    """Min-jerk time scaling on s in [0,1]. Smooth start and stop, which
    matters on a servo bus: a step command makes the arm slam and the
    whole tabletop ring."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ArmController:
    def __init__(
        self,
        backend: ArmBackend,
        limits: SafetyLimits | None = None,
        control_hz: float = 100.0,
    ) -> None:
        self.backend = backend
        self.limits = limits or SafetyLimits()
        self.control_hz = control_hz
        self.stats = ControllerStats()
        self._command = np.zeros(NUM_JOINTS)
        self._estop = False
        self._lock = threading.Lock()
        self._last_command_time = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self.backend.connected:
            self.backend.connect()
        state = self.backend.read()
        with self._lock:
            self._command = state.q.copy()
        self._last_command_time = time.perf_counter()

    def stop(self, park: bool = True) -> None:
        try:
            if park and not self._estop:
                self.park()
        finally:
            self.backend.disconnect()

    @property
    def estopped(self) -> bool:
        return self._estop

    def estop(self) -> None:
        """Freeze at the current measured position and refuse further motion.

        Deliberately holds torque rather than cutting it: a limp arm falls,
        and it may be falling onto the hand that triggered the stop.
        """
        state = self.backend.read()
        with self._lock:
            self._estop = True
            self._command = state.q.copy()
        self.backend.write(state.q)
        log.warning("E-STOP engaged at q=%s", np.round(state.q, 3))

    def release_estop(self) -> None:
        state = self.backend.read()
        with self._lock:
            self._command = state.q.copy()
            self._estop = False
        log.info("e-stop released")

    # -- state -------------------------------------------------------------
    def state(self) -> JointState:
        return self.backend.read()

    def pose(self) -> Pose:
        return model.tool_pose(self.backend.read().q[:5])

    @property
    def commanded(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    # -- low level ---------------------------------------------------------
    def _write(self, q: np.ndarray, max_speed: float | None = None, dt: float | None = None) -> None:
        """Rate-limit and dispatch a joint command."""
        max_speed = self.limits.max_speed if max_speed is None else max_speed
        dt = (1.0 / self.control_hz) if dt is None else dt

        with self._lock:
            # Checked inside the lock. estop() runs on whichever thread
            # noticed the problem -- the viewer, a watchdog -- while the
            # control loop is mid-command, and a check outside the lock
            # leaves a window where a command issued after the stop still
            # reaches the servos.
            if self._estop:
                return
            prev = self._command
            step_cap = max_speed * dt
            delta = np.clip(np.asarray(q, float) - prev, -step_cap, step_cap)
            cmd = prev + delta
            lo = np.concatenate([model.JOINT_LIMITS[:, 0] + self.limits.joint_margin,
                                 [model.GRIPPER_LIMITS[0]]])
            hi = np.concatenate([model.JOINT_LIMITS[:, 1] - self.limits.joint_margin,
                                 [model.GRIPPER_LIMITS[1]]])
            cmd = np.clip(cmd, lo, hi)
            self._command = cmd
        self.backend.write(cmd)
        self._last_command_time = time.perf_counter()
        self.stats.commands += 1

    # -- pose control ------------------------------------------------------
    def solve(self, target: Pose, *, position_only: bool = True, seed: np.ndarray | None = None):
        """IK against the safety-clamped target, warm-started from the last
        command so successive solves stay on the same branch."""
        safe, violations = self.limits.clamp_pose(target)
        if violations:
            self.stats.guard_hits += 1
            self.stats.last_violations = violations

        if seed is None:
            seed = self.commanded[:5]

        t0 = time.perf_counter()
        if position_only:
            result = model.ik_position(safe.xyz(), seed, pitch=safe.pitch or None)
        else:
            result = model.ik(safe, seed)
        self.stats.last_ik_ms = (time.perf_counter() - t0) * 1e3
        if not result.ok:
            self.stats.ik_failures += 1
        return result, safe, violations

    def servo_pose(
        self,
        target: Pose,
        *,
        position_only: bool = True,
        max_speed: float | None = None,
        dt: float | None = None,
    ) -> bool:
        """Send one tracking command toward `target`. Call at control rate.

        This is the streaming entry point used by the game loop. It never
        blocks. Returns whether IK succeeded; on failure the arm simply
        holds its previous command rather than lurching toward a
        half-solved configuration.
        """
        result, _, _ = self.solve(target, position_only=position_only)
        if not result.ok and result.pos_error > 0.02:
            return False
        q = np.concatenate([result.q, [self.commanded[5]]])
        self._write(q, max_speed=max_speed, dt=dt)
        return result.ok

    # -- blocking moves ----------------------------------------------------
    def goto_joints(self, q_target: np.ndarray, duration: float = 1.5) -> None:
        """Interpolate to a joint configuration over `duration` seconds."""
        q_target = np.asarray(q_target, float)
        if q_target.shape[0] == 5:
            q_target = np.concatenate([q_target, [self.commanded[5]]])
        q_start = self.commanded.copy()
        period = 1.0 / self.control_hz
        t0 = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - t0
            s = minimum_jerk(elapsed / duration) if duration > 0 else 1.0
            self._write(q_start + (q_target - q_start) * s,
                        max_speed=self.limits.strike_speed, dt=period)
            if elapsed >= duration:
                return
            time.sleep(max(0.0, period - (time.perf_counter() - t0 - elapsed)))

    def goto_pose(self, target: Pose, duration: float = 1.5, *, position_only: bool = True) -> bool:
        result, _, _ = self.solve(target, position_only=position_only)
        if not result.ok:
            log.warning("goto_pose: IK failed, %.1f mm off", result.pos_error * 1e3)
            return False
        self.goto_joints(result.q, duration)
        return True

    def park(self, duration: float = 2.0) -> None:
        """Return to the safe home configuration."""
        self.goto_joints(model.HOME, duration)

    # -- gripper -----------------------------------------------------------
    def set_gripper(self, opening: float) -> None:
        """`opening` in [0, 1]: 0 fully closed, 1 fully open."""
        lo, hi = model.GRIPPER_LIMITS
        value = lo + (hi - lo) * float(np.clip(opening, 0.0, 1.0))
        q = self.commanded.copy()
        q[5] = value
        self._write(q, max_speed=self.limits.strike_speed)

    # -- watchdog ----------------------------------------------------------
    def check_watchdog(self) -> bool:
        """True if commands have gone stale. The caller should hold or park."""
        return (time.perf_counter() - self._last_command_time) > self.limits.command_timeout

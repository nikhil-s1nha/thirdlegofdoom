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
from tlod.arm.power import PowerGovernor
from tlod.arm.profile import MotionProfile, ProfileLimits
from tlod.types import NUM_JOINTS, JointState, Pose

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

    # Bounds on how fast the command may change, not just how fast it may
    # move. See tlod.arm.profile: acceleration is what sets motor current,
    # so this is the limit that decides whether the supply copes.
    max_accel: float = 8.0            # rad/s^2
    max_jerk: float = 80.0            # rad/s^3

    # Watchdog: if nobody sends a command for this long, hold position.
    command_timeout: float = 0.5

    # A stalled control thread must not be allowed to authorise a large
    # step. Every rate limit here is expressed per tick as `limit * dt`,
    # so a tick that took 200 ms instead of 10 ms permits twenty times the
    # motion -- which is exactly the lurch the limits exist to prevent, and
    # it fires precisely when something has already gone wrong. Clamping dt
    # makes a late tick move less than it wanted rather than more.
    max_tick_dt: float = 0.05

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
    peak_speed: float = 0.0           # rad/s, largest commanded joint speed
    peak_accel: float = 0.0           # rad/s^2, and the acceleration with it
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
        governor: PowerGovernor | None = None,
    ) -> None:
        self.backend = backend
        # Off by default: the simulator has no power supply to overload,
        # and a governor silently slowing a simulated arm would make every
        # timing conclusion drawn from it wrong in a way nothing reports.
        self.governor = governor
        self.limits = limits or SafetyLimits()
        self.control_hz = control_hz
        self.stats = ControllerStats()
        self._command = np.zeros(NUM_JOINTS)
        self._estop = False
        self._lock = threading.Lock()
        self._last_command_time = 0.0
        self.profile = MotionProfile(np.zeros(NUM_JOINTS), self.profile_limits())
        # Time-scaling applied on top of every limit, in (0, 1]. The power
        # governor writes it; everything else just obeys it. Kept separate
        # from the limits themselves so that derating is visible and
        # reversible rather than quietly editing the configured bounds.
        self._derate = 1.0

    def profile_limits(self, max_speed: float | None = None) -> ProfileLimits:
        """Profile bounds for this command, never looser than the safety ones.

        A caller asking for a speed is asking to go *slower* -- a servo
        step, a careful approach. It substituted rather than bounded,
        which meant any primitive carrying its own speed silently
        outranked safety.max_speed: measured, a StrikeLimits retract of
        4.0 rad/s ran a commanded 3.53 on an arm configured to cap at
        3.5. The cap that gets tuned down for a marginal supply, or for a
        person standing closer, is exactly the one that must win.
        """
        return ProfileLimits(
            max_speed=self.limits.max_speed if max_speed is None
            else min(float(max_speed), self.limits.max_speed),
            max_accel=self.limits.max_accel,
            max_jerk=self.limits.max_jerk,
        )

    @property
    def derate(self) -> float:
        return self._derate

    def set_derate(self, factor: float) -> None:
        """Scale every kinematic limit by `factor` in (0, 1].

        Time-scaling, so acceleration falls as the square and the current
        that goes with it falls roughly as fast. This is the knob the
        brownout governor turns.
        """
        self._derate = float(np.clip(factor, 0.05, 1.0))

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self.backend.connected:
            self.backend.connect()
        with self._lock:
            state = self.backend.read()
            self._command = state.q.copy()
            self.profile.reset(state.q)
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
        with self._lock:
            state = self.backend.read()
            self._estop = True
            self._command = state.q.copy()
            # Discard the profile's velocity and acceleration too. Freezing
            # only the position would leave the shaper mid-motion, and
            # releasing the stop would resume the swing that caused it.
            self.profile.reset(state.q)
            self.backend.write(state.q)
        log.warning("E-STOP engaged at q=%s", np.round(state.q, 3))

    def release_estop(self) -> None:
        with self._lock:
            state = self.backend.read()
            self._command = state.q.copy()
            self.profile.reset(state.q)
            self._estop = False
        log.info("e-stop released")

    # -- state -------------------------------------------------------------
    def state(self) -> JointState:
        # Locked: on real hardware this shares the servo bus with _write()'s
        # backend.write() from the control loop's own thread (and, for
        # anything polling state() off-thread, like ArmTelemetryPublisher).
        # Two threads doing raw serial I/O on the same port at once can
        # corrupt a transaction -- harmless on MockArm, but on FeetechArm it
        # can throw or return garbage, which is exactly the kind of failure
        # that should never touch the servos.
        with self._lock:
            return self.backend.read()

    def diagnostics(self) -> dict[str, object]:
        """Backend health, read under the bus lock.

        Same reason as `state()`: on real hardware this is two dozen
        round trips on the serial port the control loop is also writing
        to, and two threads interleaving raw transactions corrupt each
        other. Callers should go through here rather than reaching for
        `controller.backend.diagnostics()`.
        """
        with self._lock:
            return self.backend.diagnostics()

    def pose(self) -> Pose:
        with self._lock:
            q = self.backend.read().q[:5]
        return model.tool_pose(q)

    @property
    def commanded(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    # -- low level ---------------------------------------------------------
    def _write(self, q: np.ndarray, max_speed: float | None = None, dt: float | None = None) -> None:
        """Shape and dispatch a joint command.

        The requested configuration is a *target*, not the value written to
        the servos. It goes through the motion profile, which bounds the
        velocity, acceleration and jerk of the command stream and keeps the
        joints synchronised. Callers therefore get the pose they asked for
        only as fast as the arm and its power supply can actually deliver
        it, which is the correct failure mode: a late arrival rather than a
        lurch and a brownout.
        """
        dt = (1.0 / self.control_hz) if dt is None else dt
        dt = min(dt, self.limits.max_tick_dt)
        requested = self.profile_limits(max_speed)
        if self.governor is not None:
            self._derate = self.governor.update(self.profile.q, requested, dt)
        limits = requested.scaled(self._derate)

        with self._lock:
            # Checked inside the lock. estop() runs on whichever thread
            # noticed the problem -- the viewer, a watchdog -- while the
            # control loop is mid-command, and a check outside the lock
            # leaves a window where a command issued after the stop still
            # reaches the servos.
            if self._estop:
                return
            # Clamp the target, not the profiled output. Clipping
            # afterwards would leave the profile integrating toward a
            # configuration it is never allowed to reach, so its internal
            # velocity would wind up against the limit and be carried into
            # the next move as a lurch away from it.
            lo = np.concatenate([model.JOINT_LIMITS[:, 0] + self.limits.joint_margin,
                                 [model.GRIPPER_LIMITS[0]]])
            hi = np.concatenate([model.JOINT_LIMITS[:, 1] - self.limits.joint_margin,
                                 [model.GRIPPER_LIMITS[1]]])
            target = np.clip(np.asarray(q, float), lo, hi)
            cmd = self.profile.step(target, dt, limits)
            self._command = cmd
            self.stats.peak_speed = max(self.stats.peak_speed, self.profile.speed)
            self.stats.peak_accel = max(self.stats.peak_accel, self.profile.accel)
            self.backend.write(cmd)
        self._last_command_time = time.perf_counter()
        self.stats.commands += 1

    def settled(self, dwell: float = 0.03) -> bool:
        """True once the commanded setpoint has held still for `dwell`.

        The dwell is not padding. The servo trails its goal by its own
        latency and slew rate, so the instant the command stops the arm is
        still arriving; returning "done" then means a caller that measures
        the tool position finds it several millimetres short, and a
        sequence of motions each starts from somewhere its predecessor did
        not intend. One servo time constant of quiet is what makes
        "finished" mean the arm is actually there.
        """
        return self.profile.rest_time >= dwell

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
    def goto_joints(self, q_target: np.ndarray, duration: float = 1.5,
                    settle_timeout: float = 2.0) -> None:
        """Interpolate to a joint configuration over `duration` seconds.

        `duration` shapes the plan; the motion profile underneath may still
        be catching up when the plan ends, because it -- not the caller --
        has the last word on how fast the arm may accelerate. So the plan is
        followed by however long it takes the profile to converge. Without
        that, a blocking move would return with the arm still travelling,
        and every caller that assumed "goto_joints returned, so we are
        there" would be wrong by a few centimetres.
        """
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
            if self._estop:
                return
            if elapsed >= duration and (self.settled() or elapsed >= duration + settle_timeout):
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

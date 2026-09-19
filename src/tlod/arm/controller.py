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
        flex_gain: float = 0.0,
        flex_offset: float = 0.0,
    ) -> None:
        self.backend = backend
        # Sag the encoders cannot see, as metres of droop per metre of
        # horizontal reach plus a constant. See `compensate`.
        self.flex_gain = flex_gain
        self.flex_offset = flex_offset
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
            # Set first, and unconditionally. Everything below can fail --
            # it talks to the same bus that is the most likely reason an
            # e-stop was called for in the first place -- and a stop that
            # raises before it has stopped anything is not a stop.
            #
            # This is not hypothetical. A sync read failed mid-strike, the
            # control loop answered it by calling estop(), estop() read the
            # bus, the read failed again, and the exception took out the
            # control thread with the arm still commanded downward. The
            # safety action was the one action that could not tolerate the
            # failure it existed to handle.
            self._estop = True
            try:
                q = self.backend.read().q
            except Exception:
                # No fresh reading, so freeze at the last command instead.
                # Slightly ahead of where the arm physically is, which is
                # the safe direction: it stops the profile advancing, and
                # the arm is already tracking toward it.
                log.warning("e-stop could not read the arm; freezing at the "
                            "last command", exc_info=True)
                q = self._command.copy()
            self._command = q.copy()
            # Discard the profile's velocity and acceleration too. Freezing
            # only the position would leave the shaper mid-motion, and
            # releasing the stop would resume the swing that caused it.
            self.profile.reset(q)
            try:
                self.backend.write(q)
            except Exception:
                log.error("e-stop could not command the arm; the servos hold "
                          "their last goal position", exc_info=True)
        log.warning("E-STOP engaged at q=%s", np.round(q, 3))

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

    def uncompensate(self, fk: Pose) -> Pose:
        """The inverse of `compensate`: forward kinematics -> real tool.

        The flex depends on horizontal radius, which the correction does
        not change, so this is exact rather than an approximation.
        """
        if not self.flex_gain and not self.flex_offset:
            return fk
        radius = float(np.hypot(fk.x, fk.y))
        return Pose(fk.x, fk.y,
                    fk.z - self.flex_gain * radius - self.flex_offset)

    def pose_fk(self) -> Pose:
        """Where the *output shafts* put the tool, straight from the model.

        Only for callers that mean the kinematic frame specifically --
        diagnostics, and the tests that pin FK. Everything about the game
        wants `pose`.
        """
        with self._lock:
            q = self.backend.read().q[:5]
        return model.tool_pose(q)

    def pose(self) -> Pose:
        """Where the tool actually is, as far as this arm can tell.

        Forward kinematics less the flex, so it is in the same frame as
        the targets `solve` accepts and as the hand positions the camera
        reports. That consistency is the whole point: `Strike` reads a
        start height here and computes a drop against a hand plane from
        the vision pipeline, and when those two were in different frames
        the strike aimed 25 mm high and the feint moved three millimetres.
        """
        return self.uncompensate(self.pose_fk())

    @property
    def commanded(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    # -- low level ---------------------------------------------------------
    def _write(self, q: np.ndarray, max_speed: float | None = None, dt: float | None = None,
               limits: ProfileLimits | None = None) -> None:
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
        # `limits` is the escape hatch for a motion that is provably not
        # approaching anything -- a flourish is joint space, has no target,
        # and its envelope is zero at both ends, so it cannot walk toward a
        # hand however it is interrupted. Everything else goes through
        # `profile_limits`, which bounds by safety rather than substituting
        # for it. The joint clamp, the e-stop check and the governor below
        # still apply either way; this widens the rate limits only.
        requested = self.profile_limits(max_speed) if limits is None else limits
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
    def compensate(self, target: Pose) -> Pose:
        """Raise a target by the droop that no sensor on this arm can see.

        There are two height errors and they need different treatment.
        The first is the servos settling low under load: they are
        proportional position controllers, so they come to rest where
        their restoring torque balances gravity. The encoders see that
        one, which is why `goto_pose` can correct it by re-aiming at what
        forward kinematics reports.

        The second is the arm bending *after* the output shaft -- link
        deflection, gearbox backlash, mount compliance. The shaft is
        exactly where the encoder says; the tool is not. Nothing on this
        machine observes it, so no amount of closing the loop on the
        encoders can remove it, and a ruler is the only instrument that
        sees it at all.

        Measured on this rig at a commanded 69 mm, with the encoder-side
        droop already corrected -- forward kinematics reading 67-69 mm
        throughout -- against a ruler on the paddle:

            radius 0.224 m   ruler 52 mm   gap 15.4 mm
            radius 0.316 m   ruler 44 mm   gap 25.0 mm
            radius 0.396 m   ruler 36 mm   gap 33.0 mm

        The slope between consecutive points is 103.7 and 100.3 mm per
        metre, so it is a straight line in the reach, which is what a
        bending beam should be: deflection goes with the moment arm.
        `0.102 * r - 0.0074` fits all three inside 0.3 mm.

        It is a *feed-forward* correction, not a loop, because the thing
        it corrects is unobservable -- and it depends on radius, which
        raising z does not change, so one shot is exact rather than
        iterative.
        """
        if not self.flex_gain and not self.flex_offset:
            return target
        radius = float(np.hypot(target.x, target.y))
        return Pose(target.x, target.y,
                    target.z + self.flex_gain * radius + self.flex_offset)

    def solve(self, target: Pose, *, position_only: bool = True, seed: np.ndarray | None = None,
              compensate: bool = True):
        """IK against the safety-clamped target, warm-started from the last
        command so successive solves stay on the same branch.

        `compensate=False` for a target that has already been raised by
        `compensate()` -- the re-aiming in `goto_pose` works in the
        compensated frame and would otherwise apply it twice.
        """
        if compensate:
            target = self.compensate(target)
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

    # How many times `goto_pose` re-aims at a target it has undershot, and
    # how close is close enough to stop.
    #
    # The servos are proportional position controllers, so under a constant
    # load they settle where their restoring torque balances gravity --
    # a steady-state offset, not a tracking failure, and one that grows
    # with reach because the gravitational torque on the shoulder does.
    # The encoders see it: FK reports the drooped height, not the
    # commanded one.
    #
    # It lands almost entirely in z. For an arm stretched out roughly
    # horizontally, an angular droop of `d` at the shoulder drops the tool
    # by about L*d -- first order in the reach -- and moves it sideways by
    # about L*d^2, which vanishes. Measured on this rig at a commanded
    # z of 69 mm: 67 mm achieved at (0.20, 0.10), 52 mm at (0.40, 0.10),
    # 43 mm at (0.36, 0.165). Same command, 24 mm of spread, and x and y
    # within a few mm throughout.
    #
    # So rather than model it, close the loop: move, read where the arm
    # actually got to, and re-aim by the difference. This converges on the
    # right pose whatever the load, the reach or the payload, because it
    # predicts nothing. Two passes take 26 mm to a couple of mm.
    REAIM_PASSES: int = 2
    REAIM_TOLERANCE: float = 0.003     # metres; stop once inside this

    def goto_pose(self, target: Pose, duration: float = 1.5, *, position_only: bool = True,
                  reaim: bool | None = None) -> bool:
        """Drive the tool to `target`, re-aiming at what the encoders report.

        `reaim=False` for a caller that wants one open-loop shot -- a
        ballistic strike, where the point is that it is fast and a second
        pass would be a second, slower descent.
        """
        result, _, _ = self.solve(target, position_only=position_only)
        if not result.ok:
            log.warning("goto_pose: IK failed, %.1f mm off", result.pos_error * 1e3)
            return False
        self.goto_joints(result.q, duration)

        if reaim is False or self.REAIM_PASSES <= 0:
            return True

        wanted = target.xyz()
        for _ in range(self.REAIM_PASSES):
            error = wanted - self.pose().xyz()
            if float(np.linalg.norm(error)) <= self.REAIM_TOLERANCE:
                break
            # Aim past the target by however far it fell short. Solving for
            # the corrected point rather than nudging joints keeps the
            # safety clamp and the IK branch choice in the loop.
            aim = Pose(*(wanted + error))
            again, _, _ = self.solve(aim, position_only=position_only)
            if not again.ok:
                break
            # Short, because it is a small correction from a standstill.
            self.goto_joints(again.q, max(duration * 0.35, 0.25))

        left = float(np.linalg.norm(wanted - self.pose().xyz()))
        if left > self.REAIM_TOLERANCE:
            log.info("goto_pose: %.1f mm off after re-aiming", left * 1e3)
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

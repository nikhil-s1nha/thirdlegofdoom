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

Measurement on the real arm added a floor to that. Below a ~250 ms ask
for an 8 cm drop the arm does not land sooner -- it lands *further from
where it was aimed*, and past a point the miss is bigger than the contact
tolerance, so the fastest strike is the one that cannot score. Short is
better right up to that floor and worse below it; the defaults sit on it.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass

import numpy as np

from tlod.arm import model
from tlod.arm.controller import ArmController, minimum_jerk
from tlod.types import Pose

log = logging.getLogger(__name__)


@dataclass(slots=True)
class StrikeLimits:
    """Bounds on anything that moves fast toward a person.

    Defaults now come from the arm rather than from the simulator. Measured
    on an SO-ARM101 at 12 V 5 A with `servo_accel: 35`, `max_accel: 35`,
    `max_jerk: 400`, `max_speed: 3.5`: an 8 cm drop asked for in 250 ms
    takes 310 ms end to end and stops 9 mm short of the target. Of that
    310 ms, 50-60 ms is `Motion._complete` waiting for `controller.settled`
    rather than travel, so the paddle is moving for ~255 ms and covers the
    8 cm at ~0.3 m/s mean, ~0.5 m/s peak. Still a third of a casual
    high-five (1-3 m/s), and still inside a 230-400 ms human escape budget.

    The old docstring said ~210 ms at ~0.7 m/s. Both halves were the
    simulator: the real arm is 100 ms slower and correspondingly gentler.
    """

    max_drop: float = 0.08              # metres; the single most important cap
    # Set against max_drop and press_depth, not chosen freely: see
    # __post_init__. `Strike` clamps its drop to max_drop, and the swing
    # has to cover the hover *and* the press below the hand, so the floor
    # the paddle can actually reach is (hover - max_drop). Hovering
    # further away is not "more warning"; past this value the paddle
    # stops above where it was aimed.
    #
    # This was 0.08, equal to max_drop, from when the strike aimed at the
    # hand plane exactly. It cost a session of hardware testing when
    # press_depth arrived: 80 + 17 > 80, so every strike's floor was
    # clamped to precisely the hand plane, and the contact sensor was
    # being asked to tell a hit from a miss across a band of zero width.
    # The logs said "floor 22 mm, hand 22 mm" on every line.
    hover_height: float = 0.063         # resting height above the target plane
    # Binding, not decorative: the measured peak commanded joint speed
    # during an 8 cm strike under the real arm's limits is 3.41 rad/s, and
    # dropping this to 2.0 stretches the same strike to 400 ms.
    strike_speed: float = 3.5           # rad/s during a strike
    # Was 4.0, on the theory that retracting fast is free because it moves
    # away. The theory is fine; the number was not. Nothing clamps it: a
    # primitive's per-call speed *replaces* `SafetyLimits.max_speed` inside
    # `ArmController.profile_limits()` rather than being min()'d against
    # it, so 4.0 really did authorise -- and reach -- 3.53 rad/s on an arm
    # configured to cap at 3.5. It bought 10 ms on a 320 ms retract. This
    # number is the only guard there is, so it matches the configured
    # ceiling.
    retract_speed: float = 3.5          # rad/s returning; away from the hand
    # How far BELOW the estimated hand surface to command the paddle.
    #
    # This was +5 mm above it, named plane_margin, on the reasoning that
    # stopping short of the hand is what keeps a wrong height estimate
    # harmless. The reasoning had the wrong guard in mind. What bounds the
    # force is `torque_limit`: the servo cannot push harder than 350/1000
    # no matter how deep it is asked to go, and the measured press against
    # a rigid book -- the stiffest thing available -- was a 0.038 lean.
    # The geometry never bounded force; it only decided whether contact
    # happened at all, and at +5 mm it decided "usually not".
    #
    # It also made detection depend on the thickness of the hand. The
    # sensor reads the torque still being spent while held, and torque is
    # only spent if the arm is *blocked short of its commanded floor*. A
    # floor above the hand means the paddle arrives freely, spends nothing
    # and scores a dodge -- having touched. Measured: floor 27 mm, a hand
    # that blocked at 29 mm, 2 mm of press, 0.037. A flatter hand, or the
    # same hand held flatter, reaches 27 mm unobstructed and reads 0.001.
    #
    # Commanding below the surface instead makes the press deep enough
    # that hand thickness stops mattering, and 10 mm was not enough of
    # that. The number now comes from the arm's own frame rather than
    # from a guess: driven to the joint angles at which the gripper rests
    # on the table, FK reports the tool at +0.2 mm, so **model z = 0 is
    # the work surface**. Against a 22 mm hand plane, 17 mm of depth puts
    # the floor at 5 mm -- five millimetres of air -- and a hand of any
    # plausible thickness is then blocking the paddle by 17-24 mm rather
    # than the 2 mm that was being asked to carry the whole decision.
    #
    # `safety.min_height` still has the last word, and is what keeps this
    # off the table -- so raising press_depth without lowering min_height
    # changes nothing at all. At 5 mm the two now meet exactly, which is
    # deliberate: the floor is the guard, not something the guard has to
    # rescue.
    press_depth: float = 0.017
    torque_limit: int = 350             # of 1000, while striking; yields on contact
    normal_torque_limit: int = 800
    # Stay down at the bottom, still at `torque_limit`, before retracting.
    #
    # This is what makes contact detectable at all, and it is worth being
    # precise about why. Measured on this arm across nothing / a book / a
    # hand, the peak load *during* the swing read 0.330 / 0.326 / 0.350 --
    # a rigid book, the easiest thing there is to feel, landed between the
    # other two. Load climbs monotonically to the ceiling in every run,
    # empty table included, because the arm is accelerating and braking its
    # own mass: the whole swing is one transient, and anything measured
    # during it describes the swing rather than what the swing hit.
    #
    # Held still at the bottom the same three read 0.001 / 0.038 / 0.037.
    # With no acceleration left to confound it, what remains is the arm
    # pressing on what is underneath. The servo's own load filter needs
    # ~250 ms to decay from the swing before that is true, and 450 ms was
    # sized for that.
    #
    # `--contact height` does not wait for any filter -- it reads the
    # encoders, which are already right the moment the arm stops -- and
    # needs 120 ms. This is that plus margin, because the hold is not
    # free: it is the servos stalled against a hand at their torque limit,
    # every strike, and sustained stall current is what a 5 A supply and
    # six servos have least of. It went in at 450 ms this afternoon and
    # the bus started dropping transactions the same afternoon.
    #
    # Raise it back to 0.45 for `--contact press`, which reads a filtered
    # torque estimate and does need the 300 ms. cmd_play warns if the hold
    # is shorter than the sensor's settle window.
    press_hold: float = 0.20
    # Load above the hover baseline, held, that counts as something being
    # there. Nothing measured 0.001 and the two real obstacles 0.037-0.038,
    # so this sits in a gap almost two orders of magnitude wide.
    press_threshold: float = 0.02
    # A backstop, not the working cadence. HandSlapGame's own dwells --
    # retract, then 0.6 s of settle, then 0.4 s of ready before the hazard
    # rate is allowed to fire -- already put ~1.3 s between strikes, so
    # this only ever fires if those change.
    min_strike_interval: float = 0.35   # seconds between strikes; thermal and safety

    def __post_init__(self) -> None:
        # The invariant that ties the three distances together. A strike
        # from `hover_height` above the hand has to travel that plus
        # `press_depth` to put the paddle below it, and `max_drop` caps
        # the travel -- so if this does not hold, the floor silently rises
        # to (hover - max_drop) and both contact sensors lose the band
        # they decide on. Nothing downstream can detect that, because a
        # floor sitting exactly at the hand plane is a perfectly
        # ordinary-looking number.
        needed = self.hover_height + self.press_depth
        if needed > self.max_drop + 1e-9:
            log.warning(
                "strike geometry cannot reach its floor: hover %.0f mm + press "
                "%.0f mm = %.0f mm of travel, but max_drop caps it at %.0f mm, so "
                "the paddle stops %.0f mm above where it is aimed. Lower "
                "hover_height to %.0f mm or raise max_drop to %.0f mm.",
                self.hover_height * 1e3, self.press_depth * 1e3, needed * 1e3,
                self.max_drop * 1e3, (needed - self.max_drop) * 1e3,
                (self.max_drop - self.press_depth) * 1e3, needed * 1e3)

    @property
    def max_hover(self) -> float:
        """The highest hover from which the paddle can still reach its floor.

        `max_drop` caps the travel and the swing has to cover the hover
        *and* the press below the hand, so this is what any caller
        choosing a hover has to clamp against -- not `max_drop` itself,
        which is the mistake that leaves the floor sitting exactly on the
        hand plane.
        """
        return self.max_drop - self.press_depth

    @property
    def reachable_floor_offset(self) -> float:
        """Depth below the hand the paddle can actually get to, in metres.

        Equals `press_depth` when the geometry is consistent, and less
        when `max_drop` is the binding constraint. Zero or negative means
        the paddle stops at or above the hand and cannot land at all.
        """
        return min(self.press_depth, self.max_drop - self.hover_height)

    def clamp_drop(self, drop: float) -> float:
        return float(np.clip(drop, 0.0, self.max_drop))


class Motion(abc.ABC):
    """A steppable movement. `step` returns True when finished."""

    name: str = "motion"

    # How long to keep ticking after a motion's plan has run out, waiting
    # for the controller's motion profile to catch up. The plan says where
    # the arm should be; the profile decides how fast it is allowed to get
    # there, and under a tight acceleration limit -- a derated one in
    # particular -- it can still be travelling when the plan ends. Finishing
    # on plan time alone would hand the next motion an arm that is still
    # moving, and a sequence of those accumulates error until a hover is
    # nowhere near where the strike expects to start from.
    settle_timeout: float = 0.75

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

    def _complete(self, controller: ArmController, plan_duration: float) -> bool:
        """Plan has run out and the arm has stopped -- or waited long enough.

        The timeout matters: a target the profile can never converge on,
        because a joint limit clamps it short, would otherwise never
        report done and would wedge the state machine driving it.
        """
        if self.elapsed < plan_duration:
            return False
        return controller.settled() or self.elapsed >= plan_duration + self.settle_timeout

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
        if self._complete(controller, self.duration):
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
        duration: float = 0.25,   # the measured floor; see StrikeLimits
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
        self._holding_since: float | None = None

    def _on_start(self, controller) -> None:
        self._q0 = controller.commanded.copy()
        start_z = model.tool_pose(self._q0[:5]).z
        # `press_depth` below the estimated hand surface, so the paddle is
        # still trying to descend when it meets the hand and the servos
        # have something to spend torque on. The guard against a wrong
        # height estimate is `torque_limit`, not this geometry.
        #
        # The drop is measured to *this*, not to the hand plane. Measuring
        # it to the plane and then subtracting made press_depth a complete
        # no-op: clamp_drop(start - hand) lands end_z exactly on the hand,
        # and max() of that against a floor below it returns the hand
        # again. Every strike stopped precisely where it used to.
        floor = self.target[2] - self.limits.press_depth
        drop = self.limits.clamp_drop(
            start_z - floor if self.depth is None else self.depth
        )
        # max_drop can still stop the swing above the floor; the max()
        # keeps an explicit `depth` from going below it.
        end_z = max(floor, start_z - drop)
        goal = Pose(float(self.target[0]), float(self.target[1]), float(end_z))
        result, safe, violations = controller.solve(goal, position_only=True)
        self.ok = result.ok
        self._q1 = np.concatenate([result.q, [self._q0[5]]])

        # What the strike will actually do, in millimetres, before it does
        # it. Every term here can quietly collapse the drop to nothing --
        # a hover that never got to height, a hand estimated at the wrong
        # depth, the safety floor, the plane margin, or IK solving to
        # somewhere other than it was asked -- and from across the table
        # they all look the same: an arm that twitched.
        reached = model.tool_pose(result.q).z if result.ok else float("nan")
        log.debug("strike: from %.0f mm to %.0f mm (drop %.0f, asked %.0f, "
                  "hand %.0f, ik reached %.0f)%s",
                  start_z * 1e3, end_z * 1e3, (start_z - end_z) * 1e3,
                  self.limits.clamp_drop(start_z - self.target[2]) * 1e3,
                  self.target[2] * 1e3, reached * 1e3,
                  f", clamped by {', '.join(violations)}" if violations else "")
        if self.ok and start_z - end_z < 0.01:
            log.warning("strike drop is only %.0f mm: hovering at %.0f mm over a "
                        "hand estimated at %.0f mm%s",
                        (start_z - end_z) * 1e3, start_z * 1e3, self.target[2] * 1e3,
                        f", clamped by {', '.join(violations)}" if violations else "")

        self._controller = controller
        self._holding_since = None
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
        if self._holding_since is None:
            s = minimum_jerk(self.elapsed / self.duration)
            controller._write(self._q0 + (self._q1 - self._q0) * s,
                              max_speed=self.limits.strike_speed, dt=dt)
            if self._complete(controller, self.duration):
                if self.limits.press_hold > 0.0:
                    self._holding_since = time.perf_counter()
                else:
                    self.finished = True
                    self._restore(controller)
            return self.finished
        # Pressing. Keep commanding the floor, and keep the lowered torque
        # limit: the point of the hold is to lean on whatever is down there
        # with the same authority the swing had, and restoring torque here
        # would both change what is being measured and press harder than
        # the strike was ever authorised to.
        controller._write(self._q1, max_speed=self.limits.strike_speed, dt=dt)
        if time.perf_counter() - self._holding_since >= self.limits.press_hold:
            self.finished = True
            self._restore(controller)
        return self.finished

    @property
    def pressing(self) -> bool:
        """Down, stopped, and leaning -- the only time load means anything."""
        return self._holding_since is not None and not self.finished

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
        if self._complete(controller, total):
            self.finished = True
        return self.finished


# -- performance -----------------------------------------------------------
#
# The point of this robot is to be entertaining, and a machine that only
# ever moves with purpose is not. These give it a body language: a taunt
# after a bluff lands, a sulk when it misses, a chomp because a gripper
# that can open and close should sometimes open and close for no reason.
#
# *Where* the performance goes is a design constraint rather than a
# matter of taste. A feint scores only if it is credible for its first
# hundred milliseconds -- that is the entire mechanic -- and a robot
# visibly clowning during a commit draws no flinch and wins nothing. So
# none of this runs during a commit. It runs while waiting, and after the
# round is already decided, which is where a person puts it too.


@dataclass(frozen=True, slots=True)
class Move:
    """One flourish: how far each joint swings, and how many times."""

    amplitudes: tuple[float, ...]      # radians, JOINT_NAMES order
    cycles: float


# Amplitudes are timid on the joints that translate the tool and generous
# on the ones that do not: a 26-degree wrist roll is unmistakable across a
# room and moves the gripper nowhere, where the same angle at the shoulder
# would sweep it through 10 cm of table.
#
# One swing each, and a slow one, because the motion profile is not a
# suggestion. Peak joint acceleration for a swing of amplitude A at f Hz
# is A(2*pi*f)^2, so a 0.5 rad wiggle at 10 Hz asks for ~2000 rad/s^2
# against a configured 35 and arrives as a 1.5-degree tremble -- correctly
# smoothed into nothing. At one cycle over 0.8 s the same amplitude needs
# ~28 rad/s^2 and survives. A single deliberate gesture also simply reads
# better than a buzz.
FLOURISHES: dict[str, Move] = {
    #          pan   lift  elbow wrist roll  grip
    "shimmy": Move((0.00, 0.00, 0.00, 0.00, 0.45, 0.00), 1.0),
    "wag":    Move((0.13, 0.00, 0.00, 0.00, 0.00, 0.00), 1.0),
    "nod":    Move((0.00, 0.00, 0.00, 0.25, 0.00, 0.00), 1.0),
    "bob":    Move((0.00, 0.10, -0.13, 0.00, 0.00, 0.00), 1.0),
    "chomp":  Move((0.00, 0.00, 0.00, 0.00, 0.00, 0.60), 1.0),
    "droop":  Move((0.00, 0.13, 0.00, 0.10, 0.00, 0.00), 0.5),
    "strut":  Move((0.09, 0.00, 0.00, 0.00, 0.35, 0.00), 1.0),
}

# Which flourishes suit which outcome. Named by mood rather than by
# result so the game reads as a performer rather than a scoreboard.
MOODS: dict[str, tuple[str, ...]] = {
    "gloat": ("shimmy", "strut", "chomp"),      # it landed one
    "sulk": ("droop", "nod"),                   # it missed
    "smug": ("wag", "shimmy"),                  # its bluff worked
    "caught": ("nod", "droop"),                 # the human held through it
    "idle": ("bob", "chomp"),
}


class Flourish(Motion):
    """A wiggle about the pose it starts from, going nowhere.

    Joint space on purpose. There is no target and no IK, the amplitude
    envelope is zero at both ends, so it returns to exactly the
    configuration it began in and cannot drift toward the hand however it
    is interrupted or replayed. That is what makes it safe to run for fun
    on a machine that also swings at people.

    It is still a Motion, so the state machine can abandon it mid-swing
    the instant something real needs doing.
    """

    name = "flourish"

    def __init__(self, move: Move, duration: float = 0.8, speed: float = 2.0) -> None:
        super().__init__()
        self.move = move
        self.duration = max(duration, 1e-3)
        self.speed = speed
        self._q0: np.ndarray | None = None

    def _on_start(self, controller) -> None:
        self._q0 = controller.commanded.copy()

    def step(self, controller, dt) -> bool:
        if self.finished:
            return True
        s = min(self.elapsed / self.duration, 1.0)
        # A half-sine envelope over a whole number of swings: the offset
        # is zero at s=0 and s=1 whatever the amplitude, so the arm ends
        # where it started without needing to be driven back.
        envelope = np.sin(np.pi * s)
        swing = np.sin(2.0 * np.pi * self.move.cycles * s)
        amplitudes = np.asarray(self.move.amplitudes, float)
        controller._write(self._q0 + amplitudes * swing * envelope,
                          max_speed=self.speed, dt=dt)
        if self._complete(controller, self.duration):
            self.finished = True
        return self.finished


def flourish(mood: str, rng=None, duration: float = 0.8, speed: float = 2.0) -> Flourish:
    """A flourish suiting `mood`, picked at random so it does not stale.

    Repetition is what makes a performance stop being funny, and this one
    runs several times a minute.
    """
    names = MOODS.get(mood) or MOODS["idle"]
    pick = (rng.choice(len(names)) if rng is not None
            else np.random.randint(len(names)))
    return Flourish(FLOURISHES[names[int(pick)]], duration=duration, speed=speed)


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

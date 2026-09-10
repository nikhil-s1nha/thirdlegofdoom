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
from tlod.arm.profile import ProfileLimits
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

    def observe(self, tool_z: float) -> None:
        """Feed the measured tool height, if the caller already has one.

        Optional, and a no-op for every motion that does not care. It
        exists so a motion can know where the arm *is* without reading the
        servo bus itself: the caller driving it is often reading the pose
        already, and a second sync read per tick during a strike is not
        free on a half-duplex bus that drops the occasional transaction.
        """

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

    # What counts as the arm having stopped, and it is deliberately a long
    # time.
    #
    # The first version used 60 ms and ended every strike mid-descent:
    # dodges quit at 17-21 mm instead of reaching the 11 mm floor, hits
    # quit at 24-27 mm against a 28 mm hand instead of pressing into it.
    # The reason is that a min-jerk plan *decelerates to zero velocity by
    # design*, so near the end of the plan the arm is always crawling --
    # whether it has arrived or not. "Moving slowly" and "stopped" are
    # indistinguishable over a short window, and 60 ms is short.
    #
    # 150 ms is longer than the recovery creep in the bench trace, which
    # covered 1-2 mm per 26 ms sample: over 150 ms that is 6-10 mm of
    # travel, far past STILL_EPSILON, so a decelerating arm keeps
    # resetting the timer and only a genuinely blocked one runs it out.
    # Nothing waits this long on a clean dodge, because arrival
    # short-circuits it -- see step().
    STILL_EPSILON: float = 0.0015     # metres of drift that still counts as stopped
    STILL_DWELL: float = 0.15         # seconds of not moving before we believe it
    # How close to the commanded floor counts as having got there. Well
    # inside CollisionPlaneContactSensor's 4 mm margin, so a strike that
    # ends this way always scores as a dodge -- which is what it is.
    ARRIVE_EPSILON: float = 0.002

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
        # Stillness, fed by observe(). See _arm_still.
        self._still_ref: float | None = None
        self._moved_at: float | None = None
        self._seen_z: float | None = None
        self._floor_z: float | None = None
        # Why the descent ended, for the log line. Guessing at this is
        # what turned one wrong constant into a session of confusion.
        self.ended_because: str = ""

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
        self._still_ref = None
        self._moved_at = None
        self._seen_z = None
        self._floor_z = float(model.tool_pose(self._q1[:5]).z) if self.ok else None
        self.ended_because = ""
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

    def observe(self, tool_z: float) -> None:
        """Record where the paddle actually is. Called by whoever is stepping us."""
        now = time.perf_counter()
        self._seen_z = float(tool_z)
        if self._still_ref is None or abs(tool_z - self._still_ref) > self.STILL_EPSILON:
            # Re-base only on real movement. Holding the reference across
            # small ticks is the point: a slow creep never exceeds the
            # threshold on any single tick, but it does accumulate against
            # a fixed reference, and a creep is exactly what the arm does
            # while it recovers from overshooting the floor.
            self._still_ref = float(tool_z)
            self._moved_at = now

    def _arrived(self) -> bool:
        """Has the paddle got to the floor it was sent to?

        Unambiguous, unlike stillness, and it short-circuits the dwell --
        so an unobstructed strike ends the moment it lands rather than
        waiting to prove it has stopped. Only a blocked one pays for the
        wait, and a blocked one is not going anywhere.
        """
        if self._seen_z is None or self._floor_z is None:
            return False
        return self._seen_z <= self._floor_z + self.ARRIVE_EPSILON

    def _arm_still(self, controller) -> bool:
        """Has the paddle stopped moving?

        Not "has it arrived" -- on a hit it never arrives, it stalls
        against the hand and stops there. Stopping is the thing both
        outcomes have in common and the thing that separates them: stopped
        at the floor is a dodge, stopped above it is a hand.

        Falls back to `controller.settled()` when nobody is feeding
        observe(), which is the old behaviour: the *commanded* setpoint
        has stopped. That is a strictly worse question -- it is true while
        the paddle is still 16 mm up and travelling at 0.18 m/s -- but a
        caller that does not observe has nothing better to offer, and
        waiting for a stillness that can never be established would hang
        every strike until the timeout.
        """
        if self._moved_at is None:
            return controller.settled()
        return time.perf_counter() - self._moved_at >= self.STILL_DWELL

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
            # The plan has to have run out *and* the arm has to have
            # stopped. Waiting on the plan alone is what made every
            # downstream reading a measurement of the swing rather than of
            # what the swing hit: `Motion._complete` asks
            # `controller.settled()`, which is true once the commanded
            # setpoint stops changing, and the arm is nowhere near it then.
            # The timeout is still the backstop, because a paddle stalled
            # against something that yields slowly might never go quiet.
            plan_done = self.elapsed >= self.duration
            timed_out = self.elapsed >= self.duration + self.settle_timeout
            arrived, still = self._arrived(), self._arm_still(controller)
            if plan_done and (arrived or still or timed_out):
                self.ended_because = ("arrived" if arrived
                                      else "stopped" if still else "timed out")
                # Which of the three it was, because they mean different
                # things and the numbers alone cannot tell them apart. A
                # descent that "stopped" high is a paddle on a hand; one
                # that "timed out" is a gate that never fired, and that is
                # a bug rather than a round.
                log.debug("strike: %s at %s mm after %.0f ms (floor %s mm)",
                          self.ended_because,
                          "?" if self._seen_z is None else f"{self._seen_z * 1e3:.0f}",
                          self.elapsed * 1e3,
                          "?" if self._floor_z is None else f"{self._floor_z * 1e3:.0f}")
                if timed_out and not (arrived or still):
                    log.warning("strike: arm never went still and never arrived; "
                                "pressing anyway after %.0f ms", self.elapsed * 1e3)
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
    """One flourish: how far each joint swings, how often, and for how long.

    `cycles` is per joint when it is a tuple, because the good gestures are
    the ones where the joints do different things at once: a nod is the arm
    rising once while the wrist dips twice, and a chomp is the wrist coming
    up once while the gripper snaps. One cycle count for the whole arm can
    only make every joint do the same thing at the same time, which is what
    made the old table read as a set of twitches.

    `duration` is per move for the same reason. A spin wants to be slow --
    amplitude at fixed acceleration is bought with time -- and a chomp
    wants to be quick, and no single number is both. None means "whatever
    the caller's default is".
    """

    amplitudes: tuple[float, ...]      # radians, JOINT_NAMES order
    cycles: float | tuple[float, ...]  # one for all joints, or one each
    duration: float | None = None      # seconds; None = the caller's
    # Rectify the swing, per joint: `sin^2` instead of `sin`, so every
    # repetition goes the same way and `cycles` counts humps. This is what lets a joint repeat at
    # all when it has nothing behind HOME -- the gripper has 0.179 rad
    # before it is shut, so a plain two-cycle chomp spends every other
    # beat driving into the closed stop, and reads as one bite and a
    # stall. Rectified, three cycles are three bites. The envelope is
    # still zero at both ends, so it returns to where it started.
    oneway: bool | tuple[bool, ...] = False

    def offsets(self, s):
        """Joint offsets from the starting pose, at phase `s` in [0, 1].

        The one definition of the shape. `Flourish` steps it a sample at a
        time, the tests sweep it, and the review tool plots it -- all from
        here, because a test that reimplements the waveform is a test of
        the reimplementation.

        `sin(pi s)` is the envelope and is zero at both ends whatever else
        happens, which is what makes every flourish end where it began.
        """
        s = np.atleast_1d(np.asarray(s, float))[:, None]
        amps = np.asarray(self.amplitudes, float)
        cyc = np.broadcast_to(np.asarray(self.cycles, float), amps.shape)
        one = np.broadcast_to(np.asarray(self.oneway, bool), amps.shape)
        swing = np.where(one,
                         np.sin(np.pi * cyc * s) ** 2,
                         np.sin(2.0 * np.pi * cyc * s))
        return amps * np.sin(np.pi * s) * swing


# Measured on this rig, joint by joint, torque off. These are the real
# limits and they are nothing like the URDF's: `model.JOINT_LIMITS` gives
# wrist_roll +-2.7 rad where the motor has 0.513 rad in total, so a
# command far outside its travel is clamped by nothing and simply does not
# arrive. That is what a spin of 1.90 rad was -- commanded in full,
# reported by the encoder as 4 degrees, at every amplitude and every speed.
#
# Two of the six barely straddle HOME, and that is the fact that shapes
# this table. wrist_roll sits 0.054 rad off its lower stop, and the gripper
# 0.179 off its closed one, so on those joints a whole-cycle swing spends
# half its time driving into a hard stop. They get half cycles, which are
# `sin^2` and travel one way only.
MEASURED_TRAVEL: dict[str, tuple[float, float]] = {
    "shoulder_pan":  (-2.051, 1.450),
    "shoulder_lift": (-1.775, 1.827),
    "elbow_flex":    (-1.793, 1.584),
    "wrist_flex":    (-1.733, 1.800),
    # A full-turn joint whose *reading* wraps, which is not the same as a
    # joint with no travel. `Calibration.to_rad` is
    # `(counts - center) * RAD_PER_COUNT` with center 2839 of 4096, so
    # turning one way walks the count to 0 and the angle to -4.354 rad;
    # one more step wraps the count to 4095 and the angle jumps to +1.927.
    # Rotating the wrist through 360 degrees therefore reads 0 -> -4.354,
    # jump, +1.927 -> 0. Measuring the ends by hand and taking min and max
    # across that discontinuity is what produced "0.513 rad of travel" and
    # sent a spin to the wrong side of the arm.
    #
    # The usable span is what can be reached without crossing the wrap, and
    # it is wildly asymmetric: 4.354 rad going negative, 1.927 positive.
    # `to_counts` clips to 0..4095 in silence, so a command past either end
    # is not refused or logged, it simply stops arriving -- which is what
    # +1.90 rad was doing, 18 counts short of the boundary.
    "wrist_roll":    (-4.354, 1.927),
    "gripper":       (-0.179, 2.100),   # opens wide, shuts at once
}

# Sized to use that travel rather than a fraction of it, and checked at
# 1.5x amplitude, which is what loop jitter can add at 400 rad/s^2. The
# gestures are 31-74 degrees where the last table managed 8-22.
#
# Bigger costs time and there is no way around it. Peak joint speed for a
# swing of amplitude A at f Hz is proportional to A*f, so at the ~9 rad/s
# the servos were measured tracking, doubling a gesture doubles its
# duration. shimmy and jig are slower than they were for exactly that
# reason; bob and spin are quicker because they got shorter, not smaller.
#
# Direction is not symmetric and matters more than size. HOME leaves the
# tool 71 mm above the table, and positive shoulder_lift, elbow_flex and
# wrist_flex all drive it down, which is how the old bow and droop came to
# hit it. Anything large goes up: negative.
FLOURISHES: dict[str, Move] = {
    #             pan    lift  elbow  wrist   roll   grip
    # Everything at once: the wrist swinging up and rolling 74 degrees the
    # long way round, the shoulder turning into it, three quick bites on
    # the way through.
    #
    # 0.62 s rather than the 0.50 it was drawn at, and the gripper is what
    # sets that. Three rectified humps in half a second move faster than
    # the profile can follow, so it lags and then overshoots -- measured
    # 0.48 rad of travel against 0.24 asked, twice the gesture and none of
    # it predictable. The overshoot disappears at 0.60. Everything else in
    # here would happily run at 0.50.
    "spin": Move((0.45, -0.30, 0.00, -0.75, -1.30, 0.28),
                 (1.0, 0.5, 1.0, 0.5, 0.5, 3.0),
                 0.62, (False, False, False, False, False, True)),
    # Nothing in common with wag, which is what a shimmy needs: no pan at
    # all. It is the wrist -- 74 degrees of roll, three times -- with the
    # gripper snapping three times through it and the shoulder lifting
    # twice underneath. Sharing a joint with another gesture is what made
    # the last one read as a slower wag.
    "shimmy": Move((0.00, -0.30, 0.00, 0.00, 1.30, 0.90),
                   (1.0, 2.0, 1.0, 1.0, 1.5, 3.0),
                   1.40, (False, True, False, False, False, True)),
    # 49 degrees of pan with the elbow rising on each end of the swing --
    # one-way, so the elbow lifts twice rather than dipping at the table
    # in between.
    "wag": Move((1.12, 0.00, -0.35, 0.00, 0.00, 0.00),
                (1.0, 1.0, 2.0, 1.0, 1.0, 1.0),
                0.82, (False, False, True, False, False, False)),
    # Rise 32 degrees, then dip twice through 34. Both dips were always
    # there -- `sin^2(2 pi s)` has exactly two humps -- but at 1.20 s
    # against a 38-degree rise they arrived as one slow sag. Quicker, and
    # with less rise competing, they read as two nods. The wrist cannot go
    # much further than this in any case: HOME sits 0.5 rad up a joint
    # that stops at 1.8, so 0.80 is most of what is left once overswing is
    # allowed for.
    "nod": Move((0.00, -0.55, 0.00, 0.80, 0.00, 0.00),
                (1.0, 0.5, 1.0, 2.0, 1.0, 1.0),
                0.90, (False, False, False, True, False, False)),
    # Reaching much further up: 73 degrees of elbow against the 52 it had.
    "bob": Move((0.00, 0.70, -1.28, 0.00, 0.00, 0.00),
                (1.0, 0.5, 0.5, 1.0, 1.0, 1.0), 0.47),
    # Three 83-degree bites with the wrist held at the top of its travel.
    # Three big ones cannot also be quick: peak speed goes as amplitude
    # times cycles over duration, so a full jaw three times needs 1.55 s
    # against the 9 rad/s the servos track. Size won over speed here
    # because a chomp that does not open is not a chomp.
    "chomp": Move((0.00, 0.00, 0.00, -1.45, 0.00, 1.38),
                  (1.0, 1.0, 1.0, 0.5, 1.0, 3.0),
                  1.55, (False, False, False, False, False, True)),
    # Twitchy on purpose at 0.4 s. Two hops of 15 degrees is all that
    # duration buys -- the same trade as chomp, run the other way.
    "jig": Move((0.28, -0.25, 0.00, 0.00, 0.00, 0.00),
                (2.0, 0.5, 1.0, 1.0, 1.0, 1.0), 0.40),
}

# Which flourishes suit which outcome. Named by mood rather than by
# result so the game reads as a performer rather than a scoreboard.
MOODS: dict[str, tuple[str, ...]] = {
    "gloat": ("spin", "shimmy", "chomp"),   # it landed one
    "sulk": ("nod", "bob"),                 # it missed
    "smug": ("wag", "jig"),                 # its bluff worked
    "caught": ("nod", "wag"),               # the human held through it
    "idle": ("bob", "jig", "chomp"),
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

    # How close to the starting configuration counts as back there.
    HOME_EPSILON = 1e-3

    def __init__(self, move: Move, duration: float = 1.2, speed: float = 12.0,
                 accel: float = 400.0, jerk: float = 8000.0) -> None:
        super().__init__()
        self.move = move
        self.duration = max(move.duration or duration, 1e-3)
        self.speed = speed
        self.limits = ProfileLimits(max_speed=speed, max_accel=accel, max_jerk=jerk)
        self._q0: np.ndarray | None = None
        # Worst case the command is stranded a whole amplitude from home
        # when the plan runs out, and walking it back is rate-limited to
        # `speed`. The inherited 0.75 s would expire mid-return on exactly
        # the big gestures this branch exists for -- a 2.2 rad spin needs
        # 0.88 s at 2.5 rad/s -- so size the backstop to the move.
        reach = float(np.max(np.abs(np.asarray(move.amplitudes, float))))
        self.settle_timeout = max(Motion.settle_timeout,
                                  reach / max(self.speed, 1e-6) + 0.25)

    def _on_start(self, controller) -> None:
        self._q0 = controller.commanded.copy()

    def step(self, controller, dt) -> bool:
        if self.finished:
            return True
        s = min(self.elapsed / self.duration, 1.0)
        controller._write(self._q0 + self.move.offsets(s)[0],
                          dt=dt, limits=self.limits)
        if self.elapsed < self.duration:
            return False
        # Finished has to mean back where it started, which is not the
        # same as out of plan. The envelope is zero at s=1, but the arm
        # only gets there if the profile kept up, and these amplitudes
        # deliberately outrun the speed a flourish is allowed: a spin asks
        # 5.8 rad/s of a 2.5 rad/s budget, so the command trails the plan
        # by design. On a loop that misses ticks the last write lands
        # short, and `settled()` is then true of the stale offset the
        # profile came to rest at -- handing back an arm parked ten
        # degrees off the pose it promised to return to. Measured under
        # load that is exactly what happened. So ask where the arm is,
        # not whether it has stopped.
        strayed = float(np.abs(controller.commanded - self._q0).max())
        if strayed <= self.HOME_EPSILON and controller.settled():
            self.finished = True
        elif self.elapsed >= self.duration + self.settle_timeout:
            # The backstop still wins over wedging the state machine, but
            # it is a bug rather than a round, so it says so.
            log.warning("flourish: %.3f rad from where it started after %.0f ms; "
                        "abandoning the return", strayed, self.elapsed * 1e3)
            self.finished = True
        return self.finished


def flourish(mood: str, rng=None, duration: float = 1.2, speed: float = 12.0,
             accel: float = 400.0, jerk: float = 8000.0) -> Flourish:
    """A flourish suiting `mood`, picked at random so it does not stale.

    Repetition is what makes a performance stop being funny, and this one
    runs several times a minute.
    """
    names = MOODS.get(mood) or MOODS["idle"]
    pick = (rng.choice(len(names)) if rng is not None
            else np.random.randint(len(names)))
    return Flourish(FLOURISHES[names[int(pick)]], duration=duration, speed=speed,
                    accel=accel, jerk=jerk)


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

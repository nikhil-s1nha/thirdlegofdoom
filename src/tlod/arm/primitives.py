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


# Sized against both limits, which is what the earlier tables were not.
# Amplitude at fixed acceleration is bought with time: a swing of
# amplitude A at f Hz needs A(2*pi*f)^2, against 35 rad/s^2 here, and the
# servo's Goal_Acceleration register is one byte, so ~39 rad/s^2 is the
# ceiling no config can raise. The limit the old table missed is speed:
# `_write(max_speed=...)` is obeyed verbatim, and at the 2.5 rad/s it used
# to pass, eight of eleven moves were clipped -- a spin asked 2.20 rad and
# delivered 1.29, so the "126 degree" entry arrived as 74. Every move here
# is checked against 3.5 rad/s, which the rig already sustains through a
# strike, and against the acceleration ceiling.
#
# Direction matters more than it looks. HOME puts the tool 71 mm above the
# table and positive shoulder_lift, elbow_flex and wrist_flex all drive it
# *down*, so the old bow and droop -- lift +0.55 -- were aiming at the
# table and hitting it. Gestures that want to be big go up: negative.
#
# The roll and the gripper still carry what they can, because they are
# where theatre is cheapest -- a radian of wrist roll moves the gripper
# nowhere, where a radian of shoulder pan sweeps it across the table.
FLOURISHES: dict[str, Move] = {
    #                pan    lift  elbow  wrist   roll   grip
    # One big one-way roll, slow because that is what buys the size: 109
    # degrees, against the 74 the old 2.20-rad entry actually delivered.
    "spin": Move((0.00, 0.00, 0.00, 0.00, 1.90, 0.00), 0.5, 0.60),
    # Three roll swings of 36 degrees rather than one of 27, which is the
    # difference between a shimmy and a wobble.
    "shimmy": Move((0.00, 0.00, 0.00, 0.00, 0.62, 0.00), 1.5, 0.55),
    # Further and faster: 31 degrees of pan in a second, from 18 in 1.2 s.
    "wag": Move((0.58, 0.00, 0.00, 0.00, 0.00, 0.00), 1.0, 0.38),
    # Rise once, dip twice while up -- a nod from a standing start rather
    # than a wrist twitch. Needs the per-joint cycle counts.
    "nod": Move((0.00, -0.40, 0.00, 0.35, 0.00, 0.00),
                (1.0, 0.5, 1.0, 2.0, 1.0, 1.0), 0.46),
    # Out further and back, but only so far: this is the one gesture that
    # travels *toward* the table, and `safety.min_height` does not protect
    # it -- `clamp_pose` guards Cartesian commands and a flourish writes
    # joint space directly. Clearance here is this table's job and nothing
    # else's. These amplitudes bottom out 29 mm up; 0.50/-0.66 reached
    # 20 mm, which is not enough air for a gesture nobody is watching the
    # height of.
    # Smaller than the rest of this table wants to be, and deliberately.
    # At 400 rad/s^2 the arm can chase a target that jumped because the
    # loop missed a tick, so a gesture can overswing its amplitude by a
    # third -- measured at 1.35x on nod. Every other move here rotates or
    # goes up, where overswinging is ugly. This one travels at the table.
    # At 0.42/-0.55 it bottomed out 29 mm up, which a 1.4x overswing turns
    # into 11 mm. These amplitudes sit 42 mm up, and 29 mm even overswung.
    "bob": Move((0.00, 0.28, -0.38, 0.00, 0.00, 0.00), 1.0, 0.36),
    # Point up, then snap twice. The snap is acceleration-limited, not
    # speed-limited: at 35 rad/s^2 a 20-degree bite cannot come round
    # faster than about 0.75 s, and 39 is all the servo has.
    "chomp": Move((0.00, 0.00, 0.00, -0.30, 0.00, 0.38),
                  (1.0, 1.0, 1.0, 0.5, 1.0, 2.0), 0.50),
    # Two hops side to side while it rises once. The lift is a half cycle
    # on purpose: a whole one is a sine, which spends half its time going
    # *down*, and down from HOME is 71 mm of air and then the table -- at
    # a whole cycle this move passed 12 mm off it.
    "jig": Move((0.42, -0.34, 0.00, 0.00, 0.00, 0.00),
                (2.0, 0.5, 1.0, 1.0, 1.0, 1.0), 0.54),
}

# Which flourishes suit which outcome. Named by mood rather than by
# result so the game reads as a performer rather than a scoreboard.
#
# bow, droop, strut and flail are gone. The first two aimed at the table;
# the last two were smaller, muddier versions of spin and jig, spending
# shoulder travel -- the expensive kind -- to look like less.
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
        self._cycles = np.broadcast_to(
            np.asarray(move.cycles, float), np.shape(move.amplitudes)).copy()
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
        # A half-sine envelope over a whole number of swings: the offset
        # is zero at s=0 and s=1 whatever the amplitude, so the arm ends
        # where it started without needing to be driven back.
        envelope = np.sin(np.pi * s)
        swing = np.sin(2.0 * np.pi * self._cycles * s)
        amplitudes = np.asarray(self.move.amplitudes, float)
        controller._write(self._q0 + amplitudes * swing * envelope,
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

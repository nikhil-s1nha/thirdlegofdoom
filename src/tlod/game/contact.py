"""Did the slap land?

This is the question a camera cannot answer, and it is worth being
explicit about why. At the moment of contact the arm is directly between
an overhead camera and the contact point, occluding exactly the thing
that needs to be seen. And at 30 fps a frame is 33 ms, on an event that
decides the round and lasts a few milliseconds. Vision is the wrong
instrument.

So contact detection is an interface. Three implementations are wired
up, one per tier:

  GeometricContactSensor        tier A. Uses ground truth, because in
                                simulation ground truth exists and
                                pretending otherwise would only be
                                theatre.
  ProximityContactSensor        tier B, real hand and simulated arm.
                                Infers contact from tracked hand position
                                versus the virtual tool, because a
                                simulated arm is never blocked by
                                anything and there is nothing else to
                                read. Honest about being an estimate.
  CollisionPlaneContactSensor   tier C, the real arm. The strike commands
                                a floor below the hand; a paddle that
                                stopped short of it was blocked, one that
                                reached it was not. Encoders at both
                                ends.

The first two exist so the game is fully playable before hardware does.
The third is what runs on the real arm, and `tlod play --real` offers no
way to select anything else.

    !! ONLY ONE THING BELOW THIS POINT IS REACHABLE !!

`ServoPressContactSensor` is, behind an explicit `--contact press`, and
its docstring says why it came back. `ServoLoadContactSensor` and
`SerialContactSensor` are kept for their measurements, not for their
behaviour. Each one's docstring records what it read on this rig, and
between them they are the argument for the sensor that replaced them --
delete them and the next person re-runs the same three experiments. They
are not reachable from the CLI. Do not wire one back in without a number
that beats the encoders.

The short version of why torque lost. Measured across nothing / a book /
a hand, peak load *during* the swing read 0.330 / 0.326 / 0.350: a rigid
book, the easiest thing there is to feel, landed *between* the other two,
because the arm braking its own mass reaches the torque cap in every run,
empty table included. Held still at the bottom the same three read 0.001
/ 0.038 / 0.037 -- which does separate them, but only after ~300 ms of
the servos stalled at their torque limit, and sustained stall current is
what a 5 A supply and six servos have least of. `Present_Current` (addr
69) separated the same three conditions by 0.006 A, one 6.5 mA
quantisation step, smaller than the jitter within a single run. A piezo
disc on a microcontroller would time an impact to microseconds, and is a
whole extra board for a game scored per round rather than per
millisecond.

The encoders answer the same question in 120 ms, with no bus traffic the
control loop was not already doing, and are right the moment the arm
stops.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np


log = logging.getLogger(__name__)


@dataclass(slots=True)
class ContactEvent:
    stamp: float
    source: str
    strength: float = 1.0


class ContactSensor(abc.ABC):
    """Reports at most one contact per arming."""

    def arm(self, blank_for: float | None = None) -> None:
        """Ready for the next strike; discard anything pending.

        `blank_for` is how long after arming to ignore, for sensors that
        cannot tell contact from the strike's own launch. A real contact
        cannot happen in the first part of a drop, because the paddle has
        not reached the hand yet.
        """

    @abc.abstractmethod
    def poll(self, **kwargs) -> ContactEvent | None: ...

    def close(self) -> None:
        pass


class GeometricContactSensor(ContactSensor):
    """Ground truth for simulation: the tool got close enough to the hand."""

    def __init__(self, radius: float = 0.045, plane_tolerance: float = 0.02) -> None:
        self.radius = radius
        self.plane_tolerance = plane_tolerance
        self._fired = False

    def arm(self, blank_for: float = 0.0) -> None:
        self._fired = False

    def poll(self, tool_xyz=None, hand_xyz=None, **kwargs) -> ContactEvent | None:
        if self._fired or tool_xyz is None or hand_xyz is None:
            return None
        tool = np.asarray(tool_xyz, float)
        hand = np.asarray(hand_xyz, float)
        horizontal = float(np.linalg.norm(tool[:2] - hand[:2]))
        vertical = float(tool[2] - hand[2])
        if horizontal <= self.radius and -self.plane_tolerance <= vertical <= self.plane_tolerance:
            self._fired = True
            return ContactEvent(time.perf_counter(), "geometric",
                                strength=1.0 - horizontal / self.radius)
        return None


class ProximityContactSensor(GeometricContactSensor):
    """Tier B: same test, but the hand position is an estimate, not truth.

    Kept as a distinct class so that a result obtained this way is never
    mistaken for a measurement, and near-misses will be scored wrongly in
    both directions. Not by as much as this used to claim, though: the
    "several centimetres of uncertainty" here was pessimism from before
    anything was measured, and vision resolves 2.5 mm on this rig against
    a hand that jitters 2.4-5 mm. What the estimate is really short of is
    not precision but *currency*, which is the whole of what follows.

    The geometry is the same as `GeometricContactSensor`'s and it is the
    right geometry here: in tier B the arm is simulated, so the paddle
    passes *through* a hand rather than being stopped by it, and the only
    question the encoders can answer is whether the hand was inside the
    band the paddle swept. What is not the same is *when* it is allowed
    to ask.

    Asking on every tick of the descent is what made this sensor score
    every committed strike as a hit, and it is worth spelling out because
    the failure looks like a threshold problem and is not one. The strike
    is aimed where the hand was at the moment of commit, so `horizontal`
    starts at ~0 and the test turns entirely on `vertical`. That enters
    the +/-`plane_tolerance` band a third of the way down -- ~105 ms into
    a ~310 ms strike -- and the hand estimate being compared is itself a
    pipeline latency old, so the round was decided on where the hand had
    been ~200 ms before the paddle would actually land. A human's dodge
    starts at ~65 ms and takes another ~150 ms to clear. None of it was
    ever in the data.

    So the reading is gated on the paddle having *arrived*, the same way
    `ServoPressContactSensor` gates on `pressing`, and for the same
    reason: the motion knows what phase it is in and nothing else does.
    Arrival is a latch rather than a live flag, because the useful reads
    keep coming after the press ends -- `HandSlapGame` freezes the tool
    at the bottom and keeps polling through the retract, which is how the
    frames covering the moment of contact get to be part of the verdict
    instead of arriving after it.

    Without a `pressing` kwarg this sensor never fires, so a caller that
    forgets it scores every round as a dodge rather than inventing hits.
    `StrikeLimits.press_hold` must be non-zero, or the strike never
    presses and nothing ever arrives; `cmd_play` sizes it.
    """

    def __init__(self, radius: float = 0.06, plane_tolerance: float = 0.035) -> None:
        super().__init__(radius, plane_tolerance)
        self._arrived = False
        # What the last judged round looked like, for `report()`. Same
        # reason CollisionPlaneContactSensor keeps one: a verdict on its
        # own cannot distinguish "the hand moved" from "the hand estimate
        # moved", and those have different fixes.
        self.last: tuple[float, float] | None = None   # horizontal, vertical

    def arm(self, blank_for: float | None = None) -> None:
        # `blank_for` is accepted and ignored -- arrival replaces it. A
        # blanking window is a guess at how long the launch takes; this
        # is the motion saying it has stopped travelling.
        super().arm()
        self._arrived = False
        self.last = None

    def report(self) -> str:
        """What the last judged round looked like, in millimetres."""
        if self.last is None:
            return "no reading (the paddle never reached the bottom)"
        horizontal, vertical = self.last
        return (f"tracked hand {horizontal * 1e3:.0f} mm to the side and "
                f"{-vertical * 1e3:+.0f} mm above the paddle "
                f"(needs {self.radius * 1e3:.0f} across, "
                f"{self.plane_tolerance * 1e3:.0f} deep)")

    def poll(self, pressing: bool = False, tool_xyz=None, hand_xyz=None,
             **kwargs) -> ContactEvent | None:
        self._arrived = self._arrived or bool(pressing)
        if not self._arrived:
            return None
        if not self._fired and tool_xyz is not None and hand_xyz is not None:
            tool = np.asarray(tool_xyz, float)
            hand = np.asarray(hand_xyz, float)
            self.last = (float(np.linalg.norm(tool[:2] - hand[:2])),
                         float(tool[2] - hand[2]))
        event = super().poll(tool_xyz=tool_xyz, hand_xyz=hand_xyz, **kwargs)
        if event is not None:
            event = ContactEvent(event.stamp, "proximity", event.strength)
        return event


class ServoLoadContactSensor(ContactSensor):
    """Detect contact from the servos' own torque feedback.

    UNUSED. Kept for the measurement, not the behaviour: across nothing /
    a book / a hand the peak load during a swing read 0.330 / 0.326 /
    0.350, putting a rigid book *between* the other two. That number is
    the reason nothing reads torque mid-swing any more.

    When the paddle meets a hand, the joints resisting the motion see
    their load rise sharply. The STS3215 reports this on Present_Load,
    and `FeetechArm.read()` already pulls it in the same bus transaction
    as position and speed, so this is free: no piezo, no microcontroller,
    no wiring.

    Only the pitch joints are watched. Shoulder pan and wrist roll are
    roughly orthogonal to a downward strike and mostly report noise.

    The baseline is captured at `arm()` rather than assumed, because
    resting load depends on the arm's configuration -- an extended arm
    holds more of its own weight than a folded one, and a fixed threshold
    would fire on posture instead of on contact.

    Two things here are traps rather than details.

    The threshold has less room than it looks. `Strike` drops the servo
    torque limit to `StrikeLimits.torque_limit` (350 of 1000) for the
    duration of the swing so the arm yields on contact, and a servo
    cannot report more load than it is allowed to produce. The entire
    detectable range during a strike is therefore the gap between the
    hover pose's resting load and that cap: if a stretched-out hover
    already sits at 0.25, a 0.12 threshold has 0.10 of headroom and can
    never fire. `peak_rise` records the largest rise ever seen so the
    first hardware session can set the threshold from a measurement
    instead of from this guess.

      !! THRESHOLD UNVERIFIED AGAINST HARDWARE !!

    Reads are wrapped because `poll` runs on the control thread inside a
    committed strike. A transient sync-read failure on the half-duplex
    bus is a known event on this rig, and an exception escaping here
    reaches `RobotApp._control_loop`, which answers a failed policy tick
    by e-stopping -- freezing the arm mid-swing, directly above the hand
    it was aiming at. A dropped reading costs one round scored as a
    dodge. That is the cheaper failure by a wide margin.
    """

    # shoulder_lift, elbow_flex, wrist_flex
    STRIKE_JOINTS: tuple[int, ...] = (1, 2, 3)

    def __init__(
        self,
        state_source,
        threshold: float = 0.12,
        joints: tuple[int, ...] | None = None,
        blank_for: float = 0.10,
    ) -> None:
        self.state_source = state_source
        self.threshold = threshold
        # How long after arming to ignore, when the caller does not say.
        # A caller that knows the strike duration should pass a fraction
        # of it; this default is for one that does not.
        self.blank_for = blank_for
        self.joints = list(self.STRIKE_JOINTS if joints is None else joints)
        self._baseline: np.ndarray | None = None
        self._blank_until = 0.0
        self._armed_at = 0.0
        self._fired = False
        # Diagnostics, for tuning the threshold on the first real session.
        self.peak_rise = 0.0
        self.read_failures = 0

    def peak_summary(self) -> str:
        """End-of-run line. A fraction of rated torque, not a height."""
        return (f"peak held load {self.peak_rise:.3f} of rated torque "
                f"(needed {self.threshold:.3f}, after "
                f"{self.settle * 1000:.0f} ms pressing)")

    def report(self) -> str:
        """What the last judged round actually looked like."""
        if self.last is None:
            return "no reading (the paddle never pressed long enough)"
        return (f"held load {self.last:+.3f} of rated torque "
                f"(needs {self.threshold:.3f}) after {self.settle * 1000:.0f} ms")

    def _load(self) -> np.ndarray | None:
        """Watched joints' load magnitudes, or None if there is no reading."""
        try:
            state = self.state_source()
        except Exception:
            self.read_failures += 1
            return None
        if state.load is None:
            return None
        return np.abs(np.asarray(state.load, float)[self.joints])

    def arm(self, blank_for: float | None = None) -> None:
        self._fired = False
        self._baseline = None
        self._armed_at = time.perf_counter()
        window = self.blank_for if blank_for is None else max(blank_for, 0.0)
        self._blank_until = self._armed_at + window

    def poll(self, **kwargs) -> ContactEvent | None:
        if self._fired or time.perf_counter() < self._blank_until:
            return None
        current = self._load()
        if current is None:
            return None
        if self._baseline is None:
            # The baseline is taken here, on the first poll after the
            # blanking window, rather than at arm().
            #
            # Taking it while hovering does not work: the joints have to
            # produce torque to accelerate the arm downward, and at
            # 35 rad/s^2 that rise clears any workable threshold on the
            # first tick of the strike. Observed on hardware -- the swing
            # was aborted about a millimetre in, every time, and read as
            # the arm failing to move rather than as a false contact.
            #
            # Measured against the descent instead, the reference is the
            # load of an arm already travelling, and what remains above
            # it is the hand.
            self._baseline = current
            return None
        rise = float(np.max(current - self._baseline))
        self.peak_rise = max(self.peak_rise, rise)
        if rise < self.threshold:
            return None
        self._fired = True
        # When and how hard, because "it hit instantly" and "it hit on
        # contact" are indistinguishable at one-second log resolution,
        # and the difference is the whole diagnosis: a rise this side of
        # the blanking window is the arm's own launch, not a hand.
        log.debug("contact: rise %.3f (threshold %.3f) at %.0f ms after arming; "
                  "baseline %s, now %s",
                  rise, self.threshold, (time.perf_counter() - self._armed_at) * 1e3,
                  np.round(self._baseline, 3), np.round(current, 3))
        return ContactEvent(time.perf_counter(), "servo_load", strength=min(rise, 1.0))


class CollisionPlaneContactSensor(ContactSensor):
    """Hit if the paddle stopped inside the band where the hand is.

    The rule, in full: the strike commands a floor *below* the hand. If
    nothing is there the paddle reaches that floor. If a hand is there the
    paddle stops on top of it -- somewhere in the band between the floor
    and the hand's own height. So a paddle that ends up inside that band
    was stopped by something, and a paddle that reaches the floor was not.
    Both positions come from the encoders, so neither is late and neither
    is filtered.

    That is all of it. The three earlier attempts in this file were all
    the same idea measured worse: load and current try to infer the block
    from torque, which Torque_Limit clamps and the swing's own braking
    swamps, and proximity asks the camera where the hand is at the one
    moment the arm is between the camera and the hand.

    Two ways it can be wrong, both worth naming because they are the ones
    to check when it misreports:

      * The floor is above the hand. Then an untouched paddle and a
        touched one stop in the same place and everything reads as a
        dodge. `StrikeLimits.press_depth` is what puts the floor below,
        and `safety.min_height` can silently clamp it back up.
      * The hand compresses to the floor. Flesh is soft and the paddle
        keeps pushing for `press_hold`; if it squashes the last few
        millimetres out of a palm, the band closes.

    `report()` exists for exactly those two, and the game logs it every
    round: guessing at which one is happening is what this class replaced.
    """

    def __init__(
        self,
        floor_source,
        margin: float = 0.001,
        settle: float = 0.12,
        band_fraction: float = 0.15,
        still_epsilon: float = 0.0006,
        still_ticks: int = 4,
        max_wait: float = 0.15,
        floor_sag: float = 0.0,
    ) -> None:
        # () -> commanded floor height, metres. Where the paddle actually
        # reached comes in on `tool_xyz`, which the caller has already read
        # this tick.
        #
        # It used to read that itself, which meant two bus transactions per
        # tick during a strike where one would do. On a half-duplex servo
        # bus that is not free: a sync read failed mid-strike, the control
        # loop e-stopped, and the arm froze above the hand. The commanded
        # position is cached in the controller and costs nothing.
        self.floor_source = floor_source
        # The absolute floor under `band_fraction`, for a band too thin
        # for a fraction to mean anything. It is not the working
        # threshold; `band_fraction` is, and on any sane geometry it wins.
        #
        # 1 mm, down from 2. The measured clusters at press_depth 8 mm are
        # +2..+4 mm short for a hand and -0..-15 for an empty table, so the
        # gap between them is the 2 mm from 0 to +2 and a threshold of
        # 2 mm sat on its upper edge -- a hit landing at exactly +2 was
        # decided by the comparison being strict. 1 mm centres it, with a
        # millimetre of room on each side.
        #
        # That the gap is only 2 mm wide is the real problem and this does
        # not fix it; see `arm.strike_torque`, which changes how hard the
        # paddle leans on the palm and so how far apart the clusters sit.
        self.margin = margin
        # The threshold that actually decides, as a fraction of the band
        # between the floor and the hand.
        #
        # Scale-free, which is the point. Absolute millimetres have to be
        # re-tuned whenever press_depth, the torque limit or the arm's
        # load changes, and each of those moves where the paddle ends up.
        #
        # This was 0.5, measured against an arm whose descent quit on
        # first touch: an empty table left the paddle 24% of the way up
        # the band and a hand stopped it at 85%, so half-way separated
        # them. Making the descent wait for the arm moved *both* clusters
        # down about 7 mm, because the paddle now presses into a hand
        # rather than resting on it. Measured again on the same rig,
        # floor 11 mm and hand plane 28 mm:
        #
        #     empty table   6-10 mm   -5 to -1 mm short   (below the floor)
        #     a hand       17-23 mm   +6 to +12 mm short
        #
        # The gap is 10-17 mm, so the threshold belongs at about 2.5 mm --
        # 15% of the band, with ~3.5 mm of room on either side. At 50% it
        # sat above most of the hand cluster and scored real hits as
        # dodges, which is the opposite of the failure it was introduced
        # to fix and a good reminder that a threshold calibrated against
        # one version of the motion does not survive changing the motion.
        #
        # `hand_xyz` is a fixed plane from `vision.hand_height`, not a
        # depth measurement -- one camera cannot get depth, so the pixel
        # ray is intersected against an assumed palm height. That makes it
        # a stable reference rather than a noisy one. Without it this
        # falls back to `margin` alone.
        self.band_fraction = band_fraction
        # Long enough for an unobstructed press to have arrived. This is
        # now the *minimum* wait, not the moment of judgement -- see
        # `still_epsilon`.
        self.settle = settle
        # The reading is taken when the paddle has actually stopped
        # moving, not at a fixed time after the press starts.
        #
        # This is the difference between a signal and noise on this rig.
        # A bench trace at the game's own geometry: the paddle is still
        # descending 120 ms into the press and does not settle until
        # ~180-220 ms, and `strike_bench` prints the gap in as many
        # words -- "settled at 55 mm, 182 ms into the hold / the game
        # would read 59 mm at 120 ms, +4 mm off the settled value."
        #
        # 4 mm of "still falling" error, against a hit/dodge signal that
        # is 2-4 mm wide. So the error is the whole signal: a dodge read
        # early looks like a hit, and a hit read late looks like a dodge,
        # which is exactly the coin-flip behaviour that made the verdicts
        # look arbitrary while every threshold in sight was correct.
        #
        # `Strike` already had this lesson -- the descent ends when the
        # arm stops rather than when the asking stops -- and the sensor
        # judging it did not.
        self.still_epsilon = still_epsilon
        self.still_ticks = still_ticks
        # A blocked paddle leaning on a palm can creep for a long time,
        # so there has to be a point where a verdict is given anyway.
        # A round judged on the timeout says so in `report()`.
        #
        # 150 ms, and the ceiling on it is the power supply rather than
        # anything about detection. `press_hold` is sized from
        # `hold_needed()`, so this lands as 350 ms of six servos stalled
        # at their torque limit every strike -- against a 200 ms default,
        # and against the afternoon `press_hold` went to 450 ms and the
        # bus started dropping transactions. Do not raise it without
        # watching `tlod power`.
        #
        # It is enough because the slow case is the *empty* one: a
        # blocked paddle stops when it meets the hand, while an
        # unobstructed one overshoots its floor and creeps back for
        # ~200 ms after the press starts. Measured on the bench, settling
        # completes 180-220 ms in, and `settle` already covers the first
        # 120 of that.
        self.max_wait = max_wait
        self._recent: list[float] = []
        self._judged_moving = False
        # How far *below* its commanded floor an unobstructed strike
        # actually comes to rest, metres. Subtracted from the floor to
        # get the height this sensor measures against.
        #
        # This is the difference between a working sensor and one that
        # scores every round a dodge, and it is not a fudge factor -- the
        # commanded floor is simply not a place the paddle ever goes. A
        # strike is ballistic on purpose: `goto_pose` re-aims at what the
        # encoders report and `Strike` deliberately does not, because a
        # second pass is a second, slower descent. So the strike carries
        # the arm's full static sag, uncorrected, and it is large.
        # Measured on the bench at 0.36/0.165 with nothing under the
        # paddle, commanded floor 20 mm:
        #
        #     settled at 11, 8, 9, 9 mm     -- 9-12 mm below the command,
        #                                      held steady 300+ ms
        #
        # against a hand in the same geometry settling at 13-16 mm. The
        # signal is a real 2-5 mm and both clusters sit ~10 mm under the
        # floor, so a threshold placed just above the floor is above
        # *both* of them and everything reads as a dodge. Which is
        # exactly what a whole session of "it hit me and said dodged"
        # looked like.
        #
        # It has to be measured, not modelled: it is the same
        # unobservable droop `ArmController.compensate` handles for
        # Cartesian targets, but that model was fitted at hover height
        # and the arm is in a very different pose at the floor.
        # `strike_bench` over an empty table prints the number to put in
        # `arm.strike_sag`. Re-measure it after anything that changes the
        # strike -- see the rule at the top of docs/hit-detection.md.
        self.floor_sag = floor_sag
        self._pressing_since: float | None = None
        self._fired = False
        self.last: tuple[float, float, float] | None = None   # reached, floor, hand
        # How far to the side of the tracked hand the paddle came down,
        # metres, for the last judged round. See MISS_RADIUS.
        self.last_lateral: float | None = None
        # How far the paddle came down from where the round was *aimed*,
        # and how far the hand has moved since. These are two different
        # failures and `last_lateral` alone cannot separate them: a big
        # number there means either the strike went somewhere the hand
        # never was, or the hand left after the strike was committed --
        # which is a dodge, and the whole point of the game.
        self.last_aim: float | None = None
        self.last_hand_moved: float | None = None
        # Diagnostics: the largest settled shortfall seen, in metres.
        self.peak_rise = 0.0
        self.read_failures = 0
        # Rounds where the paddle landed further than MISS_RADIUS from the
        # hand. Those are not dodges and counting them as such is what
        # made a whole aiming problem look like a threshold problem.
        self.misses = 0

    # Beyond this far from the tracked hand, horizontally, the paddle
    # cannot have touched the palm -- so "it reached its floor" says
    # nothing about whether the human dodged.
    #
    # This exists because the two failures are indistinguishable in the
    # round line and have opposite fixes. A dodge means detection worked
    # and the human was quick. A miss means the strike was aimed
    # somewhere the hand was not, and no threshold anywhere repairs it.
    # Measured on this rig: `where_is_my_hand --truth` reports a hand at
    # a true (250, 0) mm as (264..267, +19) -- a standing ~24 mm bias --
    # and `vision-check` puts accuracy against kinematics at 24 mm mean
    # 19. A palm is about 90 mm across, so a 24 mm bias lands the paddle
    # near the edge of the hand and sometimes past it, which is exactly
    # the observed pattern of one session hitting 7 of 14 and the next
    # hitting 0 of 7 with nothing changed in between.
    #
    # 50 mm is half a palm plus a little: inside it a stopped paddle is
    # credible, outside it the round is not evidence either way.
    MISS_RADIUS: float = 0.05

    @property
    def threshold(self) -> float:
        """Alias, so callers that tune a threshold reach the right knob."""
        return self.margin

    def _threshold(self, floor: float, hand: float) -> float:
        """How far above the floor counts as blocked, for this round.

        A fraction of the floor-to-hand band where there is a hand plane
        to measure against, and the bare margin where there is not.
        """
        if not np.isfinite(hand) or hand <= floor:
            return self.margin
        return max(self.margin, self.band_fraction * (hand - floor))

    def arm(self, blank_for: float | None = None) -> None:
        # `blank_for` is accepted and ignored -- `pressing` replaces it.
        self._fired = False
        self._pressing_since = None
        self.last = None
        self.last_lateral = None
        self.last_aim = None
        self.last_hand_moved = None
        self._recent = []
        self._judged_moving = False

    def settled_floor(self, floor: float) -> float:
        """Where an unobstructed strike from `floor` actually ends up.

        The commanded floor less the measured sag. This, not the
        commanded floor, is what "stopped short" is short *of*.
        """
        return floor - self.floor_sag

    def hold_needed(self) -> float:
        """How long the paddle must stay down for this to get a reading.

        `settle` is only the minimum now, so a caller sizing `press_hold`
        from it alone retracts the arm out from under a round that was
        still waiting for the paddle to stop.
        """
        return self.settle + self.max_wait

    def _still(self) -> bool:
        """Has the paddle stopped moving, as far as the encoders can tell?"""
        if len(self._recent) < self.still_ticks:
            return False
        window = self._recent[-self.still_ticks:]
        return (max(window) - min(window)) <= self.still_epsilon

    def peak_summary(self) -> str:
        """End-of-run line. In millimetres, because this is not a torque."""
        need = self.margin if self.last is None else self._threshold(self.last[1], self.last[2])
        line = (f"peak shortfall {self.peak_rise * 1e3:.0f} mm "
                f"(needed {need * 1e3:.0f} mm, {self.band_fraction:.0%} of the band)")
        if self.misses:
            line += (f"; {self.misses} round(s) landed more than "
                     f"{self.MISS_RADIUS * 1e3:.0f} mm from the hand and are not "
                     f"evidence either way")
        return line

    def report(self) -> str:
        """What the last judged round actually looked like, in millimetres."""
        if self.last is None:
            return "no reading (the paddle never settled at the bottom)"
        reached, floor, hand = self.last
        rest = self.settled_floor(floor)
        short = reached - rest
        need = self._threshold(floor, hand)
        floors = (f"floor {floor * 1e3:.0f} mm"
                  if not self.floor_sag
                  else f"floor {floor * 1e3:.0f} mm (rests {rest * 1e3:.0f})")
        line = (f"paddle stopped {reached * 1e3:.0f} mm, {floors}, "
                f"hand {hand * 1e3:.0f} mm -> {short * 1e3:+.0f} mm short "
                f"(needs {need * 1e3:.0f})")
        # The half of the round the heights cannot show. Without this a
        # paddle that came down 60 mm wide of the hand reads exactly like
        # a paddle the human pulled away from, and the two have nothing
        # in common except the word "dodged".
        # Without `aimed_at` there is nothing to separate the two cases
        # with, so fall back to the raw offset and say so in the weaker
        # form. `strike_bench` and the tests poll this way.
        offset = self.last_aim if self.last_aim is not None else self.last_lateral
        if offset is not None and offset > self.MISS_RADIUS:
            # The paddle went somewhere the hand was not when the round
            # was committed. Nothing about this round is evidence.
            where = "from where it aimed" if self.last_aim is not None else "from the hand"
            line += (f"  [MISSED: came down {offset * 1e3:.0f} mm {where}, so this "
                     f"round says nothing about hit or dodge -- check the aim, "
                     f"not the threshold]")
        elif self.last_hand_moved is not None and self.last_hand_moved > self.MISS_RADIUS:
            # Aimed correctly and the hand left. That is a dodge, and it
            # is the one large-offset case that is working as intended.
            line += (f"  [hand moved {self.last_hand_moved * 1e3:.0f} mm since the "
                     f"strike was committed -- a real dodge]")
        elif self.last_lateral is not None:
            line += f"  [{self.last_lateral * 1e3:.0f} mm off the hand centre]"
        if self._judged_moving:
            line += ("  [still moving when judged -- the paddle never stopped "
                     "within press_hold, so this height is a snapshot of a "
                     "descent, not a resting place]")
        return line

    def poll(self, pressing: bool = False, tool_xyz=None, hand_xyz=None,
             aimed_at=None, **kwargs) -> ContactEvent | None:
        if self._fired or tool_xyz is None:
            return None
        if not pressing:
            self._pressing_since = None
            return None
        now = time.perf_counter()
        if self._pressing_since is None:
            self._pressing_since = now
            return None
        if now - self._pressing_since < self.settle:
            return None
        try:
            floor = self.floor_source()
        except Exception:
            self.read_failures += 1
            return None
        reached = float(tool_xyz[2])
        # Wait for the paddle to stop before judging anything. Reading a
        # fixed time into the press reads a height the paddle is still
        # falling *through*, and the descent is ~4 mm from done at 120 ms
        # -- the whole width of the hit/dodge signal. See `still_epsilon`.
        self._recent.append(reached)
        if len(self._recent) > self.still_ticks:
            del self._recent[0]
        if not self._still():
            if now - self._pressing_since - self.settle < self.max_wait:
                return None
            # Out of time. Judge anyway rather than defaulting to a dodge,
            # but mark it so the round line cannot be read as a settled
            # measurement.
            self._judged_moving = True
        hand = float(hand_xyz[2]) if hand_xyz is not None else float("nan")
        self.last = (float(reached), float(floor), hand)
        # Horizontal distance from the paddle to the tracked hand. Cheap
        # -- both positions are already in hand -- and it is the only
        # thing that separates a dodge from a strike aimed off the hand.
        lateral = None
        if hand_xyz is not None and len(hand_xyz) >= 2:
            lateral = float(np.linalg.norm(
                np.asarray(tool_xyz, float)[:2] - np.asarray(hand_xyz, float)[:2]))
        # Split that into the two things it confounds. `aimed_at` is
        # where the hand was when the round was committed, so the paddle
        # against it is the aiming error, and the hand against it is how
        # far the human has moved since -- which is the dodge.
        aim = moved = None
        if aimed_at is not None and len(aimed_at) >= 2:
            goal = np.asarray(aimed_at, float)[:2]
            aim = float(np.linalg.norm(np.asarray(tool_xyz, float)[:2] - goal))
            if hand_xyz is not None and len(hand_xyz) >= 2:
                moved = float(np.linalg.norm(np.asarray(hand_xyz, float)[:2] - goal))
        # A miss is an *aiming* failure, so count it off `aim` where that
        # is known. Counting it off `lateral` charged every dodge as a
        # miss, since a hand that leaves is far away by definition.
        offset = aim if aim is not None else lateral
        was_miss = (self.last_lateral is None and self.last_aim is None
                    and offset is not None and offset > self.MISS_RADIUS)
        self.last_lateral = lateral
        self.last_aim = aim
        self.last_hand_moved = moved
        # Against where an unobstructed strike actually rests, not
        # against the commanded floor it never reaches. See `floor_sag`.
        short = float(reached) - self.settled_floor(float(floor))
        self.peak_rise = max(self.peak_rise, short)
        if was_miss:
            self.misses += 1
        if short < self._threshold(float(floor), hand):
            return None
        self._fired = True
        log.debug("collision: %s", self.report())
        return ContactEvent(now, "collision_plane", strength=min(short / 0.02, 1.0))


# The name this was published under for one commit. Kept so a config or
# script pinned to it does not break; the class above is the same sensor
# with the band stated explicitly and a report() worth reading.
ToolHeightContactSensor = CollisionPlaneContactSensor


class ServoPressContactSensor(ContactSensor):
    """Detect contact from what the arm is still pushing against once it stops.

    OPT-IN, behind `--contact press`. It works -- 0.001 / 0.038 / 0.037
    held still is a real separation, and this docstring is the record of
    how that was found -- but it costs ~300 ms of six servos stalled at
    their torque limit every single strike, and sustained stall current is
    what a 5 A supply has least of; the bus started dropping transactions
    the afternoon press_hold went to 450 ms. `CollisionPlaneContactSensor`
    answers the same question from the encoders in 120 ms with no extra
    bus traffic, so it stays the default.

    What brought this back from UNUSED is reach. The geometric sensor asks
    how far short of its floor the paddle stopped, and at 400 mm that
    question loses its answer: an empty table settles +1..+2 mm short and a
    hand +3..+5, one millimetre apart. Held load at the same reach read
    0.008 empty against 0.040-0.052 on a hand, five runs each. Same rig,
    same strike, and only one of the two still separates.

    This is the sensor that works on this arm, and it exists because the
    obvious one does not. `ServoLoadContactSensor` reads the same register
    during the swing, and measured across nothing / a book / a hand it
    returned 0.330 / 0.326 / 0.350 -- a rigid book between the other two.
    Load climbs monotonically to its ceiling in every run, empty table
    included, because the arm is accelerating and braking its own mass.
    The swing is one long transient and nothing read during it is about
    what was hit.

    Held still at the bottom, the same three conditions read 0.001 /
    0.038 / 0.037. That is the whole idea: wait for the arm to stop, wait
    for the servo's load filter to forget the swing, and then read what
    torque is still being spent. In steady state the only thing left to
    spend it on is whatever is under the paddle.

    The reading is gated on `pressing`, which `Strike` sets when it has
    finished travelling and is leaning on the floor, rather than on any
    speed threshold of this sensor's own. That is deliberate. The obvious
    alternative -- watch `dq` and call the arm still when it drops below
    some number -- needs a number nobody has measured, and the arm is also
    briefly stationary at the *start* of a drop, before the profile has
    accelerated it. `ArmController.settled()` is no better: on the
    measured traces it reported settled while the paddle still had 10 mm
    to travel. The motion knows what phase it is in; nothing else does.

    Without a `pressing` kwarg this sensor never fires, so a caller that
    forgets it scores every round as a dodge rather than inventing hits.

    `Strike.press_hold` must exceed `settle`, or the arm retracts before
    this ever gets a reading. `cmd_play` checks and warns.

    Reads are wrapped because `poll` runs on the control thread inside a
    committed strike. A transient sync-read failure on the half-duplex bus
    is a known event on this rig, and an exception escaping here reaches
    `RobotApp._control_loop`, which answers a failed policy tick by
    e-stopping -- freezing the arm mid-swing, directly above the hand it
    was aiming at. A dropped reading costs one round scored as a dodge.
    That is the cheaper failure by a wide margin.
    """

    # shoulder_lift, elbow_flex, wrist_flex
    STRIKE_JOINTS: tuple[int, ...] = (1, 2, 3)

    def __init__(
        self,
        state_source,
        threshold: float = 0.02,
        joints: tuple[int, ...] | None = None,
        settle: float = 0.30,
    ) -> None:
        self.state_source = state_source
        self.threshold = threshold
        # How long the arm must have been pressing before the reading
        # means anything. Measured: the servo's load filter takes ~250 ms
        # to decay from the swing, and at +300 ms nothing reads 0.000
        # against 0.032-0.036 for a book and a hand.
        self.settle = settle
        self.joints = list(self.STRIKE_JOINTS if joints is None else joints)
        self._baseline: np.ndarray | None = None
        self._armed_at = 0.0
        self._pressing_since: float | None = None
        self._fired = False
        # Diagnostics. `peak_rise` is the largest *settled* rise seen,
        # which is the number to set `threshold` from after a session.
        self.peak_rise = 0.0
        self.read_failures = 0
        # What the last judged round read, for `report()`. The geometric
        # sensor has carried one since it landed and this did not, so
        # selecting `--contact press` silently gave up the per-round line
        # -- and then crashed at the end of the run reaching for a
        # summary that was never written. Both are the same omission.
        self.last: float | None = None

    def peak_summary(self) -> str:
        """End-of-run line. A fraction of rated torque, not a height."""
        return (f"peak held load {self.peak_rise:.3f} of rated torque "
                f"(needed {self.threshold:.3f}, after "
                f"{self.settle * 1000:.0f} ms pressing)")

    def report(self) -> str:
        """What the last judged round actually looked like."""
        if self.last is None:
            return "no reading (the paddle never pressed long enough)"
        return (f"held load {self.last:+.3f} of rated torque "
                f"(needs {self.threshold:.3f}) after {self.settle * 1000:.0f} ms")

    def _load(self) -> np.ndarray | None:
        try:
            state = self.state_source()
        except Exception:
            self.read_failures += 1
            return None
        if state.load is None:
            return None
        return np.abs(np.asarray(state.load, float)[self.joints])

    def arm(self, blank_for: float | None = None) -> None:
        # `blank_for` is accepted and ignored: the caller's guess at how
        # long the launch takes is exactly what `pressing` replaces.
        self._fired = False
        self._baseline = None
        self._pressing_since = None
        self._armed_at = time.perf_counter()
        self.last = None

    def poll(self, pressing: bool = False, **kwargs) -> ContactEvent | None:
        if self._fired:
            return None
        load = self._load()
        if load is None:
            return None
        if self._baseline is None:
            # Taken on the first poll after arming, while the arm is still
            # at the hover holding itself statically. That is the right
            # reference precisely because the comparison happens in the
            # same regime -- one static hold against another, with the
            # swing between them excluded rather than averaged in.
            self._baseline = load
            return None
        if not pressing:
            self._pressing_since = None
            return None
        now = time.perf_counter()
        if self._pressing_since is None:
            self._pressing_since = now
            return None
        if now - self._pressing_since < self.settle:
            return None
        rise = float(np.max(load - self._baseline))
        self.last = rise
        self.peak_rise = max(self.peak_rise, rise)
        if rise < self.threshold:
            return None
        self._fired = True
        log.debug("press: rise %.3f (threshold %.3f) after %.0f ms pressing, "
                  "%.0f ms since arming; baseline %s, now %s",
                  rise, self.threshold, (now - self._pressing_since) * 1e3,
                  (now - self._armed_at) * 1e3,
                  np.round(self._baseline, 3), np.round(load, 3))
        return ContactEvent(now, "servo_press", strength=min(rise, 1.0))


class SerialContactSensor(ContactSensor):
    """Piezo impact detector on a microcontroller.

    UNUSED, and never verified against hardware.

    Expects newline-delimited `HIT <microseconds> <amplitude>` from the
    board. Read on a background thread because the game loop must never
    block on a serial read.

    Kept for anyone who does add a sidecar. It was never the recommended
    path -- this used to say ServoLoadContactSensor got the same answer
    with no extra hardware, which turned out to be wrong about
    ServoLoadContactSensor rather than about the piezo.
    """

    def __init__(self, port: str, baudrate: int = 115200, threshold: float = 0.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.threshold = threshold
        self._latest: ContactEvent | None = None
        self._lock = threading.Lock()
        self._serial = None
        self._thread: threading.Thread | None = None
        self._running = False

    def open(self) -> None:
        import serial

        self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="contact")
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", "ignore").strip()
            except Exception:
                continue
            if not line.startswith("HIT"):
                continue
            parts = line.split()
            amplitude = float(parts[2]) if len(parts) > 2 else 1.0
            if amplitude < self.threshold:
                continue
            with self._lock:
                self._latest = ContactEvent(time.perf_counter(), "piezo", amplitude)

    def arm(self, blank_for: float = 0.0) -> None:
        with self._lock:
            self._latest = None

    def poll(self, **kwargs) -> ContactEvent | None:
        with self._lock:
            event, self._latest = self._latest, None
        return event

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._serial:
            self._serial.close()

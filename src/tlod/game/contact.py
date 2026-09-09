"""Did the slap land?

This is the question a camera cannot answer, and it is worth being
explicit about why. At the moment of contact the arm is directly between
an overhead camera and the contact point, occluding exactly the thing
that needs to be seen. And at 30 fps a frame is 33 ms, on an event that
decides the round and lasts a few milliseconds. Vision is the wrong
instrument.

So contact detection is an interface with three implementations:

  GeometricContactSensor   simulation. Uses ground truth, because in
                           simulation ground truth exists and pretending
                           otherwise would only be theatre.
  ProximityContactSensor   tier B, real hand and simulated arm. Infers
                           contact from tracked hand position versus the
                           virtual tool. Honest about being an estimate.
  ServoLoadContactSensor   hardware, no extra parts. Every STS3215
                           reports Present_Load, and the driver already
                           fetches those bytes in the same sync-read as
                           position -- so contact detection costs nothing
                           and needs no sidecar board.

The first two exist so the game is fully playable before hardware does.
The third is what runs on the real arm.

A piezo disc on a microcontroller would time an impact more precisely
(microseconds, versus one control tick here). It is not worth a whole
extra board: at 100 Hz the load spike lands within 10 ms, and a slap is
scored per round, not per millisecond.
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
    mistaken for a measurement. The tracked hand carries several
    centimetres of uncertainty, so near-misses will be scored wrongly in
    both directions.
    """

    def __init__(self, radius: float = 0.06, plane_tolerance: float = 0.035) -> None:
        super().__init__(radius, plane_tolerance)

    def poll(self, **kwargs) -> ContactEvent | None:
        event = super().poll(**kwargs)
        if event is not None:
            event = ContactEvent(event.stamp, "proximity", event.strength)
        return event


class ServoLoadContactSensor(ContactSensor):
    """Detect contact from the servos' own torque feedback.

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


class ToolHeightContactSensor(ContactSensor):
    """Did the paddle get where it was sent? If not, something stopped it.

    The simplest instrument on the arm, and on measurement the best one.
    The strike commands a floor below any plausible hand; the encoders say
    where the paddle actually ended up; the difference is the thickness of
    whatever was in the way. Nothing is inferred, nothing is filtered, and
    the encoders resolve 0.087 degrees -- about 0.1 mm at the tool, against
    a hand worth twenty-odd millimetres.

    Compare with the two torque-based sensors in this file. Load is a
    commanded quantity that Torque_Limit clamps, so it pins at its ceiling
    mid-swing and reads a rigid book as indistinguishable from empty air.
    Current is a real measurement but quantised at 6.5 mA, which turned out
    to be the entire size of the effect. Held still, load does separate --
    0.001 empty against 0.037 on a hand -- but only after waiting ~300 ms
    for the servo's load filter to forget the swing. Height needs no such
    wait: the number is already correct as soon as the arm has stopped.

    Two things it does need.

    The floor has to be *below* the hand, or an untouched paddle and a
    touched one both arrive and the shortfall is zero either way. That is
    `StrikeLimits.press_depth`, and it is the same requirement the press
    sensor has, for the same reason.

    And the arm has to be capable of reaching that floor when nothing is
    there, or its own tracking error reads as a hand. It is: measured, an
    unobstructed press converges to within about 3 mm, which is why the
    threshold is 8 mm rather than something tighter.

    Reads are wrapped because `poll` runs on the control thread inside a
    committed strike, and an exception escaping here reaches
    `RobotApp._control_loop`, which answers a failed policy tick by
    e-stopping -- freezing the arm mid-swing, directly above the hand it
    was aiming at. A dropped reading costs one round scored as a dodge.
    """

    def __init__(
        self,
        height_source,
        threshold: float = 0.008,
        settle: float = 0.15,
    ) -> None:
        # () -> (reached_z, commanded_z) in metres, base frame.
        self.height_source = height_source
        self.threshold = threshold
        # Long enough for an unobstructed press to have arrived. Measured,
        # it is within 3 mm about 60 ms into the hold; this is that with
        # room to spare, and still half of what the load channel needs.
        self.settle = settle
        self._pressing_since: float | None = None
        self._fired = False
        # Diagnostics: the largest settled shortfall seen, in metres.
        self.peak_rise = 0.0
        self.read_failures = 0

    def arm(self, blank_for: float | None = None) -> None:
        # `blank_for` is accepted and ignored -- `pressing` replaces it.
        self._fired = False
        self._pressing_since = None

    def poll(self, pressing: bool = False, **kwargs) -> ContactEvent | None:
        if self._fired:
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
            reached, commanded = self.height_source()
        except Exception:
            self.read_failures += 1
            return None
        short = float(reached) - float(commanded)
        self.peak_rise = max(self.peak_rise, short)
        if short < self.threshold:
            return None
        self._fired = True
        log.debug("height: stopped %.1f mm above the %.1f mm floor "
                  "(threshold %.1f mm) after %.0f ms pressing",
                  short * 1e3, float(commanded) * 1e3, self.threshold * 1e3,
                  (now - self._pressing_since) * 1e3)
        return ContactEvent(now, "tool_height", strength=min(short / 0.03, 1.0))


class ServoPressContactSensor(ContactSensor):
    """Detect contact from what the arm is still pushing against once it stops.

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

    Expects newline-delimited `HIT <microseconds> <amplitude>` from the
    board. Read on a background thread because the game loop must never
    block on a serial read.

    Kept for anyone who does add a sidecar, but it is no longer the
    recommended path -- ServoLoadContactSensor gets the same answer with
    no extra hardware.

      !! UNVERIFIED AGAINST HARDWARE !!
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

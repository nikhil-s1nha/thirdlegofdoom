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
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ContactEvent:
    stamp: float
    source: str
    strength: float = 1.0


class ContactSensor(abc.ABC):
    """Reports at most one contact per arming."""

    def arm(self) -> None:
        """Ready for the next strike; discard anything pending."""

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

    def arm(self) -> None:
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
    """

    # shoulder_lift, elbow_flex, wrist_flex
    STRIKE_JOINTS: tuple[int, ...] = (1, 2, 3)

    def __init__(
        self,
        state_source,
        threshold: float = 0.12,
        joints: tuple[int, ...] | None = None,
    ) -> None:
        self.state_source = state_source
        self.threshold = threshold
        self.joints = joints or self.STRIKE_JOINTS
        self._baseline: np.ndarray | None = None
        self._fired = False

    def arm(self) -> None:
        self._fired = False
        state = self.state_source()
        self._baseline = (
            np.abs(state.load[list(self.joints)]) if state.load is not None else None
        )

    def poll(self, **kwargs) -> ContactEvent | None:
        if self._fired:
            return None
        state = self.state_source()
        if state.load is None:
            return None
        current = np.abs(state.load[list(self.joints)])
        baseline = self._baseline if self._baseline is not None else np.zeros_like(current)
        rise = float(np.max(current - baseline))
        if rise < self.threshold:
            return None
        self._fired = True
        return ContactEvent(time.perf_counter(), "servo_load", strength=min(rise, 1.0))


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

    def arm(self) -> None:
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

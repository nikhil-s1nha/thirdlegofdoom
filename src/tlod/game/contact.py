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
  SerialContactSensor      hardware. A piezo disc on the target pad, read
                           by a microcontroller that timestamps the impact
                           in microseconds and cannot be occluded.

The first two exist so the game is fully playable before the hardware
does. The third is the one that will be trusted.
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


class SerialContactSensor(ContactSensor):
    """Piezo impact detector on a microcontroller.

    Expects newline-delimited `HIT <microseconds> <amplitude>` from the
    board. Read on a background thread because the game loop must never
    block on a serial read.

      !! UNVERIFIED AGAINST HARDWARE !!
      Written alongside firmware/pico_sidecar.py. The protocol is fixed
      but neither end has run on a real board.
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

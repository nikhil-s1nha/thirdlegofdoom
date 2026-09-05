"""Handoff between threads.

Perception and control run at different rates and must not block each
other. Perception is bursty and occasionally slow (a detection sweep costs
more than a tracking update); control must tick like a metronome, because
a control loop with jitter produces motion with jitter.

So they do not share a queue. A queue is the wrong shape here: if control
is slower than perception the queue grows and control starts consuming
stale frames, which is precisely the failure the camera layer already
works to avoid. What control wants is always the *newest* estimate and
never a backlog.

`Latest` is that: a single slot, last write wins, reads never block and
never wait for a writer.
"""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class Latest(Generic[T]):
    """A one-slot mailbox. Last write wins; readers never block."""

    __slots__ = ("_value", "_stamp", "_lock", "_event", "_writes", "_reads")

    def __init__(self, initial: T | None = None) -> None:
        self._value = initial
        self._stamp = time.perf_counter() if initial is not None else 0.0
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._writes = 0
        self._reads = 0

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._stamp = time.perf_counter()
            self._writes += 1
        self._event.set()

    def get(self) -> T | None:
        with self._lock:
            self._reads += 1
            return self._value

    def get_fresh(self, max_age: float) -> T | None:
        """The value, but only if it was written within `max_age` seconds.

        Staleness is a first-class concept here. A control loop acting on a
        perception estimate from 400 ms ago is worse than one acting on
        nothing, because it will confidently move to the wrong place.
        """
        with self._lock:
            if self._value is None or (time.perf_counter() - self._stamp) > max_age:
                return None
            self._reads += 1
            return self._value

    def wait(self, timeout: float | None = None) -> T | None:
        """Block until a value is written, then return it."""
        if not self._event.wait(timeout):
            return None
        self._event.clear()
        return self.get()

    @property
    def age(self) -> float:
        with self._lock:
            return time.perf_counter() - self._stamp if self._stamp else float("inf")

    @property
    def stats(self) -> dict[str, int]:
        return {"writes": self._writes, "reads": self._reads}

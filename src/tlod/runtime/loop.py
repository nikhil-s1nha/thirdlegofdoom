"""Fixed-rate loops with honest timing.

Two things here that a naive `while True: work(); sleep(period)` gets wrong:

1. Drift. Sleeping for a fixed period after variable work makes the actual
   rate depend on how long the work took. This schedules against an
   absolute timeline instead, so a slow tick is followed by a shorter
   sleep and the average rate holds.

2. Silence about overruns. If the work does not fit in the period, the
   loop simply runs slow and nobody finds out until the robot behaves
   strangely. `RateLoop` counts overruns and `Timing` reports percentiles,
   because on this project every millisecond is accounted for and a
   regression in loop timing is a bug worth failing on.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class Timing:
    """Rolling latency statistics for one named stage."""

    name: str
    window: int = 240
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=240))

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    def record(self, start: float) -> float:
        dt = time.perf_counter() - start
        self.add(dt)
        return dt

    def _pct(self, p: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        k = min(int(p * len(ordered)), len(ordered) - 1)
        return ordered[k]

    @property
    def mean_ms(self) -> float:
        return 1e3 * sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p50_ms(self) -> float:
        return 1e3 * self._pct(0.50)

    @property
    def p95_ms(self) -> float:
        return 1e3 * self._pct(0.95)

    @property
    def max_ms(self) -> float:
        return 1e3 * max(self.samples) if self.samples else 0.0

    def summary(self) -> str:
        return (
            f"{self.name:<22} mean {self.mean_ms:6.2f} ms  "
            f"p50 {self.p50_ms:6.2f}  p95 {self.p95_ms:6.2f}  max {self.max_ms:6.2f}  "
            f"n={len(self.samples)}"
        )


class RateLoop:
    """Drift-free fixed-rate iteration.

        loop = RateLoop(100.0)
        while running:
            loop.tick()   # sleeps just enough to hold the rate
            do_work()
    """

    def __init__(self, hz: float, name: str = "loop") -> None:
        self.hz = hz
        self.period = 1.0 / hz
        self.name = name
        self.overruns = 0
        self.ticks = 0
        self._next = time.perf_counter()
        self.jitter = Timing(f"{name}.jitter")

    def tick(self) -> float:
        """Sleep until the next scheduled instant. Returns the overshoot."""
        now = time.perf_counter()
        remaining = self._next - now

        if remaining > 0:
            # Coarse sleep, then spin the last ~1 ms. OS sleep granularity
            # is a couple of ms, which at 100 Hz is 20% jitter; the spin
            # costs one busy core briefly and buys a metronome.
            if remaining > 0.0015:
                time.sleep(remaining - 0.0015)
            while time.perf_counter() < self._next:
                pass
            overshoot = 0.0
        else:
            overshoot = -remaining
            self.overruns += 1
            if overshoot > self.period * 5:
                # Far behind: give up on catching up rather than spinning
                # through a burst of back-to-back iterations.
                self._next = time.perf_counter()

        self._next += self.period
        self.ticks += 1
        self.jitter.add(abs(overshoot))
        return overshoot

    @property
    def overrun_rate(self) -> float:
        return self.overruns / self.ticks if self.ticks else 0.0

    def reset(self) -> None:
        self._next = time.perf_counter()
        self.overruns = 0
        self.ticks = 0

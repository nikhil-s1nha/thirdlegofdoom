"""Estimating the offset between two machines' clocks.

Every timestamp in this system means "when the shutter opened", and the
control loop uses those to decide whether an estimate is fresh enough to
act on. Split across two boards, that stops working: `perf_counter()` on
the Orange Pi and on the Pi are unrelated numbers, so a hand position
stamped on one and judged on the other is compared against nonsense.

The failure is silent, which is what makes it dangerous. Nothing errors;
the control board simply believes every estimate is either far too old
(and never acts) or impossibly fresh (and acts on stale data).

The measurement is the NTP one, without the daemon:

    t0  we send a ping                     (our clock)
    t1  they receive it and reply with t1  (their clock)
    t3  we receive the reply               (our clock)

    offset = t1 - (t0 + t3) / 2
    rtt    = t3 - t0

Assuming a symmetric path, offset is what to add to their timestamps to
put them in our terms. The assumption fails when the path is congested,
so we take the sample with the **smallest round trip** out of several
rather than averaging: the least-delayed exchange is the one least
distorted, and averaging just mixes good samples with bad.

Good enough is sub-millisecond on wired ethernet, a few ms on WiFi.
Against a ~16 ms frame interval, either is fine. What is not fine is
assuming zero.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ClockEstimate:
    offset: float          # add this to their timestamps to get ours
    rtt: float             # round trip of the sample used
    samples: int
    stamp: float           # when this was measured, our clock

    @property
    def uncertainty(self) -> float:
        """Half the round trip: the most the offset can be wrong by if the
        path is asymmetric."""
        return self.rtt / 2.0

    def age(self) -> float:
        return time.perf_counter() - self.stamp


def measure_offset(
    host: str,
    port: int,
    samples: int = 9,
    timeout: float = 0.4,
    gap: float = 0.02,
) -> ClockEstimate | None:
    """Ping the vision board and estimate its clock offset."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    best: tuple[float, float] | None = None
    taken = 0
    try:
        for _ in range(samples):
            t0 = time.perf_counter()
            try:
                sock.sendto(json.dumps({"ping": t0}).encode(), (host, port))
                data, _ = sock.recvfrom(256)
                t3 = time.perf_counter()
            except OSError:
                continue
            try:
                t1 = float(json.loads(data)["t"])
            except (ValueError, KeyError, TypeError):
                continue
            rtt = t3 - t0
            taken += 1
            if best is None or rtt < best[1]:
                best = (t1 - (t0 + t3) / 2.0, rtt)
            time.sleep(gap)
    finally:
        sock.close()

    if best is None:
        log.warning("no clock samples from %s:%d", host, port)
        return None
    offset, rtt = best
    log.info("clock offset %+.3f ms (rtt %.3f ms, %d samples)",
             offset * 1e3, rtt * 1e3, taken)
    return ClockEstimate(offset=offset, rtt=rtt, samples=taken, stamp=time.perf_counter())


class ClockResponder:
    """Answers offset pings. Runs on the vision board."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._sock: socket.socket | None = None

    def start(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.2)
        self._sock = sock
        return sock

    def poll(self) -> None:
        """Answer any pending ping. Non-blocking enough to sit in a loop.

        Replies with the timestamp taken as late as possible, so the
        reply carries the moment closest to when it goes on the wire.
        """
        if self._sock is None:
            return
        try:
            data, addr = self._sock.recvfrom(256)
        except (OSError, TimeoutError):
            return
        if b"ping" not in data:
            return
        try:
            self._sock.sendto(json.dumps({"t": time.perf_counter()}).encode(), addr)
        except OSError:
            pass

    def stop(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

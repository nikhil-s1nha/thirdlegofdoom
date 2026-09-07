"""Control side over UART: receive detections down a serial line.

Counterpart to `tlod.net.uart_publisher.UartVisionPublisher`, filling the
same `Latest[Perception]` mailbox `tlod.net.subscriber.VisionSubscriber`
does -- the game, the controller and the IK are unchanged and unaware of
which transport is underneath.

Three jobs, same as the UDP subscriber, plus one UDP does not need:

**Clock translation.** Incoming timestamps are in the sender's clock and
useless here until shifted; see `tlod.net.clock` for why this cannot be
skipped.

**Ordering.** Sequence numbers going backwards are dropped, in case a
future revision of the wire ever buffers rather than reads byte-by-byte.
On a raw UART link ordering is not actually at risk -- bytes arrive in
the order they were sent, there is no reordering hazard the way UDP
datagrams taking different network paths can have -- but the check costs
nothing and keeps this a drop-in replacement for the UDP path rather
than a subtly different contract.

**Demultiplexing.** UDP gets two channels (data port, clock port) for
the price of a second socket. A UART link is one wire, so a single
reader thread here reads every line and dispatches it by the "k" tag
from `uart_protocol` -- "data" updates the mailbox, "pong" completes
whichever `sync_clock()` call is waiting on it. Only one thread ever
calls `readline()`; two threads racing to read the same stream would
each get an arbitrary slice of it.
"""

from __future__ import annotations

import logging
import threading
import time

from tlod.net.clock import ClockEstimate
from tlod.net.protocol import Packet, decode_perception
from tlod.net.uart_protocol import DEFAULT_BAUD, decode_line, encode_clock
from tlod.runtime.signal import Latest
from tlod.types import Perception

log = logging.getLogger(__name__)


class UartVisionSubscriber:
    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud: int = DEFAULT_BAUD,
        serial_conn=None,
        resync_interval: float = 30.0,
        require_clock: bool = True,
        clock_samples: int = 9,
        clock_timeout: float = 0.4,
    ) -> None:
        self.port = port
        self.baud = baud
        self.resync_interval = resync_interval
        self.require_clock = require_clock
        self.clock_samples = clock_samples
        self.clock_timeout = clock_timeout

        self.perception: Latest[Perception] = Latest()
        self.clock: ClockEstimate | None = None
        self.received = 0
        self.dropped_stale = 0
        self.dropped_bad = 0
        self._last_seq = -1

        self._ser = serial_conn
        self._owns_ser = serial_conn is None
        self._write_lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []

        self._ping_id = 0
        self._pending: dict[int, threading.Event] = {}
        self._pong_results: dict[int, tuple[float, float]] = {}
        self._pending_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._ser is None:
            import serial

            self._ser = serial.Serial(self.port, self.baud, timeout=0.25)

        self._running = True
        t = threading.Thread(target=self._receive_loop, name="uart-rx", daemon=True)
        t.start()
        self._threads.append(t)

        self.sync_clock()
        if self.clock is None and self.require_clock:
            self.stop()
            raise RuntimeError(
                f"no clock response on {self.port}. Without a clock offset "
                "every freshness check is meaningless, so this refuses to "
                "run blind. Start the vision publisher first, or pass "
                "require_clock=False to accept the risk."
            )

        if self.resync_interval > 0:
            t = threading.Thread(target=self._resync_loop, name="clock-sync", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("listening on %s @ %d baud", self.port, self.baud)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._ser is not None and self._owns_ser:
            self._ser.close()

    def __enter__(self) -> UartVisionSubscriber:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- clock ---------------------------------------------------------------
    def sync_clock(self) -> ClockEstimate | None:
        """Same NTP-style estimate as `tlod.net.clock.measure_offset`,
        run over the UART line instead of a throwaway UDP socket: take
        several ping/pong round trips, keep the one with the smallest
        RTT, since that sample is the least distorted by queueing delay
        on either end."""
        best: tuple[float, float] | None = None
        taken = 0
        for _ in range(self.clock_samples):
            sample = self._ping_once(self.clock_timeout)
            if sample is not None:
                taken += 1
                offset, rtt = sample
                if best is None or rtt < best[1]:
                    best = (offset, rtt)
            time.sleep(0.02)

        if best is None:
            log.warning("no clock samples on %s", self.port)
            return None
        offset, rtt = best
        log.info("clock offset %+.3f ms (rtt %.3f ms, %d samples)",
                  offset * 1e3, rtt * 1e3, taken)
        self.clock = ClockEstimate(offset=offset, rtt=rtt, samples=taken,
                                    stamp=time.perf_counter())
        return self.clock

    def _ping_once(self, timeout: float) -> tuple[float, float] | None:
        if self._ser is None:
            return None
        with self._pending_lock:
            self._ping_id += 1
            ping_id = self._ping_id
            event = threading.Event()
            self._pending[ping_id] = event

        t0 = time.perf_counter()
        try:
            with self._write_lock:
                self._ser.write(encode_clock("ping", id=ping_id, t_send=t0))
        except Exception as e:
            log.debug("uart write failed: %s", e)
            with self._pending_lock:
                self._pending.pop(ping_id, None)
            return None

        got = event.wait(timeout)
        with self._pending_lock:
            self._pending.pop(ping_id, None)
            t1, t3 = self._pong_results.pop(ping_id, (None, None))
        if not got or t1 is None:
            return None
        rtt = t3 - t0
        offset = t1 - (t0 + t3) / 2.0
        return offset, rtt

    def _resync_loop(self) -> None:
        while self._running:
            time.sleep(self.resync_interval)
            if self._running:
                self.sync_clock()

    @property
    def offset(self) -> float:
        return self.clock.offset if self.clock else 0.0

    # -- receive -------------------------------------------------------------
    def _receive_loop(self) -> None:
        while self._running:
            line = self._readline()
            if not line:
                continue

            decoded = decode_line(line)
            if decoded is None:
                self.dropped_bad += 1
                continue
            kind, obj = decoded

            if kind == "pong":
                self._handle_pong(obj)
                continue
            if kind != "data":
                continue

            packet = Packet.from_dict(obj)
            if packet is None:
                self.dropped_bad += 1
                continue
            # See module docstring: not a real hazard on a raw serial
            # line, kept so this is a genuine drop-in for the UDP path.
            if packet.seq <= self._last_seq:
                self.dropped_stale += 1
                continue
            self._last_seq = packet.seq
            self.received += 1
            self.perception.set(decode_perception(packet, self.offset))

    def _handle_pong(self, obj: dict) -> None:
        t3 = time.perf_counter()
        ping_id = obj.get("id")
        t1 = obj.get("t")
        if ping_id is None or t1 is None:
            return
        with self._pending_lock:
            event = self._pending.get(ping_id)
            if event is None:
                return  # timed out already, or not ours
            self._pong_results[ping_id] = (float(t1), t3)
        event.set()

    def _readline(self) -> bytes | None:
        if self._ser is None:
            return None
        try:
            line = self._ser.readline()
        except Exception:
            return None
        return line or None

    def report(self) -> str:
        clock = (
            f"offset {self.clock.offset*1e3:+.2f} ms +/-{self.clock.uncertainty*1e3:.2f}"
            if self.clock else "NOT SYNCED"
        )
        return (
            f"  received {self.received}, reordered {self.dropped_stale}, "
            f"malformed {self.dropped_bad}\n  clock {clock}"
        )

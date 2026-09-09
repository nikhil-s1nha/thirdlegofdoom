"""Control side: receive detections. Runs on the Raspberry Pi.

Fills the same `Latest[Perception]` mailbox the in-process perception
thread used to fill, so nothing downstream can tell the difference. The
game, the controller and the IK are unchanged and unaware.

Two jobs beyond receiving:

**Clock translation.** Incoming timestamps are in the sender's clock and
useless here until shifted. The offset is measured at startup and
re-measured periodically, because clocks drift.

**Ordering.** UDP can reorder, and a stale datagram overwriting a fresh
one would make the arm chase the past. Sequence numbers going backwards
are dropped.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

from tlod.net.clock import ClockEstimate, measure_offset
from tlod.net.protocol import Packet, decode_perception
from tlod.net.publisher import DEFAULT_CLOCK_PORT, DEFAULT_PORT
from tlod.net.uart_link import FrameDecoder
from tlod.runtime.signal import Latest
from tlod.types import Perception

log = logging.getLogger(__name__)


class VisionSubscriber:
    def __init__(
        self,
        host: str = "",
        port: int = DEFAULT_PORT,
        clock_port: int = DEFAULT_CLOCK_PORT,
        resync_interval: float = 30.0,
        require_clock: bool = True,
        serial_port: str | None = None,
        serial_baudrate: int = 115200,
        max_stamp_age: float = 1.0,
        seq_resync_after: int = 30,
    ) -> None:
        self.host = host
        self.port = port
        self.clock_port = clock_port
        self.resync_interval = resync_interval
        self.require_clock = require_clock
        # Guards against a *second* publisher on the same port -- almost
        # always one left running from an earlier session. Its sequence
        # numbers are higher than a freshly started one's, so the seq
        # guard below hands it the mailbox permanently and discards the
        # publisher you actually meant to listen to. Costly to diagnose,
        # because every counter looks like ordinary packet reordering.
        #
        # max_stamp_age throws out frames that are old on their face: a
        # stale process is usually also a backlogged one, and no useful
        # perception is a second behind. seq_resync_after covers the
        # case where the newcomer is the *lower* sequence number, by
        # giving up on a sequence we have rejected this many times in a
        # row and adopting whatever is actually arriving.
        self.max_stamp_age = max_stamp_age
        self.seq_resync_after = seq_resync_after
        # UART is the learning/testing link (see docs/deployment.md). When
        # set, this reads perception off a serial port instead of the UDP
        # socket -- the two are mutually exclusive, since they stand in
        # for the same "how do I hear from the vision board" question.
        # Clock sync still goes over UDP if `host` is set: it is a
        # separate physical channel (Ethernet/WiFi) from the UART link,
        # and nothing here measures a clock offset over serial.
        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate

        self.perception: Latest[Perception] = Latest()
        self.clock: ClockEstimate | None = None
        self.received = 0
        self.dropped_stale = 0
        self.dropped_bad = 0
        self.dropped_old = 0
        self.seq_resyncs = 0
        self._last_seq = -1
        self._consecutive_stale = 0
        self._sock: socket.socket | None = None
        self._serial = None
        self._running = False
        self._threads: list[threading.Thread] = []

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self.host:
            self.sync_clock()
            if self.clock is None and self.require_clock:
                raise RuntimeError(
                    f"no clock response from {self.host}:{self.clock_port}. "
                    "Without a clock offset every freshness check is meaningless, "
                    "so this refuses to run blind. Start the vision publisher "
                    "first, or pass require_clock=False to accept the risk."
                )

        self._running = True
        targets = []

        if self.serial_port:
            import serial

            self._serial = serial.Serial(self.serial_port, self.serial_baudrate, timeout=0.25)
            targets.append((self._serial_receive_loop, "vision-rx-uart"))
            log.info("listening on %s @ %d", self.serial_port, self.serial_baudrate)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(0.25)
            self._sock = sock
            targets.append((self._receive_loop, "vision-rx"))
            log.info("listening on :%d", self.port)

        if self.host and self.resync_interval > 0:
            targets.append((self._resync_loop, "clock-sync"))
        for target, name in targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._sock:
            self._sock.close()
        if self._serial:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> VisionSubscriber:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- clock -------------------------------------------------------------
    def sync_clock(self) -> ClockEstimate | None:
        estimate = measure_offset(self.host, self.clock_port)
        if estimate is not None:
            self.clock = estimate
        return estimate

    def _resync_loop(self) -> None:
        while self._running:
            time.sleep(self.resync_interval)
            if self._running:
                self.sync_clock()

    @property
    def offset(self) -> float:
        return self.clock.offset if self.clock else 0.0

    # -- receive -----------------------------------------------------------
    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except (OSError, TimeoutError):
                continue
            self._handle(data)

    def _serial_receive_loop(self) -> None:
        decoder = FrameDecoder()
        while self._running:
            try:
                data = self._serial.read(4096)
            except OSError:
                continue
            if not data:
                continue
            for payload in decoder.feed(data):
                self._handle(payload)

    def _handle(self, data: bytes) -> None:
        """Decode one datagram/frame payload and, if it's newer than what
        we already have, store it. Shared by the UDP and UART paths so
        the reordering/staleness guard behaves identically either way."""
        packet = Packet.decode(data)
        if packet is None:
            self.dropped_bad += 1
            return
        # Old on its face, whoever sent it. Checked before the sequence
        # guard so a backlogged sender cannot claim the mailbox on the
        # strength of a high sequence number alone.
        age = time.perf_counter() - (packet.stamp + self.offset)
        if self.max_stamp_age > 0 and age > self.max_stamp_age:
            self.dropped_old += 1
            return
        # Both UDP and (framed) UART can reorder. An older packet
        # overwriting a newer one would have the arm chase the past.
        if packet.seq <= self._last_seq:
            self.dropped_stale += 1
            self._consecutive_stale += 1
            # Nothing has advanced the sequence in a long while, yet
            # fresh packets keep arriving: the sender restarted, or a
            # different one took over. Adopt it rather than ignoring the
            # only perception there is.
            if self._consecutive_stale < self.seq_resync_after:
                return
            self.seq_resyncs += 1
            log.warning("sequence went backwards for %d packets; resyncing to seq %d",
                        self._consecutive_stale, packet.seq)
        self._consecutive_stale = 0
        self._last_seq = packet.seq
        self.received += 1
        self.perception.set(decode_perception(packet, self.offset))

    def report(self) -> str:
        clock = (
            f"offset {self.clock.offset*1e3:+.2f} ms +/-{self.clock.uncertainty*1e3:.2f}"
            if self.clock else "NOT SYNCED"
        )
        extra = ""
        if self.dropped_old:
            extra = (f"\n  {self.dropped_old} dropped as older than "
                     f"{self.max_stamp_age:.1f}s -- is a second publisher running?")
        if self.seq_resyncs:
            extra += f"\n  {self.seq_resyncs} sequence resyncs (sender restarted?)"
        return (
            f"  received {self.received}, reordered {self.dropped_stale}, "
            f"stale {self.dropped_old}, malformed {self.dropped_bad}\n"
            f"  clock {clock}{extra}"
        )

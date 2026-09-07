"""Arm state, out to whoever wants to watch. For boards with no screen.

The control board (Raspberry Pi) has no display -- `tlod control` prints
a text report only at the end of a run. This is the same idea as
`tlod.vision.preview` (an MJPEG stream for a vision board with no
screen), but for joint angles instead of pixels: a small, throttled UDP
stream of what the arm is doing, so a laptop on the same network can run
a live viewer while there is no physical arm to look at (`MockArm`) or
while the real one is out of sight.

Deliberately its own tiny protocol rather than reusing `net.protocol`:
this is a different data shape (joint angles, not hand detections) with
a different consumer (a human watching a window, not the control loop),
and it can be silently dropped without anything downstream noticing --
unlike a lost `Perception` packet, a lost `TelemetryPacket` just means
one skipped frame of drawing.

**Clock sync, and why it's a third hop, not a rename of the first one.**
`net.clock` already solves "translate one board's `perf_counter()` into
another's" for the vision->control hop. That offset stays inside the
control board -- `packet.shutter` below is the vision board's shutter
time, already corrected into *the control board's* clock by the time it
gets here. Getting from "shutter, in the control board's clock" to "how
long ago was that, on my laptop" needs a second, independent offset:
control board -> Mac. Same exchange, same `ClockResponder`/
`measure_offset`, just a different pair of machines and a different
port, because clock drift is symmetric nonsense between any two clocks,
not just the first two that happened to be wired together.

Without it, `tlod arm-viewer` can only ever report *reception* age (time
since this laptop's socket got the packet), which conflates real
end-to-end latency with however long the packet sat in a WiFi queue on
the way here. With it, the HUD can show where the time actually goes:
shutter -> command (measured on the control board, clock-agnostic,
already correct) and command -> display (the new leg this file adds,
needs the offset to mean anything at all).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from tlod.net.clock import ClockEstimate, ClockResponder, measure_offset
from tlod.runtime.signal import Latest
from tlod.types import NUM_JOINTS

log = logging.getLogger(__name__)

DEFAULT_TELEMETRY_PORT = 45900
DEFAULT_TELEMETRY_CLOCK_PORT = 45901
PROTOCOL_VERSION = 1


@dataclass(slots=True)
class TelemetryPacket:
    """One snapshot of the arm, as it travels."""

    seq: int
    stamp: float                       # this sample's time, sender's clock
    q: np.ndarray                      # measured joints, radians, shape (6,)
    commanded: np.ndarray               # commanded joints, radians, shape (6,)
    estopped: bool
    hand: np.ndarray | None = None      # tracked hand position, base frame, if any
    shutter: float | None = None        # shutter time behind this state, sender's clock

    def encode(self) -> bytes:
        payload = {
            "v": PROTOCOL_VERSION,
            "seq": self.seq,
            "t": round(self.stamp, 6),
            "q": [round(float(x), 4) for x in self.q],
            "c": [round(float(x), 4) for x in self.commanded],
            "e": bool(self.estopped),
        }
        if self.hand is not None:
            payload["h"] = [round(float(x), 4) for x in self.hand]
        if self.shutter is not None:
            payload["sh"] = round(self.shutter, 6)
        return json.dumps(payload, separators=(",", ":")).encode()

    @staticmethod
    def decode(data: bytes) -> TelemetryPacket | None:
        try:
            d = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return None
        if d.get("v") != PROTOCOL_VERSION:
            return None
        try:
            q = np.array(d["q"], dtype=float)
            commanded = np.array(d["c"], dtype=float)
        except (KeyError, ValueError):
            return None
        if q.shape != (NUM_JOINTS,) or commanded.shape != (NUM_JOINTS,):
            return None
        hand = np.array(d["h"], dtype=float) if "h" in d else None
        return TelemetryPacket(
            seq=int(d.get("seq", 0)),
            stamp=float(d.get("t", 0.0)),
            q=q,
            commanded=commanded,
            estopped=bool(d.get("e", False)),
            hand=hand,
            shutter=float(d["sh"]) if "sh" in d else None,
        )


class ArmTelemetryPublisher:
    """Samples the controller (and optionally the perception mailbox) at a
    throttled rate and fires it at one or more UDP targets. Fire-and-forget,
    same reasoning as `VisionPublisher`: if nobody is watching, nothing
    breaks, and there is no handshake to get stuck on if a viewer restarts.

    Also answers clock pings on `clock_port`, same `ClockResponder` the
    vision board uses, so a viewer can translate `stamp`/`shutter` into
    its own clock instead of only measuring reception age.
    """

    def __init__(
        self,
        controller,
        targets: list[tuple[str, int]],
        perception: Latest | None = None,
        rate_hz: float = 20.0,
        clock_port: int = DEFAULT_TELEMETRY_CLOCK_PORT,
    ) -> None:
        self.controller = controller
        self.targets = targets
        self.perception = perception
        self.rate_hz = rate_hz
        self.clock_port = clock_port

        self.clock = ClockResponder(clock_port)
        self._sock: socket.socket | None = None
        self._running = False
        self._threads: list[threading.Thread] = []
        self._seq = 0
        self.sent = 0

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clock.start()
        self._running = True
        for target, name in ((self._loop, "arm-telemetry"), (self._clock_loop, "arm-clock")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("arm telemetry to %s, clock on :%d",
                 ", ".join(f"{h}:{p}" for h, p in self.targets), self.clock_port)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self.clock.stop()
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> ArmTelemetryPublisher:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _clock_loop(self) -> None:
        while self._running:
            self.clock.poll()

    def _loop(self) -> None:
        period = 1.0 / max(self.rate_hz, 0.1)
        while self._running:
            t0 = time.perf_counter()
            self._sample_and_send()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _sample_and_send(self) -> None:
        state = self.controller.state()
        snapshot = self.perception.get_fresh(0.5) if self.perception is not None else None
        hand = snapshot.hands[0].position if snapshot is not None and snapshot.hands else None
        shutter = snapshot.stamp if snapshot is not None else None

        self._seq += 1
        packet = TelemetryPacket(
            seq=self._seq,
            stamp=time.perf_counter(),
            q=state.q,
            commanded=self.controller.commanded,
            estopped=self.controller.estopped,
            hand=hand,
            shutter=shutter,
        )
        data = packet.encode()
        for target in self.targets:
            try:
                self._sock.sendto(data, target)
            except OSError as e:
                log.debug("telemetry send to %s failed: %s", target, e)
        self.sent += 1


class ArmTelemetrySubscriber:
    """Receives `TelemetryPacket`s and keeps only the newest, same
    `Latest[T]` mailbox shape as everything else in this codebase.

    Clock sync here is best-effort, not required: this feeds a HUD
    number, not a safety decision, so a viewer that can't reach the
    clock port still opens and just falls back to reporting reception
    age instead of true end-to-end latency -- unlike `VisionSubscriber`,
    which refuses to run at all without one, because there nothing
    downstream would notice the numbers were meaningless.
    """

    def __init__(
        self,
        port: int = DEFAULT_TELEMETRY_PORT,
        host: str = "",
        clock_port: int = DEFAULT_TELEMETRY_CLOCK_PORT,
        resync_interval: float = 30.0,
    ) -> None:
        self.port = port
        self.host = host
        self.clock_port = clock_port
        self.resync_interval = resync_interval

        self.latest: Latest[TelemetryPacket] = Latest()
        self.clock: ClockEstimate | None = None
        self.received = 0
        self.dropped_stale = 0
        self.dropped_bad = 0
        self._last_seq = -1
        self._sock: socket.socket | None = None
        self._running = False
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.host:
            self.sync_clock()
            if self.clock is None:
                log.warning("no clock response from %s:%d -- showing reception age, "
                            "not end-to-end latency", self.host, self.clock_port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.25)
        self._sock = sock
        self._running = True
        targets = [(self._loop, "arm-telemetry-rx")]
        if self.host and self.resync_interval > 0:
            targets.append((self._resync_loop, "arm-clock-sync"))
        for target, name in targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("arm telemetry listening on :%d", self.port)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> ArmTelemetrySubscriber:
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
        """Add this to a packet's `stamp`/`shutter` to get our clock."""
        return self.clock.offset if self.clock else 0.0

    def to_local(self, their_time: float) -> float:
        return their_time + self.offset

    # -- receive -----------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except (OSError, TimeoutError):
                continue
            packet = TelemetryPacket.decode(data)
            if packet is None:
                self.dropped_bad += 1
                continue
            if packet.seq <= self._last_seq:
                self.dropped_stale += 1
                continue
            self._last_seq = packet.seq
            self.received += 1
            self.latest.set(packet)

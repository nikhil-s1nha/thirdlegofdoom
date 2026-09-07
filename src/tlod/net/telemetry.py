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
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from tlod.runtime.signal import Latest
from tlod.types import NUM_JOINTS

log = logging.getLogger(__name__)

DEFAULT_TELEMETRY_PORT = 45900
PROTOCOL_VERSION = 1


@dataclass(slots=True)
class TelemetryPacket:
    """One snapshot of the arm, as it travels."""

    seq: int
    stamp: float                       # sender's clock, perf_counter seconds
    q: np.ndarray                      # measured joints, radians, shape (6,)
    commanded: np.ndarray               # commanded joints, radians, shape (6,)
    estopped: bool
    hand: np.ndarray | None = None      # tracked hand position, base frame, if any

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
        )


class ArmTelemetryPublisher:
    """Samples the controller (and optionally the perception mailbox) at a
    throttled rate and fires it at one or more UDP targets. Fire-and-forget,
    same reasoning as `VisionPublisher`: if nobody is watching, nothing
    breaks, and there is no handshake to get stuck on if a viewer restarts.
    """

    def __init__(
        self,
        controller,
        targets: list[tuple[str, int]],
        perception: Latest | None = None,
        rate_hz: float = 20.0,
    ) -> None:
        self.controller = controller
        self.targets = targets
        self.perception = perception
        self.rate_hz = rate_hz

        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._seq = 0
        self.sent = 0

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="arm-telemetry", daemon=True)
        self._thread.start()
        log.info("arm telemetry to %s", ", ".join(f"{h}:{p}" for h, p in self.targets))

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> ArmTelemetryPublisher:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _loop(self) -> None:
        period = 1.0 / max(self.rate_hz, 0.1)
        while self._running:
            t0 = time.perf_counter()
            self._sample_and_send()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _sample_and_send(self) -> None:
        state = self.controller.state()
        hand = None
        if self.perception is not None:
            snapshot = self.perception.get_fresh(0.5)
            if snapshot is not None and snapshot.hands:
                hand = snapshot.hands[0].position

        self._seq += 1
        packet = TelemetryPacket(
            seq=self._seq,
            stamp=time.perf_counter(),
            q=state.q,
            commanded=self.controller.commanded,
            estopped=self.controller.estopped,
            hand=hand,
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
    `Latest[T]` mailbox shape as everything else in this codebase."""

    def __init__(self, port: int = DEFAULT_TELEMETRY_PORT) -> None:
        self.port = port
        self.latest: Latest[TelemetryPacket] = Latest()
        self.received = 0
        self.dropped_stale = 0
        self.dropped_bad = 0
        self._last_seq = -1
        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.25)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="arm-telemetry-rx", daemon=True)
        self._thread.start()
        log.info("arm telemetry listening on :%d", self.port)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> ArmTelemetrySubscriber:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

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

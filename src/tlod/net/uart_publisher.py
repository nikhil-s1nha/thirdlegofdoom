"""Vision side over UART: detect, localise, publish down a serial line.

`tlod.net.publisher.VisionPublisher` (UDP/Ethernet) is the transport
`docs/deployment.md` recommends, and the one with fan-out to more than
one listener and no fixed rate ceiling. This exists for the case that
motivated it in the first place: two boards close enough together to run
a direct TTL UART line instead of a network hop -- one cable class
instead of two, no switch, no IP addressing, at the cost of a fixed baud
rate and exactly one listener.

Same pipeline as the UDP publisher -- detect, locate, track, publish --
and the same fire-and-forget philosophy: if nothing is listening on the
other end, writes just go nowhere and this keeps running. Only the wire
itself differs, so this mirrors `VisionPublisher`'s shape closely on
purpose. Sharing a base class between two implementations was considered
and rejected -- twenty duplicated lines is cheaper to read than an
abstraction built for exactly two cases, one of which (UDP) has fan-out
and multi-target semantics the other structurally cannot have.

Unlike UDP, where the clock responder owns a second socket, a UART link
is one shared wire: this runs a single reader thread that sees every
incoming line and dispatches it by the "k" tag from `uart_protocol` --
here, that is only ever a clock "ping" from the control board, answered
with a "pong". Writes (data lines and pong replies) share one lock
because pyserial's `Serial.write` is not documented as safe to call
from two threads at once, even though the two directions of a UART pair
are physically independent wires.
"""

from __future__ import annotations

import logging
import threading
import time

from tlod.net.uart_protocol import DEFAULT_BAUD, decode_line, encode_clock, encode_data
from tlod.net.protocol import encode_perception
from tlod.runtime.loop import Timing
from tlod.types import Perception

log = logging.getLogger(__name__)


class UartVisionPublisher:
    def __init__(
        self,
        camera,
        detector,
        locator,
        tracker=None,
        object_detector=None,
        port: str = "/dev/ttyS4",
        baud: int = DEFAULT_BAUD,
        serial_conn=None,
        hand_suppression_radius: float = 0.07,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.locator = locator
        self.tracker = tracker
        self.object_detector = object_detector
        self.port = port
        self.baud = baud
        self.hand_suppression_radius = hand_suppression_radius

        # A pre-opened connection is accepted so tests (and anything else
        # that wants an in-memory stand-in for a real serial port) never
        # have to touch an actual device node. Production callers leave
        # this as None and get a real `serial.Serial`.
        self._ser = serial_conn
        self._owns_ser = serial_conn is None
        self._write_lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._seq = 0
        self._last_index = -1

        self.t_detect = Timing("detect")
        self.t_total = Timing("shutter->sent")
        self.frames = 0
        self.sent = 0
        self.pings_answered = 0
        self.dropped_bad = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._ser is None:
            import serial  # pyserial: only needed on the real hardware path

            self._ser = serial.Serial(self.port, self.baud, timeout=0.25)
        self.camera.start()
        self._running = True
        for target, name in (
            (self._vision_loop, "vision"),
            (self._reader_loop, "uart-rx"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("publishing on %s @ %d baud", self.port, self.baud)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self.camera.stop()
        self.detector.close()
        if self._ser is not None and self._owns_ser:
            self._ser.close()

    def __enter__(self) -> UartVisionPublisher:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- clock: answer pings from the control board -------------------------
    def _reader_loop(self) -> None:
        while self._running:
            line = self._readline()
            if not line:
                continue
            decoded = decode_line(line)
            if decoded is None:
                self.dropped_bad += 1
                continue
            kind, obj = decoded
            if kind != "ping":
                continue
            # The reply carries the timestamp taken as late as possible,
            # so it reflects the moment closest to when it actually goes
            # on the wire -- same reasoning as ClockResponder.poll().
            with self._write_lock:
                self._ser.write(
                    encode_clock("pong", id=obj.get("id"), t=time.perf_counter())
                )
            self.pings_answered += 1

    def _readline(self) -> bytes | None:
        if self._ser is None:
            return None
        try:
            line = self._ser.readline()
        except Exception:
            return None
        return line or None

    # -- vision loop ---------------------------------------------------------
    def _vision_loop(self) -> None:
        import numpy as np

        while self._running:
            frame = self.camera.read()
            if frame is None or frame.index == self._last_index:
                time.sleep(0.001)
                continue
            self._last_index = frame.index
            self.frames += 1

            try:
                t0 = time.perf_counter()
                hands2d = self.detector.detect(frame)
                self.t_detect.record(t0)

                observations = self.locator.locate_all(hands2d)
                if self.tracker is not None:
                    self.tracker.update([o.position for o in observations], frame.stamp)
                    enriched = []
                    for obs in observations:
                        nearest = min(
                            self.tracker.tracks,
                            key=lambda t: float(np.linalg.norm(t.filter.position - obs.position)),
                            default=None,
                        )
                        enriched.append(
                            type(obs)(
                                position=obs.position, stamp=obs.stamp,
                                velocity=nearest.filter.velocity if nearest else None,
                                landmarks=obs.landmarks, handedness=obs.handedness,
                                confidence=obs.confidence,
                            )
                        )
                    observations = enriched

                objects = []
                if self.object_detector is not None:
                    objects = self.object_detector.detect(frame)
                    if self.hand_suppression_radius > 0 and observations:
                        objects = self._suppress(objects, observations)

                self.publish(
                    Perception(stamp=frame.stamp, hands=observations, objects=objects)
                )
                self.t_total.add(time.perf_counter() - frame.stamp)
            except Exception:
                log.exception("vision iteration failed")

    def _suppress(self, objects, hands):
        """See `VisionPublisher._suppress` -- identical reasoning, table-
        plane comparison so parallax doesn't hide a hand's real object."""
        import numpy as np

        projector = getattr(self.locator, "projector", None)
        if projector is None:
            return objects
        shadows = []
        for hand in hands:
            uv = projector.project(hand.position)
            if uv is None:
                continue
            on_table = projector.pixel_to_plane(uv[0], uv[1], 0.0)
            if on_table is not None:
                shadows.append(on_table)
        if not shadows:
            return objects
        return [
            d for d in objects
            if min(float(np.linalg.norm(d.position - s)) for s in shadows)
            > self.hand_suppression_radius
        ]

    # -- output ------------------------------------------------------------
    def publish(self, perception: Perception) -> None:
        if self._ser is None:
            return
        self._seq += 1
        packet = encode_perception(perception, self._seq, time.perf_counter())
        line = encode_data(packet)
        try:
            with self._write_lock:
                self._ser.write(line)
        except Exception as e:
            log.debug("uart write failed: %s", e)
            return
        self.sent += 1

    def report(self) -> str:
        return (
            f"  frames {self.frames}, published {self.sent}, "
            f"pings answered {self.pings_answered}, malformed rx {self.dropped_bad}\n"
            f"  {self.t_detect.summary()}\n"
            f"  {self.t_total.summary()}"
        )

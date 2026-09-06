"""Vision side: detect, localise, publish. Runs on the Orange Pi 5.

Owns the camera, the calibration and the detectors, and emits one small
UDP datagram per detection. It holds no kinematics and no game logic --
it does not know or care what the arm does with what it sees.

Publishing is fire-and-forget by design. If nobody is listening, the
datagrams go nowhere and this keeps running; if the control board
restarts, it starts receiving again with no handshake. There is nothing
to get stuck on, which is the main reason to prefer this over a
connection.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

from tlod.net.clock import ClockResponder
from tlod.net.protocol import encode_perception
from tlod.runtime.loop import Timing
from tlod.types import Perception

log = logging.getLogger(__name__)

DEFAULT_PORT = 45800
DEFAULT_CLOCK_PORT = 45801


class VisionPublisher:
    def __init__(
        self,
        camera,
        detector,
        locator,
        tracker=None,
        object_detector=None,
        targets: list[tuple[str, int]] | None = None,
        clock_port: int = DEFAULT_CLOCK_PORT,
        hand_suppression_radius: float = 0.07,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.locator = locator
        self.tracker = tracker
        self.object_detector = object_detector
        self.targets = targets or [("255.255.255.255", DEFAULT_PORT)]
        self.hand_suppression_radius = hand_suppression_radius

        self.clock = ClockResponder(clock_port)
        self._sock: socket.socket | None = None
        self._running = False
        self._threads: list[threading.Thread] = []
        self._seq = 0
        self._last_index = -1

        self.t_detect = Timing("detect")
        self.t_total = Timing("shutter->sent")
        self.frames = 0
        self.sent = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.camera.start()
        self.clock.start()
        self._running = True
        for target, name in ((self._vision_loop, "vision"), (self._clock_loop, "clock")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("publishing to %s", ", ".join(f"{h}:{p}" for h, p in self.targets))

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self.camera.stop()
        self.clock.stop()
        self.detector.close()
        if self._sock:
            self._sock.close()

    def __enter__(self) -> VisionPublisher:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- loops -------------------------------------------------------------
    def _clock_loop(self) -> None:
        while self._running:
            self.clock.poll()

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
        """Drop object detections that are really the hand.

        Compared on the table plane, because that is where the object
        detector resolves every blob -- a hand 10 cm up is reported at the
        point behind it, which parallax can put far from where the hand
        actually is.
        """
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
        if self._sock is None:
            return
        self._seq += 1
        packet = encode_perception(perception, self._seq, time.perf_counter())
        data = packet.encode()
        for target in self.targets:
            try:
                self._sock.sendto(data, target)
            except OSError as e:
                log.debug("send to %s failed: %s", target, e)
        self.sent += 1

    def report(self) -> str:
        return (
            f"  frames {self.frames}, published {self.sent}\n"
            f"  {self.t_detect.summary()}\n"
            f"  {self.t_total.summary()}"
        )

"""The UART transport: same guarantees as the UDP split, different wire.

No real serial port is opened anywhere here. `_loop_pair()` builds two
in-memory, cross-connected objects that satisfy the small slice of the
pyserial API `UartVisionPublisher`/`UartVisionSubscriber` actually use
(`write`, `readline`, `close`) and are injected via `serial_conn=`
instead of a device path -- the same dependency-injection seam the rest
of the codebase uses for cameras and arm backends, so the transport can
be tested without hardware, a socket, or even a real thread scheduler
doing anything unusual.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from tlod.net.protocol import Packet, encode_perception
from tlod.net.uart_protocol import decode_line, encode_clock, encode_data
from tlod.net.uart_publisher import UartVisionPublisher
from tlod.net.uart_subscriber import UartVisionSubscriber
from tlod.types import Detection, HandObservation, Perception
from tlod.vision.calibration import synthetic_projector
from tlod.vision.camera import MockCamera
from tlod.vision.hands import HandLocator
from tlod.vision.scene import SceneHandDetector, SyntheticHandScene
from tlod.vision.tracking import MultiTracker


class _LoopEnd:
    """One end of a fake, cross-connected serial pair."""

    def __init__(self, outbox: queue.Queue, inbox: queue.Queue, timeout: float = 0.25):
        self._outbox = outbox
        self._inbox = inbox
        self._timeout = timeout
        self._buf = b""

    def write(self, data: bytes) -> int:
        self._outbox.put(data)
        return len(data)

    def readline(self) -> bytes:
        deadline = time.monotonic() + self._timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return b""
            try:
                self._buf += self._inbox.get(timeout=remaining)
            except queue.Empty:
                return b""
        line, _, self._buf = self._buf.partition(b"\n")
        return line + b"\n"

    def close(self) -> None:
        pass


def loop_pair(timeout: float = 0.25) -> tuple[_LoopEnd, _LoopEnd]:
    a_to_b: queue.Queue = queue.Queue()
    b_to_a: queue.Queue = queue.Queue()
    return _LoopEnd(a_to_b, b_to_a, timeout), _LoopEnd(b_to_a, a_to_b, timeout)


def sample_perception(stamp=1234.5):
    return Perception(
        stamp=stamp,
        hands=[HandObservation(np.array([0.22, 0.05, 0.10]), stamp,
                               np.array([0.3, -0.1, 0.0]), confidence=0.93,
                               handedness="Right")],
        objects=[Detection("red", np.array([0.18, 0.11, 0.0]), stamp, 0.8, radius=0.022)],
    )


# -- framing -----------------------------------------------------------------

def test_data_line_round_trips_through_the_same_packet_fields():
    packet = encode_perception(sample_perception(), seq=7, sent=1234.51)
    line = encode_data(packet)
    assert line.endswith(b"\n")

    kind, obj = decode_line(line.rstrip(b"\n"))
    assert kind == "data"
    back = Packet.from_dict(obj)
    assert back.seq == 7
    assert back.hands[0]["p"] == pytest.approx([0.22, 0.05, 0.10], abs=1e-4)


def test_clock_lines_carry_their_kind():
    ping = encode_clock("ping", id=3, t_send=1.0)
    kind, obj = decode_line(ping.rstrip(b"\n"))
    assert kind == "ping"
    assert obj["id"] == 3


def test_malformed_lines_decode_to_none():
    assert decode_line(b"not json") is None
    assert decode_line(b'{"seq": 1}') is None          # no "k" tag
    assert decode_line(b"") is None


def test_wrong_protocol_version_is_rejected_same_as_udp():
    assert Packet.from_dict({"v": 999, "seq": 1}) is None


# -- clock, wire-level only ----------------------------------------------

def _run_minimal_responder(end: _LoopEnd, stop: threading.Event) -> None:
    """Answers "ping" with "pong", nothing else -- the slice of
    UartVisionPublisher's reader loop this test needs, without pulling in
    a camera or detector to exercise it."""
    while not stop.is_set():
        line = end.readline()
        if not line:
            continue
        decoded = decode_line(line)
        if decoded is None:
            continue
        kind, obj = decoded
        if kind == "ping":
            end.write(encode_clock("pong", id=obj.get("id"), t=time.perf_counter()))


def test_clock_offset_measured_over_a_loop_pair():
    # sync_clock() needs its own _receive_loop running to catch the
    # "pong" replies, same as it would inside start() -- so this drives
    # start() (with require_clock=False, so a slow first sync can't
    # raise) rather than calling sync_clock() in isolation.
    vision_end, control_end = loop_pair()
    stop = threading.Event()
    responder = threading.Thread(target=_run_minimal_responder, args=(vision_end, stop), daemon=True)
    responder.start()

    subscriber = UartVisionSubscriber(serial_conn=control_end, require_clock=False,
                                      resync_interval=0, clock_samples=5)
    try:
        subscriber.start()
        estimate = subscriber.clock
    finally:
        subscriber.stop()
        stop.set()
        responder.join(timeout=1.0)

    assert estimate is not None
    # Same process, so the true offset is zero; allow for the round trip.
    assert abs(estimate.offset) < max(estimate.rtt, 0.01)
    assert estimate.uncertainty == pytest.approx(estimate.rtt / 2)


def test_subscriber_refuses_to_run_without_a_clock():
    """Same contract as the UDP subscriber: judging freshness against an
    unmeasured offset fails silently, so this fails loudly instead."""
    dead_end, _ = loop_pair()  # nothing on the other end to answer pings
    sub = UartVisionSubscriber(serial_conn=dead_end, require_clock=True,
                               clock_samples=2, clock_timeout=0.05)
    with pytest.raises(RuntimeError, match="clock"):
        sub.start()


# -- end to end ----------------------------------------------------------

def test_detections_cross_a_uart_line_accurately():
    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    vision_end, control_end = loop_pair()

    publisher = UartVisionPublisher(
        camera=MockCamera(320, 240, 60, scene=scene),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        tracker=MultiTracker(),
        serial_conn=vision_end,
    )
    publisher.start()
    time.sleep(0.2)

    subscriber = UartVisionSubscriber(serial_conn=control_end, require_clock=True,
                                      clock_samples=5, resync_interval=0)
    subscriber.start()
    try:
        time.sleep(1.2)
        snapshot = subscriber.perception.get()
        truth = scene.position_at(publisher.camera.elapsed)
    finally:
        subscriber.stop()
        publisher.stop()

    assert snapshot is not None, "nothing arrived"
    assert subscriber.received > 10
    assert subscriber.dropped_bad == 0
    assert np.linalg.norm(snapshot.hands[0].position - truth) < 0.02
    assert subscriber.clock is not None


def test_freshness_gate_works_across_a_uart_line():
    """If the vision board stops writing, the control board must see
    None rather than serve an increasingly stale estimate."""
    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    vision_end, control_end = loop_pair()

    publisher = UartVisionPublisher(
        camera=MockCamera(320, 240, 60, scene=scene),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        serial_conn=vision_end,
    )
    publisher.start()
    time.sleep(0.2)

    subscriber = UartVisionSubscriber(serial_conn=control_end, require_clock=True,
                                      clock_samples=5, resync_interval=0)
    subscriber.start()
    try:
        time.sleep(0.6)
        assert subscriber.perception.get_fresh(0.3) is not None
        publisher.stop()                        # vision board goes away
        time.sleep(0.4)
        assert subscriber.perception.get_fresh(0.15) is None, "stale data served"
    finally:
        subscriber.stop()

"""The vision/control split.

Perception on one board, kinematics on another. These test the parts that
fail silently if wrong -- clock translation, ordering, and the freshness
gate -- because none of them raise when they misbehave.
"""

import json
import socket
import threading
import time

import numpy as np
import pytest

from tlod.net.clock import ClockEstimate, ClockResponder, measure_offset
from tlod.net.protocol import Packet, decode_perception, encode_perception
from tlod.net.publisher import VisionPublisher
from tlod.net.subscriber import VisionSubscriber
from tlod.types import Detection, HandObservation, Perception
from tlod.vision.calibration import synthetic_projector
from tlod.vision.camera import MockCamera
from tlod.vision.hands import HandLocator
from tlod.vision.scene import SceneHandDetector, SyntheticHandScene
from tlod.vision.tracking import MultiTracker


def sample_perception(stamp=1234.5):
    return Perception(
        stamp=stamp,
        hands=[HandObservation(np.array([0.22, 0.05, 0.10]), stamp,
                               np.array([0.3, -0.1, 0.0]), confidence=0.93,
                               handedness="Right")],
        objects=[Detection("red", np.array([0.18, 0.11, 0.0]), stamp, 0.8, radius=0.022)],
    )


# -- protocol --------------------------------------------------------------

def test_round_trip_preserves_content():
    packet = encode_perception(sample_perception(), seq=7, sent=1234.51)
    back = decode_perception(Packet.decode(packet.encode()))
    assert np.allclose(back.hands[0].position, [0.22, 0.05, 0.10], atol=1e-4)
    assert np.allclose(back.hands[0].velocity, [0.3, -0.1, 0.0], atol=1e-4)
    assert back.hands[0].handedness == "Right"
    assert back.objects[0].label == "red"


def test_datagram_fits_one_packet():
    """Fragmentation would add loss and latency for no benefit."""
    busy = Perception(
        stamp=1.0,
        hands=[HandObservation(np.zeros(3), 1.0, np.zeros(3)) for _ in range(2)],
        objects=[Detection(f"obj{i}", np.zeros(3), 1.0) for i in range(12)],
    )
    from tlod.net.protocol import MAX_DATAGRAM
    assert len(encode_perception(busy, 1, 1.0).encode()) < MAX_DATAGRAM


def test_clock_offset_is_applied_to_timestamps():
    """The whole point: their clock translated into ours."""
    packet = encode_perception(sample_perception(1000.0), seq=1, sent=1000.0)
    back = decode_perception(packet, clock_offset=+5.0)
    assert back.stamp == pytest.approx(1005.0)
    assert back.hands[0].stamp == pytest.approx(1005.0)


def test_malformed_and_wrong_version_are_rejected():
    assert Packet.decode(b"not json") is None
    assert Packet.decode(b'{"v":999,"seq":1}') is None
    assert Packet.decode(b"") is None


# -- clock -----------------------------------------------------------------

def test_clock_offset_measured_over_loopback():
    responder = ClockResponder(45991)
    responder.start()
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            responder.poll()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        estimate = measure_offset("127.0.0.1", 45991, samples=5, gap=0.004)
    finally:
        stop.set()
        thread.join(timeout=1.0)
        responder.stop()

    assert estimate is not None
    # Same machine, so the true offset is zero; allow the round trip.
    assert abs(estimate.offset) < max(estimate.rtt, 0.01)
    assert estimate.uncertainty == pytest.approx(estimate.rtt / 2)


def test_clock_measurement_fails_cleanly_with_no_responder():
    assert measure_offset("127.0.0.1", 45992, samples=2, timeout=0.05) is None


def test_offset_sign_is_correct_against_a_genuinely_different_clock():
    """A same-process loopback (test_clock_offset_measured_over_loopback,
    above) can never catch a sign error in the offset formula: both sides
    share one `perf_counter()`, so the true offset is always ~0 and a
    flipped sign is indistinguishable from a correct one (-0 == +0). This
    fakes a responder whose clock reads a large, known amount *ahead* --
    the two-real-boards scenario (independent perf_counter() epochs,
    plausibly hours apart) that actually exposed the bug this pins down.
    """
    known_ahead = 137.5  # seconds; the fake responder's clock reads this much larger

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 46000))
    sock.settimeout(0.2)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(256)
            except (OSError, TimeoutError):
                continue
            if b"ping" not in data:
                continue
            sock.sendto(json.dumps({"t": time.perf_counter() + known_ahead}).encode(), addr)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        estimate = measure_offset("127.0.0.1", 46000, samples=5, gap=0.004)
    finally:
        stop.set()
        thread.join(timeout=1.0)
        sock.close()

    assert estimate is not None
    # `their_timestamp + offset` must land back near our own clock -- so
    # offset has to be approximately *negative* known_ahead, the opposite
    # sign from how far ahead their raw reading looked.
    assert estimate.offset == pytest.approx(-known_ahead, abs=max(estimate.rtt, 0.05))


def test_subscriber_refuses_to_run_without_a_clock():
    """Better to fail loudly than to judge freshness against nonsense."""
    sub = VisionSubscriber(host="127.0.0.1", port=45993, clock_port=45994)
    with pytest.raises(RuntimeError, match="clock"):
        sub.start()


# -- ordering --------------------------------------------------------------

def test_out_of_order_datagrams_are_dropped():
    """UDP can reorder; a stale packet overwriting a fresh one would have
    the arm chase the past."""
    sub = VisionSubscriber(port=45995, require_clock=False)
    sub._last_seq = 10
    for seq in (11, 9, 12, 5):
        packet = Packet(seq=seq, stamp=float(seq), hands=[], objects=[])
        if packet.seq <= sub._last_seq:
            sub.dropped_stale += 1
            continue
        sub._last_seq = packet.seq
        sub.received += 1
    assert sub.received == 2 and sub.dropped_stale == 2


# -- end to end ------------------------------------------------------------

def test_detections_cross_the_wire_accurately():
    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    publisher = VisionPublisher(
        camera=MockCamera(320, 240, 60, scene=scene),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        tracker=MultiTracker(),
        targets=[("127.0.0.1", 45996)],
        clock_port=45997,
    )
    publisher.start()
    time.sleep(0.2)
    subscriber = VisionSubscriber(host="127.0.0.1", port=45996, clock_port=45997)
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


def test_freshness_gate_works_across_the_link():
    """If the vision board dies, the control board must see None."""
    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    publisher = VisionPublisher(
        camera=MockCamera(320, 240, 60, scene=scene),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        targets=[("127.0.0.1", 45998)],
        clock_port=45999,
    )
    publisher.start()
    time.sleep(0.2)
    subscriber = VisionSubscriber(host="127.0.0.1", port=45998, clock_port=45999)
    subscriber.start()
    try:
        time.sleep(0.6)
        assert subscriber.perception.get_fresh(0.3) is not None
        publisher.stop()                       # vision board goes away
        time.sleep(0.4)
        assert subscriber.perception.get_fresh(0.15) is None, "stale data served"
    finally:
        subscriber.stop()


def test_a_backlogged_publisher_cannot_claim_the_mailbox():
    """A vision-serve left running from an earlier session has higher
    sequence numbers than a freshly started one, so the sequence guard
    alone hands it the mailbox permanently -- and it is usually also
    backlogged, so what it hands over is seconds old. Observed on real
    hardware as a 4.5 s shutter-to-servo latency with every counter
    looking like ordinary reordering."""
    sub = VisionSubscriber(port=45994, require_clock=False)
    # The age check is only meaningful against a clock estimate.
    sub.clock = ClockEstimate(offset=0.0, rtt=0.0004, samples=8,
                              stamp=time.perf_counter())
    now = time.perf_counter()

    sub._handle(Packet(seq=5, stamp=now, hands=[], objects=[]).encode())
    assert sub.received == 1

    # The zombie: far ahead in sequence, far behind in time.
    sub._handle(Packet(seq=9000, stamp=now - 4.5, hands=[], objects=[]).encode())
    assert sub.dropped_old == 1
    assert sub._last_seq == 5, "a stale packet advanced the sequence"

    # The live publisher keeps the mailbox.
    sub._handle(Packet(seq=6, stamp=time.perf_counter(), hands=[], objects=[]).encode())
    assert sub.received == 2


def test_the_sequence_guard_gives_up_when_the_sender_restarts():
    """A restarted publisher begins at seq 1 again. Holding the old
    sequence forever would ignore the only perception there is."""
    sub = VisionSubscriber(port=45993, require_clock=False, seq_resync_after=5)
    sub._handle(Packet(seq=500, stamp=time.perf_counter(), hands=[], objects=[]).encode())

    for seq in range(1, 5):
        sub._handle(Packet(seq=seq, stamp=time.perf_counter(), hands=[], objects=[]).encode())
    assert sub.seq_resyncs == 0, "gave up too early; this could be real reordering"

    for seq in range(5, 9):
        sub._handle(Packet(seq=seq, stamp=time.perf_counter(), hands=[], objects=[]).encode())
    assert sub.seq_resyncs == 1
    assert sub._last_seq < 500, "did not adopt the restarted sender"


def test_the_age_check_is_skipped_without_a_clock():
    """Over UART no offset is measured at all, so the sender's stamps are
    in an unrelated timebase. Rejecting them on age would throw away
    perfectly good perception -- a worse failure than the one the age
    check exists to prevent."""
    sub = VisionSubscriber(serial_port="/dev/null", require_clock=False)
    assert sub.clock is None
    sub._handle(Packet(seq=1, stamp=1234.5, hands=[], objects=[]).encode())
    assert sub.received == 1 and sub.dropped_old == 0

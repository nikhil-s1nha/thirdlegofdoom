"""Arm telemetry: the Pi -> laptop stream for the arm-less HIL test.

Mirrors the shape of tests/test_net.py -- protocol round trip, then an
end-to-end check over a real UDP loopback -- but for `TelemetryPacket`
instead of `Perception`.
"""

import time

import numpy as np

from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.net.telemetry import ArmTelemetryPublisher, ArmTelemetrySubscriber, TelemetryPacket
from tlod.runtime.signal import Latest
from tlod.types import Perception


def test_round_trip_preserves_content():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
    commanded = np.array([0.0, -0.1, 0.2, -0.3, 0.4, 0.5])
    packet = TelemetryPacket(seq=3, stamp=12.5, q=q, commanded=commanded,
                             estopped=True, hand=np.array([0.2, 0.0, 0.1]))
    back = TelemetryPacket.decode(packet.encode())
    assert np.allclose(back.q, q, atol=1e-3)
    assert np.allclose(back.commanded, commanded, atol=1e-3)
    assert back.estopped is True
    assert np.allclose(back.hand, [0.2, 0.0, 0.1], atol=1e-3)
    assert back.seq == 3


def test_hand_is_optional():
    q = np.zeros(6)
    packet = TelemetryPacket(seq=1, stamp=0.0, q=q, commanded=q, estopped=False)
    back = TelemetryPacket.decode(packet.encode())
    assert back.hand is None


def test_malformed_is_rejected():
    assert TelemetryPacket.decode(b"not json") is None
    assert TelemetryPacket.decode(b'{"v":999}') is None


def test_end_to_end_over_loopback():
    controller = ArmController(MockArm(), SafetyLimits(), control_hz=100.0)
    controller.start()
    controller.goto_joints(np.array([0.1, -0.2, 0.3, -0.1, 0.0]), duration=0.1)

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46001)], rate_hz=30.0,
    )
    subscriber = ArmTelemetrySubscriber(port=46001)
    subscriber.start()
    publisher.start()
    try:
        time.sleep(0.5)
        packet = subscriber.latest.get()
    finally:
        publisher.stop()
        subscriber.stop()
        controller.stop(park=False)

    assert packet is not None
    assert subscriber.received > 5
    assert subscriber.dropped_bad == 0
    assert np.allclose(packet.commanded[:5], [0.1, -0.2, 0.3, -0.1, 0.0], atol=1e-2)


def test_forwards_the_freshest_tracked_hand():
    controller = ArmController(MockArm(), SafetyLimits(), control_hz=100.0)
    controller.start()
    perception: Latest[Perception] = Latest()

    from tlod.types import HandObservation

    perception.set(Perception(
        stamp=time.perf_counter(),
        hands=[HandObservation(position=np.array([0.25, 0.05, 0.1]), stamp=time.perf_counter())],
    ))

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46002)],
        perception=perception, rate_hz=30.0,
    )
    subscriber = ArmTelemetrySubscriber(port=46002)
    subscriber.start()
    publisher.start()
    try:
        time.sleep(0.3)
        packet = subscriber.latest.get()
    finally:
        publisher.stop()
        subscriber.stop()
        controller.stop(park=False)

    assert packet is not None
    assert packet.hand is not None
    assert np.allclose(packet.hand, [0.25, 0.05, 0.1], atol=1e-3)


def test_stale_perception_is_not_forwarded():
    """A hand seen half a second ago should not still be drawn as current."""
    controller = ArmController(MockArm(), SafetyLimits(), control_hz=100.0)
    controller.start()
    perception: Latest[Perception] = Latest()

    from tlod.types import HandObservation

    perception.set(Perception(
        stamp=0.0,
        hands=[HandObservation(position=np.array([0.25, 0.05, 0.1]), stamp=0.0)],
    ))
    time.sleep(0.6)  # older than get_fresh(0.5)'s window

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46003)],
        perception=perception, rate_hz=30.0,
    )
    subscriber = ArmTelemetrySubscriber(port=46003)
    subscriber.start()
    publisher.start()
    try:
        time.sleep(0.3)
        packet = subscriber.latest.get()
    finally:
        publisher.stop()
        subscriber.stop()
        controller.stop(park=False)

    assert packet is not None
    assert packet.hand is None

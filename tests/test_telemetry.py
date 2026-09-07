"""Arm telemetry: the Pi -> laptop stream for the arm-less HIL test.

Mirrors the shape of tests/test_net.py -- protocol round trip, then an
end-to-end check over a real UDP loopback -- but for `TelemetryPacket`
instead of `Perception`.
"""

import time

import numpy as np
import pytest

from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.net.telemetry import ArmTelemetryPublisher, ArmTelemetrySubscriber, TelemetryPacket
from tlod.runtime.signal import Latest
from tlod.types import Perception


def test_round_trip_preserves_content():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
    commanded = np.array([0.0, -0.1, 0.2, -0.3, 0.4, 0.5])
    packet = TelemetryPacket(seq=3, stamp=12.5, q=q, commanded=commanded,
                             estopped=True, hand=np.array([0.2, 0.0, 0.1]), shutter=12.3)
    back = TelemetryPacket.decode(packet.encode())
    assert np.allclose(back.q, q, atol=1e-3)
    assert np.allclose(back.commanded, commanded, atol=1e-3)
    assert back.estopped is True
    assert np.allclose(back.hand, [0.2, 0.0, 0.1], atol=1e-3)
    assert back.shutter == pytest.approx(12.3, abs=1e-4)
    assert back.seq == 3


def test_hand_and_shutter_are_optional():
    q = np.zeros(6)
    packet = TelemetryPacket(seq=1, stamp=0.0, q=q, commanded=q, estopped=False)
    back = TelemetryPacket.decode(packet.encode())
    assert back.hand is None
    assert back.shutter is None


def test_malformed_is_rejected():
    assert TelemetryPacket.decode(b"not json") is None
    assert TelemetryPacket.decode(b'{"v":999}') is None


def test_end_to_end_over_loopback():
    controller = ArmController(MockArm(), SafetyLimits(), control_hz=100.0)
    controller.start()
    controller.goto_joints(np.array([0.1, -0.2, 0.3, -0.1, 0.0]), duration=0.1)

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46001)], rate_hz=30.0,
        clock_port=46011,
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

    shutter = time.perf_counter()
    perception.set(Perception(
        stamp=shutter,
        hands=[HandObservation(position=np.array([0.25, 0.05, 0.1]), stamp=shutter)],
    ))

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46002)],
        perception=perception, rate_hz=30.0, clock_port=46012,
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
    assert packet.shutter == pytest.approx(shutter, abs=1e-3)


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
        perception=perception, rate_hz=30.0, clock_port=46013,
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
    assert packet.shutter is None


def test_subscriber_syncs_clock_when_a_host_is_given():
    controller = ArmController(MockArm(), SafetyLimits(), control_hz=100.0)
    controller.start()

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46004)], rate_hz=30.0,
        clock_port=46014,
    )
    publisher.start()
    time.sleep(0.1)  # give the clock responder a moment to bind
    subscriber = ArmTelemetrySubscriber(port=46004, host="127.0.0.1", clock_port=46014)
    try:
        subscriber.start()
        assert subscriber.clock is not None
        # Same machine: the true offset is zero, allow the round trip.
        assert abs(subscriber.offset) < max(subscriber.clock.rtt, 0.01)
    finally:
        subscriber.stop()
        publisher.stop()
        controller.stop(park=False)


def test_subscriber_falls_back_gracefully_with_no_clock_responder():
    """A viewer is a diagnostic, not a safety gate -- it should still open,
    just without a real end-to-end latency number."""
    subscriber = ArmTelemetrySubscriber(port=46005, host="127.0.0.1", clock_port=46015)
    try:
        subscriber.start()
        assert subscriber.clock is None
        assert subscriber.offset == 0.0
    finally:
        subscriber.stop()


def test_capture_to_command_latency_is_computable_without_clock_sync():
    """shutter and stamp are both the control board's own clock, so this
    leg needs no cross-machine translation at all."""
    controller = ArmController(MockArm(), SafetyLimits(), control_hz=100.0)
    controller.start()
    perception: Latest[Perception] = Latest()

    from tlod.types import HandObservation

    shutter = time.perf_counter() - 0.05  # simulate a 50 ms-old detection
    perception.set(Perception(
        stamp=shutter,
        hands=[HandObservation(position=np.array([0.25, 0.05, 0.1]), stamp=shutter)],
    ))

    publisher = ArmTelemetryPublisher(
        controller=controller, targets=[("127.0.0.1", 46006)],
        perception=perception, rate_hz=30.0, clock_port=46016,
    )
    subscriber = ArmTelemetrySubscriber(port=46006)
    subscriber.start()
    publisher.start()
    try:
        # Short wait, not the usual 0.3s: perception is set once and never
        # refreshed here, so the gap grows with elapsed time (correctly --
        # a real, unrefreshed detection *should* look increasingly stale).
        # A quick read keeps this test measuring the fixed 50 ms offset,
        # not however long the test happened to sleep.
        time.sleep(0.08)
        packet = subscriber.latest.get()
    finally:
        publisher.stop()
        subscriber.stop()
        controller.stop(park=False)

    assert packet is not None and packet.shutter is not None
    capture_to_command_ms = (packet.stamp - packet.shutter) * 1e3
    assert 30.0 < capture_to_command_ms < 200.0

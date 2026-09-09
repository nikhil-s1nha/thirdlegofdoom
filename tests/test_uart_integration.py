"""VisionPublisher/VisionSubscriber against a real serial port.

Uses one end of a pseudo-terminal pair as the "board" (opened by
`serial.Serial`, exactly as `--serial-port /dev/ttyAMA0` would be on a
Pi) and drives the other end directly with `os.read`/`os.write` from the
test. That exercises the real `pyserial` + framing code paths that only
run when `--serial-port` is passed -- not just the pure `FrameDecoder`
logic already covered in test_uart_link.py -- without depending on this
platform being able to name *both* ends of the pair (some sandboxes
cannot: `os.ttyname` on the master fd here raises ERANGE, so only the
slave is used as a `serial_port`).

Skipped if `pyserial` isn't installed (it's the optional `uart` extra).
"""

import os
import pty
import time

import numpy as np
import pytest

serial = pytest.importorskip("serial")

from tlod.net.protocol import Packet, decode_perception, encode_perception
from tlod.net.publisher import VisionPublisher
from tlod.net.subscriber import VisionSubscriber
from tlod.net.uart_link import FrameDecoder, encode_frame
from tlod.types import Detection, HandObservation, Perception
from tlod.vision.calibration import synthetic_projector
from tlod.vision.camera import MockCamera
from tlod.vision.hands import HandLocator
from tlod.vision.scene import SceneHandDetector, SyntheticHandScene
from tlod.vision.tracking import MultiTracker


def test_publisher_writes_valid_frames_to_the_serial_port():
    """The write side: what `VisionPublisher` puts on the wire is
    exactly what `FrameDecoder` + `Packet.decode` can recover."""
    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)
    os.close(slave_fd)  # VisionPublisher reopens it via serial.Serial

    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    publisher = VisionPublisher(
        camera=MockCamera(320, 240, 60, scene=scene),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        tracker=MultiTracker(),
        targets=[],  # no UDP target -- UART only
        serial_port=slave_path,
    )
    publisher.start()
    try:
        time.sleep(0.5)
        raw = os.read(master_fd, 65536)
    finally:
        publisher.stop()
        os.close(master_fd)

    decoder = FrameDecoder()
    payloads = decoder.feed(raw)
    assert payloads, "no frames arrived on the wire"
    packet = Packet.decode(payloads[0])
    assert packet is not None
    perception = decode_perception(packet)
    assert perception.hands, "expected at least one hand in the synthetic scene"


def test_subscriber_reads_frames_from_the_serial_port():
    """The read side: bytes written raw to the wire, framed by hand, show
    up in `VisionSubscriber.perception` exactly like a UDP packet would."""
    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)
    os.close(slave_fd)  # VisionSubscriber reopens it via serial.Serial

    subscriber = VisionSubscriber(serial_port=slave_path, require_clock=False)
    subscriber.start()
    try:
        truth = np.array([0.22, 0.05, 0.10])
        perception = Perception(
            stamp=1234.5,
            hands=[HandObservation(truth, 1234.5, np.zeros(3), confidence=0.9)],
            objects=[Detection("red", np.array([0.1, 0.1, 0.0]), 1234.5, 0.8)],
        )
        for seq in range(1, 4):
            packet = encode_perception(perception, seq=seq, sent=time.perf_counter())
            os.write(master_fd, encode_frame(packet.encode()))
            time.sleep(0.05)
        time.sleep(0.3)
        snapshot = subscriber.perception.get()
    finally:
        subscriber.stop()
        os.close(master_fd)

    assert snapshot is not None, "nothing arrived over the simulated UART link"
    assert subscriber.received == 3
    assert subscriber.dropped_bad == 0
    assert np.allclose(snapshot.hands[0].position, truth, atol=1e-4)

"""Headless vision verification.

The point of these checks is to fail when something is wrong, so the
tests mostly verify that they *do* fail rather than that they pass.
"""


import numpy as np
import pytest

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.vision.calibrate_flow import calibration_poses
from tlod.vision.camera import MockCamera
from tlod.vision.check import Report, Thresholds, check_against_arm, check_precision
from tlod.vision.hands import HandLocator
from tlod.vision.scene import SceneHandDetector, SyntheticHandScene
from tlod.vision.calibration import synthetic_projector


def rig():
    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    return (MockCamera(320, 240, 60, scene=scene), SceneHandDetector(scene),
            HandLocator(projector, depth_mode="size"), scene, projector)


def test_precision_check_passes_on_a_clean_pipeline():
    camera, detector, locator, _, _ = rig()
    with camera:
        report = check_precision(camera, detector, locator, duration=1.5)
    assert report.frames > 20
    assert report.detections == report.frames
    named = {r.name: r for r in report.results}
    assert named["detection rate"].passed
    assert named["teleports"].passed


def test_detection_rate_fails_when_nothing_is_found():
    """A blind pipeline must be reported as blind, not as quiet."""
    camera, _, locator, _, _ = rig()

    class Blind:
        def detect(self, frame):
            return []

        def close(self):
            pass

    with camera:
        report = check_precision(camera, Blind(), locator, duration=1.0)
    assert not report.passed
    assert not {r.name: r for r in report.results}["detection rate"].passed


def test_teleports_are_caught():
    """A detector that loses and reacquires elsewhere must be flagged."""
    camera, detector, locator, scene, projector = rig()

    class Jumpy:
        def __init__(self):
            self.n = 0

        def detect(self, frame):
            self.n += 1
            t = 0.0 if self.n % 2 else 4.0     # two far-apart points, alternating
            hand = scene.hand2d_at(t, frame.stamp)
            return [hand] if hand else []

        def close(self):
            pass

    with camera:
        report = check_precision(camera, Jumpy(), locator, duration=1.5,
                                 thresholds=Thresholds(jump_distance_m=0.02))
    assert not {r.name: r for r in report.results}["teleports"].passed


def test_depth_spread_is_informational_unless_asserted():
    """The precondition is an instruction to a human, not something the
    code can check, so it must not fail an honest run by default."""
    camera, detector, locator, _, _ = rig()
    with camera:
        loose = check_precision(camera, detector, locator, duration=1.2)
    assert {r.name: r for r in loose.results}["depth spread"].passed

    camera2, detector2, locator2, _, _ = rig()
    with camera2:
        strict = check_precision(camera2, detector2, locator2, duration=1.2,
                                 fixed_distance=True,
                                 thresholds=Thresholds(max_depth_spread_mm=1.0))
    assert not {r.name: r for r in strict.results}["depth spread"].passed


def test_accuracy_against_kinematics_passes_with_a_good_camera():
    """The check that actually validates calibration."""
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]]), max_speed=8.0),
                               SafetyLimits(), control_hz=200.0)
    controller.start()

    def perfect_locate(image):
        return model.fk(controller.state().q[:5])[:3, 3]

    camera = MockCamera(320, 240, 60)
    camera.start()
    report = check_against_arm(
        camera=camera, controller=controller,
        locate_marker=perfect_locate, poses=calibration_poses(6),
        report=Report(), settle=0.01, move_time=0.15,
    )
    controller.backend.disconnect()
    result = {r.name: r for r in report.results}["accuracy vs kinematics"]
    assert result.passed and result.value < 1.0
    assert len(report.arm_points) >= 5


def test_accuracy_fails_on_a_miscalibrated_camera():
    """A constant camera-to-base offset must be caught. This is the whole
    reason the arm check exists -- nothing camera-only can see it."""
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]]), max_speed=8.0),
                               SafetyLimits(), control_hz=200.0)
    controller.start()
    bias = np.array([0.05, 0.0, 0.0])          # 5 cm out

    camera = MockCamera(320, 240, 60)
    camera.start()
    report = check_against_arm(
        camera=camera, controller=controller,
        locate_marker=lambda img: model.fk(controller.state().q[:5])[:3, 3] + bias,
        poses=calibration_poses(6), report=Report(), settle=0.01, move_time=0.15,
    )
    controller.backend.disconnect()
    result = {r.name: r for r in report.results}["accuracy vs kinematics"]
    assert not result.passed
    assert result.value == pytest.approx(50.0, abs=2.0)   # mm


def test_missing_marker_is_reported_not_silently_passed():
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]]), max_speed=8.0),
                               SafetyLimits(), control_hz=200.0)
    controller.start()
    camera = MockCamera(320, 240, 60)
    camera.start()
    report = check_against_arm(
        camera=camera, controller=controller,
        locate_marker=lambda img: None, poses=calibration_poses(4),
        report=Report(), settle=0.01, move_time=0.1,
    )
    controller.backend.disconnect()
    assert any("NOT verified" in n for n in report.notes)


def test_report_serialises(tmp_path):
    camera, detector, locator, _, _ = rig()
    with camera:
        report = check_precision(camera, detector, locator, duration=0.8)
    path = tmp_path / "r.json"
    report.save(path)
    import json
    data = json.loads(path.read_text())
    assert "results" in data and "passed" in data


def test_preview_server_encodes_and_serves():
    from tlod.vision.preview import PreviewServer

    server = PreviewServer(port=8099, max_fps=1000)
    server.offer(np.full((48, 64, 3), 120, np.uint8))
    jpeg = server.latest()
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")   # JPEG magic


def test_preview_throttles():
    from tlod.vision.preview import PreviewServer

    server = PreviewServer(port=8098, max_fps=2.0)
    server.offer(np.zeros((16, 16, 3), np.uint8))
    first = server.latest()
    server.offer(np.full((16, 16, 3), 255, np.uint8))
    assert server.latest() is first, "second frame should have been throttled"


def test_preview_routes_take_precedence_over_the_index():
    """Anything else on the board publishes through this one server.

    `/` in particular has to be claimable: a caller with no frames to
    offer -- the scoreboard -- would otherwise serve an index page
    pointing at a stream that never produces a byte. A route that raises
    must not take the server down with it either, because the thing most
    likely to raise is a read of a policy that has just been torn down.
    """
    import socket
    import urllib.error
    import urllib.request

    from tlod.vision.preview import PreviewServer

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = PreviewServer(port=port)
    server.add_route("/", lambda: ("text/plain", b"mine"))
    server.add_route("/boom", lambda: 1 / 0)
    server.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/", timeout=3) as r:
            assert r.read() == b"mine"
        for path, code in (("/boom", 500), ("/nope", 404)):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + path, timeout=3)
            assert caught.value.code == code
    finally:
        server.stop()

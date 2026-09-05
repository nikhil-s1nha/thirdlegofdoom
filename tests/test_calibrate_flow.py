"""Calibration procedures.

Extrinsics are the calibration people get wrong, and getting them wrong
does not look like an error -- it looks like an arm that reaches
confidently for the wrong place. Worth testing the recovery end to end.
"""

import time

import numpy as np
import pytest

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.types import Frame
from tlod.vision.calibrate_flow import (
    calibration_poses, find_marker, run_extrinsics,
)
from tlod.vision.calibration import Intrinsics, Projector, synthetic_projector


class MarkerCamera:
    """Renders a green dot wherever the tool actually is."""

    def __init__(self, projector, controller, size=(1280, 720)):
        self.projector = projector
        self.controller = controller
        self.width, self.height = size
        self._n = 0

    def start(self): pass

    def stop(self): pass

    @property
    def resolution(self): return self.width, self.height

    def read(self):
        import cv2
        img = np.full((self.height, self.width, 3), 30, np.uint8)
        tip = model.fk(self.controller.state().q[:5])[:3, 3]
        uv = self.projector.project(tip)
        if uv is not None:
            cv2.circle(img, (int(uv[0]), int(uv[1])), 14, (70, 190, 90), -1)
        self._n += 1
        return Frame(image=img, stamp=time.perf_counter(), index=self._n)


def test_marker_found_and_absent():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (300, 220), 18, (70, 190, 90), -1)
    u, v = find_marker(img)
    assert abs(u - 300) < 2 and abs(v - 220) < 2
    assert find_marker(np.zeros((480, 640, 3), np.uint8)) is None


def test_marker_ignores_specks():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (100, 100), 2, (70, 190, 90), -1)
    assert find_marker(img, min_area=120) is None


def test_calibration_poses_are_reachable_and_spread():
    poses = calibration_poses()
    assert len(poses) >= 8
    for p in poses:
        assert model.ik_position(p.xyz(), model.HOME).ok, f"{p} unreachable"
    zs = {round(p.z, 3) for p in poses}
    assert len(zs) >= 3, "points confined to too few heights leave the solve ill-conditioned"


def test_extrinsics_recovers_a_known_camera_pose():
    """The end-to-end procedure, against a camera whose pose we know."""
    truth = synthetic_projector((1280, 720), (0.15, -0.45, 0.55), (0.22, 0.0, 0.0))
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]]), max_speed=8.0),
                               SafetyLimits(), control_hz=200.0)
    controller.start()
    camera = MarkerCamera(truth, controller)

    extr, residuals = run_extrinsics(
        camera, controller, truth.intr, settle=0.02, move_time=0.15
    )
    controller.backend.disconnect()

    assert np.linalg.norm(extr.t - truth.extr.t) < 0.01, (
        f"camera placed at {extr.t}, truth {truth.extr.t}"
    )
    angle = np.degrees(np.arccos(np.clip((np.trace(extr.R.T @ truth.extr.R) - 1) / 2, -1, 1)))
    assert angle < 1.0, f"orientation off by {angle:.2f} deg"
    assert max(residuals) < 5.0


def test_extrinsics_refuses_with_too_few_points():
    """Better to fail loudly than to return a confident wrong transform."""
    truth = synthetic_projector()
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]]), max_speed=8.0),
                               SafetyLimits(), control_hz=200.0)
    controller.start()
    blind = MarkerCamera(truth, controller)
    with pytest.raises(RuntimeError, match="usable points"):
        run_extrinsics(blind, controller, truth.intr, settle=0.0, move_time=0.1,
                       locate=lambda img: None)
    controller.backend.disconnect()

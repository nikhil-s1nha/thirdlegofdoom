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
from tlod.vision.calibration import synthetic_projector


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

    extr, residuals, offset, naive_rms = run_extrinsics(
        camera, controller, truth.intr, settle=0.02, move_time=0.15
    )
    controller.backend.disconnect()

    assert np.linalg.norm(extr.t - truth.extr.t) < 0.01, (
        f"camera placed at {extr.t}, truth {truth.extr.t}"
    )
    angle = np.degrees(np.arccos(np.clip((np.trace(extr.R.T @ truth.extr.R) - 1) / 2, -1, 1)))
    assert angle < 1.0, f"orientation off by {angle:.2f} deg"
    assert max(residuals) < 5.0
    # This camera puts the marker exactly at the tool point, so solving
    # for an offset must not invent one.
    assert np.linalg.norm(offset) < 0.005, f"invented a {offset} m offset"


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


def test_a_marker_off_the_tool_point_is_solved_for_not_absorbed():
    """The 8.6 px extrinsics against a 0.165 px intrinsics, explained.

    The marker is tape on a jaw, not the point forward kinematics
    reports. A *fixed* offset in the base frame would be harmless -- the
    solve absorbs it into the camera position and reprojects perfectly,
    just placing the camera slightly wrong. What makes it bite is that
    the gripper rotates between poses, so the same offset points
    somewhere different each time and no single camera pose explains all
    of them.
    """
    from tlod.vision.calibration import (
        solve_extrinsics,
        solve_extrinsics_with_marker_offset,
    )

    truth = synthetic_projector((640, 480), (0.42, 0.04, 0.44), (0.22, 0.0, 0.1))
    rng = np.random.default_rng(0)

    tool_t, tool_R = [], []
    for pose in calibration_poses():
        r = model.ik_position(pose.xyz())
        if not r.ok:
            continue
        T = model.fk(r.q)
        tool_t.append(T[:3, 3])
        tool_R.append(T[:3, :3])
    tool_t, tool_R = np.array(tool_t), np.array(tool_R)

    offset = np.array([0.0, 0.012, -0.008])          # 14 mm of tape on a jaw
    marker = tool_t + np.einsum("nij,j->ni", tool_R, offset)
    pixels = np.array([truth.project(m) for m in marker]) + rng.normal(0, 0.3, (len(marker), 2))

    naive = solve_extrinsics(tool_t, pixels, truth.intr)
    fixed, found = solve_extrinsics_with_marker_offset(tool_t, tool_R, pixels, truth.intr)

    assert fixed.rms < naive.rms / 1.5, (
        f"solving for the offset did not help: {naive.rms:.2f} -> {fixed.rms:.2f} px")
    naive_err = np.linalg.norm(naive.t - truth.extr.t)
    fixed_err = np.linalg.norm(fixed.t - truth.extr.t)
    assert fixed_err < naive_err, (
        f"camera placed no better: {naive_err * 1e3:.1f} -> {fixed_err * 1e3:.1f} mm")
    # The offset is partly degenerate with camera position -- both move the
    # marker in the image -- so this recovers its scale, not its exact value.
    assert 0.004 < np.linalg.norm(found) < 0.030


def test_no_offset_means_no_offset_invented():
    from tlod.vision.calibration import solve_extrinsics_with_marker_offset

    truth = synthetic_projector((640, 480), (0.42, 0.04, 0.44), (0.22, 0.0, 0.1))
    rng = np.random.default_rng(1)
    tool_t, tool_R = [], []
    for pose in calibration_poses():
        r = model.ik_position(pose.xyz())
        if r.ok:
            T = model.fk(r.q)
            tool_t.append(T[:3, 3])
            tool_R.append(T[:3, :3])
    tool_t, tool_R = np.array(tool_t), np.array(tool_R)
    pixels = np.array([truth.project(t) for t in tool_t]) + rng.normal(0, 0.3, (len(tool_t), 2))

    extr, found = solve_extrinsics_with_marker_offset(tool_t, tool_R, pixels, truth.intr)
    assert np.linalg.norm(found) < 0.005, f"invented {np.linalg.norm(found) * 1e3:.1f} mm"
    assert np.linalg.norm(extr.t - truth.extr.t) < 0.01

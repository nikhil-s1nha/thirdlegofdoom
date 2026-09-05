"""Calibration, projection, scene and localisation tests."""

import numpy as np
import pytest

from tlod.types import Frame
from tlod.vision.calibration import Extrinsics, Intrinsics, Projector, solve_extrinsics, synthetic_projector
from tlod.vision.hands import HandLocator, Hand2D, INDEX_MCP, PINKY_MCP
from tlod.vision.objects import ColorBlobDetector
from tlod.vision.scene import HandPath, SceneHandDetector, SyntheticHandScene


@pytest.fixture
def projector():
    return synthetic_projector()


def test_projection_round_trip_is_exact(projector):
    for p in ([0.22, 0.0, 0.0], [0.15, 0.12, 0.0], [0.30, -0.10, 0.05]):
        p = np.array(p)
        uv = projector.project(p)
        assert uv is not None
        back = projector.pixel_to_plane(*uv, plane_z=p[2])
        assert np.allclose(back, p, atol=1e-6)


def test_points_behind_the_camera_are_rejected(projector):
    behind = projector.extr.t - projector.extr.R[:, 2] * 0.5
    assert projector.project(behind) is None


def test_ray_is_unit_length(projector):
    _, d = projector.ray(640, 360)
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_extrinsics_recovered_from_arm_points(projector):
    rng = np.random.default_rng(0)
    pts = np.array([[0.10 + 0.2 * rng.random(), -0.2 + 0.4 * rng.random(), 0.3 * rng.random()]
                    for _ in range(12)])
    px = np.array([projector.project(p) for p in pts])
    e = solve_extrinsics(pts, px, projector.intr)
    assert np.allclose(e.t, projector.extr.t, atol=1e-4)
    angle = np.degrees(np.arccos(np.clip((np.trace(e.R.T @ projector.extr.R) - 1) / 2, -1, 1)))
    assert angle < 0.01
    assert e.rms < 0.01


def test_extrinsics_needs_enough_points(projector):
    with pytest.raises(RuntimeError):
        solve_extrinsics(np.zeros((3, 3)), np.zeros((3, 2)), projector.intr)


def test_intrinsics_save_load(tmp_path, projector):
    path = tmp_path / "intr.npz"
    projector.intr.save(path)
    loaded = Intrinsics.load(path)
    assert np.allclose(loaded.K, projector.intr.K)
    assert loaded.resolution == projector.intr.resolution


def test_extrinsics_save_load(tmp_path, projector):
    path = tmp_path / "extr.npz"
    projector.extr.save(path)
    loaded = Extrinsics.load(path)
    assert np.allclose(loaded.R, projector.extr.R)
    assert np.allclose(loaded.t, projector.extr.t)


def test_size_depth_recovers_true_position(projector):
    scene = SyntheticHandScene(projector)
    truth = scene.position_at(0.4)
    hand = scene.hand2d_at(0.4, 0.0)
    got = HandLocator(projector, depth_mode="size").locate(hand).position
    assert np.linalg.norm(got - truth) < 1e-3


def test_plane_depth_matches_when_height_is_known(projector):
    scene = SyntheticHandScene(projector)
    truth = scene.position_at(0.7)
    hand = scene.hand2d_at(0.7, 0.0)
    loc = HandLocator(projector, depth_mode="plane", hand_height=float(truth[2]))
    assert np.linalg.norm(loc.locate(hand).position - truth) < 1e-6


def test_auto_depth_clamps_to_the_height_band(projector):
    """A partly occluded hand reports a collapsed width and thus absurd
    depth; auto mode must not hand that to the controller."""
    scene = SyntheticHandScene(projector)
    hand = scene.hand2d_at(0.3, 0.0)
    hand.landmarks[INDEX_MCP] = hand.landmarks[PINKY_MCP] + np.array([1.0, 0.0])
    got = HandLocator(projector, depth_mode="auto", height_band=(0.0, 0.40)).locate(hand)
    assert got is None or -1e-6 <= got.position[2] <= 0.40 + 1e-6


def test_scene_path_stays_inside_the_workspace():
    """The reason the old pixel-space path was replaced."""
    scene = SyntheticHandScene(synthetic_projector())
    for t in np.linspace(0, 12, 300):
        p = scene.position_at(t)
        r = float(np.hypot(p[0], p[1]))
        assert 0.08 <= r <= 0.33, f"t={t:.2f} radius {r:.3f} outside reach"
        assert 0.015 <= p[2] <= 0.45


def test_scene_velocity_matches_finite_difference():
    path = HandPath()
    v = path.velocity_at(1.0)
    num = (path.position_at(1.0 + 1e-5) - path.position_at(1.0 - 1e-5)) / 2e-5
    assert np.allclose(v, num, atol=1e-4)


def test_scene_detector_is_deterministic(projector):
    scene = SyntheticHandScene(projector)
    d1, d2 = SceneHandDetector(scene, t0=0.0), SceneHandDetector(scene, t0=0.0)
    f = Frame(np.zeros((4, 4, 3), np.uint8), 1.25)
    assert np.allclose(d1.detect(f)[0].landmarks, d2.detect(f)[0].landmarks)


def test_hand2d_palm_center_is_the_knuckle_centroid():
    lms = np.zeros((21, 2))
    lms[[0, 5, 9, 13, 17]] = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]
    assert np.allclose(Hand2D(lms, 1.0, "Right", 0.0).palm_center, [5.0, 5.0])


def test_color_blob_detects_a_disc_on_the_table(projector):
    import cv2

    img = np.zeros((720, 1280, 3), np.uint8)
    truth = np.array([0.22, 0.03, 0.0])
    uv = projector.project(truth)
    cv2.circle(img, (int(uv[0]), int(uv[1])), 40, (60, 60, 220), -1)  # BGR red
    found = ColorBlobDetector(projector, min_area_px=200).detect(Frame(img, 0.0))
    assert found, "no blob detected"
    best = found[0]
    assert best.label == "red"
    assert np.linalg.norm(best.position - truth) < 0.01
    assert best.radius > 0


def test_color_blob_ignores_noise(projector):
    img = np.zeros((480, 640, 3), np.uint8)
    assert ColorBlobDetector(projector).detect(Frame(img, 0.0)) == []

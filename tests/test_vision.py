"""Calibration, projection, scene and localisation tests."""

import cv2
import numpy as np
import pytest

from tlod.types import Detection, Frame
from tlod.vision.calibration import (
    Extrinsics, Intrinsics, Projector, solve_extrinsics,
    synthetic_projector,
)
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


def test_hand2d_palm_center_slides_along_the_wrist_to_knuckle_axis():
    """PALM_BIAS is the knob, and it is the only thing that moves the aim.

    Set from hardware rather than anatomy: 0.8 -- which is what a plain
    centroid of the wrist and four knuckles happens to give, nobody having
    chosen it -- landed on the edge of the index finger, and 0.5, the
    anatomical middle of the palm, was further off still.
    """
    from tlod.vision.hands import PALM_BIAS

    lms = np.zeros((21, 2))
    lms[0] = [0, 0]                                                  # wrist
    lms[[5, 9, 13, 17]] = [[0, 100], [0, 100], [0, 100], [0, 100]]   # knuckle line
    centre = Hand2D(lms, 1.0, "Right", 0.0).palm_center
    assert np.allclose(centre, [0.0, PALM_BIAS * 100.0]), centre

    # Laterally it is always the centre of the knuckles, whatever the
    # bias, so it never favours the index or the pinky edge.
    lms[[5, 9, 13, 17]] = [[0, 100], [10, 100], [20, 100], [30, 100]]
    centre = Hand2D(lms, 1.0, "Right", 0.0).palm_center
    assert np.isclose(centre[0], 15.0 * PALM_BIAS), centre


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


class TestFisheye:
    """A 145 deg lens cannot be described by the pinhole model at all.

    These check the two things that make a fisheye calibration usable
    rather than merely stored: that it round-trips through the projection
    it was fitted with, and that nothing silently falls back to pinhole
    maths on the way to a base-frame coordinate.
    """

    @staticmethod
    def _wide() -> Intrinsics:
        # Roughly the Arducam B0589: 145 deg across 640 px.
        f = (640 / 2.0) / np.tan(np.deg2rad(145.0) / 2.0)
        K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
        return Intrinsics(K, np.array([-0.02, 0.004, -0.001, 0.0002]),
                          (640, 480), 0.4, model="fisheye")

    def test_project_and_normalize_are_inverses(self):
        intr = self._wide()
        # Well off-axis, where the two models disagree most.
        points = np.array([[0.30, 0.22, 1.0], [-0.45, 0.05, 1.0], [0.02, -0.38, 1.0]])
        pixels = intr.project(points)
        back = intr.normalize(pixels)
        expected = points[:, :2] / points[:, 2:3]
        assert np.allclose(back, expected, atol=1e-3)

    def test_the_model_survives_a_save_and_load(self, tmp_path):
        intr = self._wide()
        path = tmp_path / "fisheye.npz"
        intr.save(path)
        loaded = Intrinsics.load(path)
        assert loaded.model == "fisheye" and loaded.fisheye
        assert np.allclose(loaded.dist, intr.dist)

    def test_a_file_without_a_model_key_loads_as_pinhole(self, tmp_path):
        """Calibrations shot before the fisheye model existed."""
        path = tmp_path / "old.npz"
        np.savez(path, K=np.eye(3), dist=np.zeros(5),
                 resolution=np.array([640, 480]), rms=0.5)
        assert Intrinsics.load(path).model == "pinhole"

    def test_pixels_round_trip_to_the_base_frame(self):
        """The whole chain, which is where a half-applied model hides: it
        reprojects beautifully near the optical axis and is centimetres
        out at the edge of the table."""
        intr = self._wide()
        extr = Extrinsics(np.eye(3), np.array([0.0, 0.0, 0.5]), 0.0)
        proj = Projector(intr, extr)
        for point in ([0.10, 0.05, 0.9], [-0.28, 0.20, 0.7], [0.35, -0.30, 1.2]):
            pixel = proj.project(np.array(point))
            assert pixel is not None
            origin, direction = proj.ray(*pixel)
            offset = np.array(point) - origin
            # The ray must pass through the point it came from.
            cross = np.linalg.norm(np.cross(direction, offset))
            assert cross < 1e-3, f"{point} came back {cross * 1000:.2f} mm off the ray"


def test_a_detection_beyond_reach_is_dropped():
    """A pixel is a ray, and the higher up the frame it sits the more
    shallowly it meets the table -- so something on a wall or held up
    resolves to a point far past the arm, with an ordinary confidence and
    nothing downstream able to tell. Measured on hardware: a red object
    reported 670 mm and then 1010 mm behind the base of an arm whose
    reach is 330 mm, and the arm set off after it."""
    from tlod.vision.objects import ColorBlobDetector

    projector = synthetic_projector()
    w, h = projector.intr.resolution

    def blob_at(row):
        image = np.zeros((h, w, 3), np.uint8)
        cv2.rectangle(image, (w // 2 - 20, row), (w // 2 + 20, row + 40),
                      (0, 255, 0), -1)
        return Frame(image=image, stamp=0.0, index=0)

    # Higher in frame is further away on the table: 0.57 m against 0.22 m.
    far, near = blob_at(4), blob_at(400)
    ungated = ColorBlobDetector(projector, min_area_px=50, max_range=0.0)
    assert np.hypot(*ungated.detect(far)[0].position[:2]) > 0.5
    assert np.hypot(*ungated.detect(near)[0].position[:2]) < 0.3

    gated = ColorBlobDetector(projector, min_area_px=50, max_range=0.4)
    assert gated.detect(far) == [], "kept a detection past the arm's reach"
    assert gated.detect(near), "dropped one the arm can reach"


def test_one_object_split_by_a_highlight_is_one_detection():
    """A specular highlight can cut a glossy piece into two contours
    despite the morphological close, and each half then arrives as its
    own detection a few millimetres from its twin -- indistinguishable
    downstream from two real objects. Seen on hardware as a single blue
    block touched twice, 2.5 mm and 5.2 mm off centre."""
    from tlod.vision.objects import ColorBlobDetector

    detector = ColorBlobDetector(synthetic_projector())
    at = lambda x, y, r, c: Detection(  # noqa: E731
        label="blue", position=np.array([x, y, 0.0]), stamp=0.0,
        confidence=c, radius=r)

    halves = [at(0.220, 0.100, 0.020, 0.9), at(0.228, 0.100, 0.012, 0.6)]
    merged = detector._merge_overlapping(halves)
    assert len(merged) == 1
    # Area-weighted, so the centre sits nearer the larger fragment than
    # the midpoint of the two.
    assert 0.220 < merged[0].position[0] < 0.224
    assert merged[0].confidence == 0.9

    apart = [at(0.20, 0.10, 0.015, 0.9), at(0.20, -0.10, 0.015, 0.8)]
    assert len(detector._merge_overlapping(apart)) == 2, "merged two real objects"

    other = [at(0.220, 0.100, 0.020, 0.9),
             Detection(label="red", position=np.array([0.221, 0.100, 0.0]),
                       stamp=0.0, confidence=0.8, radius=0.02)]
    assert len(detector._merge_overlapping(other)) == 2, "merged across colours"


def test_closing_the_hand_detector_twice_is_harmless():
    """MediaPipe's HandLandmarker also closes itself from __del__, which
    on 1.0.x runs during interpreter shutdown -- when the globals its
    dispatcher needs are already None, so it raises from inside a
    destructor. Python prints that and continues, so a clean run ends in
    a traceback that looks like a failure and is not one."""
    from tlod.vision.hands import MediaPipeHandDetector

    class Landmarker:
        closes = 0

        def close(self):
            Landmarker.closes += 1

    detector = MediaPipeHandDetector.__new__(MediaPipeHandDetector)
    detector._landmarker = Landmarker()
    detector.close()
    detector.close()
    assert Landmarker.closes == 1


def test_a_wrong_plane_height_moves_the_hand_sideways(projector):
    """Why "it keeps aiming at my index finger" was a calibration bug.

    With depth_mode: plane the pixel ray is intersected against an assumed
    height. The camera looks down at an angle, so getting that height
    wrong slides the answer *sideways* along the table -- always the same
    direction, since the camera does not move. No landmark choice can
    correct it, which is what two sessions of turning PALM_BIAS found out
    the slow way. The tell is that the offset stays put when the hand
    rotates.
    """
    truth = np.array([0.22, 0.0, 0.028])
    u, v = projector.project(truth)

    exact = projector.pixel_to_plane(u, v, 0.028)
    assert np.linalg.norm(exact - truth) < 1e-6

    # 6 mm low, which is what configs/opi.yaml had before it was measured.
    wrong = projector.pixel_to_plane(u, v, 0.022)
    sideways = float(np.linalg.norm((wrong - truth)[:2]))
    assert sideways > 0.003, (
        "this projector is too close to straight down for the test to mean "
        "anything; the real rig sees ~0.87 mm per mm")

    # And the direction is fixed by the camera, not by the hand: the same
    # height error at a different spot on the table pushes the same way.
    other = np.array([0.24, 0.06, 0.028])
    u2, v2 = projector.project(other)
    drift_a = (wrong - truth)[:2]
    drift_b = (projector.pixel_to_plane(u2, v2, 0.022) - other)[:2]
    cos = float(np.dot(drift_a, drift_b) /
                (np.linalg.norm(drift_a) * np.linalg.norm(drift_b)))
    assert cos > 0.9, f"drift direction is not consistent across the table ({cos:.2f})"


def test_list_cameras_only_probes_capture_nodes(monkeypatch):
    """Opening every /dev/videoN in a range is what made `tlod cameras` hang.

    The Rockchip codec nodes on this board accept the open and never
    answer, so the command printed its first line and stopped -- which
    reads exactly like the camera being broken. Only nodes that claim
    V4L2_CAP_VIDEO_CAPTURE are worth touching, and nothing else was ever
    going to be a camera.
    """
    from tlod.vision import camera as cam

    opened = []

    class FakeCap:
        def __init__(self, index, *a):
            opened.append(index)
            self.index = index

        def isOpened(self):
            return self.index == 4

        def release(self):
            pass

    monkeypatch.setattr(cam, "capture_nodes", lambda: {4: "Arducam", 9: "rkvdec"})
    monkeypatch.setattr(cam.cv2, "VideoCapture", FakeCap)
    assert cam.list_cameras() == [4]
    assert opened == [4, 9], "probed something that is not a capture node"


def test_stable_paths_maps_by_id_links_to_indices(monkeypatch, tmp_path):
    """The index is enumeration order; by-id is the device's own descriptor.

    Measured on the rig: remounting the camera moved it from 1 to 0, and
    every command naming an index was then pointing at a video decoder.
    """
    import os

    from tlod.vision import camera as cam

    link = tmp_path / "usb-Arducam_B0589-video-index0"
    link.write_text("")
    monkeypatch.setattr(cam.glob if hasattr(cam, "glob") else os, "path", os.path)
    monkeypatch.setattr("glob.glob", lambda pat: [str(link)])
    monkeypatch.setattr(os.path, "realpath", lambda p: "/dev/video3")
    assert cam.stable_paths() == {3: str(link)}


def test_stable_paths_ignores_links_that_are_not_video_nodes(monkeypatch):
    """by-id also carries media and metadata nodes, which never open."""
    import os

    from tlod.vision import camera as cam

    monkeypatch.setattr("glob.glob", lambda pat: ["/dev/v4l/by-id/x", "/dev/v4l/by-id/y"])
    monkeypatch.setattr(os.path, "realpath",
                        lambda p: "/dev/media0" if p.endswith("x") else "/dev/video7")
    assert cam.stable_paths() == {7: "/dev/v4l/by-id/y"}


def test_camera_arg_takes_an_index_or_a_path():
    """`--camera 1` and `--camera /dev/v4l/by-id/...` both have to work;
    cv2.VideoCapture accepts either and the path is the one that lasts."""
    from tlod.cli import camera_arg

    assert camera_arg("1") == 1
    assert camera_arg("0") == 0
    path = "/dev/v4l/by-id/usb-Arducam_B0589_4K_HDR-video-index0"
    assert camera_arg(path) == path

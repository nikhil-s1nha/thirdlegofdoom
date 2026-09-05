"""Interactive calibration procedures.

Two jobs, and the second one is the one people get wrong.

**Intrinsics** describe the lens. Shot once per camera and resolution,
never again unless you change either. A chessboard, the standard OpenCV
routine, nothing surprising.

**Extrinsics** place the camera relative to the robot's base. This has to
be redone every time either moves, which on a tabletop robot is often,
and it is the difference between the arm reaching for what you see and
reaching two centimetres to the left of it.

The extrinsics routine here uses **the arm as its own calibration
target**: it drives the tool to a spread of known configurations and
looks for a marker on the gripper in each frame. Forward kinematics
supplies the 3D coordinates, so they are expressed in exactly the frame
the controller commands in. Any constant error in the arm model then
cancels out instead of becoming a permanent offset between what the
camera sees and where the arm goes. A chessboard on the table cannot do
that -- it is only as accurate as your ruler, and it measures the wrong
frame.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from tlod.arm import model
from tlod.vision.calibration import (
    Extrinsics,
    Intrinsics,
    calibrate_intrinsics,
    find_chessboard,
    solve_extrinsics,
)
from tlod.types import Pose

log = logging.getLogger(__name__)

# HSV band for the gripper marker. Green by default because skin, wood
# and most tabletops are not green -- red would fight with hands.
MARKER_HSV = ((40, 90, 60), (85, 255, 255))


def _spread_enough(corners: np.ndarray, previous: list[np.ndarray], min_shift: float) -> bool:
    """True if this board view differs enough from the ones already kept.

    Fifteen views of the board in nearly the same place calibrate nothing
    -- the solver needs the pattern at varied positions, distances and
    tilts to separate focal length from distortion.
    """
    centre = corners.reshape(-1, 2).mean(axis=0)
    return all(np.linalg.norm(centre - p.reshape(-1, 2).mean(axis=0)) > min_shift for p in previous)


def run_intrinsics(
    camera,
    pattern: tuple[int, int] = (9, 6),
    square: float = 0.025,
    views: int = 15,
    min_shift: float = 60.0,
    timeout: float = 180.0,
    on_progress=None,
) -> Intrinsics:
    """Collect chessboard views until there are enough, then calibrate.

    Auto-captures rather than waiting for keypresses, so it works over
    SSH on a headless board. Move the board slowly around the frame:
    corners, edges, close, far, tilted.
    """
    kept_images: list[np.ndarray] = []
    kept_corners: list[np.ndarray] = []
    deadline = time.perf_counter() + timeout
    last_index = -1

    while len(kept_images) < views and time.perf_counter() < deadline:
        frame = camera.read()
        if frame is None or frame.index == last_index:
            time.sleep(0.005)
            continue
        last_index = frame.index

        corners = find_chessboard(frame.image, pattern)
        if corners is None:
            continue
        if not _spread_enough(corners, kept_corners, min_shift):
            continue

        kept_images.append(frame.image.copy())
        kept_corners.append(corners)
        if on_progress:
            on_progress(len(kept_images), views, frame.image, corners)

    if len(kept_images) < 5:
        raise RuntimeError(
            f"only captured {len(kept_images)} usable views of the board in "
            f"{timeout:.0f}s. Check the pattern size matches --pattern (inner "
            "corners, so an 8x8 board is 7x7), and that the board is well lit "
            "and fully in frame."
        )
    return calibrate_intrinsics(kept_images, pattern, square)


def find_marker(image: np.ndarray, hsv_band=MARKER_HSV, min_area: int = 120):
    """Centroid of the largest blob in the marker colour, or None."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_band[0]), np.array(hsv_band[1]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area:
        return None
    m = cv2.moments(biggest)
    if m["m00"] == 0:
        return None
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def calibration_poses(count: int = 12) -> list[Pose]:
    """A spread of tool positions covering the working volume.

    Spread matters more than count. Points clustered in a plane leave the
    solve poorly conditioned along the camera axis, which shows up as an
    extrinsic that reprojects beautifully and puts the arm in the wrong
    place the moment it changes height.
    """
    poses: list[Pose] = []
    for z in (0.06, 0.14, 0.22):
        for x, y in ((0.16, -0.10), (0.26, -0.06), (0.26, 0.06), (0.16, 0.10)):
            poses.append(Pose(x, y, z))
    return poses[:count] if count < len(poses) else poses


def wait_until_still(controller, tolerance: float = 0.01, timeout: float = 3.0) -> bool:
    """Block until the arm stops moving, or the timeout expires.

    Calibration pairs a camera frame with a forward-kinematics
    coordinate, and those are two separate reads. If the arm is still
    drifting between them the pair is inconsistent, and inconsistent
    correspondences do not produce a noisy calibration -- they produce a
    confidently wrong one, because the solver has no way to know the two
    halves disagree.

    Waiting on measured velocity rather than sleeping a fixed interval
    also survives a heavier payload or a slower joint, where a hardcoded
    delay silently stops being enough.
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        state = controller.state()
        if state.dq is None or float(np.max(np.abs(state.dq))) < tolerance:
            return True
        time.sleep(0.02)
    return False


def run_extrinsics(
    camera,
    controller,
    intrinsics: Intrinsics,
    poses: list[Pose] | None = None,
    settle: float = 0.4,
    move_time: float = 2.0,
    locate=None,
    on_progress=None,
) -> tuple[Extrinsics, list[float]]:
    """Drive the arm to known poses and find the marker in each frame.

    `locate(image) -> (u, v) | None` defaults to colour-blob detection of
    a green marker on the gripper. Pass your own to click manually.

    Returns the extrinsics and the per-point reprojection errors, which
    are the honest quality signal: a low RMS with one wild outlier means
    one mislocated marker, not a bad camera.
    """
    poses = poses or calibration_poses()
    locate = locate or find_marker

    points_base: list[np.ndarray] = []
    points_image: list[tuple[float, float]] = []

    for i, pose in enumerate(poses, 1):
        if not controller.goto_pose(pose, duration=move_time):
            log.warning("pose %d unreachable, skipping: %s", i, pose)
            continue
        if not wait_until_still(controller):
            log.warning("arm still moving at pose %d; skipping", i)
            continue
        # Then a short extra pause: the structure rings after the joints
        # stop, and the camera may still be adjusting exposure. A blurred
        # marker biases the centroid.
        time.sleep(settle)

        frame = camera.read()
        if frame is None:
            log.warning("no frame at pose %d", i)
            continue

        uv = locate(frame.image)
        if uv is None:
            log.warning("marker not found at pose %d; skipping", i)
            continue

        # FK, not the requested pose: the arm lands where it lands, and
        # using the commanded value would fold servo error into the
        # camera calibration.
        actual = model.fk(controller.state().q[:5])[:3, 3]
        points_base.append(actual)
        points_image.append(uv)
        if on_progress:
            on_progress(i, len(poses), frame.image, uv, actual)

    if len(points_base) < 6:
        raise RuntimeError(
            f"only {len(points_base)} usable points. Check the marker is "
            "visible from the camera through the whole motion, is the only "
            "thing that colour in frame, and is not occluded by the arm."
        )

    extrinsics = solve_extrinsics(np.array(points_base), np.array(points_image), intrinsics)

    from tlod.vision.calibration import Projector

    projector = Projector(intrinsics, extrinsics)
    residuals = []
    for base, (u, v) in zip(points_base, points_image, strict=True):
        projected = projector.project(base)
        residuals.append(
            float(np.hypot(projected[0] - u, projected[1] - v)) if projected else float("inf")
        )
    return extrinsics, residuals

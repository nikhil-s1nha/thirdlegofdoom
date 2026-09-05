"""Camera calibration and the pixel -> robot-frame mapping.

Two separate problems, often conflated:

  intrinsics  what the lens does. Chessboard, standard OpenCV, done once
              per camera and never again unless you change the lens or
              resolution.

  extrinsics  where the camera sits relative to the robot's base. This is
              the one that actually matters and the one people get wrong.
              It must be redone every time the camera or the arm is moved,
              which for a tabletop robot is often.

Two ways to get extrinsics are provided. `extrinsics_from_board` needs a
chessboard placed at a measured offset from the base -- quick, but only as
accurate as your ruler. `extrinsics_from_arm_points` instead drives the arm
to a set of poses and uses forward kinematics for the 3D coordinates, so
the robot measures itself. That is both more accurate and less error-prone,
and it is the recommended path.

Depth from a single camera
--------------------------
One camera cannot measure depth. Every pixel is a ray. Turning a ray into a
point requires an assumption, and the honest thing is to make it explicit:
`pixel_to_plane` intersects the ray with a horizontal plane at a stated
height. For objects sitting on the table that plane is the table and the
result is exact. For a hand hovering above it, the height is a guess, and
the error is roughly (height error) x tan(viewing angle) -- which is why a
steeply angled overhead mount is much more forgiving here than a shallow
side view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class Intrinsics:
    K: np.ndarray            # 3x3
    dist: np.ndarray         # (5,) or (8,)
    resolution: tuple[int, int]
    rms: float = 0.0         # reprojection error from calibration, pixels

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, K=self.K, dist=self.dist, resolution=np.array(self.resolution), rms=self.rms)

    @classmethod
    def load(cls, path: str | Path) -> "Intrinsics":
        d = np.load(path)
        return cls(d["K"], d["dist"], tuple(int(v) for v in d["resolution"]), float(d["rms"]))

    @classmethod
    def approximate(cls, resolution: tuple[int, int], hfov_deg: float = 70.0) -> "Intrinsics":
        """A plausible pinhole model from the advertised field of view.

        For bring-up only, so the pipeline runs end to end before you have
        shot a calibration board. Distortion is assumed zero, which for a
        C922 at 720p costs a few pixels at the edges. Do not ship it.
        """
        w, h = resolution
        f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
        return cls(K, np.zeros(5), resolution, rms=float("nan"))


@dataclass(slots=True)
class Extrinsics:
    """Rigid transform placing the camera in the robot base frame."""

    R: np.ndarray            # 3x3, camera -> base rotation
    t: np.ndarray            # (3,), camera origin in base coordinates
    rms: float = 0.0

    @property
    def T(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, R=self.R, t=self.t, rms=self.rms)

    @classmethod
    def load(cls, path: str | Path) -> "Extrinsics":
        d = np.load(path)
        return cls(d["R"], d["t"], float(d["rms"]))


CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def find_chessboard(image: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Sub-pixel inner-corner locations, or None. `pattern` is (cols, rows)
    of *inner* corners -- an 8x8 board has a 7x7 pattern."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = cv2.findChessboardCorners(gray, pattern, CHESSBOARD_FLAGS)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def _board_object_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def calibrate_intrinsics(
    images: list[np.ndarray], pattern: tuple[int, int] = (9, 6), square: float = 0.025
) -> Intrinsics:
    """Standard chessboard intrinsics. Aim for 15+ views covering the whole
    frame, especially the corners, at varied tilts."""
    objp = _board_object_points(pattern, square)
    obj_points, img_points = [], []
    shape = None
    for img in images:
        corners = find_chessboard(img, pattern)
        if corners is None:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        shape = img.shape[1::-1]
    if len(obj_points) < 5:
        raise RuntimeError(f"only {len(obj_points)} usable views; need at least 5 (ideally 15+)")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, shape, None, None)
    return Intrinsics(K, dist.ravel(), shape, float(rms))


def extrinsics_from_board(
    image: np.ndarray,
    intr: Intrinsics,
    board_origin_in_base: np.ndarray,
    pattern: tuple[int, int] = (9, 6),
    square: float = 0.025,
    board_rotation: np.ndarray | None = None,
) -> Extrinsics:
    """Locate the camera from one view of a board at a known base-frame pose."""
    corners = find_chessboard(image, pattern)
    if corners is None:
        raise RuntimeError("chessboard not found")
    objp = _board_object_points(pattern, square)
    R_board = np.eye(3) if board_rotation is None else board_rotation
    pts_base = (R_board @ objp.T).T + np.asarray(board_origin_in_base, float)
    return solve_extrinsics(pts_base, corners.reshape(-1, 2), intr)


def solve_extrinsics(
    points_base: np.ndarray, points_image: np.ndarray, intr: Intrinsics
) -> Extrinsics:
    """PnP from >=4 correspondences between base-frame points and pixels."""
    points_base = np.asarray(points_base, np.float64).reshape(-1, 3)
    points_image = np.asarray(points_image, np.float64).reshape(-1, 2)
    if len(points_base) < 4:
        raise RuntimeError(f"need at least 4 correspondences, got {len(points_base)}")

    ok, rvec, tvec = cv2.solvePnP(
        points_base, points_image, intr.K, intr.dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    if len(points_base) >= 6:
        rvec, tvec = cv2.solvePnPRefineLM(points_base, points_image, intr.K, intr.dist, rvec, tvec)

    # solvePnP gives base -> camera; we want the camera placed in base.
    R_cb, _ = cv2.Rodrigues(rvec)
    R = R_cb.T
    t = (-R_cb.T @ tvec).ravel()

    proj, _ = cv2.projectPoints(points_base, rvec, tvec, intr.K, intr.dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - points_image) ** 2, axis=1))))
    return Extrinsics(R, t, rms)


def extrinsics_from_arm_points(
    tcp_points_base: np.ndarray, pixels: np.ndarray, intr: Intrinsics
) -> Extrinsics:
    """Hand-eye style extrinsics using the arm itself as the calibration target.

    Drive the arm to N well-spread poses, record the FK tip position for
    each and where the tip appears in the image, and solve. Better than a
    board because the 3D points come from the robot's own kinematics, so
    the result is expressed in exactly the frame the controller commands
    in -- any constant error in the arm model cancels out instead of
    turning into a systematic offset between what the camera sees and
    where the arm goes.
    """
    return solve_extrinsics(tcp_points_base, pixels, intr)


class Projector:
    """Converts between pixels and the robot base frame."""

    def __init__(self, intr: Intrinsics, extr: Extrinsics) -> None:
        self.intr = intr
        self.extr = extr
        self._Kinv = np.linalg.inv(intr.K)

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """(origin, unit direction) in base coordinates for a pixel."""
        pts = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.intr.K, self.intr.dist).reshape(2)
        d_cam = np.array([undistorted[0], undistorted[1], 1.0])
        d_base = self.extr.R @ d_cam
        return self.extr.t.copy(), d_base / np.linalg.norm(d_base)

    def pixel_to_plane(self, u: float, v: float, plane_z: float = 0.0) -> np.ndarray | None:
        """Intersect the pixel ray with the horizontal plane z = plane_z.

        Returns None when the ray is parallel to the plane or points away
        from it, which happens for pixels above the horizon and is a real
        answer, not an error.
        """
        origin, direction = self.ray(u, v)
        if abs(direction[2]) < 1e-9:
            return None
        s = (plane_z - origin[2]) / direction[2]
        if s <= 0:
            return None
        return origin + s * direction

    def project(self, point_base: np.ndarray) -> tuple[float, float] | None:
        """Base-frame point -> pixel. None if it is behind the camera."""
        p = np.asarray(point_base, float).reshape(3)
        p_cam = self.extr.R.T @ (p - self.extr.t)
        if p_cam[2] <= 1e-6:
            return None
        rvec = np.zeros(3)
        proj, _ = cv2.projectPoints(p_cam.reshape(1, 3), rvec, np.zeros(3), self.intr.K, self.intr.dist)
        u, v = proj.reshape(2)
        return float(u), float(v)


def synthetic_projector(
    resolution: tuple[int, int] = (1280, 720),
    camera_position=(0.15, -0.45, 0.55),
    look_at=(0.22, 0.0, 0.0),
) -> Projector:
    """A plausible overhead-angled camera, for tests and simulation."""
    intr = Intrinsics.approximate(resolution)
    cam = np.asarray(camera_position, float)
    target = np.asarray(look_at, float)

    forward = target - cam
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    # OpenCV camera axes: x right, y down, z forward.
    R = np.column_stack([right, down, forward])
    return Projector(intr, Extrinsics(R, cam, rms=0.0))

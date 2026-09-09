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

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class Intrinsics:
    K: np.ndarray            # 3x3
    dist: np.ndarray         # pinhole: (5,) or (8,).  fisheye: (4,)
    resolution: tuple[int, int]
    rms: float = 0.0         # reprojection error from calibration, pixels
    # Which lens model the coefficients belong to. The default five-term
    # Brown-Conrady model is a perturbation of a pinhole and stops being
    # able to describe a lens somewhere around 120 degrees: the fit does
    # not merely get worse, it cannot represent the shape at all, and the
    # residual piles up at the frame edges while the centre still looks
    # excellent. A 145-degree lens needs the equidistant fisheye model.
    #
    # Carried on the intrinsics rather than chosen at each call site so a
    # fisheye calibration cannot be paired with pinhole projection --
    # which would reproject beautifully near the optical axis and put the
    # arm centimetres out at the edge of the table, with nothing failing.
    model: str = "pinhole"   # "pinhole" | "fisheye"

    @property
    def fisheye(self) -> bool:
        return self.model == "fisheye"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, K=self.K, dist=self.dist, resolution=np.array(self.resolution),
                 rms=self.rms, model=self.model)

    @classmethod
    def load(cls, path: str | Path) -> Intrinsics:
        d = np.load(path)
        # Files written before the fisheye model existed have no "model"
        # key and are all pinhole.
        model = str(d["model"]) if "model" in d.files else "pinhole"
        return cls(d["K"], d["dist"], tuple(int(v) for v in d["resolution"]),
                   float(d["rms"]), model)

    # -- the lens, applied ---------------------------------------------------
    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels -> undistorted normalised image coordinates, shape (N, 2).

        The inverse of the lens: what the pixel would have been on an
        ideal pinhole camera of focal length 1.
        """
        pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(pts, self.K, self.dist.reshape(4, 1))
        else:
            out = cv2.undistortPoints(pts, self.K, self.dist)
        return out.reshape(-1, 2)

    def project(self, points: np.ndarray, rvec=None, tvec=None) -> np.ndarray:
        """Points in some frame -> pixels, shape (N, 2).

        `rvec`/`tvec` place those points relative to the camera; omit both
        when the points are already in camera coordinates.
        """
        pts = np.asarray(points, np.float64).reshape(-1, 1, 3)
        rvec = np.zeros(3) if rvec is None else np.asarray(rvec, np.float64)
        tvec = np.zeros(3) if tvec is None else np.asarray(tvec, np.float64)
        if self.fisheye:
            # cv2.fisheye.projectPoints insists on its own shapes and,
            # unlike the pinhole version, returns (N, 1, 2) either way.
            out, _ = cv2.fisheye.projectPoints(
                pts.reshape(-1, 1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1),
                self.K, self.dist.reshape(4, 1),
            )
        else:
            out, _ = cv2.projectPoints(pts, rvec, tvec, self.K, self.dist)
        return out.reshape(-1, 2)

    @classmethod
    def approximate(cls, resolution: tuple[int, int], hfov_deg: float = 70.0) -> Intrinsics:
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
    def load(cls, path: str | Path) -> Extrinsics:
        d = np.load(path)
        return cls(d["R"], d["t"], float(d["rms"]))


CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def _calib_flag(name: str) -> int:
    """A calibration flag, wherever this OpenCV keeps it.

    4.x exposes the fisheye flags on `cv2.fisheye`; 5.0 moved them to the
    top-level namespace and left `cv2.fisheye` with none, so referring to
    either one directly breaks on the other. Missing entirely resolves to
    0, which drops that flag rather than failing the calibration.
    """
    for holder in (cv2.fisheye, cv2):
        value = getattr(holder, name, None)
        if value is not None:
            return int(value)
    return 0


# RECOMPUTE_EXTRINSIC re-solves each view's pose between iterations, which
# a wide lens needs to converge; FIX_SKEW pins the skew term to zero,
# since no real sensor has any and leaving it free just absorbs noise.
#
# USE_INTRINSIC_GUESS is not optional here. The solver's own linear
# initialisation assumes a near-pinhole geometry, and on a 145 degree lens
# it lands nowhere near: measured against a synthetic camera of known
# focal length 101 px, starting cold gives 317 px and an RMS of 118, or
# fails outright inside InitExtrinsics. Seeded from the advertised field
# of view it recovers the focal length exactly. The advertised number only
# has to be roughly right -- it is a starting point, not an answer.
FISHEYE_CALIB_FLAGS = (
    _calib_flag("CALIB_RECOMPUTE_EXTRINSIC")
    | _calib_flag("CALIB_FIX_SKEW")
    | _calib_flag("CALIB_USE_INTRINSIC_GUESS")
)


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
    images: list[np.ndarray],
    pattern: tuple[int, int] = (9, 6),
    square: float = 0.025,
    fisheye: bool = False,
    hfov_deg: float = 145.0,
) -> Intrinsics:
    """Chessboard intrinsics. Aim for 15+ views covering the whole frame,
    especially the corners, at varied tilts.

    Set `fisheye` for a lens much wider than about 120 degrees. The
    pinhole model does not degrade gracefully past that point -- it
    cannot express the projection at all, and no number of views brings
    the residual down. `hfov_deg` then seeds the solve from the lens's
    advertised field of view; see FISHEYE_CALIB_FLAGS for why that is
    required rather than merely helpful. It is ignored for pinhole.
    """
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

    if not fisheye:
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, shape, None, None)
        return Intrinsics(K, dist.ravel(), shape, float(rms))

    # cv2.fisheye wants (1, N, C), not the (N, 1, C) its pinhole
    # counterpart takes, and rejects the latter with a size mismatch from
    # inside the arithmetic rather than anything nameable.
    K = Intrinsics.approximate(shape, hfov_deg).K
    D = np.zeros((4, 1))
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        [np.ascontiguousarray(p.reshape(1, -1, 3), np.float64) for p in obj_points],
        [np.ascontiguousarray(p.reshape(1, -1, 2), np.float64) for p in img_points],
        shape, K, D,
        flags=FISHEYE_CALIB_FLAGS,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    return Intrinsics(K, D.ravel(), shape, float(rms), model="fisheye")


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

    # solvePnP has no fisheye form, so undo the lens first and solve
    # against an ideal camera. For pinhole this is the same computation
    # either way; doing it uniformly keeps one path.
    normalized = intr.normalize(points_image).reshape(-1, 1, 2)
    eye, none = np.eye(3), np.zeros(5)

    ok, rvec, tvec = cv2.solvePnP(
        points_base, normalized, eye, none, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    if len(points_base) >= 6:
        rvec, tvec = cv2.solvePnPRefineLM(points_base, normalized, eye, none, rvec, tvec)

    # solvePnP gives base -> camera; we want the camera placed in base.
    R_cb, _ = cv2.Rodrigues(rvec)
    R = R_cb.T
    t = (-R_cb.T @ tvec).ravel()

    # Residuals in pixels, through the real lens -- the number a person
    # judges the calibration by has to be in the units they can see.
    proj = intr.project(points_base, rvec, tvec)
    rms = float(np.sqrt(np.mean(np.sum((proj - points_image) ** 2, axis=1))))
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
        undistorted = self.intr.normalize([(u, v)])[0]
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
        u, v = self.intr.project(p_cam.reshape(1, 3))[0]
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

"""Kinematic model of the SO-101 arm.

Geometry is transcribed from the official URDF, `assets/so101_new_calib.urdf`
(TheRobotStudio/SO-ARM100, Simulation/SO101). `scripts/extract_urdf.py`
regenerates the CHAIN table below from that file, so the numbers here are
checkable rather than folklore.

Why this module exists instead of an off-the-shelf IK library
-------------------------------------------------------------
The arm is marketed as 6-DOF. It has six *motors*, but one drives the
gripper, so there are five arm joints:

    shoulder_pan    yaw about the base vertical
    shoulder_lift   |
    elbow_flex      |  three parallel pitch axes
    wrist_flex      |
    wrist_roll      roll about the tool axis

Five joints cannot span SE(3). The reachable task space is five-dimensional:
position (3) + tool pitch (1) + tool roll (1). Tool yaw is a dependent
variable, fixed by whichever base pan reaches the target. A general 6-DOF IK
solver handed a full pose target will therefore chase an unreachable
orientation and either fail or converge somewhere arbitrary. Modelling the
task space correctly at 5-D makes the problem square, well conditioned, and
fast.

Solver strategy: warm-started damped least squares on the exact forward
kinematics. Warm starting from the current measured configuration is what a
control loop wants anyway -- it gives continuity between successive solves
and so avoids elbow flips mid-motion. An analytic planar seed is used for
cold starts. Damping keeps it stable through the wrist-vertical singularity
of the pitch/roll parametrisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tlod.types import Pose

# --------------------------------------------------------------------------
# Chain, transcribed from so101_new_calib.urdf.
# Each entry: (joint name, translation xyz, fixed rotation rpy, axis).
# All revolute joints in this URDF rotate about their own local +z.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Link:
    name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None  # None => fixed joint


CHAIN: tuple[Link, ...] = (
    Link("shoulder_pan",   (0.0388353, -8.97657e-09, 0.0624),  (3.14159, 0.0, -3.14159), (0, 0, 1)),
    Link("shoulder_lift",  (-0.0303992, -0.0182778, -0.0542),  (-1.5708, -1.5708, 0.0),  (0, 0, 1)),
    Link("elbow_flex",     (-0.11257, -0.028, 0.0),            (0.0, 0.0, 1.5708),       (0, 0, 1)),
    Link("wrist_flex",     (-0.1349, 0.0052, 0.0),             (0.0, 0.0, -1.5708),      (0, 0, 1)),
    Link("wrist_roll",     (0.0, -0.0611, 0.0181),             (1.5708, 0.0486795, 3.14159), (0, 0, 1)),
    Link("tcp",            (-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0),     None),
)

# Joint limits from the URDF, radians, in ARM_JOINTS order.
JOINT_LIMITS: np.ndarray = np.array(
    [
        [-1.91986, 1.91986],   # shoulder_pan
        [-1.74533, 1.74533],   # shoulder_lift
        [-1.69000, 1.69000],   # elbow_flex
        [-1.65806, 1.65806],   # wrist_flex
        [-2.74385, 2.84121],   # wrist_roll
    ],
    dtype=float,
)
GRIPPER_LIMITS: tuple[float, float] = (-0.174533, 1.74533)

# Which local axis of the TCP frame points out of the gripper. Determined
# empirically from the URDF frames (see tests/test_kinematics.py, which
# pins it), not guessed.
APPROACH_AXIS: np.ndarray = np.array([0.0, 0.0, 1.0])
# Reference axis used to measure tool roll about the approach axis.
ROLL_REF_AXIS: np.ndarray = np.array([1.0, 0.0, 0.0])


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ]
    )


def _homogeneous(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Precompute the constant part of each link transform once at import.
_STATIC: tuple[np.ndarray, ...] = tuple(
    _homogeneous(_rpy_matrix(*link.rpy), link.xyz) for link in CHAIN
)
_IS_REVOLUTE: tuple[bool, ...] = tuple(link.axis is not None for link in CHAIN)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics. `q` is 5 arm joint angles; returns the 4x4 TCP pose."""
    T = np.eye(4)
    qi = 0
    for static, revolute in zip(_STATIC, _IS_REVOLUTE, strict=True):
        T = T @ static
        if revolute:
            T = T @ _rot_z(float(q[qi]))
            qi += 1
    return T


def fk_all(q: np.ndarray) -> list[np.ndarray]:
    """Every joint frame in the chain, base first. For visualisation and
    collision checks against the tabletop."""
    frames = []
    T = np.eye(4)
    qi = 0
    for static, revolute in zip(_STATIC, _IS_REVOLUTE, strict=True):
        T = T @ static
        if revolute:
            T = T @ _rot_z(float(q[qi]))
            qi += 1
        frames.append(T.copy())
    return frames


def pose_from_matrix(T: np.ndarray) -> Pose:
    """Project a 4x4 TCP transform onto the arm's true 5-D task space."""
    p = T[:3, 3]
    R = T[:3, :3]
    a = R @ APPROACH_AXIS                       # approach direction, base frame
    horiz = float(np.hypot(a[0], a[1]))
    pitch = float(np.arctan2(a[2], horiz))

    # Roll is measured about the approach axis, against a reference frame
    # that has no roll: the horizontal vector perpendicular to the approach
    # azimuth. Degenerate when the tool points straight up or down, which is
    # a genuine singularity of this parametrisation, not a bug.
    if horiz < 1e-9:
        s_ref = np.array([0.0, 1.0, 0.0])
    else:
        az = np.arctan2(a[1], a[0])
        s_ref = np.array([-np.sin(az), np.cos(az), 0.0])
    t_ref = np.cross(a, s_ref)
    tool_ref = R @ ROLL_REF_AXIS
    roll = float(np.arctan2(np.dot(tool_ref, t_ref), np.dot(tool_ref, s_ref)))
    return Pose(float(p[0]), float(p[1]), float(p[2]), pitch, roll)


def tool_pose(q: np.ndarray) -> Pose:
    return pose_from_matrix(fk(q))


def _task_vec(q: np.ndarray) -> np.ndarray:
    return tool_pose(q).as_vec()


def _wrap(a: float | np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def task_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Residual in task space, with the two angular rows wrapped."""
    e = target - current
    e[3:] = _wrap(e[3:])
    return e


def jacobian(q: np.ndarray, base: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """5x5 task Jacobian by forward differences on the exact FK.

    Finite differences rather than an analytic derivation: the pitch/roll
    parametrisation makes the analytic form long and easy to get subtly
    wrong, and at ~100 us this is not the bottleneck of a 100 Hz loop.
    Correctness beats cleverness. `base` is the task vector at `q`, passed
    in by the solver which has already computed it, halving the FK calls.
    """
    if base is None:
        base = _task_vec(q)
    J = np.empty((5, 5))
    qp = q.copy()
    for i in range(5):
        original = qp[i]
        qp[i] = original + eps
        J[:, i] = task_error(base, _task_vec(qp)) / eps
        qp[i] = original
    return J


def position_jacobian(q: np.ndarray) -> np.ndarray:
    """Exact 3x5 position Jacobian from one forward-kinematics pass.

    For a revolute joint with world axis z_i through point p_i, moving the
    joint sweeps the tool around that axis, so the tool's velocity
    contribution is z_i x (p_e - p_i). That is exact, not an
    approximation, and it needs the joint frames -- which one fk_all()
    call already produces.

    The finite-difference path costs six FK evaluations per Jacobian.
    This costs one. On a laptop that difference is irrelevant; on a
    Raspberry Pi Zero running the control loop it is the difference
    between fitting in the tick budget and not.

    Only position. Tool pitch and roll still go through finite
    differences, because their analytic derivatives are long and easy to
    get subtly wrong -- and position-only IK, which is what the games
    actually use, never needs them.
    """
    frames = fk_all(np.asarray(q, float))
    p_e = frames[-1][:3, 3]
    J = np.empty((3, 5))
    qi = 0
    for frame, revolute in zip(frames, _IS_REVOLUTE, strict=True):
        if not revolute:
            continue
        axis = frame[:3, 2]                 # joint's own z, in world
        J[:, qi] = np.cross(axis, p_e - frame[:3, 3])
        qi += 1
    return J


def clamp_to_limits(q: np.ndarray) -> np.ndarray:
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def within_limits(q: np.ndarray, margin: float = 0.0) -> bool:
    return bool(
        np.all(q >= JOINT_LIMITS[:, 0] + margin) and np.all(q <= JOINT_LIMITS[:, 1] - margin)
    )


# Reach envelope, measured from the chain rather than asserted, so it stays
# right if the geometry is ever regenerated.
def _measure_reach() -> tuple[float, float]:
    origin = fk(np.zeros(5))[:3, 3]
    base = _STATIC[0][:3, 3]
    far = 0.0
    for _ in range(2000):
        q = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        far = max(far, float(np.linalg.norm(fk(q)[:3, 3] - base)))
    return float(np.linalg.norm(origin - base)), far


MAX_REACH: float = 0.36  # metres from the shoulder axis; see test_reach_envelope


@dataclass(frozen=True, slots=True)
class IKResult:
    q: np.ndarray          # 5 arm joint angles
    ok: bool               # converged within tolerance and inside limits
    error: np.ndarray      # final task residual
    iterations: int
    pos_error: float       # metres
    ori_error: float       # radians


HOME: np.ndarray = np.array([0.0, -0.6, 0.9, 0.5, 0.0])
"""A safe, well-conditioned configuration away from limits and the table.
Used as the cold-start IK seed and as the arm's park pose."""


def _solve_from(
    q0: np.ndarray,
    goal: np.ndarray,
    W: np.ndarray,
    pos_tol: float,
    ori_tol: float,
    max_iter: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One Levenberg-Marquardt run. Returns (q, task residual, iterations).

    Adaptive damping is what makes this reliable. A fixed damping term has
    to be tuned as a compromise: small enough to converge quickly in the
    well-conditioned interior, large enough not to blow up near the
    wrist-vertical singularity or against a joint limit. No single value
    does both, and the failures show up exactly where a game needs the arm
    most -- at full extension, reaching for a hand. LM instead raises
    damping when a step makes things worse and lowers it when the step
    helps, so it behaves like gradient descent in the hard regions and like
    Gauss-Newton in the easy ones.
    """
    q = q0.copy()
    err = task_error(_task_vec(q), goal)
    cost = float(np.sum((W @ err) ** 2))
    eye = np.eye(5)
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 - `it` is the returned iteration count
        if np.linalg.norm(err[:3]) < pos_tol and np.max(np.abs(err[3:])) < ori_tol:
            break
        J = W @ jacobian(q, base=goal - err)
        e = W @ err
        JJt = J @ J.T
        try:
            dq = J.T @ np.linalg.solve(JJt + (lam**2) * eye, e)
        except np.linalg.LinAlgError:
            lam = min(lam * 4.0, 10.0)
            continue

        # Cap the step so a large residual cannot fling the solver across
        # the workspace into a different IK branch.
        step = float(np.linalg.norm(dq))
        if step > 0.35:
            dq *= 0.35 / step

        q_try = clamp_to_limits(q + dq)
        err_try = task_error(_task_vec(q_try), goal)
        cost_try = float(np.sum((W @ err_try) ** 2))
        if cost_try < cost:
            q, err, cost = q_try, err_try, cost_try
            lam = max(lam * 0.5, 1e-4)
        else:
            lam = min(lam * 3.0, 10.0)
            if lam >= 10.0:
                break  # wedged: let the caller restart from elsewhere
    return q, err, it


def ik(
    target: Pose,
    seed: np.ndarray | None = None,
    *,
    pos_tol: float = 1e-3,          # 1 mm
    ori_tol: float = 1e-2,          # ~0.6 deg
    max_iter: int = 60,
    damping: float = 1e-2,
    weights: np.ndarray | None = None,
    restarts: int = 3,
) -> IKResult:
    """Solve for the 5 arm joints reaching `target`.

    `weights` scales the task rows in the least-squares objective. The
    default trades orientation for position: when a target is out of reach
    or near a singularity we would much rather land in the right place with
    a slightly wrong tool angle than the reverse. For a slap that is exactly
    the correct trade.

    On failure the solver restarts from perturbed seeds. A 5R arm has
    multiple IK branches (elbow up/down and wrist flips), and a local
    method only ever finds the branch its seed sits in; a restart is how it
    reaches the others.
    """
    if weights is None:
        weights = np.array([1.0, 1.0, 1.0, 0.3, 0.15])
    W = np.diag(weights)
    goal = target.as_vec()

    q0 = HOME.copy() if seed is None else clamp_to_limits(np.asarray(seed, float)[:5].copy())

    rng = np.random.default_rng(0)  # deterministic: identical inputs give identical motion
    total_iters = 0
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for attempt in range(restarts + 1):
        q, err, it = _solve_from(q0, goal, W, pos_tol, ori_tol, max_iter, damping)
        total_iters += it
        pos_err = float(np.linalg.norm(err[:3]))
        ori_err = float(np.max(np.abs(err[3:])))
        score = float(np.sum((W @ err) ** 2))
        if best is None or score < best[0]:
            best = (score, q, err)
        if pos_err < pos_tol and ori_err < ori_tol and within_limits(q):
            return IKResult(q, True, err, total_iters, pos_err, ori_err)
        # Perturb for the next branch. Widen with each failed attempt.
        scale = 0.5 * (attempt + 1)
        q0 = clamp_to_limits(HOME + rng.normal(0.0, scale, 5))

    assert best is not None
    _, q, err = best
    pos_err = float(np.linalg.norm(err[:3]))
    ori_err = float(np.max(np.abs(err[3:])))
    return IKResult(q, False, err, total_iters, pos_err, ori_err)


def _solve_position(
    goal_xyz: np.ndarray,
    q0: np.ndarray,
    pos_tol: float,
    max_iter: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Levenberg-Marquardt on position alone, analytic Jacobian.

    The hot path. Everything the games do -- hovering, striking,
    pointing, reaching for an object -- only cares where the tip is.
    """
    q = q0.copy()
    err = goal_xyz - fk(q)[:3, 3]
    cost = float(err @ err)
    eye = np.eye(3)
    it = 0
    for it in range(1, max_iter + 1):
        if np.linalg.norm(err) < pos_tol:
            break
        J = position_jacobian(q)
        try:
            dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * eye, err)
        except np.linalg.LinAlgError:
            lam = min(lam * 4.0, 10.0)
            continue
        step = float(np.linalg.norm(dq))
        if step > 0.35:
            dq *= 0.35 / step
        q_try = clamp_to_limits(q + dq)
        err_try = goal_xyz - fk(q_try)[:3, 3]
        cost_try = float(err_try @ err_try)
        if cost_try < cost:
            q, err, cost = q_try, err_try, cost_try
            lam = max(lam * 0.5, 1e-4)
        else:
            lam = min(lam * 3.0, 10.0)
            if lam >= 10.0:
                break
    return q, err, it


def ik_position(
    xyz, seed: np.ndarray | None = None, *, pitch: float | None = None, **kw
) -> IKResult:
    """Position-only IK: reach a point, let the solver pick the tool angle.

    Used for anything where only the tip location matters -- pointing,
    slapping, hovering over an object -- which is most of what this robot
    does. Orientation weights drop to near zero so the extra freedom is
    spent on hitting the point.
    """
    xyz = np.asarray(xyz, dtype=float)

    if pitch is None:
        # Fast path: no orientation constraint at all, so the analytic
        # position Jacobian is sufficient and the whole finite-difference
        # machinery can be skipped.
        pos_tol = kw.get("pos_tol", 1e-3)
        max_iter = kw.get("max_iter", 60)
        damping = kw.get("damping", 1e-2)
        restarts = kw.get("restarts", 3)
        q0 = HOME.copy() if seed is None else clamp_to_limits(np.asarray(seed, float)[:5].copy())
        rng = np.random.default_rng(0)
        best = None
        total = 0
        for attempt in range(restarts + 1):
            q, err, it = _solve_position(xyz, q0, pos_tol, max_iter, damping)
            total += it
            pos_err = float(np.linalg.norm(err))
            if best is None or pos_err < best[0]:
                best = (pos_err, q, err)
            if pos_err < pos_tol and within_limits(q):
                full = np.zeros(5)
                return IKResult(q, True, np.concatenate([err, [0.0, 0.0]]), total, pos_err, 0.0)
            q0 = clamp_to_limits(HOME + rng.normal(0.0, 0.5 * (attempt + 1), 5))
        pos_err, q, err = best
        return IKResult(q, False, np.concatenate([err, [0.0, 0.0]]), total, pos_err, 0.0)

    p = Pose(float(xyz[0]), float(xyz[1]), float(xyz[2]), pitch, 0.0)
    w = np.array([1.0, 1.0, 1.0, 0.4, 0.0])
    kw.setdefault("ori_tol", 5e-2)
    return ik(p, seed, weights=w, **kw)

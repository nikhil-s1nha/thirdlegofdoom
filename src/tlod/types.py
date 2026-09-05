"""Shared value types.

Everything that crosses a module boundary is one of these. They are all
immutable and cheap to copy, so they can be handed between threads without
locking (see tlod.runtime.signal).

Timestamps are always `time.perf_counter()` seconds from a single process.
Every observation carries the timestamp of the *physical event* it describes
(shutter open, encoder read) rather than the time it finished being computed.
That is the only way to measure end-to-end latency honestly, and for a game
where 50 ms decides the outcome it is the difference between a robot that
works and one that is mysteriously always slightly late.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Sequence

import numpy as np

# Canonical joint order. Matches the SO-101 URDF chain and the motor ids
# 1..6 on the Feetech bus, so this ordering is used for every array of
# joint values in the codebase. Do not reorder.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS: tuple[str, ...] = JOINT_NAMES[:5]  # the 5 pose-controlling joints
GRIPPER: str = "gripper"
NUM_JOINTS: int = len(JOINT_NAMES)


def joint_index(name: str) -> int:
    return JOINT_NAMES.index(name)


@dataclass(frozen=True, slots=True)
class JointState:
    """Measured or commanded joint configuration, radians, JOINT_NAMES order."""

    q: np.ndarray                 # shape (6,), radians
    stamp: float                  # perf_counter of the encoder read
    dq: np.ndarray | None = None  # shape (6,), rad/s, if the backend reports it
    load: np.ndarray | None = None  # shape (6,), -1..1 of rated torque

    def __post_init__(self) -> None:
        if self.q.shape != (NUM_JOINTS,):
            raise ValueError(f"q must be shape ({NUM_JOINTS},), got {self.q.shape}")

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(JOINT_NAMES, self.q, strict=True)}

    @staticmethod
    def from_dict(d: dict[str, float], stamp: float) -> JointState:
        return JointState(np.array([d[n] for n in JOINT_NAMES], dtype=float), stamp)


@dataclass(frozen=True, slots=True)
class Pose:
    """Tool pose in the robot base frame.

    The SO-101 has five arm joints, so it cannot reach an arbitrary 6-DOF
    pose. The reachable task space is exactly:

        (x, y, z)  tool tip position
        pitch      elevation of the tool axis, in the vertical plane
        roll       rotation about the tool axis

    Tool *yaw* is not free: it is fixed by atan2(y, x), the base pan needed
    to reach the target. Asking for a yaw is a modelling error, so there is
    deliberately no field for it.
    """

    x: float
    y: float
    z: float
    pitch: float = 0.0
    roll: float = 0.0

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def as_vec(self) -> np.ndarray:
        """5-vector in the arm's true task space, the IK residual space."""
        return np.array([self.x, self.y, self.z, self.pitch, self.roll], dtype=float)

    @staticmethod
    def from_vec(v: Sequence[float]) -> Pose:
        return Pose(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]))

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Pose:
        return replace(self, x=self.x + dx, y=self.y + dy, z=self.z + dz)


@dataclass(frozen=True, slots=True)
class Detection:
    """A thing seen on the table, in the robot base frame (metres)."""

    label: str
    position: np.ndarray          # (3,) base frame
    stamp: float                  # shutter time
    confidence: float = 1.0
    pixel: tuple[float, float] | None = None   # (u, v) centroid, for debug overlay
    radius: float = 0.0           # rough object radius, metres


@dataclass(frozen=True, slots=True)
class HandObservation:
    """A human hand, in the robot base frame.

    `velocity` and `stamp` exist so the game layer can *extrapolate*. With a
    250-450 ms sense-to-motion budget, a purely reactive slap always loses;
    the controller must aim where the hand will be, not where it was.
    """

    position: np.ndarray                 # (3,) palm centre, base frame
    stamp: float                         # shutter time
    velocity: np.ndarray | None = None   # (3,) m/s, from the tracker
    landmarks: np.ndarray | None = None  # (21, 3) base frame, if available
    handedness: str = "unknown"          # "Left" | "Right" | "unknown"
    confidence: float = 1.0

    def predict(self, dt: float) -> np.ndarray:
        """Constant-velocity extrapolation `dt` seconds past the shutter."""
        if self.velocity is None:
            return self.position
        return self.position + self.velocity * dt


@dataclass(frozen=True, slots=True)
class Frame:
    """One camera frame plus the time its shutter opened."""

    image: np.ndarray     # HxWx3 BGR
    stamp: float          # best estimate of shutter time
    index: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]  # (w, h)


@dataclass(slots=True)
class Perception:
    """Everything the vision thread knows, published as one atomic snapshot."""

    stamp: float
    hands: list[HandObservation] = field(default_factory=list)
    objects: list[Detection] = field(default_factory=list)
    frame: Frame | None = None
    vision_latency: float = 0.0  # shutter -> published, seconds

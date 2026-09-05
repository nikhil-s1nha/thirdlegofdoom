"""Synthetic scene: a hand that moves through the robot's actual workspace.

The first version of the simulator defined the fake hand's motion in
*pixels*, which was backwards. Pixels have no relationship to what the arm
can reach, so most of the trajectory fell outside the workspace and the
run was mostly safety-guard clamping -- a demo that exercised the guards
rather than the behaviour.

Here the trajectory is defined in the robot base frame, inside the reach
envelope, and *then* projected into the image. That is also the right
direction physically: the world exists, and the camera observes it.

Because the 3D truth is known, this doubles as ground truth. A tracker's
prediction can be scored against `position_at(t + horizon)` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tlod.vision.hands import INDEX_MCP, PALM_WIDTH_M, PINKY_MCP, Hand2D


@dataclass(slots=True)
class HandPath:
    """A hand orbiting through the reachable workspace.

    Default centre and radii sit inside the SO-101's envelope with margin,
    so a tracking policy spends its time tracking rather than being
    clamped.
    """

    center: tuple[float, float, float] = (0.22, 0.0, 0.10)
    radius_x: float = 0.055
    radius_y: float = 0.090
    radius_z: float = 0.035
    speed: float = 1.6              # rad/s around the orbit
    dodge_at: float | None = None   # seconds; if set, snap away once
    dodge_speed: float = 1.4        # m/s during the snap

    def position_at(self, t: float) -> np.ndarray:
        cx, cy, cz = self.center
        p = np.array(
            [
                cx + self.radius_x * np.sin(t * self.speed),
                cy + self.radius_y * np.sin(t * self.speed * 0.63),
                cz + self.radius_z * np.cos(t * self.speed * 0.81),
            ]
        )
        if self.dodge_at is not None and t > self.dodge_at:
            # A hard reversal: what a real dodge looks like, and the case
            # a constant-velocity predictor handles worst.
            dt = t - self.dodge_at
            p = p + np.array([-0.6, 0.7, 0.35]) * self.dodge_speed * dt
        return p

    def velocity_at(self, t: float, eps: float = 1e-4) -> np.ndarray:
        return (self.position_at(t + eps) - self.position_at(t - eps)) / (2 * eps)


class SyntheticHandScene:
    """Renders a HandPath through a Projector, and reports ground truth."""

    def __init__(
        self,
        projector,
        path: HandPath | None = None,
        palm_width_m: float = PALM_WIDTH_M,
    ) -> None:
        self.projector = projector
        self.path = path or HandPath()
        self.palm_width_m = palm_width_m

    # -- ground truth ------------------------------------------------------
    def position_at(self, t: float) -> np.ndarray:
        return self.path.position_at(t)

    def velocity_at(self, t: float) -> np.ndarray:
        return self.path.velocity_at(t)

    # -- observation -------------------------------------------------------
    def pixel_at(self, t: float) -> tuple[float, float] | None:
        return self.projector.project(self.position_at(t))

    def palm_width_px_at(self, t: float) -> float:
        """Apparent palm width, so size-based depth recovers the truth."""
        p = self.position_at(t)
        d = p - self.projector.extr.t
        range_m = float(np.linalg.norm(d))
        if range_m < 1e-6:
            return 1.0
        forward = self.projector.extr.R[:, 2]
        depth = float(np.dot(d, forward))
        fx = float(self.projector.intr.K[0, 0])
        return max(1.0, fx * self.palm_width_m / max(depth, 1e-6))

    def hand2d_at(self, t: float, stamp: float) -> Hand2D | None:
        uv = self.pixel_at(t)
        if uv is None:
            return None
        half = self.palm_width_px_at(t) / 2.0
        lms = np.tile(np.array(uv, dtype=float), (21, 1))
        lms[INDEX_MCP] = [uv[0] - half, uv[1]]
        lms[PINKY_MCP] = [uv[0] + half, uv[1]]
        return Hand2D(lms, 1.0, "Right", stamp)

    def render(self, t: float, width: int, height: int) -> np.ndarray:
        """A frame showing the hand and a few table markers for context."""
        img = np.full((height, width, 3), 32, dtype=np.uint8)

        # Workspace annulus on the table, so the view is legible.
        for r in (0.10, 0.20, 0.30):
            pts = []
            for a in np.linspace(0, 2 * np.pi, 72):
                uv = self.projector.project(np.array([r * np.cos(a), r * np.sin(a), 0.0]))
                if uv is not None:
                    pts.append([int(uv[0]), int(uv[1])])
            if len(pts) > 2:
                cv2.polylines(img, [np.array(pts, np.int32)], True, (60, 60, 60), 1)

        uv = self.pixel_at(t)
        if uv is not None:
            radius = max(4, int(self.palm_width_px_at(t) / 2))
            cv2.circle(img, (int(uv[0]), int(uv[1])), radius, (70, 110, 230), -1)
            cv2.circle(img, (int(uv[0]), int(uv[1])), radius, (200, 220, 255), 2)
        return img


class SceneHandDetector:
    """Reads hands straight out of a scene. No model, fully deterministic.

    Bypasses the image entirely rather than rendering and re-detecting:
    tier A exists to test control logic and timing, and putting a neural
    network in that path only adds noise and nondeterminism to a test
    whose value is being exactly repeatable.
    """

    def __init__(self, scene: SyntheticHandScene, t0: float | None = None) -> None:
        self.scene = scene
        self.t0 = t0

    def detect(self, frame) -> list[Hand2D]:
        if self.t0 is None:
            self.t0 = frame.stamp
        h = self.scene.hand2d_at(frame.stamp - self.t0, frame.stamp)
        return [h] if h is not None else []

    def close(self) -> None:
        pass

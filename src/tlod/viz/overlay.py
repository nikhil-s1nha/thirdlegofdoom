"""Draw what the robot knows onto the camera image.

Everything is projected through the same `Projector` the controller uses,
which makes this a genuine check rather than a decoration: if the arm
skeleton does not land on the arm in the picture, the extrinsics are
wrong, and that is a bug you cannot find by reading numbers.

Colours are consistent throughout:
    white    measured / ground truth
    cyan     the arm's current configuration
    amber    where the arm is being commanded
    green    a confirmed hand track
    magenta  the predicted future hand position
    red      safety limits, and anything refused
"""

from __future__ import annotations

import cv2
import numpy as np

from tlod.arm import model
from tlod.types import Pose

WHITE = (235, 235, 235)
CYAN = (220, 200, 90)
AMBER = (70, 170, 240)
GREEN = (120, 220, 130)
MAGENTA = (220, 120, 220)
RED = (80, 80, 235)
GREY = (110, 110, 110)
DARK = (55, 55, 55)


class Overlay:
    def __init__(self, projector, limits=None) -> None:
        self.projector = projector
        self.limits = limits

    # -- helpers -----------------------------------------------------------
    def _px(self, point) -> tuple[int, int] | None:
        uv = self.projector.project(np.asarray(point, float))
        if uv is None:
            return None
        u, v = uv
        if not (np.isfinite(u) and np.isfinite(v)) or abs(u) > 1e5 or abs(v) > 1e5:
            return None
        return int(u), int(v)

    def _line(self, img, a, b, color, thickness=2) -> None:
        pa, pb = self._px(a), self._px(b)
        if pa and pb:
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)

    def _circle_on_table(self, img, radius, z, color, thickness=1) -> None:
        pts = []
        for a in np.linspace(0, 2 * np.pi, 96):
            p = self._px([radius * np.cos(a), radius * np.sin(a), z])
            if p:
                pts.append(p)
        if len(pts) > 2:
            cv2.polylines(img, [np.array(pts, np.int32)], True, color, thickness, cv2.LINE_AA)

    # -- layers ------------------------------------------------------------
    def draw_workspace(self, img) -> None:
        """The table, the reach envelope, and the keep-out around the base."""
        if self.limits is None:
            return
        self._circle_on_table(img, self.limits.max_radius, self.limits.table_z, RED, 1)
        self._circle_on_table(img, self.limits.min_radius, self.limits.table_z, RED, 1)
        for r in (0.15, 0.25):
            self._circle_on_table(img, r, self.limits.table_z, DARK, 1)
        # Base axes, so the frame convention is visible at a glance.
        for axis, color in zip(np.eye(3) * 0.06, ((90, 90, 230), (90, 230, 90), (230, 160, 90))):
            self._line(img, [0, 0, 0], axis, color, 2)

    def draw_arm(self, img, q, color=CYAN, thickness=3) -> None:
        """Project the kinematic chain. The strongest available check that
        the camera calibration is right."""
        frames = model.fk_all(np.asarray(q, float)[:5])
        points = [np.zeros(3)] + [f[:3, 3] for f in frames]
        for a, b in zip(points, points[1:]):
            self._line(img, a, b, color, thickness)
        for p in points:
            px = self._px(p)
            if px:
                cv2.circle(img, px, 4, color, -1, cv2.LINE_AA)
        tip = self._px(points[-1])
        if tip:
            cv2.circle(img, tip, 9, color, 2, cv2.LINE_AA)

    def draw_hand(self, img, position, color=GREEN, label: str = "", radius: int = 12) -> None:
        p = self._px(position)
        if not p:
            return
        cv2.circle(img, p, radius, color, 2, cv2.LINE_AA)
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
        # A dropline to the table makes height legible in a 2D projection.
        foot = self._px([position[0], position[1], 0.0])
        if foot:
            cv2.line(img, p, foot, color, 1, cv2.LINE_AA)
            cv2.circle(img, foot, 3, color, 1, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (p[0] + 16, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def draw_prediction(self, img, now, future) -> None:
        a, b = self._px(now), self._px(future)
        if a and b:
            cv2.arrowedLine(img, a, b, MAGENTA, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(img, b, 7, MAGENTA, 1, cv2.LINE_AA)

    def draw_target(self, img, pose: Pose) -> None:
        p = self._px([pose.x, pose.y, pose.z])
        if not p:
            return
        s = 9
        cv2.line(img, (p[0] - s, p[1]), (p[0] + s, p[1]), AMBER, 2, cv2.LINE_AA)
        cv2.line(img, (p[0], p[1] - s), (p[0], p[1] + s), AMBER, 2, cv2.LINE_AA)

    def draw_strike_zone(self, img, center, radius: float, color=RED) -> None:
        pts = []
        for a in np.linspace(0, 2 * np.pi, 64):
            p = self._px([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), center[2]])
            if p:
                pts.append(p)
        if len(pts) > 2:
            cv2.polylines(img, [np.array(pts, np.int32)], True, color, 1, cv2.LINE_AA)

    def draw_hud(self, img, lines: list[str], anchor=(12, 24), color=WHITE) -> None:
        x, y = anchor
        width = max((len(s) for s in lines), default=0)
        if lines:
            cv2.rectangle(img, (x - 8, y - 18), (x + 8 * width + 12, y + 18 * len(lines) - 6),
                          (18, 18, 18), -1)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)

    def draw_banner(self, img, text: str, color=WHITE) -> None:
        h, w = img.shape[:2]
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        org = ((w - size[0]) // 2, h - 40)
        cv2.rectangle(img, (org[0] - 16, org[1] - size[1] - 14),
                      (org[0] + size[0] + 16, org[1] + 14), (18, 18, 18), -1)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

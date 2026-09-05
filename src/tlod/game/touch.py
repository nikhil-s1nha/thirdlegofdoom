"""Visit every object on the table, in turn.

The simplest complete demonstration of the other half of the system:
detect an object, put it in the robot's frame, plan to it, move, verify.
No game, no opponent -- just the perception-to-control path exercised end
to end on something that is not a hand.

It is also the natural place to notice calibration error. If the tool
consistently lands a centimetre to one side of every object, the
extrinsics are off, and this policy makes that obvious in a way a latency
table never will.
"""

from __future__ import annotations

import time

import numpy as np

from tlod.arm.primitives import GoToPose, Hold, Sequence
from tlod.game.base import StateMachine
from tlod.types import Pose


class TouchObjectsPolicy(StateMachine):
    name = "touch_objects"
    initial_state = "scan"

    def __init__(
        self,
        hover_height: float = 0.07,
        touch_height: float = 0.025,
        dwell: float = 0.4,
        min_confidence: float = 0.25,
    ) -> None:
        super().__init__()
        self.hover_height = hover_height
        self.touch_height = touch_height
        self.dwell = dwell
        self.min_confidence = min_confidence
        self.queue: list = []
        self.visited: list[str] = []
        self.errors: list[float] = []
        self.current = None

    def _state_scan(self, robot, controller, dt) -> None:
        self.step_motion(controller, dt)
        snapshot = robot.perception.get()
        if snapshot is None or not snapshot.objects:
            return
        seen = [d for d in snapshot.objects
                if d.confidence >= self.min_confidence and d.label not in self.visited]
        if not seen:
            return
        # Nearest first: shortest total path, and it fails fast if the
        # closest object is somehow unreachable.
        tool = controller.pose().xyz()
        self.queue = sorted(seen, key=lambda d: float(np.linalg.norm(d.position - tool)))
        self.transition("approach")

    def _state_approach(self, robot, controller, dt) -> None:
        if self.current is None:
            if not self.queue:
                # Re-scan rather than declaring victory. An object can be
                # missed on any single pass -- occluded by the arm, or
                # suppressed because the hand was over it at that instant
                # -- and a one-shot scan turns a transient occlusion into
                # a permanently skipped object.
                self.transition("scan")
                return
            self.current = self.queue.pop(0)
            p = self.current.position
            self.run_motion(
                Sequence([
                    GoToPose(Pose(float(p[0]), float(p[1]), float(p[2]) + self.hover_height), 0.7),
                    GoToPose(Pose(float(p[0]), float(p[1]), float(p[2]) + self.touch_height), 0.35),
                    Hold(self.dwell),
                    GoToPose(Pose(float(p[0]), float(p[1]), float(p[2]) + self.hover_height), 0.35),
                ]),
                controller,
            )
            return

        if self.step_motion(controller, dt):
            tool = controller.pose().xyz()
            error = float(np.linalg.norm(tool[:2] - self.current.position[:2]))
            self.errors.append(error)
            self.visited.append(self.current.label)
            self.announce(f"touched {self.current.label}, {error*1000:.1f} mm off centre")
            self.current = None

    def _state_done(self, robot, controller, dt) -> None:
        self.step_motion(controller, dt)

    @property
    def complete(self) -> bool:
        return self.state == "done"

    def update(self, robot, perception, dt) -> None:
        if robot.controller.estopped:
            return
        getattr(self, f"_state_{self.state}")(robot, robot.controller, dt)

    def hud(self) -> list[str]:
        lines = [f"touching  [{self.state}]  {len(self.visited)} done, {len(self.queue)} queued"]
        if self.errors:
            lines.append(f"mean err  {np.mean(self.errors)*1000:.1f} mm")
        if self.current is not None:
            lines.append(f"target    {self.current.label}")
        return lines

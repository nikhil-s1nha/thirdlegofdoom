"""The window.

Runs on the **main thread**, always. On macOS the GUI event loop is not
optional and a cv2 window created from a worker thread either does
nothing or crashes the process. RobotApp keeps perception and control on
their own threads precisely so the main thread is free for this.

The viewer only reads state. It never commands the arm, so dropping
frames or closing the window cannot affect the robot's behaviour -- which
also means what you see is what happened, not a rendering of what the
renderer wished had happened.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from tlod.arm import model
from tlod.viz.overlay import AMBER, CYAN, GREEN, MAGENTA, Overlay, WHITE


class Viewer:
    def __init__(
        self,
        app,
        projector,
        title: str = "third leg of doom",
        scale: float = 1.0,
        show_camera: bool = True,
    ) -> None:
        self.app = app
        self.overlay = Overlay(projector, app.controller.limits)
        self.title = title
        self.scale = scale
        self.show_camera = show_camera
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self._fps = 0.0

    def _background(self) -> np.ndarray:
        snapshot = self.app.perception.get()
        if self.show_camera and snapshot is not None and snapshot.frame is not None:
            img = snapshot.frame.image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        w, h = self.app.camera.resolution
        return np.full((h, w, 3), 26, dtype=np.uint8)

    def render_once(self) -> np.ndarray:
        img = self._background()
        snapshot = self.app.perception.get()

        self.overlay.draw_workspace(img)

        state = self.app.controller.state()
        self.overlay.draw_arm(img, state.q, CYAN, 3)
        self.overlay.draw_arm(img, self.app.controller.commanded, AMBER, 1)

        track = self.app.tracker.best()
        if track is not None:
            pos = track.filter.position
            self.overlay.draw_hand(img, pos, GREEN, f"hand#{track.id} {track.filter.speed:.2f} m/s")
            horizon = self.app.prediction_horizon
            if track.filter.speed > 0.05:
                self.overlay.draw_prediction(img, pos, track.filter.predict(horizon))

        if snapshot is not None:
            for det in snapshot.objects:
                self.overlay.draw_hand(img, det.position, (200, 200, 120), det.label, radius=8)

        pose = model.tool_pose(state.q[:5])
        hud = [
            f"policy   {self.app.policy.name}",
            f"tool     {pose.x:+.3f} {pose.y:+.3f} {pose.z:+.3f} m",
            f"latency  {self.app.t_end_to_end.p50_ms:5.1f} ms  (shutter->command)",
            f"vision   {self.app.t_vision_total.p50_ms:5.1f} ms   detect {self.app.t_detect.p50_ms:.1f} ms",
            f"control  {self.app.control_loop.ticks} ticks  {self.app.control_loop.overrun_rate*100:.1f}% over",
            f"ik       {self.app.controller.stats.ik_failures} fail  "
            f"{self.app.controller.stats.guard_hits} guard",
            f"view     {self._fps:.0f} fps",
        ]
        if self.app.controller.estopped:
            hud.append("*** E-STOP ***")
        extra = getattr(self.app.policy, "hud", None)
        if callable(extra):
            hud += list(extra())
        self.overlay.draw_hud(img, hud)

        banner = getattr(self.app.policy, "banner", None)
        if callable(banner):
            text = banner()
            if text:
                self.overlay.draw_banner(img, text)

        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        return img

    def run(self, duration: float | None = None, fps: float = 30.0) -> None:
        """Block on the main thread until the window closes or time runs out.

        Keys: q or Esc to quit, e to toggle e-stop, space for a policy
        action (games use it to start a round).
        """
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        period = 1.0 / fps
        deadline = None if duration is None else time.perf_counter() + duration
        try:
            while True:
                start = time.perf_counter()
                cv2.imshow(self.title, self.render_once())

                self._fps_n += 1
                if start - self._fps_t >= 0.5:
                    self._fps = self._fps_n / (start - self._fps_t)
                    self._fps_t, self._fps_n = start, 0

                key = cv2.waitKey(max(1, int((period - (time.perf_counter() - start)) * 1000)))
                key &= 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("e"):
                    if self.app.controller.estopped:
                        self.app.controller.release_estop()
                    else:
                        self.app.controller.estop()
                if key == ord(" "):
                    action = getattr(self.app.policy, "on_key_space", None)
                    if callable(action):
                        action()
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if deadline is not None and time.perf_counter() > deadline:
                    break
        finally:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)

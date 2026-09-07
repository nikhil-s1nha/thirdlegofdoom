"""Render arm telemetry received over the network.

For running the arm-less HIL test (`docs/deployment.md`'s two-board
split, `arm.backend=mock`, no physical SO-101) from a laptop instead of
the control board itself -- the Pi has no screen either way, real or
mock arm, so this is the same "watch it from elsewhere" idea as
`tlod.vision.preview`, but for `ArmTelemetryPacket`s instead of pixels.

Deliberately not a subclass or reuse of `tlod.viz.viewer.Viewer`: that
class is wired to a live, in-process `RobotApp` (its perception mailbox,
tracker, policy, latency stats). Here there is no `RobotApp` at all --
only a `Latest[TelemetryPacket]` mailbox filled by the network -- so
forcing it through that shape would be more code than it saves. What
*is* reused is `Overlay`: the same projection and the same `draw_arm`
call the in-process viewer makes, just fed numbers that arrived over
UDP instead of read from `app.controller`.

There is no camera image to draw over, so the background is the same
plain fill `Viewer` falls back to when it has no frame -- this window
never has one.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from tlod.arm import model
from tlod.arm.controller import SafetyLimits
from tlod.net.telemetry import ArmTelemetrySubscriber
from tlod.viz.overlay import AMBER, CYAN, GREEN, Overlay


class RemoteArmViewer:
    def __init__(
        self,
        subscriber: ArmTelemetrySubscriber,
        projector=None,
        limits: SafetyLimits | None = None,
        resolution: tuple[int, int] = (1280, 720),
        title: str = "third leg of doom (remote)",
        stale_after: float = 1.0,
    ) -> None:
        if projector is None:
            from tlod.vision.calibration import synthetic_projector

            projector = synthetic_projector(resolution)
        self.subscriber = subscriber
        self.overlay = Overlay(projector, limits or SafetyLimits())
        self.resolution = resolution
        self.title = title
        self.stale_after = stale_after
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self._fps = 0.0

    def render_once(self) -> np.ndarray:
        w, h = self.resolution
        img = np.full((h, w, 3), 26, dtype=np.uint8)
        self.overlay.draw_workspace(img)

        packet = self.subscriber.latest.get()
        age = self.subscriber.latest.age
        stale = packet is not None and age > self.stale_after

        if packet is not None:
            self.overlay.draw_arm(img, packet.q, CYAN, 3)
            self.overlay.draw_arm(img, packet.commanded, AMBER, 1)
            if packet.hand is not None:
                self.overlay.draw_hand(img, packet.hand, GREEN, "hand")
        else:
            # Nothing has arrived yet -- draw the home configuration so the
            # window shows *something* geometric rather than an empty grid.
            self.overlay.draw_arm(img, model.HOME, (90, 90, 90), 1)

        hud = [
            f"telemetry :{self.subscriber.port}",
            f"packets   {self.subscriber.received}  "
            f"bad {self.subscriber.dropped_bad}  stale-dropped {self.subscriber.dropped_stale}",
            f"age       {'no data' if packet is None else f'{age * 1e3:5.1f} ms'}",
            f"view      {self._fps:.0f} fps",
        ]
        if packet is not None and packet.estopped:
            hud.append("*** E-STOP ***")
        self.overlay.draw_hud(img, hud)

        if packet is None:
            self.overlay.draw_banner(img, "waiting for telemetry...")
        elif stale:
            self.overlay.draw_banner(img, "STALE -- control board may have stopped")

        return img

    def run(self, duration: float | None = None, fps: float = 30.0) -> None:
        """Block until the window closes or time runs out. q/Esc to quit."""
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
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if deadline is not None and time.perf_counter() > deadline:
                    break
        finally:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)

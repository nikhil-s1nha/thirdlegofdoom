"""Wiring: a perception thread, a control thread, and a policy between them.

    camera -> detect -> locate -> track ---[Latest]--- policy -> IK -> arm
             (perception thread,                      (control thread,
              camera rate, bursty)                     fixed rate, strict)

Two threads rather than one loop, because they want opposite things.
Perception is bursty: a detection sweep costs several times a tracking
update, and the camera delivers when it delivers. Control wants to tick
like a metronome. Running them together means every hiccup in vision
becomes a hiccup in motion.

They are joined by a single-slot mailbox, not a queue, so control always
reads the newest estimate and can never fall behind into a backlog.

Latency accounting is built in rather than bolted on. Every stage is
timed, and the number that actually matters -- shutter to servo command --
is measured directly by carrying the frame's shutter timestamp all the way
through to the moment a command goes out. That figure is what the
prediction horizon is set from, so it has to be real.
"""

from __future__ import annotations

import abc
import logging
import threading
import time

import numpy as np

from tlod.arm.controller import ArmController
from tlod.runtime.loop import RateLoop, Timing
from tlod.runtime.signal import Latest
from tlod.types import Frame, Perception, Pose
from tlod.vision.camera import Camera
from tlod.vision.hands import HandDetector, HandLocator
from tlod.vision.tracking import MultiTracker

log = logging.getLogger(__name__)


class Policy(abc.ABC):
    """A behaviour. Games are policies.

    `update` is called on the control thread at a fixed rate with the
    freshest perception available -- or None when perception is stale,
    which is a case every policy must handle deliberately rather than by
    continuing to act on old data.
    """

    name: str = "policy"

    def start(self, robot: RobotApp) -> None:
        pass

    @abc.abstractmethod
    def update(self, robot: RobotApp, perception: Perception | None, dt: float) -> None: ...

    def stop(self, robot: RobotApp) -> None:
        pass


class IdlePolicy(Policy):
    """Holds position. The safe default."""

    name = "idle"

    def update(self, robot, perception, dt) -> None:
        return


class TrackHandPolicy(Policy):
    """Visual servoing: hover above the tracked hand, aiming ahead of it.

    The demonstration that the whole pipeline closes. It also makes the
    latency argument visible: set `use_prediction=False` and the arm
    visibly trails the hand; set it True and it keeps up.
    """

    name = "track_hand"

    def __init__(
        self,
        hover_height: float = 0.12,
        use_prediction: bool = True,
        max_speed: float | None = None,
    ) -> None:
        self.hover_height = hover_height
        self.use_prediction = use_prediction
        self.max_speed = max_speed

    def update(self, robot, perception, dt) -> None:
        if perception is None:
            return
        track = robot.tracker.best()
        if track is None:
            return

        if self.use_prediction:
            # Aim where the hand will be once the arm can actually be
            # there: the measured sense-to-motion latency, not a guess.
            target = track.filter.predict(robot.prediction_horizon)
        else:
            target = track.filter.position

        # Refuse to swing at an estimate we do not believe. Uncertainty
        # grows with the horizon, so this fires exactly when the hand has
        # been lost or is moving unpredictably.
        if track.filter.position_uncertainty(robot.prediction_horizon) > 0.15:
            return

        robot.controller.servo_pose(
            Pose(float(target[0]), float(target[1]), float(target[2]) + self.hover_height),
            max_speed=self.max_speed,
            dt=dt,
        )


class RobotApp:
    def __init__(
        self,
        camera: Camera,
        detector: HandDetector,
        locator: HandLocator,
        controller: ArmController,
        policy: Policy | None = None,
        tracker: MultiTracker | None = None,
        object_detector=None,
        hand_suppression_radius: float = 0.07,
        control_hz: float = 100.0,
        perception_max_age: float = 0.25,
        prediction_horizon: float = 0.30,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.locator = locator
        self.controller = controller
        self.policy = policy or IdlePolicy()
        self.tracker = tracker or MultiTracker()
        self.object_detector = object_detector
        # Skin reads as red or orange to a colour segmenter, so a hand in
        # frame reliably produces a phantom object. Suppressing detections
        # that coincide with a tracked hand is the general fix -- it also
        # covers a hand holding a coloured piece, where the piece is real
        # but must not be treated as sitting on the table. Set to 0 to
        # disable.
        self.hand_suppression_radius = hand_suppression_radius
        self.control_hz = control_hz
        self.perception_max_age = perception_max_age
        self.prediction_horizon = prediction_horizon

        self.perception: Latest[Perception] = Latest()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._last_frame_index = -1

        self.t_detect = Timing("vision.detect")
        self.t_locate = Timing("vision.locate+track")
        self.t_vision_total = Timing("vision.shutter->published")
        self.t_control = Timing("control.policy+ik")
        self.t_end_to_end = Timing("shutter->servo command")
        self.control_loop = RateLoop(control_hz, "control")
        self.perception_frames = 0
        self.perception_skipped = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.camera.start()
        self.controller.start()
        self.policy.start(self)
        self._running = True
        for target, name in ((self._perception_loop, "perception"), (self._control_loop, "control")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("robot started: policy=%s control=%.0f Hz", self.policy.name, self.control_hz)

    def _suppress_hand_objects(self, objects, hands):
        """Drop object detections that are really the hand.

        The comparison has to be made on the table plane, not in 3D. An
        object detector resolves every blob by intersecting its ray with
        the table, so a hand hovering 10 cm up is reported at the point
        on the table *behind* it -- which parallax can put well over
        10 cm from where the hand actually is. Comparing that against the
        hand's true 3D position finds no match and suppresses nothing.
        Projecting the hand the same way makes it like for like.
        """
        projector = getattr(self.locator, "projector", None)
        if projector is None:
            return objects

        shadows = []
        for hand in hands:
            uv = projector.project(hand.position)
            if uv is None:
                continue
            on_table = projector.pixel_to_plane(uv[0], uv[1], 0.0)
            if on_table is not None:
                shadows.append(on_table)
        if not shadows:
            return objects
        return [
            d for d in objects
            if min(float(np.linalg.norm(d.position - s)) for s in shadows)
            > self.hand_suppression_radius
        ]

    @property
    def objects(self):
        snapshot = self.perception.get()
        return snapshot.objects if snapshot else []

    def stop(self, park: bool = True) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()
        try:
            self.policy.stop(self)
        finally:
            self.camera.stop()
            self.controller.stop(park=park)
            self.detector.close()

    def __enter__(self) -> RobotApp:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- threads -----------------------------------------------------------
    def _perception_loop(self) -> None:
        while self._running:
            frame: Frame | None = self.camera.read()
            if frame is None or frame.index == self._last_frame_index:
                # No new frame. Yield briefly rather than spinning; the
                # camera thread is the one that should be busy here.
                time.sleep(0.001)
                continue
            self._last_frame_index = frame.index
            self.perception_frames += 1

            try:
                t0 = time.perf_counter()
                hands2d = self.detector.detect(frame)
                self.t_detect.record(t0)

                t1 = time.perf_counter()
                observations = self.locator.locate_all(hands2d)
                positions = [o.position for o in observations]
                self.tracker.update(positions, frame.stamp)

                # Re-issue observations carrying the filter's velocity, so
                # consumers get a motion estimate and not just a point.
                enriched = []
                for obs in observations:
                    tr = min(
                        (t for t in self.tracker.tracks),
                        key=lambda t: float(np.linalg.norm(t.filter.position - obs.position)),
                        default=None,
                    )
                    velocity = tr.filter.velocity if tr is not None else None
                    enriched.append(
                        type(obs)(
                            position=obs.position,
                            stamp=obs.stamp,
                            velocity=velocity,
                            landmarks=obs.landmarks,
                            handedness=obs.handedness,
                            confidence=obs.confidence,
                        )
                    )
                self.t_locate.record(t1)

                objects = []
                if self.object_detector is not None:
                    objects = self.object_detector.detect(frame)
                    if self.hand_suppression_radius > 0 and enriched:
                        objects = self._suppress_hand_objects(objects, enriched)

                now = time.perf_counter()
                self.perception.set(
                    Perception(
                        stamp=frame.stamp,
                        hands=enriched,
                        objects=objects,
                        frame=frame,
                        vision_latency=now - frame.stamp,
                    )
                )
                self.t_vision_total.add(now - frame.stamp)
            except Exception:
                log.exception("perception iteration failed")
                self.perception_skipped += 1

    def _control_loop(self) -> None:
        self.control_loop.reset()
        last = time.perf_counter()
        while self._running:
            self.control_loop.tick()
            now = time.perf_counter()
            dt = now - last
            last = now

            snapshot = self.perception.get_fresh(self.perception_max_age)
            t0 = time.perf_counter()
            try:
                self.policy.update(self, snapshot, dt)
            except Exception:
                log.exception("policy failed; engaging e-stop")
                self.controller.estop()
                return
            self.t_control.record(t0)

            if snapshot is not None:
                self.t_end_to_end.add(time.perf_counter() - snapshot.stamp)

    # -- reporting ---------------------------------------------------------
    def latency_report(self) -> str:
        lines = [
            "",
            "  latency breakdown",
            "  " + "-" * 74,
        ]
        for t in (self.t_detect, self.t_locate, self.t_vision_total, self.t_control, self.t_end_to_end):
            lines.append("  " + t.summary())
        lines.append("  " + "-" * 74)
        lines.append(
            f"  control {self.control_loop.ticks} ticks, "
            f"{self.control_loop.overrun_rate*100:.1f}% overruns, "
            f"jitter p95 {self.control_loop.jitter.p95_ms:.3f} ms"
        )
        lines.append(
            f"  perception {self.perception_frames} frames, {self.perception_skipped} failed"
        )
        lines.append(
            f"  IK: {self.controller.stats.commands} commands, "
            f"{self.controller.stats.ik_failures} failures, "
            f"{self.controller.stats.guard_hits} safety-guard hits"
        )
        return "\n".join(lines)

    @property
    def measured_latency(self) -> float:
        """Shutter to servo command, seconds. What the horizon should be."""
        return self.t_end_to_end.p50_ms / 1e3

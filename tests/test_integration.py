"""End-to-end tier A: synthetic camera, scripted hand, simulated arm.

This is the milestone-1 exit criterion expressed as a test. It runs the
real threads, the real IK and the real safety layer, and asserts on the
properties that make the loop trustworthy rather than merely running:
no IK failures, no safety-guard hits, a control loop that keeps time, and
an end-to-end latency in a sane range.

Everything here is timing-sensitive, so bounds are deliberately loose
enough to survive a loaded CI machine while still catching a regression
of the kind found during bring-up (a mock camera running 30x too fast,
which pushed loop jitter to 70 ms).
"""

import threading
import time

import numpy as np

from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.arm.model import HOME
from tlod.runtime.app import IdlePolicy, RobotApp, TrackHandPolicy
from tlod.vision.calibration import synthetic_projector
from tlod.vision.hands import HandLocator
from tlod.vision.scene import SyntheticHandScene
from tlod.vision.camera import MockCamera
from tlod.vision.tracking import MultiTracker


def build(policy, control_hz=100.0, fps=60, horizon=0.10):
    projector = synthetic_projector()
    scene = SyntheticHandScene(projector)
    return RobotApp(
        camera=MockCamera(640, 480, fps, scene=scene),
        detector=__import__("tlod.vision.scene", fromlist=["x"]).SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        controller=ArmController(
            MockArm(q0=np.concatenate([HOME, [0.0]])), SafetyLimits(), control_hz
        ),
        policy=policy,
        tracker=MultiTracker(),
        control_hz=control_hz,
        prediction_horizon=horizon,
    ), scene


def test_full_loop_runs_cleanly():
    app, _ = build(TrackHandPolicy())
    with app:
        time.sleep(2.5)
        stats = app.controller.stats
        ticks = app.control_loop.ticks
        frames = app.perception_frames
        jitter = app.control_loop.jitter.p95_ms
        latency = app.measured_latency
        e2e_samples = len(app.t_end_to_end.samples)

    assert app.perception_skipped == 0, "perception raised"
    assert stats.ik_failures == 0, f"{stats.ik_failures} IK failures"
    assert stats.guard_hits == 0, f"{stats.guard_hits} safety-guard hits"
    assert stats.commands > 100, "arm was not commanded"
    assert e2e_samples > 0, "no end-to-end latency measured"
    assert 0.001 < latency < 0.20, f"implausible shutter->command latency {latency*1e3:.1f} ms"
    assert jitter < 20.0, f"control jitter p95 {jitter:.1f} ms"
    assert ticks > 150, f"control loop only ticked {ticks} times"
    # The mock camera must respect its configured rate, not free-run.
    assert 60 < frames < 260, f"perception ran at an implausible rate: {frames} frames in 2.5 s"


def test_arm_converges_toward_the_hand():
    app, scene = build(TrackHandPolicy(hover_height=0.10))
    with app:
        time.sleep(3.0)
        tool = app.controller.pose().xyz()
        hand = scene.position_at(app.camera.elapsed)
    # Not a tight bound: the hand keeps moving and the arm has finite
    # speed. The point is that it is following, not parked at HOME.
    assert np.linalg.norm(tool[:2] - hand[:2]) < 0.18, (
        f"tool {np.round(tool,3)} nowhere near hand {np.round(hand,3)}"
    )


def test_idle_policy_does_not_move_the_arm():
    app, _ = build(IdlePolicy())
    with app:
        start = app.controller.commanded.copy()
        time.sleep(1.0)
        assert np.allclose(app.controller.commanded, start)


def test_policy_exception_triggers_estop():
    """A crashing game must stop the arm, not leave it running blind."""

    class Exploding(TrackHandPolicy):
        def update(self, robot, perception, dt):
            raise RuntimeError("boom")

    app, _ = build(Exploding())
    app.start()
    try:
        time.sleep(0.5)
        assert app.controller.estopped, "e-stop was not engaged"
    finally:
        app._running = False
        app.camera.stop()
        app.controller.backend.disconnect()


def test_stale_perception_is_not_acted_on():
    """If vision dies, the policy must receive None rather than old data."""
    seen = []

    class Recording(IdlePolicy):
        def update(self, robot, perception, dt):
            seen.append(perception)

    app, _ = build(Recording())
    app.perception_max_age = 0.05
    with app:
        time.sleep(0.4)
        app.camera.stop()          # vision goes dark
        time.sleep(0.3)
    assert seen[-1] is None, "policy was handed stale perception"


def test_latency_report_is_printable():
    app, _ = build(TrackHandPolicy())
    with app:
        time.sleep(0.6)
        report = app.latency_report()
    for expected in ("vision.detect", "shutter->servo command", "overruns", "IK:"):
        assert expected in report


class TestHybridConfig:
    """`hybrid --real` is the whole robot on one machine: camera, hand
    tracking, IK and servos in one process, arm hovering over the hand.

    The command forced arm.backend to "mock" for its whole life, so the
    failure to guard against is that reverting -- an arm that does not
    move looks the same as one that was never asked to.
    """

    def test_real_reaches_the_arm_backend(self):
        from tlod.cli import hybrid_config
        from tlod.config import Config

        cfg = hybrid_config(Config(), camera=5, policy="track_hand", real=True)
        assert cfg.arm.backend == "feetech"

    def test_without_real_nothing_can_move(self):
        from tlod.cli import hybrid_config
        from tlod.config import Config

        cfg = hybrid_config(Config(), camera=5, policy="track_hand", real=False)
        assert cfg.arm.backend == "mock"

    def test_the_camera_and_hand_are_real_either_way(self):
        """Only the arm is simulated without --real; the point of the
        command is a real camera and a real hand in both modes."""
        from tlod.cli import hybrid_config
        from tlod.config import Config

        for real in (True, False):
            cfg = hybrid_config(Config(), camera=5, policy="track_hand", real=real)
            assert cfg.camera.source == "opencv"
            assert cfg.camera.index == 5
            assert cfg.vision.detector == "mediapipe"
            assert cfg.runtime.policy == "track_hand"

    def test_the_arm_config_is_otherwise_untouched(self):
        """Calibration and limits have to survive the override, or the
        arm follows a hand using factory-default joint zeros."""
        from tlod.cli import hybrid_config
        from tlod.config import Config

        base = Config.from_dict({
            "arm": {"calibration": "calib/mine.json", "port": "/dev/ttyACM0",
                    "torque_limit": 800},
            "safety": {"max_speed": 3.5},
        })
        cfg = hybrid_config(base, camera=0, policy="track_hand", real=True)
        assert cfg.arm.calibration == "calib/mine.json"
        assert cfg.arm.port == "/dev/ttyACM0"
        assert cfg.safety.max_speed == 3.5


class TestPlayConfig:
    """`play --real` is the whole game on one board: camera, hand tracking,
    IK, servos and contact detection in one process, with the arm striking
    at a person's hand.

    `cmd_play` had `--real-hand` for the camera and no flag at all for the
    arm, so tier C could not be reached. The failure to guard against is
    the one `hybrid` had: a silent revert to "mock" is indistinguishable
    from an arm that simply did not move.
    """

    def test_real_reaches_the_arm_backend(self):
        from tlod.cli import play_config
        from tlod.config import Config

        cfg = play_config(Config(), camera=5, real=True)
        assert cfg.arm.backend == "feetech"

    def test_without_real_nothing_can_move(self):
        from tlod.cli import play_config
        from tlod.config import Config

        cfg = play_config(Config(), camera=5, real=False)
        assert cfg.arm.backend == "mock"

    def test_the_camera_and_hand_are_real_either_way(self):
        """Tier B and tier C differ only in the arm; both play a real hand."""
        from tlod.cli import play_config
        from tlod.config import Config

        for real in (True, False):
            cfg = play_config(Config(), camera=5, real=real)
            assert cfg.camera.source == "opencv"
            assert cfg.camera.index == 5
            assert cfg.vision.detector == "mediapipe"

    def test_the_arm_config_is_otherwise_untouched(self):
        from tlod.cli import play_config
        from tlod.config import Config

        base = Config.from_dict({
            "arm": {"calibration": "calib/mine.json", "port": "/dev/ttyACM0"},
            "safety": {"max_speed": 3.5},
        })
        cfg = play_config(base, camera=0, real=True)
        assert cfg.arm.calibration == "calib/mine.json"
        assert cfg.arm.port == "/dev/ttyACM0"
        assert cfg.safety.max_speed == 3.5

    def test_real_refuses_to_run_without_extrinsics(self):
        """The gate has to fire before anything opens a serial port.

        Without extrinsics the camera's pose is a guess, so every strike
        would be aimed through a guessed transform -- at a hand.
        """
        import pytest

        from tlod.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main(["play", "--real", "--yes"])
        assert "extrinsics" in str(excinfo.value)


class TestStrikeAppliesTheTorqueLimit:
    """The lowered torque limit is what makes an arm swinging at a hand
    yield instead of push. It reaches the servos only through
    `backend.set_torque_limit`, looked up with getattr -- so a broken path
    fails silently, and the arm just hits harder.
    """

    def _controller(self, torque_limit=800):
        from tlod.arm.controller import ArmController, SafetyLimits
        from tlod.arm.mock import MockArm

        backend = MockArm(q0=np.concatenate([HOME, [0.0]]))
        backend.set_torque_limit(torque_limit)
        controller = ArmController(backend, SafetyLimits(), 100.0)
        controller.start()
        return controller

    def test_a_strike_lowers_the_limit_and_puts_it_back(self):
        from tlod.arm import model
        from tlod.arm.primitives import Strike, StrikeLimits

        controller = self._controller()
        limits = StrikeLimits()
        start = model.tool_pose(HOME).xyz()
        strike = Strike([start[0], start[1], start[2] - 0.05], limits, duration=0.05)
        try:
            strike.start(controller)
            assert controller.backend._torque_limit == limits.torque_limit, (
                "the strike never lowered the servo torque limit")
            deadline = time.perf_counter() + 3.0
            while not strike.step(controller, 0.01) and time.perf_counter() < deadline:
                time.sleep(0.01)
            assert strike.finished, "strike never completed"
            assert controller.backend._torque_limit == limits.normal_torque_limit
        finally:
            controller.stop(park=False)

    def test_the_restored_limit_follows_the_config(self):
        """`StrikeLimits.normal_torque_limit` defaults to 800 only because
        `arm.torque_limit` does. Changing one and not the other would let
        the first strike restore a strength the config asked against."""
        from tlod.cli import build_strike_limits
        from tlod.config import Config

        limits = build_strike_limits(Config.from_dict({"arm": {"torque_limit": 500}}))
        assert limits.normal_torque_limit == 500
        assert limits.torque_limit == 350, "the strike-time cap is not a config knob"

    def test_game_speeds_cannot_exceed_the_configured_cap(self):
        """`controller._write` uses an explicit max_speed verbatim, so a
        StrikeLimits faster than `safety.max_speed` overrides it rather
        than being clamped by it -- on the one motion aimed at a person."""
        from tlod.cli import build_strike_limits
        from tlod.config import Config

        slow = build_strike_limits(Config.from_dict({"safety": {"max_speed": 1.5}}))
        assert slow.strike_speed <= 1.5
        assert slow.retract_speed <= 1.5

        fast = build_strike_limits(Config.from_dict({"safety": {"max_speed": 99.0}}))
        assert fast.strike_speed == 3.5, "the cap must not speed the strike up"


class TestServoLoadContactWiring:
    """Contact on hardware reads Present_Load through the same controller
    the control loop commands, which is the only place the servo bus is
    serialised. Nothing had ever constructed this sensor."""

    def _load_arm(self):
        from tlod.arm.mock import MockArm
        from tlod.types import JointState

        class LoadReportingArm(MockArm):
            """MockArm that also reports Present_Load, as the STS3215 does."""

            def __init__(self):
                super().__init__(q0=np.concatenate([HOME, [0.0]]))
                self.load = np.zeros(6)

            def read(self):
                s = super().read()
                return JointState(q=s.q, stamp=s.stamp, dq=s.dq, load=self.load.copy())

        return LoadReportingArm()

    def test_it_fires_through_a_live_controller(self):
        from tlod.arm.controller import ArmController, SafetyLimits
        from tlod.game.contact import ServoLoadContactSensor

        backend = self._load_arm()
        controller = ArmController(backend, SafetyLimits(), 100.0)
        controller.start()
        try:
            # Exactly how cmd_play builds it: the controller's own state()
            # accessor, so reads take the bus lock the control loop uses.
            sensor = ServoLoadContactSensor(controller.state, threshold=0.12)
            backend.load[2] = 0.20          # elbow already loaded by the pose
            sensor.arm()
            assert sensor.poll() is None, "resting posture scored as a hit"
            backend.load[2] = 0.40          # something resisted
            event = sensor.poll()
            assert event is not None and event.source == "servo_load"
            assert sensor.poll() is None, "fired twice on one arming"
        finally:
            controller.stop(park=False)

    def test_a_failed_bus_read_does_not_escape(self):
        """A read failure here would reach RobotApp's control loop, which
        answers a policy exception by e-stopping -- freezing the arm
        mid-swing above the hand it was aiming at. Sync-read failures are a
        known event on this bus, so this is a live path."""
        from tlod.game.contact import ServoLoadContactSensor

        def failing():
            raise RuntimeError("sync read failed")

        sensor = ServoLoadContactSensor(failing)
        sensor.arm()
        assert sensor.poll() is None
        assert sensor.read_failures == 2

    def test_a_missed_baseline_is_taken_later_not_assumed_zero(self):
        """If the read at arm() fails, a zero baseline would turn the
        pose's own resting load into a hit on the first tick."""
        from tlod.game.contact import ServoLoadContactSensor
        from tlod.types import JointState

        state = {"load": None}
        sensor = ServoLoadContactSensor(
            lambda: JointState(q=np.zeros(6), stamp=0.0, load=state["load"]),
            threshold=0.12,
        )
        sensor.arm()
        state["load"] = np.array([0.0, 0.45, 0.40, 0.35, 0.0, 0.0])
        assert sensor.poll() is None, "resting load scored as contact"
        state["load"] = np.array([0.0, 0.45, 0.60, 0.35, 0.0, 0.0])
        assert sensor.poll() is not None


def test_the_overlay_stream_starts_and_stops_cleanly():
    """Watching matters more than it sounds on a headless board: without
    it a feint and a strike look the same from across the table, and so
    do a tracked hand and a lost one. The failure to guard against is a
    render thread outliving the run and holding the serial bus."""
    import socket

    from tlod.cli import _serve_overlay

    app, _ = build(TrackHandPolicy())
    with socket.socket() as probe:      # a port nothing else is on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    assert _serve_overlay(app, app.locator.projector, 0) is None, "started when not asked"

    with app:
        server = _serve_overlay(app, app.locator.projector, port)
        assert server is not None
        time.sleep(0.4)
        assert server.latest() is not None, "served no frame"
        server.stop()
    names = {t.name for t in threading.enumerate() if t.is_alive()}
    time.sleep(0.3)
    assert "overlay" not in {t.name for t in threading.enumerate() if t.is_alive()}, names

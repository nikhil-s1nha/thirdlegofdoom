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

import logging
import threading
import time

import numpy as np
import pytest

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

    def test_a_configured_camera_index_survives(self):
        """`--camera` unset must not overwrite the config with 0.

        This is the trap that cost a hardware session. `--camera`
        defaulted to 0 and was written in unconditionally, so
        `camera.index` in a config file was dead text: you could set it to
        11, pass `-c` on the command line, and still open index 0. On an
        Orange Pi 5 the camera is never index 0 -- the Rockchip codecs
        take the low indices -- so the run failed naming an index nobody
        had chosen.
        """
        from tlod.cli import play_config
        from tlod.config import Config

        base = Config.from_dict({"camera": {"index": 11}})
        assert play_config(base, None, real=True).camera.index == 11
        assert play_config(base, 3, real=True).camera.index == 3

    def test_the_camera_flag_defaults_to_leaving_the_config_alone(self):
        """Every --camera, not just play's: they all had the same clobber."""
        from tlod import cli

        seen = {}
        for command in ("play", "hybrid", "record", "vision-check"):
            import unittest.mock as mock

            target = {"play": "cmd_play", "hybrid": "cmd_hybrid",
                      "record": "cmd_record", "vision-check": "cmd_vision_check"}[command]
            with mock.patch.object(
                    cli, target,
                    lambda args, c=command: seen.update({c: args.camera}) or 0):
                assert cli.main([command]) == 0
        assert seen == dict.fromkeys(seen, None), seen

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

    def test_there_is_no_way_to_ask_for_a_torque_sensor(self, monkeypatch):
        """`--contact` is gone, and a stale invocation must fail loudly.

        It offered four ways to judge a round and three were worse in
        ways already measured, so it was not a choice -- it was a way to
        run the wrong one by accident, which is what its `proximity`
        default did for several commits after `height` landed. A script
        or a shell history carrying `--contact press` should stop rather
        than quietly do something else.
        """
        import pytest as _pytest

        from tlod import cli

        seen = {}
        monkeypatch.setattr(cli, "cmd_play", lambda args: seen.update(vars(args)) or 0)
        assert cli.main(["play"]) == 0
        assert "contact" not in seen

        for stale in ("proximity", "press", "servo", "height"):
            with _pytest.raises(SystemExit):
                cli.main(["play", "--contact", stale])

    def test_the_real_arm_is_judged_by_the_encoders(self):
        """Tier C constructs one sensor, and torque is not it.

        Reaching the construction for real needs a camera, an arm and a
        calibration, so this reads the wiring instead. Crude, but it is
        the thing that regressed: the sensor was right and the branch
        selecting it was not.
        """
        import inspect

        from tlod import cli

        body = inspect.getsource(cli.cmd_play)
        assert "CollisionPlaneContactSensor(" in body
        assert "ServoPressContactSensor(" not in body
        assert "ServoLoadContactSensor(" not in body
        assert "SerialContactSensor(" not in body

    def test_the_retired_sensors_are_kept_but_unreachable(self):
        """Kept for their measurements; wired to nothing.

        Deleting them means the next person re-runs the same three
        experiments, so they stay -- but a class nothing constructs drifts
        into looking like a live option, which is how `--contact press`
        got run on a supply that could not hold it.
        """
        import inspect

        from tlod import cli
        from tlod.game import contact

        for name in ("ServoLoadContactSensor", "ServoPressContactSensor",
                     "SerialContactSensor"):
            cls = getattr(contact, name)
            assert cls.__doc__.lstrip().splitlines()[2].strip().startswith("UNUSED"), (
                f"{name} is not constructed anywhere; its docstring has to say so")
            # Named in a comment saying why they are absent is fine;
            # constructed is not.
            assert f"{name}(" not in inspect.getsource(cli)

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
            sensor.arm(blank_for=0.0)
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
        sensor.arm(blank_for=0.0)
        assert sensor.poll() is None
        # One, not two: arm() no longer reads. The baseline is taken on
        # the first poll instead, so that it measures an arm already
        # moving rather than one still hovering.
        assert sensor.read_failures == 1

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
        sensor.arm(blank_for=0.0)
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


class TestServoPressContact:
    """The sensor that survived the hardware A/B.

    Measured on the arm across nothing / a book / a hand, peak load during
    the swing read 0.330 / 0.326 / 0.350 -- a rigid book between the other
    two, so the swing carries no information about what it hit. Held still
    at the bottom the same three read 0.001 / 0.038 / 0.037. These tests
    pin the two properties that turn the second set of numbers into a
    sensor: nothing is read until the arm is pressing, and the reference is
    the hover rather than anything sampled mid-swing.
    """

    def _rig(self, load_at_rest=0.20):
        from tlod.arm.controller import ArmController, SafetyLimits
        from tlod.arm.mock import MockArm
        from tlod.types import JointState

        class LoadReportingArm(MockArm):
            def __init__(self):
                super().__init__(q0=np.concatenate([HOME, [0.0]]))
                self.load = np.zeros(6)

            def read(self):
                s = super().read()
                return JointState(q=s.q, stamp=s.stamp, dq=s.dq, load=self.load.copy())

        backend = LoadReportingArm()
        backend.load[2] = load_at_rest
        controller = ArmController(backend, SafetyLimits(), 100.0)
        controller.start()
        return backend, controller

    def test_the_swings_own_braking_torque_is_never_read(self):
        """The failure that retired the previous sensor, as a test.

        Load reaches its ceiling during every swing, empty table included,
        because the arm is braking its own mass. A sensor that reads then
        cannot help but score a dodge as a hit.
        """
        from tlod.game.contact import ServoPressContactSensor

        backend, controller = self._rig()
        try:
            sensor = ServoPressContactSensor(controller.state, threshold=0.02)
            sensor.arm()
            assert sensor.poll(pressing=False) is None       # baseline, at hover
            backend.load[2] = 0.55                           # mid-swing, at the cap
            for _ in range(50):
                assert sensor.poll(pressing=False) is None, \
                    "fired on the strike's own braking torque"
                time.sleep(0.01)
        finally:
            controller.stop(park=False)

    def test_it_fires_only_after_the_press_has_settled(self):
        from tlod.game.contact import ServoPressContactSensor

        backend, controller = self._rig()
        try:
            sensor = ServoPressContactSensor(controller.state, threshold=0.02,
                                             settle=0.15)
            sensor.arm()
            sensor.poll(pressing=False)                      # baseline at 0.20
            backend.load[2] = 0.24                           # 0.04 rise: something there
            assert sensor.poll(pressing=True) is None, "fired before settling"
            t0 = time.perf_counter()
            event = None
            while event is None and time.perf_counter() - t0 < 1.0:
                event = sensor.poll(pressing=True)
                time.sleep(0.005)
            assert event is not None and event.source == "servo_press"
            assert time.perf_counter() - t0 >= 0.15, "settle window not honoured"
            assert sensor.poll(pressing=True) is None, "fired twice on one arming"
        finally:
            controller.stop(park=False)

    def test_an_empty_press_is_a_dodge(self):
        """0.001 over nothing, against a 0.02 threshold."""
        from tlod.game.contact import ServoPressContactSensor

        backend, controller = self._rig()
        try:
            sensor = ServoPressContactSensor(controller.state, threshold=0.02,
                                             settle=0.05)
            sensor.arm()
            sensor.poll(pressing=False)
            backend.load[2] = 0.201                          # the measured 0.001
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.4:
                assert sensor.poll(pressing=True) is None, "empty air scored as a hit"
                time.sleep(0.005)
            assert sensor.peak_rise < 0.02
        finally:
            controller.stop(park=False)

    def test_a_caller_that_forgets_pressing_scores_dodges_not_hits(self):
        from tlod.game.contact import ServoPressContactSensor

        backend, controller = self._rig()
        try:
            sensor = ServoPressContactSensor(controller.state, threshold=0.02,
                                             settle=0.0)
            sensor.arm()
            sensor.poll()
            backend.load[2] = 0.60
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.2:
                assert sensor.poll() is None
                time.sleep(0.005)
        finally:
            controller.stop(park=False)


class TestStrikeHoldsBeforeRetracting:
    """`press_hold` is what gives the sensor above anything to read."""

    def _rig(self):
        from tlod.arm.controller import ArmController, SafetyLimits
        from tlod.arm.mock import MockArm

        controller = ArmController(MockArm(q0=np.concatenate([HOME, [0.0]])),
                                   SafetyLimits(), 100.0)
        controller.start()
        return controller

    def test_it_reports_pressing_and_stays_down_for_the_hold(self):
        from tlod.arm.primitives import Strike, StrikeLimits

        controller = self._rig()
        try:
            limits = StrikeLimits()
            limits.press_hold = 0.25
            start = controller.pose()
            motion = Strike([start.x, start.y, start.z - 0.05], limits, duration=0.15)
            motion.start(controller)
            assert not motion.pressing, "pressing before the drop has begun"

            pressing_at = None
            depths = []
            t0 = time.perf_counter()
            while not motion.step(controller, 0.01) and time.perf_counter() - t0 < 3.0:
                if motion.pressing:
                    if pressing_at is None:
                        pressing_at = time.perf_counter()
                    depths.append(controller.commanded.copy())
                time.sleep(0.01)

            assert pressing_at is not None, "never reported pressing"
            held = time.perf_counter() - pressing_at
            assert held >= limits.press_hold, f"held only {held * 1e3:.0f} ms"
            # The commanded pose must not drift during the hold: the whole
            # point is a static lean on whatever is underneath.
            assert np.allclose(depths[0], depths[-1], atol=1e-9), \
                "commanded pose moved during the press"
            assert not motion.pressing, "still pressing after finishing"
        finally:
            controller.stop(park=False)

    def test_press_hold_of_zero_keeps_the_old_drop_and_go_behaviour(self):
        from tlod.arm.primitives import Strike, StrikeLimits

        controller = self._rig()
        try:
            limits = StrikeLimits()
            limits.press_hold = 0.0
            start = controller.pose()
            motion = Strike([start.x, start.y, start.z - 0.05], limits, duration=0.15)
            motion.start(controller)
            t0 = time.perf_counter()
            while not motion.step(controller, 0.01) and time.perf_counter() - t0 < 3.0:
                assert not motion.pressing
                time.sleep(0.01)
            assert motion.finished
        finally:
            controller.stop(park=False)


class TestStrikeAimsBelowTheHand:
    """A touch that spends no torque scores as a dodge.

    `ServoPressContactSensor` reads the torque the arm is still spending
    while held down, and torque is only spent when the arm is blocked
    short of its commanded floor. So the floor has to sit *below* the
    estimated hand surface. It sat 5 mm above it for a while, which made
    detection a function of how thick the hand happened to be that round:
    measured, a hand blocking at 29 mm against a 27 mm floor read 0.037,
    and the same hand held flatter reached the floor untouched and read
    0.001 after landing on it.
    """

    def _controller(self, min_height):
        from tlod.arm.controller import ArmController, SafetyLimits
        from tlod.arm.mock import MockArm

        controller = ArmController(MockArm(q0=np.concatenate([HOME, [0.0]])),
                                   SafetyLimits(min_height=min_height), 100.0)
        controller.start()
        return controller

    def _floor_reached(self, controller, hand_z, limits):
        from tlod.arm.primitives import Strike

        start = controller.pose()
        motion = Strike([start.x, start.y, hand_z], limits, duration=0.1)
        motion.start(controller)
        t0 = time.perf_counter()
        while not motion.step(controller, 0.01) and time.perf_counter() - t0 < 3.0:
            time.sleep(0.01)
        return controller.pose().z

    def test_the_commanded_floor_is_below_the_estimated_hand(self):
        from tlod.arm.primitives import StrikeLimits

        limits = StrikeLimits()
        limits.press_hold = 0.0
        controller = self._controller(min_height=0.001)
        try:
            hand_z = controller.pose().z - 0.05
            floor = self._floor_reached(controller, hand_z, limits)
            assert floor < hand_z, (
                f"floor {floor * 1e3:.1f} mm is at or above the hand at "
                f"{hand_z * 1e3:.1f} mm, so a touch would spend no torque")
            assert abs((hand_z - floor) - limits.press_depth) < 2e-3
        finally:
            controller.stop(park=False)

    def test_min_height_still_has_the_last_word(self):
        """The floor that keeps the paddle off the table outranks press_depth.

        Worth pinning because it is also the way to set press_depth and
        see no change at all: a min_height above the intended floor
        silently clamps it back, and the only symptom is a strike that
        stops high.
        """
        from tlod.arm.primitives import StrikeLimits

        limits = StrikeLimits()
        limits.press_hold = 0.0
        controller = self._controller(min_height=0.06)
        try:
            hand_z = controller.pose().z - 0.05
            assert hand_z - limits.press_depth < 0.06, "test does not exercise the clamp"
            floor = self._floor_reached(controller, hand_z, limits)
            assert floor >= 0.06 - 2e-3, f"drove to {floor * 1e3:.1f} mm, under min_height"
        finally:
            controller.stop(park=False)


class TestCollisionPlaneContact:
    """Did the paddle stop inside the band where the hand is?

    Grounded in a measured pose. Driven to the joint angles at which the
    gripper rests on the table, FK reports the tool at +0.2 mm -- so model
    z is height above the work surface, the floor sits at 5 mm, and a hand
    of 22-29 mm stops the paddle 17-24 mm short of it. Against an 8 mm
    threshold and an unobstructed press that converges within ~3 mm, that
    is a decision with an order of magnitude of margin, taken from
    encoders rather than from a filtered torque estimate.
    """

    def test_it_fires_when_the_paddle_is_stopped_short(self):
        from tlod.game.contact import CollisionPlaneContactSensor

        tool = np.array([0.22, 0.0, 0.005])
        sensor = CollisionPlaneContactSensor(lambda: 0.005, settle=0.05)
        sensor.arm()
        assert sensor.poll(pressing=True, tool_xyz=tool) is None, "fired before settling"

        tool = np.array([0.22, 0.0, 0.022])        # a hand, 17 mm of it
        t0 = time.perf_counter()
        event = None
        while event is None and time.perf_counter() - t0 < 1.0:
            event = sensor.poll(pressing=True, tool_xyz=tool)
            time.sleep(0.005)
        assert event is not None and event.source == "collision_plane"
        assert time.perf_counter() - t0 >= 0.05, "settle window not honoured"
        assert sensor.poll(pressing=True) is None, "fired twice on one arming"

    def test_reaching_the_floor_is_a_dodge(self):
        from tlod.game.contact import CollisionPlaneContactSensor

        # An unobstructed press converges to within about 3 mm, and can
        # sit slightly under the floor. Neither is a hand.
        tool = np.array([0.22, 0.0, 0.008])
        sensor = CollisionPlaneContactSensor(lambda: 0.005, margin=0.010, settle=0.0)
        sensor.arm()
        sensor.poll(pressing=True, tool_xyz=tool)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.2:
            assert sensor.poll(pressing=True, tool_xyz=tool) is None, \
                "tracking error scored as a hit"
            time.sleep(0.005)
        assert sensor.peak_rise < sensor.margin

    def test_nothing_is_read_until_the_arm_is_pressing(self):
        """Mid-swing the paddle is far above the floor by definition."""
        from tlod.game.contact import CollisionPlaneContactSensor

        tool = np.array([0.22, 0.0, 0.080])
        sensor = CollisionPlaneContactSensor(lambda: 0.005, settle=0.0)
        sensor.arm()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.2:
            assert sensor.poll(pressing=False, tool_xyz=tool) is None, \
                "fired during the descent"
            time.sleep(0.005)

    def test_a_read_failure_is_a_dodge_not_an_exception(self):
        """An exception here reaches the control loop, which e-stops --
        freezing the arm mid-swing directly above the hand."""
        from tlod.game.contact import CollisionPlaneContactSensor

        def boom():
            raise OSError("sync read failed")

        tool = np.array([0.22, 0.0, 0.030])
        sensor = CollisionPlaneContactSensor(boom, settle=0.0)
        sensor.arm()
        assert sensor.poll(pressing=True, tool_xyz=tool) is None
        assert sensor.poll(pressing=True, tool_xyz=tool) is None
        assert sensor.read_failures == 1

    def test_the_floor_leaves_the_measured_table_clear(self):
        """Regression on the frame itself, not on the sensor.

        The config used to say the table sat 17 mm below the base, which
        made every clearance figure derived from it wrong by 17 mm in the
        dangerous direction. The pose below is measured: gripper resting
        on the table.
        """
        from tlod.arm import model
        from tlod.arm.primitives import StrikeLimits
        from tlod.config import Config

        touching = np.array([-0.304, 0.210, 0.302, 1.068, -0.005])
        table_z = model.tool_pose(touching).z
        assert abs(table_z) < 0.003, f"table is at {table_z * 1e3:.1f} mm, not ~0"

        cfg = Config.load("configs/opi.yaml")
        limits = StrikeLimits()
        floor = max(cfg.vision.hand_height - limits.press_depth, cfg.safety.min_height)
        assert floor > table_z, "the commanded floor is at or below the table"
        assert floor - table_z >= 0.004, (
            f"only {(floor - table_z) * 1e3:.1f} mm of air over the table")
        # And deep enough that a thin hand still stops the paddle well
        # clear of the threshold.
        from tlod.game.contact import CollisionPlaneContactSensor
        thin_hand = 0.020
        assert thin_hand - floor > CollisionPlaneContactSensor(lambda: 0.0).margin * 1.5

    def test_it_reports_its_numbers_whichever_way_the_round_went(self):
        """A verdict alone is unfalsifiable from outside the arm.

        "dodged" looks identical whether the paddle stopped on a hand and
        the margin was too wide, the floor sat above the hand so there was
        nothing to stop short of, or the hand compressed to the floor.
        Each has a different fix. Several rounds of guesswork happened for
        want of this line.
        """
        from tlod.game.contact import CollisionPlaneContactSensor

        tool = np.array([0.22, 0.0, 0.007])
        sensor = CollisionPlaneContactSensor(lambda: 0.005, settle=0.0)
        assert "no reading" in sensor.report()

        sensor.arm()
        hand = np.array([0.22, 0.0, 0.022])
        sensor.poll(pressing=True, tool_xyz=tool, hand_xyz=hand)
        for _ in range(5):
            sensor.poll(pressing=True, tool_xyz=tool, hand_xyz=hand)
            time.sleep(0.005)
        report = sensor.report()
        assert "7 mm" in report and "5 mm" in report and "22 mm" in report, report
        assert "+2 mm short" in report, report

    def test_the_game_polls_the_sensor_while_the_paddle_is_pressing(self):
        """The wiring, which is the only thing below the whole loop can miss.

        A sensor firing in isolation says nothing about whether the game
        ever calls it while the arm is down, with a tool position, for
        long enough to clear `settle`. That is `Strike.press_hold`, the
        strike state and `run_motion` acting together, and every one of
        those has been wrong at some point in this file's history.

        Deliberately not asserted by scoring a hit off the mock arm's own
        tracking: it stops ~10 mm above the floor on its own, so a test
        written that way passes with no hand anywhere near it.
        """
        import sys

        sys.path.insert(0, "tests")
        from test_game import fake_robot

        from tlod.arm.primitives import StrikeLimits
        from tlod.game.contact import CollisionPlaneContactSensor
        from tlod.game.handslap import Difficulty, HandSlapGame, Personality

        hand = np.array([0.22, 0.0, 0.022])
        seen: list[tuple[float, object]] = []

        class Spy(CollisionPlaneContactSensor):
            def poll(self, pressing=False, tool_xyz=None, **kw):
                if pressing:
                    seen.append((time.perf_counter(), tool_xyz))
                return super().poll(pressing=pressing, tool_xyz=tool_xyz, **kw)

        sensor = Spy(lambda: 0.005)
        difficulty = Difficulty.preset("easy")
        difficulty.feint_probability = 0.0     # strikes only; feints judge elsewhere

        robot = fake_robot(hand)
        game = HandSlapGame(difficulty, limits=StrikeLimits(),
                            personality=Personality(enabled=False),
                            contact=sensor, seed=1)
        game.truth_provider = lambda: hand
        game.running = True
        try:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 25.0 and game.strikes < 1:
                game.update(robot, None, 0.01)
                time.sleep(0.01)
            # Let the strike finish so the whole pressing window is seen.
            while time.perf_counter() - t0 < 25.0 and game.state == "strike":
                game.update(robot, None, 0.01)
                time.sleep(0.01)

            assert game.strikes >= 1, "the game never struck; test proved nothing"
            assert seen, "the sensor was never polled while pressing"
            assert all(t is not None for _, t in seen), \
                "polled while pressing but with no tool position to judge from"
            window = seen[-1][0] - seen[0][0]
            assert window >= sensor.settle, (
                f"only {window * 1e3:.0f} ms of pressing reached the sensor, "
                f"which needs {sensor.settle * 1e3:.0f} ms -- press_hold is too "
                f"short or the strike state leaves early")
        finally:
            robot.controller.stop(park=False)

    def test_a_blocked_paddle_scores_a_hit_and_a_clear_one_does_not(self):
        """Both directions, at the sensor, where the heights are controlled."""
        from tlod.game.contact import CollisionPlaneContactSensor

        def verdict(reached):
            sensor = CollisionPlaneContactSensor(lambda: 0.005, settle=0.02)
            sensor.arm()
            tool = np.array([0.22, 0.0, reached])
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.5:
                if sensor.poll(pressing=True, tool_xyz=tool) is not None:
                    return True
                time.sleep(0.005)
            return False

        assert verdict(0.022), "a hand 17 mm thick scored as a dodge"
        assert not verdict(0.005), "reaching the floor scored as a hit"
        assert not verdict(0.003), "overshooting the floor scored as a hit"


class TestStrikeGeometryIsSelfConsistent:
    """max_drop, hover_height and press_depth are one constraint, not three.

    A strike from `hover_height` above the hand must travel that plus
    `press_depth` to put the paddle below it, and `max_drop` caps the
    travel. When 80 + 17 > 80 the floor silently rose to exactly the hand
    plane, and the contact sensor was asked to separate a hit from a miss
    across a band of zero width. It ran a full hardware session that way:
    every log line read "floor 22 mm, hand 22 mm". Nothing downstream can
    catch that, because a floor at the hand plane is an ordinary number.
    """

    def test_the_defaults_can_reach_below_the_hand(self):
        from tlod.arm.primitives import StrikeLimits

        limits = StrikeLimits()
        assert limits.hover_height + limits.press_depth <= limits.max_drop, (
            f"hover {limits.hover_height * 1e3:.0f} + press "
            f"{limits.press_depth * 1e3:.0f} > max_drop {limits.max_drop * 1e3:.0f}")
        assert limits.reachable_floor_offset == limits.press_depth

    def test_the_configured_arm_lands_below_the_hand(self):
        from tlod.arm.primitives import StrikeLimits
        from tlod.config import Config
        from tlod.game.contact import CollisionPlaneContactSensor

        cfg = Config.load("configs/opi.yaml")
        limits = StrikeLimits()
        hand = cfg.vision.hand_height
        hover = hand + limits.hover_height
        floor = max(hand - limits.press_depth,
                    hover - limits.clamp_drop(hover - (hand - limits.press_depth)),
                    cfg.safety.min_height)
        band = hand - floor
        assert band > 0, f"floor {floor * 1e3:.0f} mm is at or above the hand"
        margin = CollisionPlaneContactSensor(lambda: 0.0).margin
        assert band > margin * 3, (
            f"band is only {band * 1e3:.0f} mm against a {margin * 1e3:.0f} mm margin")

    def test_an_inconsistent_geometry_says_so(self):
        from tlod.arm.primitives import StrikeLimits

        limits = StrikeLimits()
        limits.hover_height = 0.08
        limits.press_depth = 0.017
        limits.max_drop = 0.08
        assert limits.reachable_floor_offset < 0.001, \
            "the paddle cannot get below the hand, and the property should say so"

    def test_the_peak_is_reported_in_the_units_it_was_measured_in(self):
        """"peak load rise 0.008" for an 8 mm shortfall reads as nothing seen."""
        from tlod.game.contact import CollisionPlaneContactSensor

        tool = np.array([0.22, 0.0, 0.030])
        sensor = CollisionPlaneContactSensor(lambda: 0.005, settle=0.0)
        sensor.arm()
        sensor.poll(pressing=True, tool_xyz=tool)
        sensor.poll(pressing=True, tool_xyz=tool)
        assert "mm" in sensor.peak_summary()
        assert "25 mm" in sensor.peak_summary(), sensor.peak_summary()


class TestCalibrateFindsItsOwnIntrinsics:
    """`calibrate extrinsics` should not ask for a path the config holds.

    The camera having moved is exactly when this command is reached for,
    and refusing to start until a path is looked up -- one the rest of
    the pipeline already loads from `camera.intrinsics` -- puts a step
    between a diagnosis and its fix for no reason.
    """

    def _args(self, tmp_path, intrinsics=""):
        from types import SimpleNamespace

        return SimpleNamespace(
            what="extrinsics", intrinsics=intrinsics, output=str(tmp_path / "e.npz"),
            camera=0, pattern="9x6", square=0.025, views=12, gripper=0.0,
            heights="", marker="red", preview=0, fisheye=False, timeout=60,
            sim=False, config=None)

    def test_it_falls_back_to_the_config(self, tmp_path, monkeypatch):
        from tlod import cli
        from tlod.config import Config

        cfg = Config()
        cfg.camera.intrinsics = str(tmp_path / "absent.npz")
        monkeypatch.setattr(cli.Config, "load", staticmethod(lambda *a, **k: cfg))

        # The path from the config is the one it complains about, which is
        # only possible if it looked there.
        with pytest.raises(SystemExit) as e:
            cli.cmd_calibrate(self._args(tmp_path))
        assert "absent.npz" in str(e.value)
        assert "config" in str(e.value)

    def test_an_explicit_flag_still_wins(self, tmp_path, monkeypatch):
        from tlod import cli
        from tlod.config import Config

        cfg = Config()
        cfg.camera.intrinsics = str(tmp_path / "from_config.npz")
        monkeypatch.setattr(cli.Config, "load", staticmethod(lambda *a, **k: cfg))

        with pytest.raises(SystemExit) as e:
            cli.cmd_calibrate(self._args(tmp_path, intrinsics=str(tmp_path / "flag.npz")))
        assert "flag.npz" in str(e.value)
        assert "from_config.npz" not in str(e.value)

    def test_extrinsics_will_not_write_over_the_intrinsics(self, tmp_path, monkeypatch):
        """The one that cost a lens calibration.

        `-o` defaulted to calib/intrinsics.npz for the whole `calibrate`
        command -- right for intrinsics, and for extrinsics it wrote its
        own solve straight over a chessboard calibration. Nothing warns:
        the run prints a camera position and an RMS and looks like it
        worked, and the loss only surfaces later as vision that is wrong
        in a way no amount of recalibrating extrinsics can fix.
        """
        from tlod import cli
        from tlod.config import Config

        cfg = Config()
        cfg.camera.intrinsics = str(tmp_path / "intrinsics.npz")
        cfg.camera.extrinsics = str(tmp_path / "extrinsics.npz")
        (tmp_path / "intrinsics.npz").write_bytes(b"precious")
        monkeypatch.setattr(cli.Config, "load", staticmethod(lambda *a, **k: cfg))

        args = self._args(tmp_path)
        args.output = str(tmp_path / "intrinsics.npz")
        with pytest.raises(SystemExit) as e:
            cli.cmd_calibrate(args)
        assert "refusing" in str(e.value)
        assert (tmp_path / "intrinsics.npz").read_bytes() == b"precious"

    def test_each_subcommand_defaults_to_its_own_file(self, tmp_path, monkeypatch):
        from tlod import cli
        from tlod.config import Config

        cfg = Config()
        cfg.camera.intrinsics = str(tmp_path / "i.npz")
        cfg.camera.extrinsics = str(tmp_path / "e.npz")
        monkeypatch.setattr(cli.Config, "load", staticmethod(lambda *a, **k: cfg))

        # Extrinsics with no -o must not resolve to the intrinsics path.
        # It fails later for want of an intrinsics file, which is fine --
        # what matters is that it did not name i.npz as its destination.
        args = self._args(tmp_path)
        args.output = ""
        with pytest.raises(SystemExit) as e:
            cli.cmd_calibrate(args)
        assert "refusing" not in str(e.value)
        assert "i.npz" in str(e.value), "should be complaining about reading it"

    def test_neither_set_says_how_to_get_them(self, tmp_path, monkeypatch):
        from tlod import cli
        from tlod.config import Config

        cfg = Config()
        cfg.camera.intrinsics = ""
        monkeypatch.setattr(cli.Config, "load", staticmethod(lambda *a, **k: cfg))

        with pytest.raises(SystemExit) as e:
            cli.cmd_calibrate(self._args(tmp_path))
        assert "calibrate intrinsics" in str(e.value)


class TestEstopSurvivesTheBusItIsStoppingFor:
    """An e-stop must not need the thing that just failed.

    From the rig: a sync read failed mid-strike, the control loop answered
    it by calling estop(), estop() read the bus, the read failed again,
    and the exception took out the control thread -- with the arm still
    commanded downward, above the hand it was aiming at. The safety action
    was the one action that could not tolerate the failure it exists to
    handle.
    """

    def _controller(self, fail_reads: bool = False, fail_writes: bool = False):
        from tlod.arm.controller import ArmController, SafetyLimits
        from tlod.arm.mock import MockArm

        class FlakyArm(MockArm):
            def __init__(self):
                super().__init__(q0=np.concatenate([HOME, [0.0]]))
                self.break_reads = False
                self.break_writes = False

            def read(self):
                if self.break_reads:
                    raise OSError("sync read failed: [TxRxResult] There is no status packet!")
                return super().read()

            def write(self, q):
                if self.break_writes:
                    raise OSError("sync write failed")
                return super().write(q)

        backend = FlakyArm()
        controller = ArmController(backend, SafetyLimits(), 100.0)
        controller.start()
        backend.break_reads = fail_reads
        backend.break_writes = fail_writes
        return backend, controller

    def test_it_still_stops_when_the_bus_is_gone(self):
        backend, controller = self._controller()
        try:
            controller.goto_pose(controller.pose().offset(dz=-0.02), duration=0.2)
            before = controller.commanded.copy()
            backend.break_reads = True
            controller.estop()                       # must not raise
            assert controller.estopped
            # Frozen at the last command, since there is no fresh reading.
            assert np.allclose(controller.commanded, before)
        finally:
            backend.break_reads = False
            controller.stop(park=False)

    def test_it_still_stops_when_writes_fail_too(self):
        backend, controller = self._controller(fail_reads=True, fail_writes=True)
        try:
            controller.estop()
            assert controller.estopped, "a failed write left the stop un-engaged"
        finally:
            backend.break_reads = backend.break_writes = False
            controller.stop(park=False)

    def test_a_command_after_a_failed_estop_is_still_refused(self):
        """The point of the flag: nothing further reaches the servos."""
        backend, controller = self._controller()
        try:
            backend.break_reads = True
            controller.estop()
            backend.break_reads = False
            before = controller.commanded.copy()
            controller.servo_pose(controller.pose().offset(dz=-0.05))
            assert np.allclose(controller.commanded, before), \
                "a command got through after the e-stop"
        finally:
            controller.stop(park=False)


class TestTheHoldIsSizedToTheSensor:
    """`press_hold` is stall time, so it is spent, not chosen freely.

    Holding at the bottom means the servos stalled against a hand at
    their torque limit for the whole window, every strike, and sustained
    stall current is what an undersized supply has least of. It went in
    at 450 ms -- sized for the torque sensor's filter -- and the bus began
    dropping transactions the same afternoon. The encoder-based sensor
    needs 120 ms, so the default is that plus margin and only a sensor
    that needs more stretches it.
    """

    def test_the_default_covers_the_encoder_sensor(self):
        from tlod.arm.primitives import StrikeLimits
        from tlod.game.contact import CollisionPlaneContactSensor

        limits, sensor = StrikeLimits(), CollisionPlaneContactSensor(lambda: 0.0)
        assert limits.press_hold > sensor.settle, (
            f"hold {limits.press_hold * 1e3:.0f} ms cannot cover a "
            f"{sensor.settle * 1e3:.0f} ms settle; every round scores as a dodge")
        # And not extravagantly longer, since the cost is stall current.
        assert limits.press_hold < sensor.settle + 0.15

    def test_a_slower_sensor_stretches_the_hold(self):
        from tlod.arm.primitives import StrikeLimits
        from tlod.cli import _size_hold_to
        from tlod.game.contact import ServoPressContactSensor

        limits = StrikeLimits()
        sensor = ServoPressContactSensor(lambda: None)
        assert limits.press_hold < sensor.settle, "test does not exercise the stretch"
        _size_hold_to(limits, sensor)
        assert limits.press_hold > sensor.settle

    def test_a_fast_sensor_does_not_shorten_it(self):
        from tlod.arm.primitives import StrikeLimits
        from tlod.cli import _size_hold_to
        from tlod.game.contact import CollisionPlaneContactSensor

        limits = StrikeLimits()
        before = limits.press_hold
        _size_hold_to(limits, CollisionPlaneContactSensor(lambda: 0.0, settle=0.01))
        assert limits.press_hold == before

    def test_cli_has_a_logger(self):
        """Every `log.` in cli.py was a NameError waiting for its branch.

        One of them killed the overlay thread, inside the handler whose
        job was to swallow a render failure.
        """
        from tlod import cli

        assert isinstance(getattr(cli, "log", None), logging.Logger)

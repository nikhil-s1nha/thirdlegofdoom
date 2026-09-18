"""Command line entry point.

Commands are grouped by what they are for:

  build     sim, hybrid, play      -- run the robot
  measure   bench                  -- turn estimates into measurements
  setup     cameras, ports, calibrate, first-light

`hybrid` is the important one before hardware exists: your real webcam and
your real hand driving a simulated arm. It exercises every part of the
perception stack for real.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import replace
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Kept in step with MARKER_BANDS in tlod.vision.calibrate_flow, and named
# here rather than imported from it so that `tlod --help` does not have to
# load OpenCV. cmd_calibrate checks the two agree.
MARKER_COLOURS = ("green", "blue", "yellow", "magenta", "red")

from tlod.config import Config

# This module had no logger of its own, so every `log.` in it was a
# NameError waiting for its branch to be taken -- one of them killed the
# overlay thread, in the handler meant to swallow a render failure.
log = logging.getLogger(__name__)


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def build_projector(cfg: Config):
    from tlod.vision.calibration import Extrinsics, Intrinsics, Projector, synthetic_projector

    if cfg.camera.intrinsics and cfg.camera.extrinsics:
        return Projector(
            Intrinsics.load(cfg.camera.intrinsics), Extrinsics.load(cfg.camera.extrinsics)
        )
    if cfg.camera.intrinsics:
        # Real lens, assumed pose. Better than nothing, clearly worse than
        # a calibrated mount; the warning is deliberate.
        logging.getLogger(__name__).warning(
            "no extrinsics: using the configured nominal camera pose. "
            "Run `tlod calibrate extrinsics` once the camera is mounted."
        )
        synth = synthetic_projector(
            (cfg.camera.width, cfg.camera.height), cfg.camera.position, cfg.camera.look_at
        )
        return Projector(Intrinsics.load(cfg.camera.intrinsics), synth.extr)
    return synthetic_projector(
        (cfg.camera.width, cfg.camera.height), cfg.camera.position, cfg.camera.look_at
    )


def build_camera(cfg: Config, scene=None, render: bool = False):
    from tlod.vision.camera import MockCamera, OpenCVCamera

    if cfg.camera.source == "mock":
        return MockCamera(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                          scene=scene, render=render)
    return OpenCVCamera(
        index=cfg.camera.index,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        fourcc=cfg.camera.fourcc,
        latency_offset=cfg.camera.latency_offset,
        autofocus=cfg.camera.autofocus,
        autoexposure=cfg.camera.autoexposure,
        exposure=cfg.camera.exposure,
    )


def build_arm(cfg: Config):
    from tlod.arm.mock import MockArm
    from tlod.arm.model import HOME

    if cfg.arm.backend == "mock":
        q0 = np.concatenate([HOME, [0.0]])
        return MockArm(
            q0=q0,
            max_speed=cfg.arm.sim_max_speed,
            accel=cfg.arm.sim_accel,
            latency=cfg.arm.sim_latency,
        )

    from tlod.arm.feetech import Calibration, FeetechArm, default_lerobot_calibration, find_ports

    port = cfg.arm.port
    if not port:
        ports = find_ports()
        if not ports:
            raise SystemExit("no serial ports found. Is the controller board plugged in and powered?")
        if len(ports) > 1:
            raise SystemExit(f"several ports found, set arm.port explicitly: {ports}")
        port = ports[0]

    calib = None
    if cfg.arm.calibration:
        calib = Calibration.load(cfg.arm.calibration)
    elif cfg.arm.lerobot_id:
        calib = Calibration.load(default_lerobot_calibration(cfg.arm.lerobot_id))

    from tlod.arm.feetech import acc_counts

    return FeetechArm(
        port=port,
        baudrate=cfg.arm.baudrate,
        calibration=calib,
        goal_acceleration=acc_counts(cfg.arm.servo_accel),
        torque_limit=cfg.arm.torque_limit,
    )


def build_limits(cfg: Config):
    """Safety limits from config. One place, so no entry point drifts."""
    from tlod.arm.controller import SafetyLimits

    return SafetyLimits(
        max_speed=cfg.safety.max_speed, strike_speed=cfg.safety.strike_speed,
        max_accel=cfg.safety.max_accel, max_jerk=cfg.safety.max_jerk,
        joint_margin=cfg.safety.joint_margin, table_z=cfg.safety.table_z,
        min_height=cfg.safety.min_height, max_radius=cfg.safety.max_radius,
        min_radius=cfg.safety.min_radius, max_height=cfg.safety.max_height,
        command_timeout=cfg.safety.command_timeout, max_tick_dt=cfg.safety.max_tick_dt,
    )


def build_governor(cfg: Config):
    """The power governor, or None if it is switched off."""
    if not cfg.power.governor:
        return None
    from tlod.arm.power import PowerBudget, PowerGovernor, PowerModel

    return PowerGovernor(PowerModel(budget=PowerBudget(
        supply_current=cfg.power.supply_current,
        headroom=cfg.power.headroom,
        min_voltage=cfg.power.min_voltage,
    )))


def build_strike_limits(cfg: Config):
    """Strike bounds, tied to the configured arm rather than to defaults.

    Two couplings that are only coincidences in the dataclass defaults:

    A `Strike` drops the servo torque limit for the duration of the swing
    so the arm yields on contact, then puts it back -- and "back" has to
    mean the limit *this config* asked for. `StrikeLimits.normal_torque_limit`
    defaults to 800 only because `arm.torque_limit` does; a config that
    lowered one and not the other would have its first strike silently
    restore the arm to a strength it was configured away from, for the
    rest of the session.

    And the game's own speeds go to `controller._write` as an explicit
    `max_speed`, which is used verbatim -- `profile_limits` substitutes
    `safety.max_speed` only when nothing was passed. So a StrikeLimits
    faster than `safety.max_speed` does not get clamped by it; it
    overrides it. That is the wrong direction for the one motion aimed at
    a person, and it matters on a rig whose config was deliberately
    slowed to keep the supply from browning out.
    """
    from tlod.arm.primitives import StrikeLimits

    cap = cfg.safety.max_speed
    defaults = StrikeLimits()
    return StrikeLimits(
        normal_torque_limit=cfg.arm.torque_limit,
        strike_speed=min(defaults.strike_speed, cap),
        retract_speed=min(defaults.retract_speed, cap),
    )


def build_detector(cfg: Config, scene=None):
    from tlod.vision.hands import MediaPipeHandDetector, NullHandDetector
    from tlod.vision.scene import SceneHandDetector

    if cfg.vision.detector == "scripted":
        if scene is None:
            raise ValueError("the scripted detector needs a scene")
        return SceneHandDetector(scene)
    if cfg.vision.detector == "none":
        return NullHandDetector()
    try:
        return MediaPipeHandDetector(
            model_path=cfg.vision.model_path,
            num_hands=cfg.vision.num_hands,
            min_detection_confidence=cfg.vision.min_detection_confidence,
            delegate=cfg.vision.delegate,
        )
    except (ImportError, RuntimeError) as exc:
        # Hands are one input among several, and the commands that build a
        # detector also run a camera, object detection, tracking, control
        # and the publisher. Losing MediaPipe should cost the hands, not
        # the board: an ImportError here means the wheel is missing or
        # broken for this interpreter (see pyproject.toml's `hands` extra
        # for which build goes where), a RuntimeError means it imported
        # but could not build a landmarker -- a missing .task bundle, or
        # a delegate the platform cannot service. Both are worth saying
        # out loud and neither is worth a traceback.
        #
        # A crash in the MediaPipe *graph* is a different animal: those
        # are abseil CHECK failures that abort the process, so there is no
        # exception to catch and nothing this can do. Hence the pins.
        logging.getLogger(__name__).warning(
            "no hand detector available: %s: %s. Hands will never be seen; "
            "everything else still runs. Install the extra with "
            "`pip install -e \".[hands]\"`.",
            type(exc).__name__, exc,
        )
        return NullHandDetector()


def build_game_app(cfg: Config, policy, *, dodging=True, opponent=None, render=False):
    """An app whose scene contains an opponent that fights back."""
    from tlod.arm.controller import ArmController, SafetyLimits
    from tlod.arm.model import HOME
    from tlod.arm.mock import MockArm
    from tlod.game.opponent import DodgingHand, DodgingHandScene
    from tlod.runtime.app import RobotApp
    from tlod.vision.camera import MockCamera
    from tlod.vision.hands import HandLocator
    from tlod.vision.scene import SceneHandDetector, SyntheticHandScene
    from tlod.vision.tracking import MultiTracker

    projector = build_projector(cfg)
    scene = (DodgingHandScene(projector, opponent or DodgingHand())
             if dodging else SyntheticHandScene(projector))

    limits = SafetyLimits(
        max_speed=cfg.safety.max_speed, strike_speed=cfg.safety.strike_speed,
        joint_margin=cfg.safety.joint_margin, table_z=cfg.safety.table_z,
        min_height=cfg.safety.min_height, max_radius=cfg.safety.max_radius,
        min_radius=cfg.safety.min_radius, max_height=cfg.safety.max_height,
        command_timeout=cfg.safety.command_timeout,
    )
    controller = ArmController(
        MockArm(q0=np.concatenate([HOME, [0.0]]), max_speed=cfg.arm.sim_max_speed,
                accel=cfg.arm.sim_accel, latency=cfg.arm.sim_latency),
        limits, cfg.runtime.control_hz,
    )
    if dodging:
        # The opponent must be able to see the arm coming.
        scene.tool_provider = lambda: controller.pose().xyz()

    app = RobotApp(
        camera=MockCamera(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                          scene=scene, render=render),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        controller=controller,
        policy=policy,
        tracker=MultiTracker(process_noise=cfg.vision.process_noise,
                             measurement_noise=cfg.vision.measurement_noise),
        control_hz=cfg.runtime.control_hz,
        perception_max_age=cfg.runtime.perception_max_age,
        prediction_horizon=cfg.runtime.prediction_horizon,
    )
    app.projector = projector
    app.scene = scene
    return app


def cmd_touch(args) -> int:
    """Detect the objects on the table and touch each one.

    The perception-to-control path on something that is not a hand, and
    the clearest way to see calibration error: a consistent offset in the
    same direction on every object means the extrinsics are wrong.

    Real hardware only. There is no synthetic version of this because
    there is nothing to synthesise: the scene renderer draws a hand and
    no objects, so a simulated run would search an empty table. It also
    could not answer the question if it did -- rendering objects through
    the same calibration that recovers them exercises the arithmetic in
    both directions and agrees with itself whatever the camera's real
    position.
    """
    from tlod.arm.controller import ArmController
    from tlod.game.touch import TouchObjectsPolicy
    from tlod.runtime.app import RobotApp
    from tlod.vision.hands import HandLocator, NullHandDetector
    from tlod.vision.objects import ColorBlobDetector
    from tlod.vision.tracking import MultiTracker

    cfg = Config.load(args.config)
    if not args.real:
        raise SystemExit(
            "  tlod touch drives the real arm against what the real camera\n"
            "  sees; pass --real to confirm you want it to move.")
    if not cfg.camera.extrinsics:
        raise SystemExit(
            "  no extrinsics configured, so the camera has no idea where the\n"
            "  arm is. Run `tlod calibrate extrinsics` first.")

    cfg = cfg.with_overrides(arm={"backend": "feetech"}, camera={"source": "opencv"})
    projector = build_projector(cfg)
    policy = TouchObjectsPolicy()
    camera = build_camera(cfg)
    controller = ArmController(build_arm(cfg), build_limits(cfg),
                               cfg.runtime.control_hz, governor=build_governor(cfg))

    print("  THE ARM WILL MOVE. Clear the workspace, keep hands away.")
    print("  Put red, green, blue or yellow objects on the table.")
    input("  press Enter when ready, Ctrl-C to abort... ")

    app = RobotApp(
        camera=camera,
        # No hand detector: this run is about objects, and loading
        # mediapipe to return nothing would be a heavy way to do it.
        detector=NullHandDetector(),
        locator=HandLocator(projector, depth_mode="size"),
        controller=controller,
        policy=policy,
        tracker=MultiTracker(),
        object_detector=ColorBlobDetector(projector, min_area_px=150),
        control_hz=cfg.runtime.control_hz,
    )
    app.projector = projector
    _run_for(app, args.duration, view=args.view, projector=projector,
             preview=getattr(args, 'preview', 0))

    print(f"\n  touched {len(policy.visited)}: {', '.join(policy.visited) or 'none'}")
    for det in app.objects:
        print(f"    {det.label:<6} at ({det.position[0]:+.3f}, "
              f"{det.position[1]:+.3f}, {det.position[2]:+.3f}) m")
    if policy.errors:
        print(f"  placement error: mean {np.mean(policy.errors)*1000:.1f} mm, "
              f"max {np.max(policy.errors)*1000:.1f} mm")
    return 0


def camera_overrides(camera: int | None) -> dict:
    """The `camera` override block for a CLI flag, honouring the config.

    `--camera` defaults to None rather than 0, so a config's own
    `camera.index` survives when the flag is not given. It used to default
    to 0 and be written in unconditionally, which made `camera.index` dead
    text in every config file for every command that takes the flag: you
    could edit it, and the run would still open index 0.

    That is unfixable from the config on a board where the camera is not
    index 0, and an Orange Pi 5 is exactly that board -- it enumerates its
    Rockchip codecs first, so the low indices are hardware that opens far
    enough to report "Not a video capture device" and the camera lands
    above /dev/video10. The failure then names the index you did not
    choose, which is the least useful thing it could say.
    """
    block: dict[str, object] = {"source": "opencv"}
    if camera is not None:
        block["index"] = camera
    return block


def play_config(cfg: Config, camera: int | None, real: bool) -> Config:
    """Config for a hand-slap run against a real hand.

    Split out from `cmd_play` for the same reason `hybrid_config` was:
    the one thing worth getting wrong here is whether `--real` reaches
    the arm backend, and that is checkable without hardware. An arm that
    was never asked to move is indistinguishable, from the outside, from
    one that was asked and failed.

    `camera` is None when the flag was not given, and then the config's
    own index stands. It used to default to 0 and be written in
    unconditionally, which meant `camera.index` in a config file was dead
    text for every command that took a `--camera` -- edit it, and the run
    still opened index 0. On a board where the camera is not index 0 that
    is unfixable from the config, and the failure names the index you did
    not choose.
    """
    return cfg.with_overrides(
        arm={"backend": "feetech" if real else "mock"},
        camera=camera_overrides(camera),
        vision={"detector": "mediapipe"},
    )


def _size_hold_to(limits, contact) -> None:
    """Hold at the bottom for at least as long as the sensor needs to read.

    The hold is not free -- it is the servos stalled at their torque
    limit, and sustained stall current is what an undersized supply has
    least of -- so the default is sized to the cheapest sensor and
    stretched only for one that needs more. Warning instead and carrying
    on would mean every round scoring as a dodge, which is a silent
    failure dressed as a game.
    """
    need = getattr(contact, "settle", 0.0) + 0.08
    if limits.press_hold < need:
        log.info("holding %.0f ms at the bottom instead of %.0f: %s needs %.0f ms "
                 "of pressing before its reading means anything",
                 need * 1e3, limits.press_hold * 1e3,
                 type(contact).__name__, contact.settle * 1e3)
        limits.press_hold = need


def cmd_play(args) -> int:
    """Play hand slap. The robot slaps; you dodge.

    Three tiers, and the contact sensor is what separates them. Whether
    the slap landed is the question a camera cannot answer -- at the
    moment of contact the arm is between an overhead camera and the
    contact point -- so each tier answers it with the instrument it
    actually has:

      tier A  `tlod play`              ground truth; the hand is simulated
      tier B  `tlod play --real-hand`  geometry; the arm is simulated, so
                                       the paddle passes through a hand
                                       rather than being stopped by one
                                       and there is nothing else to read
      tier C  `tlod play --real`       the encoders, and only the encoders

    Tier C takes no sensor argument. The strike commands a floor below
    the hand, so a paddle that stopped short of it was blocked and one
    that reached it was not, and both heights come off the encoders --
    not late, not filtered, not through the bus that the swing is already
    saturating. The torque-based alternatives are still in
    game/contact.py with their measurements; nothing constructs them.
    """
    # ServoLoadContactSensor, ServoPressContactSensor and
    # SerialContactSensor are deliberately absent: see game/contact.py for
    # what they measured and why the encoders won.
    from tlod.game.contact import (
        GeometricContactSensor,
        ProximityContactSensor,
        CollisionPlaneContactSensor,
    )
    from tlod.game.handslap import HandSlapGame, Personality
    from tlod.game.opponent import DodgingHand

    cfg = Config.load(args.config)
    contact = None
    if args.real:
        cfg = play_config(cfg, args.camera, real=True)
        if not cfg.camera.extrinsics:
            raise SystemExit(
                "  no extrinsics configured, so the camera has no idea where the\n"
                "  arm is and every strike would be aimed through a guessed pose.\n"
                "  That is tolerable for `tlod hybrid`, which only hovers. It is\n"
                "  not tolerable here: this command aims at a hand.\n"
                "  Run `tlod calibrate extrinsics` first.")

        # The controller has to exist before the game does. The contact
        # sensor reads the commanded floor through it, and HandSlapGame
        # takes its sensor at construction -- so this is app first, game
        # second, the opposite order from the branches below.
        app = build_app(cfg)
        # One object, built here and handed to the game below. It was
        # built twice -- once to check press_hold against the sensor,
        # once for HandSlapGame -- so the check was reading a throwaway
        # and could never have changed what ran.
        limits = build_strike_limits(cfg)

        # There is one sensor and no way to ask for another. The
        # alternatives all read torque, and what they measured is in
        # game/contact.py; the short version is that the arm braking its
        # own mass reaches the torque cap with an empty table under it,
        # and reading it held still instead costs 300 ms of stall current
        # per strike on a supply that cannot spare it. The encoders
        # already know where the paddle was sent and where it got to.
        from tlod.arm import model

        def _floor():
            """Where the paddle was sent, metres. No bus traffic.

            `commanded` is cached in the controller, and where the paddle
            actually got to arrives on the poll's `tool_xyz`, which the
            game has already read this tick.
            """
            return float(model.tool_pose(app.controller.commanded[:5]).z)

        contact = CollisionPlaneContactSensor(
            _floor,
            **({} if args.contact_threshold is None
               else {"margin": args.contact_threshold}),
            **({} if args.contact_band is None
               else {"band_fraction": args.contact_band}))
        source = (f"collision plane, {contact.band_fraction:.0%} of the way from the "
                  f"commanded floor up to the hand (at least "
                  f"{contact.margin * 1e3:.0f} mm), after "
                  f"{contact.settle * 1000:.0f} ms pressing")
        _size_hold_to(limits, contact)
        game = HandSlapGame(args.difficulty, limits=limits,
                            personality=Personality(enabled=not args.deadpan),
                            contact=contact, seed=args.seed)
        app.policy = game

        print(f"tier C: real hand, REAL ARM. difficulty={args.difficulty}")
        print(f"  contact from {source}")
        print("\n  THE ARM WILL MOVE, and it will strike at your hand.")
        print(f"  It drops at most {game.limits.max_drop*100:.0f} cm at a torque limit of "
              f"{game.limits.torque_limit}/1000, so it yields on contact.")
        print("  Put one hand flat on the table and keep everything else clear --")
        print("  face, other hand, cables. Ctrl-C stops it and parks the arm; with")
        print("  --view, `e` is e-stop and space pauses.")
        if not args.yes:
            input("  press Enter when ready, Ctrl-C to abort... ")
    elif args.real_hand:
        cfg = play_config(cfg, args.camera, real=False)
        game = HandSlapGame(args.difficulty, contact=ProximityContactSensor(), seed=args.seed,
                            personality=Personality(enabled=not args.deadpan))
        app = build_app(cfg)
        app.policy = game
        game.start(app)
        print(f"tier B: real hand, simulated arm. difficulty={args.difficulty}")
        print("put your hand in view and try not to get slapped. space pauses, e is e-stop.")
    else:
        game = HandSlapGame(args.difficulty, contact=GeometricContactSensor(), seed=args.seed,
                            personality=Personality(enabled=not args.deadpan))
        app = build_game_app(cfg, game,
                             opponent=DodgingHand(reaction_time=args.reaction, seed=args.seed),
                             render=args.view)
        game.truth_provider = lambda: app.scene.hand.position
        print(f"tier A: simulated opponent (reaction {args.reaction*1000:.0f} ms), "
              f"difficulty={args.difficulty}")

    try:
        _run_for(app, args.duration, view=args.view, projector=app.projector,
             preview=getattr(args, 'preview', 0),
             scoreboard=getattr(args, 'scoreboard', 0))
    finally:
        # Sensors may own a thread or a serial port (SerialContactSensor
        # does). ServoLoadContactSensor owns neither, but closing through
        # the interface is what keeps that an implementation detail
        # rather than something every caller has to know.
        game.contact.close()

    print(f"\n  final score: {game.score}  over {game.score.rounds} rounds")
    if game.score.rounds:
        print(f"  robot win rate: {game.score.robot/game.score.rounds:.0%}  "
              f"({game.strikes} strikes, {game.feints} feints)")
    if contact is not None:
        # The number to read after a hardware session: the largest
        # shortfall any round produced, in millimetres. Peak near zero
        # across a session with strikes in it means the paddle reached
        # its floor every time -- either nothing was ever under it, or
        # the floor is not actually below the hand.
        print("\n  contact: " + contact.peak_summary())
        if contact.read_failures:
            print(f"  {contact.read_failures} floor reads failed during strikes "
                  f"(scored as dodges rather than e-stopping mid-swing)")
        if game.strikes and contact.peak_rise < contact.margin:
            print("  no round ever stopped short. Either nothing was hit, or the")
            print("  floor is at or above the hand -- check the per-round line for")
            print("  a floor and a hand at the same height, and see press_depth and")
            print(f"  safety.min_height. If it is genuinely close, "
                  f"--contact-threshold {contact.peak_rise:.4f} is just under it.")
    return 0


def cmd_eval(args) -> int:
    """Sweep opponent reaction time and measure the robot's win rate.

    The design question of the project, answered numerically: does a
    short strike actually beat a human, and where is the crossover?
    """
    from tlod.game.contact import GeometricContactSensor
    from tlod.game.handslap import Difficulty, HandSlapGame, Personality
    from tlod.game.opponent import DodgingHand

    cfg = Config.load(args.config)
    reactions = [float(x) for x in args.reactions.split(",")]
    print(f"  {args.rounds} rounds per point, difficulty={args.difficulty}\n")
    print(f"  {'reaction':>9} {'rounds':>7} {'robot':>6} {'human':>6} {'win rate':>9}")
    results = []
    for reaction in reactions:
        difficulty = Difficulty.preset(args.difficulty)
        difficulty.mean_wait = args.mean_wait
        if args.no_feints:
            # Measures only the reflex half of the game. Useful for
            # isolating strike physics; misleading as a difficulty figure,
            # since feints are how the human scores.
            difficulty.feint_probability = 0.0
        # Deadpan on purpose: eval exists to measure win rates, and a
        # reaction after every round changes the tempo it measures them at
        # without changing the game it is measuring.
        game = HandSlapGame(difficulty, contact=GeometricContactSensor(), seed=args.seed,
                            personality=Personality(enabled=False))
        app = build_game_app(
            cfg, game, opponent=DodgingHand(reaction_time=reaction, seed=args.seed)
        )
        game.truth_provider = lambda a=app: a.scene.hand.position
        with app:
            deadline = time.perf_counter() + args.timeout
            while game.score.rounds < args.rounds and time.perf_counter() < deadline:
                time.sleep(0.05)
        rate = game.score.robot / game.score.rounds if game.score.rounds else float("nan")
        results.append((reaction, rate))
        print(f"  {reaction*1000:7.0f}ms {game.score.rounds:7d} {game.score.robot:6d} "
              f"{game.score.human:6d} {rate:8.0%}   "
              f"(strikes {game.strikes}, flinches {game.flinches}, holds {game.holds})")
    fair = [r for r, w in results if 0.35 <= w <= 0.65]
    if fair:
        print(f"\n  even match against a {min(fair)*1000:.0f}-{max(fair)*1000:.0f} ms reaction")
    return 0


def build_app(cfg: Config, render: bool = False):
    from tlod.arm.controller import ArmController
    from tlod.runtime.app import IdlePolicy, RobotApp, TrackHandPolicy
    from tlod.vision.hands import HandLocator
    from tlod.vision.tracking import MultiTracker

    projector = build_projector(cfg)
    scene = None
    if cfg.vision.detector == "scripted" or cfg.camera.source == "mock":
        from tlod.vision.scene import SyntheticHandScene
        scene = SyntheticHandScene(projector)
    camera = build_camera(cfg, scene=scene, render=render)
    detector = build_detector(cfg, scene)
    locator = HandLocator(
        projector,
        depth_mode=cfg.vision.depth_mode,
        hand_height=cfg.vision.hand_height,
        palm_width_m=cfg.vision.palm_width_m,
    )
    limits = build_limits(cfg)
    # The governor is None unless power.governor is set, so this costs the
    # simulated paths nothing. It matters for the paths that reach real
    # servos through here -- `hybrid --real` and `play --real` -- which
    # were the only entry points driving hardware without it.
    controller = ArmController(build_arm(cfg), limits, cfg.runtime.control_hz,
                               governor=build_governor(cfg))
    policies = {"idle": IdlePolicy, "track_hand": TrackHandPolicy}
    policy = policies.get(cfg.runtime.policy, IdlePolicy)()

    app = RobotApp(
        camera=camera,
        detector=detector,
        locator=locator,
        controller=controller,
        policy=policy,
        tracker=MultiTracker(
            process_noise=cfg.vision.process_noise,
            measurement_noise=cfg.vision.measurement_noise,
        ),
        control_hz=cfg.runtime.control_hz,
        perception_max_age=cfg.runtime.perception_max_age,
        prediction_horizon=cfg.runtime.prediction_horizon,
    )
    app.projector = projector
    return app


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _run_for(app, duration: float, view: bool = False, projector=None,
             preview: int = 0, scoreboard: int = 0) -> None:
    with app:
        server = _serve_overlay(app, projector, preview)
        board = _serve_scoreboard(app, scoreboard,
                                  server if preview and preview == scoreboard else None)
        # One object when both were asked for on the same port, two when
        # they were not, and stopping it twice would join a dead thread.
        servers = list({id(s): s for s in (server, board) if s is not None}.values())
        try:
            if view:
                # The window must own the main thread; on macOS a cv2
                # window created from a worker thread does nothing or
                # crashes. RobotApp already keeps its work on other
                # threads, so the main thread is free for exactly this.
                from tlod.viz.viewer import Viewer

                Viewer(app, projector).run(duration=duration)
            else:
                deadline = time.perf_counter() + duration
                while time.perf_counter() < deadline:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            for running in servers:
                running.stop()
        print(app.latency_report())
        pose = app.controller.pose()
        print(f"\n  final tool position: "
              f"({pose.x:+.3f}, {pose.y:+.3f}, {pose.z:+.3f}) m")


def _serve_overlay(app, projector, port: int):
    """Stream the annotated view over HTTP, for boards with no screen.

    The same frame `--view` draws -- arm skeleton, tracked hand, the
    policy's own HUD, the HIT/DODGED banner -- which is the difference
    between watching a robot move and watching it decide. Without it a
    feint and a strike are indistinguishable from across the table, and
    so are a tracked hand and a lost one.

    Rate-limited hard, because rendering reads the arm over the same
    serial bus the control loop uses, and during a strike that loop is
    already doing three transactions a tick.
    """
    if not port:
        return None
    import threading

    from tlod.vision.preview import PreviewServer
    from tlod.viz.viewer import Viewer

    server = PreviewServer(port=port, max_fps=8.0)
    server.start()
    viewer = Viewer(app, projector)
    stopping = threading.Event()

    def pump():
        while not stopping.wait(1.0 / 8.0):
            try:
                server.offer(viewer.render_once())
            except Exception:
                log.debug("overlay render failed", exc_info=True)

    threading.Thread(target=pump, name="overlay", daemon=True).start()
    original = server.stop

    def stop():
        stopping.set()
        original()

    server.stop = stop
    print(f"  watch it at http://<this board>:{port}/")
    return server


def _serve_scoreboard(app, port: int, shared=None):
    """Publish the running policy's score as a page, for the player.

    Separate from `--preview` because it answers a different question for
    a different person. The preview is for whoever is debugging: it shows
    frames, and it costs a JPEG encode and an arm read every time it does.
    This shows a number and a word, costs an attribute read, and is for
    whoever has their hand on the table and cannot also be reading a
    14 px HUD line in the corner of a throttled video stream.

    Both on the same port is allowed and shares the one socket, because
    refusing would be an arbitrary rule about ports; the scoreboard takes
    `/` in that case and the annotated stream stays at `/stream`.
    """
    if not port:
        return None
    from tlod.viz import scoreboard

    server = scoreboard.serve(app.policy, port, server=shared)
    if shared is not None:
        print(f"  scoreboard at http://<this board>:{port}/ "
              f"(the annotated view moved to /stream)")
    else:
        print(f"  scoreboard at http://<this board>:{port}/")
    return server


def cmd_sim(args) -> int:
    """Tier A: everything synthetic. Deterministic, no hardware, no camera."""
    cfg = Config.load(args.config).with_overrides(
        arm={"backend": "mock"},
        camera={"source": "mock"},
        vision={"detector": "scripted"},
        runtime={"policy": args.policy},
    )
    print(f"tier A simulation: synthetic camera, scripted hand, simulated arm "
          f"[policy={args.policy}]")
    app = build_app(cfg, render=args.view)
    _run_for(app, args.duration, view=args.view, projector=app.projector,
             preview=getattr(args, 'preview', 0))
    return 0


def hybrid_config(cfg: Config, camera: int | None, policy: str, real: bool) -> Config:
    """Config for a hybrid run: real camera and real hand either way.

    Split out from `cmd_hybrid` so the one thing worth getting wrong here
    is checkable without hardware -- that `--real` actually reaches the
    arm backend. Forcing "mock" unconditionally is what this command did
    for its whole life, and a silent revert to that is indistinguishable
    from the arm simply not moving.
    """
    return cfg.with_overrides(
        arm={"backend": "feetech" if real else "mock"},
        camera=camera_overrides(camera),
        vision={"detector": "mediapipe"},
        runtime={"policy": policy},
    )


def cmd_hybrid(args) -> int:
    """Your real webcam and real hand; simulated arm, or `--real` for both.

    With `--real` this is the whole robot on one machine: camera, hand
    tracking, IK and servos in a single process, with the arm hovering
    over the tracked hand. That is the two-board system minus the wire,
    and it is the useful shape when the camera and the servo adapter are
    plugged into the same board -- which is where they both are during
    calibration anyway.
    """
    cfg = hybrid_config(Config.load(args.config), args.camera, args.policy, args.real)
    where = "real arm" if args.real else "simulated arm"
    print(f"hybrid: real camera {cfg.camera.index}, real hand, {where} "
          f"[policy={args.policy}]")
    if args.real:
        # TrackHandPolicy hovers above the hand rather than reaching for
        # it, but it is following a person's hand in real time and the
        # only thing between it and them is the hover height.
        print("  THE ARM WILL MOVE, and it will follow your hand.")
        print("  Keep your hand flat on the table and the rest of you clear.")
        if not args.yes:
            input("  press Enter when ready, Ctrl-C to abort... ")
    print("wave your hand in front of the camera.")
    app = build_app(cfg)
    _run_for(app, args.duration, view=args.view, projector=app.projector,
             preview=getattr(args, 'preview', 0))
    return 0


def cmd_bench(args) -> int:
    from tlod.arm.model import HOME, ik_position

    if args.what in ("ik", "all"):
        q = HOME.copy()
        times, ok = [], 0
        for i in range(400):
            t = i * 0.01
            p = np.array([0.22 + 0.05 * np.sin(t * 3), 0.10 * np.sin(t * 2),
                          0.12 + 0.05 * np.cos(t * 2.5)])
            t0 = time.perf_counter()
            r = ik_position(p, q)
            times.append((time.perf_counter() - t0) * 1e3)
            if r.ok:
                q = r.q
                ok += 1
        print(f"  IK (tracking regime): {ok}/400 solved, "
              f"mean {np.mean(times):.3f} ms, p95 {np.percentile(times, 95):.3f} ms")

    if args.what in ("camera", "all"):
        cfg = Config.load(args.config)
        if cfg.camera.source == "mock" and not args.force:
            print("  camera: configured source is 'mock'; pass --force or set "
                  "camera.source=opencv to bench real hardware")
        else:
            from tlod.vision.camera import OpenCVCamera

            index = cfg.camera.index if args.camera is None else args.camera
            cam = OpenCVCamera(index=index, width=cfg.camera.width,
                               height=cfg.camera.height, fps=cfg.camera.fps)
            with cam:
                time.sleep(1.5)
                seen, t0 = set(), time.perf_counter()
                while time.perf_counter() - t0 < 3.0:
                    f = cam.read()
                    if f:
                        seen.add(f.index)
                    time.sleep(0.001)
                print(f"  camera: {cam.resolution[0]}x{cam.resolution[1]}, "
                      f"{cam.measured_fps:.1f} fps measured, {len(seen)} unique frames in 3 s")
                print("  NOTE: absolute shutter latency needs an external reference "
                      "(film a millisecond timer). camera.latency_offset is still an estimate.")

    if args.what in ("loop", "all"):
        cfg = Config.load(args.config).with_overrides(
            arm={"backend": "mock"}, camera={"source": "mock"},
            vision={"detector": "scripted"}, runtime={"policy": "track_hand"})
        app = build_app(cfg)
        with app:
            time.sleep(args.duration)
            print(app.latency_report())
            print(f"\n  measured shutter->command: {app.measured_latency*1e3:.1f} ms")
            print("  set runtime.prediction_horizon to about this, plus servo travel.")
    return 0


def cmd_record(args) -> int:
    """Capture a camera session to disk for repeatable offline tuning."""
    from tlod.vision.recording import Recorder

    cfg = Config.load(args.config).with_overrides(
        camera=camera_overrides(args.camera))
    camera = build_camera(cfg)
    print(f"  recording to {args.output} for {args.duration:.0f}s ...")
    with camera, Recorder(args.output) as rec:
        last = -1
        deadline = time.perf_counter() + args.duration
        try:
            while time.perf_counter() < deadline:
                frame = camera.read()
                if frame is not None and frame.index != last:
                    last = frame.index
                    rec.add(frame)
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n  stopped")
        print(f"  wrote {rec.count} frames")
    return 0


def cmd_replay(args) -> int:
    """Re-run a recording through the full pipeline, deterministically."""
    from tlod.vision.recording import ReplayCamera

    cfg = Config.load(args.config).with_overrides(
        vision={"detector": "mediapipe"}, arm={"backend": "mock"},
        runtime={"policy": args.policy})
    app = build_app(cfg)
    app.camera = ReplayCamera(args.path, realtime=not args.fast, loop=args.loop)
    print(f"  replaying {len(app.camera)} frames from {args.path}")
    _run_for(app, args.duration, view=args.view, projector=app.projector,
             preview=getattr(args, 'preview', 0))
    return 0


def cmd_move(args) -> int:
    """Move the tool to a position. The core capability, on its own.

    Works identically against the simulator and real hardware -- the
    backend interface is what makes `--real` a flag rather than a
    different program.
    """
    from tlod.arm.controller import ArmController
    from tlod.arm.model import HOME, tool_pose
    from tlod.types import Pose

    cfg = Config.load(args.config)
    if args.real:
        cfg = cfg.with_overrides(arm={"backend": "feetech"})

    limits = build_limits(cfg)
    controller = ArmController(build_arm(cfg), limits, cfg.runtime.control_hz,
                               governor=build_governor(cfg))
    controller.start()
    print(f"  backend {cfg.arm.backend}")
    start = controller.pose()
    print(f"  start   ({start.x:+.4f}, {start.y:+.4f}, {start.z:+.4f}) m")

    try:
        if args.home:
            controller.goto_joints(HOME, duration=args.duration)
            target = None
        elif args.joints is not None:
            q = np.array(args.joints, dtype=float)
            controller.goto_joints(q, duration=args.duration)
            target = tool_pose(q[:5]).xyz()
        else:
            target = np.array([args.x, args.y, args.z], dtype=float)
            safe, violations = limits.clamp_pose(Pose(*target))
            if violations:
                print(f"  clamped by safety: {', '.join(violations)} -> "
                      f"({safe.x:+.4f}, {safe.y:+.4f}, {safe.z:+.4f})")
            if not controller.goto_pose(Pose(args.x, args.y, args.z), duration=args.duration):
                print("  IK failed: that point is not reachable")
                controller.stop(park=False)
                return 1

        end = controller.pose()
        print(f"  end     ({end.x:+.4f}, {end.y:+.4f}, {end.z:+.4f}) m")
        if target is not None:
            error = np.linalg.norm(end.xyz() - np.asarray(target, float))
            print(f"  error   {error*1000:.2f} mm")
        print(f"  joints  {np.round(controller.commanded[:5], 4)}")
        if args.hold:
            print(f"  holding {args.hold:.1f}s ...")
            time.sleep(args.hold)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        controller.stop(park=args.park)
    return 0


def cmd_flourish(args) -> int:
    """Run flourishes one at a time, so each can be judged on its own.

    A review tool, not part of the game. The game picks by mood and buries
    the gesture in the pause after a round, which is right for playing and
    useless for deciding whether a spin reads as a spin.

    It reports asked against delivered, and the peak speed and acceleration
    each take actually reached, because those are the two ceilings that
    decide how big a gesture can be and the old table was sized against
    only one of them. `--sweep` walks the speed ceiling so the rig can say
    where its own limit is rather than being told.
    """
    from tlod.arm.controller import ArmController
    from tlod.arm.model import HOME
    from tlod.arm.primitives import FLOURISHES, MOODS, Flourish
    from tlod.types import JOINT_NAMES

    def cycles_of(mv, i):
        c = mv.cycles
        return float(c[i]) if isinstance(c, tuple) else float(c)

    def reach(mv, i):
        """The largest excursion this joint can actually make.

        `sin(pi*s) * sin(2*pi*c*s)` does not peak at 1 for every c -- at one
        cycle it peaks at 0.770 -- so the nominal amplitude is not a target
        the waveform ever hits, and measuring against it reports a clip that
        is really just the shape of the swing.
        """
        c = cycles_of(mv, i)
        u = np.linspace(0.0, 1.0, 20001)
        return abs(mv.amplitudes[i]) * float(
            np.abs(np.sin(np.pi * u) * np.sin(2.0 * np.pi * c * u)).max())

    def describe(mv):
        return ", ".join(f"{JOINT_NAMES[i]} {a:+.2f}x{cycles_of(mv, i):g}"
                         for i, a in enumerate(mv.amplitudes) if a)

    names = list(args.names) if args.names else list(FLOURISHES)
    unknown = [n for n in names if n not in FLOURISHES]
    if unknown:
        print(f"  no such flourish: {', '.join(unknown)}")
        print(f"  have: {', '.join(FLOURISHES)}")
        return 2

    if args.list:
        moods = {n: [m for m, ns in MOODS.items() if n in ns] for n in FLOURISHES}
        print(f"  {'name':8s} {'secs':>5s}  {'joints (amplitude x cycles)':52s} fires on")
        for n, mv in FLOURISHES.items():
            print(f"  {n:8s} {mv.duration or 0:5.2f}  {describe(mv):52s} "
                  f"{', '.join(moods[n]) or '-'}")
        return 0

    cfg = Config.load(args.config)
    if args.real:
        cfg = cfg.with_overrides(arm={"backend": "feetech"})
    if args.servo_accel is not None:
        cfg = cfg.with_overrides(arm={"servo_accel": args.servo_accel})
    controller = ArmController(build_arm(cfg), build_limits(cfg),
                               cfg.runtime.control_hz,
                               governor=build_governor(cfg))
    controller.start()
    speeds = ([float(s) for s in args.sweep.split(",")] if args.sweep
              else [args.speed])
    scales = [float(s) for s in str(args.scale).split(",")]
    print(f"  backend {cfg.arm.backend}   accel {args.accel:.0f} rad/s^2   "
          f"jerk {args.jerk:.0f}   servo ramp {cfg.arm.servo_accel:.0f} rad/s^2   "
          f"speed {', '.join(f'{s:.2f}' for s in speeds)} rad/s")
    print("  THE ARM WILL MOVE. Every flourish is joint space with no target and")
    print("  an envelope that is zero at both ends, so each returns to the pose")
    print("  it started from. Ctrl-C stops and parks.")
    if not args.yes:
        try:
            input("  press Enter when ready, Ctrl-C to abort... ")
        except (KeyboardInterrupt, EOFError):
            print()
            controller.stop(park=True)
            return 130

    def health():
        """Rail voltage and latched faults, if the backend has them."""
        try:
            d = controller.diagnostics()
        except Exception:
            return ""
        volts = d.get("voltage") or []
        faults = [f for f in (d.get("faults") or []) if f]
        out = f"  rail {min(volts):.1f}-{max(volts):.1f} V" if volts else ""
        return out + (f"  FAULTS {faults}" if faults else "")

    dt = 1.0 / max(cfg.runtime.control_hz, 1.0)
    rc = 0
    try:
        for name in names:
            rows = []
            for scale in scales:
                move = FLOURISHES[name]
                if scale != 1.0:
                    move = replace(move, amplitudes=tuple(
                        a * scale for a in move.amplitudes))
                amps = np.asarray(move.amplitudes, float)
                moved = [i for i, a in enumerate(amps) if a]

                for speed in speeds:
                    for take in range(1, args.repeat + 1):
                        controller.goto_joints(HOME, duration=args.settle)
                        start_q = controller.commanded.copy()
                        motion = Flourish(move, duration=args.duration, speed=speed,
                                          accel=args.accel, jerk=args.jerk)
                        print(f"\n  {name}  {describe(move)}  "
                              f"{motion.duration:.2f}s @ {speed:.2f} rad/s"
                              + (f"  take {take}" if args.repeat > 1 else ""))
                        motion.start(controller)
                        peak_cmd = np.zeros(len(amps))
                        peak_arm = np.zeros(len(amps))
                        peak_v = 0.0
                        prev, prev_t = start_q.copy(), time.perf_counter()
                        t0 = prev_t
                        while time.perf_counter() - t0 < motion.duration + 6.0:
                            done = motion.step(controller, dt)
                            now, q = time.perf_counter(), controller.commanded
                            peak_cmd = np.maximum(peak_cmd, np.abs(q - start_q))
                            if now > prev_t:
                                peak_v = max(peak_v, float(np.abs(q - prev).max() / (now - prev_t)))
                            prev, prev_t = q.copy(), now
                            try:
                                peak_arm = np.maximum(
                                    peak_arm, np.abs(controller.backend.read().q - start_q))
                            except Exception:
                                pass
                            if done:
                                break
                            time.sleep(dt)
                        elapsed = time.perf_counter() - t0
                        drift = float(np.abs(controller.commanded - start_q).max())
    
                        print(f"    {'joint':14s} {'nominal':>8s} {'possible':>9s} "
                              f"{'sent':>7s} {'arm':>7s} {'%':>5s} {'deg':>6s}")
                        for i in moved:
                            can = reach(move, i)
                            print(f"    {JOINT_NAMES[i]:14s} {abs(amps[i]):8.2f} {can:9.2f} "
                                  f"{peak_cmd[i]:7.2f} {peak_arm[i]:7.2f} "
                                  f"{peak_cmd[i] / can * 100.0:5.0f} "
                                  f"{np.degrees(peak_arm[i]):6.0f}")
                        clipped = [JOINT_NAMES[i] for i in moved
                                   if peak_cmd[i] < reach(move, i) - 0.05]
                        print(f"    {elapsed:.2f}s   peak {peak_v:.2f} rad/s of {speed:.2f}"
                              f"   back within {drift * 1e3:.1f} mrad" + health())
                        if clipped:
                            print(f"    clipped: {', '.join(clipped)}")
                        if drift > 2e-3:
                            print(f"    DID NOT RETURN: {drift:.4f} rad from its start")
                            rc = 1
                        rows.append((scale, speed,
                                     peak_cmd[moved].max() if moved else 0.0,
                                     peak_arm[moved].max() if moved else 0.0))

            if len(speeds) > 1 or len(scales) > 1:
                # `sent` is what the controller asked for, `arm` is what the
                # encoders came back with. They diverge where the servo stops
                # being able to follow, and that is the number worth having:
                # a joint whose arm column stops rising while sent keeps
                # rising has found its own limit, whatever the config says.
                print(f"\n  {name}: where the arm stops following the command")
                print(f"    {'scale':>6s} {'speed':>6s} {'sent':>7s} {'arm':>7s} "
                      f"{'tracked':>8s}")
                for scale, speed, sent, got in rows:
                    print(f"    {scale:6.2f} {speed:6.2f} {sent:7.2f} {got:7.2f} "
                          f"{(got / sent * 100.0 if sent else 0.0):7.0f}%")

            if name != names[-1] and not args.no_pause:
                try:
                    input("\n  Enter for the next one, Ctrl-C to stop... ")
                except EOFError:
                    break
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        controller.stop(park=True)
    return rc


def cmd_reach(args) -> int:
    """Probe the reachable workspace. Answers 'can it get there?'."""
    from tlod.arm.model import HOME, ik_position

    zs = [float(v) for v in args.heights.split(",")]
    print("  reachable radius by height (tool pointing freely)\n")
    for z in zs:
        radii = []
        for r in np.arange(0.05, 0.42, 0.005):
            if ik_position([r, 0.0, z], HOME).ok:
                radii.append(r)
        if radii:
            print(f"    z = {z*100:5.1f} cm   r = {min(radii)*100:5.1f} .. {max(radii)*100:5.1f} cm")
        else:
            print(f"    z = {z*100:5.1f} cm   unreachable")
    return 0


class _NullCamera:
    """Stands in for a camera on the control board, which has none."""

    def __init__(self, width=1280, height=720):
        self.width, self.height = width, height

    def start(self): pass

    def stop(self): pass

    def read(self): return None

    @property
    def resolution(self): return self.width, self.height


def cmd_vision_serve(args) -> int:
    """Vision board: detect and publish. Runs on the Orange Pi 5."""
    from tlod.net.publisher import VisionPublisher
    from tlod.vision.hands import HandLocator
    from tlod.vision.objects import ColorBlobDetector
    from tlod.vision.tracking import MultiTracker

    cfg = Config.load(args.config)
    if args.sim:
        # Synthetic camera AND synthetic detector. Running MediaPipe over
        # rendered frames would measure the model, not the link, and on
        # unrendered frames it finds nothing at all.
        cfg = cfg.with_overrides(camera={"source": "mock"}, vision={"detector": "scripted"})
    else:
        cfg = cfg.with_overrides(camera=camera_overrides(args.camera),
                                 vision={"detector": "mediapipe"})

    projector = build_projector(cfg)
    scene = None
    if cfg.vision.detector == "scripted" or cfg.camera.source == "mock":
        from tlod.vision.scene import SyntheticHandScene
        scene = SyntheticHandScene(projector)

    publisher = VisionPublisher(
        camera=build_camera(cfg, scene=scene),
        detector=build_detector(cfg, scene),
        locator=HandLocator(projector, depth_mode=cfg.vision.depth_mode,
                            hand_height=cfg.vision.hand_height,
                            palm_width_m=cfg.vision.palm_width_m),
        tracker=MultiTracker(process_noise=cfg.vision.process_noise,
                             measurement_noise=cfg.vision.measurement_noise),
        object_detector=ColorBlobDetector(projector) if args.objects else None,
        targets=[(host, args.port) for host in args.to.split(",")],
        clock_port=args.clock_port,
        serial_port=args.serial_port or None,
        serial_baudrate=args.baudrate,
    )
    preview = None
    if args.preview:
        from tlod.vision.preview import PreviewServer
        preview = PreviewServer(port=args.preview)
        preview.start()
        publisher.preview = preview
        print(f"  preview: open http://<this board>:{args.preview}/ from any browser")

    print(f"  vision board: publishing to {args.to}:{args.port}, "
          f"clock on :{args.clock_port}")
    if args.serial_port:
        print(f"  also publishing over UART: {args.serial_port} @ {args.baudrate}")
    if not cfg.camera.extrinsics:
        print("  WARNING: no extrinsics configured. Positions will be in a guessed")
        print("  camera frame and the arm will reach to the wrong place.")
    with publisher:
        try:
            deadline = time.perf_counter() + args.duration if args.duration else None
            while deadline is None or time.perf_counter() < deadline:
                time.sleep(2.0)
                print(f"\r  frames {publisher.frames}  published {publisher.sent}",
                      end="", flush=True)
        except KeyboardInterrupt:
            pass
    if preview is not None:
        preview.stop()
    print("\n" + publisher.report())
    return 0


def cmd_control(args) -> int:
    """Control board: consume detections, run the loop. Runs on the Pi."""
    from tlod.arm.controller import ArmController
    from tlod.net.subscriber import VisionSubscriber
    from tlod.runtime.app import IdlePolicy, RobotApp, TrackHandPolicy
    from tlod.vision.tracking import MultiTracker

    cfg = Config.load(args.config)
    if args.real:
        cfg = cfg.with_overrides(arm={"backend": "feetech"})

    subscriber = VisionSubscriber(
        host=args.vision_host, port=args.port, clock_port=args.clock_port,
        require_clock=not args.no_clock,
        serial_port=args.serial_port or None,
        serial_baudrate=args.baudrate,
    )
    if args.serial_port:
        print(f"  control board: listening on {args.serial_port} @ {args.baudrate} (UART)")
    else:
        print(f"  control board: listening on :{args.port}, clock from "
              f"{args.vision_host or '(none)'}")
    subscriber.start()
    if subscriber.clock:
        print(f"  clock offset {subscriber.clock.offset*1e3:+.2f} ms "
              f"(+/-{subscriber.clock.uncertainty*1e3:.2f} ms)")

    policies = {"idle": IdlePolicy, "track_hand": TrackHandPolicy}
    controller = ArmController(build_arm(cfg), build_limits(cfg), cfg.runtime.control_hz,
                               governor=build_governor(cfg))
    health = None
    if controller.governor is not None:
        from tlod.arm.power import HealthMonitor

        health = HealthMonitor(controller, controller.governor)
        print(f"  power governor on: {cfg.power.supply_current:.1f} A supply, "
              f"{controller.governor.model.budget.limit:.2f} A budget")
    app = RobotApp(
        camera=_NullCamera(),
        detector=None,
        locator=None,
        controller=controller,
        policy=policies.get(args.policy, IdlePolicy)(),
        tracker=MultiTracker(),
        control_hz=cfg.runtime.control_hz,
        perception_max_age=cfg.runtime.perception_max_age,
        prediction_horizon=cfg.runtime.prediction_horizon,
        perception_source=subscriber.perception,
    )

    telemetry = None
    if args.telemetry_to:
        from tlod.net.telemetry import ArmTelemetryPublisher

        telemetry = ArmTelemetryPublisher(
            controller=controller,
            targets=[(host, args.telemetry_port) for host in args.telemetry_to.split(",")],
            perception=subscriber.perception,
            clock_port=args.telemetry_clock_port,
        )
        telemetry.start()
        print(f"  arm telemetry -> {args.telemetry_to}:{args.telemetry_port}, "
              f"clock on :{args.telemetry_clock_port} "
              f"(watch with `tlod arm-viewer --control-host <this board's IP>`)")

    try:
        with app:
            if health is not None:
                health.start()
            deadline = time.perf_counter() + args.duration
            while time.perf_counter() < deadline:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        if health is not None:
            health.stop()
        if telemetry is not None:
            telemetry.stop()
        subscriber.stop()
    print(app.latency_report())
    if health is not None:
        print(f"\n  power: peak {health.peak_current:.2f} A, "
              f"derate ended at {controller.derate:.2f}, "
              f"{controller.governor.sag_events} rail sags"
              + (f", faults {sorted(health.faults)}" if health.faults else ""))
    print("\n  network")
    print(subscriber.report())
    if telemetry is not None:
        print(f"\n  telemetry sent {telemetry.sent}")
    return 0


def cmd_arm_viewer(args) -> int:
    """Watch a control board's arm telemetry. Runs on a laptop, not a board.

    For the arm-less HIL test: the control board (`tlod control`, no
    `--real`) drives `MockArm` and has no screen either way. Point
    `--telemetry-to` at this machine when starting `tlod control`, then
    run this here to see the skeleton move instead of only reading the
    text report at the end.

    Pass `--control-host` (the control board's own IP) to also measure
    the clock offset to it -- without that, the HUD can only report how
    long a packet sat in this laptop's socket after arriving, not the
    true shutter-to-screen latency across the whole stack.
    """
    from tlod.net.telemetry import ArmTelemetrySubscriber
    from tlod.viz.remote_viewer import RemoteArmViewer

    subscriber = ArmTelemetrySubscriber(
        port=args.port, host=args.control_host, clock_port=args.clock_port,
    )
    subscriber.start()
    print(f"  listening for arm telemetry on :{args.port} ...")
    if subscriber.clock:
        print(f"  clock offset {subscriber.clock.offset * 1e3:+.2f} ms "
              f"(+/-{subscriber.clock.uncertainty * 1e3:.2f} ms)")
    elif args.control_host:
        print("  WARNING: no clock response -- latency numbers will be reception "
              "age only, not true end-to-end")
    try:
        RemoteArmViewer(subscriber).run(duration=args.duration or None)
    finally:
        subscriber.stop()
    print(f"  packets: {subscriber.received} received, "
          f"{subscriber.dropped_bad} malformed, {subscriber.dropped_stale} reordered")
    return 0


def cmd_vision_check(args) -> int:
    """Verify the vision stack numerically. For boards with no screen.

    Two kinds of check, and the difference matters: precision (stable and
    self-consistent) needs only a camera, while accuracy (actually right)
    needs ground truth, which only the arm can provide.
    """
    from tlod.vision.calibrate_flow import MARKER_BANDS, calibration_poses, find_marker
    from tlod.vision.check import Thresholds, check_against_arm, check_precision
    from tlod.vision.hands import HandLocator

    cfg = Config.load(args.config)
    if args.sim:
        # Synthetic camera and synthetic detector together. MediaPipe on
        # unrendered mock frames finds nothing, which looks like a broken
        # pipeline rather than a misconfigured test.
        cfg = cfg.with_overrides(camera={"source": "mock"}, vision={"detector": "scripted"})
    else:
        cfg = cfg.with_overrides(camera=camera_overrides(args.camera),
                                 vision={"detector": "mediapipe"})

    projector = build_projector(cfg)
    scene = None
    if cfg.vision.detector == "scripted" or cfg.camera.source == "mock":
        from tlod.vision.scene import SyntheticHandScene
        scene = SyntheticHandScene(projector)
    camera = build_camera(cfg, scene=scene)
    locator = HandLocator(projector, depth_mode=cfg.vision.depth_mode,
                          hand_height=cfg.vision.hand_height,
                          palm_width_m=cfg.vision.palm_width_m)
    thresholds = Thresholds()

    if not cfg.camera.extrinsics:
        print("  WARNING: no extrinsics configured. Precision checks are still")
        print("  meaningful; accuracy is not, because positions are in a guessed frame.\n")

    print(f"  precision check: {args.duration:.0f}s. Put a hand in view and move it")
    print("  slowly across the frame, keeping it about the same distance away.\n")
    with camera:
        time.sleep(1.0)
        report = check_precision(
            camera, build_detector(cfg, scene), locator,
            duration=args.duration, thresholds=thresholds, save_dir=args.save_frames,
            fixed_distance=args.fixed_distance,
            on_progress=lambda r: print(
                f"\r  {r.frames} frames, {r.detections} detections", end="", flush=True),
        )
        print()

        if args.with_arm:
            from tlod.arm.controller import ArmController, SafetyLimits

            print("\n  accuracy check: THE ARM WILL MOVE. Clear the workspace.")
            print(f"  A {args.marker} marker must be on the gripper.")
            if not args.yes:
                input("  press Enter when ready, Ctrl-C to abort... ")
            controller = ArmController(build_arm(cfg), SafetyLimits(), cfg.runtime.control_hz)
            controller.start()
            try:
                def locate(image):
                    uv = find_marker(image, MARKER_BANDS[args.marker])
                    if uv is None:
                        return None
                    # Resolve the marker the same way a hand would be, so
                    # the check exercises the real path rather than a
                    # shortcut around it: intersect its ray with a plane
                    # at the tool's own height, and fall back to the
                    # table when the arm is below it and the ray misses.
                    at_tool = locator.projector.pixel_to_plane(
                        uv[0], uv[1], controller.pose().z)
                    if at_tool is not None:
                        return at_tool
                    return projector.pixel_to_plane(uv[0], uv[1], 0.0)
                report = check_against_arm(
                    camera, controller, locate, calibration_poses(args.poses),
                    report, thresholds,
                )
            finally:
                controller.stop(park=True)
        else:
            report.notes.append(
                "accuracy NOT checked -- pass --with-arm to score against kinematics"
            )

    print(report.text())
    if args.json:
        report.save(args.json)
        print(f"\n  wrote {args.json}")
    return 0 if report.passed else 2


def cmd_probe(args) -> int:
    """Read the arm without commanding it. The safest first hardware test.

    Torque is disabled, so the arm is limp and you move it by hand while
    watching the numbers. Nothing is ever commanded, so nothing can lurch.
    This is what you run before `first-light`, and it answers the
    questions that otherwise only surface once something is moving under
    power:

      * does the bus work at all, and do all six motors answer
      * does each joint read the direction you expect
      * what range does each joint actually cover
      * are any of them already hot or under load

    Support the arm before enabling this -- with torque off it will drop
    under its own weight.
    """
    from tlod.types import JOINT_NAMES

    cfg = Config.load(args.config)
    if args.real:
        cfg = cfg.with_overrides(arm={"backend": "feetech"})
    if cfg.arm.backend == "mock" and not args.force:
        print("  arm.backend is 'mock'. Pass --real for hardware, or --force to rehearse.")
        return 1

    backend = build_arm(cfg)
    backend.connect()
    if not args.keep_torque:
        backend.set_torque(False)
        print("  TORQUE OFF - the arm is limp. Support it before letting go.\n")
    else:
        print("  torque left ON - the arm will hold position.\n")

    lo = np.full(6, np.inf)
    hi = np.full(6, -np.inf)
    seen = np.zeros(6, dtype=bool)
    start = None
    period = 1.0 / max(args.rate, 0.5)

    print("  move each joint by hand through its range. Ctrl-C to finish.\n")
    try:
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            state = backend.read()
            q = state.q
            if start is None:
                start = q.copy()
            lo = np.minimum(lo, q)
            hi = np.maximum(hi, q)
            seen |= np.abs(q - start) > 0.02

            cells = " ".join(f"{n[:5]}:{v:+.3f}" for n, v in zip(JOINT_NAMES, q, strict=True))
            print(f"\r  {cells}", end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n")
        diagnostics = backend.diagnostics()
        backend.disconnect()

    print(f"  {'joint':<15} {'min':>8} {'max':>8} {'range':>8}   moved?")
    for i, name in enumerate(JOINT_NAMES):
        span = hi[i] - lo[i]
        mark = "yes" if seen[i] else "NOT SEEN"
        print(f"  {name:<15} {lo[i]:+8.3f} {hi[i]:+8.3f} {span:8.3f}   {mark}")

    if not seen.all():
        missing = [n for n, s in zip(JOINT_NAMES, seen, strict=True) if not s]
        print(f"\n  These never moved: {', '.join(missing)}")
        print("  Either you did not move them, or that motor is not answering.")
        print("  Check its 3-pin cable and that its id was set.")

    temps = diagnostics.get("temperature_c")
    volts = diagnostics.get("voltage_v")
    if temps:
        print(f"\n  temperature  {temps} C")
        if max(temps) > 55:
            print("  WARNING: a servo is hot. Let it cool before running anything.")
    if volts:
        print(f"  voltage      {volts} V")
        if min(volts) < 10.5:
            print("  WARNING: low supply voltage. Check the 12 V adapter.")
    return 0


def cmd_calibrate(args) -> int:
    """Measure the lens, then measure where the camera is.

    Extrinsics use the arm as the calibration target: it drives to a
    spread of poses and finds a marker on the gripper in each frame, with
    forward kinematics supplying the 3D coordinates. That puts the result
    in exactly the frame the controller commands in.
    """
    import cv2

    from tlod.vision.calibrate_flow import (
        MARKER_BANDS,
        calibration_poses,
        find_marker,
        run_extrinsics,
        run_intrinsics,
    )

    if set(MARKER_BANDS) != set(MARKER_COLOURS):
        raise SystemExit("marker colour lists have drifted apart; fix cli.MARKER_COLOURS")
    from tlod.vision.calibration import Intrinsics

    cfg = Config.load(args.config)
    # Resolved per subcommand, from the config, rather than one default for
    # both. It was a single `-o` defaulting to calib/intrinsics.npz for the
    # whole `calibrate` command -- correct for intrinsics and destructive
    # for extrinsics, which happily wrote its own solve over a lens
    # calibration that takes a chessboard and twenty minutes to reshoot.
    # There is no warning at the point of loss: the run prints a camera
    # position and an RMS and looks like it worked.
    default_out = (cfg.camera.intrinsics if args.what == "intrinsics"
                   else cfg.camera.extrinsics)
    out = Path(args.output or default_out or f"calib/{args.what}.npz")

    # And a hard stop, because a default is only the usual way to get this
    # wrong. These two files are not interchangeable and neither solve can
    # detect that it has been handed the other's data.
    other = (cfg.camera.extrinsics if args.what == "intrinsics"
             else cfg.camera.intrinsics)
    if other and out.resolve() == Path(other).resolve():
        raise SystemExit(
            f"refusing to write {args.what} to {out}, which is the "
            f"{'extrinsics' if args.what == 'intrinsics' else 'intrinsics'} "
            f"file in this config. Pass -o with a different path.")

    if args.what == "intrinsics":
        cfg = cfg.with_overrides(camera=camera_overrides(args.camera))
        camera = build_camera(cfg)
        print(f"  hold a {args.pattern} chessboard (inner corners) with "
              f"{args.square*1000:.0f} mm squares in view.")
        print("  move it around: corners, edges, near, far, tilted. Auto-captures.")
        preview = None
        if args.preview:
            from tlod.vision.preview import PreviewServer

            preview = PreviewServer(port=args.preview)
            preview.start()
            print(f"  watch it at http://<this board>:{args.preview}/  "
                  "-- the label says why a view is being skipped")

        def show(image, corners, status, kept, total):
            """Annotate and offer the frame. Cheap when nobody is watching."""
            if preview is None:
                return
            shown = image.copy()
            if corners is not None:
                cv2.drawChessboardCorners(shown, _pattern(args.pattern), corners, True)
            green = status == "CAPTURED"
            cv2.putText(shown, f"{kept}/{total}  {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 0) if green else (0, 0, 255), 2)
            preview.offer(shown)

        with camera:
            time.sleep(1.0)
            try:
                intr = run_intrinsics(
                    camera, pattern=_pattern(args.pattern), square=args.square,
                    views=args.views, timeout=args.timeout, fisheye=args.fisheye,
                    hfov_deg=cfg.camera.hfov_deg, on_frame=show,
                    on_progress=lambda n, total, *_: print(f"    view {n}/{total}",
                                                           flush=True),
                )
            finally:
                if preview is not None:
                    preview.stop()
        intr.save(out)
        import math

        w, _ = intr.resolution
        hfov = math.degrees(2 * math.atan(w / (2 * intr.K[0, 0])))
        print(f"\n  {intr.model} model, reprojection RMS {intr.rms:.3f} px  ->  {out}")
        print(f"  measured horizontal field of view {hfov:.0f} deg "
              f"(config says {cfg.camera.hfov_deg:.0f})")
        if intr.rms > 1.0:
            print("  WARNING: above 1 px is poor. Reshoot with more varied views,")
            print("  better light, and the board fully flat.")
        return 0

    # extrinsics
    #
    # Falls back to the config, which already names the intrinsics file
    # the rest of the pipeline loads. Demanding the flag anyway meant the
    # one command you reach for when the camera has moved refused to run
    # until you went and looked up a path the config was holding.
    intrinsics_path = args.intrinsics or cfg.camera.intrinsics
    if not intrinsics_path:
        raise SystemExit(
            "extrinsics needs intrinsics: pass --intrinsics, or set "
            "camera.intrinsics in the config. Measure them first with "
            "`tlod calibrate intrinsics`.")
    if not Path(intrinsics_path).exists():
        raise SystemExit(
            f"no intrinsics at {intrinsics_path}"
            + ("" if args.intrinsics else " (from camera.intrinsics in the config)")
            + ". Measure them first with `tlod calibrate intrinsics`.")
    if not args.intrinsics:
        print(f"  intrinsics from the config: {intrinsics_path}")
    intr = Intrinsics.load(intrinsics_path)

    from tlod.arm.controller import ArmController, SafetyLimits

    if args.sim:
        # Rehearsal: a synthetic camera that renders a marker at the true
        # tool position. Proves the whole procedure end to end -- motion,
        # detection, solve, residuals -- before it drives real hardware.
        from tlod.vision.calibration import synthetic_projector

        truth = synthetic_projector((cfg.camera.width, cfg.camera.height),
                                    cfg.camera.position, cfg.camera.look_at)
        # Force the simulated backend. The claim below is that nothing
        # moves, and a config with arm.backend already set to feetech --
        # which is exactly the config someone runs this from -- would
        # otherwise drive the real arm through all twelve poses while
        # printing that it is not.
        cfg = cfg.with_overrides(arm={"backend": "mock"})
        controller = ArmController(build_arm(cfg), SafetyLimits(), cfg.runtime.control_hz)
        camera = _MarkerCamera(truth, controller, cfg.camera.width, cfg.camera.height)
        print("  SIMULATED rehearsal: no hardware is moving.")
    else:
        cfg = cfg.with_overrides(camera=camera_overrides(args.camera),
                                 arm={"backend": "feetech"})
        camera = build_camera(cfg)
        controller = ArmController(build_arm(cfg), SafetyLimits(), cfg.runtime.control_hz)
        print("  THE ARM WILL MOVE. Clear the workspace, keep hands away.")
        print(f"  Attach a {args.marker} marker to the gripper, visible from the camera.")
        input("  press Enter when ready, Ctrl-C to abort... ")

    controller.start()
    try:
        with camera:
            time.sleep(1.0)
            heights = tuple(float(v) for v in args.heights.split(","))
            extr, residuals, marker_offset, naive_rms = run_extrinsics(
                camera, controller, intr,
                poses=calibration_poses(heights=heights),
                gripper=args.gripper,
                locate=functools.partial(find_marker,
                                         hsv_band=MARKER_BANDS[args.marker]),
                on_progress=lambda i, n, *_: print(f"    pose {i}/{n}", flush=True),
            )
    finally:
        controller.stop(park=True)

    extr.save(out)
    residuals = np.array(residuals)
    print(f"\n  camera at ({extr.t[0]:+.3f}, {extr.t[1]:+.3f}, {extr.t[2]:+.3f}) m in base frame")
    print(f"  reprojection: RMS {extr.rms:.2f} px, worst point {residuals.max():.2f} px")
    if np.linalg.norm(marker_offset) > 1e-4:
        print(f"  marker sits {np.linalg.norm(marker_offset) * 1e3:.0f} mm off the tool "
              f"point ({', '.join(f'{v * 1e3:+.0f}' for v in marker_offset)} mm in the "
              f"tool frame),")
        print(f"  which is solved for rather than assumed away -- RMS would be "
              f"{naive_rms:.2f} px without it.")
        if np.linalg.norm(marker_offset) > 0.05:
            print("  NOTE: that is a long way for tape on a gripper. Check the jaw is")
            print("  actually closed (--gripper 1 if 0 opens it) and that nothing else")
            print("  in frame is that colour.")
    print(f"  -> {out}")
    if residuals.max() > 3 * max(extr.rms, 0.5):
        print("  NOTE: one point is far worse than the rest -- likely a mislocated")
        print("  marker rather than a bad calibration. Rerun; it should settle.")
    print("\n  verify with:  tlod touch --real     (the tool must land on the objects)")
    print("  or numerically:  tlod vision-check --with-arm --marker <colour>")
    return 0


def _pattern(text: str) -> tuple[int, int]:
    cols, rows = text.lower().split("x")
    return int(cols), int(rows)


class _MarkerCamera:
    """Synthetic camera drawing a marker at the true tool position."""

    def __init__(self, projector, controller, width, height):
        self.projector = projector
        self.controller = controller
        self.width, self.height = width, height
        self._n = 0

    def start(self): pass

    def stop(self): pass

    def __enter__(self): return self

    def __exit__(self, *exc): pass

    @property
    def resolution(self): return self.width, self.height

    def read(self):
        import cv2
        from tlod.types import Frame

        img = np.full((self.height, self.width, 3), 30, np.uint8)
        tip = model_fk_tip(self.controller)
        uv = self.projector.project(tip)
        if uv is not None:
            cv2.circle(img, (int(uv[0]), int(uv[1])), 14, (70, 190, 90), -1)
        self._n += 1
        return Frame(image=img, stamp=time.perf_counter(), index=self._n)


def model_fk_tip(controller):
    from tlod.arm import model

    return model.fk(controller.state().q[:5])[:3, 3]


def cmd_cameras(args) -> int:
    """Which index is the camera, by name where the kernel will say.

    Printing bare indices was not enough on this board. An Orange Pi 5
    enumerates its Rockchip codecs first -- rkvdec, rkvenc, rga -- so the
    low indices belong to hardware that is not a camera, and index 0
    opens far enough to fail with "Not a video capture device". The name
    is what tells them apart.
    """
    from tlod.vision.camera import capture_nodes, list_cameras

    nodes = capture_nodes()
    if nodes:
        print("  v4l2 capture nodes:")
        for i, name in sorted(nodes.items()):
            print(f"    {i:>3}  {name}")
    found = list_cameras()
    print(f"  indices that actually open: {found or 'none'}")
    if not found:
        print("  (check `lsusb` for the camera; a hub that dropped it looks"
              " exactly like this)")
    print("  Indices move across reboots and replugs -- never trust last week's.")
    return 0


def cmd_ports(args) -> int:
    from tlod.arm.feetech import find_ports

    found = find_ports()
    print(f"  serial ports: {found or 'none found'}")
    if not found:
        print("  (plug in the controller board, and check it has power)")
    return 0


def cmd_first_light(args) -> int:
    """Move one joint at a time, slowly, to verify signs and limits.

    The first thing to run on a newly assembled arm, and the only safe way
    to discover that a direction convention is inverted.
    """
    from tlod.types import JOINT_NAMES

    cfg = Config.load(args.config)
    if cfg.arm.backend == "mock" and not args.force:
        print("  arm.backend is 'mock'. This command is for real hardware; "
              "pass --force to rehearse it in simulation.")
        return 1

    from tlod.arm.controller import ArmController, SafetyLimits

    controller = ArmController(build_arm(cfg), SafetyLimits(max_speed=0.4), cfg.runtime.control_hz)
    controller.start()
    print("  moving each joint +/-0.2 rad, slowly. Ctrl-C stops.")
    try:
        for i, name in enumerate(JOINT_NAMES):
            base = controller.commanded.copy()
            print(f"  [{i+1}/6] {name} ...", flush=True)
            for delta in (args.amplitude, -args.amplitude, 0.0):
                q = base.copy()
                q[i] = base[i] + delta
                controller.goto_joints(q, duration=1.5)
            measured = controller.state().q[i]
            print(f"        returned to {measured:+.4f} rad (commanded {base[i]:+.4f})")
    except KeyboardInterrupt:
        print("\n  stopped by user")
    finally:
        controller.stop(park=True)
    return 0


def cmd_power(args) -> int:
    """Measure what the arm actually draws, and whether the rail holds up.

    The point is to settle by measurement a question that is otherwise
    settled by guessing. "Fine on one joint, jitters on several" has two
    plausible causes that look identical from the outside -- a power supply
    that cannot hold the rail, or a control bug -- and they want opposite
    fixes. The servos report their own current and their own terminal
    voltage, so there is no need to choose between them on a hunch.

    It runs the same displacement twice, once one joint at a time and once
    with everything moving together. Same distance, same limits: if the
    second one sags and the first does not, that is the supply.
    """
    import numpy as np

    from tlod.arm.controller import ArmController
    from tlod.arm.model import HOME
    from tlod.arm.power import PowerBudget, PowerModel

    cfg = Config.load(args.config)
    if cfg.arm.backend == "mock":
        print("  arm.backend is 'mock'. This measures real hardware; "
              "point -c at your real-arm config.")
        return 1

    model_ = PowerModel(budget=PowerBudget(
        supply_current=cfg.power.supply_current, headroom=cfg.power.headroom,
        min_voltage=cfg.power.min_voltage))
    controller = ArmController(build_arm(cfg), build_limits(cfg), cfg.runtime.control_hz)
    controller.start()

    samples: list[dict] = []

    def sample(label: str) -> dict:
        d = controller.diagnostics()
        row = {
            "phase": label,
            "current_a": float(d["total_current_a"]),
            "min_voltage_v": float(d["min_voltage_v"]),
            "faults": [f for per in d["faults"] for f in per],
            "predicted_a": model_.total_current(controller.commanded[:5]),
        }
        samples.append(row)
        return row

    def run(label: str, target: np.ndarray, duration: float) -> None:
        worst = {"current_a": 0.0, "min_voltage_v": 99.0, "faults": []}
        t0 = time.perf_counter()
        controller.goto_joints(target, duration=duration)
        while time.perf_counter() - t0 < duration + 0.3:
            row = sample(label)
            worst["current_a"] = max(worst["current_a"], row["current_a"])
            worst["min_voltage_v"] = min(worst["min_voltage_v"], row["min_voltage_v"])
            worst["faults"] += row["faults"]
            break
        print(f"  {label:<28} peak {worst['current_a']:5.2f} A   "
              f"min rail {worst['min_voltage_v']:5.2f} V"
              + (f"   FAULTS {sorted(set(worst['faults']))}" if worst["faults"] else ""))

    print(f"\n  supply configured as {cfg.power.supply_current:.1f} A, "
          f"planning budget {model_.budget.limit:.2f} A")
    try:
        controller.goto_joints(HOME, duration=2.0)
        idle = sample("idle at home")
        print(f"\n  {'idle, holding home':<28} {idle['current_a']:5.2f} A   "
              f"rail {idle['min_voltage_v']:5.2f} V"
              f"   (model predicts {idle['predicted_a']:.2f} A)")
        if not model_.supports_holding(HOME):
            print("  !! the model says this supply cannot even hold the arm up. "
                  "No amount of motion profiling fixes that.")

        amp = args.amplitude
        print()
        for i, name in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll")):
            q = np.array(HOME, dtype=float)
            q[i] += amp
            run(f"{name} alone", q, args.duration)
            controller.goto_joints(HOME, duration=args.duration)

        print()
        together = np.array(HOME, dtype=float) + amp
        run("all five together", together, args.duration)
        controller.goto_joints(HOME, duration=args.duration)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        controller.stop(park=True)

    single = max((s["current_a"] for s in samples if "alone" in s["phase"]), default=0.0)
    multi = max((s["current_a"] for s in samples if "together" in s["phase"]), default=0.0)
    sag = min((s["min_voltage_v"] for s in samples), default=0.0)
    faults = sorted({f for s in samples for f in s["faults"]})

    print("\n  verdict")
    print(f"    worst single-joint draw   {single:5.2f} A")
    print(f"    worst multi-joint draw    {multi:5.2f} A")
    print(f"    lowest rail voltage seen  {sag:5.2f} V")
    if faults:
        print(f"    servo faults latched      {faults}")

    over = multi > model_.budget.limit
    sagging = sag < cfg.power.min_voltage
    if sagging or "voltage" in faults:
        print("\n    The rail is sagging under load. This is the supply, not the code.")
        print("    Fit a bigger one (12 V 5 A) and a bulk capacitor; see docs/power.md.")
        print("    Meanwhile set power.governor: true to keep the arm inside what you have.")
    elif over:
        print("\n    Draw exceeds the planning budget but the rail is holding. "
              "The supply is coping;")
        print("    raise power.supply_current if that rating is honest, or leave the "
              "governor on.")
    else:
        print("\n    Draw and rail voltage are both within budget. If the arm still "
              "jitters, it is")
        print("    not power -- check `tlod sim` overruns and the serial bus.")

    if args.json:
        import json
        Path(args.json).write_text(json.dumps(samples, indent=2))
        print(f"\n  wrote {args.json}")
    return 1 if (sagging or faults) else 0


def cmd_config(args) -> int:
    cfg = Config.load(args.config)
    cfg.save(args.output)
    print(f"  wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tlod", description="Third Leg of Doom robot")
    p.add_argument("-c", "--config", default=None, help="YAML config path")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sim", help="tier A: fully synthetic run")
    s.add_argument("--duration", type=float, default=5.0)
    s.add_argument("--policy", default="track_hand")
    s.add_argument("--view", action="store_true", help="open a window")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="stream the annotated view on this port, e.g. 8080; "
                        "shows what the robot sees and decides, for boards "
                        "with no screen")
    s.set_defaults(func=cmd_sim)

    s = sub.add_parser("hybrid", help="tier B: real camera and hand, simulated arm")
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.add_argument("--policy", default="track_hand")
    s.add_argument("--view", action="store_true", help="open a window")
    s.add_argument("--real", action="store_true",
                   help="drive the real arm as well, so it follows your hand. "
                        "Camera and servo adapter must be on this machine")
    s.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="stream the annotated view on this port, e.g. 8080; "
                        "shows what the robot sees and decides, for boards "
                        "with no screen")
    s.set_defaults(func=cmd_hybrid)

    s = sub.add_parser("bench", help="measure what is currently estimated")
    s.add_argument("what", choices=["ik", "camera", "loop", "all"], default="all", nargs="?")
    s.add_argument("--duration", type=float, default=5.0)
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("record", help="capture a camera session to disk")
    s.add_argument("-o", "--output", default="recordings/session")
    s.add_argument("--duration", type=float, default=20.0)
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("replay", help="re-run a recording through the pipeline")
    s.add_argument("path")
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--policy", default="track_hand")
    s.add_argument("--fast", action="store_true", help="ignore original timing")
    s.add_argument("--loop", action="store_true")
    s.add_argument("--view", action="store_true")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="stream the annotated view on this port, e.g. 8080; "
                        "shows what the robot sees and decides, for boards "
                        "with no screen")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("touch", help="detect table objects and touch each one")
    s.add_argument("--duration", type=float, default=25.0)
    s.add_argument("--view", action="store_true")
    s.add_argument("--real", action="store_true",
                   help="required: this command drives the real arm against "
                        "what the real camera sees")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="stream the annotated view on this port, e.g. 8080; "
                        "shows what the robot sees and decides, for boards "
                        "with no screen")
    s.set_defaults(func=cmd_touch)

    s = sub.add_parser("play", help="play hand slap; the robot slaps, you dodge")
    s.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard"])
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--reaction", type=float, default=0.22, help="simulated human reaction, s")
    s.add_argument("--real-hand", action="store_true", help="tier B: use the webcam")
    s.add_argument("--real", action="store_true",
                   help="tier C: real camera, real hand AND the real arm, on this "
                        "machine. The arm will strike at your hand")
    # There is no --contact any more. It offered four ways to judge a
    # round and three of them were worse in ways that had already been
    # measured: `servo` reads load during the swing, where a rigid book
    # scored *between* nothing and a hand; `press` reads it held still,
    # which separates 0.001 from 0.037 but only after twice the settling
    # time, at the servos' torque limit, on a 5 A supply that already
    # drops bus transactions under sustained stall; `proximity` asks a
    # camera about the one moment the arm is between it and the hand. A
    # flag whose other settings are all known-worse is not a choice, it
    # is a way to run the wrong one by accident -- which is exactly what
    # its `proximity` default did for several commits after `height`
    # landed. The encoders answer it directly, so they answer it.
    s.add_argument("--contact-threshold", type=float, default=None,
                   dest="contact_threshold",
                   help="--real only: the absolute floor, in metres, under "
                        "--contact-band. Rarely the one to reach for; the band "
                        "fraction is what decides unless the band is very thin")
    s.add_argument("--contact-band", type=float, default=None,
                   dest="contact_band", metavar="FRACTION",
                   help="--real only: how far up from the commanded floor toward "
                        "the hand a paddle must have stopped to count as blocked, "
                        "as a fraction (default 0.5). Measured on this rig: an "
                        "empty table stalls the paddle 24%% of the way up, a hand "
                        "stops it at 85%%, so half-way separates them. Lower it if "
                        "real hits are scoring as dodges, raise it if the empty "
                        "table scores as hits. The end-of-run line prints the peak "
                        "it saw")
    s.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--view", action="store_true")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="stream the annotated view on this port, e.g. 8080; "
                        "shows what the robot sees and decides, for boards "
                        "with no screen")
    s.add_argument("--scoreboard", type=int, default=0, metavar="PORT",
                   help="serve the live score on this port, e.g. 8090; the "
                        "whole page flashes HIT/DODGED/FLINCH/HELD as each "
                        "round resolves, which is the part a player can read "
                        "without taking their eyes off the arm. Costs no "
                        "encoding, so it is fine to leave on. May share a "
                        "port with --preview")
    s.add_argument("--deadpan", action="store_true",
                   help="no fidgeting and no taunts. The robot plays exactly the "
                        "same game; it just stops performing, which is what you "
                        "want when measuring it rather than watching it")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("eval", help="sweep opponent reaction time, measure win rate")
    s.add_argument("--reactions", default="0.15,0.20,0.25,0.30,0.40")
    s.add_argument("--rounds", type=int, default=12)
    s.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard"])
    s.add_argument("--mean-wait", type=float, default=0.6, dest="mean_wait")
    s.add_argument("--no-feints", action="store_true",
                   help="measure strike physics alone, without the feint game")
    s.add_argument("--timeout", type=float, default=90.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("move", help="move the tool to a position")
    s.add_argument("x", type=float, nargs="?", default=0.22)
    s.add_argument("y", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.12)
    s.add_argument("--joints", type=float, nargs=5, metavar=("J1", "J2", "J3", "J4", "J5"))
    s.add_argument("--home", action="store_true", help="go to the home configuration")
    s.add_argument("--duration", type=float, default=1.5)
    s.add_argument("--hold", type=float, default=0.0, help="stay there for N seconds")
    s.add_argument("--park", action="store_true", help="return home afterwards")
    s.add_argument("--real", action="store_true", help="drive real hardware")
    s.set_defaults(func=cmd_move)

    s = sub.add_parser("flourish", help="run each flourish on its own, to judge it")
    s.add_argument("names", nargs="*", metavar="NAME",
                   help="which to run; default all of them, in order")
    s.add_argument("--list", action="store_true",
                   help="print the table -- joints, sizes, which mood fires each -- and exit")
    s.add_argument("--duration", type=float, default=1.2,
                   help="seconds, for moves that do not carry their own")
    s.add_argument("--speed", type=float, default=12.0,
                   help="rad/s ceiling (Personality.flourish_speed)")
    s.add_argument("--accel", type=float, default=400.0,
                   help="rad/s^2 ceiling for the flourish itself")
    s.add_argument("--jerk", type=float, default=8000.0, help="rad/s^3 ceiling")
    s.add_argument("--scale", default="1.0", metavar="A,B,C",
                   help="multiply every amplitude by each of these, to find "
                        "where a joint stops following the command")
    s.add_argument("--servo-accel", type=float, default=None, dest="servo_accel",
                   help="the servo's own Goal_Acceleration ramp, rad/s^2. One "
                        "byte, so ~39 is its maximum; 0 disables the ramp for "
                        "whatever the motor will give. SHARED WITH THE STRIKE "
                        "-- recheck hit/dodge after changing it")
    s.add_argument("--sweep", default=None, metavar="A,B,C",
                   help="run each move at these speed ceilings and show where "
                        "more speed stops buying more gesture")
    s.add_argument("--repeat", type=int, default=1, help="takes per flourish")
    s.add_argument("--settle", type=float, default=1.0,
                   help="seconds to return HOME before each, so they start alike")
    s.add_argument("--no-pause", action="store_true", dest="no_pause",
                   help="do not wait for Enter between them")
    s.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    s.add_argument("--real", action="store_true", help="force the feetech backend")
    s.set_defaults(func=cmd_flourish)

    s = sub.add_parser("reach", help="probe the reachable workspace")
    s.add_argument("--heights", default="0.02,0.05,0.10,0.15,0.20,0.30")
    s.set_defaults(func=cmd_reach)

    s = sub.add_parser("vision-serve", help="vision board: detect and publish (Orange Pi)")
    s.add_argument("--to", default="255.255.255.255", help="control board host(s), comma separated")
    s.add_argument("--port", type=int, default=45800)
    s.add_argument("--clock-port", type=int, default=45801, dest="clock_port")
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.add_argument("--objects", action="store_true", help="also publish table objects")
    s.add_argument("--duration", type=float, default=0.0, help="0 = run until stopped")
    s.add_argument("--sim", action="store_true", help="synthetic camera, for testing the link")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="serve an annotated MJPEG view on this port (e.g. 8081)")
    s.add_argument("--serial-port", default="", dest="serial_port",
                   help="also publish over this UART device (e.g. /dev/ttyS4). "
                        "Learning/testing link -- see docs/deployment.md")
    s.add_argument("--baudrate", type=int, default=115200)
    s.set_defaults(func=cmd_vision_serve)

    s = sub.add_parser("control", help="control board: consume detections, run the loop (Pi)")
    s.add_argument("--vision-host", default="", dest="vision_host",
                   help="vision board address, for clock sync")
    s.add_argument("--port", type=int, default=45800)
    s.add_argument("--clock-port", type=int, default=45801, dest="clock_port")
    s.add_argument("--policy", default="track_hand")
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--real", action="store_true", help="drive real servos")
    s.add_argument("--no-clock", action="store_true", dest="no_clock",
                   help="run without a clock offset (freshness checks become meaningless)")
    s.add_argument("--serial-port", default="", dest="serial_port",
                   help="receive perception over this UART device instead of UDP "
                        "(e.g. /dev/ttyAMA0). Learning/testing link, mutually "
                        "exclusive with --vision-host/--port")
    s.add_argument("--baudrate", type=int, default=115200)
    s.add_argument("--telemetry-to", default="", dest="telemetry_to",
                   help="stream this arm's joint state to host(s) (comma separated) "
                        "for `tlod arm-viewer`, e.g. your laptop's IP")
    s.add_argument("--telemetry-port", type=int, default=45900, dest="telemetry_port")
    s.add_argument("--telemetry-clock-port", type=int, default=45901, dest="telemetry_clock_port",
                   help="answers clock pings from `tlod arm-viewer --control-host`, "
                        "so it can report true end-to-end latency")
    s.set_defaults(func=cmd_control)

    s = sub.add_parser("arm-viewer", help="watch a control board's arm telemetry (laptop)")
    s.add_argument("--port", type=int, default=45900)
    s.add_argument("--control-host", default="", dest="control_host",
                   help="control board's IP, to measure clock offset for true "
                        "end-to-end latency (optional; falls back to reception age)")
    s.add_argument("--clock-port", type=int, default=45901, dest="clock_port")
    s.add_argument("--duration", type=float, default=0.0, help="0 = run until the window closes")
    s.set_defaults(func=cmd_arm_viewer)

    s = sub.add_parser("vision-check", help="verify vision numerically; for headless boards")
    s.add_argument("--marker", default="green", choices=sorted(MARKER_COLOURS),
                   help="colour of the marker on the gripper, for --with-arm. "
                        "Pick one absent from the rest of the frame: the largest "
                        "blob of that colour wins, whatever it belongs to")
    s.add_argument("--duration", type=float, default=20.0)
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.add_argument("--with-arm", action="store_true", dest="with_arm",
                   help="also score accuracy against forward kinematics (moves the arm)")
    s.add_argument("--poses", type=int, default=8)
    s.add_argument("--save-frames", default="", dest="save_frames",
                   help="write annotated JPEGs here for later inspection")
    s.add_argument("--json", default="", help="write the report as JSON")
    s.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    s.add_argument("--fixed-distance", action="store_true", dest="fixed_distance",
                   help="you held the hand at a constant distance; enforce depth stability")
    s.add_argument("--sim", action="store_true")
    s.set_defaults(func=cmd_vision_check)

    s = sub.add_parser("probe", help="read the arm with torque off; safest first test")
    s.add_argument("--real", action="store_true", help="drive real hardware")
    s.add_argument("--duration", type=float, default=120.0)
    s.add_argument("--rate", type=float, default=10.0, help="reads per second")
    s.add_argument("--keep-torque", action="store_true",
                   help="do not disable torque (the arm will hold position)")
    s.add_argument("--force", action="store_true", help="rehearse against the simulator")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("calibrate", help="measure the lens, then where the camera is")
    s.add_argument("what", choices=["intrinsics", "extrinsics"])
    s.add_argument("-o", "--output", default="",
                   help="where to write the result. Defaults to camera.intrinsics "
                        "or camera.extrinsics from the config, whichever this "
                        "subcommand produces")
    s.add_argument("--camera", type=int, default=None,
                   help="v4l2 index; overrides camera.index in the config. `tlod cameras` lists them by name")
    s.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    s.add_argument("--square", type=float, default=0.025, help="square size, metres")
    s.add_argument("--views", type=int, default=15)
    s.add_argument("--gripper", type=float, default=0.0,
                   help="gripper opening to hold during extrinsics: 0 is one end "
                        "of its travel, 1 the other. Which end is closed depends "
                        "on the sign in your arm calibration, so check rather "
                        "than assume -- an open jaw puts the marker off the tool "
                        "point that forward kinematics reports")
    s.add_argument("--heights", default="0.06,0.14,0.22",
                   help="tool heights to calibrate at, metres (extrinsics). "
                        "Narrow it if the arm hides the marker when raised, but "
                        "as little as possible: the spread in z conditions the solve")
    s.add_argument("--marker", default="green", choices=sorted(MARKER_COLOURS),
                   help="colour of the marker on the gripper (extrinsics). Pick "
                        "one absent from the rest of the frame: the largest blob "
                        "of that colour wins, whatever it belongs to")
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="serve the annotated camera view on this port while "
                        "capturing, e.g. 8080; for headless boards")
    s.add_argument("--fisheye", action="store_true",
                   help="equidistant fisheye model; needed above ~120 deg, where "
                        "the default pinhole model cannot fit at all")
    s.add_argument("--timeout", type=float, default=180.0)
    s.add_argument("--intrinsics", default="",
                   help="extrinsics: path to the intrinsics .npz. Defaults to "
                        "camera.intrinsics from the config")
    s.add_argument("--sim", action="store_true", help="rehearse without hardware")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("cameras", help="list camera indices")
    s.set_defaults(func=cmd_cameras)

    s = sub.add_parser("ports", help="list serial ports")
    s.set_defaults(func=cmd_ports)

    s = sub.add_parser("first-light", help="verify a newly assembled arm, one joint at a time")
    s.add_argument("--amplitude", type=float, default=0.2)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_first_light)

    s = sub.add_parser("power", help="measure current draw and rail sag; diagnoses brownout")
    s.add_argument("--amplitude", type=float, default=0.4,
                   help="radians each joint travels, per move")
    s.add_argument("--duration", type=float, default=0.8,
                   help="seconds per move; shorter means higher acceleration")
    s.add_argument("--json", default=None, help="write the raw samples here")
    s.set_defaults(func=cmd_power)

    s = sub.add_parser("config", help="write the effective config to a file")
    s.add_argument("-o", "--output", default="configs/effective.yaml")
    s.set_defaults(func=cmd_config)

    args = p.parse_args(argv)
    _log_setup(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

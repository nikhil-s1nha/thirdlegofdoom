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
import logging
import sys
import time
from pathlib import Path

import numpy as np

from tlod.config import Config


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

    return FeetechArm(
        port=port,
        baudrate=cfg.arm.baudrate,
        calibration=calib,
        goal_acceleration=cfg.arm.goal_acceleration,
        torque_limit=cfg.arm.torque_limit,
    )


def build_detector(cfg: Config, scene=None):
    from tlod.vision.hands import MediaPipeHandDetector
    from tlod.vision.scene import SceneHandDetector

    if cfg.vision.detector == "scripted":
        if scene is None:
            raise ValueError("the scripted detector needs a scene")
        return SceneHandDetector(scene)
    return MediaPipeHandDetector(
        model_path=cfg.vision.model_path,
        num_hands=cfg.vision.num_hands,
        min_detection_confidence=cfg.vision.min_detection_confidence,
        delegate=cfg.vision.delegate,
    )


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
    """
    from tlod.arm.controller import ArmController, SafetyLimits
    from tlod.arm.mock import MockArm
    from tlod.arm.model import HOME
    from tlod.game.touch import TouchObjectsPolicy
    from tlod.runtime.app import RobotApp
    from tlod.vision.camera import MockCamera
    from tlod.vision.hands import HandLocator
    from tlod.vision.objects import ColorBlobDetector
    from tlod.vision.scene import SceneHandDetector, SyntheticHandScene
    from tlod.vision.tracking import MultiTracker

    cfg = Config.load(args.config)
    projector = build_projector(cfg)
    scene = SyntheticHandScene(projector)
    policy = TouchObjectsPolicy()
    controller = ArmController(
        MockArm(q0=np.concatenate([HOME, [0.0]]), max_speed=cfg.arm.sim_max_speed,
                accel=cfg.arm.sim_accel),
        SafetyLimits(), cfg.runtime.control_hz)

    app = RobotApp(
        # Objects have to be visible, so this run renders pixels and the
        # detector actually looks at them -- unlike the hand path, which
        # short-circuits to scene truth for determinism.
        camera=MockCamera(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                          scene=scene, render=True),
        detector=SceneHandDetector(scene),
        locator=HandLocator(projector, depth_mode="size"),
        controller=controller,
        policy=policy,
        tracker=MultiTracker(),
        object_detector=ColorBlobDetector(projector, min_area_px=150),
        control_hz=cfg.runtime.control_hz,
    )
    app.projector = projector
    print(f"  scene has {len(scene.objects)} objects: "
          f"{', '.join(o.label for o in scene.objects)}")
    _run_for(app, args.duration, view=args.view, projector=projector)
    print(f"\n  touched {len(policy.visited)}: {', '.join(policy.visited) or 'none'}")
    if policy.errors:
        print(f"  placement error: mean {np.mean(policy.errors)*1000:.1f} mm, "
              f"max {np.max(policy.errors)*1000:.1f} mm")
        truth = {o.label: np.array(o.position) for o in scene.objects}
        for det in app.objects:
            if det.label in truth:
                err = np.linalg.norm(det.position - truth[det.label])
                print(f"    {det.label:<6} detected {err*1000:5.1f} mm from true position")
    return 0


def cmd_play(args) -> int:
    """Play hand slap. The robot slaps; you dodge."""
    from tlod.game.contact import GeometricContactSensor, ProximityContactSensor
    from tlod.game.handslap import HandSlapGame
    from tlod.game.opponent import DodgingHand

    cfg = Config.load(args.config)
    if args.real_hand:
        cfg = cfg.with_overrides(camera={"source": "opencv", "index": args.camera},
                                 vision={"detector": "mediapipe"})
        game = HandSlapGame(args.difficulty, contact=ProximityContactSensor(), seed=args.seed)
        app = build_app(cfg)
        app.policy = game
        game.start(app)
        print(f"tier B: real hand, simulated arm. difficulty={args.difficulty}")
        print("put your hand in view and try not to get slapped. space pauses, e is e-stop.")
    else:
        game = HandSlapGame(args.difficulty, contact=GeometricContactSensor(), seed=args.seed)
        app = build_game_app(cfg, game,
                             opponent=DodgingHand(reaction_time=args.reaction, seed=args.seed),
                             render=args.view)
        game.truth_provider = lambda: app.scene.hand.position
        print(f"tier A: simulated opponent (reaction {args.reaction*1000:.0f} ms), "
              f"difficulty={args.difficulty}")

    _run_for(app, args.duration, view=args.view, projector=app.projector)
    print(f"\n  final score: {game.score}  over {game.score.rounds} rounds")
    if game.score.rounds:
        print(f"  robot win rate: {game.score.robot/game.score.rounds:.0%}  "
              f"({game.strikes} strikes, {game.feints} feints)")
    return 0


def cmd_eval(args) -> int:
    """Sweep opponent reaction time and measure the robot's win rate.

    The design question of the project, answered numerically: does a
    short strike actually beat a human, and where is the crossover?
    """
    from tlod.game.contact import GeometricContactSensor
    from tlod.game.handslap import Difficulty, HandSlapGame
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
        game = HandSlapGame(difficulty, contact=GeometricContactSensor(), seed=args.seed)
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
    from tlod.arm.controller import ArmController, SafetyLimits
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
    limits = SafetyLimits(
        max_speed=cfg.safety.max_speed,
        strike_speed=cfg.safety.strike_speed,
        joint_margin=cfg.safety.joint_margin,
        table_z=cfg.safety.table_z,
        min_height=cfg.safety.min_height,
        max_radius=cfg.safety.max_radius,
        min_radius=cfg.safety.min_radius,
        max_height=cfg.safety.max_height,
        command_timeout=cfg.safety.command_timeout,
    )
    controller = ArmController(build_arm(cfg), limits, cfg.runtime.control_hz)
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


def _run_for(app, duration: float, view: bool = False, projector=None) -> None:
    with app:
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
        print(app.latency_report())
        pose = app.controller.pose()
        print(f"\n  final tool position: "
              f"({pose.x:+.3f}, {pose.y:+.3f}, {pose.z:+.3f}) m")


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
    _run_for(app, args.duration, view=args.view, projector=app.projector)
    return 0


def cmd_hybrid(args) -> int:
    """Tier B: your real webcam and real hand, simulated arm."""
    cfg = Config.load(args.config).with_overrides(
        arm={"backend": "mock"},
        camera={"source": "opencv", "index": args.camera},
        vision={"detector": "mediapipe"},
        runtime={"policy": args.policy},
    )
    print(f"tier B hybrid: real camera {args.camera}, real hand, simulated arm "
          f"[policy={args.policy}]")
    print("wave your hand in front of the camera.")
    app = build_app(cfg)
    _run_for(app, args.duration, view=args.view, projector=app.projector)
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

            cam = OpenCVCamera(index=args.camera, width=cfg.camera.width,
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
        camera={"source": "opencv", "index": args.camera})
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
    _run_for(app, args.duration, view=args.view, projector=app.projector)
    return 0


def cmd_move(args) -> int:
    """Move the tool to a position. The core capability, on its own.

    Works identically against the simulator and real hardware -- the
    backend interface is what makes `--real` a flag rather than a
    different program.
    """
    from tlod.arm.controller import ArmController, SafetyLimits
    from tlod.arm.model import HOME, tool_pose
    from tlod.types import Pose

    cfg = Config.load(args.config)
    if args.real:
        cfg = cfg.with_overrides(arm={"backend": "feetech"})

    limits = SafetyLimits(
        max_speed=cfg.safety.max_speed, strike_speed=cfg.safety.strike_speed,
        joint_margin=cfg.safety.joint_margin, table_z=cfg.safety.table_z,
        min_height=cfg.safety.min_height, max_radius=cfg.safety.max_radius,
        min_radius=cfg.safety.min_radius, max_height=cfg.safety.max_height,
    )
    controller = ArmController(build_arm(cfg), limits, cfg.runtime.control_hz)
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
        cfg = cfg.with_overrides(camera={"source": "opencv", "index": args.camera},
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
    )
    print(f"  vision board: publishing to {args.to}:{args.port}, "
          f"clock on :{args.clock_port}")
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
    print("\n" + publisher.report())
    return 0


def cmd_control(args) -> int:
    """Control board: consume detections, run the loop. Runs on the Pi."""
    from tlod.arm.controller import ArmController, SafetyLimits
    from tlod.net.subscriber import VisionSubscriber
    from tlod.runtime.app import IdlePolicy, RobotApp, TrackHandPolicy
    from tlod.vision.tracking import MultiTracker

    cfg = Config.load(args.config)
    if args.real:
        cfg = cfg.with_overrides(arm={"backend": "feetech"})

    subscriber = VisionSubscriber(
        host=args.vision_host, port=args.port, clock_port=args.clock_port,
        require_clock=not args.no_clock,
    )
    print(f"  control board: listening on :{args.port}, clock from "
          f"{args.vision_host or '(none)'}")
    subscriber.start()
    if subscriber.clock:
        print(f"  clock offset {subscriber.clock.offset*1e3:+.2f} ms "
              f"(+/-{subscriber.clock.uncertainty*1e3:.2f} ms)")

    limits = SafetyLimits(
        max_speed=cfg.safety.max_speed, strike_speed=cfg.safety.strike_speed,
        joint_margin=cfg.safety.joint_margin, table_z=cfg.safety.table_z,
        min_height=cfg.safety.min_height, max_radius=cfg.safety.max_radius,
        min_radius=cfg.safety.min_radius, max_height=cfg.safety.max_height,
    )
    policies = {"idle": IdlePolicy, "track_hand": TrackHandPolicy}
    app = RobotApp(
        camera=_NullCamera(),
        detector=None,
        locator=None,
        controller=ArmController(build_arm(cfg), limits, cfg.runtime.control_hz),
        policy=policies.get(args.policy, IdlePolicy)(),
        tracker=MultiTracker(),
        control_hz=cfg.runtime.control_hz,
        perception_max_age=cfg.runtime.perception_max_age,
        prediction_horizon=cfg.runtime.prediction_horizon,
        perception_source=subscriber.perception,
    )
    try:
        with app:
            deadline = time.perf_counter() + args.duration
            while time.perf_counter() < deadline:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        subscriber.stop()
    print(app.latency_report())
    print("\n  network")
    print(subscriber.report())
    return 0


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
    from tlod.vision.calibrate_flow import run_extrinsics, run_intrinsics
    from tlod.vision.calibration import Intrinsics

    cfg = Config.load(args.config)
    out = Path(args.output)

    if args.what == "intrinsics":
        cfg = cfg.with_overrides(camera={"source": "opencv", "index": args.camera})
        camera = build_camera(cfg)
        print(f"  hold a {args.pattern} chessboard (inner corners) with "
              f"{args.square*1000:.0f} mm squares in view.")
        print("  move it around: corners, edges, near, far, tilted. Auto-captures.")
        with camera:
            time.sleep(1.0)
            intr = run_intrinsics(
                camera, pattern=_pattern(args.pattern), square=args.square,
                views=args.views, timeout=args.timeout,
                on_progress=lambda n, total, *_: print(f"    view {n}/{total}", flush=True),
            )
        intr.save(out)
        print(f"\n  reprojection RMS {intr.rms:.3f} px  ->  {out}")
        if intr.rms > 1.0:
            print("  WARNING: above 1 px is poor. Reshoot with more varied views,")
            print("  better light, and the board fully flat.")
        return 0

    # extrinsics
    if not args.intrinsics:
        raise SystemExit("extrinsics needs --intrinsics pointing at the .npz from the first step")
    intr = Intrinsics.load(args.intrinsics)

    from tlod.arm.controller import ArmController, SafetyLimits

    if args.sim:
        # Rehearsal: a synthetic camera that renders a marker at the true
        # tool position. Proves the whole procedure end to end -- motion,
        # detection, solve, residuals -- before it drives real hardware.
        from tlod.vision.calibration import synthetic_projector

        truth = synthetic_projector((cfg.camera.width, cfg.camera.height),
                                    cfg.camera.position, cfg.camera.look_at)
        controller = ArmController(build_arm(cfg), SafetyLimits(), cfg.runtime.control_hz)
        camera = _MarkerCamera(truth, controller, cfg.camera.width, cfg.camera.height)
        print("  SIMULATED rehearsal: no hardware is moving.")
    else:
        cfg = cfg.with_overrides(camera={"source": "opencv", "index": args.camera},
                                 arm={"backend": "feetech"})
        camera = build_camera(cfg)
        controller = ArmController(build_arm(cfg), SafetyLimits(), cfg.runtime.control_hz)
        print("  THE ARM WILL MOVE. Clear the workspace, keep hands away.")
        print("  Attach a green marker to the gripper, visible from the camera.")
        input("  press Enter when ready, Ctrl-C to abort... ")

    controller.start()
    try:
        with camera:
            time.sleep(1.0)
            extr, residuals = run_extrinsics(
                camera, controller, intr,
                on_progress=lambda i, n, *_: print(f"    pose {i}/{n}", flush=True),
            )
    finally:
        controller.stop(park=True)

    extr.save(out)
    residuals = np.array(residuals)
    print(f"\n  camera at ({extr.t[0]:+.3f}, {extr.t[1]:+.3f}, {extr.t[2]:+.3f}) m in base frame")
    print(f"  reprojection: RMS {extr.rms:.2f} px, worst point {residuals.max():.2f} px")
    print(f"  -> {out}")
    if residuals.max() > 3 * max(extr.rms, 0.5):
        print("  NOTE: one point is far worse than the rest -- likely a mislocated")
        print("  marker rather than a bad calibration. Rerun; it should settle.")
    print("\n  verify with:  tlod touch --view    (the drawn arm must land on the real arm)")
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
    from tlod.vision.camera import list_cameras

    found = list_cameras()
    print(f"  camera indices that open: {found or 'none'}")
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
    s.set_defaults(func=cmd_sim)

    s = sub.add_parser("hybrid", help="tier B: real camera and hand, simulated arm")
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--policy", default="track_hand")
    s.add_argument("--view", action="store_true", help="open a window")
    s.set_defaults(func=cmd_hybrid)

    s = sub.add_parser("bench", help="measure what is currently estimated")
    s.add_argument("what", choices=["ik", "camera", "loop", "all"], default="all", nargs="?")
    s.add_argument("--duration", type=float, default=5.0)
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("record", help="capture a camera session to disk")
    s.add_argument("-o", "--output", default="recordings/session")
    s.add_argument("--duration", type=float, default=20.0)
    s.add_argument("--camera", type=int, default=0)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("replay", help="re-run a recording through the pipeline")
    s.add_argument("path")
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--policy", default="track_hand")
    s.add_argument("--fast", action="store_true", help="ignore original timing")
    s.add_argument("--loop", action="store_true")
    s.add_argument("--view", action="store_true")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("touch", help="detect table objects and touch each one")
    s.add_argument("--duration", type=float, default=25.0)
    s.add_argument("--view", action="store_true")
    s.set_defaults(func=cmd_touch)

    s = sub.add_parser("play", help="play hand slap; the robot slaps, you dodge")
    s.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard"])
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--reaction", type=float, default=0.22, help="simulated human reaction, s")
    s.add_argument("--real-hand", action="store_true", help="tier B: use the webcam")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--view", action="store_true")
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

    s = sub.add_parser("reach", help="probe the reachable workspace")
    s.add_argument("--heights", default="0.02,0.05,0.10,0.15,0.20,0.30")
    s.set_defaults(func=cmd_reach)

    s = sub.add_parser("vision-serve", help="vision board: detect and publish (Orange Pi)")
    s.add_argument("--to", default="255.255.255.255", help="control board host(s), comma separated")
    s.add_argument("--port", type=int, default=45800)
    s.add_argument("--clock-port", type=int, default=45801, dest="clock_port")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--objects", action="store_true", help="also publish table objects")
    s.add_argument("--duration", type=float, default=0.0, help="0 = run until stopped")
    s.add_argument("--sim", action="store_true", help="synthetic camera, for testing the link")
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
    s.set_defaults(func=cmd_control)

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
    s.add_argument("-o", "--output", default="calib/intrinsics.npz")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    s.add_argument("--square", type=float, default=0.025, help="square size, metres")
    s.add_argument("--views", type=int, default=15)
    s.add_argument("--timeout", type=float, default=180.0)
    s.add_argument("--intrinsics", default="", help="extrinsics: path to the intrinsics .npz")
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

    s = sub.add_parser("config", help="write the effective config to a file")
    s.add_argument("-o", "--output", default="configs/effective.yaml")
    s.set_defaults(func=cmd_config)

    args = p.parse_args(argv)
    _log_setup(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

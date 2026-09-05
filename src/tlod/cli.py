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
        rng = np.random.default_rng(0)
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
            print(f"  set runtime.prediction_horizon to about this, plus servo travel.")
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

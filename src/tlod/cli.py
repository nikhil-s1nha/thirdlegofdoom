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
    from tlod.vision.calibrate_flow import calibration_poses, find_marker
    from tlod.vision.check import Thresholds, check_against_arm, check_precision
    from tlod.vision.hands import HandLocator

    cfg = Config.load(args.config)
    if args.sim:
        # Synthetic camera and synthetic detector together. MediaPipe on
        # unrendered mock frames finds nothing, which looks like a broken
        # pipeline rather than a misconfigured test.
        cfg = cfg.with_overrides(camera={"source": "mock"}, vision={"detector": "scripted"})
    else:
        cfg = cfg.with_overrides(camera={"source": "opencv", "index": args.camera},
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
            print("  A green marker must be on the gripper.")
            if not args.yes:
                input("  press Enter when ready, Ctrl-C to abort... ")
            controller = ArmController(build_arm(cfg), SafetyLimits(), cfg.runtime.control_hz)
            controller.start()
            try:
                def locate(image):
                    uv = find_marker(image)
                    if uv is None:
                        return None
                    depth_plane = projector.pixel_to_plane(uv[0], uv[1], 0.0)
                    # Resolve the marker the same way a hand would be, so
                    # the check exercises the real path rather than a
                    # shortcut around it.
                    return locator.projector.pixel_to_plane(uv[0], uv[1],
                                                            controller.pose().z) or depth_plane
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
        print(f"\n  {intr.model} model, reprojection RMS {intr.rms:.3f} px  ->  {out}")
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
    s.add_argument("--duration", type=float, default=20.0)
    s.add_argument("--camera", type=int, default=0)
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
    s.add_argument("-o", "--output", default="calib/intrinsics.npz")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    s.add_argument("--square", type=float, default=0.025, help="square size, metres")
    s.add_argument("--views", type=int, default=15)
    s.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="serve the annotated camera view on this port while "
                        "capturing, e.g. 8080; for headless boards")
    s.add_argument("--fisheye", action="store_true",
                   help="equidistant fisheye model; needed above ~120 deg, where "
                        "the default pinhole model cannot fit at all")
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

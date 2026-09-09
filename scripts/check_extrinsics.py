"""Does the camera agree with the arm about where the arm is?

An extrinsics RMS is in pixels, which is not a unit anybody can act on,
and it is computed from the same points that produced the fit -- so it
reports how well the solve explained its own data, not whether the
answer is right. This drives the arm to fresh poses and asks the useful
question instead: project where forward kinematics says the tool is, find
the marker in the image, and report the gap in both pixels and
millimetres on the table.

Millimetres are what matters. A hand is about 90 mm across, so an error
of 20 mm is a graze instead of a hit and 50 mm is a miss.

    python3 scripts/check_extrinsics.py red

The arm moves. Clear the workspace.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")
from tlod.arm import model  # noqa: E402
from tlod.cli import build_arm, build_limits  # noqa: E402
from tlod.arm.controller import ArmController  # noqa: E402
from tlod.config import Config  # noqa: E402
from tlod.types import Pose  # noqa: E402
from tlod.vision.calibrate_flow import MARKER_BANDS, find_marker, wait_until_still  # noqa: E402
from tlod.vision.calibration import Extrinsics, Intrinsics, Projector  # noqa: E402

colour = sys.argv[1] if len(sys.argv) > 1 else "red"
cfg = Config.load(sys.argv[2] if len(sys.argv) > 2 else "configs/opi.yaml")
band = MARKER_BANDS[colour]

intr = Intrinsics.load(cfg.camera.intrinsics)
extr = Extrinsics.load(cfg.camera.extrinsics)
projector = Projector(intr, extr)
print(f"\n  camera at ({extr.t[0]:+.3f}, {extr.t[1]:+.3f}, {extr.t[2]:+.3f}) m, "
      f"fit rms {extr.rms:.2f} px\n")

# Deliberately not the poses the extrinsics was fitted on: a calibration
# that only agrees with its own training points has said nothing.
POSES = [
    Pose(0.20, -0.08, 0.08), Pose(0.24, 0.00, 0.10), Pose(0.20, 0.08, 0.08),
    Pose(0.18, -0.04, 0.13), Pose(0.24, 0.04, 0.12), Pose(0.22, 0.00, 0.16),
]

from tlod.vision.camera import OpenCVCamera  # noqa: E402

camera = OpenCVCamera(index=cfg.camera.index, width=cfg.camera.width,
                      height=cfg.camera.height, fps=cfg.camera.fps,
                      fourcc=cfg.camera.fourcc)
controller = ArmController(build_arm(cfg), build_limits(cfg), cfg.runtime.control_hz)
controller.start()
camera.start()
errors_px: list[float] = []
try:
    controller.set_gripper(0.0)
    time.sleep(0.8)
    for i, pose in enumerate(POSES, 1):
        if not controller.goto_pose(pose, duration=2.0):
            print(f"  {i}. unreachable, skipped")
            continue
        wait_until_still(controller)
        time.sleep(0.4)
        frame = camera.read()
        if frame is None:
            continue
        seen = find_marker(frame.image, band)
        truth = model.fk(controller.state().q[:5])[:3, 3]
        expect = projector.project(truth)
        if seen is None or expect is None:
            print(f"  {i}. marker not seen")
            continue
        err_px = float(np.hypot(seen[0] - expect[0], seen[1] - expect[1]))
        errors_px.append(err_px)
        # Convert to millimetres at the tool's own distance from the
        # camera, which is what a pixel is worth where the arm is -- not
        # at the table, and not averaged over the frame.
        depth = float(np.linalg.norm(truth - extr.t))
        mm = err_px * depth / intr.K[0, 0] * 1000.0
        print(f"  {i}. expected ({expect[0]:6.1f},{expect[1]:6.1f})  "
              f"saw ({seen[0]:6.1f},{seen[1]:6.1f})  "
              f"off by {err_px:5.1f} px = {mm:5.1f} mm")
finally:
    camera.stop()
    controller.stop(park=False)

if errors_px:
    a = np.array(errors_px)
    print(f"\n  mean {a.mean():.1f} px, worst {a.max():.1f} px, over {len(a)} poses")
    print("  under ~5 px is good; a palm is about 90 mm across\n")

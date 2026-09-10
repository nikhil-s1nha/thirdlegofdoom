"""Where does the camera think your hand is? In millimetres, against a ruler.

The arm does not move and is never opened. This is the camera half of
the pipeline on its own, so that a disagreement between where the paddle
lands and where your palm actually is can be pinned on one side or the
other instead of being argued about.

    python3 scripts/where_is_my_hand.py

Put a hand flat on the table, hold still, and read the number. Then
measure the same point with a ruler: x is forward from the centre of the
arm's base, y is to the left, both to the middle of your palm. The
difference between those two numbers is the whole bug, and its direction
says which knob fixes it.

WHAT THE DIRECTION MEANS

  Mostly along the camera's line of sight
      `vision.hand_height` is wrong. With depth_mode: plane the pixel ray
      is intersected against an assumed height, and the camera looks down
      at an angle, so being wrong about the height slides the answer
      sideways -- on this rig about 0.87 mm for every mm of height error,
      always the same direction. The height is measurable: `--contact
      height` reports where the paddle stops when a hand blocks it.

  Some other fixed direction
      The extrinsics are off. `scripts/check_extrinsics.py` measures that
      directly by driving the arm to known poses and asking the camera
      where it thinks the tool is.

  It moves when you rotate your hand without moving it
      Then it is not a fixed offset at all and the landmarks are at
      fault -- PALM_BIAS in vision/hands.py, which slides the aim along
      the wrist-to-knuckles axis. Note that this is the *only* thing that
      knob can do: sideways, the aim is the mean of all four knuckles at
      every setting of it.

That last case is worth testing first because it is free: put the hand
down, read the number, rotate it 180 degrees in place, read it again.
"""

import sys
import time

sys.path.insert(0, "src")
import numpy as np  # noqa: E402

from tlod.cli import build_camera, build_detector, build_projector  # noqa: E402
from tlod.config import Config  # noqa: E402
from tlod.vision.hands import PALM_BIAS, HandLocator  # noqa: E402

cfg = Config.load(sys.argv[1] if len(sys.argv) > 1 else "configs/opi.yaml")
projector = build_projector(cfg)
# A scripted detector or a mock camera needs a scene to read hands from,
# the same wiring build_app() does. Without it this only ran against real
# hardware, which is a poor property for a script whose whole job is
# telling you whether the real hardware is lying to you.
scene = None
if cfg.vision.detector == "scripted" or cfg.camera.source == "mock":
    from tlod.vision.scene import SyntheticHandScene
    scene = SyntheticHandScene(projector)
camera = build_camera(cfg, scene=scene)
detector = build_detector(cfg, scene)
locator = HandLocator(projector, depth_mode=cfg.vision.depth_mode,
                      hand_height=cfg.vision.hand_height,
                      palm_width_m=cfg.vision.palm_width_m)

print(f"\n  assuming hands sit at {cfg.vision.hand_height * 1e3:.0f} mm "
      f"(vision.hand_height, depth_mode {cfg.vision.depth_mode})")
print(f"  aiming {PALM_BIAS:.2f} of the way from wrist to knuckles (PALM_BIAS)")
print("\n  x is forward from the base, y is to the left, millimetres.")
print("  Hold still, then measure the middle of your palm with a ruler.")
print("  Ctrl-C to stop.\n")

# `with` rather than start/stop by hand: Camera is a context manager and
# its grab thread has to be joined even when this exits on Ctrl-C.
try:
    with camera:
        last = 0.0
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if time.perf_counter() - last < 0.5:
                continue
            last = time.perf_counter()
            hands = detector.detect(frame)
            if not hands:
                print("  no hand in view")
                continue
            obs = locator.locate(hands[0])
            if obs is None:
                print("  hand seen, but its ray misses the assumed plane")
                continue
            u, v = hands[0].palm_center
            p = obs.position
            print(f"  palm at x {p[0] * 1e3:+7.1f}  y {p[1] * 1e3:+7.1f}  "
                  f"z {p[2] * 1e3:+6.1f} mm     (pixel {u:.0f}, {v:.0f}, "
                  f"{hands[0].handedness.lower()})")
except KeyboardInterrupt:
    print("\n  stopped")
finally:
    close = getattr(detector, "close", None)
    if callable(close):
        close()

"""Where does the camera think your hand is? In millimetres, against a ruler.

The arm does not move and is never opened. This is the camera half of
the pipeline on its own, so that a disagreement between where the paddle
lands and where your palm actually is can be pinned on one side or the
other instead of being argued about.

    python3 scripts/where_is_my_hand.py
    python3 scripts/where_is_my_hand.py --truth 0.19 0.17

Put a hand flat on the table, hold still, and read the number. Then
measure the same point with a ruler: x is forward from the centre of the
arm's base, y is to the left, both to the middle of your palm.

Give those two numbers back with --truth and this does the rest. The
difference on its own only tells you there is a bug; the arithmetic below
tells you which one, and guessing between them cost several sessions.

HOW IT TELLS THEM APART

A pixel is a *ray*, not a point. Every height along that ray is a
different answer, and `depth_mode: plane` picks one by assuming the hand
sits at `vision.hand_height`. So there are two ways to be wrong, and they
look identical from a single reading:

  The ray is right, the height is wrong
      Then the true palm lies somewhere on the ray, just not where the
      assumed height put it. Solving for the height that lands the ray on
      your ruler measurement gives a sensible number, and that number is
      what `vision.hand_height` should be.

  The ray itself is wrong
      Then no height on it passes near the true palm, because the camera
      is not where the extrinsics say it is. The residual below stays
      large whatever height is tried, and no config value fixes it --
      `scripts/check_extrinsics.py` measures the extrinsics directly by
      driving the arm to known poses.

The third possibility is that the offset is not fixed in the table frame
at all but attached to the hand, which would make it a landmark problem
(PALM_BIAS in vision/hands.py, which slides the aim along the
wrist-to-knuckles axis -- and note that is the *only* thing it can do:
sideways, the aim is the mean of all four knuckles at every setting).
Testing that is free: read the number, rotate the hand 180 degrees
without sliding it, read it again. If the number holds still, it is one
of the two above.
"""

import sys
import time

sys.path.insert(0, "src")
import numpy as np  # noqa: E402

from tlod.cli import build_camera, build_detector, build_projector  # noqa: E402
from tlod.config import Config  # noqa: E402
from tlod.vision.hands import PALM_BIAS, HandLocator  # noqa: E402

argv = sys.argv[1:]
truth = None
if "--truth" in argv:
    i = argv.index("--truth")
    truth = np.array([float(argv[i + 1]), float(argv[i + 2])])
    del argv[i:i + 3]
cfg = Config.load(argv[0] if argv else "configs/opi.yaml")
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

_explained: set[str] = set()


def diagnose(projector, u, v, seen, truth) -> None:
    """Which of the two errors is this? Solved, not guessed.

    A pixel is a ray. Its horizontal position is linear in height:

        xy(z) = a + b * z,   b = ray_xy / ray_z

    so the height that brings the ray closest to the ruler measurement
    has a closed form, and the distance left over at that height says
    whether the ray goes near the true palm at all. A small residual
    means the ray is right and only the assumed height was wrong -- and
    then the solved height is the answer. A large one means no height on
    this ray reaches the true palm, so the ray is wrong, which is the
    extrinsics and not something a config value can fix.
    """
    origin, direction = projector.ray(u, v)
    if abs(direction[2]) < 1e-9:
        return
    b = direction[:2] / direction[2]
    a = origin[:2] - origin[2] * b

    off = seen[:2] - truth
    # Height that puts the ray closest to the truth, and what is left.
    z = float(np.dot(b, truth - a) / np.dot(b, b))
    residual = float(np.linalg.norm(a + b * z - truth))
    plausible = residual < 0.008 and 0.0 <= z <= 0.06
    verdict = "height" if plausible else "extrinsics"

    print(f"    off by {np.linalg.norm(off) * 1e3:5.1f} mm  "
          f"(x {off[0] * 1e3:+.1f}, y {off[1] * 1e3:+.1f})   "
          f"best-fit height {z * 1e3:+.0f} mm, residual {residual * 1e3:.0f} mm"
          f"  -> {verdict}")

    # The explanation once per verdict, not once per frame. Sixty identical
    # paragraphs is how a clear answer gets lost in its own output.
    if verdict in _explained:
        return
    _explained.add(verdict)
    slope = float(np.linalg.norm(b))
    print(f"\n    this ray moves {slope:.2f} mm sideways per mm of height error")
    if plausible:
        print(f"    the ray passes {residual * 1e3:.1f} mm from your palm at "
              f"z = {z * 1e3:.0f} mm, so the ray is right and only the height")
        print(f"    was wrong: set vision.hand_height to {z:.3f}\n")
    else:
        print(f"    no height on this ray gets closer than {residual * 1e3:.0f} mm "
              f"(best would be z = {z * 1e3:.0f} mm, which is")
        print("    not where a flat hand sits). The ray itself is wrong, so this is")
        print("    the extrinsics and no value of vision.hand_height touches it:")
        print("\n        tlod -c configs/opi.yaml calibrate extrinsics \\")
        print("            --marker red --gripper 0 --preview 8080\n")


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
            if truth is not None:
                diagnose(projector, u, v, p, truth)
except KeyboardInterrupt:
    print("\n  stopped")
finally:
    close = getattr(detector, "close", None)
    if callable(close):
        close()

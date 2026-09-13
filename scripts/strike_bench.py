"""One strike at a time, with the numbers, and nothing else running.

The game makes a strike hard to judge: it decides when to commit, it
bluffs, and it aborts the swing the moment contact fires -- so a strike
that never travelled and a strike that landed instantly look identical
from the outside. This does one strike when you ask for it, and prints
what the arm actually did.

    python3 scripts/strike_bench.py                  # over the default spot
    python3 scripts/strike_bench.py 0.22 0.00        # over a point you pick

Press Enter to strike, Ctrl-C to stop. Contact is not detected and the
swing is never aborted, so every run travels the whole way.

The load trace is the point. Run it once over an empty table and once
over a book, and compare: the difference between those two is the only
honest basis for a contact threshold. On this arm the swing raises all
three pitch joints by ~0.12 on its own, purely from braking, at the same
instant every time -- so a threshold set without that comparison detects
the arm rather than the hand.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")
from tlod.arm import model  # noqa: E402
from tlod.arm.controller import ArmController  # noqa: E402
from tlod.arm.primitives import Strike, StrikeLimits  # noqa: E402
from tlod.cli import build_arm, build_governor, build_limits  # noqa: E402
from tlod.config import Config  # noqa: E402
from tlod.types import Pose  # noqa: E402

WATCHED = (1, 2, 3)          # shoulder_lift, elbow_flex, wrist_flex

x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.22
y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
config = sys.argv[3] if len(sys.argv) > 3 else "configs/opi.yaml"

cfg = Config.load(config)
limits = StrikeLimits()
plane = cfg.vision.hand_height                 # where a flat palm sits
hover = Pose(x, y, plane + limits.hover_height)

print(f"\n  strike bench: over ({x:+.3f}, {y:+.3f}), target plane {plane * 1000:.0f} mm")
print(f"  hover {limits.hover_height * 1000:.0f} mm, drop clamped to "
      f"{limits.max_drop * 1000:.0f} mm, torque {limits.torque_limit}/1000 while striking")
print("\n  THE ARM WILL MOVE. Clear the workspace.")
input("  press Enter when ready, Ctrl-C to abort... ")

controller = ArmController(build_arm(cfg), build_limits(cfg), cfg.runtime.control_hz,
                           governor=build_governor(cfg))
controller.start()
period = 1.0 / cfg.runtime.control_hz
n = 0
try:
    while True:
        controller.goto_pose(hover, duration=1.2)
        time.sleep(0.3)
        started = controller.pose()
        print(f"\n  hovering at {started.z * 1000:.0f} mm")
        input("  Enter to strike, Ctrl-C to stop... ")

        motion = Strike([x, y, plane], limits, duration=0.25)
        motion.start(controller)
        trace: list[tuple[float, float, np.ndarray]] = []
        t0 = time.perf_counter()
        while not motion.step(controller, period):
            state = controller.state()
            load = (np.abs(np.asarray(state.load, float))[list(WATCHED)]
                    if state.load is not None else np.zeros(3))
            trace.append((time.perf_counter() - t0,
                          model.fk(state.q[:5])[2, 3], load))
            time.sleep(period)

        n += 1
        ended = controller.pose()
        if not trace:
            print("  no samples -- the motion finished before the first read")
            continue
        base = trace[0][2]
        rises = [float(np.max(load - base)) for _, _, load in trace]
        peak_at, peak = max(zip((t for t, _, _ in trace), rises), key=lambda p: p[1])

        print(f"\n  strike {n}")
        print(f"    travelled  {started.z * 1000:.0f} -> {ended.z * 1000:.0f} mm "
              f"({(started.z - ended.z) * 1000:.0f} mm) in "
              f"{trace[-1][0] * 1000:.0f} ms")
        print(f"    peak load rise {peak:.3f} at {peak_at * 1000:.0f} ms")
        print("    ms     z mm    load rise (shoulder, elbow, wrist)")
        # Every other sample: enough to see the shape, short enough to read.
        for (t, z, load), rise in list(zip(trace, rises))[::2]:
            print(f"    {t * 1000:5.0f}  {z * 1000:6.0f}    "
                  f"{np.round(load - base, 3)}  max {rise:+.3f}")
except KeyboardInterrupt:
    print("\n  stopped")
finally:
    controller.stop(park=False)

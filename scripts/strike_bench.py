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

The traces are the point. Run it once over an empty table and once over
a book, and compare: the difference between those two is the only honest
basis for a contact threshold. On this arm the swing raises all three
pitch joints by ~0.12 on its own, purely from braking, at the same
instant every time -- so a threshold set without that comparison detects
the arm rather than the hand.

Two signals are traced, and they are not the same measurement:

  load     Present_Load, addr 60. The PWM duty the servo is commanding.
           Torque_Limit hard-clamps it, and `Strike` drops that limit to
           350/1000 for the swing, so on this arm load pins at ~0.35
           over an *empty* table. A signal already at its ceiling cannot
           rise further for a hand. This is why load-based contact
           detection was abandoned.
  current  Present_Current, addr 69, in amps. What the motor actually
           draws. Not clamped by Torque_Limit. Holding a rotor back at
           unchanged duty collapses its back-EMF and the current climbs,
           so this still has headroom exactly where load has none.

Measured on this arm, neither does. Over an empty table the wrist load
peaks at the 0.350 ceiling; over a hand it peaks *lower*, at 0.326.
Current peaked at 0.052 A empty and 0.058 A over a hand -- 0.006 A
apart, which is exactly one 6.5 mA quantisation step, and smaller than
the sample-to-sample jitter within either run. Neither channel can see a
hand.

Which is why the third column exists. `Strike` drops Torque_Limit to 350
for the swing, and at 350 the arm cannot track its own command even with
nothing under it: it is asked down to 27 mm and reaches 44 mm. That
17 mm shortfall is the servos already saturated against gravity and
friction alone, before a hand is involved -- the same "no headroom left"
failure as load, in a different register. If raising the torque limit
closes that gap over an empty table, the remaining lag becomes a real
contact signal. If it does not, this arm cannot feel a hand at all and
vision is the only judge available.
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
        trace: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
        t0 = time.perf_counter()
        while not motion.step(controller, period):
            state = controller.state()
            load = (np.abs(np.asarray(state.load, float))[list(WATCHED)]
                    if state.load is not None else np.zeros(3))
            amps = (np.abs(np.asarray(state.current, float))[list(WATCHED)]
                    if state.current is not None else np.zeros(3))
            # Where the servos were *told* to be, alongside where they are.
            # The gap is the third candidate signal: a servo blocked by a
            # hand falls further behind its command than one moving freely.
            trace.append((time.perf_counter() - t0,
                          model.fk(state.q[:5])[2, 3],
                          model.fk(controller.commanded[:5])[2, 3], load, amps))
            time.sleep(period)

        n += 1
        ended = controller.pose()
        if not trace:
            print("  no samples -- the motion finished before the first read")
            continue
        # Baselines from the first sample: resting load and current both
        # depend on the arm's configuration, so an absolute threshold would
        # fire on posture instead of on contact.
        load_base, amp_base = trace[0][3], trace[0][4]
        load_rise = [float(np.max(load - load_base)) for _, _, _, load, _ in trace]
        amp_rise = [float(np.max(amps - amp_base)) for _, _, _, _, amps in trace]
        lag = [z - cz for _, z, cz, _, _ in trace]
        stamps = [t for t, _, _, _, _ in trace]
        load_at, load_peak = max(zip(stamps, load_rise), key=lambda p: p[1])
        amp_at, amp_peak = max(zip(stamps, amp_rise), key=lambda p: p[1])
        lag_at, lag_peak = max(zip(stamps, lag), key=lambda p: p[1])

        print(f"\n  strike {n}")
        print(f"    travelled  {started.z * 1000:.0f} -> {ended.z * 1000:.0f} mm "
              f"({(started.z - ended.z) * 1000:.0f} mm) in "
              f"{trace[-1][0] * 1000:.0f} ms")
        print(f"    peak load rise    {load_peak:+.3f}      at {load_at * 1000:4.0f} ms"
              f"   (ceiling {limits.torque_limit / 1000:.3f})")
        print(f"    peak current rise {amp_peak:+.3f} A    at {amp_at * 1000:4.0f} ms")
        print(f"    peak lag behind command {lag_peak * 1000:+.0f} mm at {lag_at * 1000:4.0f} ms")
        print(f"    commanded floor {trace[-1][2] * 1000:.0f} mm, reached "
              f"{trace[-1][1] * 1000:.0f} mm")
        if float(np.max([np.max(a) for _, _, _, _, a in trace])) == 0.0:
            print("    current reads zero throughout -- this servo firmware does not"
                  "\n    populate Present_Current, so only load is available")
        print("    ms     z mm   cmd mm  lag    load rise            current rise A")
        # Every other sample: enough to see the shape, short enough to read.
        for (t, z, cz, load, amps), lr, ar in list(zip(trace, load_rise, amp_rise))[::2]:
            print(f"    {t * 1000:5.0f}  {z * 1000:6.0f}  {cz * 1000:6.0f} "
                  f"{(z - cz) * 1000:+5.0f}   "
                  f"{np.round(load - load_base, 3)} {lr:+.3f}   "
                  f"{np.round(amps - amp_base, 3)} {ar:+.3f}")
except KeyboardInterrupt:
    print("\n  stopped")
finally:
    controller.stop(park=False)

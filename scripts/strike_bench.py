"""One strike at a time, with the numbers, and nothing else running.

The game makes a strike hard to judge: it decides when to commit, it
bluffs, and it aborts the swing the moment contact fires -- so a strike
that never travelled and a strike that landed instantly look identical
from the outside. This does one strike when you ask for it, and prints
what the arm actually did.

    python3 scripts/strike_bench.py                  # over the default spot
    python3 scripts/strike_bench.py 0.22 0.00        # over a point you pick
    python3 scripts/strike_bench.py --torque 600     # with more to press with

Press Enter to strike, Ctrl-C to stop. Type a word first ("book", "hand")
to label the run. Contact is not detected and the swing is never aborted,
so every run travels the whole way.

WHAT THIS MEASURES, AND WHY THE FIRST VERSION OF IT MEASURED NOTHING

Three signals, and they are not the same measurement:

  load     Present_Load, addr 60. The PWM duty the servo is *commanding*.
           Torque_Limit hard-clamps it, and the strike lowers that limit
           so the arm yields on contact. A clamped signal cannot rise.
  current  Present_Current, addr 69, in amps. What the motor actually
           draws. Not clamped by Torque_Limit, but quantised at 6.5 mA.
  lag      Commanded height minus reached height. A servo held back by
           something falls further behind its command.

Measured on this arm at Torque_Limit 350, across nothing / a book / a
hand, all three scrambled: load 0.330 / 0.326 / 0.350, current 0.065 /
0.078 / 0.071. A rigid book -- the easiest possible thing to feel --
landed in the middle of the other two on both channels.

The reason is in the traces rather than the peaks. Load climbs
monotonically from the first sample to the last in *every* run, empty
table included, and arrives at the ceiling before the paddle arrives
anywhere. That is the arm accelerating and braking its own mass. The
whole swing is one long transient, so a peak taken during it is a
measurement of the swing, not of what the swing hit.

Hence the hold. After the drop the arm stays down, still at the strike's
torque limit, and is sampled again once it has stopped moving. In steady
state there is no acceleration left to confound anything: whatever load
and current remain are the arm pressing on what is underneath it. Empty
table against book is then a difference in a static quantity, which is
the only comparison that was ever going to work.

The starting height matters as much as the signal. A hover that comes up
short shortens the drop, and the endpoint scatter between two strikes
from different heights is larger than the difference this is looking
for -- so the hover is verified before each strike rather than assumed,
and a run that starts low says so instead of quietly polluting the A/B.
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
HOLD = 0.5                   # seconds pressing at the bottom before retract
SETTLE = 0.30                # of that, discarded before reading; see below
HOVER_TOLERANCE = 0.004      # metres; a start further off than this is flagged

argv = [a for a in sys.argv[1:]]
torque = None
if "--torque" in argv:
    i = argv.index("--torque")
    torque = int(argv[i + 1])
    del argv[i:i + 2]

x = float(argv[0]) if len(argv) > 0 else 0.22
y = float(argv[1]) if len(argv) > 1 else 0.0
config = argv[2] if len(argv) > 2 else "configs/opi.yaml"

cfg = Config.load(config)
limits = StrikeLimits()
if torque is not None:
    limits.torque_limit = torque
# Strike now holds at the bottom by itself, which is what makes contact
# detectable in the game. Here that would double up with the bench's own
# hold and fold the press into the reported drop time, so the primitive's
# hold is switched off and this script does the holding -- which keeps
# "how long did the drop take" an honest number.
limits.press_hold = 0.0
plane = cfg.vision.hand_height                 # where a flat palm sits
hover = Pose(x, y, plane + limits.hover_height)
# The floor Strike will command from a full-height hover, computed the
# same way it computes it, so the printout can say how far short of its
# own target the arm stopped.
floor = max(plane + limits.plane_margin,
            hover.z - limits.clamp_drop(hover.z - plane))

print(f"\n  strike bench: over ({x:+.3f}, {y:+.3f}), target plane {plane * 1000:.0f} mm")
print(f"  hover {hover.z * 1000:.0f} mm, floor {floor * 1000:.0f} mm, "
      f"torque {limits.torque_limit}/1000 while striking and holding")
print(f"  holding {HOLD * 1000:.0f} ms at the bottom before retracting")
print("\n  THE ARM WILL MOVE. Clear the workspace.")
input("  press Enter when ready, Ctrl-C to abort... ")

controller = ArmController(build_arm(cfg), build_limits(cfg), cfg.runtime.control_hz,
                           governor=build_governor(cfg))
controller.start()
period = 1.0 / cfg.runtime.control_hz
set_limit = getattr(controller.backend, "set_torque_limit", None)
n = 0


def sample(t0):
    """One (elapsed, reached z, commanded z, load, current) row."""
    state = controller.state()
    load = (np.abs(np.asarray(state.load, float))[list(WATCHED)]
            if state.load is not None else np.zeros(3))
    amps = (np.abs(np.asarray(state.current, float))[list(WATCHED)]
            if state.current is not None else np.zeros(3))
    return (time.perf_counter() - t0,
            model.fk(state.q[:5])[2, 3],
            model.fk(controller.commanded[:5])[2, 3],
            load, amps)


try:
    while True:
        # Two attempts: after a strike the arm starts from the bottom, and
        # one 1.2 s move does not always finish the climb. An unmatched
        # start height is the single easiest way to ruin this comparison.
        for _ in range(2):
            controller.goto_pose(hover, duration=1.2)
            time.sleep(0.3)
            if abs(controller.pose().z - hover.z) <= HOVER_TOLERANCE:
                break
        started = controller.pose()
        short = hover.z - started.z
        print(f"\n  hovering at {started.z * 1000:.0f} mm"
              + (f"  -- {short * 1000:.0f} mm SHORT of {hover.z * 1000:.0f}, this run "
                 f"is not comparable with a full-height one" if short > HOVER_TOLERANCE else ""))
        label = input("  Enter to strike (or type a label first), Ctrl-C to stop... ").strip()

        motion = Strike([x, y, plane], limits, duration=0.25)
        motion.start(controller)
        bottom = Pose(x, y, max(plane + limits.plane_margin,
                                started.z - limits.clamp_drop(started.z - plane)))
        trace = []
        t0 = time.perf_counter()
        while not motion.step(controller, period):
            trace.append(sample(t0))
            time.sleep(period)
        drop_ms = (time.perf_counter() - t0) * 1000

        # The hold. Strike restores the normal torque limit when it
        # finishes, so put it back: the point is to press on whatever is
        # down there with the same authority the swing had.
        if callable(set_limit):
            set_limit(limits.torque_limit)
        hold = []
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < HOLD:
            controller.servo_pose(bottom, max_speed=limits.strike_speed, dt=period)
            hold.append(sample(t0))
            time.sleep(period)
        if callable(set_limit):
            set_limit(limits.normal_torque_limit)

        n += 1
        ended = controller.pose()
        if not trace:
            print("  no samples -- the motion finished before the first read")
            continue

        load_base, amp_base = trace[0][3], trace[0][4]
        rows = trace + hold
        load_rise = [float(np.max(l - load_base)) for *_, l, _ in rows]
        amp_rise = [float(np.max(a - amp_base)) for *_, a in rows]
        stamps = [r[0] for r in rows]
        load_at, load_peak = max(zip(stamps, load_rise), key=lambda p: p[1])
        amp_at, amp_peak = max(zip(stamps, amp_rise), key=lambda p: p[1])

        # The steady-state numbers: everything after SETTLE seconds of
        # pressing, by which point the arm has stopped arriving and the
        # servo's load filter has forgotten the swing. This is the same
        # window ServoPressContactSensor reads, so the numbers below are
        # directly comparable with its threshold.
        settled = [r for r in hold if r[0] > drop_ms / 1000 + SETTLE]
        title = f"strike {n}" + (f" -- {label}" if label else "")
        print(f"\n  {title}")
        print(f"    travelled  {started.z * 1000:.0f} -> {ended.z * 1000:.0f} mm "
              f"({(started.z - ended.z) * 1000:.0f} mm) in {drop_ms:.0f} ms, "
              f"floor was {bottom.z * 1000:.0f} mm "
              f"({(ended.z - bottom.z) * 1000:+.0f} mm short)")
        print(f"    peak during swing: load {load_peak:+.3f} at {load_at * 1000:4.0f} ms"
              f" (ceiling {limits.torque_limit / 1000:.3f}), "
              f"current {amp_peak:+.3f} A at {amp_at * 1000:4.0f} ms")
        if settled:
            sl = np.mean([np.max(r[3] - load_base) for r in settled])
            sa = np.mean([np.max(r[4] - amp_base) for r in settled])
            sz = np.mean([r[1] for r in settled])
            print(f"    HELD at {sz * 1000:.0f} mm: load {sl:+.3f}, current {sa:+.3f} A"
                  f"   <- compare this line between runs")
        if float(np.max([np.max(r[4]) for r in rows])) == 0.0:
            print("    current reads zero throughout -- this servo firmware does not"
                  "\n    populate Present_Current, so only load is available")
        print("    ms     z mm   cmd mm  lag    load rise            current rise A")
        for (t, z, cz, load, amps), lr, ar in list(zip(rows, load_rise, amp_rise))[::2]:
            mark = "  hold" if t * 1000 > drop_ms else ""
            print(f"    {t * 1000:5.0f}  {z * 1000:6.0f}  {cz * 1000:6.0f} "
                  f"{(z - cz) * 1000:+5.0f}   {np.round(load - load_base, 3)} {lr:+.3f}   "
                  f"{np.round(amps - amp_base, 3)} {ar:+.3f}{mark}")
except KeyboardInterrupt:
    print("\n  stopped")
finally:
    if callable(set_limit):
        set_limit(limits.normal_torque_limit)
    controller.stop(park=False)

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
  current  Present_Current, addr 69. Retired: it separated nothing from a
           book from a hand by 0.006 A, one 6.5 mA quantisation step, and
           the fifteen-byte sync read it needed was costing transactions
           on a bus that already drops the occasional one. The column
           stays and reads zero, because the null result is worth being
           able to see rather than rediscover.
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
from tlod.arm.primitives import Strike  # noqa: E402
from tlod.cli import (  # noqa: E402
    build_arm, build_governor, build_limits, build_strike_limits,
)
from tlod.config import Config  # noqa: E402
from tlod.game.contact import CollisionPlaneContactSensor  # noqa: E402
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
# The game's limits, not the defaults. Bare `StrikeLimits()` has
# `tip_offset` 0 and the default `press_depth`, so the bench was measuring
# a different strike from the one being tuned -- and worse, it still felt
# `safety.min_height` through the controller's clamp, so raising that for
# the paddle left the bench commanding a floor the arm was already below
# and reporting a 2 mm "strike".
limits = build_strike_limits(cfg)
if torque is not None:
    limits.torque_limit = torque
# Strike now holds at the bottom by itself, which is what makes contact
# detectable in the game. Here that would double up with the bench's own
# hold and fold the press into the reported drop time, so the primitive's
# hold is switched off and this script does the holding -- which keeps
# "how long did the drop take" an honest number.
limits.press_hold = 0.0
# CollisionPlaneContactSensor.settle -- how far into the press the game
# takes its one reading. Imported rather than repeated so this cannot
# drift away from the sensor it is reporting on.
SENSOR_SETTLE = CollisionPlaneContactSensor(lambda: 0.0).settle
plane = cfg.vision.hand_height                 # where a flat palm sits
# Tool-point heights, so the tip lands where `plane` says. `Hover` and
# `Strike` add `tip_offset` themselves; this reproduces it because the
# bench builds its own poses rather than driving those motions.
hover = Pose(x, y, plane + limits.tip_offset + limits.hover_height)
def commanded_floor(start_z: float) -> float:
    """Where `Strike` will actually send the paddle from `start_z`.

    This has to be Strike._on_start's arithmetic exactly, and for a while
    it was not. The old version measured the drop to the *hand plane* and
    took max() against a floor below it -- `max(plane - press_depth,
    start - clamp_drop(start - plane))` -- which is the bug Strike itself
    was fixed for: the second term lands on the plane, and max() of the
    plane against something below it returns the plane. So the bench
    reported the hand plane as the floor and held the arm there, while
    the strike it had just run commanded 17 mm lower.

    It only showed up when the hover came up short, because with a full
    hover both terms clamp to the same place. A 12 mm low hover was enough
    to make "floor was 28 mm (-5 mm short)" out of a strike that had in
    fact commanded 11 mm and stopped 16 mm above it.
    """
    floor = plane + limits.tip_offset - limits.press_depth
    return max(floor, start_z - limits.clamp_drop(start_z - floor))


floor = commanded_floor(hover.z)

print(f"\n  strike bench: over ({x:+.3f}, {y:+.3f}), target plane {plane * 1000:.0f} mm")
print(f"  hover {hover.z * 1000:.0f} mm, floor {floor * 1000:.0f} mm, "
      f"torque {limits.torque_limit}/1000 while striking and holding")
print(f"  holding {HOLD * 1000:.0f} ms at the bottom before retracting")
print("\n  THE ARM WILL MOVE. Clear the workspace.")
input("  press Enter when ready, Ctrl-C to abort... ")

# The flex compensation has to be here too, or the bench and the game are
# not measuring the same thing. `ArmController.compensate` raises every
# Cartesian target by the droop no sensor can see -- 33 mm at full reach
# on this rig -- and `pose()` takes it back off. Built without it, this
# script commands and reports in the kinematic frame while `tlod play`
# commands and reports in the real one, so the empty-table cluster
# measured here and the hand cluster measured in a play session are in
# frames a whole signal apart. That is exactly the mixed-frame bug the
# strike itself was fixed for; the bench kept it.
controller = ArmController(build_arm(cfg), build_limits(cfg), cfg.runtime.control_hz,
                           governor=build_governor(cfg),
                           flex_gain=cfg.arm.flex_gain,
                           flex_offset=cfg.arm.flex_offset)
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
    # Real frame, both of them. These used to be raw forward kinematics,
    # which was self-consistent while the controller had no flex
    # compensation and became a 37 mm lie the moment it did -- the
    # summary above reads `controller.pose()` and printed "travelled 92
    # -> 18 mm" over a trace whose own rows said 129 -> 55.
    return (time.perf_counter() - t0,
            controller.uncompensate(model.tool_pose(state.q[:5])).z,
            controller.uncompensate(model.tool_pose(controller.commanded[:5])).z,
            load, amps)


try:
    while True:
        # Climb until it actually gets there. Two 1.2 s attempts was not
        # enough from the bottom of a strike -- a whole session came up
        # 11-17 mm short on every single run, and the bench dutifully
        # warned and then measured them anyway. A short hover means a
        # short drop, which is exactly the variable being controlled for.
        for attempt in range(5):
            controller.goto_pose(hover, duration=1.2 if attempt else 1.6)
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
        bottom = Pose(x, y, commanded_floor(started.z))
        trace = []
        t0 = time.perf_counter()
        while True:
            row = sample(t0)
            # Feed the measured height in, exactly as HandSlapGame does.
            # Without this `Strike` falls back to `controller.settled()`
            # and the bench measures the *old* descent gate -- which is
            # the one thing it must not do, because the whole point of
            # this script is to model what the game will see. It ran that
            # way for one session and produced a "the game would read"
            # line describing behaviour that was no longer shipping.
            motion.observe(row[1])
            if motion.step(controller, period):
                break
            trace.append(row)
            time.sleep(period)
        drop_ms = (time.perf_counter() - t0) * 1000
        descent_ended = getattr(motion, "ended_because", "")

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
              f"({(ended.z - bottom.z) * 1000:+.0f} mm short)"
              + (f", descent {descent_ended}" if descent_ended else ""))
        print(f"    peak during swing: load {load_peak:+.3f} at {load_at * 1000:4.0f} ms"
              f" (ceiling {limits.torque_limit / 1000:.3f}), "
              f"current {amp_peak:+.3f} A at {amp_at * 1000:4.0f} ms")
        # When the arm actually stopped, and what the game's sensor would
        # have read before it did.
        #
        # This is the question the trace answers and the summary used to
        # hide. `Strike` reports done when the *commanded* setpoint has
        # settled, which is not the arm: the paddle still has travel left
        # in it and carries on past, then recovers. Meanwhile
        # CollisionPlaneContactSensor waits `settle` (120 ms) into the
        # press and reads once. If the arm is still moving then, the
        # reading is not the shortfall -- it is a snapshot of the rebound,
        # and it is wrong in whichever direction the arm happens to be
        # going.
        if hold:
            zs = [r[1] for r in hold]
            final = float(np.mean(zs[-5:]))
            stop_at = hold[-1][0]
            for i, (t, *_) in enumerate(hold):
                if all(abs(r[1] - final) <= 0.001 for r in hold[i:]):
                    stop_at = t
                    break
            hold_t0 = hold[0][0]
            # The number `arm.strike_sag` wants, when this run is over an
            # empty table: how far under its commanded floor an
            # unobstructed strike actually stops. Printed because it is
            # what the contact sensor measures against, and guessing it
            # is what made every round read as a dodge.
            if final < bottom.z:
                print(f"    -> arm.strike_sag: {(bottom.z - final):.4f}  "
                      f"(if nothing was under the paddle)")
            print(f"    settled at {final * 1000:.0f} mm, "
                  f"{(stop_at - hold_t0) * 1000:.0f} ms into the hold "
                  f"({(final - bottom.z) * 1000:+.0f} mm short of the floor)")
            at_read = next((r for r in hold if r[0] - hold_t0 >= SENSOR_SETTLE), None)
            if at_read is not None:
                print(f"    the game would read {at_read[1] * 1000:.0f} mm at "
                      f"{SENSOR_SETTLE * 1000:.0f} ms "
                      f"({(at_read[1] - bottom.z) * 1000:+.0f} mm short) "
                      f"-- {(at_read[1] - final) * 1000:+.0f} mm off the settled value")
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

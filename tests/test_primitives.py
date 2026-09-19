"""Motion primitive tests.

Weighted toward the safety properties, because the strike primitive is
the part of this system that moves fast toward a person.
"""

import time

import numpy as np
import pytest

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.arm.primitives import (
    FLOURISHES, MEASURED_TRAVEL, Feint, Flourish, GoTo, GoToPose, Hold, Hover, Retract, Sequence,
    Strike, StrikeLimits, flourish,
)
from tlod.types import JOINT_NAMES, Pose


@pytest.fixture
def controller():
    c = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])), control_hz=200.0)
    c.start()
    yield c
    c.backend.disconnect()


def drive(motion, controller, limit=6.0, dt=0.005):
    motion.start(controller)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if motion.step(controller, dt):
            return True
        time.sleep(dt)
    return False


def test_goto_reaches_configuration(controller):
    target = model.HOME + np.array([0.2, 0.1, -0.1, 0.05, 0.0])
    assert drive(GoTo(target, duration=0.4), controller)
    assert np.allclose(controller.commanded[:5], target, atol=1e-3)


def test_goto_pose_reaches_point(controller):
    assert drive(GoToPose(Pose(0.23, 0.04, 0.13), duration=0.4), controller)
    assert np.linalg.norm(controller.pose().xyz() - np.array([0.23, 0.04, 0.13])) < 3e-3


def test_goto_pose_gives_up_on_unreachable(controller):
    m = GoToPose(Pose(3.0, 3.0, 3.0), duration=0.2)
    assert drive(m, controller)
    assert not m.ok


def test_hover_sits_above_the_target(controller):
    limits = StrikeLimits(hover_height=0.09)
    target = np.array([0.22, 0.0, 0.03])
    assert drive(Hover(target, limits, duration=0.4), controller)
    assert abs(controller.pose().z - (target[2] + 0.09)) < 4e-3


def test_strike_goes_below_the_plane_but_only_by_press_depth(controller):
    """The bound on how far under the hand the paddle is allowed to go.

    This test used to assert the opposite -- that the paddle never went
    below the plane at all -- on the reasoning that stopping short of the
    hand is what keeps a wrong height estimate harmless. That reasoning
    named the wrong guard. What bounds the force is `torque_limit`: at
    350/1000 the arm leans on a rigid book by 0.038 of rated torque and
    stops, however deep it is asked to go. The geometry only decided
    whether contact happened at all, and above the plane it decided
    "usually not" -- both contact sensors ask whether the paddle was
    stopped *short* of its floor, and a floor above the hand is one that
    a touched paddle and an untouched one both reach.

    So the guarantee is now two-sided, and this pins both sides.
    """
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.05])
    drive(Hover(target, limits, duration=0.4), controller)
    strike = Strike(target, limits, duration=0.2)
    strike.start(controller)
    lowest = 1e9
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        done = strike.step(controller, 0.005)
        lowest = min(lowest, controller.pose().z)
        if done:
            break
        time.sleep(0.005)

    floor = target[2] - limits.press_depth
    assert lowest >= floor - 2e-3, \
        f"tool reached {lowest * 1e3:.1f} mm, under the {floor * 1e3:.1f} mm floor"
    # And it has to actually get there, or there is no band to judge in.
    assert lowest <= target[2] - 1e-3, \
        f"tool stopped at {lowest * 1e3:.1f} mm, never reaching under the plane"


def test_strike_does_not_finish_while_the_arm_is_still_moving(controller):
    """The bug the whole contact chain rested on.

    `Motion._complete` asks `controller.settled()`, which is true once the
    *commanded* setpoint stops changing. On the real arm that is true
    while the paddle is 16 mm above its floor and travelling at 0.18 m/s,
    and it goes on moving for another ~360 ms. Everything downstream --
    `pressing`, and through it every contact sensor -- was therefore
    reading an arm in flight and calling the gap a hand.

    A lagging arm is simulated here by feeding observe() a height that
    trails the command, because MockArm tracks its command exactly and so
    cannot reproduce the failure on its own.
    """
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.05])
    drive(Hover(target, limits, duration=0.4), controller)

    strike = Strike(target, limits, duration=0.2)
    strike.start(controller)
    # The arm "moves" for 0.5 s, well past the plan's 0.2 s but inside
    # settle_timeout, so this tests the stillness gate and not the backstop.
    moving_for = 0.5
    assert moving_for < strike.settle_timeout, "would test the timeout instead"
    t0 = time.perf_counter()
    moving_until = t0 + moving_for
    finished_at = None
    while time.perf_counter() - t0 < 3.0:
        now = time.perf_counter()
        # A steadily descending height while moving, then a fixed one.
        strike.observe(0.20 - 0.05 * min(now - t0, moving_for))
        if strike.step(controller, 0.005):
            finished_at = now - t0
            break
        assert not strike.pressing or now >= moving_until, (
            f"pressing at {(now - t0) * 1e3:.0f} ms, while the arm is still moving")
        time.sleep(0.005)

    assert finished_at is not None, "strike never finished"
    # It must have waited out the motion plus the stillness dwell, rather
    # than stopping when the plan ran out at 0.2 s.
    assert finished_at >= moving_for + Strike.STILL_DWELL, (
        f"finished at {finished_at * 1e3:.0f} ms; the arm was moving until "
        f"{moving_for * 1e3:.0f} ms")


def test_a_decelerating_arm_is_not_a_stopped_arm(controller):
    """The regression. A min-jerk plan ends at zero velocity by design, so
    near the bottom the arm is always crawling -- arrived or not. A short
    dwell reads that as stopped and ends the descent in mid-air.

    Measured consequence, on hardware, with a 60 ms dwell: dodges quit at
    17-21 mm instead of reaching the 11 mm floor, and hits quit at
    24-27 mm against a 28 mm hand instead of pressing into it. Everything
    ended early, so everything read as blocked.

    The creep rate here is the one from the bench trace: ~1 mm per 26 ms.
    """
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.05])
    drive(Hover(target, limits, duration=0.4), controller)

    strike = Strike(target, limits, duration=0.2)
    strike.start(controller)
    floor = target[2] - limits.press_depth
    # The real arm only crawls over the last centimetre or so; the swing
    # itself is fast. Starting the creep 15 mm up is what the bench trace
    # shows, and creeping the whole drop at this rate would just run out
    # settle_timeout and test the backstop instead.
    start_z = floor + 0.015
    creep = 0.001 / 0.026          # metres per second, from the trace

    t0 = time.perf_counter()
    ended_z = None
    while time.perf_counter() - t0 < 3.0:
        z = max(floor, start_z - creep * (time.perf_counter() - t0))
        strike.observe(z)
        if strike.step(controller, 0.005):
            ended_z = z
            break
        time.sleep(0.005)

    assert ended_z is not None, "strike never finished"
    assert ended_z <= floor + strike.ARRIVE_EPSILON, (
        f"descent ended at {ended_z * 1e3:.0f} mm with the floor at "
        f"{floor * 1e3:.0f} mm -- a crawling arm was called a stopped one")
    assert strike.ended_because == "arrived", strike.ended_because


def test_an_unobstructed_strike_ends_on_arrival_not_on_the_dwell(controller):
    """A clean dodge must not pay the stillness wait.

    Arrival is unambiguous where stillness is not, so it short-circuits.
    Without that every dodge would sit at the bottom for STILL_DWELL with
    the servos stalled, which is the current the supply has least of.
    """
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.05])
    drive(Hover(target, limits, duration=0.4), controller)

    strike = Strike(target, limits, duration=0.2)
    strike.start(controller)
    floor = target[2] - limits.press_depth
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        strike.observe(floor)          # already there
        if strike.step(controller, 0.005):
            break
        time.sleep(0.005)
    elapsed = time.perf_counter() - t0
    assert strike.ended_because == "arrived", strike.ended_because
    assert elapsed < 0.2 + Strike.STILL_DWELL + limits.press_hold, (
        f"took {elapsed * 1e3:.0f} ms; arrival should have skipped the dwell")


def test_strike_presses_as_soon_as_the_arm_stops(controller):
    """Stopped, not arrived. On a hit the paddle never arrives -- it stalls
    on the hand -- so waiting for arrival would hang every hit."""
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.05])
    drive(Hover(target, limits, duration=0.4), controller)

    strike = Strike(target, limits, duration=0.2)
    strike.start(controller)
    # Blocked well above the floor from the outset, as a hand would.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        strike.observe(0.09)
        if strike.step(controller, 0.005):
            break
        time.sleep(0.005)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2 + Strike.STILL_DWELL + limits.press_hold + 0.4, (
        f"a blocked strike took {elapsed * 1e3:.0f} ms; it should press and go, "
        "not wait out the settle_timeout")


def test_strike_still_completes_when_nobody_observes(controller):
    """Every caller that does not feed observe() keeps the old behaviour.

    Falling back matters more than it looks: waiting for a stillness that
    can never be established would hang each strike until settle_timeout,
    turning a silent improvement into a silent 0.75 s tax.
    """
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.05])
    drive(Hover(target, limits, duration=0.4), controller)
    strike = Strike(target, limits, duration=0.2)
    assert drive(strike, controller, limit=3.0), "strike never finished unobserved"


def test_strike_respects_max_drop(controller):
    """A caller asking for a huge strike gets a capped one."""
    limits = StrikeLimits(max_drop=0.05, hover_height=0.25)
    target = np.array([0.22, 0.0, 0.02])
    drive(Hover(target, limits, duration=0.5), controller)
    before = controller.pose().z
    drive(Strike(target, limits, duration=0.25), controller)
    assert (before - controller.pose().z) <= limits.max_drop + 5e-3


def test_strike_lowers_then_restores_torque_limit(controller):
    limits = StrikeLimits(torque_limit=300, normal_torque_limit=800)
    target = np.array([0.22, 0.0, 0.04])
    drive(Hover(target, limits, duration=0.3), controller)
    strike = Strike(target, limits, duration=0.2)
    strike.start(controller)
    assert controller.backend.diagnostics()["torque_limit"] == 300
    while not strike.step(controller, 0.005):
        time.sleep(0.005)
    assert controller.backend.diagnostics()["torque_limit"] == 800


def test_strike_restores_torque_limit_even_if_ik_fails(controller):
    limits = StrikeLimits()
    strike = Strike(np.array([9.0, 9.0, 9.0]), limits, duration=0.1)
    drive(strike, controller)
    assert controller.backend.diagnostics()["torque_limit"] == limits.normal_torque_limit


def test_feint_returns_to_where_it_started(controller):
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.03])
    drive(Hover(target, limits, duration=0.4), controller)
    before = controller.commanded.copy()
    assert drive(Feint(target, limits, out=0.08, back=0.12), controller)
    assert np.allclose(controller.commanded, before, atol=5e-3)


def test_feint_does_not_reach_the_target(controller):
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.03])
    drive(Hover(target, limits, duration=0.4), controller)
    feint = Feint(target, limits, fraction=0.4, out=0.08, back=0.12)
    feint.start(controller)
    lowest = 1e9
    while not feint.step(controller, 0.005):
        lowest = min(lowest, controller.pose().z)
        time.sleep(0.005)
    assert lowest > target[2] + 0.02, "a feint that lands is just a slow strike"


def test_sequence_runs_in_order(controller):
    limits = StrikeLimits()
    target = np.array([0.22, 0.0, 0.03])
    seq = Sequence([Hover(target, limits, 0.3), Hold(0.05),
                    Strike(target, limits, 0.2), Retract(model.HOME, limits, 0.3)])
    assert drive(seq, controller, limit=8.0)
    assert np.allclose(controller.commanded[:5], model.HOME, atol=1e-2)


def test_sequence_abort_stops_everything(controller):
    seq = Sequence([Hold(5.0), Hold(5.0)])
    seq.start(controller)
    seq.step(controller, 0.005)
    seq.abort()
    assert all(m.finished for m in seq.motions)


def test_motions_are_interruptible(controller):
    """A game must be able to abandon a motion mid-flight."""
    m = GoTo(model.HOME + 0.5, duration=3.0)
    m.start(controller)
    m.step(controller, 0.005)
    m.abort()
    assert m.step(controller, 0.005) is True


def test_hold_does_not_move(controller):
    before = controller.commanded.copy()
    assert drive(Hold(0.05), controller)
    assert np.allclose(controller.commanded, before)


# -- performance -----------------------------------------------------------

@pytest.fixture
def rig():
    """A controller with the real rig's limits rather than the defaults.

    Acceleration is what bounds a flourish, and the shipped configs allow
    35 rad/s^2 where SafetyLimits defaults to 8. Tuning a performance
    against the default would produce one nobody on the actual robot
    could see.
    """
    c = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])),
                      SafetyLimits(max_speed=3.5, max_accel=35.0, max_jerk=400.0),
                      control_hz=200.0)
    c.start()
    yield c
    c.backend.disconnect()


def test_every_flourish_ends_where_it_started(rig):
    """The property that makes it safe to do this for fun on a machine
    that also swings at people: joint space, no target, and an envelope
    that is zero at both ends, so however it is interrupted or replayed
    it cannot walk the arm toward the hand."""
    for name, move in FLOURISHES.items():
        start = rig.commanded.copy()
        assert drive(Flourish(move), rig), f"{name} never finished"
        drift = float(np.abs(rig.commanded - start).max())
        assert drift < 2e-3, f"{name} left the arm {drift:.4f} rad from where it began"


def test_every_flourish_stays_within_its_amplitudes(rig):
    """A wag is 9 degrees of shoulder, not 90. Joints that translate the
    tool are held timid on purpose; the ones that do not are where the
    performance lives."""
    for name, move in FLOURISHES.items():
        start = rig.commanded.copy()
        motion = Flourish(move)
        motion.start(rig)
        worst = np.zeros(6)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            done = motion.step(rig, 0.005)
            worst = np.maximum(worst, np.abs(rig.commanded - start))
            if done:
                break
            time.sleep(0.005)
        # Bounded overswing rather than none, and the bound is the point.
        # This asserted `worst <= amplitude` back when a flourish ran under
        # safety.max_accel, 35 rad/s^2, which was too low to chase the step
        # a missed control tick puts in the target -- so the profile
        # smoothed the jump away and never overshot. A flourish now carries
        # its own 400 rad/s^2, which is what makes it quick, and the same
        # jump gets followed: measured 1.35x on nod under loop jitter.
        # That is a real cost of the speed and not a bug, but a factor of
        # two would be, so this still catches an amplitude typo or a
        # runaway envelope.
        allowed = np.abs(np.asarray(move.amplitudes, float)) * 1.5 + 2e-3
        assert np.all(worst <= allowed), f"{name} overswung: {worst} > {allowed}"


def test_no_flourish_commands_past_the_motors_travel():
    """The limit that actually stops a gesture, and the one nothing checks.

    `model.JOINT_LIMITS` comes from the URDF and is far wider than the
    motors: it allows wrist_roll +-2.7 rad where the joint has 0.513 in
    total. `_write` clamps to the URDF, so a command outside the real
    travel is stopped by nothing in software -- it is issued in full, and
    the encoder reports the joint sitting where it started. That is what a
    1.90 rad spin was, and it read as 4 degrees at every speed and size.

    Checked at 1.5x amplitude to match the overswing jitter can add.
    """
    s = np.linspace(0.0, 1.0, 2000)
    home = np.concatenate([model.HOME, [0.0]])
    for name, move in FLOURISHES.items():
        offsets = move.offsets(s) * 1.5
        for i, joint in enumerate(JOINT_NAMES):
            if not move.amplitudes[i]:
                continue
            lo, hi = MEASURED_TRAVEL[joint]
            reach = home[i] + offsets[:, i]
            assert reach.min() >= lo, (
                f"{name} drives {joint} to {reach.min():+.3f}, past its {lo:+.3f} stop")
            assert reach.max() <= hi, (
                f"{name} drives {joint} to {reach.max():+.3f}, past its {hi:+.3f} stop")


def test_one_way_joints_get_one_way_swings():
    """A whole cycle is a sine and spends half its time going backwards.

    Two joints have almost nothing behind them -- wrist_roll starts 0.054
    rad off its lower stop and the gripper 0.179 off its closed one -- so
    on those a whole-cycle swing drives into a hard stop for half of every
    cycle. Measured, that is exactly what halved the old shimmy: 0.62 rad
    commanded on the roll, 0.22 reached.
    """
    home = np.concatenate([model.HOME, [0.0]])
    for name, move in FLOURISHES.items():
        cycles = np.broadcast_to(np.asarray(move.cycles, float),
                                 np.shape(move.amplitudes))
        for i, joint in enumerate(JOINT_NAMES):
            if not move.amplitudes[i]:
                continue
            lo, hi = MEASURED_TRAVEL[joint]
            behind = min(home[i] - lo, hi - home[i])
            # A rectified joint only ever goes one way, so having nothing
            # behind HOME costs it nothing -- that is the whole point of
            # `oneway`, and it is how the gripper chomps three times.
            rectified = np.broadcast_to(np.asarray(move.oneway, bool),
                                        np.shape(move.amplitudes))[i]
            if behind < 0.25 and not rectified and cycles[i] % 1.0 == 0.0:
                raise AssertionError(
                    f"{name} swings {joint} through whole cycles, but it has "
                    f"only {behind:.3f} rad on one side of HOME -- use a half "
                    f"cycle so the swing goes one way")


def test_no_flourish_can_reach_the_table(rig):
    """The guard that overswing actually threatens.

    `safety.min_height` does not cover any of this: `clamp_pose` bounds
    Cartesian commands and a flourish writes joint space, so nothing
    downstream is checking how close these get to the work surface. HOME
    is only 71 mm up and positive shoulder_lift, elbow_flex and wrist_flex
    all drive the tool *down*, which is how the old bow and droop came to
    hit the table.

    Checked at 1.5x amplitude, matching the overswing the test above
    allows, because the clearance has to survive the overshoot rather than
    just the nominal swing.
    """
    s = np.linspace(0.0, 1.0, 2000)
    for name, move in FLOURISHES.items():
        offsets = move.offsets(s) * 1.5
        lowest = min(model.tool_pose(model.HOME + o[:5]).xyz()[2]
                     for o in offsets[::10])
        assert lowest > 0.020, f"{name} reaches {lowest * 1e3:.0f} mm off the table"


def test_a_flourish_is_actually_visible(rig):
    """A taunt nobody can see is not a taunt.

    Asserted in absolute terms rather than as a fraction of the nominal
    amplitude, because the motion profile is entitled to round the peaks
    off and the question here is only whether an audience would notice.
    Three degrees would not do; this asks for rather more.
    """
    for name, move in FLOURISHES.items():
        moved = np.asarray(move.amplitudes) != 0
        start = rig.commanded.copy()
        motion = Flourish(move)
        motion.start(rig)
        peak = np.zeros(6)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            done = motion.step(rig, 0.005)
            peak = np.maximum(peak, np.abs(rig.commanded - start))
            if done:
                break
            time.sleep(0.005)
        assert peak[moved].max() > 0.08, f"{name} barely moved: {peak[moved].max():.3f} rad"


def test_moods_pick_from_their_own_repertoire():
    rng = np.random.default_rng(0)
    for mood in ("gloat", "sulk", "smug", "caught", "idle"):
        seen = {tuple(flourish(mood, rng=rng).move.amplitudes) for _ in range(40)}
        assert seen, mood
    # An unknown mood falls back rather than raising: a missing reaction
    # should cost a joke, not a round.
    assert flourish("triumphant-despair", rng=rng) is not None


def test_tip_offset_puts_the_paddle_where_the_geometry_means_it(controller):
    """The offset between the tool point and the end of the tool.

    Every height in this system is written in tool-point coordinates and
    every one of them is about the paddle, so a tool with any length at
    all breaks the equivalence. Measured on the rig: the paddle hangs
    69 mm below the tool point, so a floor of 13 mm was commanding the tip
    56 mm underground. Every strike bottomed out on the table, and the
    encoders -- which read the servo shaft, not the flexing link -- went on
    reporting that it had stopped where it was asked to.
    """
    limits = StrikeLimits(tip_offset=0.069, press_depth=0.010)
    hand = np.array([0.24, 0.0, 0.030])

    # Hover clearance is for the tip, so the tool sits a tip higher again.
    drive(Hover(hand, limits, duration=0.4), controller)
    tool_z = controller.pose().z
    assert abs((tool_z - limits.tip_offset) - (hand[2] + limits.hover_height)) < 4e-3, (
        f"tip hovers at {(tool_z - limits.tip_offset) * 1e3:.0f} mm, wanted "
        f"{(hand[2] + limits.hover_height) * 1e3:.0f}")

    # And the floor puts the tip press_depth below the hand, not below the
    # table. This is the number that was 56 mm underground.
    strike = Strike(hand, limits, duration=0.2)
    strike.start(controller)
    lowest = 1e9
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        done = strike.step(controller, 0.005)
        lowest = min(lowest, controller.pose().z)
        if done:
            break
        time.sleep(0.005)
    tip_floor = lowest - limits.tip_offset
    assert tip_floor > 0.0, f"tip driven {abs(tip_floor) * 1e3:.0f} mm below the table"
    want = hand[2] - limits.press_depth
    assert abs(tip_floor - want) < 3e-3, (
        f"tip floor {tip_floor * 1e3:.1f} mm, wanted {want * 1e3:.1f}")


def test_a_bare_gripper_is_unaffected_by_the_offset(controller):
    """tip_offset defaults to 0, so nothing without a paddle changes."""
    plain = StrikeLimits()
    assert plain.tip_offset == 0.0
    hand = np.array([0.24, 0.0, 0.030])
    drive(Hover(hand, plain, duration=0.4), controller)
    assert abs(controller.pose().z - (hand[2] + plain.hover_height)) < 4e-3


def test_min_height_has_to_cover_the_tool_it_is_guarding():
    """`safety.min_height` is a tool-point floor, so a tool that hangs
    below the tool point needs it raised by that much or the guard permits
    the tip through the table -- which is what drove the gripper into the
    wood at a commanded 20 mm."""
    from tlod.config import Config

    cfg = Config.load("configs/opi.yaml")
    tip = cfg.arm.tip_offset
    assert cfg.safety.min_height >= tip, (
        f"min_height {cfg.safety.min_height * 1e3:.0f} mm allows the tip "
        f"{(tip - cfg.safety.min_height) * 1e3:.0f} mm below the table")

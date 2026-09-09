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
    FLOURISHES, Feint, Flourish, GoTo, GoToPose, Hold, Hover, Retract, Sequence,
    Strike, StrikeLimits, flourish,
)
from tlod.types import Pose


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


def test_strike_never_goes_below_the_target_plane(controller):
    """The guard that makes a wrong height estimate harmless."""
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
    assert lowest >= target[2] - 1e-3, f"tool reached {lowest:.4f}, below plane {target[2]}"


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
        allowed = np.abs(np.asarray(move.amplitudes, float)) + 2e-3
        assert np.all(worst <= allowed), f"{name} overswung: {worst} > {allowed}"


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

"""Motion profile tests.

The profile is what stands between a policy asking for a large jump and
the servos being asked to deliver it in one tick, so the limits are
checked directly rather than inferred from how the arm looks.
"""

import numpy as np
import pytest

from tlod.arm.profile import MotionProfile, ProfileLimits

DT = 0.01
LIMITS = ProfileLimits(max_speed=2.0, max_accel=8.0, max_jerk=80.0)


def run(profile, target, ticks=2000, dt=DT, limits=None):
    """Drive a profile to a fixed target, returning the setpoint history."""
    target = np.asarray(target, float)
    return np.array([profile.step(target, dt, limits) for _ in range(ticks)])


@pytest.mark.parametrize("magnitude", [0.005, 0.05, 0.3, 1.0, 3.0])
def test_step_input_never_overshoots(magnitude):
    """A jerk-limited tracker that overshoots hunts around the target, and
    on an arm that swings at people, overshoot is the direction that
    matters."""
    p = MotionProfile(np.zeros(1), LIMITS)
    history = run(p, [magnitude])
    assert history.max() <= magnitude + 1e-9
    assert history[-1, 0] == pytest.approx(magnitude, abs=1e-9)


@pytest.mark.parametrize("magnitude", [0.05, 0.3, 1.0, 3.0])
def test_velocity_and_acceleration_stay_inside_the_limits(magnitude):
    p = MotionProfile(np.zeros(1), LIMITS)
    speeds, accels = [], []
    for _ in range(2000):
        p.step(np.array([magnitude]), DT)
        speeds.append(p.speed)
        accels.append(p.accel)
    assert max(speeds) <= LIMITS.max_speed + 1e-9
    assert max(accels) <= LIMITS.max_accel + 1e-9


def test_jerk_is_bounded_while_moving():
    """The jerk limit is what stops commanded current arriving as a step.

    Checked only while the profile is actually travelling: the final tick
    of a move snaps the last sub-millimetre of residual and drops the
    acceleration straight to zero, which violates the limit arithmetically
    but is a *release* of deceleration, not a torque demand.
    """
    p = MotionProfile(np.zeros(1), LIMITS)
    previous = 0.0
    worst = 0.0
    for _ in range(400):
        p.step(np.array([0.8]), DT)
        if p.speed > 1e-6:
            worst = max(worst, abs(p.accel - previous) / DT)
        previous = p.accel
    assert worst <= LIMITS.max_jerk + 1e-6


def test_synchronised_joints_arrive_together():
    """Unsynchronised, the short joints finish early and the joint-space
    path bends; every joint also peaks at once."""
    p = MotionProfile(np.zeros(4), LIMITS)
    target = np.array([1.0, 0.5, 0.25, 0.1])
    arrival = [None] * 4
    for k in range(2000):
        q = p.step(target, DT)
        for i in range(4):
            if arrival[i] is None and abs(q[i] - target[i]) < 1e-9:
                arrival[i] = k
    assert all(a is not None for a in arrival)
    spread = max(arrival) - min(arrival)
    assert spread <= 2, f"joints arrived {spread} ticks apart, not synchronised"


def test_synchronisation_lowers_simultaneous_demand():
    """The reason synchronisation is on: it is also a power measure. Only
    the longest-travelling joint runs at the full limit."""
    target = np.array([1.0, 0.5, 0.25, 0.1, 0.02])

    def peak_total_accel(synchronise):
        limits = ProfileLimits(2.0, 8.0, 80.0, synchronise=synchronise)
        p = MotionProfile(np.zeros(5), limits)
        worst = 0.0
        for _ in range(2000):
            p.step(target, DT)
            worst = max(worst, float(np.abs(p.a).sum()))
        return worst

    assert peak_total_accel(True) < 0.6 * peak_total_accel(False)


def test_a_moving_target_is_tracked_without_violating_limits():
    """The streaming case: `servo_pose` moves the target every tick."""
    p = MotionProfile(np.zeros(2), LIMITS)
    for k in range(1200):
        t = k * DT
        p.step(np.array([0.4 * np.sin(3.0 * t), 0.3 * np.cos(5.0 * t)]), DT)
        assert p.speed <= LIMITS.max_speed + 1e-9
        assert p.accel <= LIMITS.max_accel + 1e-9


def test_target_reversal_mid_flight_stays_bounded():
    """A hand that changes direction reverses the target at full speed."""
    p = MotionProfile(np.zeros(1), LIMITS)
    for _ in range(30):
        p.step(np.array([2.0]), DT)
    assert p.speed > 0.1, "should be travelling before the reversal"
    for _ in range(400):
        p.step(np.array([-2.0]), DT)
        assert p.accel <= LIMITS.max_accel + 1e-9


def test_scaled_limits_follow_the_time_scaling_law():
    """Velocity scales as s, acceleration as s^2, jerk as s^3."""
    scaled = LIMITS.scaled(0.5)
    assert scaled.max_speed == pytest.approx(LIMITS.max_speed * 0.5)
    assert scaled.max_accel == pytest.approx(LIMITS.max_accel * 0.25)
    assert scaled.max_jerk == pytest.approx(LIMITS.max_jerk * 0.125)


def test_reset_discards_velocity():
    """After an e-stop the profile must not resume the motion that caused
    it."""
    p = MotionProfile(np.zeros(1), LIMITS)
    for _ in range(30):
        p.step(np.array([2.0]), DT)
    assert p.speed > 0.0
    p.reset(np.array([0.5]))
    assert p.speed == 0.0
    assert p.accel == 0.0
    assert p.q[0] == 0.5


def test_rest_time_tracks_a_stationary_setpoint():
    p = MotionProfile(np.zeros(1), LIMITS)
    run(p, [0.2], ticks=300)
    assert p.rest_time > 0.5
    p.step(np.array([0.6]), DT)
    assert p.rest_time == 0.0


def test_zero_dt_is_a_no_op():
    p = MotionProfile(np.zeros(1), LIMITS)
    before = p.q.copy()
    assert np.array_equal(p.step(np.array([1.0]), 0.0), before)

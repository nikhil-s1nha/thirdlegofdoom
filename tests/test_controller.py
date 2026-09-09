"""Safety and motion tests.

This robot is designed to move quickly toward a human hand. These tests
exist because the guards are the only thing standing between a bug and
a bruise, so they are checked directly rather than assumed.
"""

import numpy as np
import pytest

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits, minimum_jerk
from tlod.arm.mock import MockArm
from tlod.types import Pose


@pytest.fixture
def controller():
    c = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])), control_hz=200.0)
    c.start()
    yield c
    c.backend.disconnect()


def test_clamp_pose_enforces_table():
    lim = SafetyLimits()
    safe, hits = lim.clamp_pose(Pose(0.22, 0.0, -0.5))
    assert safe.z >= lim.table_z + lim.min_height
    assert "min_height" in hits


def test_clamp_pose_enforces_reach():
    lim = SafetyLimits()
    safe, hits = lim.clamp_pose(Pose(2.0, 0.0, 0.1))
    assert np.hypot(safe.x, safe.y) <= lim.max_radius + 1e-9
    assert "max_radius" in hits


def test_clamp_pose_pushes_out_of_the_base():
    lim = SafetyLimits()
    safe, hits = lim.clamp_pose(Pose(0.01, 0.0, 0.1))
    assert np.hypot(safe.x, safe.y) >= lim.min_radius - 1e-9
    assert "min_radius" in hits


def test_clamp_pose_at_origin_is_defined():
    """Degenerate input must not divide by zero."""
    safe, _ = SafetyLimits().clamp_pose(Pose(0.0, 0.0, 0.1))
    assert np.isfinite([safe.x, safe.y, safe.z]).all()
    assert np.hypot(safe.x, safe.y) > 0


def test_valid_pose_passes_through_untouched():
    safe, hits = SafetyLimits().clamp_pose(Pose(0.22, 0.05, 0.12))
    assert hits == []
    assert (safe.x, safe.y, safe.z) == (0.22, 0.05, 0.12)


def test_goto_pose_reaches_target(controller):
    assert controller.goto_pose(Pose(0.22, 0.05, 0.12), duration=0.5)
    got = controller.pose().xyz()
    assert np.linalg.norm(got - np.array([0.22, 0.05, 0.12])) < 2e-3


def test_commands_respect_joint_margin(controller):
    controller.goto_joints(model.JOINT_LIMITS[:, 1] + 1.0, duration=0.3)
    q = controller.commanded[:5]
    assert model.within_limits(q, margin=controller.limits.joint_margin - 1e-6)


def test_speed_limit_is_enforced(controller):
    """A single tick must never command more than max_speed * dt."""
    before = controller.commanded.copy()
    far = model.HOME + np.array([1.5, 1.0, -1.0, 0.8, 1.2])
    controller._write(np.concatenate([far, [0.0]]), max_speed=1.0, dt=0.01)
    step = np.abs(controller.commanded - before).max()
    assert step <= 1.0 * 0.01 + 1e-9, f"stepped {step} rad in one tick"


def test_acceleration_limit_is_enforced(controller):
    """The limit that decides motor current, and so whether the supply
    copes. A velocity cap alone permits a standing start to full speed in
    one tick, which is what the old rate clamp did."""
    far = np.concatenate([model.HOME + 1.0, [0.0]])
    for _ in range(50):
        controller._write(far, dt=0.01)
    assert controller.profile.accel <= controller.limits.max_accel + 1e-9
    assert controller.stats.peak_accel <= controller.limits.max_accel + 1e-9


def test_a_late_tick_cannot_authorise_a_bigger_step(controller):
    """Every limit is applied as `limit * dt`, so an unbounded dt is an
    unbounded step -- and dt blows up exactly when the control thread has
    already stalled, which is the worst moment to permit a lurch."""
    far = np.concatenate([model.HOME + 2.0, [0.0]])
    before = controller.commanded.copy()
    controller._write(far, dt=5.0)
    step = np.abs(controller.commanded - before).max()
    ceiling = controller.limits.max_speed * controller.limits.max_tick_dt
    assert step <= ceiling + 1e-9, f"a 5 s tick moved {step:.4f} rad"


def test_derate_scales_the_limits(controller):
    """Time-scaling: velocity by s, acceleration by s squared."""
    controller.set_derate(0.5)
    far = np.concatenate([model.HOME + 1.0, [0.0]])
    for _ in range(200):
        controller._write(far, dt=0.01)
    assert controller.profile.speed <= 0.5 * controller.limits.max_speed + 1e-9
    assert controller.profile.accel <= 0.25 * controller.limits.max_accel + 1e-9


def test_governor_slows_the_arm_on_a_small_supply():
    """The whole point: an arm that is too heavy for its brick should move
    slowly rather than brown out."""
    from tlod.arm.power import PowerBudget, PowerGovernor, PowerModel

    def peak_speed(supply):
        governor = (None if supply is None else
                    PowerGovernor(PowerModel(budget=PowerBudget(supply_current=supply))))
        c = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])),
                          control_hz=100.0, governor=governor)
        c.start()
        far = np.concatenate([model.HOME + 1.0, [0.0]])
        worst = 0.0
        for _ in range(400):
            c._write(far, dt=0.01)
            worst = max(worst, c.profile.speed)
        c.backend.disconnect()
        return worst

    ungoverned = peak_speed(None)
    starved = peak_speed(2.0)
    ample = peak_speed(10.0)
    assert starved < 0.7 * ungoverned, "a 2 A supply should force a slowdown"
    assert ample == pytest.approx(ungoverned, rel=0.05), "10 A needs no slowdown"


def test_estop_clears_the_profile_velocity(controller):
    """Releasing an e-stop must not resume the swing that triggered it."""
    far = np.concatenate([model.HOME + 1.0, [0.0]])
    for _ in range(40):
        controller._write(far, dt=0.01)
    assert controller.profile.speed > 0.1
    controller.estop()
    assert controller.profile.speed == 0.0
    controller.release_estop()
    assert controller.profile.speed == 0.0


def test_settled_waits_for_the_arm_not_just_the_command(controller):
    """`settled` gates every blocking move, so it has to mean the servo
    got there, not that the setpoint stopped changing."""
    target = np.concatenate([model.HOME + 0.05, [0.0]])
    assert not controller.settled()
    for _ in range(400):
        controller._write(target, dt=0.005)
    assert controller.settled()
    assert np.allclose(controller.commanded, target, atol=1e-6)


def test_estop_freezes_and_blocks(controller):
    controller.estop()
    frozen = controller.commanded.copy()
    controller.servo_pose(Pose(0.3, 0.1, 0.2))
    controller.goto_joints(model.HOME + 0.5, duration=0.1)
    assert np.allclose(controller.commanded, frozen), "e-stop must block all motion"
    assert controller.estopped


def test_estop_holds_torque_rather_than_dropping(controller):
    """A limp arm falls, possibly onto the hand that triggered the stop."""
    controller.estop()
    assert controller.backend.diagnostics()["torque"] is True


def test_release_estop_restores_motion(controller):
    controller.estop()
    controller.release_estop()
    assert not controller.estopped
    assert controller.goto_pose(Pose(0.22, 0.0, 0.12), duration=0.3)


def test_gripper_maps_zero_to_one(controller):
    """Endpoints map to the joint limits -- but only after enough ticks.

    set_gripper is rate-limited like every other command, so it slews to
    the target rather than jumping. A gripper that slams shut is a broken
    gripper, and on a robot that plays with hands it is worse than that.
    """
    for _ in range(400):
        controller.set_gripper(0.0)
    assert np.isclose(controller.commanded[5], model.GRIPPER_LIMITS[0], atol=1e-3)
    for _ in range(400):
        controller.set_gripper(1.0)
    assert np.isclose(controller.commanded[5], model.GRIPPER_LIMITS[1], atol=1e-3)


def test_gripper_is_rate_limited(controller):
    """One call must not traverse the whole range."""
    controller.set_gripper(0.0)
    start = controller.commanded[5]
    controller.set_gripper(1.0)
    span = model.GRIPPER_LIMITS[1] - model.GRIPPER_LIMITS[0]
    assert abs(controller.commanded[5] - start) < span / 2


def test_gripper_clamps_out_of_range(controller):
    controller.set_gripper(99.0)
    assert controller.commanded[5] <= model.GRIPPER_LIMITS[1] + 1e-9


def test_servo_pose_declines_unreachable_targets(controller):
    assert controller.servo_pose(Pose(5.0, 5.0, 5.0)) is False


def test_minimum_jerk_endpoints_and_monotonicity():
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    assert minimum_jerk(-5.0) == 0.0 and minimum_jerk(5.0) == 1.0
    xs = np.linspace(0, 1, 50)
    ys = [minimum_jerk(x) for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:], strict=False)), "must not reverse"
    # Zero velocity at both ends is the point of min-jerk.
    assert minimum_jerk(0.01) < 0.001
    assert minimum_jerk(0.99) > 0.999


def test_watchdog_detects_stale_commands(controller):
    controller.limits.command_timeout = 0.0
    assert controller.check_watchdog()

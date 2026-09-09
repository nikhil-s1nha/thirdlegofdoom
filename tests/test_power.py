"""Power model tests.

The model decides how fast the arm is allowed to move on a given supply,
so the checks here are mostly against physics that can be worked out by
hand, and against the servo's own spec sheet. A power budget that is
quietly wrong is worse than none: it would license exactly the motions
that brown the arm out.
"""

import numpy as np
import pytest

from tlod.arm import model
from tlod.arm.power import (
    GRAVITY, PowerBudget, PowerGovernor, PowerModel, ServoModel, _COM, _MASS,
)
from tlod.arm.profile import ProfileLimits

KGCM = 0.0980665  # N.m per kg.cm, for comparing against the spec sheet


def test_stall_point_is_reproduced():
    """The end of the curve a current limit actually cares about."""
    servo = ServoModel()
    current = servo.current(np.array([servo.stall_torque]), np.array([0.0]))
    assert float(current[0]) == pytest.approx(servo.stall_current, rel=0.03)


def test_rated_point_is_within_the_spec_sheet_s_own_disagreement():
    """The manufacturer's three operating points are not mutually
    consistent; the model is fitted to stall and runs about 20% high at
    the rated point. Pinned so the size of that gap cannot grow unnoticed
    -- it is the reason the model is calibratable."""
    check = ServoModel().check()
    assert check["rated_a"] == pytest.approx(0.9, rel=0.25)
    assert check["rated_a"] > 0.9, "expected the model to be the conservative one"


def test_gravity_torque_matches_a_hand_calculation():
    """Arm held horizontal: the shoulder carries the whole arm's weight.

    Gravity is a uniform field, so the moment of five separate weights is
    exactly the moment of their combined mass at their combined centre --
    which is a calculation short enough to do independently of the model's
    per-link loop.
    """
    q = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0])
    frames = model.fk_all(q)
    coms = [f[:3, :3] @ _COM[k] + f[:3, 3] for k, f in enumerate(frames[:5])]
    # Links outboard of shoulder_lift only. The shoulder link itself sits
    # inboard of this joint and is carried by the base, not by this motor.
    outboard = slice(1, 5)
    total = sum(_MASS[outboard])
    centre = sum(m * c for m, c in zip(_MASS[outboard], coms[outboard], strict=True)) / total
    axis = frames[1][:3, 2]
    by_hand = abs(float(axis @ np.cross(centre - frames[1][:3, 3], total * GRAVITY)))

    assert abs(PowerModel().gravity_torque(q)[1]) == pytest.approx(by_hand, rel=1e-9)
    # ... and that it is the right order of magnitude: getting on for half
    # a kilogram on a 10 cm lever, about a third of one servo's rated
    # torque and a seventh of its stall torque.
    assert 0.3 < by_hand < 0.6


def test_vertical_axes_carry_no_gravity_load():
    """shoulder_pan is a yaw axis, so gravity has no moment about it in
    any configuration. Not exactly zero: the URDF writes its right angles
    as 3.14159 rather than pi, which tilts the axis by about a
    microradian."""
    for q in (np.zeros(5), model.HOME, np.array([0.5, -0.3, 0.7, 0.2, 1.0])):
        assert abs(PowerModel().gravity_torque(q)[0]) < 1e-5


def test_effective_inertia_is_positive_and_decreases_outboard():
    """Each joint carries everything beyond it, so the wrist sees least."""
    inertia = PowerModel().effective_inertia(model.HOME)
    assert (inertia > 0).all()
    assert inertia[4] < inertia[3] < inertia[2]


def test_inertia_of_the_last_joint_is_just_the_gripper():
    """wrist_roll turns the gripper about its own axis and nothing else,
    so its inertia should be tiny and configuration-independent."""
    a = PowerModel().effective_inertia(model.HOME)[4]
    b = PowerModel().effective_inertia(np.array([1.0, -0.2, 0.4, -0.6, 0.3]))[4]
    assert a == pytest.approx(b, rel=1e-6)
    assert a < 1e-4


def test_current_rises_with_acceleration_in_both_directions():
    """A budget must not be cheaper for accelerating the way gravity pulls;
    the limit has to hold for the return trip too."""
    pm = PowerModel()
    q = model.HOME
    up = pm.total_current(q, np.full(5, 10.0))
    down = pm.total_current(q, np.full(5, -10.0))
    still = pm.total_current(q)
    assert up == pytest.approx(down)
    assert up > still


def test_moving_joints_cost_more_than_holding_ones():
    """180 mA of running friction per servo is most of a 2 A budget once
    six of them turn, and is the reason multi-joint moves are the ones
    that fail."""
    pm = PowerModel()
    holding = pm.total_current(model.HOME, speed=np.zeros(5))
    turning = pm.total_current(model.HOME, speed=np.full(5, 1.0))
    assert turning - holding == pytest.approx(5 * 0.15, abs=0.02)


def test_feasible_scale_brings_a_move_inside_the_budget():
    pm = PowerModel(budget=PowerBudget(supply_current=2.0))
    q = model.HOME
    accel, speed = np.full(5, 20.0), np.full(5, 1.5)
    assert pm.total_current(q, accel, speed) > pm.budget.limit

    s = pm.feasible_scale(q, accel, speed)
    assert 0.0 < s < 1.0
    assert pm.total_current(q, accel * s * s, speed * s) <= pm.budget.limit + 1e-6


def test_feasible_scale_is_one_when_there_is_headroom():
    pm = PowerModel(budget=PowerBudget(supply_current=10.0))
    assert pm.feasible_scale(model.HOME, np.full(5, 8.0), np.full(5, 1.0)) == 1.0


def test_feasible_scale_bottoms_out_when_gravity_alone_is_too_much():
    """Slowing down cannot reduce the cost of holding the arm up, so the
    model must report a floor rather than pretend a slower move fits."""
    pm = PowerModel(budget=PowerBudget(supply_current=0.4))
    assert not pm.supports_holding(model.HOME)
    assert pm.feasible_scale(model.HOME, np.full(5, 8.0), np.full(5, 1.0)) == pytest.approx(0.1)


def test_a_two_amp_supply_cannot_run_five_joints_at_the_default_limits():
    """The headline number: this is why the arm works one joint at a time
    and not otherwise."""
    pm = PowerModel(budget=PowerBudget(supply_current=2.0))
    scale = pm.worst_case_scale(model.HOME, ProfileLimits())
    assert scale < 0.6

    generous = PowerModel(budget=PowerBudget(supply_current=5.0))
    assert generous.worst_case_scale(model.HOME, ProfileLimits()) == 1.0


class TestGovernor:
    def test_derate_settles_on_the_feasible_scale(self):
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=2.0)))
        limits = ProfileLimits()
        for _ in range(2000):
            scale = gov.update(model.HOME, limits, 0.01)
        expected = gov.model.worst_case_scale(model.HOME, limits)
        assert scale == pytest.approx(expected, abs=1e-3)

    def test_a_healthy_rail_does_not_derate(self):
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=10.0)))
        for _ in range(500):
            gov.note_voltage(12.0)
            scale = gov.update(model.HOME, ProfileLimits(), 0.01)
        assert scale == pytest.approx(1.0, abs=1e-3)

    def test_a_sagging_rail_derates(self):
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=10.0)))
        gov.note_voltage(9.5)
        assert gov.voltage_scale < 1.0
        assert gov.sag_events == 1
        for _ in range(200):
            scale = gov.update(model.HOME, ProfileLimits(), 0.01)
        assert scale < 1.0

    def test_recovery_is_slower_than_the_cut(self):
        """A supply that has just recovered is the one that sags again."""
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=10.0)),
                            cut_time=0.1, recover_time=3.0)
        gov.note_voltage(8.0)
        gov.update(model.HOME, ProfileLimits(), 0.2)
        dropped = gov.scale
        assert dropped < 0.5

        for _ in range(20):
            gov.note_voltage(12.0)
        gov.update(model.HOME, ProfileLimits(), 0.2)
        assert gov.scale < dropped + 0.2, "recovered far too quickly"


class TestGovernorCost:
    """The governor has to fit inside a control tick.

    Measured on a Raspberry Pi 5: worst_case_scale is ~11.9 ms, against a
    20 ms budget at 50 Hz, and more than policy, IK and the servo bus put
    together. It ran every tick and drove the control loop to 96.7%
    overruns and a 4.5 s command latency.
    """

    def test_the_pose_term_is_not_recomputed_every_tick(self):
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=2.0)),
                            pose_interval=0.2)
        for _ in range(100):                      # 1 s at 100 Hz
            gov.update(model.HOME, ProfileLimits(), 0.01)
        assert gov.pose_evaluations <= 6, "still recomputing at control rate"

    def test_caching_does_not_change_where_the_derate_settles(self):
        limits = ProfileLimits()
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=2.0)),
                            pose_interval=0.2)
        for _ in range(2000):
            scale = gov.update(model.HOME, limits, 0.01)
        assert scale == pytest.approx(gov.model.worst_case_scale(model.HOME, limits),
                                      abs=1e-3)

    def test_a_sagging_rail_is_answered_between_pose_evaluations(self):
        """The voltage term is the one that must not wait for the cache:
        it arrives from HealthMonitor asynchronously, and delaying it is
        the failure the governor exists to prevent."""
        gov = PowerGovernor(PowerModel(budget=PowerBudget(supply_current=10.0)),
                            pose_interval=10.0)
        gov.update(model.HOME, ProfileLimits(), 0.01)
        before = gov.pose_evaluations
        gov.note_voltage(8.0)
        for _ in range(20):
            scale = gov.update(model.HOME, ProfileLimits(), 0.01)
        assert scale < 0.9, "sag ignored until the next pose evaluation"
        assert gov.pose_evaluations == before, "test did not exercise the cache"

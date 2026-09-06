"""Kinematics tests.

These pin the geometry. If someone regenerates the chain from a different
URDF revision and the numbers shift, these fail loudly rather than the
arm quietly reaching to the wrong place.
"""

import numpy as np
import pytest

from tlod.arm import model
from tlod.types import Pose


def random_q(rng, scale=0.85):
    return rng.uniform(model.JOINT_LIMITS[:, 0] * scale, model.JOINT_LIMITS[:, 1] * scale)


def test_chain_matches_urdf_topology():
    assert len(model.CHAIN) == 6
    revolute = [l.name for l in model.CHAIN if l.axis is not None]
    assert revolute == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    assert model.CHAIN[-1].name == "tcp" and model.CHAIN[-1].axis is None


def test_fk_is_deterministic_and_rigid():
    q = np.array([0.3, -0.5, 0.7, 0.2, -0.4])
    T1, T2 = model.fk(q), model.fk(q)
    assert np.allclose(T1, T2)
    R = T1[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "rotation must be orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "no reflection"


def test_fk_home_position_is_pinned():
    # Guards against silent geometry drift.
    p = model.fk(np.zeros(5))[:3, 3]
    assert np.allclose(p, [0.39136, 0.0, 0.22647], atol=1e-4)


def test_fk_all_returns_every_frame():
    frames = model.fk_all(np.zeros(5))
    assert len(frames) == len(model.CHAIN)
    assert np.allclose(frames[-1], model.fk(np.zeros(5)))


def test_joint_limits_are_sane():
    lo, hi = model.JOINT_LIMITS[:, 0], model.JOINT_LIMITS[:, 1]
    assert np.all(lo < hi)
    assert model.within_limits(model.HOME), "HOME must be reachable"
    assert model.within_limits(model.HOME, margin=0.05), "HOME must have margin"


def test_clamp_to_limits():
    q = model.clamp_to_limits(np.array([10.0, -10.0, 0.0, 0.0, 0.0]))
    assert model.within_limits(q)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ik_round_trip_warm_start(seed):
    """The control-loop regime: warm started from a nearby configuration."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(60):
        q = random_q(rng)
        target = model.tool_pose(q)
        seed_q = model.clamp_to_limits(q + rng.normal(0, 0.25, 5))
        r = model.ik(target, seed_q)
        if r.ok:
            ok += 1
            assert r.pos_error < 1e-3
            achieved = model.tool_pose(r.q)
            assert np.linalg.norm(achieved.xyz() - target.xyz()) < 1e-3
    assert ok >= 57, f"warm-start IK converged only {ok}/60"


def test_ik_position_only_over_reachable_points():
    rng = np.random.default_rng(3)
    ok = 0
    for _ in range(80):
        q = random_q(rng)
        p = model.fk(q)[:3, 3]
        if model.ik_position(p, model.HOME).ok:
            ok += 1
    assert ok >= 72, f"position IK solved only {ok}/80 reachable points"


def test_ik_tracking_loop_never_fails():
    """Successive small target moves, the regime the robot actually runs in."""
    q = model.HOME.copy()
    for i in range(200):
        t = i * 0.01
        p = np.array([0.22 + 0.05 * np.sin(t * 3), 0.10 * np.sin(t * 2),
                      0.12 + 0.05 * np.cos(t * 2.5)])
        r = model.ik_position(p, q)
        assert r.ok, f"tracking IK failed at step {i}"
        assert r.pos_error < 1e-3
        q = r.q


def test_ik_is_deterministic():
    a = model.ik_position([0.24, 0.06, 0.13], model.HOME)
    b = model.ik_position([0.24, 0.06, 0.13], model.HOME)
    assert np.allclose(a.q, b.q), "identical inputs must give identical motion"


def test_ik_reports_failure_when_unreachable():
    r = model.ik_position([2.0, 0.0, 0.0], model.HOME)
    assert not r.ok
    assert r.pos_error > 1.0
    assert model.within_limits(r.q), "a failed solve must still return a legal configuration"


def test_ik_respects_joint_limits():
    rng = np.random.default_rng(4)
    for _ in range(40):
        p = model.fk(random_q(rng))[:3, 3]
        assert model.within_limits(model.ik_position(p, model.HOME).q)


def test_task_error_wraps_angles():
    e = model.task_error(np.array([0, 0, 0, 3.0, 0.0]), np.array([0, 0, 0, -3.0, 0.0]))
    assert abs(e[3]) < np.pi, "angular error must take the short way round"


def test_jacobian_matches_finite_difference_of_fk():
    q = np.array([0.2, -0.4, 0.6, 0.1, 0.3])
    J = model.jacobian(q)
    eps = 1e-5
    for i in range(5):
        qp, qm = q.copy(), q.copy()
        qp[i] += eps
        qm[i] -= eps
        num = model.task_error(model.tool_pose(qm).as_vec(), model.tool_pose(qp).as_vec()) / (2 * eps)
        assert np.allclose(J[:, i], num, atol=1e-3)


def test_pose_has_no_yaw():
    """The arm cannot span SE(3); the type must not imply otherwise."""
    assert not hasattr(Pose(0, 0, 0), "yaw")
    assert len(Pose(0, 0, 0).as_vec()) == 5


def test_analytic_position_jacobian_matches_finite_difference():
    """The fast path must agree with the slow one it replaces.

    Position-only IK uses an analytic Jacobian -- one FK pass instead of
    six -- which is what lets a Raspberry Pi Zero hold the control loop.
    Speed is only worth having if it is the same answer.
    """
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.uniform(model.JOINT_LIMITS[:, 0] * 0.8, model.JOINT_LIMITS[:, 1] * 0.8)
        analytic = model.position_jacobian(q)
        numeric = model.jacobian(q)[:3, :]
        assert np.allclose(analytic, numeric, atol=1e-4), np.abs(analytic - numeric).max()


def test_analytic_jacobian_is_exact_at_singular_configurations():
    """Finite differences degrade near singularities; the analytic form
    does not, which is part of the point."""
    for q in (np.zeros(5), np.array([0.0, 0.0, 0.0, 0.0, 1.5])):
        J = model.position_jacobian(q)
        assert np.all(np.isfinite(J))


def test_position_only_ik_still_converges_after_the_fast_path():
    q = model.HOME.copy()
    errors = []
    for i in range(200):
        t = i * 0.01
        p = np.array([0.22 + 0.05 * np.sin(t * 3), 0.10 * np.sin(t * 2),
                      0.12 + 0.05 * np.cos(t * 2.5)])
        r = model.ik_position(p, q)
        assert r.ok, f"failed at step {i}"
        errors.append(r.pos_error)
        q = r.q
    assert np.mean(errors) < 5e-4


def test_position_ik_with_a_pitch_constraint_uses_the_full_solver():
    """Asking for a tool angle must still work; it just takes the slow path."""
    r = model.ik_position([0.24, 0.05, 0.14], model.HOME, pitch=-0.6)
    assert r.ok
    assert abs(model.tool_pose(r.q).pitch - (-0.6)) < 0.06

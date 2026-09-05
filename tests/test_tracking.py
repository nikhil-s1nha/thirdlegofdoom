"""Prediction tests.

The central claim of the project is that prediction beats reaction given
the latency budget. That claim is tested here numerically, not asserted
in a comment.
"""

import numpy as np

from tlod.vision.tracking import ConstantVelocityFilter, MultiTracker

LATENCY = 0.30  # the sense-to-motion budget we must overcome


def test_filter_converges_on_constant_velocity():
    v = np.array([0.6, -0.4, 0.1])
    p0 = np.array([0.2, 0.1, 0.1])
    f = ConstantVelocityFilter(p0, 0.0)
    for k in range(1, 90):
        t = k / 60
        f.update(p0 + v * t, t)
    assert np.allclose(f.velocity, v, atol=0.12)
    assert np.allclose(f.position, p0 + v * (89 / 60), atol=0.02)


def test_prediction_beats_reaction():
    """The whole thesis, as a number."""
    rng = np.random.default_rng(0)
    v = np.array([0.9, -0.7, 0.2])
    p0 = np.array([0.25, 0.15, 0.10])
    tr = MultiTracker()
    reactive, predictive = [], []
    for k in range(90):
        t = k / 60
        tr.update([p0 + v * t + rng.normal(0, 0.01, 3)], t)
        b = tr.best()
        if b and k > 15:
            future = p0 + v * (t + LATENCY)
            reactive.append(np.linalg.norm(b.filter.position - future))
            predictive.append(np.linalg.norm(b.filter.predict(LATENCY) - future))
    assert np.mean(predictive) < np.mean(reactive) / 3, (
        f"prediction {np.mean(predictive):.3f} m vs reaction {np.mean(reactive):.3f} m"
    )


def test_uncertainty_grows_with_horizon():
    f = ConstantVelocityFilter(np.zeros(3), 0.0)
    assert f.position_uncertainty(0.6) > f.position_uncertainty(0.0)


def test_uncertainty_shrinks_with_observations():
    rng = np.random.default_rng(1)
    f = ConstantVelocityFilter(np.zeros(3), 0.0)
    early = f.position_uncertainty(LATENCY)
    for k in range(1, 60):
        f.update(np.array([0.01 * k, 0, 0]) + rng.normal(0, 0.005, 3), k / 60)
    assert f.position_uncertainty(LATENCY) < early


def test_tracks_are_not_confirmed_immediately():
    """One false detection must not make the arm lunge."""
    tr = MultiTracker()
    tr.update([np.array([0.2, 0.0, 0.1])], 0.0)
    assert tr.best() is None
    for k in range(1, 5):
        tr.update([np.array([0.2, 0.0, 0.1])], k / 60)
    assert tr.best() is not None


def test_two_hands_keep_separate_identities():
    tr = MultiTracker()
    a = np.array([0.20, 0.15, 0.10])
    b = np.array([0.20, -0.15, 0.10])
    for k in range(12):
        t = k / 60
        tr.update([a + np.array([0.01, 0, 0]) * k, b + np.array([0.01, 0, 0]) * k], t)
    confirmed = [t for t in tr.tracks if t.confirmed]
    assert len(confirmed) == 2
    assert len({t.id for t in confirmed}) == 2


def test_lost_tracks_are_dropped():
    tr = MultiTracker(max_misses=3)
    for k in range(6):
        tr.update([np.array([0.2, 0.0, 0.1])], k / 60)
    for k in range(6, 20):
        tr.update([], k / 60)
    assert tr.tracks == []


def test_association_gate_rejects_teleports():
    """A detection far from any track starts a new one rather than
    yanking an existing track across the table."""
    tr = MultiTracker(max_distance=0.10)
    for k in range(5):
        tr.update([np.array([0.2, 0.0, 0.1])], k / 60)
    first = tr.best().id
    tr.update([np.array([0.2, 0.9, 0.1])], 5 / 60)
    assert len(tr.tracks) == 2
    assert any(t.id == first for t in tr.tracks)


def test_filter_ignores_backwards_time():
    f = ConstantVelocityFilter(np.zeros(3), 1.0)
    before = f.position.copy()
    f.predict_to(0.5)
    assert np.allclose(f.position, before)

"""Game logic, safety gating, and the simulated opponent.

Heaviest on the gates. The game is what decides to swing a fast arm at a
person, so "it refuses to strike when X" is the part worth pinning down.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.arm.primitives import StrikeLimits
from tlod.game.contact import (
    ContactEvent, GeometricContactSensor, ProximityContactSensor,
)
from tlod.game.handslap import Difficulty, HandSlapGame
from tlod.game.opponent import DodgingHand


# -- fakes -----------------------------------------------------------------

class FakeFilter:
    """Stand-in for ConstantVelocityFilter with directly settable outputs.

    The real filter exposes `speed` and `position_uncertainty` as derived
    read-only values, which is right for production and useless for a
    test that needs to place the tracker in a specific state.
    """

    def __init__(self, position, speed=0.0, uncertainty=0.01):
        self.position = np.asarray(position, float)
        self.velocity = np.zeros(3)
        self.speed = float(speed)
        self._uncertainty = float(uncertainty)
        self.stamp = 0.0

    def predict(self, horizon):
        return self.position + self.velocity * horizon

    def position_uncertainty(self, horizon=0.0):
        return self._uncertainty


class FakeTrack:
    def __init__(self, position, speed=0.0, uncertainty=0.01):
        self.filter = FakeFilter(position, speed, uncertainty)
        self.id = 0
        self.hits = 10
        self.confirmed = True


def fake_robot(hand_position=None, speed=0.0, uncertainty=0.01, estopped=False):
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])),
                               SafetyLimits(), control_hz=200.0)
    controller.start()
    if estopped:
        controller.estop()

    track = FakeTrack(hand_position, speed, uncertainty) if hand_position is not None else None
    tracker = SimpleNamespace(best=lambda: track, tracks=[track] if track else [])
    return SimpleNamespace(controller=controller, tracker=tracker,
                           prediction_horizon=0.1, scene=None)


# -- difficulty ------------------------------------------------------------

def test_difficulty_presets_are_ordered():
    easy, normal, hard = (Difficulty.preset(n) for n in ("easy", "normal", "hard"))
    assert easy.hover_height > normal.hover_height > hard.hover_height
    assert easy.strike_duration > normal.strike_duration > hard.strike_duration
    assert easy.feint_probability < hard.feint_probability


def test_unknown_difficulty_raises():
    with pytest.raises(KeyError):
        Difficulty.preset("impossible")


# -- contact ---------------------------------------------------------------

def test_geometric_contact_hit_and_miss():
    s = GeometricContactSensor(radius=0.045)
    s.arm()
    assert s.poll(tool_xyz=[0.22, 0, 0.02], hand_xyz=[0.22, 0, 0.02]) is not None
    s.arm()
    assert s.poll(tool_xyz=[0.22, 0, 0.02], hand_xyz=[0.32, 0, 0.02]) is None


def test_contact_fires_once_per_arming():
    s = GeometricContactSensor()
    s.arm()
    assert s.poll(tool_xyz=[0.22, 0, 0.02], hand_xyz=[0.22, 0, 0.02]) is not None
    assert s.poll(tool_xyz=[0.22, 0, 0.02], hand_xyz=[0.22, 0, 0.02]) is None


def test_contact_needs_vertical_proximity():
    """Hovering above the hand is not a hit."""
    s = GeometricContactSensor(plane_tolerance=0.02)
    s.arm()
    assert s.poll(tool_xyz=[0.22, 0, 0.12], hand_xyz=[0.22, 0, 0.02]) is None


def test_contact_handles_missing_inputs():
    s = GeometricContactSensor()
    s.arm()
    assert s.poll() is None
    assert s.poll(tool_xyz=[0, 0, 0], hand_xyz=None) is None


def test_proximity_sensor_labels_itself_distinctly():
    """A proximity result must never be mistaken for a measurement."""
    s = ProximityContactSensor()
    s.arm()
    e = s.poll(tool_xyz=[0.22, 0, 0.02], hand_xyz=[0.22, 0, 0.03])
    assert isinstance(e, ContactEvent) and e.source == "proximity"


# -- gating ----------------------------------------------------------------

def test_no_strike_without_a_hand():
    game = HandSlapGame(seed=0)
    robot = fake_robot(None)
    game.update(robot, None, 0.01)
    assert game.state == "idle"
    assert "no hand" in game.reason


def test_no_strike_when_estopped():
    game = HandSlapGame(seed=0)
    robot = fake_robot([0.22, 0.0, 0.03], estopped=True)
    for _ in range(20):
        game.update(robot, None, 0.01)
    assert game.state == "idle"


def test_rejects_an_uncertain_hand():
    """Do not swing at an estimate we do not believe."""
    game = HandSlapGame(seed=0)
    robot = fake_robot([0.22, 0.0, 0.03], uncertainty=0.5)
    game.update(robot, None, 0.01)
    assert game.state == "idle"
    assert "uncertain" in game.reason


def test_rejects_a_hand_out_of_reach():
    game = HandSlapGame(seed=0)
    robot = fake_robot([1.5, 0.0, 0.03])
    game.update(robot, None, 0.01)
    assert game.state == "idle"
    assert "out of reach" in game.reason


def test_rejects_a_hand_inside_the_base_keepout():
    game = HandSlapGame(seed=0)
    robot = fake_robot([0.01, 0.0, 0.03])
    game.update(robot, None, 0.01)
    assert "out of reach" in game.reason


def test_cooldown_blocks_rapid_restrikes():
    game = HandSlapGame(seed=0)
    game.limits = StrikeLimits(min_strike_interval=10.0)
    game.last_strike = time.perf_counter()
    robot = fake_robot([0.22, 0.0, 0.03])
    assert game._may_strike(robot) is False
    assert "cooling" in game.reason


def test_acquires_a_valid_hand():
    game = HandSlapGame(seed=0)
    robot = fake_robot([0.22, 0.0, 0.03])
    game.update(robot, None, 0.01)
    assert game.state == "acquire"


# -- commit timing ---------------------------------------------------------

def test_no_commit_immediately_on_entering_ready():
    """Striking the instant it arrives reads as a glitch, not a decision."""
    game = HandSlapGame(seed=0)
    game.transition("ready")
    assert game._commit_probability(FakeTrack([0.22, 0, 0.03], speed=0.0), 0.01) == 0.0


def test_a_still_hand_is_more_tempting_than_a_moving_one():
    game = HandSlapGame(seed=0)
    game.state_since = time.perf_counter() - 5.0
    still = FakeTrack([0.22, 0, 0.03], speed=0.0)
    moving = FakeTrack([0.22, 0, 0.03], speed=0.6)
    assert game._commit_probability(still, 0.01) > game._commit_probability(moving, 0.01)


def test_commit_probability_is_bounded():
    game = HandSlapGame(seed=0)
    game.state_since = time.perf_counter() - 60.0
    assert 0.0 <= game._commit_probability(FakeTrack([0.22, 0, 0.03]), 1.0) <= 0.5


def test_timing_is_unpredictable_but_seeded():
    """Unpredictable to a player, reproducible for a test."""
    def wait_samples(seed):
        g = HandSlapGame(seed=seed)
        g.state_since = time.perf_counter() - 5.0
        return [float(g.rng.random()) for _ in range(20)]

    assert wait_samples(1) == wait_samples(1)
    assert wait_samples(1) != wait_samples(2)


# -- scoring ---------------------------------------------------------------

def test_score_records_hit_and_dodge():
    game = HandSlapGame(seed=0)
    robot = fake_robot([0.22, 0.0, 0.03])
    game._resolve(robot, robot.controller, hit=True)
    assert (game.score.robot, game.score.human, game.score.rounds) == (1, 0, 1)
    game._resolve(robot, robot.controller, hit=False)
    assert (game.score.robot, game.score.human, game.score.rounds) == (1, 1, 2)


def test_pause_toggle():
    game = HandSlapGame(seed=0)
    assert game.running
    game.on_key_space()
    assert not game.running


def test_hud_and_banner_are_strings():
    game = HandSlapGame(seed=0)
    assert all(isinstance(s, str) for s in game.hud())
    assert isinstance(game.banner(), str)


# -- opponent --------------------------------------------------------------

def test_opponent_rests_when_nothing_threatens():
    hand = DodgingHand(seed=0)
    for i in range(120):
        hand.update(i * 0.016, np.array([0.22, 0.0, 0.30]))
    assert hand.state == "rest"
    assert np.linalg.norm(hand.position - hand.home) < 0.05


def test_opponent_withdraws_from_a_descending_tool():
    hand = DodgingHand(reaction_time=0.10, seed=0)
    tool = np.array([0.22, 0.0, 0.13])
    t = 0.0
    for _ in range(200):
        t += 0.008
        tool = tool - np.array([0.0, 0.0, 0.0025])
        hand.update(t, tool)
        if hand.state != "rest":
            break
    assert hand.dodges >= 1


def test_opponent_reaction_time_is_respected():
    """A slower human notices at the same time but moves later."""
    def first_move(reaction):
        hand = DodgingHand(reaction_time=reaction, seed=0)
        tool = np.array([0.22, 0.0, 0.13])
        t = 0.0
        for _ in range(400):
            t += 0.008
            tool = tool - np.array([0.0, 0.0, 0.0025])
            if hand.update(t, tool) is not None and hand.state == "withdraw":
                return t
        return None

    fast, slow = first_move(0.10), first_move(0.30)
    assert fast is not None and slow is not None
    assert slow > fast


def test_opponent_returns_home_after_dodging():
    hand = DodgingHand(reaction_time=0.05, seed=0)
    tool = np.array([0.22, 0.0, 0.13])
    t = 0.0
    for _ in range(60):
        t += 0.008
        tool = tool - np.array([0.0, 0.0, 0.003])
        hand.update(t, tool)
    for _ in range(600):   # threat gone
        t += 0.008
        hand.update(t, np.array([0.22, 0.0, 0.30]))
    assert hand.state == "rest"
    assert np.linalg.norm(hand.position - hand.home) < 0.02


def test_opponent_integration_is_frame_rate_independent():
    """Withdrawal speed must be physics, not a function of call rate."""
    def travel(step):
        hand = DodgingHand(reaction_time=0.0, seed=0)
        hand.state = "withdraw"
        hand._last_update = 0.0
        t = 0.0
        for _ in range(int(0.20 / step)):
            t += step
            hand.update(t, None)
        return float(np.linalg.norm(hand.position - hand.home))

    assert travel(0.008) == pytest.approx(travel(0.004), rel=0.15)

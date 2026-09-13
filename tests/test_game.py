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
from tlod.arm.primitives import Hover, Retract, Strike, StrikeLimits
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
    assert easy.hover_height >= normal.hover_height >= hard.hover_height
    # Not a strict ordering any more, and deliberately so: the arm has one
    # honest floor on strike duration and both of the harder presets sit on
    # it. Easy is allowed to be slower; nothing is allowed to be faster.
    assert easy.strike_duration >= normal.strike_duration >= hard.strike_duration
    # More feints means *easier*, not harder. A feint is the human's
    # scoring opportunity: hold through it and they take the point. This
    # assertion was the other way round while the game was still a pure
    # dodge contest, and inverted when flinch scoring was introduced.
    assert easy.feint_probability > normal.feint_probability > hard.feint_probability
    # With strike speed off the table, these carry the difficulty.
    assert easy.mean_wait > normal.mean_wait > hard.mean_wait
    assert easy.settle_bonus < normal.settle_bonus < hard.settle_bonus


def test_no_preset_asks_for_a_strike_the_arm_cannot_land():
    """Measured floor: an 8 cm drop asked for in under 250 ms lands short.

    The arm answers a 0.21 s ask with 0.27 s and 14 mm of error, and a
    0.18 s ask with 0.23 s and 30 mm -- past the contact sensor's 20 mm
    plane tolerance, so the fastest strike is the one that cannot score.
    `hard` used to ask for 0.17 s.
    """
    for name in ("easy", "normal", "hard"):
        assert Difficulty.preset(name).strike_duration >= 0.25, name


def test_no_preset_hovers_beyond_the_reachable_drop():
    """`Strike` clamps the drop, so a high hover stops short of the hand."""
    max_drop = StrikeLimits().max_drop
    for name in ("easy", "normal", "hard"):
        assert Difficulty.preset(name).hover_height <= max_drop, name


def test_a_hover_above_max_drop_is_clamped_not_obeyed():
    """The guard behind the preset bound, for callers passing their own.

    A 12 cm hover against an 8 cm clamped drop leaves the paddle 4 cm above
    the hand at the end of the strike, which is twice the contact tolerance
    -- an "easy" difficulty that is really a broken one.
    """
    game = HandSlapGame(Difficulty(hover_height=0.12), seed=0)
    assert game.limits.hover_height == game.limits.max_drop


def test_unknown_difficulty_raises():
    with pytest.raises(KeyError):
        Difficulty.preset("impossible")


# -- limits against the measured arm ---------------------------------------

def drive(motion, controller, dt=0.005, limit=4.0):
    """Step a motion to completion in real time, as the control loop does."""
    motion.start(controller)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if motion.step(controller, dt):
            return True
        time.sleep(dt)
    return False


def test_a_strike_from_too_high_a_hover_cannot_reach_the_hand():
    """Why `hover_height` is capped at `max_drop` rather than trusted.

    The drop is clamped for safety, so hovering higher does not buy the
    human more warning -- it ends the strike that much above the hand.
    From the old easy preset's 12 cm, 4 cm short: twice the contact
    sensor's plane tolerance, so no strike on easy could ever score.
    """
    controller = fake_robot([0.22, 0.0, 0.03]).controller
    target = np.array([0.22, 0.0, 0.03])
    limits = StrikeLimits(hover_height=0.12)          # against a 0.08 max_drop
    assert drive(Hover(target, limits), controller)
    assert drive(Strike(target, limits, duration=0.3), controller)

    short_by = controller.pose().z - target[2]
    assert short_by == pytest.approx(limits.hover_height - limits.max_drop, abs=5e-3)

    sensor = GeometricContactSensor()
    sensor.arm()
    assert sensor.poll(tool_xyz=controller.pose().xyz(), hand_xyz=target) is None


def test_retract_stays_inside_the_configured_speed_ceiling():
    """`StrikeLimits` speeds are not clamped by `SafetyLimits`, so test them.

    `ArmController.profile_limits()` *substitutes* a per-call speed for
    `SafetyLimits.max_speed` rather than min()-ing it against the ceiling.
    A `retract_speed` of 4.0 therefore reached a measured 3.53 rad/s on an
    arm configured to cap at 3.5, for 10 ms on a 320 ms retract. The number
    in StrikeLimits is the only guard there is.
    """
    safety = SafetyLimits(max_speed=3.5, max_accel=35.0, max_jerk=400.0)  # the real arm
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])),
                               safety, control_hz=100.0)
    controller.start()
    limits, target = StrikeLimits(), np.array([0.22, 0.0, 0.03])
    assert drive(Hover(target, limits), controller)
    hover_q = controller.commanded.copy()
    assert drive(Strike(target, limits, duration=0.25), controller)

    controller.stats.peak_speed = 0.0
    assert drive(Retract(hover_q, limits, duration=0.28), controller)
    assert controller.stats.peak_speed <= safety.max_speed


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


# -- servo-load contact ----------------------------------------------------

def test_servo_load_contact_fires_on_a_load_spike():
    """Contact from the servos' own torque feedback -- no extra hardware."""
    from tlod.game.contact import ServoLoadContactSensor
    from tlod.types import JointState

    load = np.zeros(6)
    state = lambda: JointState(q=np.zeros(6), stamp=0.0, load=load.copy())
    sensor = ServoLoadContactSensor(state, threshold=0.12)
    sensor.arm(blank_for=0.0)               # no launch to ignore here
    assert sensor.poll() is None            # resting: sets the baseline
    load[2] = 0.30                          # elbow resists
    event = sensor.poll()
    assert event is not None and event.source == "servo_load"
    assert sensor.poll() is None            # once per arming


def test_servo_load_baseline_ignores_posture():
    """A loaded resting pose must not read as a hit.

    Resting load depends on configuration -- an extended arm holds more of
    its own weight -- so the baseline is captured at arm(), not assumed.
    """
    from tlod.game.contact import ServoLoadContactSensor
    from tlod.types import JointState

    load = np.array([0.0, 0.45, 0.40, 0.35, 0.0, 0.0])   # heavy but static
    state = lambda: JointState(q=np.zeros(6), stamp=0.0, load=load.copy())
    sensor = ServoLoadContactSensor(state, threshold=0.12)
    sensor.arm()
    assert sensor.poll() is None, "static posture load registered as contact"


def test_servo_load_ignores_pan_and_roll():
    """Joints orthogonal to a downward strike mostly report noise."""
    from tlod.game.contact import ServoLoadContactSensor
    from tlod.types import JointState

    load = np.zeros(6)
    state = lambda: JointState(q=np.zeros(6), stamp=0.0, load=load.copy())
    sensor = ServoLoadContactSensor(state, threshold=0.12)
    sensor.arm()
    load[0] = 0.9      # shoulder_pan
    load[4] = 0.9      # wrist_roll
    assert sensor.poll() is None


def test_servo_load_handles_backends_without_load():
    from tlod.game.contact import ServoLoadContactSensor
    from tlod.types import JointState

    sensor = ServoLoadContactSensor(lambda: JointState(q=np.zeros(6), stamp=0.0))
    sensor.arm()
    assert sensor.poll() is None


def test_the_retract_target_follows_the_hand():
    """After a strike the arm returns to where it was hovering, which has
    to be above the hand *now*, not where it was when first acquired.

    `ready` servos continuously to follow a drifting hand, and `settle`
    returns to `ready` rather than re-acquiring, so a pose captured once
    at acquisition goes stale by however far the hand has moved since.
    Observed on hardware as the arm wandering off to one side after a
    hit instead of lifting straight back up.
    """
    game = HandSlapGame("normal", seed=1)
    hand = np.array([0.24, 0.09, 0.03])
    robot = fake_robot(hand)
    try:
        stale = np.concatenate([model.HOME, [0.0]])
        game.hover_q = stale.copy()
        game.transition("ready")
        game.state_since = time.perf_counter() - 5.0    # past the settle-in gate

        for _ in range(4000):
            game._state_ready(robot, robot.controller, 0.005)
            if game.state in ("strike", "feint"):
                break
        assert game.state in ("strike", "feint"), "never committed"

        assert not np.allclose(game.hover_q, stale), (
            "retract target is still the pose from acquisition")
        hovering = model.tool_pose(game.hover_q[:5]).xyz()
        gap = float(np.linalg.norm(hovering[:2] - hand[:2]))
        assert gap < 0.06, f"would retract {gap * 1000:.0f} mm from the hand"
    finally:
        robot.controller.stop(park=False)


# -- performance -----------------------------------------------------------

def test_nothing_performs_during_a_commit():
    """The mechanic the whole game rests on: a feint only scores while it
    is credible, so a robot mugging on the way down draws no flinch and
    wins nothing. Performance belongs either side of a bluff, never
    inside it."""
    game = HandSlapGame("normal", seed=3)
    robot = fake_robot(np.array([0.22, 0.0, 0.03]))
    try:
        game.transition("ready")
        game.state_since = time.perf_counter() - 5.0
        for _ in range(4000):
            game._state_ready(robot, robot.controller, 0.005)
            if game.state in ("strike", "feint"):
                break
        assert game.state in ("strike", "feint"), "never committed"
        assert game.motion is not None
        assert game.motion.name in ("strike", "feint"), (
            f"committed with a {game.motion.name} running")
    finally:
        robot.controller.stop(park=False)


def test_the_reaction_matches_the_outcome_and_happens_once():
    game = HandSlapGame("normal", seed=3)
    robot = fake_robot(np.array([0.22, 0.0, 0.03]))
    try:
        game.last_result = "HIT"
        game._performed = False
        game.transition("settle")
        game._state_settle(robot, robot.controller, 0.005)
        first = game.motion
        assert first is not None and first.name == "flourish"

        game.motion = None                    # as if it had run to the end
        game._state_settle(robot, robot.controller, 0.005)
        assert game.motion is None, "gloated twice for one round"
    finally:
        robot.controller.stop(park=False)


def test_deadpan_neither_sways_nor_performs():
    """Measuring the robot and watching it are different jobs."""
    from tlod.game.handslap import Personality

    game = HandSlapGame("normal", seed=3, personality=Personality(enabled=False))
    robot = fake_robot(np.array([0.22, 0.0, 0.03]))
    try:
        assert game._sway() == (0.0, 0.0)
        game.last_result = "HIT"
        game._performed = False
        game.transition("settle")
        game._state_settle(robot, robot.controller, 0.005)
        assert game.motion is None
    finally:
        robot.controller.stop(park=False)


def test_the_sway_is_horizontal_and_small():
    """A vertical bob would change the height a strike starts from, and
    with it the depth it lands at -- which is the bug we just spent a
    session chasing, reintroduced as a joke."""
    game = HandSlapGame("normal", seed=3)
    radius = game.personality.sway_radius
    seen = [game._sway() for _ in range(200)]
    assert all(np.hypot(x, y) <= radius + 1e-9 for x, y in seen)
    assert len(seen[0]) == 2, "the sway must not have a z component"


def test_the_strike_launch_does_not_read_as_contact():
    """Servo load cannot tell the torque of accelerating the arm from the
    torque of meeting a hand, and at 35 rad/s^2 the launch clears any
    workable threshold. Observed on hardware: contact fired on the first
    tick of every strike, the swing was aborted about a millimetre in,
    and it read as the arm failing to move rather than as a false hit."""
    from tlod.game.contact import ServoLoadContactSensor

    launching = np.array([0.0, 0.40, 0.35, 0.30, 0.0, 0.0])   # torque to accelerate
    descending = np.array([0.0, 0.10, 0.08, 0.06, 0.0, 0.0])  # coasting down
    on_contact = descending + 0.25

    load = {"value": launching}
    sensor = ServoLoadContactSensor(
        lambda: SimpleNamespace(load=load["value"]), threshold=0.12)

    sensor.arm(blank_for=0.05)
    assert sensor.poll() is None, "fired during the launch"
    time.sleep(0.06)

    load["value"] = descending
    assert sensor.poll() is None, "the first poll after blanking is the baseline"
    assert sensor.poll() is None, "coasting read as contact"

    load["value"] = on_contact
    event = sensor.poll()
    assert event is not None and event.source == "servo_load"
    assert sensor.poll() is None, "fired twice for one strike"

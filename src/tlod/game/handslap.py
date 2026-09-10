"""Hand slap. The robot is the slapper; you are the dodger.

Why this way round: latency taxes only the responder. By initiating, the
robot spends its pipeline delay *before* the strike, where nobody is
waiting on it, and it aims at a hand that is nearly stationary. That
delay is now measured rather than estimated: shutter to servo is ~53 ms,
~85 ms from photons to the arm starting to move, not the ~250 ms this
docstring used to claim. See docs/slap-analysis.md.

That removes the speed problem and leaves a better one, though the
arithmetic has moved. The measured 8 cm strike is asked for in 250 ms and
takes 310 ms, of which 50-60 ms is the settle check at the end rather
than travel; contact fires at ~70% of the travel, so the paddle reaches
the hand ~180 ms after it starts moving, and the first ~25% of the
min-jerk covers only a few millimetres, so the human sees it at ~65 ms.
That leaves ~115 ms to react and clear -- against a human budget of
230-400 ms.

So the robot does not win *narrowly*, which is what the old text said; it
wins by about a reaction time, and it did so by slightly more when the
figures were simulated. Nor is it a race that can be tuned: every strike
the arm can honestly produce, from the 250 ms floor to the 400 ms ask
that is already too slow to read as a slap, lands well inside the human
budget. The dodge is not a contest at any setting the hardware offers.

Which leaves the same conclusion the simulator reached, for a better
reason: the interesting engineering is not reaction time, it is deciding
*when* to commit. A robot that strikes on a fixed rhythm is trivially
beaten by counting.

The commit decision here is a hazard rate: each tick carries a small
probability of striking, rising the longer the robot has been waiting.
This gives an unpredictable delay with a bounded tail -- the human can
never learn the timing, but never waits forever either. The rate is
modulated by how settled the hand is, because a hand that has just
stopped moving belongs to someone who has just stopped paying attention.

Feints exist because the robot owns the clock. A feint costs the human a
flinch, and the recovery from the flinch is an opening.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from tlod.arm import model
from tlod.arm.primitives import Feint, Hover, Retract, Strike, StrikeLimits, flourish
from tlod.game.base import StateMachine
from tlod.game.contact import ContactSensor, GeometricContactSensor
from tlod.types import Pose

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Rules:
    """Scoring.

    Dodging alone is not the game. Measurement showed why: contact fires
    at roughly 70% of the strike travel and motion onset costs the human
    the first quarter, so a human only ever gets ~45% of the travel to
    react. On the real arm that is ~115 ms of the measured 255 ms of
    travel in an 8 cm strike -- more than the ~95 ms the simulated arm
    left, and still half of the fastest human. Slowing the arm to
    compensate takes ~650 ms per strike, which no longer reads as a slap,
    and striking from further away is both slower *and* harder-hitting --
    the wrong direction on safety, and beyond `StrikeLimits.max_drop` it
    does not strike at all, because the drop is clamped and the paddle
    stops short of the hand.

    Real hand-slap is slapper-favoured too. What makes it a game is that
    the dodger is punished for flinching. So a feint that draws a flinch
    scores for the robot, and holding still through one scores for the
    human. The human's job becomes reading intent, which is a decision
    rather than a reflex, and it does not require beating physics.
    """

    flinch_distance: float = 0.045    # hand movement during a feint that counts
    hold_reward: bool = True          # holding through a feint scores for the human


@dataclass(slots=True)
class Personality:
    """How much the robot performs, as opposed to plays.

    The premise is a silly robot, so it fidgets while it waits and reacts
    when a round ends.

    The fidget is free -- `ready` is time already spent deciding when to
    commit. The reaction is not, quite: `settle` was 0.6 s of doing
    nothing, and a gesture big enough to read across a room takes about
    twice that, because amplitude and speed trade against each other
    under a fixed acceleration limit and this branch has chosen
    amplitude. So a round here runs roughly half a second longer than a
    deadpan one. That is the price of the performance, and on this branch
    it is worth paying; `--deadpan` is how you stop paying it.

    What is deliberately *not* here is any performance during a commit.
    A feint scores only while it is credible, and a robot mugging on the
    way down draws no flinch and wins nothing. The comedy has to live
    either side of the bluff, never inside it.

    `sway` is the one that touches play, and only helpfully: a hover that
    drifts in a small circle is harder to read the moment of commitment
    off than one that sits perfectly still. It is horizontal on purpose
    -- a vertical bob would change the height a strike starts from, and
    so the depth it lands at.
    """

    enabled: bool = True
    sway_radius: float = 0.022        # metres, horizontal only
    sway_period: float = 1.8          # seconds per lap
    flourish_duration: float = 1.2    # fallback; every move carries its own
    # Raising this past 3.5 used to do nothing, and the reason is worth
    # keeping: `profile_limits` bounds a caller's speed by
    # `safety.max_speed` rather than substituting for it, so on a rig
    # configured at 3.5 a flourish asking 4.5 got 3.5. Measured -- a spin
    # swept at 2.5/3.0/3.5/4.0/4.5 delivered 1.78/1.87/1.90/1.90/1.90 and
    # never commanded above 2.82 rad/s. The flourish now passes its own
    # ProfileLimits instead, which is why these numbers bite at all.
    flourish_speed: float = 12.0      # rad/s
    flourish_accel: float = 400.0     # rad/s^2
    flourish_jerk: float = 8000.0     # rad/s^3
    # These sit far above safety.max_accel and are meant to. A flourish is
    # the one motion that provably cannot approach anything -- joint space,
    # no target, an envelope that is zero at both ends, and never during a
    # commit -- so it does not have to be judged under limits that exist to
    # bound a swing aimed at a hand. Keeping it there was costing the whole
    # performance: measured on the rig, the profile rather than the servo
    # was holding a spin to a third of the speed it was allowed.


@dataclass(slots=True)
class Difficulty:
    """How hard the robot is to beat.

    Tuned by *how often it offers the human a scoring chance* -- feints to
    read, hesitation to sit through -- not by crippling the arm. A slower
    arm hits softer and reads as broken.

    Measuring the arm turned that into a two-sided rule, because a strike
    the arm cannot produce is dishonest in exactly the same way a
    deliberately slow one is. Two bounds fall out, and the presets below
    stay inside them:

      * `strike_duration` has a floor of 0.25 s. Asking for less does not
        get the paddle there sooner; it gets the same strike landing
        further from where it was aimed.
      * `hover_height` has a ceiling of `StrikeLimits.max_hover`, which
        is `max_drop` less `press_depth`. The drop is clamped at
        `max_drop` and has to cover the hover *and* the press below the
        hand, so a higher hover ends the strike short of its floor rather
        than giving the human more warning.

    Within those, the dial that actually changes the game is the feint
    rate -- which is the one the design wanted all along.
    """

    # <= StrikeLimits.max_hover, which is max_drop less press_depth. This
    # was 0.08 -- equal to max_drop, correct only while the strike aimed
    # at the hand plane rather than below it.
    hover_height: float = 0.055       # travel and impact, not reaction time
    strike_duration: float = 0.25     # the measured floor; slower is allowed, faster is not
    feint_probability: float = 0.45   # the human's main scoring opportunity
    mean_wait: float = 1.8            # seconds of expected hesitation
    settle_bonus: float = 2.5         # how much a still hand tempts a strike

    @classmethod
    def preset(cls, name: str) -> Difficulty:
        return {
            # Provenance, because these no longer all come from one place.
            #
            # HARDWARE (measured on the arm; 8 cm drop, one asked duration
            # per row, distance from the target at the end of the move):
            #
            #     ask 0.40 -> 0.45 s,  3.7 mm
            #     ask 0.30 -> 0.35 s,  7.9 mm
            #     ask 0.25 -> 0.31 s,  8.8 mm
            #     ask 0.21 -> 0.27 s, 14.1 mm
            #     ask 0.18 -> 0.23 s, 29.8 mm
            #
            # Every row overshoots the ask by a near-constant 50-60 ms,
            # which is `Motion._complete` waiting on `controller.settled`
            # rather than travel, so it does not shrink when the ask does.
            # Accuracy is what shrinks: below a 0.25 s ask the arm stops
            # further and further short, and by 0.18 it misses by more
            # than the contact sensor's 20 mm plane tolerance -- the
            # "fastest" strike is the one that cannot score. So
            # `strike_duration` and `hover_height` are hardware numbers
            # now, and 0.25 is a floor rather than a preference.
            #
            # SIMULATED (unchanged): feint_probability, mean_wait and
            # settle_bonus are still calibrated against the 250 ms
            # simulated opponent in `tlod eval`, and still have to be
            # re-tuned against real people in tier B. Treat the win rates
            # in docs/slap-analysis.md as stale in two directions: they
            # were measured with an arm that lands where it is told, and
            # with an `easy` hover that could not land at all.
            #
            # What changed, and why it is not a difficulty regression:
            # `hard` asked for 0.17 s, got 0.23 s and a 3 cm miss -- both
            # slower *and* blinder than `normal`, while the label promised
            # faster. `easy` hovered at 12 cm against an 8 cm clamped
            # drop, so its strike stopped 4 cm above the hand and could
            # not score at all; it was "easy" because it was broken. Both
            # now sit on honest geometry and separate on the dials that
            # work.
            #
            # `easy` then repeated the same mistake once more, at 0.08:
            # right while the strike aimed at the hand plane, wrong the
            # moment press_depth put the floor below it, since the drop
            # must now cover the hover *and* the press. The ceiling is
            # StrikeLimits.max_hover, and it is 0.055 -- `max_drop` less
            # `press_depth` less `HOVER_SLACK`, the last of which exists
            # because a hover that arrives high lifts the floor and the
            # floor is what contact is measured against.
            "easy": cls(hover_height=0.055,     # the most travel that can still land
                        strike_duration=0.30,   # a measured row: 0.35 s, 7.9 mm
                        feint_probability=0.65, mean_wait=2.4, settle_bonus=1.4),
            "normal": cls(),                    # the floor: 0.25 -> 0.31 s, 8.8 mm
            # Shares the floor with `normal` because there is nothing
            # below it. Hard is harder by feinting a third as often,
            # hesitating less, and pouncing harder on a settled hand. Its
            # shorter hover gives away less wind-up and lands softer at
            # the same duration; it does not change how long the human
            # gets, which the duration sets on its own.
            "hard": cls(hover_height=0.052, strike_duration=0.25,
                        feint_probability=0.25, mean_wait=1.3, settle_bonus=3.5),
        }[name]


class HandSlapGame(StateMachine):
    """Robot as slapper.

    States:
        idle      -> nothing to hit
        acquire   -> move above the hand
        ready     -> hover, track, decide when to commit
        feint     -> bait a flinch
        strike    -> committed
        resolve   -> hit or dodge
        settle    -> brief pause, then back
    """

    name = "hand_slap"
    initial_state = "idle"

    def __init__(
        self,
        difficulty: str | Difficulty = "normal",
        limits: StrikeLimits | None = None,
        contact: ContactSensor | None = None,
        seed: int | None = None,
        auto_start: bool = True,
        truth_provider=None,
        rules: Rules | None = None,
        personality: Personality | None = None,
    ) -> None:
        super().__init__()
        self.difficulty = (
            difficulty if isinstance(difficulty, Difficulty) else Difficulty.preset(difficulty)
        )
        self.limits = limits or StrikeLimits()
        # Clamped, not copied. `Strike` clamps its drop to `max_drop`, and
        # the swing has to cover the hover *and* `press_depth` below the
        # hand, so a hover above `max_hover` ends the strike short of its
        # floor -- which is how the old `easy` preset shipped a broken
        # difficulty rather than a gentle one.
        #
        # The ceiling used to be `max_drop`, which was right only while
        # the strike aimed at the hand plane exactly. Once press_depth
        # arrived, this line was quietly overwriting a corrected
        # StrikeLimits with the Difficulty's stale 0.08 on every single
        # game, so the floor came out at the hand plane and both contact
        # sensors were judging across a band of zero width. The warning in
        # StrikeLimits.__post_init__ could not catch it: it runs at
        # construction, and this runs after.
        wanted = self.difficulty.hover_height
        self.limits.hover_height = min(wanted, self.limits.max_hover)
        if wanted > self.limits.max_hover + 1e-9:
            log.warning(
                "difficulty asks to hover %.0f mm above the hand but the strike "
                "can only reach its floor from %.0f mm (max_drop %.0f less "
                "press_depth %.0f); hovering lower instead",
                wanted * 1e3, self.limits.max_hover * 1e3,
                self.limits.max_drop * 1e3, self.limits.press_depth * 1e3)
        self.contact = contact or GeometricContactSensor()
        self.rng = np.random.default_rng(seed)
        self.running = auto_start
        # In simulation, score against the true hand position. Scoring
        # against the tracker's estimate systematically over-credits the
        # robot: the filter lags a fast withdrawal, so a hand that has
        # already escaped still reads as being under the tool.
        self.truth_provider = truth_provider
        self.rules = rules or Rules()
        self.personality = personality or Personality()
        self._performed = False
        self.hand_at_commit: np.ndarray | None = None
        self._pending: str | None = None
        self._tool_at_bottom: np.ndarray | None = None
        self._strike_ended_because: str = ""
        self.flinches = 0
        self.holds = 0

        self.strike_target: np.ndarray | None = None
        self.hover_q: np.ndarray | None = None
        self.last_strike: float = 0.0
        self.last_result: str = ""
        self.reason: str = ""
        self.strikes = 0
        self.feints = 0
        # The gesture the arm is performing, and how many it has done.
        # Kept separate from `last_result` because they are separate
        # events: the verdict lands when the round is judged, the gesture
        # a second or so later once the retract has finished, and anything
        # reacting to the performance wants the second one.
        self.last_flourish: str = ""
        self.flourishes = 0
        # Why it is not playing, said out loud. `self.reason` has always
        # existed and has always gone only to `hud()`, which needs --view
        # or --preview to see -- so on a headless board a game that
        # refuses every hand prints nothing at all and looks broken. It is
        # not: three times on this rig the answer was "out of reach" or
        # "uncertain", and every counter in the run summary read healthy
        # while the arm sat still. Logged on change rather than per tick.
        self._said_reason: str = ""
        self._said_reason_at: float = 0.0

    # -- gating ------------------------------------------------------------
    def _hand(self, robot):
        """The hand we are willing to hit, or None with a reason set."""
        track = robot.tracker.best()
        if track is None:
            self.reason = "no hand"
            return None
        pos = track.filter.position
        uncertainty = track.filter.position_uncertainty(0.05)
        if uncertainty > 0.08:
            self.reason = f"uncertain ({uncertainty*100:.0f} cm)"
            return None

        radius = float(np.hypot(pos[0], pos[1]))
        lim = robot.controller.limits
        if not (lim.min_radius <= radius <= lim.max_radius):
            self.reason = "out of reach"
            return None
        if not (lim.table_z - 0.02 <= pos[2] <= lim.max_height - self.limits.hover_height):
            self.reason = "bad height"
            return None
        self.reason = ""
        return track

    def _sway(self) -> tuple[float, float]:
        """A small horizontal circle to hover on, or nothing.

        Restlessness, mostly -- a robot that holds perfectly still looks
        switched off. It earns its place in play too: a hover that drifts
        is harder to read the instant of commitment off than one that is
        motionless, and reading that instant is the human's whole job.

        Horizontal only. A vertical bob would change the height the
        strike starts from, and with it the depth it lands at.
        """
        if not self.personality.enabled or self.personality.sway_radius <= 0:
            return 0.0, 0.0
        angle = 2.0 * np.pi * time.perf_counter() / max(self.personality.sway_period, 0.1)
        r = self.personality.sway_radius
        return r * float(np.cos(angle)), r * float(np.sin(angle))

    def _perform(self, controller) -> None:
        """React to the round that just ended, in the pause after it.

        `settle` already sat still for 0.6 s between rounds; this fills
        that with something to watch rather than adding time to it. Once
        per round -- a robot that gloats twice is a robot with a bug.
        """
        if self._performed or not self.personality.enabled:
            return
        self._performed = True
        mood = {"HIT": "gloat", "DODGED": "sulk",
                "FLINCH": "smug", "HELD": "caught"}.get(self.last_result, "idle")
        gesture = flourish(mood, rng=self.rng,
                           duration=self.personality.flourish_duration,
                           speed=self.personality.flourish_speed,
                           accel=self.personality.flourish_accel,
                           jerk=self.personality.flourish_jerk)
        # Named so anything watching can react to the gesture rather than
        # to the verdict. The two are not the same event: the verdict is
        # announced the moment the round is judged, and the arm starts
        # performing a second or so later, once the retract has finished.
        self.last_flourish = gesture.move_name
        self.flourishes += 1
        self.run_motion(gesture, controller)

    def _may_strike(self, robot) -> bool:
        if robot.controller.estopped:
            self.reason = "e-stopped"
            return False
        if time.perf_counter() - self.last_strike < self.limits.min_strike_interval:
            self.reason = "cooling down"
            return False
        return True

    def _commit_probability(self, track, dt: float) -> float:
        """Per-tick hazard. Unpredictable, but with a bounded tail.

        Base rate gives the configured mean wait. A hand that has gone
        still multiplies it, because stillness means the human has
        stopped actively expecting the strike.
        """
        base = dt / max(self.difficulty.mean_wait, 0.1)
        settled = float(np.exp(-track.filter.speed / 0.06))
        rate = base * (1.0 + self.difficulty.settle_bonus * settled)
        # Do not strike in the first moments of hovering; it reads as a
        # glitch rather than a decision, and the human has not settled.
        if self.in_state < 0.4:
            return 0.0
        return float(np.clip(rate, 0.0, 0.5))

    # -- policy ------------------------------------------------------------
    def update(self, robot, perception, dt) -> None:
        controller = robot.controller
        if controller.estopped:
            self.transition("idle")
            return
        if not self.running:
            self.step_motion(controller, dt)
            return

        handler = getattr(self, f"_state_{self.state}")
        handler(robot, controller, dt)

    # -- states ------------------------------------------------------------
    def _announce_reason(self) -> None:
        """Say why no hand was accepted, once per distinct reason.

        Per-tick would be a hundred lines a second; only-once would hide a
        reason that changed ten minutes into a session. So: on change, and
        again if the same one persists for half a minute.
        """
        now = time.perf_counter()
        if self.reason == self._said_reason and now - self._said_reason_at < 30.0:
            return
        self._said_reason, self._said_reason_at = self.reason, now
        if self.reason:
            log.info("not playing: %s", self.reason)

    def _state_idle(self, robot, controller, dt) -> None:
        self.step_motion(controller, dt)
        track = self._hand(robot)
        self._announce_reason()
        if track is not None:
            self.transition("acquire")
            self.run_motion(Hover(track.filter.position, self.limits, duration=0.6), controller)

    def _state_acquire(self, robot, controller, dt) -> None:
        done = self.step_motion(controller, dt)
        track = self._hand(robot)
        if track is None:
            self.transition("idle")
            self.run_motion(Retract(model.HOME, self.limits, duration=0.6), controller)
            return
        if done:
            # A starting value only; `ready` refreshes it on commitment.
            self.hover_q = controller.commanded.copy()
            self.transition("ready")

    def _state_ready(self, robot, controller, dt) -> None:
        self.step_motion(controller, dt)
        track = self._hand(robot)
        if track is None:
            self.transition("idle")
            self.run_motion(Retract(model.HOME, self.limits, duration=0.6), controller)
            return

        # Keep hovering over the hand as it drifts. Slow, so the tracking
        # itself does not telegraph the strike.
        pos = track.filter.position
        sway_x, sway_y = self._sway()
        controller.servo_pose(
            Pose(float(pos[0] + sway_x), float(pos[1] + sway_y),
                 float(pos[2]) + self.limits.hover_height),
            max_speed=1.0, dt=dt,
        )

        if not self._may_strike(robot):
            return
        roll = self.rng.random()
        if roll < self._commit_probability(track, dt):
            # Where to come back to. Captured now, at the moment of
            # commitment, rather than kept from `acquire`: `ready` servos
            # continuously to follow the hand, and `settle` returns here
            # rather than re-acquiring, so a pose saved at acquisition is
            # stale by however far the hand has drifted since -- which on
            # a hand that has moved across the table means retracting to
            # a point nowhere near it, and looks from the outside like
            # the arm wandering off after a hit.
            self.hover_q = controller.commanded.copy()
            if self.rng.random() < self.difficulty.feint_probability:
                self.feints += 1
                self.hand_at_commit = np.array(self._hand_for_scoring(robot)
                                               if self._hand_for_scoring(robot) is not None
                                               else pos, float)
                self.transition("feint")
                self.run_motion(Feint(pos, self.limits), controller)
            else:
                self._begin_strike(controller, track)

    def _state_feint(self, robot, controller, dt) -> None:
        # A flinch is the point of a feint. If the hand bolts while the
        # robot is only pretending, that is a score.
        hand = self._hand_for_scoring(robot)
        if hand is not None and self.hand_at_commit is not None:
            moved = float(np.linalg.norm(np.asarray(hand) - self.hand_at_commit))
            if moved > self.rules.flinch_distance:
                if self.motion is not None:
                    self.motion.abort()
                self._resolve_feint(robot, controller, flinched=True)
                return
        if self.step_motion(controller, dt):
            # A flinch is a *reaction*, so it cannot have happened yet: a
            # feint lasts 280 ms and a human reacts in 230-400, then the
            # camera takes another ~100 to show it. Scoring at the end of
            # the motion calls every round a hold. Keep watching through
            # the retract instead, which is where the reaction lands.
            self._pending = "feint"
            self.transition("resolve")
            self._retract(controller)

    def _resolve_feint(self, robot, controller, flinched: bool) -> None:
        self.last_strike = time.perf_counter()
        self.score.rounds += 1
        if flinched:
            self.flinches += 1
            self.score.robot += 1
            self.last_result = "FLINCH"
            self.announce(f"flinched on a feint ({self.score})")
        elif self.rules.hold_reward:
            self.holds += 1
            self.score.human += 1
            self.last_result = "HELD"
            self.announce(f"held through a feint ({self.score})")
        self.hand_at_commit = None
        self._performed = False
        if self.state != "resolve":
            self.transition("resolve")
            self._retract(controller)

    def _begin_strike(self, controller, track) -> None:
        # Aim a touch ahead: the hand is nearly stationary, so this is a
        # small correction, not the load-bearing prediction a dodging
        # robot would need.
        pos = track.filter.predict(self.difficulty.strike_duration * 0.5)
        self.strike_target = np.array(pos, float)
        self.hand_at_commit = None
        # Blank the sensor for the launch. Servo load cannot tell the
        # torque of accelerating the arm from the torque of meeting a
        # hand, and the paddle has not reached the hand yet anyway --
        # contact fires at ~70% of the travel, so nothing before 40% of
        # it is real.
        self.contact.arm(self.difficulty.strike_duration * 0.4)
        self.strikes += 1
        self.transition("strike")
        self.run_motion(
            Strike(self.strike_target, self.limits, duration=self.difficulty.strike_duration),
            controller,
        )

    def _hand_for_scoring(self, robot):
        if self.truth_provider is not None:
            try:
                return self.truth_provider()
            except Exception:
                pass
        track = robot.tracker.best()
        return track.filter.position if track else None

    def _state_strike(self, robot, controller, dt) -> None:
        tool = controller.pose().xyz()
        hand = self._hand_for_scoring(robot)
        # Hand the motion the height we just read, so it can tell whether
        # the arm has actually stopped without paying for a second sync
        # read on the same tick. `Strike` ends its descent on this rather
        # than on the commanded setpoint going quiet, which it does while
        # the paddle is still well above the floor and moving.
        if self.motion is not None:
            self.motion.observe(float(tool[2]))
            # Kept here because `step_motion` drops the motion the moment
            # it finishes, and the retract replaces it before the round is
            # reported -- so by the time anything asks, the strike is gone.
            self._strike_ended_because = getattr(self.motion, "ended_because", "")
        # Whether the paddle has stopped travelling and is leaning on
        # whatever is under it. Only a load-based sensor uses this, and it
        # is the only thing that tells one apart from the swing's own
        # braking torque -- see ServoPressContactSensor.
        pressing = bool(getattr(self.motion, "pressing", False))
        if self.contact.poll(tool_xyz=tool, hand_xyz=hand,
                             pressing=pressing) is not None:
            self._report_contact()
            self._resolve(robot, controller, hit=True)
            return
        if self.step_motion(controller, dt):
            # The paddle is down, but the camera has not caught up: the
            # hand estimate is a pipeline-latency old, so a hand pulled
            # during the swing still reads as sitting under the tool.
            # Judging here scores that as a hit. So freeze where the tool
            # got to and keep asking, through the retract, until the
            # frames covering the moment of contact have arrived.
            self._tool_at_bottom = tool
            self._pending = "strike"
            self.transition("resolve")
            self._retract(controller)

    def _report_contact(self) -> None:
        """Say what the sensor saw, every round, whichever way it went.

        A verdict on its own is unfalsifiable from the outside: "dodged"
        looks the same whether the paddle stopped on a hand and the margin
        was too wide, or the floor was above the hand so there was nothing
        to stop short of, or the hand simply compressed. Each of those has
        a different fix and they are indistinguishable without the
        numbers, which is how several rounds of guesswork happened.
        """
        report = getattr(self.contact, "report", None)
        if callable(report):
            why = self._strike_ended_because or "?"
            self.announce(f"    {report()}  [descent {why}]")

    def _resolve(self, robot, controller, hit: bool) -> None:
        self.last_strike = time.perf_counter()
        self.score.rounds += 1
        if hit:
            self.score.robot += 1
            self.last_result = "HIT"
            self.announce(f"hit  ({self.score})")
        else:
            self.score.human += 1
            self.last_result = "DODGED"
            self.announce(f"dodged ({self.score})")
        self._performed = False
        self.transition("resolve")
        target = self.hover_q if self.hover_q is not None else np.concatenate([model.HOME, [0.0]])
        self.run_motion(Retract(target, self.limits, duration=0.28), controller)

    def _retract(self, controller) -> None:
        target = self.hover_q if self.hover_q is not None else np.concatenate([model.HOME, [0.0]])
        self.run_motion(Retract(target, self.limits, duration=0.28), controller)

    def _judging_window(self, robot) -> float:
        """How long to keep watching after a motion ends.

        The pipeline's own measured shutter-to-command latency, when
        there is one, because that is exactly how far behind the hand
        estimate is. Doubled: once for the lag on the evidence, once more
        because a reaction has to happen before it can be seen.
        """
        latency = getattr(robot, "measured_latency", 0.0) or 0.12
        return float(np.clip(latency * 2.0, 0.12, 0.45))

    def _state_resolve(self, robot, controller, dt) -> None:
        done = self.step_motion(controller, dt)

        # Still deciding the round the retract belongs to.
        if self._pending == "strike":
            hand = self._hand_for_scoring(robot)
            if (hand is not None
                    and self.contact.poll(tool_xyz=self._tool_at_bottom,
                                          hand_xyz=hand) is not None):
                self._pending = None
                self._resolve(robot, controller, hit=True)
                return
        elif self._pending == "feint":
            hand = self._hand_for_scoring(robot)
            if hand is not None and self.hand_at_commit is not None:
                moved = float(np.linalg.norm(np.asarray(hand) - self.hand_at_commit))
                if moved > self.rules.flinch_distance:
                    self._pending = None
                    self._resolve_feint(robot, controller, flinched=True)
                    return

        if self._pending is not None and self.in_state < self._judging_window(robot):
            return
        if self._pending == "strike":
            self._pending = None
            self._report_contact()
            self._resolve(robot, controller, hit=False)
            return
        if self._pending == "feint":
            self._pending = None
            self._resolve_feint(robot, controller, flinched=False)
            return

        if done:
            self.transition("settle")

    def _state_settle(self, robot, controller, dt) -> None:
        idle = self.step_motion(controller, dt)
        if idle:
            self._perform(controller)
            idle = self.motion is None
        # The dwell, and then however much of the reaction is still
        # playing -- capped, so a motion that never reports done cannot
        # wedge the game in a victory dance.
        # 2.4 s was sized against gestures that were being silently clipped
        # to about half their amplitude. The full-size ones take longer --
        # a spin runs 1.9 s, and the retract it follows is still finishing
        # when settle begins -- so the old cap cut the reaction off partway
        # and handed `ready` an arm mid-swing. This is the reaction window
        # only; nothing about how a round is judged depends on it.
        if self.in_state < 0.6 or (not idle and self.in_state < 3.0):
            return
        self.last_result = ""
        self.transition("ready" if self._hand(robot) is not None else "idle")

    # -- ui ----------------------------------------------------------------
    def on_key_space(self) -> None:
        self.running = not self.running
        self.announce("resumed" if self.running else "paused")

    def hud(self) -> list[str]:
        lines = [
            f"game     hand slap  [{self.state}]",
            f"score    {self.score}   rounds {self.score.rounds}",
            f"strikes  {self.strikes}   feints {self.feints}",
            f"flinches {self.flinches}   holds  {self.holds}",
        ]
        if self.reason:
            lines.append(f"waiting  {self.reason}")
        if not self.running:
            lines.append("PAUSED (space)")
        return lines

    def banner(self) -> str:
        return self.last_result

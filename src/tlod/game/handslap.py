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

import time
from dataclasses import dataclass

import numpy as np

from tlod.arm import model
from tlod.arm.primitives import Feint, Hover, Retract, Strike, StrikeLimits
from tlod.game.base import StateMachine
from tlod.game.contact import ContactSensor, GeometricContactSensor
from tlod.types import Pose


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
      * `hover_height` has a ceiling of `StrikeLimits.max_drop`. The drop
        is clamped there, so a higher hover ends the strike short of the
        hand rather than giving the human more warning.

    Within those, the dial that actually changes the game is the feint
    rate -- which is the one the design wanted all along.
    """

    hover_height: float = 0.08        # travel and impact, not reaction time; <= max_drop
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
            "easy": cls(hover_height=0.08,      # the most travel that can still land
                        strike_duration=0.30,   # a measured row: 0.35 s, 7.9 mm
                        feint_probability=0.65, mean_wait=2.4, settle_bonus=1.4),
            "normal": cls(),                    # the floor: 0.25 -> 0.31 s, 8.8 mm
            # Shares the floor with `normal` because there is nothing
            # below it. Hard is harder by feinting a third as often,
            # hesitating less, and pouncing harder on a settled hand. Its
            # shorter hover gives away less wind-up and lands softer at
            # the same duration; it does not change how long the human
            # gets, which the duration sets on its own.
            "hard": cls(hover_height=0.06, strike_duration=0.25,
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
    ) -> None:
        super().__init__()
        self.difficulty = (
            difficulty if isinstance(difficulty, Difficulty) else Difficulty.preset(difficulty)
        )
        self.limits = limits or StrikeLimits()
        # Clamped, not copied. `Strike` clamps its drop to `max_drop`, so a
        # hover above that ends the strike (hover - max_drop) above the
        # hand and nothing can ever land -- which is how the old `easy`
        # preset shipped a broken difficulty rather than a gentle one. A
        # caller supplying its own Difficulty gets the same guard.
        self.limits.hover_height = min(self.difficulty.hover_height, self.limits.max_drop)
        self.contact = contact or GeometricContactSensor()
        self.rng = np.random.default_rng(seed)
        self.running = auto_start
        # In simulation, score against the true hand position. Scoring
        # against the tracker's estimate systematically over-credits the
        # robot: the filter lags a fast withdrawal, so a hand that has
        # already escaped still reads as being under the tool.
        self.truth_provider = truth_provider
        self.rules = rules or Rules()
        self.hand_at_commit: np.ndarray | None = None
        self.flinches = 0
        self.holds = 0

        self.strike_target: np.ndarray | None = None
        self.hover_q: np.ndarray | None = None
        self.last_strike: float = 0.0
        self.last_result: str = ""
        self.reason: str = ""
        self.strikes = 0
        self.feints = 0

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
    def _state_idle(self, robot, controller, dt) -> None:
        self.step_motion(controller, dt)
        track = self._hand(robot)
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
        controller.servo_pose(
            Pose(float(pos[0]), float(pos[1]), float(pos[2]) + self.limits.hover_height),
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
            self._resolve_feint(robot, controller, flinched=False)

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
        self.transition("resolve")
        target = self.hover_q if self.hover_q is not None else np.concatenate([model.HOME, [0.0]])
        self.run_motion(Retract(target, self.limits, duration=0.28), controller)

    def _begin_strike(self, controller, track) -> None:
        # Aim a touch ahead: the hand is nearly stationary, so this is a
        # small correction, not the load-bearing prediction a dodging
        # robot would need.
        pos = track.filter.predict(self.difficulty.strike_duration * 0.5)
        self.strike_target = np.array(pos, float)
        self.hand_at_commit = None
        self.contact.arm()
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
        if self.contact.poll(tool_xyz=tool, hand_xyz=hand) is not None:
            self._resolve(robot, controller, hit=True)
            return
        if self.step_motion(controller, dt):
            self._resolve(robot, controller, hit=False)

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
        self.transition("resolve")
        target = self.hover_q if self.hover_q is not None else np.concatenate([model.HOME, [0.0]])
        self.run_motion(Retract(target, self.limits, duration=0.28), controller)

    def _state_resolve(self, robot, controller, dt) -> None:
        if self.step_motion(controller, dt):
            self.transition("settle")

    def _state_settle(self, robot, controller, dt) -> None:
        self.step_motion(controller, dt)
        if self.in_state < 0.6:
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

"""A simulated human hand that dodges.

Needed because the game cannot be evaluated against a hand that orbits
obliviously. A real opponent rests, watches, and yanks away when the arm
commits -- and the whole design question is whether the strike beats that
reaction. Modelling it explicitly turns "will this work?" into a
measurable win rate, and makes `reaction_time` a dial to sweep.

The model is deliberately simple and slightly generous to the human:

  rest        small drift around a home position
  notice      the tool has started descending
  judge       is this a real strike, or a feint?
  react       after `reaction_time`, begin withdrawing
  withdraw    accelerate away and back
  return      come back to rest after a pause

The judgement is the interesting part, and an earlier version of this
model did not have it. Withdrawal was triggered by mere motion, so the
only thing separating a flinch from a hold was whether the human happened
to be fast enough to move during the feint window -- which made *faster*
simulated humans flinch more, exactly backwards. A slow one "held" by
being too sluggish to react at all, not by deciding anything, and the
game degenerated into a coin flip on feint frequency with no skill in it.

`commit_fraction` fixes that by being the actual skill: how far the tool
must descend, as a fraction of the hover gap, before the human believes
it and bolts. Twitchy players (low) escape real strikes but flinch at
feints. Patient players (high) hold through feints but get caught. That
tradeoff is the game.

Human numbers for reference: simple visual reaction 150-250 ms, hand
withdrawal 80-150 ms. `reaction_time` here covers the first; the second
falls out of `withdraw_speed` and the distance needed to escape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tlod.vision.scene import SyntheticHandScene


@dataclass
class DodgingHand:
    home: np.ndarray = field(default_factory=lambda: np.array([0.22, 0.0, 0.03]))
    reaction_time: float = 0.22       # visual reaction; the dial that matters
    withdraw_speed: float = 1.3       # m/s once moving
    withdraw_distance: float = 0.18
    trigger_height: float = 0.14      # notice the tool below this
    trigger_displacement: float = 0.004   # metres of descent that reads as motion
    commit_fraction: float = 0.35     # how convinced before bolting; the skill dial
    drift: float = 0.012              # idle wander, metres
    return_delay: float = 0.45
    seed: int | None = None

    def __post_init__(self) -> None:
        self.home = np.asarray(self.home, float)
        self.rng = np.random.default_rng(self.seed)
        self.state = "rest"
        self.position = self.home.copy()
        self._noticed_at: float | None = None
        self._state_since = 0.0
        # Pull back and away, essentially horizontal. An earlier version
        # withdrew with a large upward component, which lifted the hand
        # into the descending tool and registered as a hit -- the model
        # was scoring its own escape as a collision. Nobody dodges a
        # downward slap by raising their hand.
        self._escape_dir = np.array([-0.60, 0.79, 0.06])
        self._escape_dir /= np.linalg.norm(self._escape_dir)
        self._rest_tool_z: float | None = None
        self._last_update: float | None = None
        self._hover_gap: float | None = None
        self.dodges = 0
        self.times_hit = 0

    def reset(self) -> None:
        self.state = "rest"
        self.position = self.home.copy()
        self._noticed_at = None
        self._rest_tool_z = None
        self._last_update = None

    def _descent(self, tool_z: float) -> float:
        """How far the tool has dropped below its resting height.

        Motion onset is detected by displacement rather than by a speed
        threshold. A speed gate was tried first and modelled the human
        badly: a real servo ramps up under its acceleration limit, so the
        tool did not cross 0.25 m/s until roughly half the strike was
        over, and the simulated human "noticed" 98 ms late for reasons
        that have nothing to do with human perception. Someone braced for
        a slap reacts to the arm starting to move, not to it reaching a
        particular speed.
        """
        if self._rest_tool_z is None:
            self._rest_tool_z = tool_z
        # Track the highest recent position as the reference, so the
        # baseline follows the arm back up between strikes.
        self._rest_tool_z = max(self._rest_tool_z * 0.995 + tool_z * 0.005, tool_z)
        return float(self._rest_tool_z - tool_z)

    def update(self, t: float, tool_xyz) -> np.ndarray:
        # Integrate against real elapsed time. This used to advance by a
        # hardcoded 0.008 s per call, which made the withdrawal speed a
        # function of the camera frame rate rather than of the physics --
        # so the modelled human escaped at roughly half the speed
        # configured, and never got away.
        dt = 0.0 if self._last_update is None else max(0.0, min(t - self._last_update, 0.1))
        self._last_update = t

        tool = np.asarray(tool_xyz, float) if tool_xyz is not None else None
        descent = self._descent(float(tool[2])) if tool is not None else 0.0

        if self.state == "rest":
            wander = self.rng.normal(0, self.drift * 0.06, 3)
            wander[2] *= 0.3
            self.position = self.position + wander
            pull = (self.home - self.position) * 0.04
            self.position = self.position + pull

            if tool is not None:
                horizontal = float(np.linalg.norm(tool[:2] - self.position[:2]))
                above = float(tool[2] - self.position[2])
                near = horizontal < 0.12 and -0.02 < above < self.trigger_height

                # Remember how high it was hovering; the judgement is
                # relative to that gap, not to an absolute distance.
                if near and descent < self.trigger_displacement:
                    self._hover_gap = above

                gap = self._hover_gap or self.trigger_height
                convinced = descent > max(gap * self.commit_fraction,
                                          self.trigger_displacement)
                if near and convinced and self._noticed_at is None:
                    self._noticed_at = t
                if self._noticed_at is not None and (t - self._noticed_at) >= self.reaction_time:
                    self.state = "withdraw"
                    self._state_since = t
                    self.dodges += 1

        elif self.state == "withdraw":
            travelled = float(np.linalg.norm(self.position - self.home))
            if travelled >= self.withdraw_distance:
                self.state = "wait"
                self._state_since = t
            else:
                self.position = self.position + self._escape_dir * self.withdraw_speed * dt

        elif self.state == "wait":
            if t - self._state_since > self.return_delay:
                self.state = "return"
                self._state_since = t

        elif self.state == "return":
            delta = self.home - self.position
            if np.linalg.norm(delta) < 0.01:
                self.state = "rest"
                self.position = self.home.copy()
                self._noticed_at = None
                self._rest_tool_z = None
            else:
                self.position = self.position + delta * 0.06

        return self.position.copy()


class DodgingHandScene(SyntheticHandScene):
    """A scene whose hand fights back.

    `tool_provider` is a callable returning the current tool position, so
    the opponent can see the arm coming. Wired by the CLI to the live
    controller.
    """

    def __init__(self, projector, hand: DodgingHand | None = None, tool_provider=None) -> None:
        super().__init__(projector)
        self.hand = hand or DodgingHand()
        self.tool_provider = tool_provider
        self._cache: tuple[float, np.ndarray] | None = None

    def position_at(self, t: float) -> np.ndarray:
        """Advance the opponent at most once per timestamp.

        This method looks pure but is not -- it steps a simulation. The
        rendering path calls it twice per frame (once for the pixel, once
        for apparent palm width), and the second call arrived with dt=0,
        which zeroed the measured descent speed and discarded the previous
        sample. The opponent could therefore never see a strike coming and
        never dodged: the robot won 100% of rounds even against a 100 ms
        reaction time. Caching by timestamp makes repeated calls
        idempotent, which is what every caller already assumed.
        """
        if self._cache is not None and self._cache[0] == t:
            return self._cache[1].copy()
        tool = None
        if self.tool_provider is not None:
            try:
                tool = self.tool_provider()
            except Exception:
                tool = None
        position = self.hand.update(t, tool)
        self._cache = (t, position.copy())
        return position

    def velocity_at(self, t: float) -> np.ndarray:
        return np.zeros(3)

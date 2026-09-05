"""Game scaffolding: a state machine that is also a Policy.

Games run on the control thread, so every state handler must return
promptly. Long movements are expressed as steppable Motions rather than
loops, and anything time-based is compared against a deadline rather than
slept on.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field

from tlod.arm.primitives import Motion
from tlod.runtime.app import Policy

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Score:
    robot: int = 0
    human: int = 0
    rounds: int = 0

    def reset(self) -> None:
        self.robot = self.human = self.rounds = 0

    def __str__(self) -> str:
        return f"robot {self.robot} - {self.human} human"


class StateMachine(Policy):
    """A policy with named states and a current motion."""

    name = "state_machine"
    initial_state: str = "idle"

    def __init__(self) -> None:
        self.state: str = self.initial_state
        self.state_since: float = time.perf_counter()
        self.motion: Motion | None = None
        self.score = Score()
        self.log: list[str] = []

    # -- state helpers -----------------------------------------------------
    def transition(self, state: str, note: str = "") -> None:
        if state == self.state:
            return
        log.debug("state %s -> %s %s", self.state, state, note)
        self.state = state
        self.state_since = time.perf_counter()

    @property
    def in_state(self) -> float:
        return time.perf_counter() - self.state_since

    def run_motion(self, motion: Motion, controller) -> None:
        if self.motion is not None:
            self.motion.abort()
        self.motion = motion
        motion.start(controller)

    def step_motion(self, controller, dt: float) -> bool:
        """Advance the current motion. True when there is none left."""
        if self.motion is None:
            return True
        if self.motion.step(controller, dt):
            self.motion = None
            return True
        return False

    def announce(self, text: str) -> None:
        self.log.append(text)
        if len(self.log) > 8:
            self.log.pop(0)
        log.info("%s", text)

    # -- Policy ------------------------------------------------------------
    @abc.abstractmethod
    def update(self, robot, perception, dt) -> None: ...

    def hud(self) -> list[str]:
        return [f"state    {self.state} ({self.in_state:.1f}s)", f"score    {self.score}"]

    def banner(self) -> str:
        return ""

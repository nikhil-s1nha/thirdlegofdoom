"""The one interface every arm implementation satisfies.

Nothing above this line knows whether it is driving servos or a simulation.
That is what lets the game logic, the kinematics and the whole perception
stack be written and tested before the hardware arrives, and it is why
`tlod run --sim` and `tlod run --real` are the same code path.

Contract:
  * `read()` returns the *measured* state with the timestamp of the sample.
  * `write()` sets a position target and returns immediately. It never
    blocks on motion completing. Trajectory generation lives one layer up,
    in ArmController, so that both backends behave identically.
  * angles are radians, JOINT_NAMES order, always.
"""

from __future__ import annotations

import abc

import numpy as np

from tlod.types import JointState


class ArmBackend(abc.ABC):
    """Position-controlled 6-motor arm."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> JointState:
        """Measured joint positions, radians, shape (6,)."""

    @abc.abstractmethod
    def write(self, q: np.ndarray) -> None:
        """Set goal positions, radians, shape (6,). Non-blocking."""

    @abc.abstractmethod
    def set_torque(self, enabled: bool) -> None:
        """Energise or release all joints. Releasing drops the arm, so
        callers are responsible for it being somewhere safe to fall from."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...

    # Optional: backends that can report it override this.
    def diagnostics(self) -> dict[str, object]:
        """Temperatures, voltages, loads. Empty when unsupported."""
        return {}

    def __enter__(self) -> "ArmBackend":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

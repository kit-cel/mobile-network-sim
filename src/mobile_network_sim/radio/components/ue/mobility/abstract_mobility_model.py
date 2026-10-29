"""Abstract mobility model."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mobility_state import UEMobilityState


class AbstractMobilityModel(ABC):
    """Abstract base class for mobility models."""

    NAME: str
    speed_kph: float

    def __init__(self, seed: int | None) -> None:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = str(cls.__name__)

    def reset(self, seed: int | None = None) -> None:
        """Reset the random-number generator.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the new generator. When omitted, reuse the model's
            configured seed.
        """
        if seed is None:
            seed = self.seed
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    @property
    def speed_mps(self) -> float:
        """Get the speed of the UE in m/s."""
        return self.speed_kph / 3.6

    def get_random_angle_deg(self) -> float:
        """
        Get a random angle in degrees.

        Returns
        -------
        float
            Random angle in degrees, in [-180, 180).
        """
        return self.rng.uniform(-180, 180)

    def get_random_angle_rad(self) -> float:
        """
        Get a random angle in radians.

        Returns
        -------
        float
            Random angle in radians, in [-π, π).
        """
        return self.rng.uniform(-np.pi, np.pi)

    @abstractmethod
    def update_ue_position(
        self,
        mobility_state: "UEMobilityState",
        time_increment_s: float,
    ) -> "UEMobilityState":
        """
        Get new position and direction of movement angle.

        Parameters
        ----------
        mobility_state : MobilityState
            Current position of the UE.
        time_increment_s : float
            Time delta, i.e., length of one time step in seconds.

        Returns
        -------
        MobilityState
            Updated position of the UE.
        """

"""User equipment entity."""

from dataclasses import dataclass, field

import numpy as np

from ....utils.types import FloatArr


@dataclass(slots=True)
class UEMobilityState:
    """
    Class to represent the mobility state of a user equipment (UE).

    Attributes
    ----------
    xy : FloatArr
        Current position of the UE in 2D space (x, y).
    angle_rad : float
        Current movement direction of the UE in radians.
    prev_xy : FloatArr
        Previous position of the UE in 2D space (x, y).
    """

    xy: FloatArr = field(
        default_factory=lambda: np.array([np.nan, np.nan], dtype=float)
    )
    angle_rad: float = np.nan

    prev_xy: FloatArr = field(
        default_factory=lambda: np.array([np.nan, np.nan], dtype=float)
    )

    _initial_position_set: bool = False

    def set_initial_position(
        self, initial_xy: FloatArr, initial_angle_rad: float
    ) -> None:
        """
        Set the initial position and angle of the UE.

        Parameters
        ----------
        initial_xy : FloatArr
            Initial position of the UE in 2D space (x, y).
        initial_angle_rad : float
            Initial movement direction of the UE in radians.
        """
        if self._initial_position_set:
            raise ValueError("Initial position has already been set.")
        self.xy = initial_xy
        self.angle_rad = initial_angle_rad
        self._initial_position_set = True

    def update(self, new_xy: FloatArr, new_angle_rad: float) -> None:
        """
        Update the position and store the previous position and angle.

        Parameters
        ----------
        new_xy : FloatArr
            New position of the UE in 2D space (x, y).
        new_angle_rad : float
            New movement direction of the UE in radians.
        """
        if not self._initial_position_set:
            raise ValueError("Initial position must be set before updating.")
        self.prev_xy = self.xy
        self.xy = new_xy
        self.angle_rad = new_angle_rad

    @classmethod
    def reset(cls) -> "UEMobilityState":
        """Create a mobility state containing the default unset values.

        Returns
        -------
        UEMobilityState
            New state with positions and angle set to NaN.
        """
        return cls()

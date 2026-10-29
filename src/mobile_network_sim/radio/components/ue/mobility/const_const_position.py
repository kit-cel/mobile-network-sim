"""Movement functions."""

from typing import TYPE_CHECKING

from .abstract_mobility_model import AbstractMobilityModel

if TYPE_CHECKING:
    from .mobility_state import UEMobilityState


class ConstPosition(AbstractMobilityModel):
    """Constant position movement model."""

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.speed_kph = 0.0

    def update_ue_position(
        self,
        mobility_state: "UEMobilityState",
        time_increment_s: float,
    ) -> "UEMobilityState":
        """Leave the UE mobility state unchanged.

        Parameters
        ----------
        mobility_state : UEMobilityState
            Current UE position and heading.
        time_increment_s : float
            Duration of the update step in seconds. It has no effect for this
            model.

        Returns
        -------
        UEMobilityState
            The unchanged input state.
        """
        return mobility_state

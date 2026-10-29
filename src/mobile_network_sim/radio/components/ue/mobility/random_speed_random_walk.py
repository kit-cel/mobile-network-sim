"""Random-speed random-walk mobility model."""

from typing import TYPE_CHECKING

import numpy as np

from .const_speed_corr_random_walk import ConstSpeedCorrRandomWalk

if TYPE_CHECKING:
    from .mobility_state import UEMobilityState


class RandomSpeedRandomWalk(ConstSpeedCorrRandomWalk):
    """Random-speed random-walk mobility model."""

    def __init__(
        self,
        *,
        speed_kph: float | None = None,
        random_speed_corr: float = 0.0,
        random_speed_lim_kph: tuple[float, float] = (0.0, 120.0),
        max_turn_angle_deg: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        speed_kph : float | None
            Initial speed of the UE in kph. If None, a random speed is chosen
            from the range specified by `random_speed_lim_kph`.
        random_speed_corr : float, optional
            Correlation factor for the random speed update. Default is 0.0.
        random_speed_lim_kph : tuple[float, float], optional
            Range for the random speed update. Default is (0.0, 120.0).
        max_turn_angle_deg : float, optional
            Maximum turn angle in degrees. Default is 1.0.
        seed : int | None, optional
            Optional random seed.
        """
        super().__init__(
            speed_kph=0.0,
            max_turn_angle_deg=max_turn_angle_deg,
            seed=seed,
        )

        self.random_speed_corr = random_speed_corr
        self.vmin_kph, self.vmax_kph = sorted(random_speed_lim_kph)

        if speed_kph is None:
            self._initial_speed_kph = None
            speed_kph = self._get_random_speed_kph()
        else:
            self._initial_speed_kph = speed_kph
        self.speed_kph = speed_kph

    def _get_random_speed_kph(self) -> float:
        return self.rng.uniform(self.vmin_kph, self.vmax_kph)

    def _update_speed(self) -> None:
        """Update the speed of the UE."""
        if self.random_speed_corr == 0:
            self.speed_kph = self._get_random_speed_kph()
        else:
            # IIR filter-like correlated speed update
            new_speed_kph = (
                self.random_speed_corr * self.speed_kph
                + (1 - self.random_speed_corr) * self._get_random_speed_kph()
            )
            self.speed_kph = np.clip(new_speed_kph, self.vmin_kph, self.vmax_kph)

    def reset(self, seed: int | None = None) -> None:
        """Reset the random generator and restore or resample speed.

        Parameters
        ----------
        seed : int | None, optional
            Random seed. If ``None``, reuse the model's initial seed.
        """
        super().reset(seed)
        if self._initial_speed_kph is not None:
            self.speed_kph = self._initial_speed_kph
        else:
            self.speed_kph = self._get_random_speed_kph()

    def update_ue_position(
        self,
        mobility_state: "UEMobilityState",
        time_increment_s: float,
    ) -> "UEMobilityState":
        """Update speed and advance the correlated random walk.

        Parameters
        ----------
        mobility_state : UEMobilityState
            Current UE position and heading. It is updated in place.
        time_increment_s : float
            Duration of the update step in seconds.

        Returns
        -------
        UEMobilityState
            State after updating speed, heading, and position.
        """
        self._update_speed()
        return super().update_ue_position(
            mobility_state=mobility_state,
            time_increment_s=time_increment_s,
        )

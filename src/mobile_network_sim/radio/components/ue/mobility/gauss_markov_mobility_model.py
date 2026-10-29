r"""Gauss--Markov mobility model.

The Gauss-Markov mobility model introduces temporal correlation in speed and
movement direction. The speed and heading are updated as autoregressive
processes. Its speed recurrence follows the Gauss--Markov velocity component
in Liang and Haas. Applying the same recurrence to heading is a
simulator-specific extension.

The parameter alpha controls the memory of the process. For alpha=0, movement
is close to memoryless random mobility. For alpha->1, movement approaches
constant-speed motion.

References
----------
.. [1] B. Liang and Z. J. Haas, "Predictive Distance-Based Mobility
   Management for PCS Networks," in *Proceedings of IEEE INFOCOM 1999*,
   vol. 3, pp. 1377--1384, 1999, section II-A, equation (4).
   doi:10.1109/INFCOM.1999.752157.
"""

import numpy as np

from .abstract_mobility_model import AbstractMobilityModel
from .mobility_state import UEMobilityState


class GaussMarkovMobility(AbstractMobilityModel):
    """Gauss-Markov mobility model with correlated speed and heading."""

    def __init__(
        self,
        mean_speed_kph: float,
        alpha: float = 0.95,
        speed_std_kph: float = 1.0,
        angle_std_deg: float = 1.0,
        mean_angle_rad: float | None = None,
        min_speed_kph: float = 0.0,
        max_speed_kph: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the Gauss-Markov mobility model.

        Parameters
        ----------
        mean_speed_kph : float
            Mean speed of the UE in km/h.
        alpha : float, optional
            Memory parameter for the autoregressive process, by default 0.95.
        speed_std_kph : float, optional
            Standard deviation of speed fluctuations in km/h, by default 1.0.
        angle_std_deg : float, optional
            Standard deviation of heading fluctuations in degrees, by default 1.0.
        mean_angle_rad : float | None, optional
            Mean heading of the UE in radians, by default None.
        min_speed_kph : float, optional
            Minimum speed of the UE in km/h, by default 0.0.
        max_speed_kph : float | None, optional
            Maximum speed of the UE in km/h, by default None.
        seed : int | None, optional
            Random seed for reproducibility, by default None.
        """
        super().__init__(seed=seed)

        if mean_speed_kph < 0.0:
            raise ValueError("mean_speed_kph must be non-negative.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if speed_std_kph < 0.0:
            raise ValueError("speed_std_kph must be non-negative.")
        if angle_std_deg < 0.0:
            raise ValueError("angle_std_deg must be non-negative.")
        if max_speed_kph is not None and max_speed_kph < min_speed_kph:
            raise ValueError("max_speed_kph must be >= min_speed_kph.")

        self.mean_speed_kph = mean_speed_kph
        self.speed_kph = mean_speed_kph
        self.alpha = alpha
        self.speed_std_kph = speed_std_kph
        self.angle_std_rad = np.deg2rad(angle_std_deg)
        self.mean_angle_rad = mean_angle_rad
        self.min_speed_kph = min_speed_kph
        self.max_speed_kph = max_speed_kph

    def reset(self, seed: int | None = None) -> None:
        """Reset the random generator and speed state.

        Parameters
        ----------
        seed : int | None, optional
            Random seed. If ``None``, reuse the model's initial seed.
        """
        super().reset(seed)
        self.speed_kph = self.mean_speed_kph

    def update_ue_position(
        self,
        mobility_state: UEMobilityState,
        time_increment_s: float,
    ) -> UEMobilityState:
        r"""
        Update the UE position according to the Gauss--Markov mobility model.

        At each step, speed and heading are updated as

        .. math::

           v_t &= \alpha v_{t-1} + (1-\alpha)\bar v
             + \sqrt{1-\alpha^2}\,\sigma_v w_t, \\
           \theta_t &= \alpha\theta_{t-1} + (1-\alpha)\bar\theta
             + \sqrt{1-\alpha^2}\,\sigma_\theta z_t,

        where :math:`w_t` and :math:`z_t` are independent standard-normal
        variables. The configured speed bounds are then applied and the
        heading is wrapped to :math:`[0,2\pi)`. Position advances by

        .. math::

           \mathbf{x}_t = \mathbf{x}_{t-1} + v_t\,\Delta t
           [\cos(\theta_t),\sin(\theta_t)].

        Parameters
        ----------
        mobility_state : UEMobilityState
            Current mobility state of the UE. It is updated in place.
        time_increment_s : float
            Duration of the update step in seconds.

        Returns
        -------
        UEMobilityState
            State after updating speed, heading, and position.

        Notes
        -----
        If ``mean_angle_rad`` is ``None``, the current heading is used as
        :math:`\bar\theta`; the heading therefore becomes a Gaussian turning
        process without mean reversion. Clipping the speed changes the
        unconstrained Gaussian stationary distribution.
        """

        noise_scale = np.sqrt(max(0.0, 1.0 - self.alpha**2))

        new_speed_kph = (
            self.alpha * self.speed_kph
            + (1.0 - self.alpha) * self.mean_speed_kph
            + noise_scale * self.speed_std_kph * self.rng.normal()
        )

        if self.max_speed_kph is None:
            new_speed_kph = max(self.min_speed_kph, new_speed_kph)
        else:
            new_speed_kph = np.clip(
                new_speed_kph,
                self.min_speed_kph,
                self.max_speed_kph,
            )

        theta_bar = (
            mobility_state.angle_rad
            if self.mean_angle_rad is None
            else self.mean_angle_rad
        )

        new_angle_rad = (
            self.alpha * mobility_state.angle_rad
            + (1.0 - self.alpha) * theta_bar
            + noise_scale * self.angle_std_rad * self.rng.normal()
        )
        new_angle_rad = np.mod(new_angle_rad, 2.0 * np.pi)

        self.speed_kph = float(new_speed_kph)

        displacement = (
            self.speed_mps
            * time_increment_s
            * np.array([np.cos(new_angle_rad), np.sin(new_angle_rad)])
        )

        mobility_state.update(
            new_xy=mobility_state.xy + displacement,
            new_angle_rad=new_angle_rad,
        )
        return mobility_state

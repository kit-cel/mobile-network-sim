r"""Constant-speed correlated random walk mobility model.

This model implements a constant-speed correlated random walk (CRW), where the
movement direction evolves according to

.. math::

   \theta_{t+1} = \theta_t + \Delta\theta_t,

with bounded random turning angle

.. math::

   \Delta\theta_t \sim \mathcal{U}(-\theta_{\max},\theta_{\max}).

This introduces temporal correlation between consecutive movement directions
and yields smoother trajectories than a memoryless random walk.

The use of movement lengths and turning-angle distributions follows the
correlated-random-walk framework of Kareiva and Shigesada. Constant step
lengths and bounded uniform turning angles are choices of this simulator, not
distributions prescribed by that paper.

References
----------
.. [1] P. M. Kareiva and N. Shigesada, "Analyzing Insect Movement as a
   Correlated Random Walk," *Oecologia*, vol. 56, nos. 2--3,
   pp. 234--238, 1983, equations (1)--(2).
   doi:10.1007/BF00379695.
"""

import numpy as np

from .abstract_mobility_model import AbstractMobilityModel
from .mobility_state import UEMobilityState


class ConstSpeedCorrRandomWalk(AbstractMobilityModel):
    """Constant-speed correlated random walk mobility model."""

    def __init__(
        self,
        speed_kph: float,
        max_turn_angle_deg: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if speed_kph < 0.0:
            raise ValueError("speed_kph must be non-negative.")
        if not 0.0 <= max_turn_angle_deg <= 180.0:
            raise ValueError("max_turn_angle_deg must be in [0, 180].")

        self.speed_kph = speed_kph
        self.max_turn_angle_deg = max_turn_angle_deg
        self.max_turn_angle_rad = np.deg2rad(max_turn_angle_deg)

    def update_ue_position(
        self,
        mobility_state: UEMobilityState,
        time_increment_s: float,
    ) -> UEMobilityState:
        r"""
        Update the UE position using a correlated random-walk mobility model.

        At each update step, a heading change is sampled uniformly from
        :math:`\mathcal{U}(-\theta_{\max},\theta_{\max})`. The resulting
        angle is wrapped to :math:`[0,2\pi)`.

        The UE then moves at constant speed according to

        .. math::

           \mathbf{x}_t = \mathbf{x}_{t-1} + v\,\Delta t
           [\cos(\theta_t),\sin(\theta_t)],

        where :math:`v` is the configured UE speed and :math:`\Delta t` is
        the update duration.

        Parameters
        ----------
        mobility_state : UEMobilityState
            Current UE position and heading. It is updated in place.
        time_increment_s : float
            Duration of the update step in seconds.

        Returns
        -------
        UEMobilityState
            The updated input state.

        Notes
        -----
        Step length is deterministic for a fixed ``time_increment_s``.
        """

        delta_theta = self.rng.uniform(
            -self.max_turn_angle_rad,
            self.max_turn_angle_rad,
        )

        new_angle_rad = np.mod(
            mobility_state.angle_rad + delta_theta,
            2.0 * np.pi,
        )

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

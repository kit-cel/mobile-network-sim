r"""Random waypoint mobility model.

The Random Waypoint (RWP) model repeatedly selects a random destination inside
a simulation region, moves toward it with a selected speed, optionally pauses
at the destination, and then selects a new destination.

The original model chooses destinations in a simulation area and moves to each
at a newly sampled speed, optionally pausing between trips. This implementation
is a variant because the destination distribution is supplied by
``sample_waypoint_fn`` rather than fixed by the model.

Notes
-----
Random waypoint does not generally start in its stationary distribution.
Uniform speed sampling can cause the time-averaged speed to drift during a
transient, and destinations sampled uniformly over an area lead to a
non-uniform time-averaged spatial distribution. A strictly positive minimum
speed, as used by the default configuration, avoids the zero-speed pathology
but does not make the initial position and trip state stationary.

References
----------
.. [1] D. B. Johnson and D. A. Maltz, "Dynamic Source Routing in Ad Hoc
   Wireless Networks," in *Mobile Computing*, T. Imielinski and H. F. Korth,
   eds., pp. 153--181, 1996, section 5.1.
   doi:10.1007/978-0-585-29603-6_5.
.. [2] J. Yoon, M. Liu, and B. Noble, "Random Waypoint Considered Harmful,"
   in *Proceedings of IEEE INFOCOM 2003*, vol. 2, pp. 1312--1321, 2003,
   sections I--III. doi:10.1109/INFCOM.2003.1208967.
.. [3] T. Camp, J. Boleng, and V. Davies, "A Survey of Mobility Models for
   Ad Hoc Network Research," *Wireless Communications and Mobile Computing*,
   vol. 2, no. 5, pp. 483--502, 2002, section 2.
   doi:10.1002/wcm.72.
"""

from typing import Callable, TYPE_CHECKING

import numpy as np

from .abstract_mobility_model import AbstractMobilityModel

if TYPE_CHECKING:
    from .mobility_state import UEMobilityState


class RandomWaypoint(AbstractMobilityModel):
    """Random waypoint mobility model."""

    def __init__(
        self,
        sample_waypoint_fn: Callable[[np.random.Generator], np.ndarray],
        waypoint_speed_lim_kph: tuple[float, float] = (3.0, 30.0),
        waypoint_pause_time_lim_s: tuple[float, float] = (0.0, 0.0),
        waypoint_tolerance_m: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the random waypoint mobility model.

        Parameters
        ----------
        sample_waypoint_fn : Callable[[numpy.random.Generator], numpy.ndarray]
            Function to sample a random waypoint using the supplied UE-owned
            generator. It returns a two-dimensional ``(x, y)`` position.
        waypoint_speed_lim_kph : tuple[float, float], optional
            Range for the waypoint speed update. Default is (3.0, 30.0).
        waypoint_pause_time_lim_s : tuple[float, float], optional
            Range for the waypoint pause time update. Default is (0.0, 0.0).
        waypoint_tolerance_m : float, optional
            Tolerance for reaching the waypoint. Default is 1e-6.
        seed : int | None, optional
            Optional random seed.
        """
        super().__init__(seed=seed)

        self.sample_waypoint_fn = sample_waypoint_fn
        self.v_min_kph, self.v_max_kph = sorted(waypoint_speed_lim_kph)
        self.pause_min_s, self.pause_max_s = sorted(waypoint_pause_time_lim_s)
        self.waypoint_tolerance_m = waypoint_tolerance_m

        if self.v_min_kph < 0.0:
            raise ValueError("Minimum speed must be non-negative.")
        if self.v_max_kph == 0.0:
            raise ValueError("Maximum speed must be positive.")
        if self.pause_min_s < 0.0:
            raise ValueError("Pause times must be non-negative.")

        self.speed_kph = self._sample_speed_kph()
        self._target_xy: np.ndarray | None = None
        self._remaining_pause_s = 0.0

    def _sample_speed_kph(self) -> float:
        return float(self.rng.uniform(self.v_min_kph, self.v_max_kph))

    def _sample_pause_time_s(self) -> float:
        return float(self.rng.uniform(self.pause_min_s, self.pause_max_s))

    def _sample_waypoint(self) -> np.ndarray:
        return np.asarray(self.sample_waypoint_fn(self.rng), dtype=float)

    def reset(self, seed: int | None = None) -> None:
        """Reset the random generator and discard the active trip.

        Parameters
        ----------
        seed : int | None, optional
            Random seed. If ``None``, reuse the model's initial seed.
        """
        super().reset(seed)
        self.speed_kph = self._sample_speed_kph()
        self._target_xy = None
        self._remaining_pause_s = 0.0

    def update_ue_position(
        self,
        mobility_state: "UEMobilityState",
        time_increment_s: float,
    ) -> "UEMobilityState":
        """Advance the UE toward its current or newly sampled waypoint.

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
        A step that reaches a waypoint stops exactly at that waypoint. Any
        unused portion of the step is not applied to the pause or next trip.
        """
        if self._remaining_pause_s > 0.0:
            self._remaining_pause_s = max(
                0.0,
                self._remaining_pause_s - time_increment_s,
            )
            return mobility_state

        if self._target_xy is None:
            self._target_xy = self._sample_waypoint()
            self.speed_kph = self._sample_speed_kph()

        direction_vec = self._target_xy - mobility_state.xy
        distance_to_target_m = float(np.linalg.norm(direction_vec))

        if distance_to_target_m <= self.waypoint_tolerance_m:
            self._target_xy = None
            self._remaining_pause_s = self._sample_pause_time_s()
            return mobility_state

        new_angle_rad = float(np.arctan2(direction_vec[1], direction_vec[0]))
        new_angle_rad = np.mod(new_angle_rad, 2.0 * np.pi)

        step_length_m = self.speed_mps * time_increment_s

        if step_length_m >= distance_to_target_m:
            new_xy = self._target_xy.copy()
            self._target_xy = None
            self._remaining_pause_s = self._sample_pause_time_s()
        else:
            unit_direction = direction_vec / distance_to_target_m
            new_xy = mobility_state.xy + step_length_m * unit_direction

        mobility_state.update(new_xy=new_xy, new_angle_rad=new_angle_rad)
        return mobility_state

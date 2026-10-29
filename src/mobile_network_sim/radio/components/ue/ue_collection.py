"""
Module for the UserEquipmentCollection class.
This class manages multiple UEs and provides vectorized operations.
"""

from copy import deepcopy

import numpy as np

from mobile_network_sim.seeding import (
    RadioEpisodeSeeds,
    radio_episode_seeds,
)

from .ue import UserEquipment
from ...geometry.simulation_geometry import SimulationGeometry
from ...utils.types import FloatArr
from ...config.ue_config import UserEquipmentConfig


class UserEquipmentCollection:
    """Container for multiple UEs with vectorized aggregate operations."""

    def __init__(
        self,
        n_ue: int,
        ue_configs: UserEquipmentConfig | list[UserEquipmentConfig],
        geo_handler: SimulationGeometry,
    ):
        self.n_ue = n_ue
        self.geo_handler = geo_handler

        self.ue_list = self._build_user_equipments(ue_configs)
        self._set_initial_positions()

        self.ue_heights = np.asarray([ue.height for ue in self.ue_list])
        self.nf_db_ues = np.array([ue.rx_nf_db for ue in self.ue_list])

        self._ue_mobility_states = [ue.mobility_state for ue in self.ue_list]

    def _build_user_equipments(
        self, ue_configs: UserEquipmentConfig | list[UserEquipmentConfig]
    ) -> list["UserEquipment"]:
        """
        Build a list of UserEquipment instances based on the provided configurations.

        Parameters
        ----------
        ue_configs : UserEquipmentConfig | list[UserEquipmentConfig]
            A single UserEquipmentConfig or a list of UserEquipmentConfig instances.
            If a single config is provided, it will be duplicated for each UE with unique seeds.

        Returns
        -------
        list[UserEquipment]
            A list of UserEquipment instances.
        """
        if isinstance(ue_configs, UserEquipmentConfig):
            ue_configs_list: list[UserEquipmentConfig] = []
            for ue_i in range(self.n_ue):
                ue_i_config = deepcopy(ue_configs)
                print(f"UE {ue_i} ", end="")
                ue_i_config.update_recursively(({"seed": ue_i_config.seed + ue_i + 1}))
                ue_configs_list.append(ue_i_config)
        else:
            ue_configs_list = ue_configs

        if len(ue_configs_list) != self.n_ue:
            raise ValueError(
                f"Number of UE configs ({len(ue_configs_list)}) not matching n_ue ({self.n_ue})."
            )

        ue_list = []
        for ue_id, ue_i_config in enumerate(ue_configs_list):
            ue_list.append(
                UserEquipment(
                    ue_id=ue_id,
                    config=ue_i_config,
                    sample_waypoint_fn=self._sample_waypoint,
                )
            )

        return ue_list

    def _sample_waypoint(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a waypoint from the requesting UE's mobility stream."""
        return self.geo_handler.roi.sample_positions(1, rng=rng)[0]

    def _set_initial_positions(self) -> None:
        """Set initial positions for all UEs."""
        init_xy = self.geo_handler.roi.sample_positions(self.n_ue)
        for ue, xy in zip(self.ue_list, init_xy):
            ue.set_initial_position(xy)

    def reset(self, seed: int | RadioEpisodeSeeds | None = None) -> None:
        """Reset all UEs to new random positions.

        Parameters
        ----------
        seed : int, RadioEpisodeSeeds, or None, optional
            Episode root or fully resolved component streams. If ``None``,
            reset operations generate their own seeds.
        """
        self.reset_ue(ue_idxs=list(range(self.n_ue)), seed=seed)

    def reset_ue(
        self,
        ue_idxs: int | list[int],
        seed: int | RadioEpisodeSeeds | None = None,
    ) -> None:
        """
        Reset a specific UE or a list of UEs.

        Parameters
        ----------
        ue_idxs : int | list[int]
            Index or list of indices of UEs to reset.
        seed : int, RadioEpisodeSeeds, or None
            Episode root or resolved streams. Initial positions and mobility
            innovations use separate streams. If ``None``, new unseeded
            streams are created.
        """
        plan = None if seed is None else radio_episode_seeds(seed)
        self.geo_handler.roi.reset(
            seed=None if plan is None else plan.initial_positions_seed
        )
        boundary = self.geo_handler.boundary_handler
        if hasattr(boundary, "set_seed"):
            boundary.set_seed(None if plan is None else plan.boundary_seed)
        if isinstance(ue_idxs, int):
            ue_idxs = [ue_idxs]
        for idx in ue_idxs:
            self.ue_list[idx].reset(
                seed=(
                    plan.ue_mobility_seed(idx)
                    if plan is not None
                    else None
                )
            )
            position_rng = (
                None
                if plan is None
                else np.random.default_rng(plan.ue_initial_position_seed(idx))
            )
            rand_xy = self.geo_handler.roi.sample_positions(1, rng=position_rng)[0]
            self.ue_list[idx].set_initial_position(rand_xy)
        self._ue_mobility_states = [ue.mobility_state for ue in self.ue_list]

    def move_ues_in_roi(self, time_increment_s: float) -> None:
        """
        Update the positions of all UEs based on their mobility models.

        Parameters
        ----------
        time_increment_s : float
            Time increment in seconds for the position update.
        """
        for ue in self.ue_list:
            ue.update_position(time_increment_s)

        self.xy, self.angles_rad = self.geo_handler.keep_positions_in_roi(
            old_xy=self.prev_xy,
            new_xy=self.xy,
            move_angles_rad=self.angles_rad,
        )

    @property
    def xy(self) -> FloatArr:
        """
        Stacked UE positions.

        Returns
        -------
        FloatArr
            Stacked UE positions. Shape: (n_ue, 2).
        """
        return np.asarray([s.xy for s in self._ue_mobility_states], dtype=float)

    @xy.setter
    def xy(self, value: FloatArr) -> None:
        """
        Update UE positions.

        Parameters
        ----------
        value : FloatArr
            New UE positions. Shape: (n_ue, 2).
        """
        for s, xy in zip(self._ue_mobility_states, value):
            s.xy[:] = xy

    @property
    def prev_xy(self) -> FloatArr:
        """
        Stacked previous UE positions.

        Returns
        -------
        FloatArr
            Stacked previous UE positions. Shape: (n_ue, 2).
        """
        return np.asarray([s.prev_xy for s in self._ue_mobility_states], dtype=float)

    # No setter for prev_xy since it's managed internally by each UEMobilityState

    @property
    def xyz(self) -> FloatArr:
        """
        Return stacked UE positions in 3D (x, y, height).

        Returns
        -------
        FloatArr
            Stacked UE positions in 3D. Shape: (n_ue, 3).
        """
        xy = self.xy
        heights = self.ue_heights[:, np.newaxis]
        return np.hstack((xy, heights))

    # No setter for xyz since it's derived from xy and heights

    @property
    def angles_rad(self) -> FloatArr:
        """
        UE movement angles.

        Returns
        -------
        FloatArr
            UE movement angles in radians. Shape: (n_ue,).
        """
        return np.asarray([s.angle_rad for s in self._ue_mobility_states], dtype=float)

    @angles_rad.setter
    def angles_rad(self, value: FloatArr) -> None:
        """
        Update UE movement angles.

        Parameters
        ----------
        value : FloatArr
            New UE movement angles in radians. Shape: (n_ue,).
        """
        for s, angle in zip(self._ue_mobility_states, value):
            s.angle_rad = float(angle)

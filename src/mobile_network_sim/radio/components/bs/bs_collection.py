"""
Module for the BaseStationCollection class.
This class manages multiple BSs and provides vectorized operations.
"""

from copy import deepcopy

import numpy as np

from ...config.bs_config import BaseStationConfig
from ...config.ran_config import RadioAccessNetworkConfig
from ...geometry.simulation_geometry import SimulationGeometry
from ...utils.types import FloatArr
from ..antennas.abstract_ant import AbstractAntenna
from ..antennas.ant_array import PanelArrayAnt
from .bs import BaseStation


class BaseStationCollection:
    """Container for multiple BSs with vectorized aggregate operations."""

    # Reference switches used by the isolated runtime benchmark.
    COMBINED_BEAM_GAINS = True
    GROUPED_BEAM_GAINS = True

    def __init__(
        self,
        config: "RadioAccessNetworkConfig",
        geo_handler: SimulationGeometry,
        resource_storage: str = "legacy",
        beam_dtype: str | np.dtype = "float64",
    ):
        self.ran_config = config
        self.resource_storage = resource_storage
        self.beam_dtype = np.dtype(beam_dtype)

        self.n_macro_bs = geo_handler.layout.n_macro_bs
        self.n_micro_bs = geo_handler.layout.n_micro_bs
        self.macro_bs_list, self.micro_bs_list = self._build_base_stations(geo_handler)

        # Derived attributes
        # Scalar attributes
        self.n_bs = self.n_macro_bs + self.n_micro_bs
        self.n_cells = sum(bs.n_sectors for bs in self.bs_list)

        # Shape (n_bs,) arrays
        self.n_cells_per_bs = [bs.n_sectors for bs in self.bs_list]
        self.bs_heights = np.array([bs.pos.height for bs in self.bs_list])

        # Shape (n_cells,) arrays
        self.p_tx_dbm_cells = np.array(
            [bs.p_tx_dbm for bs in self.bs_list for _ in range(bs.n_sectors)]
        )
        self.nf_db_cells = np.array(
            [bs.rx_nf_db for bs in self.bs_list for _ in range(bs.n_sectors)]
        )
        groups: dict[tuple, list[tuple[int, int, AbstractAntenna]]] = {}
        cell = 0
        for bs_index, bs in enumerate(self.bs_list):
            for ant in bs._sector_ant_list:
                # Only identical built-in panels may share an evaluation.
                key = (
                    (
                        ant.config.model_dump_json(),
                        ant.fc_ghz,
                        ant.slant_deg,
                        ant.beam_dtype.str,
                    )
                    if type(ant) is PanelArrayAnt
                    else (cell,)
                )
                groups.setdefault(key, []).append((cell, bs_index, ant))
                cell += 1
        self._gain_groups = list(groups.values())
        self._gain_group_indices = [
            (
                np.array([item[0] for item in group], dtype=np.intp),
                np.array([item[1] for item in group], dtype=np.intp),
            )
            for group in self._gain_groups
        ]

    def get_max_mean_ant_gains(
        self, phi_az: FloatArr, theta_el: FloatArr
    ) -> tuple[FloatArr, FloatArr]:
        """Return signal and linear-mean interference gains in cell order.

        Group only immutable, identical panel configurations; the last axis of
        inputs remains the BS axis, including when image axes are present.
        """
        if not self.COMBINED_BEAM_GAINS:
            return self.get_max_ant_gains(phi_az, theta_el), self.get_mean_ant_gains(
                phi_az, theta_el
            )
        if phi_az.shape != theta_el.shape or phi_az.shape[-1] != self.n_bs:
            raise ValueError("Expected matching angle arrays with a final BS axis")
        maxima = np.empty((*phi_az.shape[:-1], self.n_cells), dtype=np.float64)
        means = np.empty_like(maxima)
        for group, (cells, stations) in zip(
            self._gain_groups, self._gain_group_indices, strict=True
        ):
            if self.GROUPED_BEAM_GAINS:
                best, mean = group[0][2].get_max_mean_beam_gain_db(
                    phi_az[..., stations], theta_el[..., stations]
                )
                maxima[..., cells], means[..., cells] = best, mean
            else:
                for cell, station, ant in group:
                    maxima[..., cell], means[..., cell] = ant.get_max_mean_beam_gain_db(
                        phi_az[..., station], theta_el[..., station]
                    )
        return maxima, means

    def _build_base_stations(
        self, geo_handler: SimulationGeometry
    ) -> tuple[list["BaseStation"], list["BaseStation"]]:
        """Helper function to build a list of BaseStation instances."""

        def _prepare_bs_configs(
            bs_config: BaseStationConfig | list["BaseStationConfig"], n_bs: int
        ) -> list["BaseStationConfig"]:
            if isinstance(bs_config, BaseStationConfig):
                bs_config = [bs_config]
            if len(bs_config) == 1:
                return [deepcopy(bs_config[0]) for _ in range(n_bs)]
            if len(bs_config) == n_bs:
                return bs_config
            raise ValueError("Length of BS config does not match number of BSs.")

        macro_bs_configs = _prepare_bs_configs(
            self.ran_config.macro_bs, self.n_macro_bs
        )
        micro_bs_configs = _prepare_bs_configs(
            self.ran_config.micro_bs, self.n_micro_bs
        )

        macro_bs_list = [
            BaseStation(
                gnb_id=gnb_id,
                config=config_i,
                rat_config=self.ran_config.rat,
                pos_xy=xy_i,
                resource_storage=self.resource_storage,
                beam_dtype=self.beam_dtype,
            )
            for gnb_id, (xy_i, config_i) in enumerate(
                zip(geo_handler.layout.macro_bs_xy, macro_bs_configs)
            )
        ]
        micro_bs_list = [
            BaseStation(
                gnb_id=gnb_id + self.n_macro_bs,
                config=config_i,
                rat_config=self.ran_config.rat,
                pos_xy=xy_i,
                resource_storage=self.resource_storage,
                beam_dtype=self.beam_dtype,
            )
            for gnb_id, (xy_i, config_i) in enumerate(
                zip(geo_handler.layout.micro_bs_xy, micro_bs_configs)
            )
        ]

        return macro_bs_list, micro_bs_list

    def reset(self, seed: int | None = None) -> None:
        """Reset all base stations to their initial state.

        Parameters
        ----------
        seed : int or None, optional
            Seed forwarded to every base station. The current base-station
            implementation has no dynamic state and ignores it.
        """
        for bs in self.bs_list:
            bs.reset(seed=seed)

    def get_max_ant_gains(self, phi_az: FloatArr, theta_el: FloatArr) -> FloatArr:
        """Return the best-beam gain for all cells.

        Parameters
        ----------
        phi_az : FloatArr
            Global azimuth angles in degrees, with base station on the final
            axis. The array shape must match ``theta_el``.
        theta_el : FloatArr
            Global elevation angles in degrees, with base station on the final
            axis. The array shape must match ``phi_az``.

        Returns
        -------
        FloatArr
            Maximum gains in dB, with cells on the final axis.
        """

        return np.concatenate(
            [
                np.moveaxis(  # Move 'cell' axis to last position
                    bs.get_max_beam_gain_of_all_sectors(
                        phi_az[..., idx], theta_el[..., idx]
                    ),
                    0,
                    -1,
                )
                for idx, bs in enumerate(self.bs_list)
            ],
            axis=-1,
        )

    def get_mean_ant_gains(self, phi_az: FloatArr, theta_el: FloatArr) -> FloatArr:
        """Return the mean beam gain for all cells.

        Parameters
        ----------
        phi_az : FloatArr
            Global azimuth angles in degrees, with base station on the final
            axis. The array shape must match ``theta_el``.
        theta_el : FloatArr
            Global elevation angles in degrees, with base station on the final
            axis. The array shape must match ``phi_az``.

        Returns
        -------
        FloatArr
            Mean gains in dB, with cells on the final axis.
        """
        return np.concatenate(
            [
                np.moveaxis(  # Move 'cell' axis to last position
                    bs.get_mean_ant_gain_of_all_sectors(
                        phi_az[..., idx], theta_el[..., idx]
                    ),
                    0,
                    -1,
                )
                for idx, bs in enumerate(self.bs_list)
            ],
            axis=-1,
        )

    def get_beam_gains(self, phi_az: FloatArr, theta_el: FloatArr) -> list[FloatArr]:
        """Return all beam gains grouped by base station.

        Parameters
        ----------
        phi_az : FloatArr
            Global azimuth angles in degrees, with base station on the final
            axis. The array shape must match ``theta_el``.
        theta_el : FloatArr
            Global elevation angles in degrees, with base station on the final
            axis. The array shape must match ``phi_az``.

        Returns
        -------
        list of FloatArr
            One gain array in dB per base station. A list is returned because
            sectors may use different numbers of beams.
        """
        return [  # No of beams per sector can vary, so return list
            bs.get_all_beam_gains_of_all_sectors(phi_az[..., idx], theta_el[..., idx])
            for idx, bs in enumerate(self.bs_list)
        ]

    @property
    def bs_list(self) -> list["BaseStation"]:
        """Get a combined list of all BSs."""
        return self.macro_bs_list + self.micro_bs_list

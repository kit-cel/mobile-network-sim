"""This module defines the BaseStation class, which represents a base station in a network."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...config.antenna_array_config import AntennaArrayConfig
from ...utils.types import FloatArr
from ..antennas.abstract_ant import AbstractAntenna
from ..antennas.ant_array import PanelArrayAnt
from ..antennas.ant_omnidir import OmnidirAnt

if TYPE_CHECKING:
    from ...config.bs_config import BaseStationConfig
    from ...config.rat_config import RATConfig


def generate_pci(site_id: int, cell_idx: int, max_cells_per_site: int) -> int:
    """
    Generate a PCI (Physical Cell ID) based on the site ID and cell index.

    Parameters
    ----------
    site_id : int
        The ID of the site.
    cell_idx : int
        The index of the cell within the site.
    max_cells_per_site : int
        The maximum number of cells per site.

    Returns
    -------
    int
        The generated PCI.
    """
    cell_id_bit_shift = int(np.ceil(np.log2(max_cells_per_site)))
    return (site_id << cell_id_bit_shift) | cell_idx


@dataclass(slots=True)
class BaseStationPosition:
    """
    Data class for BS position information.

    Attributes
    ----------
    height : float
        Height of the BS in meters (antenna height).
    xy : FloatArr
        2D position of the BS. Shape: (2,)
    xy_combined : FloatArr
        Combined 2D positions of the original BS and virtual BSs. Shape: (1 + n_virtual_bs, 2)
    xyz : FloatArr
        3D position of the original BS. Shape: (3,)
    xyz_combined : FloatArr
        Combined 3D positions of the original BS and virtual BSs. Shape: (1 + n_virtual_bs, 3)
    """

    height: float
    xy: FloatArr
    xyz: FloatArr = field(init=False)

    def __post_init__(self):
        self.xyz = np.append(self.xy, self.height)


class BaseStation:
    """
    Base station (BS) entity class.

    Parameters
    ----------
    gnb_id : int
        Unique ID of this BS. This is not the cell ID/PCI, the PCI belongs to a
        cell, i.e., sector
    pos_xy : FloatArr
        2D position of the BS in the xy-plane. Shape: (2,)
    config : BaseStationConfig
        Configuration object for the BS.

    Attributes
    ----------
    gnb_id : int
        Unique ID of this BS. This is not the cell ID/PCI, the PCI belongs to a
        cell, i.e., sector
    config : BaseStationConfig
        Configuration object for the BS.
    rat : RATConfig
        RAT configuration extracted from the BS config.
    height : float
        Height of the BS in meters.
    p_tx_dbm : float
        Transmission power of the BS in dBm.
    noise_figure_db : float
        Noise figure of the BS receiver in dB.
    pos : BaseStationGeoInfo
        Position information of the BS, including 2D and 3D positions of the
        main BS and virtual BSs, as well as combined arrays for easy access.

    _sector_ant_list : list[AbstractAntenna]
        List of antenna objects for each sector of the BS.
    _pci_list : list[int]
        List of PCI values for each sector of the BS.
    """

    MAX_CELLS_PER_SITE: int = 6  # cell = sector

    _sector_ant_list: list[AbstractAntenna]
    _pci_list: list[int]

    def __init__(
        self,
        gnb_id: int,
        config: "BaseStationConfig",
        rat_config: "RATConfig",
        pos_xy: FloatArr,
        resource_storage: str = "legacy",
        beam_dtype: str | np.dtype = "float64",
    ):
        self.gnb_id = gnb_id
        self.resource_storage = resource_storage
        self.beam_dtype = np.dtype(beam_dtype)
        self.config = config
        self.rat_config = rat_config

        # BS parameters
        self.p_tx_dbm = config.p_tx_dbm
        self.rx_nf_db = config.rx_noise_figure_db

        # Init 2D and 3D positions including virtual BSs (for wraparound)
        self.pos = BaseStationPosition(height=config.height_m, xy=pos_xy)

        # Create antennas for each sector
        self._create_antennas()

    def _create_antennas(self) -> None:
        self._sector_ant_list = []
        self._pci_list = []

        sector_az_off_deg = self.config.sector_azimuth_offset_deg

        for idx in range(self.config.n_sectors):
            pci = generate_pci(self.gnb_id, idx, self.MAX_CELLS_PER_SITE)
            self._pci_list.append(pci)

            ant_config = deepcopy(self.config.ant)
            ant: AbstractAntenna

            if ant_config.ant_name == OmnidirAnt.NAME:
                ant = OmnidirAnt(pci=pci, config=ant_config)
            elif ant_config.ant_name == PanelArrayAnt.NAME:
                if not isinstance(ant_config, AntennaArrayConfig):
                    raise ValueError("Config is not of type 'AntennaArrayConfig'.")
                ant_config.phi_sector_az_bearing_deg = (
                    sector_az_off_deg + 360 * idx / self.config.n_sectors
                )
                ant = PanelArrayAnt(
                    pci=pci,
                    config=ant_config,
                    fc_ghz=self.rat_config.fc_ghz,
                    resource_storage=self.resource_storage,
                    beam_dtype=self.beam_dtype,
                )
            else:
                raise ValueError(f"Unknown antenna: {ant_config.ant_name}")

            self._sector_ant_list.append(ant)

    def reset(self, seed: int | None = None) -> None:
        """Reset the base-station state.

        Parameters
        ----------
        seed : int or None, optional
            Seed reserved for resettable base-station components. The current
            base-station implementation has no dynamic state and ignores it.
        """

    def get_max_beam_gain_of_all_sectors(
        self, phi_az: FloatArr, theta_el: FloatArr
    ) -> FloatArr:
        """Return the best-beam gain for every sector.

        Parameters
        ----------
        phi_az : FloatArr
            Global azimuth angles in degrees. The array shape must match
            ``theta_el``.
        theta_el : FloatArr
            Global elevation angles in degrees. The array shape must match
            ``phi_az``.

        Returns
        -------
        FloatArr
            Maximum antenna gains in dB, with shape
            ``(n_sectors, *phi_az.shape)``.
        """
        gain_db = np.array(
            [
                ant.get_max_beam_gain_db(phi_az, theta_el)
                for ant in self._sector_ant_list
            ]
        )
        return gain_db

    def get_mean_ant_gain_of_all_sectors(
        self, phi_az: FloatArr, theta_el: FloatArr
    ) -> FloatArr:
        """Return the mean beam gain for every sector.

        Parameters
        ----------
        phi_az : FloatArr
            Global azimuth angles in degrees. The array shape must match
            ``theta_el``.
        theta_el : FloatArr
            Global elevation angles in degrees. The array shape must match
            ``phi_az``.

        Returns
        -------
        FloatArr
            Mean antenna gains in dB, with shape
            ``(n_sectors, *phi_az.shape)``.
        """
        mean_gain_db = np.array(
            [
                ant.get_mean_beam_gain_db(phi_az, theta_el)
                for ant in self._sector_ant_list
            ]
        )
        return mean_gain_db

    def get_all_beam_gains_of_all_sectors(
        self, phi_az: FloatArr, theta_el: FloatArr
    ) -> FloatArr:
        """Return every beam gain for every sector.

        Parameters
        ----------
        phi_az : FloatArr
            Global azimuth angles in degrees. The array shape must match
            ``theta_el``.
        theta_el : FloatArr
            Global elevation angles in degrees. The array shape must match
            ``phi_az``.

        Returns
        -------
        FloatArr
            Antenna gains in dB. The leading dimension enumerates sectors and
            the following dimensions contain beams and the input angle shape.
        """
        all_beam_gains_db = np.array(
            [
                ant.get_all_beam_gains_db(phi_az, theta_el)
                for ant in self._sector_ant_list
            ]
        )
        return all_beam_gains_db

    def get_pci_list(self) -> list[int]:
        """Return the physical cell identifiers of all sectors.

        Returns
        -------
        list of int
            Physical cell identifiers in sector order.
        """
        return self._pci_list

    @property
    def n_sectors(self) -> int:
        """Return the number of sectors of this BS."""
        if self._sector_ant_list is None:
            raise ValueError("Sector antenna list is not initialized.")
        return len(self._sector_ant_list)

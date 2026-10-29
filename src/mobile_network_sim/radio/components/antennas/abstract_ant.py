"""Abstract antenna class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...config.antenna_config import AntennaConfig


class AbstractAntenna(ABC):
    """Abstract antenna class."""

    NAME: str

    PHI_3dB: float
    THETA_3dB: float

    def __init__(self, pci: int, config: "AntennaConfig") -> None:
        self.pci = pci
        self.config = config
        self.g_max_dbi = config.g_max_dbi

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    def get_max_mean_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return maximum and linear-mean gains, expressed in dB.

        Antennas without a shared codebook implementation retain their existing
        individual methods through this compatibility path.
        """
        return (
            self.get_max_beam_gain_db(phi_deg, theta_deg),
            self.get_mean_beam_gain_db(phi_deg, theta_deg),
        )

    @abstractmethod
    def get_max_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """
        Get the maximum beam gain in dBi for given azimuth (phi) and elevation
        (theta) angle(s).

        Parameters
        ----------
        phi_deg : float | np.ndarray
            The azimuth angle in degrees.
        theta_deg : float | np.ndarray
            The elevation angle in degrees.

        Returns
        -------
        float | np.ndarray
            The antenna gain in dBi.
        """

    @abstractmethod
    def get_mean_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """
        Get the mean beam gain in dBi for given azimuth (phi) and elevation
        (theta) angle(s).

        Parameters
        ----------
        phi_deg : float | np.ndarray
            The azimuth angle in degrees.
        theta_deg : float | np.ndarray
            The elevation angle in degrees.

        Returns
        -------
        float | np.ndarray
            The antenna gain in dBi.
        """

    @abstractmethod
    def get_all_beam_gains_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """
        Get the beam gain in dBi for all beams in the codebook for given
        azimuth (phi) and elevation (theta) angle(s).

        Parameters
        ----------
        phi_deg : float | np.ndarray
            The azimuth angle in degrees.
        theta_deg : float | np.ndarray
            The elevation angle in degrees.

        Returns
        -------
        np.ndarray
            The antenna gain in dBi for all beams, with shape (..., num_beams).
        """

    @abstractmethod
    def get_ant_element_radiation_pattern_db(
        self,
        phi_deg: float | np.ndarray,
        theta_deg: float | np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Get the 3D radiation pattern of the antenna element in dB for given
        azimuth (phi) and elevation (theta) angle(s).

        Parameters
        ----------
        phi_deg : float | np.ndarray
            The azimuth angle in degrees.
        theta_deg : float | np.ndarray
            The elevation angle in degrees.
        normalize : bool, optional
            Whether to normalize the pattern to its maximum gain, by default False

        Returns
        -------
        float | np.ndarray
            The 3D radiation pattern in dB.
        """

    @abstractmethod
    def get_ant_composite_radiation_pattern_db(
        self,
        phi_deg: float | np.ndarray,
        theta_deg: float | np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Get the 3D radiation pattern of the antenna (including array factor if
        applicable) in dB for a given azimuth (phi) and elevation (theta) angle(s).

        Parameters
        ----------
        phi_deg : float | np.ndarray
            The azimuth angle in degrees.
        theta_deg : float | np.ndarray
            The elevation angle in degrees.
        normalize : bool, optional
            Whether to normalize the pattern to its maximum gain, by default False

        Returns
        -------
        float | np.ndarray
            The 3D radiation pattern in dB.
        """

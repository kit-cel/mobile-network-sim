"""Omnidirectional antenna class."""

from typing import TYPE_CHECKING
import warnings

import numpy as np

from .abstract_ant import AbstractAntenna

if TYPE_CHECKING:
    from ...config.antenna_config import AntennaConfig


class OmnidirAnt(AbstractAntenna):
    """Ideal omnidirectional antenna with constant gain."""

    def __init__(self, *, pci: int, config: "AntennaConfig") -> None:
        """
        Initialize the omnidirectional antenna.

        Parameters
        ----------
        pci : int
            Unique cell ID (physical cell ID) of the cell, this antenna belongs to.
        config : AntennaConfig
            Antenna configuration.
        """
        super().__init__(pci=pci, config=config)

        if self.g_max_dbi != 0:
            warnings.warn(
                "Omnidirectional antenna gain is not zero. This may lead to unexpected behavior."
            )

    def get_max_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """Return the constant maximum beam gain.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Zenith angles in degrees, broadcastable with ``phi_deg``.

        Returns
        -------
        numpy.ndarray
            Constant gain in dB with the broadcast input shape.
        """
        return self.get_ant_element_radiation_pattern_db(
            phi_deg, theta_deg, normalize=False
        )

    def get_mean_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """Return the constant mean beam gain.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Zenith angles in degrees, broadcastable with ``phi_deg``.

        Returns
        -------
        numpy.ndarray
            Constant gain in dB with the broadcast input shape.
        """
        return self.get_max_beam_gain_db(phi_deg, theta_deg)

    def get_all_beam_gains_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """Return the gain for the antenna's single effective beam.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Zenith angles in degrees, broadcastable with ``phi_deg``.

        Returns
        -------
        numpy.ndarray
            Constant gain in dB with the broadcast input shape.
        """
        return self.get_max_beam_gain_db(phi_deg, theta_deg)

    def get_ant_element_radiation_pattern_db(
        self,
        phi_deg: float | np.ndarray,
        theta_deg: float | np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Evaluate the omnidirectional element pattern.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Zenith angles in degrees, broadcastable with ``phi_deg``.
        normalize : bool, optional
            If ``True``, return zero dB. Otherwise include the configured
            maximum element gain.

        Returns
        -------
        numpy.ndarray
            Element gain in dB with the broadcast input shape.
        """
        phi_deg, _ = np.broadcast_arrays(phi_deg, theta_deg)
        pattern = np.zeros_like(phi_deg, dtype=float)

        if not normalize:
            pattern += self.g_max_dbi
        return pattern

    def get_ant_composite_radiation_pattern_db(
        self,
        phi_deg: float | np.ndarray,
        theta_deg: float | np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Evaluate the omnidirectional composite radiation pattern.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Zenith angles in degrees, broadcastable with ``phi_deg``.
        normalize : bool, optional
            If ``True``, return zero dB. Otherwise include the configured
            maximum element gain.

        Returns
        -------
        numpy.ndarray
            Composite gain in dB with the broadcast input shape.
        """
        return self.get_ant_element_radiation_pattern_db(
            phi_deg, theta_deg, normalize=normalize
        )

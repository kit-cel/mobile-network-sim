"""Panel arrays with an ETSI TR 138 901 single-element power pattern.

Panel geometry and the single-element radiation power pattern follow clause
7.3 and Table 7.3-1 of ETSI TR 138 901 V18.0.0. Beam-codebook construction and
beam selection are simulator-specific and are not standardized procedures.
"""

from copy import deepcopy
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ...geometry.geo_utils import gcs_to_lcs, lcs_to_gcs, xy_to_global_angles
from ...utils import commuinations as utc
from ...utils.consts import LIGHT_SPEED_MPS
from .abstract_ant import AbstractAntenna

if TYPE_CHECKING:
    from ...config.antenna_array_config import AntennaArrayConfig

NUMERIC_EPS = 1e-12


class PanelArrayAnt(AbstractAntenna):
    """Model a rectangular panel array with fixed analog steering beams.

    The single-element power pattern follows ETSI TR 138 901 V18.0.0,
    Table 7.3-1. The finite steering codebook is a simulator-specific layer on
    top of that element pattern.

    Notes
    -----
    The panel lies in the local y-z plane and its broadside points along local
    +x. Local ``phi`` is azimuth around z and local ``theta`` is zenith, with
    90 degrees on the horizon. Globally, ``phi`` is azimuth from +x and
    ``theta`` is again the zenith angle.
    """

    cell_id: int
    config: "AntennaArrayConfig"

    slant_deg: float = 0.0  # Slant angle for arrays (not implemented)

    def __init__(
        self,
        pci: int,
        config: "AntennaArrayConfig",
        fc_ghz: float,
        *,
        resource_storage: str = "legacy",
        beam_dtype: str | np.dtype = "float64",
    ) -> None:
        super().__init__(pci=pci, config=config)
        self.beam_dtype = np.dtype(beam_dtype)
        if self.beam_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("beam_dtype must be float32 or float64")
        self.complex_dtype = np.dtype(
            "complex64" if self.beam_dtype.itemsize == 4 else "complex128"
        )
        self.beam_config = config.beamforming
        self.fc_ghz = fc_ghz
        if not 0 < self.fc_ghz < 100.0:
            raise ValueError("fc_ghz must be between 0 and 100.0")
        self.wavelength_m = LIGHT_SPEED_MPS / (self.fc_ghz * 1e9)
        self.wavenumber_rad_per_m = 2.0 * np.pi / self.wavelength_m

        self.elem_spacing_h_m = config.element_spacing_h_lambda * self.wavelength_m
        self.elem_spacing_v_m = config.element_spacing_v_lambda * self.wavelength_m

        self.sector_azimuth_deg = config.phi_sector_az_bearing_deg
        self.mechanical_downtilt_deg = config.mechanical_downtilt_deg

        if resource_storage == "shared_mmap":
            from ...resources import ArrayResource, shared_arrays

            def build():
                self._prepare_static_arrays()
                return ArrayResource(
                    {
                        name: getattr(self, name)
                        for name in (
                            "elem_y_m",
                            "elem_z_m",
                            "codebook_weights",
                            "_conjugate_weights",
                        )
                    }
                )

            self._shared_resource = shared_arrays(
                (
                    "panel",
                    config.model_dump_json(),
                    float(fc_ghz),
                    self.slant_deg,
                    self.beam_dtype.str,
                ),
                build,
            )
            self.__dict__.update(self._shared_resource.arrays)
        else:
            self._prepare_static_arrays()

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        resource = getattr(self, "_shared_resource", None)
        if resource is not None:
            for array in resource.arrays.values():
                memo[id(array)] = array
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def _prepare_static_arrays(self):
        self.elem_y_m, self.elem_z_m = self._calculate_element_positions()
        self.codebook_weights = self._build_codebook_weights()
        self.codebook_weights.flags.writeable = False
        self._conjugate_weights = np.conj(self.codebook_weights)
        self._conjugate_weights.flags.writeable = False

    def _calculate_element_positions(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        elem_h_idx = np.tile(
            np.arange(self.config.n_element_h, dtype=self.beam_dtype),
            self.config.m_element_v,
        )
        elem_v_idx = np.repeat(
            np.arange(self.config.m_element_v, dtype=self.beam_dtype),
            self.config.n_element_h,
        )

        elem_h_center = 0.5 * (self.config.n_element_h - 1)
        elem_v_center = 0.5 * (self.config.m_element_v - 1)

        elem_y_m = (elem_h_idx - elem_h_center) * self.elem_spacing_h_m
        elem_z_m = (elem_v_idx - elem_v_center) * self.elem_spacing_v_m

        return elem_y_m, elem_z_m

    # Array steering and codebook weights --------------------------------------
    def steering_vector(
        self, phi_deg: NDArray[np.float64], theta_deg: NDArray[np.float64]
    ) -> NDArray[np.complexfloating]:
        """Compute the local array steering vector.

        For a panel in local y-z plane and local broadside along +x:
            u_y = sin(theta) * sin(phi)
            u_z = cos(theta)

        Parameters
        ----------
        phi_deg : numpy.ndarray
            Local azimuth angles in degrees.
        theta_deg : numpy.ndarray
            Local zenith angles in degrees.

        Returns
        -------
        numpy.ndarray
            Complex steering vectors with shape ``(n_elements, n_angles)``.
        """
        phi_rad = np.deg2rad(np.asarray(phi_deg, dtype=self.beam_dtype).ravel())
        theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=self.beam_dtype).ravel())

        u_y = np.sin(theta_rad) * np.sin(phi_rad)
        u_z = np.cos(theta_rad)

        phase_rad = self.wavenumber_rad_per_m * (
            self.elem_y_m[:, None] * u_y[None, :]
            + self.elem_z_m[:, None] * u_z[None, :]
        )
        return np.exp(1j * phase_rad)

    def build_steering_weights(
        self, beam_azimuth_deg: float, beam_elevation_deg: float
    ) -> NDArray[np.complexfloating]:
        """
        Build fixed beam weights in local coordinates.

        Parameters
        ----------
        beam_azimuth_deg : float
            Azimuth relative to local broadside.
        beam_elevation_deg : float
            Elevation relative to local horizon.

        Returns
        -------
        numpy.ndarray
            Unit-norm complex weights with one value per antenna element.
        """
        beam_zenith_deg = 90.0 - beam_elevation_deg
        steering_vec = self.steering_vector(
            phi_deg=np.array([beam_azimuth_deg], dtype=np.float64),
            theta_deg=np.array([beam_zenith_deg], dtype=np.float64),
        )[:, 0]

        return (steering_vec / np.linalg.norm(steering_vec)).astype(
            self.complex_dtype, copy=False
        )

    def _build_codebook_weights(self) -> NDArray[np.complexfloating]:
        if self.beam_config is None:
            if self.n_elements > 1:
                warn(
                    "No beamforming config provided. Using single broadside beam,"
                    " which may lead to low gain for off-broadside angles."
                )
            return self.build_steering_weights(0.0, 0.0)[None, :]

        beam_azimuths_deg = self.beam_config.az_list_deg
        beam_elevations_deg = self.beam_config.el_list_deg

        codebook_weights = np.empty(
            (len(beam_azimuths_deg), self.n_elements), dtype=self.complex_dtype
        )
        for beam_idx, (azimuth_deg, elevation_deg) in enumerate(
            zip(beam_azimuths_deg, beam_elevations_deg, strict=True)
        ):
            codebook_weights[beam_idx, :] = self.build_steering_weights(
                azimuth_deg,
                elevation_deg,
            )
        return codebook_weights

    # 3GPP element pattern and gain --------------------------------------------
    def element_gain_db(
        self, phi_lcs_deg: NDArray[np.float64], theta_lcs_deg: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Calculate single-element gain for local spherical angles.

        Parameters
        ----------
        phi_lcs_deg : numpy.ndarray
            Local azimuth :math:`\phi'` in degrees.
        theta_lcs_deg : numpy.ndarray
            Local zenith angle :math:`\theta'` in degrees.

        Returns
        -------
        numpy.ndarray
            Element gain in dBi, broadcast to the input-angle shape.

        Notes
        -----
        The vertical and horizontal attenuation cuts are

        .. math::

           A_V(\theta') = -\min\left(
               12\left(\frac{\theta'-90^\circ}{\theta_{3\mathrm{dB}}}
               \right)^2,\, \mathrm{SLA}_V\right)

        and

        .. math::

           A_H(\phi') = -\min\left(
               12\left(\frac{\phi'}{\phi_{3\mathrm{dB}}}\right)^2,
               \, A_{\max}\right).

        The combined gain is

        .. math::

           G(\theta',\phi') = G_{\max} -
               \min\left(-[A_V(\theta')+A_H(\phi')],\, A_{\max}\right).

        References
        ----------
        .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
           100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
           ETSI TR 138 901 V18.0.0, Table 7.3-1, May 2024.
           https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
           18.00.00_60/tr_138901v180000p.pdf
        """
        horiz_pattern_db = self._horizontal_element_pattern_db(phi_lcs_deg)
        vert_pattern_db = self._vertical_element_pattern_db(theta_lcs_deg)
        elem_pattern_db = -np.minimum(
            -(horiz_pattern_db + vert_pattern_db),
            self.config.front_back_attenuation_db,
        )
        return self.g_max_dbi + elem_pattern_db

    def _horizontal_element_pattern_db(
        self, phi_lcs_deg: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return -np.minimum(
            12.0 * (phi_lcs_deg / self.config.azimuth_3db_bw_deg) ** 2,
            self.config.front_back_attenuation_db,
        )

    def _vertical_element_pattern_db(
        self, theta_lcs_deg: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return -np.minimum(
            12.0 * ((theta_lcs_deg - 90.0) / self.config.elevation_3db_bw_deg) ** 2,
            self.config.vertical_sidelobe_attenuation_db,
        )

    # Composite array gain and codebook evaluation -----------------------------
    def _compute_beam_gain_db_of_codebook_beam(
        self, phi_global_deg: np.ndarray, theta_global_deg: np.ndarray, beam_idx: int
    ) -> np.ndarray:
        """
        Compute the directional gain for a single beam from the beamforming codebook.

        The gain is computed as the sum of the element radiation pattern and the
        array factor corresponding to the selected beam index.

        Parameters
        ----------
        phi_global_deg : np.ndarray
            Azimuth angles in the global coordinate system (GCS) in degrees.
            Shape: (n_points,).
        theta_global_deg : np.ndarray
            Elevation (zenith) angles in the global coordinate system (GCS) in degrees.
            Shape: (n_points,).
        beam_idx : int
            Index of the beam in the codebook. Must satisfy
            ``0 <= beam_idx < n_beams``.

        Returns
        -------
        np.ndarray
            Directional gain of the selected beam in dB for each input angle pair.
            Shape: (n_points,).
        """
        if not 0 <= beam_idx < self.codebook_weights.shape[0]:
            raise ValueError("beam_idx out of range.")

        phi_global_deg, theta_global_deg = np.broadcast_arrays(
            np.asarray(phi_global_deg, dtype=np.float64),
            np.asarray(theta_global_deg, dtype=np.float64),
        )
        input_shape = phi_global_deg.shape

        phi_local_deg, theta_local_deg = gcs_to_lcs(
            phi_gcs=phi_global_deg,
            theta_gcs=theta_global_deg,
            bearing=self.sector_azimuth_deg,
            downtilt=self.mechanical_downtilt_deg,
            slant=self.slant_deg,
        )

        elem_gain_db = self.element_gain_db(phi_local_deg, theta_local_deg).ravel()
        steering_vec = self.steering_vector(
            phi_local_deg.ravel(), theta_local_deg.ravel()
        )
        beam_weights = self.codebook_weights[beam_idx, :]
        array_factor_lin = np.abs(np.conj(beam_weights) @ steering_vec) ** 2

        gain_db = elem_gain_db + 10.0 * np.log10(
            np.maximum(array_factor_lin, NUMERIC_EPS)
        )
        return gain_db.reshape(input_shape)

    def _compute_all_codebook_gains_db(
        self, phi_global_deg: np.ndarray, theta_global_deg: np.ndarray
    ) -> np.ndarray:
        """
        Compute the directional gains for all beams in the codebook.

        The gain is computed for each beam as the sum of the element radiation
        pattern and the corresponding array factor.

        Parameters
        ----------
        phi_global_deg : np.ndarray
            Azimuth angles in the global coordinate system (GCS) in degrees.
            Shape: ``input_shape``.
        theta_global_deg : np.ndarray
            Elevation (zenith) angles in the global coordinate system (GCS) in degrees.
            Shape: ``input_shape``.

        Returns
        -------
        np.ndarray
            Directional gains in dB for all beams and input angle pairs.
            Shape: ``(n_beams, *input_shape)``, where each leading row
            corresponds to one beam.
        """
        element, factors = self._compute_codebook_factors(
            phi_global_deg, theta_global_deg
        )
        return element[None, ...] + 10.0 * np.log10(factors)

    def _compute_codebook_factors(self, phi_global_deg, theta_global_deg):
        """Return the element dB gain and clipped linear beam factors."""
        phi_global_deg, theta_global_deg = np.broadcast_arrays(
            np.asarray(phi_global_deg, dtype=np.float64),
            np.asarray(theta_global_deg, dtype=np.float64),
        )
        input_shape = phi_global_deg.shape

        phi_local_deg, theta_local_deg = gcs_to_lcs(
            phi_gcs=phi_global_deg,
            theta_gcs=theta_global_deg,
            bearing=self.sector_azimuth_deg,
            downtilt=self.mechanical_downtilt_deg,
            slant=self.slant_deg,
        )

        elem_gain_db = self.element_gain_db(phi_local_deg, theta_local_deg).ravel()
        steering_vec = self.steering_vector(
            phi_local_deg.ravel(), theta_local_deg.ravel()
        )

        array_factor_lin = np.abs(self._conjugate_weights @ steering_vec) ** 2
        factors = np.maximum(array_factor_lin, NUMERIC_EPS)
        return elem_gain_db.reshape(input_shape), factors.reshape(
            (self.codebook_weights.shape[0], *input_shape)
        )

    def _compute_max_beam_gain_and_beam_idx(
        self, phi_global_deg: np.ndarray, theta_global_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the maximum achievable beam gain and corresponding beam index
        for each direction using the beamforming codebook.

        For each input angle pair, the gain is evaluated for all beams and the
        beam yielding the maximum gain is selected.

        Parameters
        ----------
        phi_global_deg : np.ndarray
            Azimuth angles in the global coordinate system (GCS) in degrees.
            Shape: (n_points,).
        theta_global_deg : np.ndarray
            Elevation (zenith) angles in the global coordinate system (GCS) in degrees.
            Shape: (n_points,).

        Returns
        -------
        best_gain_db : np.ndarray
            Maximum directional gain in dB for each input angle pair.
            Shape: (n_points,).
        best_beam_idx : np.ndarray
            Index of the beam achieving the maximum gain for each input angle pair.
            Shape: (n_points,). Each entry satisfies
            ``0 <= best_beam_idx[i] < n_beams``.
        """
        all_gains_db = self._compute_all_codebook_gains_db(
            phi_global_deg, theta_global_deg
        )
        best_beam_idx = np.argmax(all_gains_db, axis=0)
        best_gain_db = np.take_along_axis(
            all_gains_db,
            best_beam_idx[None, ...],
            axis=0,
        )[0]
        return best_gain_db, best_beam_idx

    def _compute_mean_beam_gain(
        self, phi_global_deg: np.ndarray, theta_global_deg: np.ndarray
    ) -> np.ndarray:
        """
        Compute the mean beam gain across the codebook for each direction.

        For each input angle pair, the gain is evaluated for all beams and the
        mean gain is calculated.

        Parameters
        ----------
        phi_global_deg : np.ndarray
            Azimuth angles in the global coordinate system (GCS) in degrees.
            Shape: (n_points,).
        theta_global_deg : np.ndarray
            Elevation (zenith) angles in the global coordinate system (GCS) in degrees.
            Shape: (n_points,).

        Returns
        -------
        mean_gain_db : np.ndarray
            Mean directional gain in dB across all beams for each input angle pair.
            Shape: (n_points,).
        """
        all_gains_db = self._compute_all_codebook_gains_db(
            phi_global_deg, theta_global_deg
        )
        mean_gain_db = utc.lin_to_db(np.mean(utc.db_to_lin(all_gains_db), axis=0))
        return mean_gain_db

    # Beam center lines projected to the UE plane ------------------------------
    def beam_footprint_points(
        self, h_ant_m: float, h_map_m: float
    ) -> list[tuple[float, float, int]]:
        """Return projected beam-center points on the UE plane.

        Parameters
        ----------
        h_ant_m : float
            Antenna height in meters.
        h_map_m : float
            Height of the projection plane in meters.

        Returns
        -------
        list[tuple[float, float, int]]
            Tuples containing x-coordinate, y-coordinate, and beam index.
        """
        if self.beam_config is None:
            return []

        footprint_points: list[tuple[float, float, int]] = []

        delta_height_m = h_ant_m - h_map_m

        phi_global_deg, theta_global_deg = lcs_to_gcs(
            np.asarray(self.beam_config.az_list_deg),
            90.0 - np.asarray(self.beam_config.el_list_deg),
            bearing=self.sector_azimuth_deg,
            downtilt=self.mechanical_downtilt_deg,
            slant=self.slant_deg,
        )

        for beam_idx, (beam_phi_global_deg, beam_theta_global_deg) in enumerate(
            zip(phi_global_deg, theta_global_deg, strict=True)
        ):
            beam_elevation_global_deg = 90.0 - beam_theta_global_deg
            if beam_elevation_global_deg >= 0.0:
                continue

            ground_range_m = delta_height_m / np.tan(
                np.deg2rad(-beam_elevation_global_deg)
            )
            x_end_m = ground_range_m * np.cos(np.deg2rad(beam_phi_global_deg))
            y_end_m = ground_range_m * np.sin(np.deg2rad(beam_phi_global_deg))
            footprint_points.append((x_end_m, y_end_m, beam_idx))

        return footprint_points

    # Ground-plane gain map for visualization ----------------------------------
    def best_codebook_gain_map_db(
        self, x_grid_m: np.ndarray, y_grid_m: np.ndarray, h_ant_m: float, h_map_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the best codebook gain map for visualization.

        Parameters
        ----------
        x_grid_m : np.ndarray
            2D array of x coordinates on the ground plane.
        y_grid_m : np.ndarray
            2D array of y coordinates on the ground plane.
        h_ant_m : float
            Height of the antenna above the ground plane.
        h_map_m : float
            Height of the map points (usually the UE height).

        Returns
        -------
        best_gain_db : np.ndarray
            2D array of the best codebook gain in dB at each (x, y) point.
        best_beam_idx : np.ndarray
            2D array of the index of the best beam for each (x, y) point.
        """
        phi_global_deg, theta_global_deg = xy_to_global_angles(
            x_grid_m, y_grid_m, h_ant_m, h_map_m
        )
        best_gain_db, best_beam_idx = self._compute_max_beam_gain_and_beam_idx(
            phi_global_deg=phi_global_deg.ravel(),
            theta_global_deg=theta_global_deg.ravel(),
        )
        return best_gain_db.reshape(x_grid_m.shape), best_beam_idx.reshape(
            x_grid_m.shape
        )

    # Abstract method implementations ------------------------------------------
    def get_max_mean_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the codebook once for signal and interference gains.

        Reduce clipped linear factors before conversion to dB. The public
        beam-index API retains its first-beam tie convention.
        """
        element, factors = self._compute_codebook_factors(
            np.asarray(phi_deg), np.asarray(theta_deg)
        )
        return (
            element + 10.0 * np.log10(np.max(factors, axis=0)),
            element + 10.0 * np.log10(np.mean(factors, axis=0)),
        )

    def get_max_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """Calculate the maximum gain across the beam codebook.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Global azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Global zenith angles in degrees.

        Returns
        -------
        numpy.ndarray
            Maximum gain in dB for each input angle.
        """
        gain_db, _ = self._compute_max_beam_gain_and_beam_idx(
            np.asarray(phi_deg), np.asarray(theta_deg)
        )
        return gain_db

    def get_mean_beam_gain_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """Calculate the mean gain across the beam codebook.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Global azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Global zenith angles in degrees.

        Returns
        -------
        numpy.ndarray
            Mean gain in dB for each input angle.
        """
        mean_gain_db = self._compute_mean_beam_gain(
            np.asarray(phi_deg), np.asarray(theta_deg)
        )
        return mean_gain_db

    def get_all_beam_gains_db(
        self, phi_deg: float | np.ndarray, theta_deg: float | np.ndarray
    ) -> np.ndarray:
        """Calculate gains for every beam in the codebook.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Global azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Global zenith angles in degrees.

        Returns
        -------
        numpy.ndarray
            Beam gains in dB for each input angle.
        """
        all_gains_db = self._compute_all_codebook_gains_db(
            np.asarray(phi_deg), np.asarray(theta_deg)
        )
        return all_gains_db

    def get_ant_element_radiation_pattern_db(
        self,
        phi_deg: float | np.ndarray,
        theta_deg: float | np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Evaluate the element radiation pattern at global angles.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Global azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Global zenith angles in degrees.
        normalize : bool, optional
            Whether to shift the maximum returned gain to zero dB.

        Returns
        -------
        numpy.ndarray
            Element gain in dB for each input angle.
        """
        phi_local_deg, theta_local_deg = gcs_to_lcs(
            phi_gcs=np.asarray(phi_deg),
            theta_gcs=np.asarray(theta_deg),
            bearing=self.sector_azimuth_deg,
            downtilt=self.mechanical_downtilt_deg,
            slant=self.slant_deg,
        )
        pattern = self.element_gain_db(phi_local_deg, theta_local_deg)
        if normalize:
            pattern -= np.max(pattern)
        return pattern

    def get_ant_composite_radiation_pattern_db(
        self,
        phi_deg: float | np.ndarray,
        theta_deg: float | np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Evaluate the composite array radiation pattern at global angles.

        Parameters
        ----------
        phi_deg : float | numpy.ndarray
            Global azimuth angles in degrees.
        theta_deg : float | numpy.ndarray
            Global zenith angles in degrees.
        normalize : bool, optional
            Whether to shift the maximum returned gain to zero dB.

        Returns
        -------
        numpy.ndarray
            Composite gain in dB for each input angle.
        """
        pattern, _ = self._compute_max_beam_gain_and_beam_idx(
            np.asarray(phi_deg), np.asarray(theta_deg)
        )
        if normalize:
            pattern -= np.max(pattern)
        return pattern

    # Properties ---------------------------------------------------------------
    @property
    def n_elements(self) -> int:
        """Total number of antenna elements in the panel array."""
        return self.config.m_element_v * self.config.n_element_h

    @property
    def expected_main_lobe_elevation(self) -> float:
        """Expected main lobe elevation in degrees."""
        return 90.0 - self.mechanical_downtilt_deg

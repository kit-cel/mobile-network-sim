"""Wrap-around radio topology."""

import numpy as np

from .. import geo_utils as utg
from .abstract_topology import AbstractTopology, FloatArr


class WraparoundRadioTopology(AbstractTopology):
    """Radio topology using periodic base-station images.

    Notes
    -----
    Periodic copies remove artificial edges from a finite regular layout and
    are the geometric realization of a toroidal cellular topology. This class
    exposes both the minimum-distance image and every candidate image; the LSF
    configuration determines which candidates are used for radio coupling.

    References
    ----------
    .. [1] M. Iridon and D. Matula, "A 6-Regular Torus Graph Family with
       Applications to Cellular and Interconnection Networks," *Journal of
       Graph Algorithms and Applications*, vol. 6, no. 4, pp. 373--404,
       2002, sections 2--3. doi:10.7155/jgaa.00058.
    """

    def __init__(
        self, wraparound_shifts: FloatArr, bs_xy: FloatArr | None = None
    ) -> None:
        """
        Initialize wrap-around radio topology.

        Parameters
        ----------
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, the topology will rely on the BS
            positions provided at initialization.
        wraparound_shifts : FloatArr of shape (n_shifts, 2)
            Translation vectors used to generate periodic BS images. The zero
            shift should be included.
        """
        self.wraparound_shifts = wraparound_shifts
        self.bs_xy = bs_xy

        if bs_xy is None:
            self._shifted_bs_xy = None
        else:
            self._shifted_bs_xy = bs_xy[None, :, :] + wraparound_shifts[:, None, :]

    def displacement_2d(
        self, ue_xy: FloatArr, bs_xy: FloatArr | None = None
    ) -> FloatArr:
        """
        Compute minimum-image 2D displacement vectors from BSs to UEs.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at initialization.

        Returns
        -------
        FloatArr of shape (..., n_bs, 2)
            Displacement vectors to the closest periodic BS image.
        """
        ue_xy = self._ensure_ue_xy_shape(ue_xy)
        shifted_bs_xy = self._ensure_shifted_bs_xy(bs_xy)

        disp = ue_xy[..., None, None, :] - shifted_bs_xy  # (..., n_shifts, n_bs, 2)
        dist_sq = np.sum(disp * disp, axis=-1)  # (..., n_shifts, n_bs)
        best_shift_idx = np.argmin(dist_sq, axis=-2)  #  (..., n_bs)

        best_disp = np.take_along_axis(  # Best displacement for each UE-BS pair
            disp, best_shift_idx[..., None, :, None], axis=-3
        )
        return np.squeeze(best_disp, axis=-3)

    def candidate_displacements_2d(
        self, ue_xy: FloatArr, bs_xy: FloatArr | None = None
    ) -> FloatArr:
        """
        Compute 2D displacement vectors to all periodic BS images.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at
            initialization.

        Returns
        -------
        FloatArr of shape (..., n_images, n_bs, 2)
            Displacement vectors from each periodic BS image to each UE.
        """
        ue_xy = self._ensure_ue_xy_shape(ue_xy)
        shifted_bs_xy = self._ensure_shifted_bs_xy(bs_xy)
        return ue_xy[..., None, None, :] - shifted_bs_xy

    def candidate_distances_and_angles(
        self,
        ue_xy: FloatArr,
        ue_heights: float | FloatArr,
        bs_heights: FloatArr,
        bs_xy: FloatArr | None = None,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        """
        Compute distances and angles to all periodic BS images.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        ue_heights : float | FloatArr of shape (...)
            UE heights in meters.
        bs_heights : FloatArr of shape (n_bs,)
            BS heights in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at
            initialization.

        Returns
        -------
        d_2d_m : FloatArr of shape (..., n_images, n_bs)
            2D distances in meters.
        d_3d_m : FloatArr of shape (..., n_images, n_bs)
            3D distances in meters.
        phi_az_rad : FloatArr of shape (..., n_images, n_bs)
            Azimuth angle in radians.
        theta_el_rad : FloatArr of shape (..., n_images, n_bs)
            Elevation/zenith angle in radians.
        """
        ue_xy = self._ensure_ue_xy_shape(ue_xy)
        disp_2d = self.candidate_displacements_2d(ue_xy=ue_xy, bs_xy=bs_xy)

        ue_heights = np.asarray(ue_heights, dtype=np.float64)
        if ue_heights.ndim == 0:
            ue_heights = np.full(ue_xy.shape[:-1], ue_heights, dtype=np.float64)
        else:
            ue_heights = np.reshape(ue_heights, ue_xy.shape[:-1])
        if ue_heights.shape != ue_xy.shape[:-1]:
            raise ValueError(
                f"ue_heights must have shape {ue_xy.shape[:-1]} or be scalar, "
                f"got {ue_heights.shape}."
            )

        bs_heights = np.asarray(bs_heights, dtype=np.float64)
        dz = ue_heights[..., None, None] - bs_heights[None, None, :]
        dz = np.broadcast_to(dz, disp_2d.shape[:-1])
        disp_3d = np.concatenate([disp_2d, dz[..., None]], axis=-1)

        _, az_rad_, elev_rad_ = utg.cartesian_to_spherical(disp_3d)
        phi_az_rad, theta_el_rad = utg.to_physics_convention(az_rad_, elev_rad_)

        d_2d_m = np.linalg.norm(disp_2d, axis=-1)
        d_3d_m = np.linalg.norm(disp_3d, axis=-1)

        return d_2d_m, d_3d_m, phi_az_rad, theta_el_rad

    def maybe_wraparound(
        self, ue_xy: FloatArr, bs_xy: FloatArr | None = None
    ) -> FloatArr:
        """
        Optionally wrap UE positions to the closest periodic BS image.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at initialization.

        Returns
        -------
        FloatArr of shape (..., 2)
            UE positions wrapped to the closest periodic BS image.
        """
        ue_xy = self._ensure_ue_xy_shape(ue_xy)
        if bs_xy is not None:
            bs_xy = self._ensure_bs_xy_shape(bs_xy)

        disp = self.displacement_2d(ue_xy=ue_xy, bs_xy=bs_xy)
        return ue_xy - disp

    def _ensure_shifted_bs_xy(self, bs_xy: FloatArr | None = None) -> FloatArr:
        if bs_xy is not None:
            bs_xy = self._ensure_bs_xy_shape(bs_xy)
            # Shape: (n_shifts, n_bs, 2)
            return bs_xy[None, :, :] + self.wraparound_shifts[:, None, :]
        if self._shifted_bs_xy is not None:
            return self._shifted_bs_xy
        raise ValueError(
            "BS positions must be provided either at initialization or as an argument."
        )

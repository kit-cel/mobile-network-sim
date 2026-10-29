"""Abstract radio topology interface."""

from abc import ABC, abstractmethod

import numpy as np

from .. import geo_utils as utg
from ...utils.types import FloatArr


class AbstractTopology(ABC):
    """Interface for geometric radio-distance computation."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    @abstractmethod
    def displacement_2d(
        self, ue_xy: FloatArr, bs_xy: FloatArr | None = None
    ) -> FloatArr:
        """
        Compute 2D displacement vectors from BSs to UEs.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at initialization.

        Returns
        -------
        FloatArr of shape (..., n_bs, 2)
            Displacement vectors ``ue_xy - bs_xy`` in meters.
        """

    def distances_2d(self, ue_xy: FloatArr, bs_xy: FloatArr | None = None) -> FloatArr:
        """
        Compute 2D distances between UEs and BSs.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at initialization.

        Returns
        -------
        FloatArr of shape (..., n_bs)
            2D distances in meters.
        """
        disp = self.displacement_2d(ue_xy=ue_xy, bs_xy=bs_xy)
        return np.linalg.norm(disp, axis=-1)

    def distances_3d(
        self,
        ue_xy: FloatArr,
        ue_heights_m: FloatArr,
        bs_heights_m: FloatArr,
    ) -> FloatArr:
        """Compute three-dimensional UE-to-BS distances.

        Parameters
        ----------
        ue_xy : FloatArr
            UE positions in meters with shape ``(..., 2)``.
        ue_heights_m : FloatArr
            UE heights in meters.
        bs_heights_m : FloatArr
            BS heights in meters.

        Returns
        -------
        FloatArr
            Three-dimensional distances in meters.
        """
        distances_2d = self.distances_2d(ue_xy)
        return self.distances_3d_by_distances_2d(
            distances_2d=distances_2d,
            ue_heights_m=ue_heights_m,
            bs_heights_m=bs_heights_m,
        )

    def distances_3d_by_distances_2d(
        self, distances_2d: FloatArr, ue_heights_m: FloatArr, bs_heights_m: FloatArr
    ) -> FloatArr:
        """Compute 3D distances from precomputed horizontal distances.

        Parameters
        ----------
        distances_2d : FloatArr
            Horizontal UE-to-BS distances in meters.
        ue_heights_m : FloatArr
            UE heights in meters.
        bs_heights_m : FloatArr
            BS heights in meters.

        Returns
        -------
        FloatArr
            Three-dimensional distances in meters.
        """
        self._ensure_shapes_for_3dd(distances_2d, ue_heights_m, bs_heights_m)
        height_diff = bs_heights_m[None, :] - ue_heights_m[:, None]
        return np.sqrt(distances_2d**2 + height_diff**2)

    def distances_and_angles(
        self,
        ue_xy: FloatArr,
        ue_heights: float | FloatArr,
        bs_heights: FloatArr,
        bs_xy: FloatArr | None = None,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        """
        Compute 2D/3D distances and azimuth/elevation angles from BSs to UEs.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        ue_heights : float | FloatArr of shape (...)
            UE heights in meters.
        bs_heights : FloatArr of shape (n_bs,)
            BS heights in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at initialization.

        Returns
        -------
        d_2d_m : FloatArr of shape (..., n_bs)
        d_3d_m : FloatArr of shape (..., n_bs)
        phi_az_rad : FloatArr of shape (..., n_bs)
            Azimuth angle in radians.
        theta_el_rad : FloatArr of shape (..., n_bs)
            Elevation/zenith angle in radians.
        """
        ue_xy = self._ensure_ue_xy_shape(ue_xy)

        # Shape: (..., n_bs, 2), vector from nearest BS image to UE
        disp_2d = self.displacement_2d(ue_xy=ue_xy, bs_xy=bs_xy)

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

        # Shape: (..., n_bs), vertical displacement from BS to UE
        dz = ue_heights[..., None] - bs_heights[None, :]

        # Shape: (..., n_bs, 3), vector from BS to UE
        disp_3d = np.concatenate([disp_2d, dz[..., None]], axis=-1)

        # Get angles in spherical coordinates
        _, az_rad_, elev_rad_ = utg.cartesian_to_spherical(disp_3d)

        # Convert angles to physics convention
        phi_az_rad, theta_el_rad = utg.to_physics_convention(az_rad_, elev_rad_)

        # 2D and 3D distances
        d_2d_m = np.linalg.norm(disp_2d, axis=-1)
        d_3d_m = np.linalg.norm(disp_3d, axis=-1)

        return d_2d_m, d_3d_m, phi_az_rad, theta_el_rad

    def candidate_distances_and_angles(
        self,
        ue_xy: FloatArr,
        ue_heights: float | FloatArr,
        bs_heights: FloatArr,
        bs_xy: FloatArr | None = None,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        """
        Compute candidate image distances and angles from BSs to UEs.

        Non-periodic topologies have exactly one candidate image. Wrap-around
        topologies override this method to expose all periodic BS images.

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
        d_2d_m : FloatArr of shape (..., 1, n_bs)
            2D distances in meters.
        d_3d_m : FloatArr of shape (..., 1, n_bs)
            3D distances in meters.
        phi_az_rad : FloatArr of shape (..., 1, n_bs)
            Azimuth angle in radians.
        theta_el_rad : FloatArr of shape (..., 1, n_bs)
            Elevation/zenith angle in radians.
        """
        d_2d, d_3d, phi_az, theta_el = self.distances_and_angles(
            ue_xy=ue_xy,
            ue_heights=ue_heights,
            bs_heights=bs_heights,
            bs_xy=bs_xy,
        )
        return (
            d_2d[..., None, :],
            d_3d[..., None, :],
            phi_az[..., None, :],
            theta_el[..., None, :],
        )

    @staticmethod
    def _ensure_ue_xy_shape(ue_xy: FloatArr) -> FloatArr:
        ue_xy = np.asarray(ue_xy)
        if ue_xy.ndim == 1 and ue_xy.shape[0] == 2:
            ue_xy = ue_xy[None, :]
        elif ue_xy.ndim < 2 or ue_xy.shape[-1] != 2:
            raise ValueError(f"ue_xy must have shape (..., 2), got {ue_xy.shape}")
        return ue_xy

    @staticmethod
    def _ensure_bs_xy_shape(bs_xy: FloatArr) -> FloatArr:
        bs_xy = np.asarray(bs_xy)
        if bs_xy.ndim == 1 and bs_xy.shape[0] == 2:
            bs_xy = bs_xy[None, :]
        elif bs_xy.ndim != 2 or bs_xy.shape[1] != 2:
            raise ValueError(f"bs_xy must have shape (n_bs, 2), got {bs_xy.shape}")
        return bs_xy

    @staticmethod
    def _ensure_shapes_for_3dd(
        distances_2d: FloatArr, ue_heights_m: FloatArr, bs_heights_m: FloatArr
    ) -> None:
        """Ensure that the input shapes are compatible for 3D distance calculation."""
        if distances_2d.shape != (*ue_heights_m.shape, *bs_heights_m.shape):
            raise ValueError(
                "Incompatible shapes for 3D distance calculation: "
                f"distances_2d: {distances_2d.shape}, "
                f"ue_heights_m: {ue_heights_m.shape}, "
                f"bs_heights_m: {bs_heights_m.shape}"
            )

"""Shadow Fading Map using Cholesky Decomposition."""

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..grf import GaussianRandomField

if TYPE_CHECKING:
    from ...geometry.mobility.abstract_boundary_handler import AbstractBoundaryHandler
    from ...geometry.roi.abstract_roi import AbstractROI


class ShadowFadingCholesky(GaussianRandomField):
    """
    Class to compute shadow fading using Cholesky decomposition based on a grid of points.
    """

    shadowing_z_maps: np.ndarray | None = None

    def __init__(
        self,
        *,
        roi: "AbstractROI",
        boundary_handler: "AbstractBoundaryHandler",
        bs_pos_2d: NDArray[np.float32],
        grf: GaussianRandomField | None = None,
        grid_spacing: float = 10.0,
        d_decorr: float = 20.0,
        sigma: float = 8.0,
        map_corr_factor: float | None = None,
        progress_bar: bool = False,
        map_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        resource_storage: str = "legacy",
        factor_dtype: str | np.dtype = "float64",
    ):
        """
        Initialize the ShadowFadingCholesky class.
        Parameters
        ----------
        roi : AbstractROI
            Region of interest (ROI) where the shadow fading is computed.
        grf : GaussianRandomField, optional
            Precomputed Gaussian random field. If not provided, a new one will be generated.
        grid_spacing : float, optional
            Spacing between grid points in meters. Default is 10.0.
        d_decorr : float, optional
            The :math:`1/e` correlation distance in meters for the exponential
            autocorrelation function. Default is 20.0.
        sigma : float, optional
            Standard deviation of the latent shadow fading field. Use 1.0 when
            final dB scaling is applied per link.
        roi_extension_factor: float, optional
            Inflation factor for the region of interest. This determines how far the grid extends
            beyond the ROI. Default is 3.0.
        """
        if grf is not None:
            if grf.resource_storage == "shared_mmap":
                grf = GaussianRandomField.copy(grf, seed=seed)
            self.__dict__.update(grf.__dict__)
        else:
            super().__init__(
                roi=roi,
                boundary_handler=boundary_handler,
                grid_spacing=grid_spacing,
                d_decorr=d_decorr,
                sigma=sigma,
                field_corr_factor=map_corr_factor,
                roi_extension_factor=1,
                progress_bar=progress_bar,
                map_dtype=map_dtype,
                seed=seed,
                resource_storage=resource_storage,
                factor_dtype=factor_dtype,
            )

        self.bs_positions = bs_pos_2d

    def get_grf(self, seed: int | None = None) -> GaussianRandomField:
        """Return this instance as a base Gaussian random field copy.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for the copied field.

        Returns
        -------
        GaussianRandomField
            Base-class copy without shadow-fading-specific state.
        """
        return GaussianRandomField.copy(self, seed=seed)

    def generate_shadow_fading_maps(self) -> None:
        """
        Generate unit-normal latent shadow fading maps based on the Cholesky decomposition.
        """
        n_bs = self.bs_positions.shape[0]
        fields = self.generate_fields(n_fields=n_bs)

        if self.field_corr_factor is None:  # BS-level distance-based correlation
            bs_distance_mat = self._compute_pairwise_distance_matrix(self.bs_positions)
            bs_cholesky_mat = self._compute_cholesky_matrix(bs_distance_mat)
            fields_ = fields.reshape(fields.shape[0], -1)
            shadowing_z_maps = (bs_cholesky_mat @ fields_).reshape(fields.shape)
        else:  # Map-level correlation
            shadowing_z_maps = fields

        self.shadowing_z_maps = shadowing_z_maps

    def get_sf_values(self, positions: np.ndarray, *, interpolation=None) -> np.ndarray:
        """
        Get shadow fading values at specified positions.

        Parameters
        ----------
        positions : np.ndarray
            2D array of shape (n_positions, 2) containing the x and y coordinates.

        Returns
        -------
        np.ndarray
            Unit-normal latent shadow fading values at the specified positions.
        """
        if self.shadowing_z_maps is None:
            raise ValueError("Latent shadow fading maps have not been generated yet.")
        return self.get_grf_values(
            self.shadowing_z_maps, positions, interpolation=interpolation
        ).T

    def validate_shadowing_mean_std(self) -> None:
        """Validate the mean and standard deviation of the latent shadow fading maps."""
        if self.shadowing_z_maps is None:
            raise ValueError("Latent shadow fading maps have not been generated yet.")
        super().validate_mean_std(self.shadowing_z_maps)

    def validate_shadowing_correlation(self) -> None:
        """Validate the spatial correlation of the latent shadow fading maps."""
        if self.shadowing_z_maps is None:
            raise ValueError("Latent shadow fading maps have not been generated yet.")
        super().validate_spatial_correlation(self.shadowing_z_maps)

    def plot_pairwise_distances(
        self, save: bool = False, path: str | None = None
    ) -> None:
        """
        Plots the pairwise distances matrix.

        Parameters
        ----------
        save : bool, optional
            If True, saves the plot to the specified path. Default is False.
        path : str, optional
            Path to save the plot. If None, the plot will not be saved. Default is None.
        """
        fig, ax = plt.subplots(constrained_layout=True)
        im = ax.imshow(
            self.distance_mat, cmap="viridis", origin="lower", aspect="equal"
        )
        plt.colorbar(im, ax=ax, label="Distance (m)")
        ax.set_aspect("equal")
        ax.set_xlabel("Index X")
        ax.set_ylabel("Index Y")
        ax.set_title("Pairwise Distances Matrix")

        if save and path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(path, dpi=300)
            print(f"Distances plot saved to {path}")

    def plot_shadow_fading_map(
        self,
        map_idx: int,
        wrapped: bool = False,
        interpolation_factor: int = 1,
        alpha: float = 0.75,
        save: bool = False,
        path: str | None = None,
    ) -> None:
        """
        Plots the latent shadow fading map or its wrapped version.

        Parameters
        ----------
        map_idx : int
            Index of the shadow fading map to plot.
        wrapped : bool, optional
            If True, plots the wrapped shadow fading map.
        interpolation_factor : int, optional
            Interpolation factor for the shadow fading map.
        alpha : float, optional
            Transparency level for the shadow fading map. Must be in the range (0, 1].
        save : bool, optional
            If True, saves the plot to the specified path.
        path : str, optional
            Path to save the plot. If None, the plot will not be saved.
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be in the range (0, 1].")
        if self.shadowing_z_maps is None:
            raise ValueError("Latent shadow fading maps have not been generated yet.")

        sfm = self.get_interpolated_fields(
            self.shadowing_z_maps,
            interpolation_factor=interpolation_factor,
            wrapped=wrapped,
        )

        bounds = self.wrapped_map_bounds if wrapped else self.grid.grid_bounds

        fig, ax = plt.subplots()
        im = ax.imshow(
            sfm[map_idx],
            cmap="viridis",
            extent=bounds,
            origin="lower",
            # aspect="equal",
            alpha=alpha,
        )
        plt.colorbar(im, ax=ax, label="Latent Shadow Fading (standard normal)")
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[-2:])
        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate (m)")
        ax.set_ylabel("Y Coordinate (m)")
        ax.set_title(f"Latent Shadow Fading Map {map_idx}")

        if save and path is not None:
            os.makedirs(path, exist_ok=True)
            fig.savefig(path, dpi=300)
            print(f"Shadow fading map saved to {path}")

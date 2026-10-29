"""Test the ROI implementations with a simple simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ..geometry.layout import layout_dict
from ..geometry.roi import roi_dict
from ..geometry.topology import EuclideanRadioTopology, WraparoundRadioTopology
from ..geometry.mobility import ReflectiveBoundary, WraparoundBoundary
from ..utils.types import FloatArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..config.geometry_config import GeometryConfig
    from ..config.layout_config import NetworkLayoutConfig
    from ..config.roi_config import ROIConfig
    from ..geometry.layout.abstract_layout import AbstractNetworkLayout
    from ..geometry.mobility.abstract_boundary_handler import AbstractBoundaryHandler
    from ..geometry.roi.abstract_roi import AbstractROI
    from ..geometry.topology.abstract_topology import AbstractTopology


def get_layout(
    layout_config: "NetworkLayoutConfig",
) -> "AbstractNetworkLayout":
    """Get the network layout based on the configuration."""
    cl = layout_dict.get(layout_config.layout, None)
    if cl is None:
        raise ValueError(
            f"Unsupported layout type: {layout_config.layout}. "
            f"Supported types: {list(layout_dict.keys())}."
        )
    return cl(config=layout_config)


def get_roi(roi_config: "ROIConfig", layout: "AbstractNetworkLayout") -> "AbstractROI":
    """Get the ROI based on the configuration and layout."""
    cl = roi_dict.get(roi_config.roi, None)
    if cl is None:
        raise ValueError(
            f"Unsupported ROI type: {roi_config.roi}. "
            f"Supported types: {list(roi_dict.keys())}."
        )
    return cl(
        config=roi_config,
        isd_m=layout.config.isd_m,
        n_tiers=layout.config.n_tiers,
        rotation_offset_deg=layout.config.rotation_offset_deg,
    )


def get_topology(
    layout: "AbstractNetworkLayout", network_wraparound: bool
) -> "AbstractTopology":
    """Get the radio topology for a given layout."""
    if network_wraparound:
        topology = WraparoundRadioTopology(
            bs_xy=layout.bs_xy, wraparound_shifts=layout.wraparound_shifts
        )
    else:
        topology = EuclideanRadioTopology(bs_xy=layout.bs_xy)
    return topology


def get_mobility_handler(
    roi: "AbstractROI", layout: "AbstractNetworkLayout", mobility_wraparound: bool
) -> "AbstractBoundaryHandler":
    """Get the mobility handler for a given ROI and layout."""
    if mobility_wraparound:
        return WraparoundBoundary(roi=roi, layout=layout)
    return ReflectiveBoundary(roi=roi, rand_reflection=True)


class SimulationGeometry:
    """
    Class for handling the geometry of the simulation.
    Including layout, ROI, topology, and mobility.
    """

    layout: "AbstractNetworkLayout"
    roi: "AbstractROI"

    topology: "AbstractTopology"
    boundary_handler: "AbstractBoundaryHandler"

    def __init__(self, config: "GeometryConfig") -> None:
        self.config = config

        self.layout = get_layout(self.config.layout)
        self.topology = get_topology(
            self.layout,
            network_wraparound=self.config.network_wraparound,
        )
        self.roi = get_roi(self.config.region, self.layout)
        self.boundary_handler = get_mobility_handler(
            roi=self.roi,
            layout=self.layout,
            mobility_wraparound=self.config.mobility_wraparound,
        )

    def keep_positions_in_roi(
        self, old_xy: FloatArr, new_xy: FloatArr, move_angles_rad: FloatArr
    ) -> tuple[FloatArr, FloatArr]:
        """
        Keep the positions within the ROI according to the mobility handler.

        Parameters
        ----------
        old_xy : FloatArr of shape (..., 2)
            Previous positions in meters.
        new_xy : FloatArr of shape (..., 2)
            Candidate next positions in meters.
        move_angles_rad : FloatArr of shape (...,)
            Current movement angles in radians.

        Returns
        -------
        corrected_xy : FloatArr of shape (..., 2)
            Corrected positions inside the ROI.
        corrected_angle_rad : FloatArr of shape (...,)
            Corrected movement angles in radians.
        """
        new_pos, new_move_angles_rad = self.boundary_handler.keep_ues_in_roi(
            old_xy=old_xy, new_xy=new_xy, move_angles_rad=move_angles_rad
        )
        return new_pos, new_move_angles_rad

    def get_rand_positions_in_roi(self, n_samples: int) -> FloatArr:
        """Sample random positions within the ROI.

        Parameters
        ----------
        n_samples : int
            Number of positions to sample.

        Returns
        -------
        FloatArr
            Sampled positions in meters with shape ``(n_samples, 2)``.
        """
        return self.roi.sample_positions(n=n_samples)

    def get_2d_distances(self, ue_positions_xy: FloatArr) -> FloatArr:
        """
        Get 2D distances from UEs to BSs.

        Parameters
        ----------
        ue_positions_xy : FloatArr of shape (..., 2)
            UE positions in meters.

        Returns
        -------
        FloatArr of shape (..., n_bs)
            2D distances in meters.
        """
        return self.topology.distances_2d(ue_positions_xy)

    def get_3d_distances(
        self,
        ue_positions_xy: FloatArr,
        ue_heights: FloatArr,
        bs_heights: FloatArr,
    ) -> FloatArr:
        """
        Get 3D distances from UEs to BSs.

        Parameters
        ----------
        ue_positions_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        ue_heights : FloatArr of shape (...)
            UE heights in meters.
        bs_heights : FloatArr of shape (n_bs,)
            BS heights in meters.

        Returns
        -------
        FloatArr of shape (..., n_bs)
            3D distances in meters.
        """
        return self.topology.distances_3d(
            ue_xy=ue_positions_xy,
            ue_heights_m=ue_heights,
            bs_heights_m=bs_heights,
        )

    def get_distances_and_angles(
        self,
        ue_positions_xy: FloatArr,
        ue_heights: FloatArr,
        bs_heights: FloatArr,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        """
        Get 2D/3D distances and azimuth/elevation angles from UEs to BSs.

        Parameters
        ----------
        ue_positions_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        ue_heights : FloatArr of shape (...)
            UE heights in meters.
        bs_heights : FloatArr of shape (n_bs,)
            BS heights in meters.

        Returns
        -------
        d_2d_m : FloatArr of shape (..., n_bs)
            2D distances in meters.
        d_3d_m : FloatArr of shape (..., n_bs)
            3D distances in meters.
        phi_az_rad : FloatArr of shape (..., n_bs)
            Azimuth angle in radians.
        theta_el_rad : FloatArr of shape (..., n_bs)
            Elevation/zenith angle in radians.
        """
        return self.topology.distances_and_angles(
            ue_xy=ue_positions_xy,
            ue_heights=ue_heights,
            bs_heights=bs_heights,
        )

    def get_candidate_distances_and_angles(
        self,
        ue_positions_xy: FloatArr,
        ue_heights: FloatArr,
        bs_heights: FloatArr,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        """
        Get distances and angles for all topology candidate BS images.

        Parameters
        ----------
        ue_positions_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        ue_heights : FloatArr of shape (...)
            UE heights in meters.
        bs_heights : FloatArr of shape (n_bs,)
            BS heights in meters.

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
        return self.topology.candidate_distances_and_angles(
            ue_xy=ue_positions_xy,
            ue_heights=ue_heights,
            bs_heights=bs_heights,
        )

    def plot_geometry(self, show: bool = True) -> tuple["Figure", "Axes"]:
        """Plot the layout and region of interest.

        Parameters
        ----------
        show : bool, optional
            Whether to display the figure.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and axes containing the geometry.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        fig, ax = self.layout.plot_layout(fig, ax, topology=self.topology, show=False)
        fig, ax = self.roi.plot_polygon(fig, ax, show=False)
        fig, ax = self.boundary_handler.plot_virtual_polygons(fig, ax, show=False)
        ax.set_title("Simulation Geometry: Layout and ROI")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        if show:
            plt.show()
        return fig, ax

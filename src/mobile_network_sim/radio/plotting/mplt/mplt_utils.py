"""Helper functions for matplotlib-based plotting of network environments."""

from typing import Any, TYPE_CHECKING


from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d  # pylint: disable=E0611

from ..plot_utils import interpolate_maps_and_extract_roi_values
from ...components.antennas.ant_omnidir import OmnidirAnt
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap
    from ...components.bs.bs_collection import BaseStationCollection
    from ...geometry.simulation_geometry import SimulationGeometry
    from ...geometry.layout.abstract_layout import AbstractNetworkLayout
    from ...geometry.roi.abstract_roi import AbstractROI


def add_bs_positions(
    ax: "Axes",
    geo: "SimulationGeometry",
    show_wrap_around_bs: bool = False,
    annotate_wrap_around_bs: bool = False,
    marker_size: int | None = None,
) -> None:
    """
    Plot BS positions and optionally wrap-around copies.

    Parameters
    ----------
    bs_collection : BaseStationCollection
        Collection of base stations containing positions and metadata.
    ax : matplotlib.axes.Axes
        Target axes.
    show_wrap_around_bs : bool, default False
        If True, plot wrap-around BS positions when available.
    annotate_wrap_around_bs : bool, default False
        If True, annotate wrap-around BS copies with their IDs.
    marker_size : int | None, default 50
        Size of the scatter plot markers for BS positions.
    """
    # Default values
    zorder = 5
    if marker_size is None:
        fig_size = ax.get_figure().get_size_inches()  # type: ignore
        marker_size = int(5 * (fig_size[0] + fig_size[1]) / 10)

    network_layout = geo.layout

    # Plot macro and micro BSs
    macro_bs_pos = network_layout.macro_bs_xy
    ax.set_aspect("equal")
    ax.scatter(
        macro_bs_pos[:, 0],
        macro_bs_pos[:, 1],
        edgecolors="tab:red",
        facecolors="tab:red",
        marker="o",
        label="Macro BS",
        zorder=zorder,
        s=marker_size,
    )

    if network_layout.n_micro_bs > 0:
        micro_bs_pos = network_layout.micro_bs_xy
        ax.set_aspect("equal")
        ax.scatter(
            micro_bs_pos[:, 0],
            micro_bs_pos[:, 1],
            edgecolors="tab:green",
            facecolors="tab:green",
            marker="o",
            label="micro BS",
            zorder=zorder,
            s=marker_size,
        )

    # Annotate each BS with its ID
    for i, bs_xy in enumerate(network_layout.bs_xy):
        ax.annotate(
            f"{i}",
            tuple(bs_xy),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            zorder=zorder,
        )

    # Plot virtual macro and micro BSs for wrap-around and annotate them
    if geo.config.network_wraparound and show_wrap_around_bs:
        virtual_macro_bs_pos = network_layout.macro_bs_xy_virtual.reshape(-1, 2)
        ax.scatter(
            virtual_macro_bs_pos[:, 0],
            virtual_macro_bs_pos[:, 1],
            edgecolors="tab:red",
            facecolors="white",
            marker="o",
            label="Virtual macro BS",
            zorder=zorder,
            s=marker_size,
        )

        if network_layout.n_micro_bs > 0:
            virtual_micro_bs_pos = network_layout.micro_bs_xy_virtual.reshape(-1, 2)
            ax.scatter(
                virtual_micro_bs_pos[:, 0],
                virtual_micro_bs_pos[:, 1],
                edgecolors="tab:green",
                facecolors="white",
                marker="o",
                label="Virtual micro BS",
                zorder=zorder,
                s=marker_size,
            )

        # Annotate each wrap-around BS with its ID
        if annotate_wrap_around_bs:
            for i, bs_xy in enumerate(network_layout.bs_xy_virtual.reshape(-1, 2)):
                ax.annotate(
                    f"{i} (virtual)",
                    tuple(bs_xy),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    zorder=zorder,
                )


def add_ant_boresights(ax: "Axes", bs_collection: "BaseStationCollection") -> None:
    """
    Plot the BS antenna boresights as arrows.

    Parameters
    ----------
    bs_positions : numpy.ndarray
        Array of shape (2,) or (n, 2) with BS positions to anchor boresight arrows.
    ax : matplotlib.axes.Axes
        Target axes.
    """
    for bs in bs_collection.bs_list:
        if bs.config.ant.ant_name == OmnidirAnt.NAME:
            return

        for i in range(bs.n_sectors):
            arrow_len = 15
            angle_rad = np.deg2rad(i * 120.0 + bs.config.sector_azimuth_offset_deg)

            ax.arrow(
                bs.pos.xy[0],
                bs.pos.xy[1],
                arrow_len * np.cos(angle_rad),
                arrow_len * np.sin(angle_rad),
                head_width=10,
                head_length=10,
                fc="tab:orange",
                ec="tab:orange",
                alpha=1.0,
            )


def add_macro_cell_edges(
    ax: "Axes",
    geo: "SimulationGeometry",
    bs_collection: "BaseStationCollection",
    show_wrap_around_bs: bool = False,
) -> None:
    """
    Plot cell boundaries (Voronoi diagram) for given BS positions.

    Parameters
    ----------
    bs_positions : numpy.ndarray
        Array of shape (n_bs, 2) with BS positions.
    ax : matplotlib.axes.Axes
        Target axes.
    """
    # Omnidirectional coverage area

    bs_pos = (
        geo.layout.macro_bs_xy_virtual.reshape(-1, 2)
        if show_wrap_around_bs
        else geo.layout.macro_bs_xy
    )
    vor = Voronoi(bs_pos)
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue  # skip open ridges
        v0, v1 = vor.vertices[vpair]
        ax.plot(
            [v0[0], v1[0]],
            [v0[1], v1[1]],
            color="k",
            linestyle="-.",
            alpha=0.5,
            lw=1,
        )
    ax.plot([], [], color="k", lw=1, label="Omnidirectional cell edges")

    # Cell edges between sectors
    isd = geo.layout.config.isd_m
    cell_center_pos: list[FloatArr] = []
    for bs in bs_collection.bs_list:
        if bs.config.ant.ant_name == OmnidirAnt.NAME:
            if show_wrap_around_bs:
                cell_center_pos.extend(
                    bs.pos.xy + geo.layout.wraparound_shifts.reshape(-1, 2)
                )
            else:
                cell_center_pos.append(bs.pos.xy)
        else:
            for i in range(bs.n_sectors):
                angle_rad = np.deg2rad(i * 120.0 + bs.config.sector_azimuth_offset_deg)
                offset = isd / 3 * np.array([np.cos(angle_rad), np.sin(angle_rad)])
                positions = (
                    bs.pos.xy + geo.layout.wraparound_shifts.reshape(-1, 2)
                    if show_wrap_around_bs
                    else bs.pos.xy[np.newaxis, :]
                )
                cell_center_pos.extend(positions + offset)
    voronoi_plot_2d(
        Voronoi(np.asarray(cell_center_pos)),
        ax=ax,
        show_points=False,
        show_vertices=False,
        line_colors="tab:blue",
        line_width=1,
    )

    ax.plot([], [], color="tab:blue", lw=1, label="Macro cell edges")


def add_roi(ax: "Axes", roi: "AbstractROI", inflation: float = 1.05) -> None:
    """
    Plot the region of interest boundary as a polygon.

    Parameters
    ----------
    roi : AbstractROI
        Region of interest providing polygon coordinates.
    ax : matplotlib.axes.Axes
        Target axes.
    inflation : float, default 1.05
        Factor to expand axes limits around the ROI for display.
    """
    coords = roi.polygon_coordinates
    polygon = Polygon(
        coords,
        closed=True,
        edgecolor="k",
        alpha=0.15,
        linewidth=3,
    )
    ax.add_patch(polygon)
    ax.set_xlim((inflation * coords[:, 0].min(), inflation * coords[:, 0].max()))
    ax.set_ylim((inflation * coords[:, 1].min(), inflation * coords[:, 1].max()))
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")


def add_value_map(
    fig: "Figure",
    ax: "Axes",
    geo: "SimulationGeometry",
    value_map: np.ndarray,
    *,
    vmap_kwargs: dict[str, Any] | None = None,
) -> tuple["Figure", "Axes"]:
    """Add a value map to an existing plot."""
    if vmap_kwargs is None:
        vmap_kwargs = {}
    alpha = vmap_kwargs.get("vmap_alpha", 1.0)
    cbar_label = vmap_kwargs.get("vmap_colorbar_label", "Value")
    cbar_ticks = vmap_kwargs.get("vmap_cbar_ticks", None)
    cbar_ticks_labels = vmap_kwargs.get("vmap_cbar_ticks_labels", None)
    cmap = vmap_kwargs.get("vmap_cmap", "viridis")
    interpolation_factor = vmap_kwargs.get("vmap_interpolation_factor", 1.0)
    no_value_interpolation = vmap_kwargs.get("no_value_interpolation", False)
    title = vmap_kwargs.get("title", "Value Map")
    vmin = vmap_kwargs.get("vmin", None)
    vmax = vmap_kwargs.get("vmax", None)

    value_map = interpolate_maps_and_extract_roi_values(
        geo,
        value_map,
        ip_factor=int(interpolation_factor),
        no_value_interpolation=no_value_interpolation,
    )

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)
    cax = ax.imshow(
        value_map,
        origin="lower",
        cmap=cmap,
        extent=geo.roi.bounding_box,
        aspect="auto",
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(cax, ax=ax, label=cbar_label)
    if cbar_ticks is not None:
        cbar.set_ticks(ticks=cbar_ticks)
        if cbar_ticks_labels is not None:
            cbar.set_ticklabels(cbar_ticks_labels)
    ax.set_title(title)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    return fig, ax

"""Plotting utility functions for the simulation environment and results."""

# pylint: disable=E0611
import importlib
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
from warnings import warn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import plotly.graph_objects as go
from ..geometry.grid import Grid

try:
    importlib.import_module("mobile_network_sim.radio.plotting.plotly._title_renderer")
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..geometry.simulation_geometry import SimulationGeometry


def save(
    fig: "Figure",
    ax: "Axes",
    *,
    dpi: int = 300,
    filename: str | None = None,
    save_pickle: bool = False,
    output_dir: str | Path,
) -> None:
    """
    Save the figure to a file.

    Args:
        fig (Figure): The matplotlib figure to save.
        ax (Axes): The matplotlib axes containing the plot.
        env_config (NetworkEnvConfig | None): The environment configuration.
        dpi (int, optional): The resolution of the saved plot in dots per inch. Defaults to 300.
    """
    if filename is None:
        filename = (
            ax.get_title().replace(" ", "_").replace("(", "").replace(")", "").lower()
        )

    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    fig.savefig(path / filename, dpi=dpi)
    if save_pickle:
        with open(path / f"{filename}.pkl", "wb") as f:
            pickle.dump(fig, f)
    print(f"Saved plot to: {path / filename}")


def show(
    pyqt_apps: list[object] | None = None,
    plotly_figures: list[go.Figure] | None = None,
    pyplot_figures: list["Figure"] | None = None,
) -> None:
    """Show the plot."""
    if plotly_figures is not None:
        for fig in plotly_figures:
            fig.show(
                renderer="titleBrowser",
                browser_tab_title=fig.layout.title.text,  # type: ignore
            )

    if pyqt_apps is not None:
        for app in pyqt_apps:
            app.exec_()  # type: ignore[attr-defined]

    if pyplot_figures is not None:
        for fig in pyplot_figures:
            fig.show()
    else:
        plt.show()


def matplot_to_pyqtgraph(fig: "Figure", dpi: int = 200):
    """
    Convert a Matplotlib figure to a QPixmap for use in PyQtGraph.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure to convert.
    dpi : int, optional
        Resolution in dots-per-inch for rendering.

    Returns
    -------
    pixmap : QPixmap
        PyQtGraph-compatible QPixmap representing the figure.
    width : int
        Width of the figure in pixels.
    height : int
        Height of the figure in pixels.
    """
    from PyQt5.QtGui import QImage, QPixmap

    # Render Matplotlib figure to canvas
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Get RGBA image from canvas
    img = np.asarray(canvas.buffer_rgba()).reshape(height, width, 4)
    # Convert RGBA to RGB
    img_rgb = img[:, :, :3].copy()

    # Convert to QImage
    qimg = QImage(img_rgb.data, width, height, 3 * width, QImage.Format_RGB888)

    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(qimg)

    return pixmap, width, height


def interpolate_maps_and_extract_roi_values(
    geo: "SimulationGeometry",
    value_maps: np.ndarray,
    ip_factor: int,
    no_value_interpolation: bool = False,
) -> np.ndarray:
    """
    Interpolate the value maps to a higher resolution.

    Parameters
    ----------
    geo : SimulationGeometry
        Simulation geometry containing the ROI and boundary handler for interpolation.
    value_maps : np.ndarray
        The input value maps to interpolate.
    ip_factor : int
        The factor by which to interpolate the maps.
    no_value_interpolation : bool, optional
        If True, do not interpolate the values and instead use nearest neighbor
        interpolation, e.g., for categorical data. Defaults to False.

    Returns
    -------
    np.ndarray
        The interpolated value maps.
    """
    r_ip_max = 10
    do_squueze = False

    if not 1 <= ip_factor <= r_ip_max:
        raise ValueError(f"interpolation_factor must be between 1 and {r_ip_max}.")
    if value_maps.ndim not in (2, 3):
        raise ValueError("value_map must be either 2D or 3D (n_maps, H, W).")
    if any(dim <= 0 for dim in value_maps.shape):
        raise ValueError("Value map must have non-zero dimensions.")

    if value_maps.ndim == 2:
        do_squueze = True
        value_maps = value_maps[np.newaxis, ...]

    n_maps, _, nx = value_maps.shape

    grid_res = getattr(geo.roi, "grid_res", None)
    if grid_res is None:
        grid_res = geo.roi.bb_size[0] // nx
        warn(
            f"Warning: ROI does not have `grid_res`. Using inferred grid resolution {grid_res}m."
        )

    grid = Grid(*geo.roi.bb_size, grid_res, centered=True)
    ip_grid = Grid.interpolate_grid(grid, ip_factor) if ip_factor > 1 else grid
    coords, idxs = geo.roi.get_positions_and_idxs_in_roi(
        *Grid.flatten(ip_grid.xy, ip_grid.idxs)
    )

    if no_value_interpolation:
        ip_values = grid.get_nearest_surrounding_values(
            value_maps,
            coords,
            geo.boundary_handler,
        )
    else:
        ip_values = grid.get_interpolated_values(
            value_maps, coords, geo.boundary_handler
        )

    ip_maps = np.full((n_maps, *ip_grid.shape), np.nan)
    ip_maps[:, idxs[:, 1], idxs[:, 0]] = ip_values

    if do_squueze:
        return np.squeeze(ip_maps)
    return ip_maps

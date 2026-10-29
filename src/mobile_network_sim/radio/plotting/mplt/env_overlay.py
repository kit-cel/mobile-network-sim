"""Plotting functions for the simulation environment and results."""

from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from . import mplt_utils as uplt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ...simulation.simulation_env import RadioEnvironment


def plot_environment(
    env: "RadioEnvironment",
    inflation: float = 1.2,
    annotate_wrap_around_bs: bool = False,
    show_wrap_around_bs: bool = True,
    show_legend: bool = True,
    vmap: np.ndarray | None = None,
    vmap_kwargs: dict[str, Any] | None = None,
) -> tuple["Figure", "Axes"]:
    """
    Plot the simulation environment with base stations (BSs) and the ROI boundary.

    Parameters
    ----------
    network_env : NetworkEnvironment
        The network environment containing the simulation data.
    inflation : float, default 1.2
        Factor to inflate the ROI axes limits for display.
    annotate_wrap_around_bs : bool, default False
        If True, annotate wrap-around BS copies with their IDs.
    show_ant_sectors : bool, default False
        If True, show the antenna sectors.
    show_legend : bool, default True
        If True, show the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Axes with the environment plot.
    """
    fig, ax = plt.subplots(constrained_layout=True)

    _vmap_kwargs: dict[str, Any] = {
        "vmap_interpolation_factor": 1,
        "vmap_cmap": "jet",
        "vmap_colorbar_label": "Value",
        "vmap_alpha": 1.0,
    }
    if vmap_kwargs is not None:
        _vmap_kwargs.update(vmap_kwargs)

    if isinstance(vmap, np.ndarray) and np.prod(vmap.shape) != 0:
        if vmap.ndim != 2:
            raise ValueError(f"Value map must be 2D array, got shape {vmap.shape}.")
        fig, ax = uplt.add_value_map(
            fig=fig,
            ax=ax,
            geo=env.geo_handler,
            value_map=vmap,
            vmap_kwargs=_vmap_kwargs,
        )

    uplt.add_macro_cell_edges(ax, env.geo_handler, env.bs_set, show_wrap_around_bs)
    uplt.add_ant_boresights(ax, env.bs_set)
    uplt.add_roi(ax, env.geo_handler.roi, inflation)
    uplt.add_bs_positions(
        ax, env.geo_handler, show_wrap_around_bs, annotate_wrap_around_bs
    )

    ax.set_title("Simulation Environment")
    ax.set_aspect("equal")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Simulation Environment")

    return fig, ax

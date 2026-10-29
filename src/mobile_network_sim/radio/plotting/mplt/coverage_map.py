"""Plotting functions for the simulation environment and results."""

from typing import Any, TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import mplt_utils as uplt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ...simulation.simulation_env import RadioEnvironment


def plot_coverage_map(
    env: "RadioEnvironment",
    vmap: np.ndarray,
    vmap_kwargs: dict[str, Any] | None = None,
    inflation: float = 1.2,
    annotate_wrap_around_bs: bool = False,
    show_wrap_around_bs: bool = True,
    show_legend: bool = True,
) -> tuple["Figure", "Axes"]:
    """
    Plot the simulation environment with base stations (BSs), ROI boundary,
    and a serving-cell coverage map derived from a 3D value map.

    Parameters
    ----------
    network_env : SimulationEnvironment
        The simulation environment containing the simulation data.
    vmap : np.ndarray
        Value map of shape (n_cells, x_shape, y_shape). The serving cell is
        computed via argmax over the first axis.
    vmap_kwargs : dict[str, Any] | None, default None
        Optional plotting kwargs.
    inflation : float, default 1.2
        Factor to inflate the ROI axes limits for display.
    annotate_wrap_around_bs : bool, default False
        If True, annotate wrap-around BS copies with their IDs.
    show_wrap_around_bs : bool, default True
        If True, show wrap-around BS copies.
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
        "vmap_colorbar_label": "Value",
        "vmap_alpha": 1.0,
    }
    if vmap_kwargs is not None:
        if "cmap" in vmap_kwargs.keys():
            vmap_kwargs.pop("cmap")  # prevent overriding the custom colormap
        _vmap_kwargs.update(vmap_kwargs)

    if vmap.ndim != 3:
        raise ValueError(f"Value map must be 3D array, got shape {vmap.shape}.")

    n_cells = vmap.shape[0]
    base_cmap = plt.get_cmap("turbo", n_cells)
    colors = base_cmap(np.arange(n_cells))

    # reproducible shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(colors)

    cmap = mcolors.ListedColormap(colors)
    _vmap_kwargs.update(
        {
            "cbar_ticks": np.arange(n_cells) + 0.5,
            "cbar_ticks_labels": [f"{i}" for i in range(n_cells)],
            "no_value_interpolation": True,
            "vmin": 0,
            "vmax": n_cells,
            "vmap_cmap": cmap,
        }
    )

    fig, ax = uplt.add_value_map(
        fig=fig,
        ax=ax,
        geo=env.geo_handler,
        value_map=np.argmax(vmap, axis=0),
        vmap_kwargs=_vmap_kwargs,
    )

    uplt.add_macro_cell_edges(ax, env.geo_handler, env.bs_set, show_wrap_around_bs)
    uplt.add_ant_boresights(ax, env.bs_set)
    uplt.add_roi(ax, env.geo_handler.roi, inflation)
    uplt.add_bs_positions(
        ax, env.geo_handler, show_wrap_around_bs, annotate_wrap_around_bs
    )

    ax.set_title("Coverage map (serving cell ID)")
    ax.set_aspect("equal")

    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Coverage map (serving cell ID)")

    return fig, ax

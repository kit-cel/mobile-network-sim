"""Plotting functions for the simulation environment and results."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "figure.figsize": (6, 4),
        "figure.autolayout": True,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.loc": "lower right",
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.5,
    }
)


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_cdf(
    data: np.ndarray | list[np.ndarray],
    labels: str | list[str],
    vlines: list[float] | None = None,
    vlines_labels: list[str] | None = None,
    vline_colors: list[str] | None = None,
    **kwargs,
) -> tuple["Figure", "Axes"]:
    """
    Plot the cumulative distribution function (CDF) of the given data.
    Args:
        data (np.ndarray): Input data for which to plot the CDF
        **kwargs: Additional keyword arguments for plt.plot()
    Returns:
        tuple[Figure, Axes]: Matplotlib Figure and Axes objects
    """
    data_list = data if isinstance(data, list) else [data]
    labels_list = labels if isinstance(labels, list) else [labels]
    if len(data_list) != len(labels_list):
        raise ValueError("Length of data and labels must be the same.")

    title = kwargs.pop("title", "Cumulative Distribution Function (CDF)")
    xlabel = kwargs.pop("xlabel", "Value")
    ylabel = kwargs.pop("ylabel", "Cumulative Probability")
    xlim = kwargs.pop("xlim", None)

    fig, ax = plt.subplots()
    for d, label in zip(data_list, labels_list):
        sorted_data = np.sort(d.flatten())
        print(f"Ignore {np.isnan(sorted_data).sum()} NaN values in CDF plot.")
        sorted_data = sorted_data[~np.isnan(sorted_data)]

        cdf_data = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf_data, label=label)

    if vlines is not None:
        for idx, vline in enumerate(vlines):
            vline_label = (
                vlines_labels[idx] if vlines_labels and idx < len(vlines_labels) else ""
            )
            vline_color = (
                vline_colors[idx] if vline_colors and idx < len(vline_colors) else "r"
            )
            ax.axvline(x=vline, color=vline_color, linestyle="--", label=vline_label)

    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_title(rf"{title}")
    ax.set_xlabel(rf"{xlabel}")
    ax.set_ylabel(rf"{ylabel}")
    ax.grid(True)
    ax.legend()

    return fig, ax

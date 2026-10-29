"""Helper functions for plotting antenna patterns."""

from typing import Any, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

if TYPE_CHECKING:
    from .abstract_ant import AbstractAntenna


def plot_elevation_pattern(
    ant: "AbstractAntenna",
    fig_handle: tuple[Figure, Axes] | None = None,
    phi_slice: float = 0.0,
    num_points: int = 361,
    normalize: bool = False,
) -> "tuple[Figure, Axes]":
    """
    Plot the elevation pattern of an antenna array.

    Parameters
    ----------
    fig : plt.Figure | None, optional
        Matplotlib figure to plot on. If None, creates a new figure.
    ax : plt.Axes | None, optional
        Matplotlib axes to plot on. If None, uses current axes.
    phi_slice : float, optional
        Azimuth angle for the vertical cut, by default 0.0
    num_points : int, optional
        Number of elevation sample points, by default 361
    normalize : bool, optional
        Whether to normalize the pattern to its maximum gain, by default False

    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if fig_handle is not None:
        if len(fig_handle) != 2:
            raise ValueError("fig_handle must be a tuple of (Figure, Axes)")
        fig, ax = fig_handle
        if not isinstance(fig, Figure) or not isinstance(ax, Axes):
            raise ValueError("fig_handle must contain a Figure and an Axes")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    theta = np.linspace(0, 180, num_points, dtype=np.float64)
    phi = np.full_like(theta, phi_slice)

    ant_pattern = ant.get_ant_composite_radiation_pattern_db(
        phi, theta, normalize=normalize
    )

    ax.plot(theta, ant_pattern, "k-", linewidth=2)
    ax.set_title("Elevation Pattern")
    ax.set_xlabel("Elevation θ (°)")
    ax.set_ylabel("Gain (dBi)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)
    if normalize:
        ax.set_ylim(np.min(ant_pattern) - 5, 5)
    else:
        ax.set_ylim(np.min(ant_pattern) - 5, np.max(ant_pattern) + 5)
    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_vertical_cut_ant_array_pattern(
    ant: "AbstractAntenna", phi_slices: list[float], **kwargs: dict[str, Any]
) -> tuple["Figure", "Axes"]:
    """
    Plot vertical cut(s) of the composite antenna array radiation pattern.

    Generates elevation pattern plot(s) at specified azimuth angles,
    showing the complete array pattern with beamforming applied.

    Parameters
    ----------
    phi_slices : list[float]
        Azimuth angle(s) in degrees at which to plot vertical cuts.
        Common values: [0] for principal plane, [-90, 0, 90] for multiple cuts.
    num_points : int, optional
        Number of elevation sample points, by default 361 (1° resolution).
    coordinate_system : str, optional
        Plot coordinate system, either "polar" or "cartesian", by default "cartesian".
    ylim : tuple[float, float] | None, optional
        Y-axis limits for Cartesian plots in dBi, e.g., (-10, 30).
        Ignored for polar plots. If None, auto-scaled.
    yticks : list[float] | None, optional
        Custom y-axis tick positions for Cartesian plots in dBi.
        Ignored for polar plots. If None, uses default ticks.
    """
    if np.any((np.asarray(phi_slices) < -180) | (np.asarray(phi_slices) > 180)):
        raise ValueError("Invalid phi_slices, must be in [-180, 180].")

    num_points = kwargs.get("num_points", 361)
    coordinate_system = kwargs.get("coordinate_system", "cartesian")
    ylim = kwargs.get("ylim", None)
    normalize = kwargs.get("normalize", False)

    if not isinstance(num_points, int) or num_points < 2:
        raise ValueError("num_points must be an integer greater than 1.")
    if ylim is not None and not isinstance(ylim, tuple):
        raise ValueError("ylim must be a tuple of (ymin, ymax) or None.")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean value.")

    fig = plt.figure(figsize=(8, 6))
    if coordinate_system == "polar":
        ax = fig.add_subplot(111, polar=True)
    elif coordinate_system == "cartesian":
        ax = fig.add_subplot(111)
    else:
        raise ValueError("coordinate_system must be 'polar' or 'cartesian'")

    phi_deg = np.linspace(-180, 180, num_points, dtype=np.float64)
    theta_deg = np.linspace(0, 180, num_points, dtype=np.float64)

    ant_pattern = ant.get_ant_composite_radiation_pattern_db(
        *np.meshgrid(phi_deg, theta_deg), normalize=normalize
    )

    for phi in phi_slices:
        idx = (np.abs(phi_deg - phi)).argmin()
        theta = np.radians(theta_deg) if coordinate_system == "polar" else theta_deg
        ax.plot(theta, ant_pattern[idx, :], label=rf"$\varphi={phi}°$")

    if coordinate_system == "cartesian":
        ax.grid(True)
        ax.set_xlabel(r"Elevation $\theta \, (°)$")
        ax.set_ylabel(r"Gain $(\mathrm{dB})$")
        ax.set_xlim((0, 180))
        if ylim is None:
            ylim = ax.set_ylim(
                (
                    np.maximum(np.min(ant_pattern), -100),
                    np.maximum(0, np.max(ant_pattern)),
                )
            )
        ax.set_ylim(ylim)
        ax.legend()

    ax.set_title(r"Vertical Cut Antenna elevation Pattern", va="bottom")
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Vertical Cut Pattern of Antenna Array")
    return fig, ax

"""Wrap-around boundary model for user mobility.."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import translate

from .abstract_boundary_handler import AbstractBoundaryHandler
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..layout.abstract_layout import AbstractNetworkLayout
    from ..roi.abstract_roi import AbstractROI


class WraparoundBoundary(AbstractBoundaryHandler):
    """Wrap-around boundary model for user mobility."""

    def __init__(
        self,
        roi: "AbstractROI",
        layout: "AbstractNetworkLayout",
        *,
        eps_m: float = 1e-6,
        warn: bool = True,
    ) -> None:
        super().__init__(roi)

        self.layout = layout
        self.eps_m = eps_m
        self.warn = warn

        self.virtual_offsets_and_polygons = [
            (
                np.asarray(offset, dtype=float),
                translate(
                    self.roi.polygon,
                    xoff=float(offset[0]),
                    yoff=float(offset[1]),
                ),
            )
            for offset in self.roi.wraparound_shifts[1:]
        ]

    def keep_ues_in_roi(
        self, old_xy: FloatArr, new_xy: FloatArr, move_angles_rad: FloatArr
    ) -> tuple[FloatArr, FloatArr]:
        """Wrap movements that leave the region of interest.

        Parameters
        ----------
        old_xy : FloatArr
            Positions before the movement step, with shape ``(..., 2)``.
        new_xy : FloatArr
            Proposed positions after the step, with the same shape as
            ``old_xy``.
        move_angles_rad : FloatArr
            Movement headings in radians, with shape ``old_xy.shape[:-1]``.

        Returns
        -------
        tuple[FloatArr, FloatArr]
            Wrapped positions and unchanged headings.

        Raises
        ------
        ValueError
            If the position or heading shapes are incompatible.
        """
        if not old_xy.shape == new_xy.shape and old_xy.shape[-1] == 2:
            raise ValueError(
                "old_xy and new_xy must have shape (..., 2), got "
                f"{old_xy.shape} and {new_xy.shape}."
            )
        if old_xy.shape[:-1] != move_angles_rad.shape:
            raise ValueError(
                "old_xy/new_xy and move_angles_rad must have compatible shapes."
                f" Got {old_xy.shape} and {move_angles_rad.shape}."
            )

        mask_out = np.asarray(self.roi.outside_roi(new_xy), dtype=bool)
        if not np.any(mask_out):
            return new_xy, move_angles_rad

        corrected_xy = self.roi.wrap_positions(new_xy, eps_m=self.eps_m)
        return corrected_xy.reshape(new_xy.shape), move_angles_rad

    def maybe_wrap_positions(self, pos_xy: FloatArr) -> FloatArr:
        """
        Wrap positions to the ROI if they are outside the ROI.

        Parameters
        ----------
        pos_xy : FloatArr
            Positions to be wrapped. Shape (..., 2).

        Returns
        -------
        FloatArr
            Wrapped positions. Shape (..., 2).
        """
        pos_xy = np.asarray(pos_xy, dtype=np.float32)

        is_1d = pos_xy.ndim == 1
        if is_1d:
            pos_xy = pos_xy[None, :]

        wrapped = self.roi.wrap_positions(pos_xy, eps_m=self.eps_m)
        return wrapped[0] if is_1d else wrapped

    def plot_virtual_polygons(
        self,
        fig: "Figure | None" = None,
        ax: "Axes | None" = None,
        show: bool = False,
        save: bool = False,
        legend: bool = False,
    ) -> tuple["Figure", "Axes"]:
        """Plot the virtual ROI polygons used for wrap-around handling.

        Parameters
        ----------
        fig : Figure | None, optional
            Existing Matplotlib figure.
        ax : Axes | None, optional
            Existing Matplotlib axes.
        show : bool, optional
            Whether to display the figure.
        save : bool, optional
            Whether to save the figure as ``virtual_rois.png``.
        legend : bool, optional
            Whether to label offsets and display a legend.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and axes containing the virtual polygons.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        for offset, polygon in self.virtual_offsets_and_polygons:
            x, y = polygon.exterior.xy
            if legend:
                ax.plot(x, y, label=f"Virtual ROI shifted by {offset}")
            else:
                ax.plot(x, y)

        if legend:
            ax.legend()
        if show:
            plt.show()
        if save:
            plt.savefig("virtual_rois.png")
        return fig, ax

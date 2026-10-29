"""Abstract class for handling the boundaries of the region of interest (ROI) in mobility models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


from ...utils.types import FloatArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..roi.abstract_roi import AbstractROI


class AbstractBoundaryHandler(ABC):
    """Abstract class for handling the boundaries of the region of interest (ROI) in mobility models."""

    def __init__(self, roi: "AbstractROI"):
        self.roi = roi

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    @abstractmethod
    def keep_ues_in_roi(
        self, old_xy: FloatArr, new_xy: FloatArr, move_angles_rad: FloatArr
    ) -> tuple[FloatArr, FloatArr]:
        """
        Apply boundary handling to a UE movement step.

        Parameters
        ----------
        old_xy : Float32Arr of shape (..., 2)
            Previous UE position in meters.
        new_xy : Float32Arr of shape (..., 2)
            Candidate next UE position in meters.
        move_angles_rad : Float32Arr of shape (...,)
            Current movement angles in radians.

        Returns
        -------
        corrected_xy : Float32Arr of shape (..., 2)
            Corrected UE position inside the ROI.
        corrected_angle_rad : Float32Arr of shape (...,)
            Corrected movement angle in radians.
        """

    @abstractmethod
    def plot_virtual_polygons(
        self,
        fig: "Figure | None" = None,
        ax: "Axes | None" = None,
        show: bool = False,
        save: bool = False,
    ) -> tuple["Figure", "Axes"]:
        """
        Plot the virtual ROI polygons used for wrap-around handling.

        Parameters
        ----------
        fig : Figure | None, optional
            Matplotlib figure to plot on. If None, a new figure is created.
        ax : Axes | None, optional
            Matplotlib axes to plot on. If None, new axes are created.
        show : bool, optional
            Whether to call plt.show() after plotting. Default is False.
        save : bool, optional
            Whether to save the plot to a file. Default is False.

        Returns
        -------
        fig : Figure
            The matplotlib figure containing the plot.
        ax : Axes
            The matplotlib axes containing the plot.
        """

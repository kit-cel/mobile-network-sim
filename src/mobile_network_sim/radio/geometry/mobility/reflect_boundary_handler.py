"""Reflective boundary model for user mobility.."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .abstract_boundary_handler import AbstractBoundaryHandler
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..roi.abstract_roi import AbstractROI


class ReflectiveBoundary(AbstractBoundaryHandler):
    """Reflective boundary model for user mobility."""

    seed: int
    np_rng: np.random.Generator

    def __init__(self, roi: "AbstractROI", rand_reflection: bool = False) -> None:
        super().__init__(roi)
        self.rand_reflection = rand_reflection
        self.set_seed(seed=roi.config.seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the random seed used for randomized reflections.

        Parameters
        ----------
        seed : int | None
            Seed for the random-number generator. If ``None``, a seed is
            sampled from NumPy's global random state.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)

    def keep_ues_in_roi(
        self, old_xy: FloatArr, new_xy: FloatArr, move_angles_rad: FloatArr
    ) -> tuple[FloatArr, FloatArr]:
        """Reflect movements that cross the region boundary.

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
            Corrected positions and headings, preserving their input shapes.

        Raises
        ------
        ValueError
            If the input shapes are incompatible or a crossing cannot be
            associated with a boundary intersection.
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

        mask_out = np.asarray(self.roi.outside_roi(new_xy))
        if not np.any(mask_out):
            return new_xy, move_angles_rad

        pos_shape = old_xy.shape
        angle_shape = move_angles_rad.shape

        flat_old = old_xy.reshape(-1, 2)
        flat_new = new_xy.reshape(-1, 2)
        flat_angle = move_angles_rad.reshape(-1)
        flat_outside = mask_out.reshape(-1)

        flat_corrected_xy = flat_new.copy()
        flat_corrected_angle = flat_angle.copy()

        for idx in np.flatnonzero(flat_outside):
            xy_i, angle_i = self._reflect_one_pos(
                old_xy=flat_old[idx],
                new_xy=flat_new[idx],
                move_angle_rad=float(flat_angle[idx]),
            )
            flat_corrected_xy[idx] = xy_i
            flat_corrected_angle[idx] = angle_i

        return (
            flat_corrected_xy.reshape(pos_shape),
            flat_corrected_angle.reshape(angle_shape),
        )

    def _reflect_one_pos(
        self,
        old_xy: FloatArr,
        new_xy: FloatArr,
        move_angle_rad: float,
    ) -> tuple[FloatArr, float]:
        """
        Reflect one UE at the ROI boundary and preserve remaining travel distance.

        Parameters
        ----------
        old_xy : FloatArr
            The old position of the UE before the movement step. Shape (2,).
        new_xy : FloatArr
            The new position of the UE after the movement step. Shape (2,).
        move_angle_rad : float
            The movement angle of the UE in radians.

        Returns
        -------
        tuple[FloatArr, float]
            The reflected position of the UE and the new movement angle in radians.
        """
        if self.roi.inside_roi(new_xy):
            return new_xy.astype(np.float32, copy=False), move_angle_rad

        hit_xy = self.roi.first_boundary_intersection(old_xy, new_xy)
        if hit_xy is None:
            raise ValueError("UE left ROI, but no boundary intersection was found.")

        normal = self.roi.boundary_normal(hit_xy)

        displacement = np.asarray(new_xy, dtype=np.float64) - np.asarray(
            old_xy, dtype=np.float64
        )
        total_dist = np.linalg.norm(displacement)

        if total_dist == 0.0:
            return old_xy.astype(np.float32, copy=False), move_angle_rad

        direction = displacement / total_dist

        dist_to_hit = np.linalg.norm(hit_xy - old_xy)
        remaining_dist = total_dist - dist_to_hit

        reflected_direction = direction - 2.0 * np.dot(direction, normal) * normal
        reflected_angle = float(
            np.arctan2(reflected_direction[1], reflected_direction[0]) % (2.0 * np.pi)
        )

        reflected_xy = hit_xy + remaining_dist * reflected_direction

        # Handle multiple reflections in one large step.
        while not self.roi.inside_roi(reflected_xy):
            reflected_start_xy = hit_xy + 1e-4 * reflected_direction
            reflected_xy, reflected_angle = self._reflect_one_pos(
                old_xy=reflected_start_xy.astype(np.float32),
                new_xy=reflected_xy.astype(np.float32),
                move_angle_rad=reflected_angle,
            )

        return reflected_xy.astype(np.float32, copy=False), reflected_angle

    def plot_virtual_polygons(
        self,
        fig: "Figure | None" = None,
        ax: "Axes | None" = None,
        show: bool = False,
        save: bool = False,
    ) -> tuple["Figure", "Axes"]:
        """Return axes for visualizing this boundary handler.

        Reflective boundaries have no virtual polygons, so this method only
        creates plotting objects when they were not supplied.

        Parameters
        ----------
        fig : Figure | None, optional
            Existing Matplotlib figure.
        ax : Axes | None, optional
            Existing Matplotlib axes.
        show : bool, optional
            Retained for interface compatibility; currently ignored.
        save : bool, optional
            Retained for interface compatibility; currently ignored.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and axes used for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        return fig, ax

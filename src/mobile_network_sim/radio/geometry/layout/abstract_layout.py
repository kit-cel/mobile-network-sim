"""Abstract base class for network layout of base stations (BSs)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from ..topology.wraparound_topology import WraparoundRadioTopology
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..topology.abstract_topology import AbstractTopology
    from ...config.layout_config import NetworkLayoutConfig


@dataclass(frozen=True)
class InterBSDistanceConstraints:
    """
    Inter-BS distance constraints for layout validation. If None, no constraint is applied.

    Parameters
    ----------
    macro_macro : tuple[float, float] | None
        Minimum and maximum distance between macro BSs.
    macro_micro : tuple[float, float] | None
        Minimum and maximum distance between macro and micro BSs.
    micro_micro : tuple[float, float] | None
        Minimum and maximum distance between micro BSs.
    """

    macro_macro: tuple[float, float] | None = None
    macro_micro: tuple[float, float] | None = None
    micro_micro: tuple[float, float] | None = None


class AbstractNetworkLayout(ABC):
    """Base class for fixed BS/site geometry."""

    _macro_bs_xy: FloatArr  # shape (n_macro_bs, 2)
    _micro_bs_xy: FloatArr  # shape (n_micro_bs, 2)

    _wraparound_shifts: FloatArr  # shape (n_shifts, 2)

    def __init__(self, config: "NetworkLayoutConfig") -> None:
        """
        Parameters
        ----------
        config : NetworkLayoutConfig
            Configuration for the network layout.
        """
        if config.layout != self.__class__.__name__:
            raise ValueError(
                f"Config layout_class_name {config.layout} does not match "
                f"the class name {self.__class__.__name__}"
            )
        self.config = config
        self.inter_bs_distance_constraints: InterBSDistanceConstraints | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_macro_bs={self.n_macro_bs}, "
            f"n_micro_bs={self.n_micro_bs})"
        )

    @classmethod
    @abstractmethod
    def compute_num_of_macro_bs(cls, **kwargs: Any) -> int:
        """Get the number of macro BSs in the cluster.

        Returns
        -------
        int
            Number of macro base stations.
        """

    @classmethod
    @abstractmethod
    def compute_num_of_micro_bs(cls, **kwargs: Any) -> int:
        """Get the number of micro BSs in the cluster.

        Returns
        -------
        int
            Number of micro base stations.
        """

    @classmethod
    def compute_num_of_bs(cls, **kwargs: Any) -> int:
        """Get the total number of base stations in the cluster.

        Returns
        -------
        int
            Total number of macro and micro base stations.
        """
        return cls.compute_num_of_macro_bs(**kwargs) + cls.compute_num_of_micro_bs(
            **kwargs
        )

    @property
    def n_macro_bs(self) -> int:
        """Number of macro BSs."""
        return self._macro_bs_xy.shape[0]

    @property
    def n_micro_bs(self) -> int:
        """Number of micro BSs."""
        return self._micro_bs_xy.shape[0]

    @property
    def n_bs(self) -> int:
        """Total number of BSs."""
        return self.n_macro_bs + self.n_micro_bs

    @property
    def macro_bs_xy(self) -> FloatArr:
        """Macro BS positions, shape (n_macro_bs, 2)."""
        if not hasattr(self, "_macro_bs_xy") or self._macro_bs_xy is None:
            raise ValueError("Macro BS positions have not been computed.")
        return self._macro_bs_xy

    @property
    def micro_bs_xy(self) -> FloatArr:
        """Micro BS positions, shape (n_micro_bs, 2)."""
        if not hasattr(self, "_micro_bs_xy") or self._micro_bs_xy is None:
            raise ValueError("Micro BS positions have not been computed.")
        return self._micro_bs_xy

    @property
    def bs_xy(self) -> FloatArr:
        """All BS positions, shape (n_bs, 2)."""
        return np.vstack((self.macro_bs_xy, self.micro_bs_xy))

    @property
    def macro_bs_xy_virtual(self) -> FloatArr:
        """Virtual Macro BS positions for wraparound, shape (n_macro_bs, 2)."""
        if self._wraparound_shifts is None:
            raise ValueError("Wraparound shifts have not been computed.")
        virtual_positions = []
        for shift in self._wraparound_shifts:
            virtual_positions.append(self.macro_bs_xy + shift)
        return np.vstack(virtual_positions)

    @property
    def bs_xy_virtual(self) -> FloatArr:
        """Virtual BS positions for wraparound, shape (n_bs * n_shifts, 2)."""
        return np.vstack((self.macro_bs_xy_virtual, self.micro_bs_xy_virtual))

    @property
    def micro_bs_xy_virtual(self) -> FloatArr:
        """Virtual Micro BS positions for wraparound, shape (n_micro_bs, 2)."""
        if self._wraparound_shifts is None:
            raise ValueError("Wraparound shifts have not been computed.")
        virtual_positions = []
        for shift in self._wraparound_shifts:
            virtual_positions.append(self.micro_bs_xy + shift)
        return np.vstack(virtual_positions)

    @property
    def wraparound_shifts(self) -> FloatArr:
        """
        Translation vectors for periodic layout copies, shape (n_shifts, 2).

        The zero shift must be included as the first entry.
        """
        if not hasattr(self, "_wraparound_shifts") or self._wraparound_shifts is None:
            raise ValueError("Wraparound shifts have not been computed.")
        return self._wraparound_shifts

    def validate_inter_bs_distance_constraints(
        self, topology: "AbstractTopology", warn_only: bool = False
    ) -> None:
        """Validate configured inter-base-station distance ranges.

        Parameters
        ----------
        topology : AbstractTopology
            Topology used to compute pairwise distances.
        warn_only : bool, optional
            If ``True``, warn instead of raising for violations.

        Raises
        ------
        ValueError
            If a constraint is violated and ``warn_only`` is ``False``.
        """

        def _get_min_max_distances(
            bs_xy_1: FloatArr, bs_xy_2: FloatArr
        ) -> tuple[float, float]:
            if bs_xy_1.shape[0] == 0 or bs_xy_2.shape[0] == 0:
                return np.inf, 0.0  # No BSs, so no distances to violate constraints
            d_min = np.inf
            d_max = 0.0
            for bs_pos in bs_xy_1:
                for other_pos in bs_xy_2:
                    if np.array_equal(bs_pos, other_pos):
                        continue
                    d = topology.distances_2d(bs_pos, other_pos)
                    d_min = min(d_min, d.item())
                    d_max = max(d_max, d.item())
            return d_min, d_max

        d_ma_ma = _get_min_max_distances(self._macro_bs_xy, self._macro_bs_xy)
        d_mi_mi = _get_min_max_distances(self._micro_bs_xy, self._micro_bs_xy)
        d_ma_mi = _get_min_max_distances(self._macro_bs_xy, self._micro_bs_xy)
        w = len(f"{max(d_ma_ma + d_mi_mi + d_ma_mi):.2f}")
        print(
            f"{self.__class__.__name__} inter-BS distances:\n"
            f"  Macro-Macro: min {d_ma_ma[0]:{w}.2f} m, max {d_ma_ma[1]:{w}.2f} m\n"
            f"  Micro-Micro: min {d_mi_mi[0]:{w}.2f} m, max {d_mi_mi[1]:{w}.2f} m\n"
            f"  Macro-Micro: min {d_ma_mi[0]:{w}.2f} m, max {d_ma_mi[1]:{w}.2f} m"
        )

        cs = self.inter_bs_distance_constraints
        msg = ""
        if cs is not None and cs.macro_macro is not None:
            for d in d_ma_ma:
                if not cs.macro_macro[0] <= d <= cs.macro_macro[1]:
                    msg += f"Macro-Macro distance {d:.2f} m not in {cs.macro_macro} m."

        if cs is not None and cs.macro_micro is not None:
            for d in d_ma_mi:
                if not cs.macro_micro[0] <= d <= cs.macro_micro[1]:
                    msg += f"Macro-Micro distance {d:.2f} m not in {cs.macro_micro} m."
        if cs is not None and cs.micro_micro is not None:
            for d in d_mi_mi:
                if not cs.micro_micro[0] <= d <= cs.micro_micro[1]:
                    msg += f"Micro-Micro distance {d:.2f} m not in {cs.micro_micro} m."
        if msg:
            if not warn_only:
                raise ValueError(msg)
            warn(msg)

    def plot_layout(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        topology: "AbstractTopology | None" = None,
        show: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plot the network layout.

        Parameters
        ----------
        fig : Figure | None, optional
            Existing Matplotlib figure.
        ax : Axes | None, optional
            Existing Matplotlib axes.
        topology : AbstractTopology | None, optional
            Topology whose periodic copies should also be drawn.
        show : bool, optional
            Whether to display the figure.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and axes containing the layout.
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(  # Plot macro BSs
            self._macro_bs_xy[:, 0],
            self._macro_bs_xy[:, 1],
            c="blue",
            marker="^",
            label="Macro BS",
        )

        if self._micro_bs_xy.shape[0] > 0:
            ax.scatter(  # Plot micro BSs
                self._micro_bs_xy[:, 0],
                self._micro_bs_xy[:, 1],
                c="orange",
                marker="o",
                label="Micro BS",
            )

        if topology is not None and isinstance(topology, WraparoundRadioTopology):
            shifts = topology.wraparound_shifts
            for shift in shifts:
                shifted_macro_bs = self._macro_bs_xy + shift
                ax.scatter(
                    shifted_macro_bs[:, 0],
                    shifted_macro_bs[:, 1],
                    c="blue",
                    marker="^",
                    alpha=0.3,
                    label=(
                        "Macro BS (wraparound)"
                        if np.array_equal(shift, shifts[0])
                        else None
                    ),
                )
                if self._micro_bs_xy.shape[0] > 0:
                    shifted_micro_bs = self._micro_bs_xy + shift
                    ax.scatter(
                        shifted_micro_bs[:, 0],
                        shifted_micro_bs[:, 1],
                        c="orange",
                        marker="o",
                        alpha=0.3,
                        label=(
                            "Micro BS (wraparound)"
                            if np.array_equal(shift, shifts[0])
                            else None
                        ),
                    )

        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Network Layout")
        ax.legend()
        if show:
            plt.show()
        return fig, ax

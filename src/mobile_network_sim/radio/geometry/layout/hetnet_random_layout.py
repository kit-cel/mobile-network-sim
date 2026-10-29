"""
Class for layout of heterogeneous network (HetNet) including macro BSs located
in a hexagonal pattern and micro BSs located randomly within the hexagons.
"""

from typing import Any, TYPE_CHECKING

import numpy as np
from shapely.geometry import Point, Polygon

from .abstract_layout import InterBSDistanceConstraints
from .hexagonal_layout import HexagonalMacroLayout
from ..topology.wraparound_topology import WraparoundRadioTopology
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from ...config.layout_config import NetworkLayoutConfig


class HetNetRandomLayout(HexagonalMacroLayout):
    """
    Layout of heterogeneous network (HetNet) including macro BSs located
    in a hexagonal pattern and micro BSs located randomly within the hexagons.
    """

    # Default parameters
    N_SECTORS = 3  # Number of sectors/cells/TRxP per macro site position
    N_MICRO_CELLS_PER_MACRO_CELL = 3  # Number of micro cells per macro cell
    D_REL_MACRO_MIN = 0.3  # Relative distance from macro BS based on ISD
    D_REL_MICRO_MIN = 0.1  # Relative distance from micro BS based on ISD
    N_VIRTUAL_BS = 7  # Number of virtual BSs around the cluster

    seed: int
    np_rng: np.random.Generator

    def __init__(self, config: "NetworkLayoutConfig") -> None:
        """
        Initialize the HetNetRandomLayout.

        Notes
        -----
        Macro BS positions are computed in the parent class (HexagonalMacroLayout)
        Wraparound shifts are computed in the parent class (HexagonalMacroLayout)
        """
        super().__init__(config=config)

        # Micro BSs are randomly placed within the macro cells using rejection sampling
        self.inter_bs_distance_constraints = InterBSDistanceConstraints(
            macro_macro=(self.D_REL_MACRO_MIN * self.isd_m, np.inf),
            macro_micro=(self.D_REL_MACRO_MIN * self.isd_m, np.inf),
            micro_micro=(self.D_REL_MICRO_MIN * self.isd_m, np.inf),
        )
        self.set_seed(seed=config.seed)
        self._micro_bs_xy = self._compute_micro_bs_pos(
            self.n_tiers, self.isd_m, self.rotation_offset_deg
        )

    def set_seed(self, seed: int | None) -> None:
        """Reset the layout's random-number generator.

        Parameters
        ----------
        seed : int or None
            Generator seed. When ``None``, draw a seed from NumPy's legacy
            global generator.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)

    def _compute_micro_bs_pos(
        self,
        n_tiers: int,
        isd_m: float,
        rotation_offset_deg: float,
        max_attempts: int = 10_000,
    ) -> FloatArr:
        """Compute the positions of the micro BSs."""
        macro_bs_positions = self._compute_macro_bs_pos(
            n_tiers=n_tiers, isd_m=isd_m, rotation_offset_deg=rotation_offset_deg
        )

        r_cell = isd_m / 3

        cell_center_positions = []  # Cell center position of hexagonal macro cells
        for macro_pos in macro_bs_positions:
            for i in range(3):
                angle = np.deg2rad(rotation_offset_deg + i * 120, dtype=float)
                pos = macro_pos + r_cell * np.array([np.cos(angle), np.sin(angle)])
                cell_center_positions.append(pos)
        cell_center_positions = np.array(cell_center_positions)

        corner_positions = []  # Base offsets of corners of macro cells
        for k in range(6):
            angle = ((rotation_offset_deg + np.pi / 3) + 2 * k - 1) * np.pi / 6
            corner_positions.append(r_cell * np.array([np.cos(angle), np.sin(angle)]))
        corner_positions = np.array(corner_positions)

        macro_cell_polygons = []  # Hexagonal macro cell polygons
        for macro_cell_pos in cell_center_positions:
            macro_cell_polygons.append(
                Polygon(np.asarray(corner_positions) + macro_cell_pos)
            )

        topology = WraparoundRadioTopology(wraparound_shifts=self.wraparound_shifts)

        micro_cell_positions = []
        for macro_cell in macro_cell_polygons:
            for _ in range(self.N_MICRO_CELLS_PER_MACRO_CELL):
                attempts = 0
                while True:  # Rejection sampling of micro cell positions
                    # Sample random point within bounding box of macro cell
                    rand_xy = self._poisson_with_seed(macro_cell, num_points=1)
                    # rand_xy = topology.get_real_position(rand_xy)

                    d_macro_min = np.min(
                        topology.distances_2d(ue_xy=rand_xy, bs_xy=macro_bs_positions)
                    )
                    attempts += 1
                    if attempts > max_attempts:
                        raise RuntimeError(
                            f"Failed to place micro BS after {max_attempts} attempts. "
                            "Consider adjusting the distance constraints."
                        )
                    if d_macro_min < self.D_REL_MACRO_MIN * isd_m:
                        continue  # Reject position if too close to macro BSs

                    if len(micro_cell_positions) > 0:
                        d_micro_min = np.min(
                            topology.distances_2d(
                                rand_xy, np.array(micro_cell_positions)
                            )
                        )
                        if d_micro_min < self.D_REL_MICRO_MIN * isd_m:
                            continue  # Reject position if too close to existing micro BSs

                    micro_cell_positions.append(rand_xy)  # Accept position
                    break

        return np.array(micro_cell_positions, dtype=np.float32)

    def _poisson_with_seed(self, polygon: Polygon, num_points: int) -> FloatArr:
        """
        Generate Poisson-distributed random points within a given shape using a fixed seed.

        Parameters
        ----------
        polygon : Polygon
            The polygon within which to generate points.
        num_points : int
            The number of points to generate.

        Returns
        -------
        FloatArr
            Array of shape (num_points, 2) with the generated points.
        """
        bbox = np.array(polygon.bounds)
        result = np.empty((num_points, 2), dtype=np.float32)

        i_points = 0
        while i_points < num_points:
            x, y = (
                self.np_rng.uniform(bbox[0], bbox[2]),
                self.np_rng.uniform(bbox[1], bbox[3]),
            )
            if polygon.contains(Point(x, y)):
                result[i_points] = (x, y)
                i_points += 1
        return result.squeeze()

    @classmethod
    def compute_num_of_micro_bs(cls, **kwargs: Any) -> int:
        """
        Compute the number of micro cells.

        Other Parameters
        ----------------
        n_tiers : int
            Number of tiers of the hexagonal layout (rings around the center BS).

        Returns
        -------
        int
            Number of micro BSs.

        Notes
        -----
        Each macro cell has 3 sectors and each sector has 3 micro cells.
        Hence, the total number of micro BSs is given by: 3 * 3 * number_of_macro_BSs.
        """
        n_tiers = kwargs.get("n_tiers", None)
        if n_tiers is None:
            raise ValueError("Missing argument 'n_tiers'.")
        return (
            cls.N_SECTORS
            * cls.N_MICRO_CELLS_PER_MACRO_CELL
            * cls.compute_num_of_macro_bs(n_tiers=n_tiers)
        )

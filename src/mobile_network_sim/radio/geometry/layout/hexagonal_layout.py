"""Class for hexagonal macro layout of base stations (BSs)."""

from typing import Any, TYPE_CHECKING

import numpy as np
from ...utils.types import FloatArr

from .abstract_layout import AbstractNetworkLayout

if TYPE_CHECKING:
    from ...config.layout_config import NetworkLayoutConfig


class HexagonalMacroLayout(AbstractNetworkLayout):
    """Hexagonal macro BS layout."""

    def __init__(self, config: "NetworkLayoutConfig") -> None:
        super().__init__(config=config)

        self.n_tiers = config.n_tiers
        self.isd_m = config.isd_m
        self.rotation_offset_deg = config.rotation_offset_deg

        self._macro_bs_xy = self._compute_macro_bs_pos(
            n_tiers=self.n_tiers,
            isd_m=self.isd_m,
            rotation_offset_deg=self.rotation_offset_deg,
        )
        self._micro_bs_xy = np.empty((0, 2), dtype=np.float32)

        self._wraparound_shifts = self._compute_wraparound_shifts()

    @classmethod
    def compute_num_of_macro_bs(cls, **kwargs: Any) -> int:
        """Compute the number of macro base stations.

        Returns
        -------
        int
            Number of macro base stations, including the center site.

        Raises
        ------
        ValueError
            If ``n_tiers`` is not supplied.
        """
        n_tiers = kwargs.get("n_tiers", None)
        if n_tiers is None:
            raise ValueError("Missing argument 'n_tiers'.")
        return 1 + 3 * n_tiers * (n_tiers + 1)

    @classmethod
    def compute_num_of_micro_bs(cls, **kwargs: Any) -> int:
        """Return the number of micro base stations in this layout.

        Returns
        -------
        int
            Always zero because this layout contains only macro sites.
        """
        return 0

    def _compute_macro_bs_pos(
        self, n_tiers: int, isd_m: float, rotation_offset_deg: float
    ) -> FloatArr:
        """
        Compute macro BS positions for a hexagonal multi-tier layout.

        The layout starts with one BS at the origin. Each additional tier adds a
        ring of BSs around it. Their positions are determined by the inter-site
        distance and the rotation offset.


        Returns
        -------
        FloatArr
            Array of BS coordinates with shape `(num_sites, 2)`.
        """

        positions = [[0.0, 0.0]]
        for i in range(1, n_tiers + 1):
            for j in range(6):
                alpha = j * np.pi / 3 + np.deg2rad(rotation_offset_deg)
                for k in range(i):
                    x = (i - k) * isd_m * np.cos(alpha) + k * isd_m * np.cos(
                        alpha + np.pi / 3
                    )
                    y = (i - k) * isd_m * np.sin(alpha) + k * isd_m * np.sin(
                        alpha + np.pi / 3
                    )
                    positions.append([x, y])

        return np.array(positions, dtype=np.float32)

    def _compute_wraparound_shifts(self) -> FloatArr:
        """
        Compute the translation vectors for the wraparound cluster copies.

        The method calculates the distance and rotation of the neighboring cluster
        copies from the inter-site distance and number of tiers. It returns the zero
        shift for the original cluster and six shifts for the surrounding copies.

        Returns
        -------
        FloatArr
            Array of two-dimensional wraparound shift vectors.
        """
        n_cluster_copies = 1
        a = self.isd_m * (1 + self.n_tiers)
        b = self.isd_m * self.n_tiers
        gamma = 2 / 3 * np.pi
        isd_ = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(gamma))

        rotation_offset_deg = (
            np.rad2deg(np.arctan(np.sqrt(3) * self.n_tiers / (2 + 3 * self.n_tiers)))
            + self.rotation_offset_deg
        )

        return self._compute_macro_bs_pos(
            n_tiers=n_cluster_copies,
            isd_m=isd_,
            rotation_offset_deg=rotation_offset_deg,
        )

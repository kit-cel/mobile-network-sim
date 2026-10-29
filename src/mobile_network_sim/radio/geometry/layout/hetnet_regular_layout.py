"""
Class for layout of heterogeneous network (HetNet) including macro BSs located
in a hexagonal pattern and micro BSs located on the edges of the hexagons.
"""

from typing import Any, TYPE_CHECKING

import numpy as np

from .hexagonal_layout import HexagonalMacroLayout
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from ...config.layout_config import NetworkLayoutConfig


class HetNetRegularLayout(HexagonalMacroLayout):
    """
    Layout of heterogeneous network (HetNet) including macro BSs located
    in a hexagonal pattern and micro BSs located on the edges of the hexagons.
    """

    def __init__(self, config: "NetworkLayoutConfig") -> None:
        """
        Initialize the HetNetRegularLayout.

        Notes
        -----
        Macro BS positions are computed in the parent class (HexagonalMacroLayout)
        Wraparound shifts are computed in the parent class (HexagonalMacroLayout)
        """
        super().__init__(config=config)

        self._micro_bs_xy = self._compute_micro_bs_pos(
            n_tiers=self.n_tiers,
            isd_m=self.isd_m,
            rotation_offset_deg=self.rotation_offset_deg,
        )

    def _compute_micro_bs_pos(
        self, n_tiers: int, isd_m: float, rotation_offset_deg: float
    ) -> FloatArr:
        """
        Compute the positions of the micro BSs.

        First, the macro BS positions are generated. Three micro BSs are then
        placed around each macro BS at equal angles and at half the inter-site
        distance.

        Returns
        -------
        FloatArr
            Array of two-dimensional micro BS positions.
        """

        macro_bs_pos = self._compute_macro_bs_pos(
            n_tiers=n_tiers, isd_m=isd_m, rotation_offset_deg=rotation_offset_deg
        )

        r = isd_m / 2
        angle_off = 60 + self.config.rotation_offset_deg

        micro_bs_pos = []
        for macro_pos in macro_bs_pos:
            for i in range(3):
                angle = np.deg2rad(angle_off + i * 120, dtype=float)
                pos = macro_pos + r * np.array([np.cos(angle), np.sin(angle)])
                micro_bs_pos.append(pos)

        return np.array(micro_bs_pos, dtype=np.float32)

    @classmethod
    def compute_num_of_micro_bs(cls, **kwargs: Any) -> int:
        """Compute the number of regularly placed micro base stations.

        Returns
        -------
        int
            Three times the number of macro base stations.

        Raises
        ------
        ValueError
            If ``tiers`` is not supplied.
        """
        tiers = kwargs.get("tiers", None)
        if tiers is None:
            raise ValueError("Missing argument 'tier'.")
        return 3 * cls.compute_num_of_macro_bs(tiers=tiers)

"""Hexagonal region of interest (ROI) implementation."""

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry.polygon import Polygon

from .abstract_roi import AbstractROI
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from ...config.roi_config import ROIConfig


class HexagonROI(AbstractROI):
    """Hexagonal region of interest (ROI)."""

    def __init__(
        self,
        config: "ROIConfig",
        isd_m: float,
        n_tiers: int,
        rotation_offset_deg: float,
    ) -> None:
        super().__init__(
            config=config,
            isd_m=isd_m,
            n_tiers=n_tiers,
            rotation_offset_deg=rotation_offset_deg,
        )

        self.cluster_size = self._get_cluster_size()
        self.translation_length = np.sqrt(self.cluster_size) * self.isd_m

        self.first_wraparound_dir_angle_rad = self._get_first_wraparound_angle()

        self._polygon = self._generate_polygon()
        self._wraparound_shifts = self._get_wraparound_shifts()

    def _get_cluster_size(self) -> int:
        """Return the number of sites in the periodic hexagonal cluster."""
        i = self.n_tiers + 1
        j = self.n_tiers
        return i**2 + i * j + j**2

    def _get_first_wraparound_angle(self) -> float:
        """Return the angle of the first wraparound translation vector."""
        i = self.n_tiers + 1
        j = self.n_tiers
        return np.arctan2(np.sqrt(3.0) * j, 2.0 * i + j)

    def _generate_polygon(self) -> Polygon:
        """Generate the central hexagonal ROI polygon."""
        hex_circumradius = self.translation_length / np.sqrt(3.0)

        angle_offset = self.first_wraparound_dir_angle_rad + np.deg2rad(
            self.rot_off_deg
        )

        corners: list[FloatArr] = []

        for idx in range(6):
            angle = (2 * idx - 1) * np.pi / 6.0 + angle_offset

            corner = np.array(
                [
                    hex_circumradius * np.cos(angle),
                    hex_circumradius * np.sin(angle),
                ]
            )

            corners.append(corner)

        return Polygon(np.asarray(corners, dtype=np.float64))

    def _get_wraparound_shifts(self) -> FloatArr:
        """Get wraparound shifts for the periodic hexagonal cluster."""
        off = self.first_wraparound_dir_angle_rad + np.deg2rad(self.rot_off_deg)

        positions: list[FloatArr] = [np.array([0.0, 0.0])]

        for idx in range(6):
            alpha = idx * np.pi / 3.0 + off

            positions.append(
                np.array(
                    [
                        self.translation_length * np.cos(alpha),
                        self.translation_length * np.sin(alpha),
                    ]
                )
            )

        return np.asarray(positions, dtype=np.float64)

    def wrap_position(self, pos_xy: FloatArr, eps_m: float = 1e-6) -> FloatArr:
        """Wrap a position from a translated hexagonal ROI copy.

        Parameters
        ----------
        pos_xy : FloatArr
            Position with shape ``(2,)``.
        eps_m : float, optional
            Boundary tolerance in meters.

        Returns
        -------
        FloatArr
            Wrapped position with shape ``(2,)``.
        """
        return super().wrap_position(pos_xy=pos_xy, eps_m=eps_m)

"""Module for multi-hexagonal ROI."""

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry.polygon import Polygon

from .abstract_roi import AbstractROI
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from ...config.roi_config import ROIConfig


class MultiHexagonROI(AbstractROI):
    """Region of interest (ROI) consisting of multiple hexagonal cells."""

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
        """Generate the polygon using hexagonal layout coordinates."""
        hex_center_locs = self._compute_hexagon_centers()
        outer_ring = hex_center_locs[-6 * self.n_tiers :]

        hex_circumradius_m = self.isd_m / np.sqrt(3.0)
        alpha_off_rad = np.deg2rad(self.rot_off_deg)

        offset = 0.0
        corners: list[FloatArr] = []

        for idx, bs_pos_xy in enumerate(outer_ring):
            start = 0 if idx % self.n_tiers == 0 else 1

            for k in range(start, 3):
                angle = (offset + 2 * k - 1) * np.pi / 6.0 + alpha_off_rad

                corner = bs_pos_xy + hex_circumradius_m * np.array(
                    [np.cos(angle), np.sin(angle)],
                    dtype=np.float64,
                )
                corners.append(corner)

            if (idx + 1) % self.n_tiers == 0:
                offset += 2.0

        return Polygon(np.asarray(corners, dtype=np.float64))

    def _compute_hexagon_centers(self) -> FloatArr:
        """Calculate the BS locations in a hexagonal grid."""
        positions: list[FloatArr] = [np.array([0.0, 0.0], dtype=np.float64)]

        alpha_off_rad = np.deg2rad(self.rot_off_deg)

        for tier in range(1, self.n_tiers + 1):
            for side in range(6):
                alpha = side * np.pi / 3.0 + alpha_off_rad

                dir_1 = np.array([np.cos(alpha), np.sin(alpha)])
                dir_2 = np.array(
                    [
                        np.cos(alpha + np.pi / 3.0),
                        np.sin(alpha + np.pi / 3.0),
                    ]
                )

                for step in range(tier):
                    pos = (tier - step) * self.isd_m * dir_1 + step * self.isd_m * dir_2
                    positions.append(pos)

        return np.asarray(positions, dtype=np.float64)

    def _get_wraparound_shifts(self) -> FloatArr:
        """Get wraparound shifts for the periodic hexagonal cluster."""
        positions: list[FloatArr] = [np.array([0.0, 0.0])]

        off = self.first_wraparound_dir_angle_rad + np.deg2rad(self.rot_off_deg)

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
        """Wrap a position from a translated multi-hexagonal ROI copy.

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

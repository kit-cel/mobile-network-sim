"""Module for rectangular ROI."""

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry.polygon import Polygon

from .abstract_roi import AbstractROI
from ...utils.types import FloatArr

if TYPE_CHECKING:
    from ...config.roi_config import ROIConfig


class RectangularROI(AbstractROI):
    """Compact rectangular toroidal ROI aligned to the hexagonal cluster lattice."""

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

        self.period_a, self.period_b = self._get_period_vectors()
        self.axis_x = self.period_a / np.linalg.norm(self.period_a)
        self.axis_y = np.array([-self.axis_x[1], self.axis_x[0]], dtype=np.float64)
        self.x_span = float(np.linalg.norm(self.period_a))
        self.y_span = float(abs(np.cross(self.period_a, self.period_b)) / self.x_span)
        self.twist_x = float(np.dot(self.period_b, self.axis_x))

        self._polygon = self._generate_polygon()
        self._wraparound_shifts = self._get_wraparound_shifts()

    def _get_period_vectors(self) -> tuple[FloatArr, FloatArr]:
        """Return primitive period vectors of the hexagonal cluster lattice."""
        i = self.n_tiers + 1
        j = self.n_tiers
        cluster_size = i**2 + i * j + j**2
        translation_length = np.sqrt(cluster_size) * self.isd_m
        first_angle = np.arctan2(np.sqrt(3.0) * j, 2.0 * i + j)
        off = first_angle + np.deg2rad(self.rot_off_deg)

        t1 = translation_length * np.array([np.cos(off), np.sin(off)])
        t2 = translation_length * np.array(
            [np.cos(off + np.pi / 3.0), np.sin(off + np.pi / 3.0)]
        )

        return np.asarray(t1, dtype=np.float64), np.asarray(t2, dtype=np.float64)

    def _generate_polygon(self) -> Polygon:
        """Generate the rectangular ROI polygon."""
        corners_uv = np.array(
            [
                [-self.x_span / 2.0, -self.y_span / 2.0],
                [self.x_span / 2.0, -self.y_span / 2.0],
                [self.x_span / 2.0, self.y_span / 2.0],
                [-self.x_span / 2.0, self.y_span / 2.0],
            ],
            dtype=np.float64,
        )
        return Polygon(self._from_local(corners_uv))

    def _get_wraparound_shifts(self) -> FloatArr:
        """Get torus shifts from the primitive hexagonal period vectors."""
        positions: list[FloatArr] = [np.array([0.0, 0.0], dtype=np.float64)]

        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                if ix == 0 and iy == 0:
                    continue

                positions.append(ix * self.period_a + iy * self.period_b)

        return np.asarray(positions, dtype=np.float64)

    def wrap_position(self, pos_xy: FloatArr, eps_m: float = 1e-6) -> FloatArr:
        """Wrap a position using the compact rectangular edge pairing.

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

        Raises
        ------
        RuntimeError
            If the maximum number of wrap operations is exceeded.
        ValueError
            If the wrapped position is not inside the ROI.
        """
        local = self._to_local(np.asarray(pos_xy, dtype=np.float64)[None, :])[0]
        x, y = float(local[0]), float(local[1])

        for _ in range(10):
            wrapped = False
            if y > self.y_span / 2.0 + eps_m:
                y -= self.y_span
                x -= self.twist_x
                wrapped = True
            elif y < -self.y_span / 2.0 - eps_m:
                y += self.y_span
                x += self.twist_x
                wrapped = True

            if x > self.x_span / 2.0 + eps_m:
                x -= self.x_span
                wrapped = True
            elif x < -self.x_span / 2.0 - eps_m:
                x += self.x_span
                wrapped = True

            if not wrapped:
                break
        else:
            raise RuntimeError("Maximum compact rectangular wrap attempts exceeded.")

        wrapped_xy = self._from_local(np.asarray([[x, y]], dtype=np.float64))[0]
        if not self.inside_roi(wrapped_xy, eps_m=eps_m):
            raise ValueError(
                "Compact rectangular wrap did not return an in-ROI position. "
                f"pos_xy={pos_xy}, wrapped_xy={wrapped_xy}"
            )
        return wrapped_xy.astype(np.float32, copy=False)

    def _to_local(self, pos_xy: FloatArr) -> FloatArr:
        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        return np.column_stack([pos_xy @ self.axis_x, pos_xy @ self.axis_y])

    def _from_local(self, pos_uv: FloatArr) -> FloatArr:
        pos_uv = np.asarray(pos_uv, dtype=np.float64)
        return pos_uv[:, 0:1] * self.axis_x + pos_uv[:, 1:2] * self.axis_y

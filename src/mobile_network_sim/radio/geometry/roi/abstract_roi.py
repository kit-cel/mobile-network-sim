"""Abstract region of interest (ROI) class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pointpats
import pointpats.random
from shapely import Geometry, Point, Polygon, LineString, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping

from ...utils.types import BoolArr, FloatArr, IntArr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ...config.roi_config import ROIConfig


class AbstractROI(ABC):
    """Region of interest."""

    NAME: str

    seed: int
    _rng: np.random.Generator
    _polygon: Polygon
    _wraparound_shifts: np.ndarray

    def __init__(
        self,
        config: "ROIConfig",
        isd_m: float,
        n_tiers: int,
        rotation_offset_deg: float,
    ) -> None:
        self.config = config

        self.n_tiers = n_tiers
        self.isd_m = isd_m
        self.rot_off_deg = rotation_offset_deg

        self.set_seed(config.seed)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fingerprint={self.fingerprint})"

    def reset(self, seed: int | None = None) -> None:
        """Reset the ROI random-number state.

        Parameters
        ----------
        seed : int or None, optional
            Random seed. If ``None``, generate a seed from NumPy's default
            random-number generator.
        """
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        """
        Set the random seed for reproducibility.

        Parameters
        ----------
        seed : int
            Seed for random number generator.

        Notes
        -----
        Owns a dedicated :class:`numpy.random.Generator` (``self._rng``)
        passed explicitly to every :mod:`pointpats` call in
        :meth:`sample_positions`. ``pointpats`` >= 2.6 (SPEC 7) samples from
        its own freshly-created, unseeded ``default_rng()`` instance
        whenever no ``rng=`` argument is given, so seeding NumPy's legacy
        global ``np.random`` state (the previous approach here) has no
        effect on it and silently made every ROI position sample
        non-reproducible regardless of the seed passed to this method.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample_positions(
        self, n: int, *, rng: np.random.Generator | None = None
    ) -> FloatArr:
        """
        Get random positions within the region-of-interest.

        Parameters
        ----------
        n : int
            Number of positions to generate.
        rng : numpy.random.Generator or None, optional
            Explicit stream to use. When omitted, the ROI-owned stream is
            used. Mobility models such as RandomWaypoint pass their own
            stream so future waypoints cannot consume the initial-position
            stream or another UE's mobility stream.

        Returns
        -------
        FloatArr
            Random positions of shape (n, 2).
        """
        positions = pointpats.random.poisson(
            self._polygon, size=n, rng=self._rng if rng is None else rng
        )
        if n == 1:
            return positions[np.newaxis, :]
        return positions

    def inside_roi(self, pos_xy: FloatArr, eps_m: float = 0.0) -> bool | BoolArr:
        """
        Check if a position or positions are inside or on the boundary of the polygon.

        Parameters
        ----------
        pos_xy : FloatArr
            Position given as np.array([x, y]) or of shape (..., 2).
        eps_m : float, optional
            Buffer distance in meters to consider points near the boundary as inside.

        Returns
        -------
        bool | BoolArr
            True if the position is inside or on the boundary of the polygon,
        """
        pos_xy = self.ensure_pos_dim(pos_xy)
        polygon = self._polygon.buffer(eps_m) if eps_m > 0.0 else self._polygon

        inside = np.fromiter(
            (polygon.covers(Point(pt)) for pt in pos_xy),
            dtype=np.bool_,
            count=len(pos_xy),
        )
        if inside.size > 1:
            return inside
        return bool(inside[0])

    def outside_roi(self, pos_xy: FloatArr, eps_m: float = 0.0) -> bool | BoolArr:
        """
        Check if positions are outside the polygon.

        Parameters
        ----------
        pos_xy : FloatArr
            Position given as np.array([x, y]) or of shape (..., 2).
        eps_m : float, default=0.0
            Numerical tolerance in meters. Positive values expand the polygon
            slightly before testing membership.

        Returns
        -------
        bool | BoolArr
            True if the position is outside the polygon.
        """
        inside = self.inside_roi(pos_xy, eps_m=eps_m)

        if isinstance(inside, np.ndarray):
            return ~inside
        return not inside

    def get_positions_and_idxs_in_roi(
        self, pos_xy: FloatArr, idxs: IntArr
    ) -> tuple[FloatArr, IntArr]:
        """
        Get the positions and indices of grid points in the region of interest (ROI).

        Parameters
        ----------
        pos_xy : FloatArr
            Array of shape (N, 2) with positions to check.
        idxs : IntArr
            Array of shape (N, 2) with indices corresponding to the positions.

        Returns
        -------
        tuple[FloatArr, IntArr]
            Tuple containing:
            - positions_in_roi: Array of shape (N, 2) with positions in the ROI.
            - indices_in_roi: Array of shape (N, 2) with indices of positions in the ROI.
        """
        mask = np.asarray(self.inside_roi(pos_xy)).astype(bool)
        return pos_xy[mask], idxs[mask]

    def get_closest_point_in_roi(self, position: FloatArr) -> FloatArr:
        """
        Get the closest point in the ROI to a given position.

        Parameters
        ----------
        position : FloatArr
            Position given as np.array([x, y]).

        Returns
        -------
        FloatArr
            Closest point in the ROI.
        """
        if not self.inside_roi(position):
            # If the position is not in the ROI, find the closest point in the ROI
            closest_point = self._polygon.exterior.interpolate(
                self._polygon.exterior.project(Point(position))
            )
            return np.array(closest_point.xy).flatten()
        return position

    def get_intersections(self, geometry: Geometry) -> BaseGeometry:
        """
        Get the intersection of a geometry with the ROI boundary.

        Parameters
        ----------
        geometry : Geometry
            A Shapely geometry to intersect with the ROI boundary.

        Returns
        -------
        BaseGeometry
            The intersection geometry of the input with the ROI boundary.
        """
        return self._polygon.boundary.intersection(geometry)

    def first_boundary_intersection(
        self, start_xy: FloatArr, end_xy: FloatArr
    ) -> FloatArr | None:
        """Return the first intersection of a segment with the ROI boundary.

        Parameters
        ----------
        start_xy : FloatArr
            Segment start position with shape ``(2,)``.
        end_xy : FloatArr
            Segment end position with shape ``(2,)``.

        Returns
        -------
        FloatArr or None
            Boundary position with shape ``(2,)``, or ``None`` if the segment
            does not intersect the boundary.
        """
        line = LineString([start_xy, end_xy])
        intersection = self._polygon.boundary.intersection(line)

        points = self._extract_points(intersection)
        if not points:
            return None

        start = np.asarray(start_xy, dtype=np.float64)

        first_point = min(
            points,
            key=lambda p: float(np.linalg.norm(np.asarray(p.coords[0]) - start)),
        )

        return np.asarray(first_point.coords[0], dtype=np.float64)

    def boundary_normal(self, boundary_xy: FloatArr) -> FloatArr:
        """Return the outward unit normal at a boundary point.

        Parameters
        ----------
        boundary_xy : FloatArr
            Boundary position with shape ``(2,)``.

        Returns
        -------
        FloatArr
            Outward unit-normal vector with shape ``(2,)``.

        Raises
        ------
        ValueError
            If no polygon edge can be used to determine the normal.
        """
        p = np.asarray(boundary_xy, dtype=np.float64)

        vertices = np.asarray(self._polygon.exterior.coords[:-1], dtype=np.float64)

        best_dist = np.inf
        best_normal: np.ndarray | None = None

        centroid = np.asarray(self._polygon.centroid.coords[0], dtype=np.float64)

        for a, b in zip(vertices, np.roll(vertices, shift=-1, axis=0)):
            edge = b - a
            edge_len_sq = float(np.dot(edge, edge))

            if edge_len_sq == 0.0:
                continue

            tau = np.clip(np.dot(p - a, edge) / edge_len_sq, 0.0, 1.0)
            closest = a + tau * edge
            dist = np.linalg.norm(p - closest)

            if dist < best_dist:
                tangent = edge / np.linalg.norm(edge)

                normal_1 = np.array([tangent[1], -tangent[0]])
                normal_2 = -normal_1

                # Pick the normal pointing away from polygon centroid.
                outward_ref = closest - centroid
                normal = normal_1 if np.dot(normal_1, outward_ref) >= 0.0 else normal_2

                best_dist = dist
                best_normal = normal

        if best_normal is None:
            raise ValueError("Could not determine boundary normal.")

        return best_normal

    def plot_polygon(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        show: bool = False,
        save: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plot the ROI polygon for debugging purposes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None, optional
            Existing figure. A new figure is created when either ``fig`` or
            ``ax`` is ``None``.
        ax : matplotlib.axes.Axes or None, optional
            Existing axes associated with ``fig``.
        show : bool, optional
            Whether to display the figure.
        save : bool, optional
            Whether to save the figure as
            ``<ROI class name>_polygon.png``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the polygon.
        ax : matplotlib.axes.Axes
            Axes containing the polygon.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        x, y = self._polygon.exterior.xy
        ax.plot(x, y, color="blue")
        ax.set_title(f"{self.__class__.__name__} Polygon")
        ax.set_aspect("equal", adjustable="datalim")
        if show:
            plt.show()
        if save:
            fig.savefig(f"{self.__class__.__name__}_polygon.png")
        return fig, ax

    def _create_fingerprint(self) -> str:
        """Deterministic fingerprint (hash of ROI attributes)."""
        if not hasattr(self, "_polygon"):
            raise ValueError("Polygon must be generated before creating fingerprint.")
        payload = {
            "class": self.NAME,
            "config": self.config.to_dict(),
            "polygon": [mapping(self._polygon)],
        }
        data = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @abstractmethod
    def _generate_polygon(self) -> Polygon:
        """
        Generate the polygon for the region-of-interest.

        Returns
        -------
        Polygon
            Polygon representing the region-of-interest.
        """

    @abstractmethod
    def _get_wraparound_shifts(self) -> FloatArr:
        """
        Get the wrap-around shifts for this ROI and the given layout.

        Returns
        -------
        FloatArr
            Array of shape (M, 2) with wrap-around shift vectors.
        """

    @abstractmethod
    def wrap_position(self, pos_xy: FloatArr, eps_m: float = 1e-6) -> FloatArr:
        """Wrap a position from a periodic ROI copy into the central ROI.

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
        ValueError
            If the position cannot be associated with a periodic ROI copy.
        """
        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        point = Point(pos_xy)

        best_wrapped_xy: FloatArr | None = None
        best_dist = np.inf

        for offset in self.wraparound_shifts[1:]:
            wrapped_xy = pos_xy - offset
            if self.inside_roi(wrapped_xy, eps_m=eps_m):
                return wrapped_xy.astype(np.float32, copy=False)

            shifted_polygon = Polygon(
                np.asarray(self.polygon.exterior.coords, dtype=np.float64) + offset
            )
            if not shifted_polygon.buffer(eps_m).covers(point):
                continue

            dist = self.polygon.distance(Point(wrapped_xy))
            if dist < best_dist:
                best_dist = dist
                best_wrapped_xy = wrapped_xy

        if best_wrapped_xy is not None:
            raise ValueError(
                "Position matched at least one virtual ROI, but no wrapped "
                "candidate lies inside the original ROI. "
                f"pos_xy={pos_xy}, best_wrapped_xy={best_wrapped_xy}, "
                f"best_dist={best_dist}"
            )

        raise ValueError(
            f"Position {pos_xy} is outside the ROI, but not inside any virtual ROI copy."
        )

    def wrap_positions(
        self, pos_xy: FloatArr, eps_m: float = 1e-6, max_attempts: int = 10
    ) -> FloatArr:
        """Wrap positions from periodic ROI copies into the central ROI.

        Parameters
        ----------
        pos_xy : FloatArr
            One position with shape ``(2,)`` or positions with shape
            ``(..., 2)``.
        eps_m : float, optional
            Boundary tolerance in meters.
        max_attempts : int, optional
            Maximum number of wrapping operations per outside position.

        Returns
        -------
        FloatArr
            Wrapped positions with the same shape as ``pos_xy``.

        Raises
        ------
        RuntimeError
            If a position remains outside after ``max_attempts`` operations.
        ValueError
            If an outside position cannot be associated with a periodic ROI
            copy.
        """
        pos_xy = np.asarray(pos_xy, dtype=np.float32)

        is_1d = pos_xy.ndim == 1
        if is_1d:
            pos_xy = pos_xy[None, :]

        flat_pos = pos_xy.reshape(-1, 2)
        flat_wrapped = flat_pos.copy()
        mask_out = np.asarray(self.outside_roi(flat_wrapped, eps_m=eps_m), dtype=bool)

        for idx in np.flatnonzero(mask_out):
            pos = flat_wrapped[idx]
            for _ in range(max_attempts):
                if not self.outside_roi(pos, eps_m=eps_m):
                    break
                if self._is_near_boundary(pos, eps_m=eps_m):
                    pos = self._nudge_inside(pos, eps_m=eps_m)
                    break
                new_pos = self.wrap_position(pos, eps_m=eps_m)
                if self._is_near_boundary(new_pos, eps_m=eps_m):
                    pos = self._nudge_inside(new_pos, eps_m=eps_m)
                    break
                if np.allclose(new_pos, pos, atol=eps_m, rtol=0.0):
                    break
                pos = new_pos
            else:
                raise RuntimeError("Maximum wrap-around attempts exceeded.")
            flat_wrapped[idx] = pos

        wrapped = flat_wrapped.reshape(pos_xy.shape)
        return wrapped[0] if is_1d else wrapped

    def _is_near_boundary(self, pos_xy: FloatArr, eps_m: float) -> bool:
        point = Point(np.asarray(pos_xy, dtype=np.float64))
        return self.polygon.distance(point) <= max(100.0 * eps_m, 1e-4)

    def _nudge_inside(self, pos_xy: FloatArr, eps_m: float) -> FloatArr:
        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        centroid = np.asarray(self.polygon.centroid.coords[0], dtype=np.float64)
        direction = centroid - pos_xy
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            return pos_xy.astype(np.float32, copy=False)
        return (pos_xy + direction / norm * max(100.0 * eps_m, 1e-4)).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _extract_points(geometry: BaseGeometry) -> list[Point]:
        """Extract point geometries from a Shapely intersection result."""
        if geometry.is_empty:
            return []
        if isinstance(geometry, Point):
            return [geometry]
        if isinstance(geometry, MultiPoint):
            return list(geometry.geoms)
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            return [Point(coords[0]), Point(coords[-1])]
        return []

    @staticmethod
    def ensure_pos_dim(xy: FloatArr) -> FloatArr:
        """Ensure that a position array has shape ``(..., 2)``.

        Parameters
        ----------
        xy : FloatArr
            One position with shape ``(2,)`` or an array with final dimension
            two.

        Returns
        -------
        FloatArr
            Position array with at least two dimensions.

        Raises
        ------
        ValueError
            If ``xy`` does not have a final dimension of size two.
        """
        xy = np.asarray(xy)
        if xy.ndim == 1 and xy.shape[0] == 2:
            xy = xy[None, :]
        elif xy.ndim < 2 or xy.shape[-1] != 2:
            raise ValueError(f"pos_xy must have shape (..., 2), got {xy.shape}")
        return xy

    @property
    def polygon(self) -> Polygon:
        """Return the polygon of the ROI."""
        return self._polygon

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the polygon.

        Returns
        -------
        tuple[float, float, float, float]
            (x_min, x_max, y_min, y_max) of the bounding box.
        """
        x_min, y_min, x_max, y_max = self._polygon.bounds
        return x_min, x_max, y_min, y_max

    @property
    def bb_size(self) -> tuple[float, float]:
        """
        Return the size of the bounding box of the polygon.

        Returns
        -------
        tuple[float, float]
            (width, height) of the polygon.
        """
        x_min, y_min, x_max, y_max = self._polygon.bounds
        return x_max - x_min, y_max - y_min

    @property
    def polygon_coordinates(self) -> FloatArr:
        """Return the coordinates of the polygon."""
        return np.array(self._polygon.exterior.coords)

    @property
    def wraparound_shifts(self) -> FloatArr:
        """Get the wrap-around shifts for this ROI and the given layout."""
        if not hasattr(self, "_wraparound_shifts"):
            raise ValueError("Wrap-around shifts not computed.")
        return self._wraparound_shifts

    @property
    def fingerprint(self) -> str:
        """Return the fingerprint of the ROI."""
        return self._create_fingerprint()

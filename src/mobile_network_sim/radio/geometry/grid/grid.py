"""Grid creation utilities for 5G simulation."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..mobility.abstract_boundary_handler import AbstractBoundaryHandler
from ..mobility.wrap_boundary_handler import WraparoundBoundary


@dataclass(frozen=True)
class InterpolationGeometry:
    """Sample-local corner indices/weights, never random field values."""

    indices: np.ndarray
    weights: np.ndarray
    grid_key: tuple


class Grid:
    """Class to represent a grid of coordinates."""

    CACHE_CORNERS = True
    MAX_CACHED_CELLS = 4096

    def __getstate__(self):
        """Persist geometry only, never references to live boundary handlers."""
        return {
            key: value for key, value in self.__dict__.items() if key != "_corner_cache"
        }

    def _corner_indices(self, x_idx, y_idx, boundary_handler):
        """Reuse exact wrapped corner indices; weights remain position-specific."""
        if not hasattr(self, "_corner_cache"):
            self._corner_cache = OrderedDict()
        # The handler belongs to this geometry for its entire lifetime. Keeping
        # the object in the key prevents object-id reuse and cross-ROI hits.
        keys = [
            (boundary_handler, int(x), int(y))
            for x, y in zip(x_idx, y_idx, strict=True)
        ]
        indices = []
        for key in keys:
            value = self._corner_cache.get(key) if self.CACHE_CORNERS else None
            if value is None:
                x0 = self.xmin + key[1] * self.spacing
                y0 = self.ymin + key[2] * self.spacing
                points = np.array(
                    [
                        [x0, y0],
                        [x0 + self.spacing, y0],
                        [x0, y0 + self.spacing],
                        [x0 + self.spacing, y0 + self.spacing],
                    ]
                )
                if isinstance(boundary_handler, WraparoundBoundary):
                    points = boundary_handler.maybe_wrap_positions(points)
                _, value = self._get_nearest_grid_point(points)
                if self.CACHE_CORNERS:
                    self._corner_cache[key] = value
                    while len(self._corner_cache) > self.MAX_CACHED_CELLS:
                        self._corner_cache.popitem(last=False)
            elif self.CACHE_CORNERS:
                self._corner_cache.move_to_end(key)
            indices.append(value)
        return np.stack(indices, axis=1).reshape(-1, 2)

    def __init__(
        self, x_width: float, y_width: float, spacing: float, centered: bool = True
    ):
        """
        Initialize a grid of coordinates.

        Parameters
        ----------
        x_width : float
            Width of the grid in the x-direction.
        y_width : float
            Width of the grid in the y-direction.
        spacing : float
            Spacing between grid points.
        centered : bool, optional
            If True, the grid will contain an odd number of points for symmetry.
        """
        self.x_width = x_width
        self.y_width = y_width
        self.spacing = spacing
        self.centered = centered

        self.xy, self.idxs = self.create_grid(x_width, y_width, spacing, centered)
        self.xy_bins = self.compute_grid_edges(self.xy, spacing)

        self.xmin, self.ymin = self.xy.min(axis=(0, 1)).tolist()
        self.xmax, self.ymax = self.xy.max(axis=(0, 1)).tolist()

    def _get_nearest_grid_point(
        self, coordinates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the nearest grid points and their indices to the specified coordinates (x, y).

        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (..., 2) containing (x, y) coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The nearest grid point coordinates (..., 2) and their indices (..., 2).
        """
        x = coordinates[..., 0]
        y = coordinates[..., 1]

        x_idx = np.round((x - self.xmin) / self.spacing).astype(int)
        y_idx = np.round((y - self.ymin) / self.spacing).astype(int)

        x_idx = np.clip(x_idx, 0, self.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, self.shape[0] - 1)

        nearest_points = self.xy[y_idx, x_idx]
        idxs = np.stack([y_idx, x_idx], axis=-1)
        return nearest_points, idxs

    def _get_surrounding_point_data(
        self,
        value_maps: np.ndarray,
        coordinates: np.ndarray,
        boundary_handler: AbstractBoundaryHandler | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare surrounding-point data for grid-based sampling.

        Parameters
        ----------
        value_maps : np.ndarray
            Array of shape (n_fields, height, width) or (height, width).
        coordinates : np.ndarray
            Array of shape (n_points, 2) containing (x, y) coordinates.
        boundary_handler : AbstractBoundaryHandler | None
            Boundary handler for wrapping coordinates. If None, no wrapping is applied.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - values : np.ndarray
                Values at the four surrounding grid points, shape
                (4, n_fields, n_points), ordered as
                [lower-left, lower-right, upper-left, upper-right].
            - x : np.ndarray
                X coordinates of shape (n_points,).
            - y : np.ndarray
                Y coordinates of shape (n_points,).
            - cell_origins : np.ndarray
                Lower-left corner coordinates of the enclosing cell, shape
                (n_points, 2), with columns (x0, y0).
        """
        if value_maps.shape[-2:] != self.shape:
            raise ValueError(
                f"Value map {value_maps.shape[-2:]} does not match grid {self.shape}."
            )
        if value_maps.ndim == 2:
            value_maps = value_maps[np.newaxis]
        elif value_maps.ndim != 3:
            raise ValueError(
                "value_maps must be of shape (n_fields, height, width) or (height, width)"
            )

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n_points, 2)")

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_idx = np.floor((x - self.xmin) / self.spacing).astype(int)
        y_idx = np.floor((y - self.ymin) / self.spacing).astype(int)

        x0 = self.xmin + x_idx * self.spacing
        y0 = self.ymin + y_idx * self.spacing
        cell_origins = np.column_stack((x0, y0))

        if self.CACHE_CORNERS and 0 < len(x_idx) <= 32:
            idxs = self._corner_indices(x_idx, y_idx, boundary_handler)
        else:
            idxs = self._uncached_corner_indices(x0, y0, boundary_handler)

        values = value_maps[:, idxs[:, 0], idxs[:, 1]]
        values = values.reshape(value_maps.shape[0], 4, -1).transpose(1, 0, 2)
        return values, x, y, cell_origins

    def _uncached_corner_indices(self, x0, y0, boundary_handler):
        """Original vectorized geometry path, also used for large map queries."""
        surrounding_points = (
            np.array(
                [
                    [x0, y0],
                    [x0 + self.spacing, y0],
                    [x0, y0 + self.spacing],
                    [x0 + self.spacing, y0 + self.spacing],
                ]
            )
            .transpose(0, 2, 1)
            .reshape(-1, 2)
        )

        if isinstance(boundary_handler, WraparoundBoundary):
            surrounding_points = boundary_handler.maybe_wrap_positions(
                surrounding_points
            )

        _, idxs = self._get_nearest_grid_point(surrounding_points)

        return idxs

    def get_interpolated_values(
        self,
        value_maps: np.ndarray,
        coordinates: np.ndarray,
        boundary_handler: AbstractBoundaryHandler | None = None,
    ) -> np.ndarray:
        """
        Interpolate values at specified coordinates based on the grid.

        Parameters
        ----------
        value_maps : np.ndarray
            Array of shape (n_fields, height, width) or (height, width) containing
            field values.
        coordinates : np.ndarray
            Array of shape (n_points, 2) containing (x, y) coordinates for
            interpolation.
        boundary_handler : AbstractBoundaryHandler | None
            Boundary handler for wrapping coordinates. If None, no wrapping is applied.

        Returns
        -------
        np.ndarray
            Interpolated values at the specified coordinates, shape
            (n_fields, n_points).
        """
        return self.interpolate_prepared(
            value_maps, self.prepare_interpolation(coordinates, boundary_handler)
        )

    @property
    def interpolation_key(self) -> tuple:
        """Regular-grid geometry, evaluated from current parameters."""
        return (self.shape, self.spacing, self.xmin, self.ymin, self.xmax, self.ymax)

    def prepare_interpolation(self, coordinates, boundary_handler=None):
        """Prepare a single sample/query for any fields on this exact grid."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n_points, 2)")
        x, y = coordinates[:, 0], coordinates[:, 1]
        x_idx = np.floor((x - self.xmin) / self.spacing).astype(int)
        y_idx = np.floor((y - self.ymin) / self.spacing).astype(int)
        x0, y0 = self.xmin + x_idx * self.spacing, self.ymin + y_idx * self.spacing
        idxs = (
            self._corner_indices(x_idx, y_idx, boundary_handler)
            if self.CACHE_CORNERS and 0 < len(x_idx) <= 32
            else self._uncached_corner_indices(x0, y0, boundary_handler)
        )
        x_frac = ((x - x0) / self.spacing)[None, :]
        y_frac = ((y - y0) / self.spacing)[None, :]
        weights = np.array(
            [
                (1 - x_frac) * (1 - y_frac),
                x_frac * (1 - y_frac),
                (1 - x_frac) * y_frac,
                x_frac * y_frac,
            ]
        )
        return InterpolationGeometry(idxs, weights, self.interpolation_key)

    def interpolate_prepared(self, value_maps, geometry: InterpolationGeometry):
        """Apply field-specific NaN normalization to prepared geometry."""
        if geometry.grid_key != self.interpolation_key:
            raise ValueError("Interpolation geometry belongs to a different grid")
        if value_maps.ndim not in (2, 3) or value_maps.shape[-2:] != self.shape:
            raise ValueError("Value map does not match the interpolation grid")
        if value_maps.ndim == 2:
            value_maps = value_maps[None]
        idxs = geometry.indices
        values = value_maps[:, idxs[:, 0], idxs[:, 1]]
        values = values.reshape(value_maps.shape[0], 4, -1).transpose(1, 0, 2)
        weights = geometry.weights

        weights_masked = np.where(~np.isnan(values), weights, 0.0)
        weight_sum = np.sum(weights_masked, axis=0, keepdims=True)
        weights_norm = np.divide(
            weights_masked,
            weight_sum,
            out=np.zeros_like(weights_masked),
            where=weight_sum > 0,
        )

        return np.sum(np.nan_to_num(values) * weights_norm, axis=0)

    def get_nearest_surrounding_values(
        self,
        value_maps: np.ndarray,
        coordinates: np.ndarray,
        boundary_handler: AbstractBoundaryHandler | None = None,
    ) -> np.ndarray:
        """
        Return the value of the closest surrounding grid point for each coordinate.

        For every query coordinate, the enclosing grid cell is determined and the
        value of the nearest one of its four corner points is selected. If some of
        the surrounding values are NaN, they are ignored. If all four are NaN, the
        result is NaN for that query point.

        Parameters
        ----------
        value_maps : np.ndarray
            Array of shape (n_fields, height, width) or (height, width) containing
            field values.
        coordinates : np.ndarray
            Array of shape (n_points, 2) containing (x, y) coordinates.
        boundary_handler : AbstractBoundaryHandler | None
            Boundary handler for wrapping coordinates. If None, no wrapping is applied.

        Returns
        -------
        np.ndarray
            Array of shape (n_fields, n_points) containing the selected values.
        """
        values, x, y, cell_origins = self._get_surrounding_point_data(
            value_maps=value_maps,
            coordinates=coordinates,
            boundary_handler=boundary_handler,
        )

        x0 = cell_origins[:, 0]
        y0 = cell_origins[:, 1]

        dx = np.stack(
            [x - x0, x - (x0 + self.spacing), x - x0, x - (x0 + self.spacing)],
            axis=0,
        )
        dy = np.stack(
            [y - y0, y - y0, y - (y0 + self.spacing), y - (y0 + self.spacing)],
            axis=0,
        )
        dist_sq = dx**2 + dy**2  # (4, n_points)

        valid_mask = ~np.isnan(values)  # (4, n_fields, n_points)
        dist_sq_masked = np.where(valid_mask, dist_sq[:, np.newaxis, :], np.inf)

        nearest_idx = np.argmin(dist_sq_masked, axis=0)  # (n_fields, n_points)

        result = np.take_along_axis(
            values,
            nearest_idx[np.newaxis, :, :],
            axis=0,
        ).squeeze(axis=0)  # (n_fields, n_points)

        all_invalid = ~np.any(valid_mask, axis=0)  # (n_fields, n_points)
        if np.sum(all_invalid) > 0:
            result[all_invalid] = np.nan

        return result

    @staticmethod
    def create_grid(
        x_width: float, y_width: float, spacing: float, centered: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a grid of coordinates based on the specified size and spacing.

        Parameters
        ----------
        x_width : float
            Width of the grid in the x-direction.
        y_width : float
            Width of the grid in the y-direction.
        spacing : float
            Spacing between grid points.
        centered : bool, optional
            If True, the grid will contain an odd number of points for symmetry.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - x_grid: 2D array of x-coordinates.
            - y_grid: 2D array of y-coordinates.
            - x_idx_grid: 2D array of x indices.
            - y_idx_grid: 2D array of y indices.
        """
        nx = int(x_width // spacing)
        ny = int(y_width // spacing)

        if centered:  # Odd number of grid points for symmetry
            nx -= nx % 2 == 0
            ny -= ny % 2 == 0

        x_coords = spacing * (np.arange(nx) - (nx - 1) // 2)
        y_coords = spacing * (np.arange(ny) - (ny - 1) // 2)

        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")

        x_idx_grid, y_idx_grid = np.meshgrid(
            np.arange(x_grid.shape[1]), np.arange(x_grid.shape[0]), indexing="xy"
        )

        xy_grid = np.stack((x_grid, y_grid), axis=-1)
        xy_idx_grid = np.stack((x_idx_grid, y_idx_grid), axis=-1)

        return xy_grid, xy_idx_grid

    @staticmethod
    def compute_grid_edges(xy: np.ndarray, grid_spacing: float) -> np.ndarray:
        """
        Compute the edges of grid cells based on their center positions.

        Parameters
        ----------
        xy : np.ndarray
            2D array of shape (nx, ny, 2) containing the center coordinates
            of the grid cells.
        grid_spacing : float
            Spacing between grid points.

        Returns
        -------
        np.ndarray
            2D array of shape (nx + 1, ny + 1, 2) containing the edges of the grid cells.
        """
        nx, ny, _ = xy.shape

        xy_bins = np.zeros((nx + 1, ny + 1, 2), dtype=xy.dtype)
        xy_bins[1:, 1:, :] = xy + (grid_spacing / 2)

        xy_bins[1:, 0, :] = xy_bins[1:, 1, :] - np.array([grid_spacing, 0])
        xy_bins[0, 1:, :] = xy_bins[1, 1:, :] - np.array([0, grid_spacing])
        xy_bins[0, 0, :] = xy_bins[1, 1, :] - np.array([grid_spacing, grid_spacing])

        return xy_bins

    @staticmethod
    def flatten(*grid_arrays: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        Flatten one or more grids of coordinates into 2D arrays of shape (N, 2),
        where N is the number of grid points.

        Other Parameters
        ----------------
        *xy_grids : np.ndarray
            One or more 2D arrays of shape (height, width, 2) representing grid coordinates.

        Returns
        -------
        tuple[np.ndarray, ...] or np.ndarray
            If one array is given, returns a single flattened array of shape (N, 2).
            If multiple arrays are given, returns a tuple of flattened arrays.
        """
        flattened = tuple(grid.reshape(-1, 2) for grid in grid_arrays)
        return flattened[0] if len(flattened) == 1 else flattened

    @classmethod
    def interpolate_grid(cls, grid: "Grid", interpolation_factor: float) -> "Grid":
        """
        Create a new grid with interpolated coordinates based on the original grid.

        Parameters
        ----------
        grid : Grid
            The original grid to interpolate.
        interpolation_factor : float
            Factor by which to interpolate the grid spacing.

        Returns
        -------
        Grid
            A new Grid instance with interpolated coordinates.
        """
        xmin, xmax, ymin, ymax = grid.map_bounds
        return cls(xmax - xmin, ymax - ymin, grid.spacing / interpolation_factor, True)

    @property
    def x_grid(self) -> np.ndarray:
        """Return the x-coordinates of the grid."""
        return self.xy[:, :, 0]

    @property
    def y_grid(self) -> np.ndarray:
        """Return the y-coordinates of the grid."""
        return self.xy[:, :, 1]

    @property
    def x_coordinates(self) -> np.ndarray:
        """Return the x-coordinates of the grid."""
        return self.x_grid[0, :]

    @property
    def y_coordinates(self) -> np.ndarray:
        """Return the y-coordinates of the grid."""
        return self.y_grid[:, 0]

    @property
    def x_bin_grid(self) -> np.ndarray:
        """Return the x-coordinates of the grid bins."""
        return self.xy_bins[:, :, 0]

    @property
    def y_bin_grid(self) -> np.ndarray:
        """Return the y-coordinates of the grid bins."""
        return self.xy_bins[:, :, 1]

    @property
    def x_bins(self) -> np.ndarray:
        """Return the x-coordinates of the grid bins."""
        return self.x_bin_grid[0, :]

    @property
    def y_bins(self) -> np.ndarray:
        """Return the y-coordinates of the grid bins."""
        return self.y_bin_grid[:, 0]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape (rows, columns) of the grid."""
        return self.xy.shape[:2]

    @property
    def minima(self) -> tuple[float, float]:
        """Return the minimum coordinates of the grid."""
        return self.xmin, self.ymin

    @property
    def maxima(self) -> tuple[float, float]:
        """Return the maximum coordinates of the grid."""
        return self.xmax, self.ymax

    @property
    def grid_size(self) -> tuple[float, float]:
        """Return the size of the grid in the x and y directions."""
        return self.xmax - self.xmin, self.ymax - self.ymin

    @property
    def grid_bounds(self) -> tuple[float, float, float, float]:
        """Return the bounds of the grid as (min_x, max_x, min_y, max_y)."""
        return self.xmin, self.xmax, self.ymin, self.ymax

    @property
    def map_bounds(self) -> tuple[float, float, float, float]:
        """Return the bounds of the original map as (xmin, xmax, ymin, ymax)."""
        xmin, xmax = -self.x_width / 2, self.x_width / 2
        ymin, ymax = -self.y_width / 2, self.y_width / 2
        return xmin, xmax, ymin, ymax

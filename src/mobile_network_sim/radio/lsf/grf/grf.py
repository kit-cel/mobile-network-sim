r"""Gaussian random fields with exponential spatial correlation."""

import contextlib
import hashlib
import json
import os
import pickle
import tempfile
import warnings
from collections import OrderedDict
from copy import copy, deepcopy
from threading import RLock
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cholesky
from tqdm import tqdm as pbar

from ....paths import cache_root
from ...geometry.grid.grid import Grid
from ...geometry.mobility.abstract_boundary_handler import AbstractBoundaryHandler
from ...geometry.mobility.wrap_boundary_handler import WraparoundBoundary

_MATRIX_POOL: OrderedDict[tuple, tuple] = OrderedDict()
_MATRIX_POOL_LOCK = RLock()
_MATRIX_POOL_LIMIT = 1024 * 1024 * 1024

if TYPE_CHECKING:
    from ...geometry.roi.abstract_roi import AbstractROI


class GaussianRandomField:
    r"""Generate Gaussian random fields over a region of interest.

    The covariance between two grid points separated by distance
    :math:`\Delta x` is

    .. math::

       R(\Delta x) = \exp\left(-\frac{\Delta x}{d_{\mathrm{decorr}}}\right).

    Thus, ``d_decorr`` is the :math:`1/e` correlation distance used by
    3GPP TR 38.901, not a half-correlation distance. A covariance matrix
    :math:`\mathbf{C}` is formed from this kernel and factored as
    :math:`\mathbf{C}=\mathbf{L}\mathbf{L}^{\mathsf{T}}`; multiplying
    independent standard-normal samples by :math:`\mathbf{L}` produces the
    correlated grid samples.

    Notes
    -----
    Some earlier mobile-radio specifications define a decorrelation distance
    at which correlation equals one half and therefore include
    :math:`\ln(2)` in the exponent. That convention is not used here.

    The Cholesky construction follows the core random-field method described
    by Yang et al. The simulator interpolates generated maps separately and
    does not implement that paper's ordinary-kriging stage.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clause 7.4.4, equation (7.4-5), May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    .. [2] ETSI, *Selection procedures for the choice of radio transmission
       technologies of the UMTS*, ETSI TR 101 112 V3.2.0,
       clause B.1.4.1.4 and Annex D.4, April 1998.
       https://www.etsi.org/deliver/etsi_tr/101100_101199/101112/
       03.02.00_60/tr_101112v030200p.pdf
    .. [3] M. Gudmundson, "Correlation Model for Shadow Fading in Mobile
       Radio Systems," *Electronics Letters*, vol. 27, no. 23,
       pp. 2145--2146, 1991. doi:10.1049/el:19911328.
    .. [4] Y. Yang, P. Wang, and S. J. Brandenberg, "An Algorithm for
       Generating Spatially Correlated Random Fields Using Cholesky
       Decomposition and Ordinary Kriging," *Computers and Geotechnics*,
       vol. 147, article 104783, 2022. doi:10.1016/j.compgeo.2022.104783.
    """

    CACHE_DIR: str | None = ".cache"
    USE_CACHED_MATRICES: bool = True
    CACHE_SCHEMA: str = "v3_seed_independent_cholesky_exp_e1"
    USE_PROCESS_CACHE: bool = True

    @staticmethod
    def clear_process_cache() -> None:
        """Release retained deterministic resources without touching disk caches."""
        with _MATRIX_POOL_LOCK:
            _MATRIX_POOL.clear()

    def _process_cache_key(self):
        if not (
            self.USE_PROCESS_CACHE
            and self.USE_CACHED_MATRICES
            and self.CACHE_DIR is not None
        ):
            return None
        path = os.path.join(self._cache_dir, self._cache_filename)
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return None
        return (path, stat.st_mtime_ns, stat.st_size)

    def _remember_matrices(self) -> None:
        key = self._process_cache_key()
        if key is None:
            return
        arrays = (
            self.coordinates_in_roi,
            self.idxs_in_roi,
            self.distance_mat,
            self.cholesky_mat,
        )
        size = sum(array.nbytes for array in arrays)
        if size > _MATRIX_POOL_LIMIT:
            return
        for array in arrays:
            array.flags.writeable = False
        with _MATRIX_POOL_LOCK:
            _MATRIX_POOL[key] = (deepcopy(self.grid), arrays, size)
            _MATRIX_POOL.move_to_end(key)
            while sum(entry[2] for entry in _MATRIX_POOL.values()) > _MATRIX_POOL_LIMIT:
                _MATRIX_POOL.popitem(last=False)

    grid: Grid
    resource_status: str
    # Compatibility for legacy deserialized/partially reconstructed objects.
    # Every normal constructor stores its explicitly supplied instance dtype.
    factor_dtype: np.dtype = np.dtype("float64")

    coordinates_in_roi: np.ndarray
    idxs_in_roi: np.ndarray

    distance_mat: np.ndarray
    cholesky_mat: np.ndarray

    seed: int
    np_rng: np.random.Generator

    def __init__(
        self,
        roi: "AbstractROI",
        boundary_handler: AbstractBoundaryHandler,
        *,
        grid_spacing: float = 10.0,
        d_decorr: float = 20.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        field_corr_factor: float | None = None,
        roi_extension_factor: float = 1.0,
        progress_bar: bool = True,
        map_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        _init_setup_matrices: bool = True,
        resource_storage: str = "legacy",
        factor_dtype: str | np.dtype = "float64",
    ):
        """Initialize a Gaussian random field generator.

        Parameters
        ----------
        roi : AbstractROI
            Region of interest (ROI).
        boundary_handler : AbstractBoundaryHandler
            Boundary handler used when field values are sampled outside the
            region of interest.
        grid_spacing : float, optional
            Spacing between grid points in meters.
        d_decorr : float, optional
            The :math:`1/e` correlation distance in meters for the exponential
            autocorrelation function.
        mu : float, optional
            Mean of the Gaussian random field. Default is 0.0.
        sigma : float, optional
            Standard deviation of the Gaussian random field.
        field_corr_factor : float, optional
            Correlation factor for the fields. If None, fields are uncorrelated. If specified, it
            should be in the range [0, 1] to blend correlated and uncorrelated fields.
        roi_extension_factor: float, optional
            Inflation factor for the region of interest. This determines how far the grid extends
            beyond the ROI.
        progress_bar : bool, optional
            If True, displays a progress bar during computations.
        map_dtype : str or numpy.dtype, optional
            Floating-point dtype used to store generated maps.
        seed : int | None, optional
            Random seed for reproducibility. If None, uses a random seed based on the system time.
        _init_setup_matrices : bool, optional
            If True, initializes the grid and correlation matrices. This is
            used internally to avoid redundant initialization when copying an
            instance.
        """
        if grid_spacing <= 0:
            raise ValueError("grid_spacing must be a positive number.")
        if d_decorr <= 0:
            raise ValueError("d_decorr must be a positive number.")
        if sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        if field_corr_factor is not None and not 0 <= field_corr_factor <= 1:
            raise ValueError("field_corr_factor must be in the range [0, 1].")
        if roi_extension_factor < 1:
            raise ValueError("roi_extension_factor must be at least 1.")

        self.roi = roi
        self.boundary_handler = boundary_handler
        self.grid_spacing = grid_spacing

        self.d_decorr = d_decorr
        self.alpha = 1.0 / d_decorr
        self.mu = mu
        self.sigma = sigma

        self.field_corr_factor = field_corr_factor
        self.roi_extension_factor = roi_extension_factor
        self.progress_bar = progress_bar
        self.map_dtype = np.dtype(map_dtype)
        self.resource_storage = resource_storage
        self.factor_dtype = np.dtype(factor_dtype)
        if self.factor_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("factor_dtype must be float32 or float64")

        if _init_setup_matrices:
            self._setup_matrices()

        self.set_seed(seed)

    def _setup_matrices(self) -> None:
        if self.resource_storage == "shared_mmap":
            from ...resources import attach_grf, grf_resource

            attach_grf(self, grf_resource(self))
            return
        key = self._process_cache_key()
        with _MATRIX_POOL_LOCK:
            entry = _MATRIX_POOL.get(key)
            if entry is not None:
                _MATRIX_POOL.move_to_end(key)
                self.grid = deepcopy(entry[0])
                (
                    self.coordinates_in_roi,
                    self.idxs_in_roi,
                    self.distance_mat,
                    self.cholesky_mat,
                ) = entry[1]
                return
        # Try to load cached matrices
        if self.USE_CACHED_MATRICES and self._load_cached_matrices():
            self._remember_matrices()
            return

        self._compute_matrices()
        if self.USE_CACHED_MATRICES:
            self._save_cached_matrices()
            self._remember_matrices()

    def _compute_matrices(self) -> None:
        """Build deterministic geometry without publishing or realizing fields."""
        if self.factor_dtype != np.dtype("float64"):
            # Reuse the historical float64 factorization, if already cached.
            # Never change RNG state, grid geometry or an existing cache file.
            source = copy(self)
            source.factor_dtype = np.dtype("float64")
            source._setup_matrices()
            self.grid = deepcopy(source.grid)
            for name in ("coordinates_in_roi", "idxs_in_roi", "distance_mat"):
                setattr(self, name, getattr(source, name))
            self.cholesky_mat = source.cholesky_mat.astype(self.factor_dtype)
            return
        # Create the grid
        xsize, ysize = self.roi.bb_size
        self.grid = Grid(
            xsize * self.roi_extension_factor,
            ysize * self.roi_extension_factor,
            self.grid_spacing,
            centered=True,
        )

        # Get coordinates and indices in the ROI
        self.coordinates_in_roi, self.idxs_in_roi = (
            self.roi.get_positions_and_idxs_in_roi(
                *Grid.flatten(self.grid.xy, self.grid.idxs)
            )
        )

        self.distance_mat = self._compute_pairwise_distance_matrix(
            self.coordinates_in_roi
        )
        self.cholesky_mat = self._compute_cholesky_matrix(self.distance_mat)

    def _save_cached_matrices(self) -> None:
        """Save seed-independent GRF geometry and correlation factors."""
        if self.CACHE_DIR is None:
            warnings.warn("cache_dir is None, not saving GRF matrices to cache.")
            return
        cache_dir = self._cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = self._cache_filename
        cache_path = os.path.join(cache_dir, cache_filename)

        fd, tmp_cache_path = tempfile.mkstemp(
            prefix=f"{cache_filename}.", suffix=".tmp", dir=cache_dir
        )
        os.close(fd)
        try:
            with open(tmp_cache_path, "wb") as f:
                pickle.dump(
                    {
                        "cache_schema": self.CACHE_SCHEMA,
                        "cache_identity": self._matrix_cache_identity,
                        "grid": self.grid,
                        "coordinates_in_roi": self.coordinates_in_roi,
                        "idxs_in_roi": self.idxs_in_roi,
                        "distance_mat": self.distance_mat,
                        "cholesky_mat": self.cholesky_mat,
                        "map_dtype": self.map_dtype.name,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(tmp_cache_path, cache_path)
            print(
                "Saved seed-independent GRF correlation cache "
                "(grid, ROI coordinates/indices, pairwise ROI-grid-distance matrix, "
                f"Cholesky factor; no random-field realization): {cache_path}"
            )
        except OSError as e:
            print(f"Failed to save GRF matrices to cache: {e}")
            with contextlib.suppress(OSError):
                os.remove(tmp_cache_path)

    @property
    def _cache_filename(self) -> str:
        """Return a collision-resistant key for deterministic matrix inputs."""
        payload = json.dumps(
            self._matrix_cache_identity, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"grf_geometry_{self.CACHE_SCHEMA}_{digest}.pkl"

    @staticmethod
    def _array_fingerprint(values: np.ndarray) -> str:
        array = np.ascontiguousarray(values)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def _matrix_cache_identity(self) -> dict[str, object]:
        """Inputs to the distance matrix and Cholesky factor.

        The realization seed, field mean/variance, and storage map dtype are
        intentionally absent: none participates in these deterministic
        matrices. Exact floating-point values are hashed instead of rounded
        filename fragments, avoiding collisions such as 5.001 vs. 5.004 m.

        Base-station positions are intentionally absent as well. This cache
        covers correlation between ROI-grid points. LOS and shadowing models
        compute their separate correlation between BS-specific fields from
        the current ``bs_pos_2d`` every time a realization is generated.
        Changing a random-layout seed therefore reuses this cache only when
        the ROI grid itself is unchanged, while still recomputing the
        BS-distance-dependent matrix.
        """
        return {
            "cache_schema": self.CACHE_SCHEMA,
            **(
                {
                    "factor_dtype": self.factor_dtype.name,
                    "factor_recipe": "float64_cholesky_cast_v1",
                }
                if self.factor_dtype != np.dtype("float64")
                else {}
            ),
            "roi_fingerprint": self.roi.fingerprint,
            "wraparound_shifts_sha256": self._array_fingerprint(
                self.roi.wraparound_shifts
            ),
            "grid_spacing_m": float(self.grid_spacing),
            "decorrelation_distance_m": float(self.d_decorr),
            "roi_extension_factor": float(self.roi_extension_factor),
        }

    def _load_cached_matrices(self, *, remove_invalid: bool = True) -> bool:
        """Load cached GRF matrices if available."""
        if self.CACHE_DIR is None:
            return False
        cache_dir = self._cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            return False

        cache_filename = self._cache_filename
        cache_path = os.path.join(cache_dir, cache_filename)

        if not os.path.isfile(cache_path):
            return False

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            if not isinstance(cached_data, dict):
                raise ValueError(  # noqa: TRY004 - preserve the cache error contract
                    f"Invalid cache payload type {type(cached_data).__name__}"
                )
            required_keys = {
                "cache_schema",
                "cache_identity",
                "grid",
                "coordinates_in_roi",
                "idxs_in_roi",
                "distance_mat",
                "cholesky_mat",
            }
            missing_keys = required_keys - set(cached_data.keys())
            if missing_keys:
                raise ValueError(f"Missing cache keys: {sorted(missing_keys)}")
            if cached_data["cache_schema"] != self.CACHE_SCHEMA:
                raise ValueError(
                    "Cached correlation convention does not match the current model."
                )
            if cached_data["cache_identity"] != self._matrix_cache_identity:
                raise ValueError("Cached matrix inputs do not match this GRF.")
            self.grid = cached_data["grid"]
            self.coordinates_in_roi = cached_data["coordinates_in_roi"]
            self.idxs_in_roi = cached_data["idxs_in_roi"]
            self.distance_mat = cached_data["distance_mat"]
            self.cholesky_mat = cached_data["cholesky_mat"]
            print(
                "Loaded seed-independent GRF correlation cache "
                "(grid, ROI coordinates/indices, pairwise ROI-grid-distance matrix, "
                f"Cholesky factor; no random-field realization): {cache_path}"
            )
            return True
        except (
            EOFError,
            OSError,
            ValueError,
            KeyError,
            pickle.UnpicklingError,
            ModuleNotFoundError,
        ) as e:
            print(f"Failed to load GRF matrices from cache: {e}")
            if remove_invalid:
                with contextlib.suppress(OSError):
                    os.remove(cache_path)
            return False

    @property
    def _cache_dir(self) -> str:
        if self.CACHE_DIR is None:
            raise RuntimeError(
                "GRF cache directory requested while caching is disabled."
            )
        if self.CACHE_DIR == ".cache":
            return str(cache_root())
        return str(cache_root() / self.CACHE_DIR)

    @classmethod
    def copy(
        cls: "type[GaussianRandomField]",
        grf: "GaussianRandomField",
        seed: int | None = None,
    ) -> "GaussianRandomField":
        """
        Create a copy of the GaussianRandomField instance with a new random seed.

        Parameters
        ----------
        grf : GaussianRandomField
            The GaussianRandomField instance to copy.
        seed : int | None, optional
            Random seed for the new instance.

        Returns
        -------
        GaussianRandomField
            Copy that reuses the precomputed grids and matrices.
        """
        roi, boundary = (
            deepcopy((grf.roi, grf.boundary_handler))
            if grf.resource_storage == "shared_mmap"
            else (grf.roi, grf.boundary_handler)
        )
        grf_copy = cls(
            roi=roi,
            boundary_handler=boundary,
            grid_spacing=grf.grid_spacing,
            d_decorr=grf.d_decorr,
            mu=grf.mu,
            sigma=grf.sigma,
            field_corr_factor=grf.field_corr_factor,
            map_dtype=grf.map_dtype,
            seed=seed,
            _init_setup_matrices=False,
            resource_storage=grf.resource_storage,
            factor_dtype=grf.factor_dtype,
            roi_extension_factor=grf.roi_extension_factor,
        )

        grf_copy._update_grids_and_matrices(grf)

        return grf_copy

    def _update_grids_and_matrices(self, grf: "GaussianRandomField") -> None:
        if hasattr(grf, "_shared_resource"):
            from ...resources import attach_grf

            attach_grf(self, grf._shared_resource)
            self.resource_status = grf.resource_status
            return
        self.grid = deepcopy(grf.grid)
        self.coordinates_in_roi = deepcopy(grf.coordinates_in_roi)
        self.idxs_in_roi = deepcopy(grf.idxs_in_roi)
        self.distance_mat = deepcopy(grf.distance_mat)
        self.cholesky_mat = deepcopy(grf.cholesky_mat)

    def release_shared_resources(self) -> None:
        """Drop this owner's mappings; a subsequent reset can reopen them."""
        if hasattr(self, "_shared_resource"):
            from ...resources import GRF_ARRAYS

            for name in (*GRF_ARRAYS, "grid", "_shared_resource"):
                self.__dict__.pop(name, None)

    def set_seed(self, seed: int | None) -> None:
        """Set the random seed for field generation.

        Parameters
        ----------
        seed : int | None
            Random seed. If ``None``, sample one from NumPy's global state.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        # Get state of the RNG for reproducibility if needed
        self.np_rng_state = self.np_rng.bit_generator.state

    def _compute_pairwise_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise distance matrix for coordinates in the region of interest (ROI).

        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (n, 2) where n is the number of coordinates and each
            coordinate is given by (x, y) in meters.

        Returns
        -------
        np.ndarray
            Pairwise distance matrix of shape (n, n) where n is the number of coordinates.
        """
        n = coordinates.shape[0]

        offsets = self.roi.wraparound_shifts
        wrapped_coords = coordinates[:, np.newaxis, :] - offsets[np.newaxis, :, :]

        iterator = pbar(
            range(1, n),
            desc=f"{self.__class__.__name__} - generate map",
            unit="positions",
            unit_scale=True,
            disable=not self.progress_bar or n < 1_000,
        )

        distance_mat = np.zeros((n, n))
        for i in iterator:
            deltas = coordinates[i] - wrapped_coords[:i]  # shape (i, num_wraps, 2)
            dists = np.linalg.norm(deltas, axis=2).min(axis=1)
            distance_mat[i, :i] = dists

        return distance_mat + distance_mat.T  # Symmetric matrix

    def _compute_cholesky_matrix(self, distance_mat: np.ndarray) -> np.ndarray:
        """
        Computes the covariance matrix using an exponential ACF model.

        Returns
        -------
        np.ndarray
            Cholesky decomposition of the covariance matrix.
        """
        corr_mat = np.exp(-self.alpha * distance_mat)
        return cholesky(corr_mat, lower=True, check_finite=False).astype(
            self.factor_dtype, copy=False
        )

    def generate_fields(self, n_fields: int, normalize: bool = True) -> np.ndarray:
        """
        Generate Gaussian random fields based on the correlation matrix.

        Parameters
        ----------
        n_fields : int
            Number of fields to generate.
        normalize : bool, optional
            If True, normalize the generated fields to have zero mean and unit variance.
            Default is True.

        Returns
        -------
        np.ndarray
            Generated fields of shape (n_fields, height, width), where height and width
            are determined by the grid shape.
        """
        if not hasattr(self, "cholesky_mat"):
            self._setup_matrices()
        n_fields_ = n_fields if self.field_corr_factor is None else n_fields + 1
        uncorr_samples = self.sigma * self.np_rng.standard_normal(
            (n_fields_, self.coordinates_in_roi.shape[0])
        )
        # Keep the exact original float64 random draw and its shape. Only the
        # deterministic matrix multiplication and normalization are reduced.
        uncorr_samples = uncorr_samples.astype(self.factor_dtype, copy=False)

        corr_values = uncorr_samples @ self.cholesky_mat.T

        if self.field_corr_factor is not None:
            # Linear correlation blending to correlate the fields
            base_map = corr_values[0]
            uncorrelated_maps = corr_values[1:]

            field_values = (
                np.sqrt(self.field_corr_factor) * base_map
                + np.sqrt(1 - self.field_corr_factor) * uncorrelated_maps
            )
        else:
            field_values = corr_values

        if normalize:
            field_values -= np.mean(field_values, axis=1, keepdims=True)
            field_stds = np.std(field_values, axis=1, keepdims=True)
            if np.any(field_stds == 0):
                raise ValueError(
                    "Cannot normalize a generated field with zero variance."
                )
            field_values /= field_stds
            field_values += self.mu
            field_values *= self.sigma

        fields = np.zeros((n_fields, *self.grid.shape), dtype=self.map_dtype)
        for i in range(n_fields):
            fields[i, self.idxs_in_roi[:, 1], self.idxs_in_roi[:, 0]] = field_values[
                i, :
            ]

        return fields

    def _get_fields_values_for_coord(
        self, fields: np.ndarray, coordinate: np.ndarray
    ) -> float | np.ndarray:
        """
        Get the field values at a specific coordinate, handling wrapping if necessary.
        """
        if self.roi.outside_roi(coordinate):
            if not isinstance(self.boundary_handler, WraparoundBoundary):
                if fields.ndim == 2:
                    return 0.0
                return np.zeros(fields.shape[0], dtype=fields.dtype)
            coordinate = self.boundary_handler.maybe_wrap_positions(coordinate)

        xi = np.clip(
            np.round((coordinate[0] - self.grid.xmin) / self.grid_spacing),
            0,
            self.grid.shape[1] - 1,
        ).astype(int)
        yi = np.clip(
            np.round((coordinate[1] - self.grid.ymin) / self.grid_spacing),
            0,
            self.grid.shape[0] - 1,
        ).astype(int)

        if fields.ndim == 2:
            return fields[yi, xi]
        return fields[:, yi, xi]

    def get_grf_values(
        self, fields: np.ndarray, coordinates: np.ndarray, *, interpolation=None
    ) -> np.ndarray:
        """
        Get the values of the fields at specified coordinates. Bilinear interpolation is used
        to compute the values at non-grid points.

        Parameters
        ----------
        fields : np.ndarray
            Fields to use, shape (n_fields, height, width).
        coordinates : np.ndarray
            Coordinates where to get the field values, shape (N, 2) = (N, (x, y)).

        Returns
        -------
        np.ndarray
            Interpolated field values at the specified coordinates, shape (n_fields, N).
        """
        if interpolation is not None:
            return self.grid.interpolate_prepared(fields, interpolation)
        return self.grid.get_interpolated_values(
            fields, coordinates, boundary_handler=self.boundary_handler
        )

    def get_wrapped_fields(
        self,
        fields: np.ndarray,
        idxs_in_roi: np.ndarray | None = None,
        spacing: float | None = None,
    ) -> np.ndarray:
        """
        Get the wrapped fields for the region of interest (ROI).

        Parameters
        ----------
        fields : np.ndarray
            Fields to wrap, shape (n_fields, height, width).
        idxs_in_roi : np.ndarray, optional
            Indices of the fields in the ROI. If None, uses self.idxs_in_roi.
        spacing : float, optional
            Spacing of the grid. If None, uses self.grid_spacing.

        Returns
        -------
        np.ndarray
            Wrapped fields.

        Notes
        -----
        If the ROI is not wrapped, the original fields are returned without wrapping.
        If interpolated fields are provided, `idxs_in_roi` and `spacing` must be specified.
        """
        if not isinstance(self.boundary_handler, WraparoundBoundary):
            return fields

        idxs_in_roi = self.idxs_in_roi if idxs_in_roi is None else idxs_in_roi
        spacing = self.grid_spacing if spacing is None else spacing

        if fields.ndim == 2:
            fields = fields[np.newaxis, :, :]

        idx_offsets = np.round(self.roi.wraparound_shifts / spacing).astype(int)

        idxs_with_offsets = (
            idxs_in_roi[np.newaxis, :, :] + idx_offsets[:, np.newaxis, :]
        )
        idxs_with_offsets -= idxs_with_offsets.min(axis=(0, 1))[
            np.newaxis, np.newaxis, :
        ]

        idxs_max = idxs_with_offsets.max(axis=(0, 1))
        wrapped_fields = np.zeros(
            (fields.shape[0], idxs_max[1] + 1, idxs_max[0] + 1), dtype=fields.dtype
        )
        for x_idxs, y_idxs in idxs_with_offsets.transpose(0, 2, 1):
            wrapped_fields[:, y_idxs, x_idxs] = fields[
                :, idxs_in_roi[:, 1], idxs_in_roi[:, 0]
            ]

        return wrapped_fields

    def get_interpolated_fields(
        self,
        fields: np.ndarray,
        interpolation_factor: int,
        normalize: bool = True,
        wrapped: bool = False,
    ) -> np.ndarray:
        """
        Interpolate the fields to a higher resolution based on the interpolation factor.

        Parameters
        ----------
        fields : np.ndarray
            Fields to interpolate, shape (n_fields, height, width).
        interpolation_factor : int
            Factor by which to interpolate the fields.
        normalize : bool, optional
            If True, normalize the interpolated fields to match the original fields'
            mean and standard deviation.
        wrapped : bool, optional
            If True, wrap the interpolated fields to the region of interest (ROI).

        Returns
        -------
        np.ndarray
            Interpolated fields of shape:
            (n_fields, height * interpolation_factor, width * interpolation_factor).
        """
        if interpolation_factor <= 1:
            return fields

        n_fields = fields.shape[0]

        f_mu = fields.mean(axis=(1, 2), keepdims=True)
        f_sigma = fields.std(axis=(1, 2), keepdims=True)

        ip_grid = Grid.interpolate_grid(self.grid, interpolation_factor)

        ip_coords_roi, ip_idxs_roi = self.roi.get_positions_and_idxs_in_roi(
            *Grid.flatten(ip_grid.xy, ip_grid.idxs)
        )

        ip_field_values = self.get_grf_values(fields, ip_coords_roi)

        ip_fields = np.zeros((n_fields, *ip_grid.shape), dtype=fields.dtype)
        ip_fields[:, ip_idxs_roi[:, 1], ip_idxs_roi[:, 0]] = ip_field_values

        if normalize:
            ip_fields = ip_fields - f_mu
            ip_fields /= np.where(f_sigma > 0, f_sigma, 1.0)

        if wrapped:
            return self.get_wrapped_fields(
                ip_fields,
                idxs_in_roi=ip_idxs_roi,
                spacing=self.grid_spacing / interpolation_factor,
            )
        return ip_fields

    def validate_mean_std(self, fields: np.ndarray, atol: float = 0.5) -> None:
        """
        Validates the mean and standard deviation of the fields.

        Parameters
        ----------
        fields : np.ndarray
            Fields to validate, shape (n_fields, height, width).
        atol : float, optional
            Absolute tolerance for the mean and standard deviation checks. Default is 0.5 dB.
        """
        field_values = self.get_grf_values(fields, self.coordinates_in_roi)
        mean = np.mean(field_values)
        std = np.std(field_values)

        print(
            "=== Mean and Standard Deviation Check ===\n",
            f"Mean: {mean:.3f} (expected: {self.mu:.1f})\n",
            f"Std Dev: {std:.3f} (expected: {self.sigma:.1f})",
        )

        if abs(mean - self.mu) > atol:
            print("WARNING: Mean deviates significantly from expected value.")
        if abs(std - self.sigma) > atol:
            print("WARNING: Std dev deviates significantly from expected value.")

    def validate_spatial_correlation(
        self, fields: np.ndarray, n_pairs: int = 1_000, n_angles: int = 16
    ) -> None:
        """
        Validates the correlation of shadow fading values at different distances.

        Parameters
        ----------
        fields : numpy.ndarray
            Fields whose spatial correlation is evaluated.
        n_pairs : int, optional
            Number of pairs of positions to test at different distances. Default is 1,000.
        n_angles : int, optional
            Number of angles to sample around each base position. Default is 16.
        """
        if n_pairs <= 0:
            raise ValueError("Number of pairs must be a positive integer.")
        if n_pairs < 30:
            warnings.warn(
                "Using a small number of pairs may lead to unreliable correlation estimates."
            )
        if n_angles <= 0:
            raise ValueError("Number of angles must be a positive integer.")
        if n_angles < 3:
            warnings.warn(
                "Using fewer than 3 angles may lead to unreliable correlation estimates."
            )

        dist = np.array([5, 10, 20, 50, 100, 200])  # distances in m
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)[np.newaxis, :]
        base_pos = self.roi.sample_positions(n_pairs)

        offset_vectors = (
            np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            * dist[:, np.newaxis, np.newaxis]
        )
        offset_pos = (
            base_pos[:, np.newaxis, np.newaxis, :] + offset_vectors[np.newaxis, :, :, :]
        ).reshape(-1, 2)
        if isinstance(self.boundary_handler, WraparoundBoundary):
            offset_pos = self.boundary_handler.maybe_wrap_positions(offset_pos)

        sfv1 = self.get_grf_values(fields[0], base_pos)
        sfv2 = self.get_grf_values(fields[0], offset_pos).reshape(
            -1, n_pairs, len(dist), n_angles
        )

        corr_theo = np.exp(-self.alpha * dist)

        print("=== Correlation vs. Distance Check ===")
        f1_rep = np.repeat(sfv1, n_angles)
        for i, (r_theo, d) in enumerate(zip(corr_theo, dist)):
            f2 = sfv2[:, :, i, :].reshape(-1)
            r_emp = np.corrcoef(f1_rep, f2)[0, 1]
            print(f"d = {d:5.1f} m | r_emp = {r_emp:.3f} | r_theo = {r_theo:.3f}")

    @property
    def wrapped_map_bounds(self) -> tuple[float, float, float, float]:
        """Return the wrapped bounds of the map grid as (min_x, max_x, min_y, max_y)."""
        if not isinstance(self.boundary_handler, WraparoundBoundary):
            return self.grid.grid_bounds
        xmin, ymin = self.roi.wraparound_shifts.min(axis=0)
        xmax, ymax = self.roi.wraparound_shifts.max(axis=0)
        return (
            self.grid.xmin + xmin,
            self.grid.xmax + xmax,
            self.grid.ymin + ymin,
            self.grid.ymax + ymax,
        )

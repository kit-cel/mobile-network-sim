"""Module for the LSF map."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import tempfile
from typing import TYPE_CHECKING
import warnings

import numpy as np

from ...paths import cache_root

from ..geometry.grid.grid import Grid
from ..utils.types import FloatArr

if TYPE_CHECKING:
    from .lsf import LSFModel
    from ..geometry.simulation_geometry import SimulationGeometry

MapArray = FloatArr | np.memmap


@dataclass(frozen=True, slots=True)
class CoverageMaps:
    """Container for large-scale fading maps."""

    pl_db: MapArray
    cg_db: MapArray
    rsrp_dbm: MapArray
    sinr_db: MapArray
    los: MapArray
    sf_db: MapArray

    def as_dict(self) -> dict[str, MapArray]:
        """Return the maps as a dictionary.

        Returns
        -------
        dict[str, MapArray]
            Mapping from map names to map arrays.
        """
        return {
            "pl_db": self.pl_db,
            "cg_db": self.cg_db,
            "rsrp_dbm": self.rsrp_dbm,
            "sinr_db": self.sinr_db,
            "los": self.los,
            "sf_db": self.sf_db,
        }

    @staticmethod
    def map_names() -> list[str]:
        """Return the map names in a fixed order.

        Returns
        -------
        list[str]
            Map names in serialization order.
        """
        return [
            "pl_db",
            "cg_db",
            "rsrp_dbm",
            "sinr_db",
            "los",
            "sf_db",
        ]

    @classmethod
    def from_dict(cls, maps_dict: Mapping[str, MapArray]) -> "CoverageMaps":
        """Create coverage maps from a mapping.

        Parameters
        ----------
        maps_dict : Mapping[str, MapArray]
            Mapping containing every name returned by :meth:`map_names`.

        Returns
        -------
        CoverageMaps
            Reconstructed coverage-map collection.
        """
        return cls(
            pl_db=maps_dict["pl_db"],
            cg_db=maps_dict["cg_db"],
            rsrp_dbm=maps_dict["rsrp_dbm"],
            sinr_db=maps_dict["sinr_db"],
            los=maps_dict["los"],
            sf_db=maps_dict["sf_db"],
        )


class MapGenerator:
    """Large-scale fading map generator for the network environment."""

    CACHE_DIR: str = ".cache"
    USE_CACHED_MAPS: bool = True
    CACHE_SCHEMA: str = "v3_seeded_realization"

    def __init__(
        self,
        lsf: "LSFModel",
        geo: "SimulationGeometry",
        *,
        map_dtype: str | np.dtype = "float32",
    ) -> None:
        self.lsf = lsf
        self.geo = geo
        self.map_dtype = np.dtype(map_dtype)

    def create_maps(self) -> CoverageMaps:
        """Create or load large-scale-fading maps for the environment.

        Returns
        -------
        CoverageMaps
            Generated coverage maps, or compatible maps loaded from cache.
        """
        if self.USE_CACHED_MAPS:
            maps = self._load_cached_maps()
            if maps is not None:
                shape = tuple(int(v) for v in maps.pl_db.shape)
                print(
                    "Loaded cached coverage-map realization "
                    f"(radio_map_seed={self.lsf.seed}, "
                    f"maps={CoverageMaps.map_names()}, shape={shape}, "
                    f"dtype={self.map_dtype.name}) from "
                    f"{self._maps_cache_dir} [key={self._cache_base_name}]."
                )
                return maps
            print("No valid cached maps found. Creating new maps...")
        else:
            print("Creating maps without caching...")

        grid = Grid(*self.geo.roi.bb_size, self.lsf.config.lsf_maps_grid_res_m, True)
        est_mem_usage_gb = self._estimate_map_memory_usage_gb(grid)
        if est_mem_usage_gb > 2.0:
            warnings.warn(
                f"Estimated memory usage for map creation is high: "
                f"{est_mem_usage_gb:.2f} GB. "
                f"Consider increasing map_grid_res_m or disabling map creation."
            )

        if self.USE_CACHED_MAPS:
            return self._create_cached_maps(grid)
        return self._create_uncached_maps(grid)

    def _estimate_map_memory_usage_gb(self, grid: Grid) -> float:
        """Estimate the memory usage for map creation in GB."""
        n_grid_points = int(np.prod(grid.shape))
        n_maps = len(CoverageMaps.map_names())
        bytes_per_float = self.map_dtype.itemsize

        total_bytes = n_grid_points * self.lsf.bs_set.n_cells * n_maps * bytes_per_float
        return total_bytes / (1024**3)

    def _create_maps(self, grid: Grid) -> CoverageMaps:
        """Create coverage maps in memory."""
        flat_xy, flat_idxs = Grid.flatten(grid.xy, grid.idxs)
        roi_pos_2d, roi_idxs = self.geo.roi.get_positions_and_idxs_in_roi(
            flat_xy, flat_idxs
        )

        heights = np.full(
            (roi_pos_2d.shape[0], 1),
            1.5,  # Typical UE height in meters
            dtype=self.lsf.internal_dtype,
        )
        roi_pos_3d = np.hstack(
            [roi_pos_2d.astype(self.lsf.internal_dtype), heights]
        ).astype(
            self.lsf.internal_dtype,
            copy=False,
        )

        d_2d, d_3d, phi_az_rad, theta_el_rad = self.geo.get_distances_and_angles(
            roi_pos_2d,
            ue_heights=heights,
            bs_heights=self.lsf.bs_set.bs_heights,
        )
        candidate_kwargs = {}
        if self.lsf.config.wraparound_coupling_method == "max_coupling_over_images":
            cand_d_2d, cand_d_3d, cand_phi_az_rad, cand_theta_el_rad = (
                self.geo.get_candidate_distances_and_angles(
                    roi_pos_2d,
                    ue_heights=heights,
                    bs_heights=self.lsf.bs_set.bs_heights,
                )
            )
            candidate_kwargs = {
                "candidate_d_2d": cand_d_2d,
                "candidate_d_3d": cand_d_3d,
                "candidate_phi_az_deg": np.rad2deg(cand_phi_az_rad),
                "candidate_theta_el_deg": np.rad2deg(cand_theta_el_rad),
            }

        # Coverage maps are diagnostic artefacts. Creating one must not move
        # the live episode's hard-LOS/channel RNGs forward; otherwise a cold
        # cache and a warm cache generate different subsequent trajectories.
        with self.lsf.preserve_random_state():
            signal_metrics = self.lsf.calculate_lsf_metrics(
                pos_3d=roi_pos_3d,
                d_2d=d_2d,
                d_3d=d_3d,
                phi_az_deg=np.rad2deg(phi_az_rad),
                theta_el_deg=np.rad2deg(theta_el_rad),
                **candidate_kwargs,
            )

        roi_x_idxs, roi_y_idxs = roi_idxs.T
        maps_shape = (self.lsf.bs_set.n_cells, *grid.shape)

        def build_map(values: FloatArr) -> FloatArr:
            """Create a full map with NaN outside the ROI."""
            out = np.full(maps_shape, np.nan, dtype=self.map_dtype)
            out[:, roi_y_idxs, roi_x_idxs] = np.asarray(
                values,
                dtype=self.map_dtype,
            ).T
            return out

        coverage_maps = CoverageMaps(
            pl_db=build_map(signal_metrics.pl_db),
            cg_db=build_map(signal_metrics.cg_db),
            rsrp_dbm=build_map(signal_metrics.rsrp_dbm),
            sinr_db=build_map(signal_metrics.sinr_db),
            los=build_map(signal_metrics.los),
            sf_db=build_map(signal_metrics.sf_db),
        )
        return coverage_maps

    def _create_cached_maps(self, grid: Grid) -> CoverageMaps:
        """Create maps and store them as memmap-backed arrays."""
        coverage_maps = self._create_maps(grid)
        maps_shape = coverage_maps.pl_db.shape
        dtype = self.map_dtype

        memmaps: dict[str, np.memmap] = {}
        for name, raw_map in coverage_maps.as_dict().items():
            map_path = self._map_file_path(name)
            fd, temporary_map_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(map_path)}.",
                suffix=".tmp",
                dir=self._maps_cache_dir,
            )
            os.close(fd)
            try:
                mm = np.memmap(
                    temporary_map_path,
                    dtype=dtype,
                    mode="w+",
                    shape=maps_shape,
                )
                mm[:] = np.asarray(raw_map, dtype=dtype)
                mm.flush()
                del mm
                os.replace(temporary_map_path, map_path)
            finally:
                if os.path.exists(temporary_map_path):
                    os.remove(temporary_map_path)
            memmaps[name] = np.memmap(
                map_path,
                dtype=dtype,
                mode="r",
                shape=maps_shape,
            )

        meta = {
            "cache_schema": self.CACHE_SCHEMA,
            "base_name": self._cache_base_name,
            "identity": self._cache_identity,
            "shape": tuple(int(v) for v in maps_shape),
            "dtype": dtype.str,
            "names": CoverageMaps.map_names(),
            "map_dtype": self.map_dtype.name,
            "internal_dtype": self.lsf.internal_dtype.name,
            "geo_config_fingerprint": self.geo.config.fingerprint,
            "lsf_config_fingerprint": self.lsf.config.fingerprint,
            "ran_config_fingerprint": self.lsf.bs_set.ran_config.fingerprint,
            "roi_fingerprint": self.geo.roi.fingerprint,
        }

        meta_path = os.path.join(
            self._maps_cache_dir, f"meta_{self._cache_base_name}.pkl"
        )
        fd, temporary_meta_path = tempfile.mkstemp(
            prefix=f"{os.path.basename(meta_path)}.",
            suffix=".tmp",
            dir=self._maps_cache_dir,
        )
        os.close(fd)
        try:
            with open(temporary_meta_path, "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_meta_path, meta_path)
        finally:
            if os.path.exists(temporary_meta_path):
                os.remove(temporary_meta_path)

        print(
            "Saved coverage-map realization to cache "
            f"(radio_map_seed={self.lsf.seed}, "
            f"maps={CoverageMaps.map_names()}, shape={maps_shape}, "
            f"dtype={self.map_dtype.name}): {self._maps_cache_dir} "
            f"[key={self._cache_base_name}]."
        )
        return CoverageMaps.from_dict(memmaps)

    def _create_uncached_maps(self, grid: Grid) -> CoverageMaps:
        """Create maps without caching."""
        return self._create_maps(grid)

    def _load_cached_maps(self) -> CoverageMaps | None:
        """Load cached maps as read-only memmap objects."""
        meta_path = os.path.join(
            self._maps_cache_dir, f"meta_{self._cache_base_name}.pkl"
        )
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return None

        if not isinstance(meta, dict):
            print(f"Invalid meta format in cache: expected dict, got {type(meta)}")
            return None
        expected_keys = {
            "cache_schema",
            "base_name",
            "identity",
            "shape",
            "dtype",
            "names",
            "map_dtype",
            "internal_dtype",
            "geo_config_fingerprint",
            "lsf_config_fingerprint",
            "ran_config_fingerprint",
            "roi_fingerprint",
        }
        if set(meta) != expected_keys:
            print(
                "Cached coverage-map metadata schema mismatch: "
                f"expected {sorted(expected_keys)}, got {sorted(meta)}"
            )
            return None
        if meta["cache_schema"] != self.CACHE_SCHEMA:
            return None
        if meta["base_name"] != self._cache_base_name:
            return None
        if meta["identity"] != self._cache_identity:
            print("Cached coverage-map identity does not match this realization.")
            return None
        diff_values = {
            "geo_config_fingerprint": meta["geo_config_fingerprint"]
            != self.geo.config.fingerprint,
            "lsf_config_fingerprint": meta["lsf_config_fingerprint"]
            != self.lsf.config.fingerprint,
            "ran_config_fingerprint": meta["ran_config_fingerprint"]
            != self.lsf.bs_set.ran_config.fingerprint,
            "roi_fingerprint": meta["roi_fingerprint"] != self.geo.roi.fingerprint,
            "map_dtype": meta.get("map_dtype") != self.map_dtype.name,
            "internal_dtype": meta.get("internal_dtype")
            != self.lsf.internal_dtype.name,
        }
        if any(diff_values.values()):
            print(
                "Cached maps config mismatch:",
                {k: v for k, v in diff_values.items() if v},
            )
            return None

        maps_shape = tuple(meta["shape"])
        dtype = np.dtype(meta.get("dtype", self.map_dtype))
        names = meta.get("names", CoverageMaps.map_names())

        if names != CoverageMaps.map_names():
            print(
                f"Cached maps have unexpected names: {names}. "
                f"Expected: {CoverageMaps.map_names()}."
            )
            return None

        maps_dict: dict[str, np.memmap] = {}
        expected_nbytes = int(np.prod(maps_shape, dtype=np.int64)) * dtype.itemsize
        for name in CoverageMaps.map_names():
            path = self._map_file_path(name)
            if not os.path.exists(path):
                print(f"Cached map file not found: {path}")
                return None
            actual_nbytes = os.path.getsize(path)
            if actual_nbytes != expected_nbytes:
                print(
                    f"Cached map file has wrong size: {path} "
                    f"({actual_nbytes} != {expected_nbytes} bytes)."
                )
                return None
            try:
                maps_dict[name] = np.memmap(
                    path,
                    dtype=dtype,
                    mode="r",
                    shape=maps_shape,
                )
            except (OSError, ValueError) as error:
                print(f"Failed to open cached map file {path}: {error}")
                return None

        return CoverageMaps.from_dict(maps_dict)

    def _map_file_path(self, name: str) -> str:
        """Return the memmap file path for one map."""
        return os.path.join(
            self._maps_cache_dir,
            f"maps_{name}_{self._cache_base_name}.dat",
        )

    @property
    def _maps_cache_dir(self) -> str:
        base_dir = (
            cache_root()
            if self.CACHE_DIR == ".cache"
            else cache_root() / self.CACHE_DIR
        )
        cache_dir = os.path.join(str(base_dir), "maps")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @property
    def _cache_base_name(self) -> str:
        payload = json.dumps(
            self._cache_identity, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.CACHE_SCHEMA}_seed{self.lsf.seed}_{digest}"

    @staticmethod
    def _array_fingerprint(values: np.ndarray) -> str:
        """Hash an array including shape and dtype, not only its raw bytes."""
        array = np.ascontiguousarray(values)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def _cache_identity(self) -> dict[str, object]:
        """Exact inputs that determine one realized coverage-map collection.

        Config seeds are deliberately excluded from ``BaseConfig.fingerprint``;
        this identity therefore records the active radio-map seed explicitly.
        Actual BS positions are hashed as well, so two realizations of a random
        HetNet layout cannot alias even when their serialized layout parameters
        are identical.
        """
        return {
            "cache_schema": self.CACHE_SCHEMA,
            "radio_map_seed": int(self.lsf.seed),
            "lsf_config": self.lsf.config.to_dict(),
            "geometry_config": self.geo.config.to_dict(),
            "ran_config": self.lsf.bs_set.ran_config.to_dict(),
            "roi_fingerprint": self.geo.roi.fingerprint,
            "bs_positions_sha256": self._array_fingerprint(self.geo.layout.bs_xy),
            "wraparound_shifts_sha256": self._array_fingerprint(
                self.geo.layout.wraparound_shifts
            ),
            "ue_noise_dbm": np.asarray(self.lsf.p_noise_ue_dbm).tolist(),
            "map_dtype": self.map_dtype.name,
            "internal_dtype": self.lsf.internal_dtype.name,
            "map_names": CoverageMaps.map_names(),
        }

"""Read-only deterministic arrays, retained only while a simulator uses them."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import warnings
from copy import copy
from pathlib import Path
from threading import RLock
from weakref import WeakValueDictionary

import numpy as np


class ArrayResource:
    def __init__(self, arrays: dict[str, np.ndarray], metadata=None):
        self.arrays = arrays
        self.metadata = metadata
        for array in arrays.values():
            array.flags.writeable = False

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


_LIVE: WeakValueDictionary = WeakValueDictionary()
_LOCK = RLock()


def shared_arrays(key, build) -> ArrayResource:
    """Serialize construction in a process without retaining unused resources."""
    with _LOCK:
        resource = _LIVE.get(key)
        if resource is None:
            resource = build()
            _LIVE[key] = resource
        return resource


GRF_ARRAYS = ("coordinates_in_roi", "idxs_in_roi", "distance_mat", "cholesky_mat")


def grf_resource(grf) -> ArrayResource:
    """Open immutable NPY arrays, migrating a valid legacy cache if available."""
    identity = dict(grf._matrix_cache_identity, storage_schema="npy_v1")
    key = json.dumps(identity, sort_keys=True)
    directory = (
        Path(grf._cache_dir) / (grf._cache_filename + ".npy-v1")
        if grf.USE_CACHED_MATRICES and grf.CACHE_DIR is not None
        else None
    )

    def private():
        try:
            loaded = directory is not None and grf._load_cached_matrices(
                remove_invalid=False
            )
        except OSError:
            loaded = False
        if not loaded:
            grf._compute_matrices()
        arrays = {name: getattr(grf, name) for name in GRF_ARRAYS}
        grid = grf.grid
        for name in ("xy", "idxs"):
            arrays["grid_" + name] = getattr(grid, name)
        # Grid edge arrays are small, but are immutable too.
        for array in grid.xy_bins:
            array.flags.writeable = False
        return ArrayResource(arrays, grid)

    def read():
        assert directory is not None
        with (directory / "metadata.pkl").open("rb") as stream:
            metadata = pickle.load(stream)
        if metadata["identity"] != identity:
            raise ValueError("Shared GRF identity mismatch")
        arrays = {
            name: np.load(
                directory / (name + ".npy"), mmap_mode="r", allow_pickle=False
            )
            for name in (*GRF_ARRAYS, "grid_xy", "grid_idxs")
        }
        for name, array in arrays.items():
            if (array.shape, array.dtype.str) != metadata["arrays"][name]:
                raise ValueError("Shared GRF array metadata mismatch")
        grid = metadata["grid"]
        grid.xy, grid.idxs = arrays["grid_xy"], arrays["grid_idxs"]
        return ArrayResource(arrays, grid)

    def build():
        if directory is None:
            grf.resource_status = "memory_only"
            return private()
        try:
            if directory.exists():
                resource = read()
            else:
                resource = private()
                directory.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix="grf-publish-", dir=directory.parent
                ) as temporary:
                    stage = Path(temporary) / "entry"
                    stage.mkdir()
                    for name, array in resource.arrays.items():
                        np.save(stage / (name + ".npy"), array, allow_pickle=False)
                    with (stage / "metadata.pkl").open("wb") as stream:
                        pickle.dump(
                            {
                                "identity": identity,
                                "grid": resource.metadata,
                                "arrays": {
                                    n: (a.shape, a.dtype.str)
                                    for n, a in resource.arrays.items()
                                },
                            },
                            stream,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    try:
                        os.rename(stage, directory)
                    except OSError:
                        if not directory.is_dir():
                            raise
                        # Another process published the same immutable identity.
                resource = read()
            grf.resource_status = "shared_mmap"
            return resource
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, KeyError) as exc:
            warnings.warn(
                f"Shared GRF cache unavailable; using private arrays: {exc}",
                stacklevel=2,
            )
            grf.resource_status = "private_fallback"
            return private()

    resource = shared_arrays((str(directory), key), build)
    if not hasattr(grf, "resource_status"):
        grf.resource_status = (
            "shared_mmap"
            if isinstance(resource.arrays["cholesky_mat"], np.memmap)
            else "memory_only"
        )
    return resource


def attach_grf(grf, resource: ArrayResource) -> None:
    grf._shared_resource = resource
    for name in GRF_ARRAYS:
        setattr(grf, name, resource.arrays[name])
    grf.grid = copy(resource.metadata)
    grf.grid.__dict__.pop("_corner_cache", None)

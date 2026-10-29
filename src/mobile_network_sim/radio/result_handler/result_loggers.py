"""Module for logging simulation results to HDF5 files."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from ..simulation.simulation_env import RadioEnvironment, RadioStepResult


def _sanitize_for_text_serialization(x: Any) -> Any:
    """Recursively sanitize an object for JSON serialization."""
    if isinstance(x, type):
        return f"{x.__module__}.{x.__qualname__}"
    if hasattr(x, "model_dump") and callable(x.model_dump):
        return _sanitize_for_text_serialization(
            x.model_dump(mode="python", serialize_as_any=True)
        )
    if hasattr(x, "dict") and callable(x.dict):
        return _sanitize_for_text_serialization(x.dict())
    if is_dataclass(x):
        return _sanitize_for_text_serialization(asdict(x))  # type: ignore
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _sanitize_for_text_serialization(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_text_serialization(v) for v in x]
    if isinstance(x, set):
        return [_sanitize_for_text_serialization(v) for v in sorted(x, key=str)]
    if hasattr(x, "__dict__"):
        return _sanitize_for_text_serialization(vars(x))
    return x


class BaseResultsLogger(ABC):
    """Abstract base class for logging environment results."""

    @abstractmethod
    def new_episode(self) -> None:
        """Call this at the start of each new episode."""

    @abstractmethod
    def log_step(self, step_result: "RadioStepResult") -> None:
        """Save one simulation-step result.

        Parameters
        ----------
        step_result : RadioStepResult
            Result produced by one radio-environment step.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush and close any resources."""


class HDF5ResultsLogger(BaseResultsLogger):
    """Log environment results to a chunked HDF5 file.

    Parameters
    ----------
    env : RadioEnvironment
        Environment that defines the configuration and result schema.
    log_path : str
        Destination HDF5 file path.
    n_ep : int
        Number of episode groups to allocate.
    n_steps : int
        Number of steps allocated in each episode group.
    chunk_size : int, optional
        Maximum chunk length along the step dimension.
    storage_dtype : str or numpy.dtype, optional
        Dtype used to persist floating-point result arrays.

    Attributes
    ----------
    ep_idx : int
        Index of the active episode, starting at ``-1`` before the first call
        to :meth:`new_episode`.
    step_idx : int
        Next step position in the active episode.
    """

    def __init__(
        self,
        env: "RadioEnvironment",
        log_path: str,
        n_ep: int,
        n_steps: int,
        chunk_size: int = 1_000,
        storage_dtype: str | np.dtype = "float32",
        buffer_size: int | None = None,
    ) -> None:
        self.env = env
        self.log_path = log_path
        self.n_ep = n_ep
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.storage_dtype = np.dtype(storage_dtype)
        self.buffer_size = min(
            chunk_size if buffer_size is None else buffer_size, n_steps
        )
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self._pending = 0
        self._buffers: dict[str, np.ndarray] = {}
        self._datasets: dict[str, Any] = {}
        template = env.get_dummy_step_result().to_dict()
        sample_bytes = sum(
            arr.size * self._dataset_dtype(arr).itemsize for arr in template.values()
        )
        # Bound logger retention for large multi-UE simulations as well.
        self.buffer_size = min(
            self.buffer_size, max(1, (16 * 1024**2) // max(1, sample_bytes))
        )

        self.config = env.config

        self.ep_idx = -1
        self.step_idx = 0

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.h5_file = h5py.File(self.log_path, "w")

        cfg_dict = _sanitize_for_text_serialization(vars(self.config))
        meta = self.h5_file.require_group("meta")
        meta.create_dataset(
            "config",
            data=json.dumps(cfg_dict),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        # Provenance / bookkeeping
        meta.attrs["created_utc"] = datetime.now(UTC).isoformat()
        meta.attrs["n_ep"] = self.n_ep
        meta.attrs["n_steps"] = self.n_steps
        meta.attrs["storage_dtype"] = self.storage_dtype.name

        for ep in range(n_ep):
            group = self.h5_file.create_group(f"ep_{ep:04d}")
            for key, arr in template.items():
                shape = (self.n_steps,) + arr.shape
                chunks = (min(chunk_size, self.n_steps),) + arr.shape
                group.create_dataset(
                    key,
                    shape=shape,
                    dtype=self._dataset_dtype(arr),
                    chunks=chunks,
                    compression="gzip",
                )

    def new_episode(self) -> None:
        """Call this at the start of each new episode."""
        self.flush()
        self.ep_idx += 1
        self.step_idx = 0
        if self.ep_idx >= self.n_ep:
            raise ValueError("Exceeded the number of episodes in HDF5 file.")
        group = self.h5_file[f"ep_{self.ep_idx:04d}"]
        self._datasets = dict(group.items())
        self._buffers = {
            key: np.empty((self.buffer_size, *dataset.shape[1:]), dtype=dataset.dtype)
            for key, dataset in self._datasets.items()
        }

    def log_step(self, step_result: "RadioStepResult") -> None:
        """Write one result at the current episode and step indices.

        Parameters
        ----------
        step_result : RadioStepResult
            Result arrays to write.
        """
        if self.ep_idx < 0 or self.step_idx >= self.n_steps:
            raise ValueError("No active episode or exceeded its sample count")
        for key, arr in step_result.to_dict().items():
            # Assignment casts once and detaches the caller's mutable sample.
            self._buffers[key][self._pending] = arr
        self._pending += 1
        self.step_idx += 1
        if self._pending == self.buffer_size or self.step_idx == self.n_steps:
            self.flush()

    def flush(self) -> None:
        """Write a partial block in sample order without changing the schema."""
        if self._pending:
            start = self.step_idx - self._pending
            for key, buffer in self._buffers.items():
                self._datasets[key][start : self.step_idx] = buffer[: self._pending]
            self._pending = 0

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        self.flush()
        self.h5_file.close()

    def _dataset_dtype(self, arr: np.ndarray | np.ma.MaskedArray) -> np.dtype:
        """Return the persisted dtype for one result array."""
        dtype = np.asarray(arr).dtype
        if np.issubdtype(dtype, np.floating):
            return self.storage_dtype
        return dtype

    def _cast_result_array(
        self, arr: np.ndarray | np.ma.MaskedArray
    ) -> np.ndarray | np.ma.MaskedArray:
        """Cast floating result arrays to the configured storage dtype."""
        dtype = np.asarray(arr).dtype
        if np.issubdtype(dtype, np.floating):
            return arr.astype(self.storage_dtype, copy=False)
        return arr

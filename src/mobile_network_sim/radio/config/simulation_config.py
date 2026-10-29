"""Top-level radio simulation configuration."""

from typing import Any

import numpy as np
from pydantic import Field, field_validator

from .abstract_config import AbstractConfig, StrictNonNegativeInt, StrictPositiveInt
from .env_config import RadioConfig


class SimulationDTypeConfig(AbstractConfig):
    """Configure floating-point types for storage and calculations.

    Attributes
    ----------
    storage : {"float32", "float64"}
        Dtype used for persisted simulation results.
    internal : {"float32", "float64"}
        Dtype of LSF metrics; not a global geometry/beam precision switch.
    maps : {"float16", "float32", "float64"}
        Dtype used for coverage and random-field maps.
    beam : {"float32", "float64"}
        Experimental steering/codebook/factor precision (complex64/complex128).
        Global geometry and the element pattern retain float64.
    grf : {"float32", "float64"}
        Experimental GRF factor/matmul/normalization precision. Random draws
        retain their original float64 sequence before conversion.
    """

    storage: str = "float32"
    internal: str = "float64"
    maps: str = "float32"
    # Independent experimental calculation precisions; legacy defaults remain.
    beam: str = "float64"
    grf: str = "float64"

    @field_validator("storage", "internal", "beam", "grf", mode="before")
    @classmethod
    def _validate_working_dtype(cls, value: Any) -> str:
        dtype = np.dtype(value)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(
                "storage, internal, beam and grf must be float32 or float64"
            )
        return dtype.name

    @field_validator("maps", mode="before")
    @classmethod
    def _validate_map_dtype(cls, value: Any) -> str:
        dtype = np.dtype(value)
        if dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
            raise ValueError("maps must be float16, float32, or float64")
        return dtype.name


class SimulationConfig(AbstractConfig):
    """Configure a deterministic radio-only simulation.

    Attributes
    ----------
    seed : int
        Non-negative random seed.
    time_resolution_ms : {1, 10, 20, 40}
        Duration of a simulation step in milliseconds.
    num_steps : int
        Positive number of steps used by standard runners.
    dtypes : SimulationDTypeConfig
        Storage, internal, and map dtypes.
    radio : RadioConfig
        Geometry, channel, network, and UE configuration.
    """

    seed: StrictNonNegativeInt = 42
    time_resolution_ms: StrictPositiveInt = 10
    num_steps: StrictPositiveInt = 1000
    dtypes: SimulationDTypeConfig = Field(default_factory=SimulationDTypeConfig)
    radio: RadioConfig

    @field_validator("time_resolution_ms", mode="before")
    @classmethod
    def _validate_resolution(cls, value: int) -> int:
        if value not in (1, 10, 20, 40):
            raise ValueError("time_resolution_ms must be one of 1, 10, 20, or 40")
        return value

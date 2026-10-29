"""Compatibility imports for radio configuration modules."""

from ...config import (
    BaseConfig,
    StrictNonNegativeFloat,
    StrictNonNegativeInt,
    StrictPositiveFloat,
    StrictPositiveInt,
    StrictProbability,
)

AbstractConfig = BaseConfig
StrictProb = StrictProbability

__all__ = [
    "AbstractConfig",
    "StrictNonNegativeFloat",
    "StrictNonNegativeInt",
    "StrictPositiveFloat",
    "StrictPositiveInt",
    "StrictProb",
]

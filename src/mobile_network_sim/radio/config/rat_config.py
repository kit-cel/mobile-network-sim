"""Radio Access Technology Configuration Class"""

from warnings import warn

import numpy as np
from pydantic import Field, field_validator, StrictFloat

from .abstract_config import AbstractConfig, StrictPositiveFloat


class RATConfig(AbstractConfig):
    """Configuration for the radio access technology (RAT)."""

    fc_ghz: StrictPositiveFloat = Field(
        default=3.5,
        description="Carrier frequency in GHz.",
    )
    bw_mhz: StrictPositiveFloat = Field(
        default=10.0,
        description="System bandwidth in MHz.",
    )

    rsrp_clipping_enabled: bool = Field(
        default=False,
        description=(
            "Whether to clip RSRP values to a specified range. If True, RSRP "
            "values will be clipped to the range defined by `rsrp_dbm_min` and "
            "`rsrp_dbm_max`. If False, no clipping will be applied."
        ),
    )
    rsrp_dbm_min: StrictFloat = Field(
        default=-np.inf,
        description="Minimum RSRP in dBm.",
    )
    rsrp_dbm_max: StrictFloat = Field(
        default=np.inf,
        description="Maximum RSRP in dBm.",
    )

    @field_validator("fc_ghz", mode="before")
    @classmethod
    def _validate_carrier_freq(cls, value: float) -> float:
        """Validate the carrier frequency."""
        if value <= 0 or value > 50:
            raise ValueError("Carrier frequency must be in (0, 50] GHz.")
        return value

    @field_validator("bw_mhz", mode="before")
    @classmethod
    def _validate_bandwidth(cls, value: float) -> float:
        """Validate the system bandwidth."""
        if value > 100:
            warn("Bandwidth values above 1000 MHz are unusual for current RATs.")
        return value

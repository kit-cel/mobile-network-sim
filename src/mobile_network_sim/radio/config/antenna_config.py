"""Antenna Configuration Class"""

from pydantic import Field, field_validator

from .abstract_config import AbstractConfig, StrictNonNegativeFloat
from ..components.antennas.ant_array import PanelArrayAnt
from ..components.antennas.ant_omnidir import OmnidirAnt

ant_type_valid = (
    PanelArrayAnt.NAME,
    OmnidirAnt.NAME,
)


class AntennaConfig(AbstractConfig):
    """Configuration for the antenna."""

    ant_name: str = Field(
        default=OmnidirAnt.NAME,
        description=f"Type of the antenna. Valid: {ant_type_valid}.",
    )
    g_max_dbi: StrictNonNegativeFloat = Field(
        default=0.0,
        description="Single-element maximum directional gain in dBi.",
    )

    @field_validator("ant_name", mode="before")
    @classmethod
    def _validate_ant_type(cls, value: str) -> str:
        """Validate the antenna type."""
        if value not in ant_type_valid:
            raise ValueError(f"Antenna type must be one of {ant_type_valid}.")
        return value

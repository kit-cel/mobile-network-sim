"""Network Layout Configuration Class"""

from pydantic import Field, field_validator, StrictStr

from .abstract_config import (
    AbstractConfig,
    StrictPositiveFloat,
    StrictPositiveInt,
    StrictNonNegativeFloat,
    StrictNonNegativeInt,
)
from ..geometry.layout import layout_dict


class NetworkLayoutConfig(AbstractConfig):
    """Configuration for the network layout."""

    # Unlike the other nested component seeds, this seed is not replaced at
    # each episode reset: it realizes a random HetNet layout once when the
    # environment is constructed. It must therefore survive serialization and
    # participate in fingerprints.
    seed: StrictNonNegativeInt = Field(default=42, exclude=False)

    layout: StrictStr = Field(
        description=f"Type of network layout. Options: {list(layout_dict.keys())}",
    )

    isd_m: StrictPositiveFloat = Field(
        default=500.0, description="Inter-site distance in meters."
    )
    n_tiers: StrictPositiveInt = Field(
        default=1, description="Number of BS rings in a hexagonal layout."
    )
    rotation_offset_deg: StrictNonNegativeFloat = Field(
        default=0.0,
        description="Rotation offset of the layout in degrees.",
    )

    # Derived attributes and validators ----------------------------------------

    # Field validators (before) ------------------------------------------------
    @field_validator("layout", mode="before")
    @classmethod
    def _validate_layout(cls, value: str) -> str:
        """Validate the layout."""
        if value not in layout_dict:
            raise ValueError(
                f"Invalid network layout type: {value}. "
                f"Supported types: {list(layout_dict.keys())}."
            )
        return value

    @field_validator("rotation_offset_deg", mode="before")
    @classmethod
    def _validate_rotation_offset_deg(cls, value: float) -> float:
        """Validate the rotation offset."""
        if not 0 <= value < 360:
            raise ValueError("Rotation offset must be in the range [0, 360) degrees.")
        return value

    # Model validators (after) -------------------------------------------------

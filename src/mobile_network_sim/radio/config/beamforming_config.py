"""Beam codebook configuration module."""

from pydantic import Field, StrictFloat, model_validator

from .abstract_config import AbstractConfig


class BeamformingConfig(AbstractConfig):
    """Configuration for beam codebook generation."""

    az_list_deg: list[StrictFloat] = Field(
        default_factory=list,
        description="List of azimuth angles in degrees for codebook beams.",
    )
    el_list_deg: list[StrictFloat] = Field(
        default_factory=list,
        description="List of elevation angles in degrees for codebook beams.",
    )

    @model_validator(mode="after")
    def _validate_angles(self) -> "BeamformingConfig":
        for az in self.az_list_deg:
            if not -180.0 <= az < 180.0:
                raise ValueError(f"Azimuth angles must be in [-180, 180). Got {az}.")
        for el in self.el_list_deg:
            if not -90.0 <= el <= 90.0:
                raise ValueError(f"Elevation angles must be in [-90, 90]. Got {el}.")
        if len(self.az_list_deg) != len(self.el_list_deg):
            raise ValueError(
                "Azimuth and elevation lists must have the same length. "
                f"Got {len(self.az_list_deg)} azimuths and {len(self.el_list_deg)} elevations."
            )
        return self

"""Base Station Configuration Class"""

from typing import Any

from pydantic import (
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
    StrictFloat,
)

from .abstract_config import AbstractConfig, StrictNonNegativeFloat, StrictPositiveInt
from .antenna_array_config import AntennaArrayConfig
from .antenna_config import AntennaConfig
from ..components.antennas import OmnidirAnt, PanelArrayAnt

n_sectors_valid = (1, 3, 6)


class BaseStationConfig(AbstractConfig):
    """Configuration for the base station."""

    p_tx_dbm: StrictFloat = Field(
        default=43.0,
        description="Transmit power in dBm.",
    )
    rx_noise_figure_db: StrictFloat = Field(
        default=5.0,
        description="BS noise figure in dB.",
    )

    height_m: StrictNonNegativeFloat = Field(
        default=25.0,
        description="Height of the BS in meters.",
    )
    n_sectors: StrictPositiveInt = Field(
        default=1,
        description=f"Number of sectors in the BS. Values: {n_sectors_valid}.",
    )
    sector_azimuth_offset_deg: StrictNonNegativeFloat = Field(
        default=0.0,
        description="Azimuth offset of the sector in degrees.",
    )

    # ``ant`` is deliberately polymorphic: the validator below turns a
    # PanelArrayAnt mapping into AntennaArrayConfig.  Without
    # SerializeAsAny, Pydantic serializes the runtime subclass through the
    # AntennaConfig annotation and silently drops every array/beamforming
    # field.  That produced H5 provenance which claimed a 1x1, non-beamformed
    # antenna although the simulation actually used a 16x8 codebook array.
    ant: SerializeAsAny[AntennaConfig]

    @field_validator("n_sectors", mode="before")
    @classmethod
    def _validate_n_sectors(cls, value: int) -> int:
        """Validate the number of sectors."""
        if value not in n_sectors_valid:
            raise ValueError(f"Number of sectors must be one of {n_sectors_valid}.")
        return value

    @field_validator("sector_azimuth_offset_deg", mode="before")
    @classmethod
    def _validate_sector_azimuth_offset_deg(cls, value: int) -> int:
        """Validate the sector azimuth offset."""
        if value < 0 or value >= 360:
            raise ValueError("Sector azimuth offset must be in [0, 360) degrees.")
        return value

    @field_validator("ant", mode="before")
    @classmethod
    def _choose_ant_config_class(cls, data: Any) -> AntennaConfig:
        """Create the correct antenna config subclass from the raw ant input."""
        if isinstance(data, AntennaConfig):
            return data

        if not isinstance(data, dict):
            raise TypeError("Invalid antenna configuration format.")

        ant_name = data.get("ant_name")
        if ant_name is None:
            raise ValueError("Antenna name must be specified in the configuration.")

        if ant_name == OmnidirAnt.NAME:
            return AntennaConfig(**data)

        if ant_name == PanelArrayAnt.NAME:
            return AntennaArrayConfig(**data)

        raise ValueError(f"Unsupported ant_name: {ant_name}")

    @model_validator(mode="after")
    def _validate_ant_compatibility_with_sectors(self) -> "BaseStationConfig":
        """Validate that the antenna configuration is compatible with the number of sectors."""
        if self.ant.ant_name == OmnidirAnt.NAME and self.n_sectors != 1:
            raise ValueError(
                "Omnidirectional antenna is only compatible with 1 sector. "
                "Please set n_sectors to 1 or choose a different antenna type."
            )
        return self

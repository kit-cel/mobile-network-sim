"""Large-scale fading configuration."""

from typing import ClassVar, Literal

from pydantic import StrictStr, Field, field_validator, StrictBool


from .abstract_config import AbstractConfig, StrictPositiveFloat, StrictNonNegativeFloat
from ..lsf.channels import channel_models, UrbanMacro
from ..lsf.los_models import los_models, ConstLOS


class LargeScaleFadingConfig(AbstractConfig):
    """Configure channel, LOS, shadowing, and coverage-map calculations.

    Attributes
    ----------
    channel_model : str
        Registered path-loss channel model name.
    los_model : str
        Registered line-of-sight model name.
    soft_los_map_grid_spacing_m : float
        Soft-LOS random-field grid spacing in metres.
    soft_los_d_decorr_m : float
        Soft-LOS :math:`1/e` correlation distance in metres.
    shadowing_enabled : bool
        Whether shadow fading is included.
    shadowing_map_grid_spacing_m : float
        Shadow-fading map grid spacing in metres.
    shadowing_std_los_db, shadowing_std_nlos_db : float
        LOS and NLOS shadow-fading standard deviations in decibels.
    shadowing_d_decorr_m : float
        Shadow-fading :math:`1/e` correlation distance in metres.
    wraparound_coupling_method : {"nearest_image", "max_coupling_over_images"}
        Rule used to select periodic base-station images for link calculations.
    create_lsf_maps : bool
        Whether the standard runner precomputes coverage maps.
    lsf_maps_grid_res_m : float
        Coverage-map grid resolution in metres.
    """

    WRAPAROUND_COUPLING_METHODS: ClassVar[set[str]] = {
        "nearest_image",
        "max_coupling_over_images",
    }

    # General LSF model parameters
    channel_model: StrictStr = Field(
        default=UrbanMacro.NAME,
        description=f"Type of channel model to use for LSF. Options: {list(channel_models.keys())}",
    )

    # LOS/NLOS model parameters
    los_model: StrictStr = Field(
        default=ConstLOS.NAME,
        description=f"Type of LOS model to use for LSF. Options: {list(los_models.keys())}",
    )
    soft_los_map_grid_spacing_m: StrictPositiveFloat = Field(
        default=20.0,
        description="Grid spacing for soft LOS maps in meters.",
    )
    soft_los_d_decorr_m: StrictNonNegativeFloat = Field(
        default=50.0,
        description=(
            "1/e correlation distance for the soft LOS model in meters, "
            "using the 3GPP exponential-correlation convention."
        ),
    )

    ### Shadowing model parameters
    shadowing_enabled: StrictBool = Field(
        default=True, description="Enable shadowing effects in the simulation."
    )
    shadowing_map_grid_spacing_m: StrictPositiveFloat = Field(
        default=20.0,
        description="Grid spacing for shadowing maps in meters.",
    )
    shadowing_std_los_db: StrictNonNegativeFloat = Field(
        default=4.0, description="LOS shadowing standard deviation in dB."
    )
    shadowing_std_nlos_db: StrictNonNegativeFloat = Field(
        default=6.0, description="NLOS shadowing standard deviation in dB."
    )
    shadowing_d_decorr_m: StrictNonNegativeFloat = Field(
        default=20.0,
        description=(
            "1/e correlation distance for shadowing in meters, using the "
            "3GPP exponential-correlation convention."
        ),
    )
    wraparound_coupling_method: Literal[
        "nearest_image",
        "max_coupling_over_images",
    ] = Field(
        default="nearest_image",
        description=(
            "Method used to handle periodic BS images in coupling calculations. "
            "'nearest_image' keeps the current minimum-distance image behavior; "
            "'max_coupling_over_images' evaluates all wrap-around images and "
            "selects the image with maximum coupling per UE-cell link."
        ),
    )

    # LSF map parameters
    create_lsf_maps: StrictBool = Field(
        default=True, description="Flag to create LSF maps for the simulation."
    )
    lsf_maps_grid_res_m: StrictPositiveFloat = Field(
        default=10.0, description="Grid resolution for LSF maps in meters."
    )

    # Field validators (before) ------------------------------------------------
    @field_validator("channel_model", mode="before")
    @classmethod
    def _validate_channel_model(cls, channel_model: str) -> str:
        """Validate the channel model."""
        if channel_model not in channel_models:
            raise ValueError(
                f"Invalid channel model type {channel_model}. "
                f"Supported types: {list(channel_models.keys())}"
            )
        return channel_model

    @field_validator("los_model", mode="before")
    @classmethod
    def _validate_los_model(cls, los_model: str) -> str:
        """Validate the LOS model."""
        if los_model not in los_models:
            raise ValueError(
                f"Invalid LOS model type {los_model}. "
                f"Supported types: {list(los_models.keys())}"
            )
        return los_model

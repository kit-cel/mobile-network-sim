"""User-equipment configuration."""

from pydantic import Field, StrictFloat, StrictStr, field_validator, model_validator

from .abstract_config import AbstractConfig, StrictNonNegativeFloat, StrictProb
from ..components.ue.mobility import (
    mobility_models,
    ConstPosition,
    ConstSpeedCorrRandomWalk,
    RandomSpeedRandomWalk,
    RandomWaypoint,
)


class UserEquipmentConfig(AbstractConfig):
    """Configure UE radio properties and mobility behavior.

    Attributes
    ----------
    height_m : float
        UE antenna height in metres.
    mobility_model : str
        Registered mobility model name.
    speed_kph : float
        Constant speed in kilometres per hour for models that use it.
    max_turn_angle_deg : float
        Maximum correlated-random-walk heading change per update in degrees.
    random_speed_lim_kph : tuple of float
        Minimum and maximum random-walk speeds in kilometres per hour.
    random_speed_corr : float
        Random-speed correlation in the interval ``[0, 1]``.
    gauss_markov_alpha : float
        Gauss-Markov memory coefficient in the interval ``[0, 1]``.
    gauss_markov_mean_speed_kph : float
        Gauss-Markov mean speed in kilometres per hour.
    gauss_markov_speed_std_kph : float
        Gauss-Markov speed standard deviation in kilometres per hour.
    gauss_markov_angle_std_deg : float
        Gauss-Markov heading standard deviation in degrees.
    gauss_markov_min_speed_kph, gauss_markov_max_speed_kph : float or None
        Lower and optional upper speed bounds for Gauss-Markov mobility.
    waypoint_speed_lim_kph : tuple of float
        Random-waypoint minimum and maximum speeds in kilometres per hour.
    waypoint_pause_time_lim_s : tuple of float
        Random-waypoint minimum and maximum pause durations in seconds.
    waypoint_tolerance_m : float
        Distance in metres at which a waypoint is considered reached.
    power_class_dbm : float
        UE transmit-power class in dBm.
    rx_noise_figure_db : float
        UE receiver noise figure in decibels.
    """

    height_m: StrictNonNegativeFloat = Field(
        default=1.5,
        description="Height of the UE in meters.",
    )

    mobility_model: StrictStr = Field(
        default=ConstPosition.NAME,
        description=(
            "Mobility model of the UE. " f"Supported models are: {mobility_models}"
        ),
    )

    # Common speed parameter, used by constant-speed mobility models.
    speed_kph: StrictNonNegativeFloat = Field(
        default=0.0,
        description="Constant UE speed in km/h.",
    )

    # Correlated Randowm Walk (CRW) parameter.
    max_turn_angle_deg: StrictNonNegativeFloat = Field(
        default=1.0,
        description=(
            "Maximum CRW turning angle in degrees per mobility update. "
            "Only used by ConstSpeedCorrRandomWalk."
        ),
    )

    # Random-speed random walk parameters.
    random_speed_lim_kph: tuple[StrictNonNegativeFloat, StrictNonNegativeFloat] = Field(
        default=(0.0, 120.0),
        description="Speed range for RandomSpeedRandomWalk mobility in km/h.",
    )
    random_speed_corr: StrictProb = Field(
        default=0.0,
        description="Speed correlation for RandomSpeedRandomWalk mobility.",
    )

    # Gauss-Markov parameters.
    gauss_markov_alpha: StrictProb = Field(
        default=0.95,
        description=(
            "Gauss-Markov memory parameter alpha in [0, 1]. "
            "alpha=0 gives memoryless updates; alpha=1 approaches constant motion."
        ),
    )
    gauss_markov_mean_speed_kph: StrictNonNegativeFloat = Field(
        default=30.0,
        description="Mean speed for the Gauss-Markov mobility model in km/h.",
    )
    gauss_markov_speed_std_kph: StrictNonNegativeFloat = Field(
        default=1.0,
        description="Speed standard deviation for the Gauss-Markov model in km/h.",
    )
    gauss_markov_angle_std_deg: StrictNonNegativeFloat = Field(
        default=1.0,
        description="Heading-angle standard deviation for the Gauss-Markov model in degrees.",
    )
    gauss_markov_min_speed_kph: StrictNonNegativeFloat = Field(
        default=0.0,
        description="Minimum allowed Gauss-Markov speed in km/h.",
    )
    gauss_markov_max_speed_kph: StrictNonNegativeFloat | None = Field(
        default=None,
        description="Maximum allowed Gauss-Markov speed in km/h.",
    )

    # Random Waypoint parameters.
    waypoint_speed_lim_kph: tuple[
        StrictNonNegativeFloat,
        StrictNonNegativeFloat,
    ] = Field(
        default=(3.0, 30.0),
        description="Speed range for Random Waypoint mobility in km/h.",
    )
    waypoint_pause_time_lim_s: tuple[
        StrictNonNegativeFloat,
        StrictNonNegativeFloat,
    ] = Field(
        default=(0.0, 0.0),
        description="Pause-time range for Random Waypoint mobility in seconds.",
    )
    waypoint_tolerance_m: StrictNonNegativeFloat = Field(
        default=1e-6,
        description=(
            "Distance threshold below which a Random Waypoint target is considered reached."
        ),
    )

    power_class_dbm: StrictFloat = Field(
        default=23.0,
        description="UE power class in dBm.",
    )
    rx_noise_figure_db: StrictFloat = Field(
        default=7.0,
        description="UE noise figure in dB.",
    )

    @field_validator("mobility_model")
    @classmethod
    def _validate_mobility_model(cls, value: str) -> str:
        if value not in mobility_models:
            raise ValueError(
                f"Invalid mobility model '{value}'. "
                f"Supported models are: {mobility_models}"
            )
        return value

    @model_validator(mode="after")
    def _validate_speed_and_model(self) -> "UserEquipmentConfig":
        if self.mobility_model == ConstPosition.NAME and self.speed_kph != 0.0:
            raise ValueError("For ConstPosition mobility model, speed_kph must be 0.0.")

        if (
            self.mobility_model == ConstSpeedCorrRandomWalk.NAME
            and self.max_turn_angle_deg > 180.0
        ):
            raise ValueError(
                "For ConstSpeedCorrRandomWalk, max_turn_angle_deg must be <= 180."
            )

        if self.mobility_model == RandomSpeedRandomWalk.NAME:
            vmin, vmax = self.random_speed_lim_kph
            if vmax < vmin:
                raise ValueError("random_speed_lim_kph must satisfy min <= max.")

        if (
            self.gauss_markov_max_speed_kph is not None
            and self.gauss_markov_max_speed_kph < self.gauss_markov_min_speed_kph
        ):
            raise ValueError(
                "gauss_markov_max_speed_kph must be >= gauss_markov_min_speed_kph."
            )

        vmin, vmax = self.waypoint_speed_lim_kph
        if vmax < vmin:
            raise ValueError("waypoint_speed_lim_kph must satisfy min <= max.")
        if vmax == 0.0 and self.mobility_model == RandomWaypoint.NAME:
            raise ValueError(
                "For RandomWaypoint, the maximum waypoint speed must be positive."
            )

        pmin, pmax = self.waypoint_pause_time_lim_s
        if pmax < pmin:
            raise ValueError("waypoint_pause_time_lim_s must satisfy min <= max.")

        return self

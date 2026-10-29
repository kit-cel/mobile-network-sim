"""Radio environment configuration."""

from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from .abstract_config import AbstractConfig, StrictPositiveInt
from .geometry_config import GeometryConfig
from .lsf_config import LargeScaleFadingConfig
from .ran_config import RadioAccessNetworkConfig
from .ue_config import UserEquipmentConfig


class RadioConfig(AbstractConfig):
    """Configure geometry, propagation, network, and UEs.

    Attributes
    ----------
    geometry : GeometryConfig
        Layout, region, and wraparound configuration.
    channel : LargeScaleFadingConfig
        Channel, LOS, shadowing, and coverage-map configuration.
    num_ues : int
        Positive number of simulated UEs.
    ues : UserEquipmentConfig or list of UserEquipmentConfig
        One shared UE configuration or one configuration per UE. A list must
        contain either one item or ``num_ues`` items.
    network : RadioAccessNetworkConfig
        Base-station, antenna, and radio-access-technology configuration.
    """

    geometry: GeometryConfig
    # Persist either mode so explicit legacy controls survive config round trips.
    resource_storage: Literal["legacy", "shared_mmap"] = "shared_mmap"
    channel: LargeScaleFadingConfig
    num_ues: StrictPositiveInt = 1
    ues: UserEquipmentConfig | list[UserEquipmentConfig] = Field(
        default_factory=UserEquipmentConfig
    )
    network: RadioAccessNetworkConfig

    @field_validator("ues", mode="before")
    @classmethod
    def _normalize_ue_config(
        cls,
        value: (
            UserEquipmentConfig
            | dict[str, Any]
            | list[UserEquipmentConfig]
            | list[dict[str, Any]]
        ),
    ) -> UserEquipmentConfig | list[UserEquipmentConfig]:
        if isinstance(value, UserEquipmentConfig):
            return value
        if isinstance(value, dict):
            return UserEquipmentConfig(**value)
        if isinstance(value, list) and all(
            isinstance(item, UserEquipmentConfig) for item in value
        ):
            return cast(list[UserEquipmentConfig], value)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return [UserEquipmentConfig(**cast(dict[str, Any], item)) for item in value]
        raise ValueError(
            "ues must be one UE configuration or a list of UE configurations"
        )

    @model_validator(mode="after")
    def _validate_ue_count(self) -> "RadioConfig":
        if isinstance(self.ues, list) and len(self.ues) not in (1, self.num_ues):
            raise ValueError(
                f"Number of UE configs ({len(self.ues)}) must be 1 or num_ues ({self.num_ues})"
            )
        return self

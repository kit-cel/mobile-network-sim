"""Radio Access Network (RAN) configuration module."""

from typing import Any, cast
from pydantic import Field, field_validator

from .abstract_config import AbstractConfig
from .bs_config import BaseStationConfig
from .rat_config import RATConfig


class RadioAccessNetworkConfig(AbstractConfig):
    """Configuration for the radio access network (RAN)."""

    macro_bs: BaseStationConfig | list[BaseStationConfig] = Field(
        default_factory=list,
        description="List of macro BS configurations.",
    )
    micro_bs: BaseStationConfig | list[BaseStationConfig] = Field(
        default_factory=list,
        description="List of micro BS configurations.",
    )

    rat: RATConfig = Field(
        description="Configuration for the radio access technology (RAT).",
    )

    @field_validator("macro_bs", "micro_bs", mode="before")
    @classmethod
    def _normalize_bs_config(
        cls,
        bs_config: (
            BaseStationConfig
            | dict[str, Any]
            | list[BaseStationConfig]
            | list[dict[str, Any]]
        ),
    ) -> BaseStationConfig | list[BaseStationConfig]:
        """Ensure that the BS configuration is a list."""
        if isinstance(bs_config, BaseStationConfig):
            return bs_config
        if isinstance(bs_config, dict):
            return [BaseStationConfig(**bs_config)]
        if isinstance(bs_config, list):
            if all(isinstance(bs, BaseStationConfig) for bs in bs_config):
                return cast(list[BaseStationConfig], bs_config)
            if all(isinstance(bs, dict) for bs in bs_config):
                return [
                    BaseStationConfig(**cast(dict[str, Any], bs)) for bs in bs_config
                ]
        raise ValueError("Invalid BS configuration format.")

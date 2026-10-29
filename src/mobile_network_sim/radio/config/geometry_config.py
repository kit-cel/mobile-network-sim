"""Network geometry configuration."""

from pydantic import Field, StrictBool

from .abstract_config import AbstractConfig
from .layout_config import NetworkLayoutConfig
from .roi_config import ROIConfig


class GeometryConfig(AbstractConfig):
    """Configure the network layout and simulation region.

    Attributes
    ----------
    mobility_wraparound : bool
        Apply periodic boundaries to UE mobility when enabled.
    network_wraparound : bool
        Include periodic network images in topology calculations when enabled.
    layout : NetworkLayoutConfig
        Base-station layout configuration.
    region : ROIConfig
        Region-of-interest shape and dimensions.
    """

    mobility_wraparound: StrictBool = Field(
        default=False, description="Enable wraparound effect for mobility."
    )
    network_wraparound: StrictBool = Field(
        default=False, description="Enable wraparound effect for the network topology."
    )

    layout: NetworkLayoutConfig = Field(
        description="Configuration for the network layout.",
    )
    region: ROIConfig = Field(
        description="Configuration for the region of interest.",
    )

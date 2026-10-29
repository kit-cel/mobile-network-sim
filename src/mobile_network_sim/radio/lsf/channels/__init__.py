"""Channel model registry for implemented 5G scenarios."""

from .abstract_channel_model import AbstractChannelModel
from .fiveg_rma import RuralMacro
from .fiveg_uma import UrbanMacro

channel_models: dict[str, type[AbstractChannelModel]] = {
    RuralMacro.NAME: RuralMacro,
    UrbanMacro.NAME: UrbanMacro,
}

__all__ = ["RuralMacro", "UrbanMacro"]

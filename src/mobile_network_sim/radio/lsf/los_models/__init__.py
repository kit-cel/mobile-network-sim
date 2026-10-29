"""LOS models for large-scale fading."""

from .abstract_los import AbstractLOSModel
from .const_los_nlos import ConstLOS, ConstNLOS
from .rma_los import RMaLOSModel
from .uma_los import UMaLOSModel
from .uma_soft_los import UMaLOSModelSoft

los_models: dict[str, type[AbstractLOSModel]] = {
    ConstLOS.NAME: ConstLOS,
    ConstNLOS.NAME: ConstNLOS,
    RMaLOSModel.NAME: RMaLOSModel,
    UMaLOSModel.NAME: UMaLOSModel,
    UMaLOSModelSoft.NAME: UMaLOSModelSoft,
}

__all__ = [
    # "AbstractLOSModel",
    "ConstLOS",
    "ConstNLOS",
    "RMaLOSModel",
    "UMaLOSModel",
    "UMaLOSModelSoft",
]

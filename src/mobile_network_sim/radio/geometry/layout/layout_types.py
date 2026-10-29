"""Layout types."""

from .abstract_layout import AbstractNetworkLayout

from .hetnet_random_layout import HetNetRandomLayout
from .hetnet_regular_layout import HetNetRegularLayout
from .hexagonal_layout import HexagonalMacroLayout

layout_dict: dict[str, type[AbstractNetworkLayout]] = {
    HetNetRandomLayout.__name__: HetNetRandomLayout,
    HetNetRegularLayout.__name__: HetNetRegularLayout,
    HexagonalMacroLayout.__name__: HexagonalMacroLayout,
}

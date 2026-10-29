"""ROI types."""

from .abstract_roi import AbstractROI
from .hexagon_roi import HexagonROI
from .multi_hexagon_roi import MultiHexagonROI
from .rectangle_roi import RectangularROI

roi_dict: dict[str, "type[AbstractROI]"] = {
    HexagonROI.NAME: HexagonROI,
    MultiHexagonROI.NAME: MultiHexagonROI,
    RectangularROI.NAME: RectangularROI,
}

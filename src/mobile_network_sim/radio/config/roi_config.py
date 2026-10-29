"""Network Layout Configuration Class"""

from typing import TYPE_CHECKING

from pydantic import Field, field_validator, StrictStr

from .abstract_config import AbstractConfig
from ..geometry.roi.hexagon_roi import HexagonROI
from ..geometry.roi.multi_hexagon_roi import MultiHexagonROI
from ..geometry.roi.rectangle_roi import RectangularROI

if TYPE_CHECKING:
    from ..geometry.roi.abstract_roi import AbstractROI

roi_cl_dict: dict[str, "type[AbstractROI]"] = {
    HexagonROI.NAME: HexagonROI,
    MultiHexagonROI.NAME: MultiHexagonROI,
    RectangularROI.NAME: RectangularROI,
}


class ROIConfig(AbstractConfig):
    """Configuration for the network layout."""

    roi: StrictStr = Field(
        default=MultiHexagonROI.NAME,
        description=f"Type of region of interest (ROI). Options: {list(roi_cl_dict.keys())}",
    )

    # Field validators (before) ------------------------------------------------
    @field_validator("roi", mode="before")
    @classmethod
    def _validate_roi(cls, roi: str) -> str:
        """Validate the ROI type."""
        if roi not in roi_cl_dict:
            raise ValueError(
                f"Invalid ROI type: {roi}. "
                f"Supported types: {list(roi_cl_dict.keys())}."
            )
        return roi

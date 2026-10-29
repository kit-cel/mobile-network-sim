"""Panel-array geometry and single-element-pattern configuration."""

from pydantic import Field, StrictFloat, model_validator


from .abstract_config import (
    StrictNonNegativeFloat,
    StrictPositiveFloat,
    StrictPositiveInt,
)
from .antenna_config import AntennaConfig
from .beamforming_config import BeamformingConfig


class AntennaArrayConfig(AntennaConfig):
    """Configure a rectangular panel array and its beam codebook.

    The geometry follows the tuple :math:`(M_g, N_g, M, N, P)` in clause 7.3
    of ETSI TR 138 901. The default element-pattern values are taken from
    Table 7.3-1: 65-degree horizontal and vertical half-power beamwidths,
    30 dB vertical sidelobe and front-to-back attenuation, and 8 dBi maximum
    element gain. Beam-codebook settings are simulator-specific.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clause 7.3 and Table 7.3-1, May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    """

    # Inherited parameters from AntennaConfig:
    g_max_dbi: StrictNonNegativeFloat = Field(
        default=8.0,  # Overwrite default from AntennaConfig
        description="Single-element maximum directional gain in dBi.",
    )

    # Panel-array geometry: (M_g, N_g, M, N, P)
    m_panel_v: StrictPositiveInt = Field(
        default=1,
        description="Number of panel rows, i.e. M_g.",
    )
    n_panel_h: StrictPositiveInt = Field(
        default=1,
        description="Number of panel columns, i.e. N_g.",
    )
    m_element_v: StrictPositiveInt = Field(
        default=1,
        description="Number of vertically stacked elements per polarization in one panel, i.e. M.",
    )
    n_element_h: StrictPositiveInt = Field(
        default=1,
        description="Number of horizontal element columns in one panel, i.e. N.",
    )
    n_pol: StrictPositiveInt = Field(
        default=1,
        description="Number of polarizations P. Allowed values: 1 or 2.",
    )

    # Spacing in wavelengths
    panel_spacing_v_lambda: StrictPositiveFloat = Field(
        default=0.5,
        description="Vertical spacing d_g,V between adjacent panels in wavelengths.",
    )
    panel_spacing_h_lambda: StrictPositiveFloat = Field(
        default=0.5,
        description="Horizontal spacing d_g,H between adjacent panels in wavelengths.",
    )
    element_spacing_v_lambda: StrictPositiveFloat = Field(
        default=0.5,
        description="Vertical spacing d_V between adjacent elements within "
        "one panel in wavelengths.",
    )
    element_spacing_h_lambda: StrictPositiveFloat = Field(
        default=0.5,
        description="Horizontal spacing d_H between adjacent elements within "
        "one panel in wavelengths.",
    )

    # Element pattern parameters
    azimuth_3db_bw_deg: StrictPositiveFloat = Field(
        default=65.0,
        description="Element horizontal half-power beamwidth in degrees.",
    )
    elevation_3db_bw_deg: StrictPositiveFloat = Field(
        default=65.0,
        description="Element vertical half-power beamwidth in degrees.",
    )
    front_back_attenuation_db: StrictNonNegativeFloat = Field(
        default=30.0,
        description="Front-to-back attenuation A_max in dB.",
    )
    vertical_sidelobe_attenuation_db: StrictNonNegativeFloat = Field(
        default=30.0,
        description="Vertical sidelobe attenuation SLA_V in dB.",
    )

    # Orientation
    phi_sector_az_bearing_deg: StrictFloat = Field(
        default=0.0,
        description="Sector azimuth in global coordinates, in degrees.",
    )
    mechanical_downtilt_deg: StrictFloat = Field(
        default=10.0,
        description="Mechanical downtilt of the full panel array in degrees.",
    )

    # Beam codebook parameters (optional)
    beamforming: BeamformingConfig | None = Field(
        default=None,
        description="Optional configuration for beam codebook generation. If None, "
        "a default codebook will be generated based on the array geometry and "
        "element pattern.",
    )

    # Derived properties
    @property
    def n_elem_total_h(self) -> int:
        """Total number of element positions in horizontal direction."""
        return self.n_panel_h * self.n_element_h

    @property
    def m_elem_total_v(self) -> int:
        """Total number of element positions in vertical direction."""
        return self.m_panel_v * self.m_element_v

    @property
    def n_elem_per_polarization(self) -> int:
        """Total number of element positions per polarization."""
        return self.n_elem_total_h * self.m_elem_total_v

    @property
    def n_ports(self) -> int:
        """Total number of antenna ports/elements including polarization."""
        return self.n_elem_per_polarization * self.n_pol

    # Validators
    @model_validator(mode="after")
    def _validate_sector_azimuth(self) -> "AntennaArrayConfig":
        if not 0.0 <= self.phi_sector_az_bearing_deg <= 360.0:
            raise ValueError(
                "`sector_azimuth_deg` must be in [0, 360]. "
                f"Got {self.phi_sector_az_bearing_deg}."
            )
        return self

    @model_validator(mode="after")
    def _validate_mechanical_downtilt(self) -> "AntennaArrayConfig":
        if not -90.0 <= self.mechanical_downtilt_deg <= 90.0:
            raise ValueError(
                "`mechanical_downtilt_deg` must be in [-90, 90]. "
                f"Got {self.mechanical_downtilt_deg}."
            )
        return self

    @model_validator(mode="after")
    def _validate_n_polarizations(self) -> "AntennaArrayConfig":
        if self.n_pol not in (1, 2):
            raise ValueError("`n_polarizations` must be 1 or 2. " f"Got {self.n_pol}.")
        return self

    @model_validator(mode="after")
    def _validate_pattern_parameters(self) -> "AntennaArrayConfig":
        if not 0.0 < self.azimuth_3db_bw_deg <= 360.0:
            raise ValueError(
                "`azimuth_3db_bw_deg` must be in (0, 360]. "
                f"Got {self.azimuth_3db_bw_deg}."
            )
        if not 0.0 < self.elevation_3db_bw_deg <= 180.0:
            raise ValueError(
                "`elevation_3db_bw_deg` must be in (0, 180]. "
                f"Got {self.elevation_3db_bw_deg}."
            )
        return self

    @model_validator(mode="after")
    def _validate_beamforming_config(self) -> "AntennaArrayConfig":
        if self.beamforming is None and self.n_elem_per_polarization > 1:
            raise ValueError(
                "Beamforming config must be provided for arrays with > 1 elements."
            )
        return self

    @model_validator(mode="after")
    def _validate_panel_spacing(self) -> "AntennaArrayConfig":
        """
        Panel spacing consistency check:
        For physically separated panels, panel spacing is usually not
        smaller than the size of one panel in the same direction.
        """
        min_panel_spacing_h = self.n_element_h * self.element_spacing_h_lambda
        min_panel_spacing_v = self.m_element_v * self.element_spacing_v_lambda

        if self.n_panel_h > 1 and self.panel_spacing_h_lambda < min_panel_spacing_h:
            raise ValueError(
                "`panel_spacing_h_lambda` should usually be at least one panel width "
                f"({min_panel_spacing_h:.3f} lambda) when n_panel_h > 1. "
                f"Got {self.panel_spacing_h_lambda}."
            )
        if self.m_panel_v > 1 and self.panel_spacing_v_lambda < min_panel_spacing_v:
            raise ValueError(
                "`panel_spacing_v_lambda` should usually be at least one panel height "
                f"({min_panel_spacing_v:.3f} lambda) when n_panel_v > 1. "
                f"Got {self.panel_spacing_v_lambda}."
            )
        return self

"""RRC configuration module."""

from typing import Any, Literal
from pydantic import Field, StrictFloat, field_validator, model_validator

from ..config import (
    BaseConfig,
    StrictNonNegativeInt,
    StrictPositiveInt,
)


class RRCConfig(BaseConfig):
    """Configure the step-based RRC state machine.

    Attributes
    ----------
    t_res_ms : int
        Duration of one RRC simulation step in milliseconds.
    phy_update_interval_ms : int
        Interval between physical-layer updates in milliseconds.
    msr_interval_ms : int
        Measurement-processing interval in milliseconds.
    log_interval_ms : int or None
        Logging interval in milliseconds, or ``None`` to disable logging.
    n310, n311 : int
        Consecutive out-of-sync and in-sync indication thresholds.
    t310_ms : int
        Radio-link-failure detection timer duration in milliseconds.
    q_in_db, q_out_db : float
        In-sync and out-of-sync SINR thresholds in decibels.
    l1_filter_sample_interval_ms : int
        Layer-1 filter sampling interval in milliseconds.
    l1_q_in_filter_len_ms, l1_q_out_filter_len_ms : int
        Layer-1 filter window lengths in milliseconds.
    l3_filter_coef : int
        Layer-3 measurement filter coefficient.
    l3_filter_sample_reference_interval_ms : int
        Reference sampling interval for the layer-3 filter in milliseconds.
    ho_event_config : dict
        Measurement-event configuration passed to the event handler.
    enable_ttt_abort : bool
        Whether a running time-to-trigger may be retargeted.
    t_ho_prep_ms_simulated, t_ho_exec_ms_simulated : int
        Simulated handover preparation and execution durations in milliseconds.
    handover_mode : {"standard", "cho_idealized"}
        Standard HO requires positive preparation. The explicit CHO idealization
        requires zero trigger-time preparation and assumes all requested targets
        were prepared and their commands delivered earlier; it does not model
        that preparation, candidate admission, or configuration validity.
    t_rlfr_ms_simulated : int
        Simulated radio-link-failure recovery duration in milliseconds.
    t_mts_ms : int
        Minimum time of stay used for ping-pong detection in milliseconds.

    Notes
    -----
    Update intervals and procedure durations used as discrete counters must be
    aligned with ``t_res_ms``. Additional alignment checks are performed when
    an :class:`~mobile_network_sim.rrc.RRCStepUE` is created.
    """

    # General RRC parameters
    t_res_ms: StrictPositiveInt = Field(
        default=0, description="RRC time resolution in milliseconds."
    )
    phy_update_interval_ms: StrictPositiveInt = Field(
        default=0, description="PHY update interval in milliseconds."
    )
    msr_interval_ms: StrictPositiveInt = Field(
        default=0, description="Measurement interval in milliseconds."
    )
    log_interval_ms: StrictPositiveInt | None = Field(
        default=None,
        description="Logging interval in milliseconds. If None, logging is disabled.",
    )

    # Counters and timers for radio link monitoring
    n310: StrictNonNegativeInt = Field(
        default=1,
        description="Consecutive out-of-sync indications; default n1 in 3GPP TS 38.331 section 9.2.3.",
    )
    n311: StrictNonNegativeInt = Field(
        default=1,
        description="Consecutive in-sync indications; default n1 in 3GPP TS 38.331 section 9.2.3.",
    )
    t310_ms: StrictNonNegativeInt = Field(
        default=1_000, description="Timer T310 duration in ms."
    )

    # RLM QoS thresholds and filter parameters
    q_in_db: StrictFloat = Field(
        default=-6.0,
        description="Radio link outage threshold Q-in parameter in dB.",
    )
    q_out_db: StrictFloat = Field(
        default=-8.0,
        description="Radio link recovery threshold Q-out parameter in dB.",
    )
    l1_filter_sample_interval_ms: StrictPositiveInt = Field(
        default=10,
        description=(
            "L1 filter sample interval in ms. Currently unused: L1/L3 "
            "measurement filtering (see MeasurementHandler) samples at "
            "t_res_ms, since PHY updates arrive at most once per RRC step "
            "in this simulator. Reserved for when PHY updates faster than "
            "t_res_ms are supported."
        ),
    )
    l1_q_in_filter_len_ms: StrictPositiveInt = Field(
        default=100, description="L1 filter length in ms for Q-in signal."
    )
    l1_q_out_filter_len_ms: StrictPositiveInt = Field(
        default=200, description="L1 filter length in ms for Q-out signal."
    )

    # HO event triggering and L3 filter parameters
    l3_filter_coef: StrictNonNegativeInt = Field(
        default=1,
        description=(
            "L3 filter coefficient for measurements triggering HO events; "
            "one of 0-9, 11, 13, 15, 17, or 19."
        ),
    )
    l3_filter_sample_reference_interval_ms: StrictPositiveInt = Field(
        default=200,
        description="L3 filter sample reference interval in ms.",
    )
    ho_event_config: dict[str, Any] = Field(
        default_factory=dict, description="HO event configuration."
    )
    enable_ttt_abort: bool = Field(
        default=False,
        description="Allow TTT retargeting when another neighbor becomes best.",
    )

    # Simulation times for RRC procedures in ms
    handover_mode: Literal["standard", "cho_idealized"] = Field(
        default="standard",
        description="Explicit protocol semantics; cho_idealized assumes prior preparation.",
    )
    t_ho_prep_ms_simulated: StrictNonNegativeInt = Field(
        default=40,
        description="Trigger-time preparation in ms; zero requires cho_idealized mode.",
    )
    t_ho_exec_ms_simulated: StrictPositiveInt = Field(
        default=40, description="HO execution duration in ms; zero is unsupported."
    )
    t_rlfr_ms_simulated: StrictNonNegativeInt = Field(
        default=1000, description="Re-establishment simulation duration in ms."
    )
    t_mts_ms: StrictNonNegativeInt = Field(
        default=1000, description="Minimum-time-of-stay for ping-pong detection."
    )

    @field_validator("l3_filter_coef")
    @classmethod
    def _validate_l3_filter_coef(cls, value: int) -> int:
        allowed = {*range(10), 11, 13, 15, 17, 19}
        if value not in allowed:
            raise ValueError(
                "L3 filter coefficient must be one of " "0-9, 11, 13, 15, 17, or 19."
            )
        return value

    @model_validator(mode="after")
    def _validate(self) -> "RRCConfig":
        self._validate_handover_mode()
        if self.phy_update_interval_ms < self.t_res_ms:
            raise ValueError(
                "PHY update interval must be greater than or equal to the time "
                "resolution of the RRC simulation."
            )
        if self.phy_update_interval_ms % self.t_res_ms != 0:
            raise ValueError(
                "PHY update interval must be a multiple of the time resolution of "
                "the RRC simulation."
            )
        if self.msr_interval_ms < self.t_res_ms:
            raise ValueError(
                "Measurement interval must be greater than or equal to the time "
                "resolution of the RRC simulation."
            )
        if self.msr_interval_ms % self.t_res_ms != 0:
            raise ValueError(
                "Measurement interval must be a multiple of the time resolution of "
                "the RRC simulation."
            )
        if self.log_interval_ms is not None:
            if self.log_interval_ms < self.t_res_ms:
                raise ValueError(
                    "Logging interval must be greater than or equal to the time "
                    "resolution of the RRC simulation."
                )
            if self.log_interval_ms % self.t_res_ms != 0:
                raise ValueError(
                    "Logging interval must be a multiple of the time resolution of "
                    "the RRC simulation."
                )
        return self

    def _validate_handover_mode(self) -> None:
        """Reject silent protocol changes, including unvalidated model copies."""
        if self.t_ho_exec_ms_simulated <= 0:
            raise ValueError(
                "t_ho_exec_ms_simulated must be positive; zero is unsupported"
            )
        if self.handover_mode == "standard":
            if self.t_ho_prep_ms_simulated <= 0:
                raise ValueError(
                    "Standard HO requires t_ho_prep_ms_simulated > 0. "
                    "To study zero trigger-time preparation, explicitly select "
                    "handover_mode='cho_idealized'. This changes protocol semantics "
                    "and assumes prior target preparation; it is not full CHO."
                )
        elif self.handover_mode == "cho_idealized":
            if self.t_ho_prep_ms_simulated != 0:
                raise ValueError(
                    "handover_mode='cho_idealized' requires t_ho_prep_ms_simulated=0; "
                    "use handover_mode='standard' for positive preparation"
                )
        else:
            raise ValueError(f"Unsupported handover_mode: {self.handover_mode!r}")

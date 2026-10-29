import numpy as np
import pytest
from pydantic import ValidationError

from mobile_network_sim.rrc import RRCConfig, RRCState, RRCStepUE
from mobile_network_sim.rrc.rrc_utils import l3_filtering


def _config(**updates: object) -> RRCConfig:
    values: dict[str, object] = {
        "t_res_ms": 40,
        "phy_update_interval_ms": 40,
        "msr_interval_ms": 40,
        "n310": 2,
        "n311": 2,
        "t310_ms": 80,
        "ho_event_config": {"a3": {"ttt_ms": 40, "hys": 0.0, "offset": 3.0}},
        "t_ho_prep_ms_simulated": 40,
        "t_ho_exec_ms_simulated": 40,
        "t_rlfr_ms_simulated": 80,
        "t_mts_ms": 80,
    }
    values.update(updates)
    return RRCConfig.model_validate(values)


def test_initial_access_selects_best_cell() -> None:
    ue = RRCStepUE(0, _config())
    ue.step(
        time_ms=0,
        rsrp_dbm=np.asarray([-90.0, -75.0]),
        sinr_db=np.asarray([1.0, 10.0]),
    )
    assert ue.state == RRCState.CONNECTED_NORMAL
    assert ue.mob.get_pcell_if_available() == 1


def test_learning_action_reports_acceptance_and_rejection() -> None:
    ue = RRCStepUE(0, _config())
    ue.set_learning_mode(enable_agent_mode=True, enable_eval=True)
    measurements = {
        "rsrp_dbm": np.asarray([-70.0, -60.0]),
        "sinr_db": np.asarray([10.0, 5.0]),
    }
    assert not ue.step(time_ms=0, action=0, **measurements)

    assert ue.step(time_ms=40, action=1, **measurements)
    assert ue.state == RRCState.CONNECTED_HO_PREP
    assert ue.timers.prep.active

    # A second request while preparation is active is rejected and must not be
    # exposed to learning code as an accepted measurement report.
    assert not ue.step(time_ms=80, action=1, **measurements)


def test_a3_leave_cancels_active_ttt_and_resets_target_once() -> None:
    # RSRP/SINR are L3-filtered (a 3GPP-style exponential moving average,
    # see MeasurementHandler) before A3 evaluation, so a single-step swap no
    # longer crosses the A3 threshold immediately -- the swapped measurement
    # must be held for a few samples for the filtered gap to exceed the
    # configured offset. Step counts here were derived by simulating the
    # filter's convergence for this test's config (l3_filter_coef=1,
    # l3_filter_sample_reference_interval_ms=200 defaults, t_res_ms=40).
    ue = RRCStepUE(
        0,
        _config(ho_event_config={"a3": {"ttt_ms": 80, "hys": 0.0, "offset": 3.0}}),
    )
    ue.step(
        time_ms=0,
        rsrp_dbm=np.asarray([-70.0, -90.0]),
        sinr_db=np.asarray([10.0, 0.0]),
    )
    for time_ms in (40, 80, 120):
        ue.step(
            time_ms=time_ms,
            rsrp_dbm=np.asarray([-90.0, -70.0]),
            sinr_db=np.asarray([0.0, 10.0]),
        )

    assert ue.timers.ttt.active
    assert ue.mob.get_target_cell_if_available() == 1

    ue.step(
        time_ms=160,
        rsrp_dbm=np.asarray([-70.0, -90.0]),
        sinr_db=np.asarray([10.0, 0.0]),
    )

    assert not ue.timers.ttt.active
    assert ue.mob.get_target_cell_if_available() is None


def test_timer_durations_must_align_to_grid() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        RRCStepUE(0, _config(t_ho_prep_ms_simulated=30))


def test_config_rejects_misaligned_update_interval() -> None:
    with pytest.raises(ValidationError, match="multiple"):
        _config(phy_update_interval_ms=50)


@pytest.mark.parametrize("coefficient", [*range(10), 11, 13, 15, 17, 19])
def test_config_accepts_supported_l3_filter_coefficients(coefficient: int) -> None:
    assert _config(l3_filter_coef=coefficient).l3_filter_coef == coefficient


@pytest.mark.parametrize("coefficient", [-1, 10, 12, 14, 16, 18, 20])
def test_config_rejects_unsupported_l3_filter_coefficients(coefficient: int) -> None:
    with pytest.raises(ValidationError, match="l3_filter_coef"):
        _config(l3_filter_coef=coefficient)


@pytest.mark.parametrize("coefficient", [-1, 10, 12, 14, 16, 18, 20])
def test_l3_filter_rejects_unsupported_coefficients(coefficient: int) -> None:
    values = np.asarray([1.0])
    with pytest.raises(ValueError, match="k must be one of"):
        l3_filtering(values, values, coefficient, t_sample=200)

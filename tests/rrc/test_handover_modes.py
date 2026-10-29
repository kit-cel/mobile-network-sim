"""Protocol opt-in, zero-preparation timing, and source/target failure boundaries."""

import numpy as np
import pytest

from mobile_network_sim.rrc import RRCConfig, RRCState, RRCStepUE


def config(**updates):
    fields = {
        "t_res_ms": 40,
        "phy_update_interval_ms": 40,
        "msr_interval_ms": 40,
        "n310": 1,
        "n311": 1,
        "t310_ms": 80,
        "l1_q_in_filter_len_ms": 40,
        "l1_q_out_filter_len_ms": 40,
        "l3_filter_coef": 0,
        "ho_event_config": {"a3": {"ttt_ms": 0, "hys": 0.0, "offset": 0.0}},
        "t_ho_prep_ms_simulated": 40,
        "t_ho_exec_ms_simulated": 80,
        "t_rlfr_ms_simulated": 80,
    }
    return RRCConfig.model_validate({**fields, **updates})


def idealized_ue():
    cfg = config(handover_mode="cho_idealized", t_ho_prep_ms_simulated=0)
    with pytest.warns(RuntimeWarning, match="CHO_IDEALIZED.*not a full"):
        ue = RRCStepUE(0, cfg)
    ue.set_learning_mode(True, True)
    return ue


def step(ue, t, action=0, source=10.0, target=20.0):
    return ue.step(
        time_ms=t * 40,
        rsrp_dbm=np.asarray([-60.0, -70.0]),
        sinr_db=np.asarray([source, target]),
        action=action,
        action_mode="target_cell",
    )


def test_zero_preparation_requires_explicit_protocol_change():
    with pytest.raises(ValueError, match="explicitly select.*cho_idealized"):
        config(t_ho_prep_ms_simulated=0)
    with pytest.raises(ValueError, match="requires t_ho_prep_ms_simulated=0"):
        config(handover_mode="cho_idealized")
    ordinary = config()
    with pytest.raises(ValueError, match="explicitly select"):
        ordinary.t_ho_prep_ms_simulated = 0
    # Assignment validators can leave a Pydantic object invalid after an error.
    # The simulator must never execute it if a caller catches that error.
    with pytest.raises(ValueError, match="Standard HO requires"):
        RRCStepUE(0, ordinary)


def test_mode_is_retained_in_yaml_and_warns_at_execution(tmp_path):
    cfg = config(handover_mode="cho_idealized", t_ho_prep_ms_simulated=0)
    cfg.export_as_yaml(tmp_path / "cho.yaml")
    restored = RRCConfig.from_yaml(tmp_path / "cho.yaml")
    assert restored.handover_mode == "cho_idealized"
    assert restored.fingerprint != config().fingerprint
    with pytest.warns(RuntimeWarning, match="Every requested target.*in advance"):
        RRCStepUE(0, restored)


@pytest.mark.parametrize("mode", ["standard", "cho_idealized"])
def test_zero_execution_is_rejected_even_in_cho(mode):
    with pytest.raises(ValueError, match="t_ho_exec_ms_simulated"):
        config(
            handover_mode=mode,
            t_ho_prep_ms_simulated=0 if mode == "cho_idealized" else 40,
            t_ho_exec_ms_simulated=0,
        )
    cfg = config()
    with pytest.raises(ValueError, match="t_ho_exec_ms_simulated"):
        cfg.t_ho_exec_ms_simulated = 0


@pytest.mark.parametrize(
    "update",
    [{"t_ho_exec_ms_simulated": 0}, {"t_ho_prep_ms_simulated": 0}],
)
def test_unvalidated_model_copy_cannot_bypass_runtime_guards(update):
    invalid = config().model_copy(update=update)
    with pytest.raises(ValueError, match="t_ho_.*ms_simulated"):
        RRCStepUE(0, invalid)


def test_prepared_command_does_not_require_healthy_source_at_execution():
    ue = idealized_ue()
    step(ue, 0)
    assert step(ue, 1, action=2, source=-30.0)
    assert ue.flags_now.t310_start == ue.flags_now.t310_cancel == 1
    assert ue.flags_now.prep_start == ue.flags_now.prep_expire == 1
    assert ue.flags_now.exec_start == 1
    assert ue.flags_now.hof_cmd == ue.flags_now.rlf_start == 0
    assert ue.state == RRCState.CONNECTED_HO_EXEC
    assert ue.timers.exec.remaining_ms == 80
    assert not ue.timers.prep.active and not ue.up_connected
    assert not step(ue, 2, action=2, source=-30.0)
    assert ue.timers.exec.remaining_ms == 40
    step(ue, 3, source=-30.0)
    assert ue.flags_now.ho_success == 1
    assert ue.mob.pcell == 1 and ue.up_connected


def test_standard_ho_still_fails_command_delivery_with_active_t310():
    ue = RRCStepUE(0, config())
    ue.set_learning_mode(True, True)
    step(ue, 0)
    assert step(ue, 1, action=2, source=-30.0)
    assert ue.state == RRCState.CONNECTED_HO_PREP
    step(ue, 2, source=-30.0)
    assert ue.flags_now.hof_cmd == ue.flags_now.rlf_start == 1
    assert not ue.flags_now.exec_start


def test_cho_cannot_revive_a_connection_after_t310_expiry():
    ue = idealized_ue()
    step(ue, 0)
    step(ue, 1, source=-30.0)
    step(ue, 2, source=-30.0)
    assert not step(ue, 3, action=2, source=-30.0)
    assert ue.state == RRCState.RLF_RECOVERY
    assert ue.flags_now.rlf_no_ho == ue.flags_now.rlf_start == 1
    assert not ue.flags_now.prep_start and not ue.flags_now.exec_start


def test_cho_still_checks_target_at_the_full_execution_deadline():
    ue = idealized_ue()
    step(ue, 0)
    assert step(ue, 1, action=2)
    step(ue, 2, target=-30.0)
    assert ue.state == RRCState.CONNECTED_HO_EXEC
    assert not ue.flags_now.ho_failure
    step(ue, 3, target=-30.0)
    assert ue.flags_now.hof_exec == ue.flags_now.rlf_start == 1
    assert not ue.flags_now.ho_success


def test_back_to_back_cho_preserves_both_full_execution_intervals():
    ue = idealized_ue()
    step(ue, 0)
    assert step(ue, 1, action=2)
    step(ue, 2)
    assert step(ue, 3, action=1)
    assert ue.flags_now.ho_success == ue.flags_now.exec_start == 1
    assert ue.mob.pcell == 1 and ue.timers.exec.remaining_ms == 80
    step(ue, 4)
    assert ue.timers.exec.remaining_ms == 40
    step(ue, 5)
    assert ue.flags_now.ho_success == 1 and ue.mob.pcell == 0
    assert ue.up_connected


@pytest.mark.parametrize("step_ms,prep_ms,exec_ms", [(20, 60, 100), (10, 30, 80)])
def test_duration_contract_is_not_hardcoded_to_a_40_ms_grid(step_ms, prep_ms, exec_ms):
    cfg = config(
        t_res_ms=step_ms,
        phy_update_interval_ms=step_ms,
        msr_interval_ms=step_ms,
        t_ho_prep_ms_simulated=prep_ms,
        t_ho_exec_ms_simulated=exec_ms,
    )
    ue = RRCStepUE(0, cfg)
    ue.set_learning_mode(True, True)
    end = 1 + (prep_ms + exec_ms) // step_ms
    for t in range(end + 1):
        ue.step(
            time_ms=t * step_ms,
            rsrp_dbm=np.asarray([-60.0, -70.0]),
            sinr_db=np.asarray([10.0, 20.0]),
            action=2 if t == 1 else 0,
            action_mode="target_cell",
        )
        assert ue.flags_now.prep_start == (t == 1)
        assert ue.flags_now.exec_start == (t == 1 + prep_ms // step_ms)
        assert ue.flags_now.ho_success == (t == end)

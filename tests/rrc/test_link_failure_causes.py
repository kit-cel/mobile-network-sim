"""Link-failure cause taxonomy.

``rlf_start`` alone conflates two very different events: a link lost
while simply connected, and a link lost *because* a handover went wrong.
Those matter separately when comparing a learned policy against A3 -- a
policy that hands over too eagerly and one that hands over too late both
show up as "more RLFs", while their failure modes are opposites.

The four causes are mutually exclusive and exhaustive: exactly one is set
on every sample that starts RLF recovery, so the per-cause counts always
sum to ``num_rlf``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobile_network_sim.rrc import RRCConfig, RRCStepUE
from mobile_network_sim.rrc.flags import (
    HANDOVER_FAILURE_CAUSES,
    LINK_FAILURE_CAUSES,
    RRCFlags,
)

N_CELLS = 3
GOOD_SINR = 20.0
BAD_SINR = -25.0


def rrc_config(**overrides: object) -> RRCConfig:
    """Short-timer RRC profile so failures fit in a compact trace."""
    fields: dict[str, object] = {
        "t_res_ms": 40,
        "phy_update_interval_ms": 40,
        "msr_interval_ms": 40,
        "n310": 1,
        "n311": 1,
        "t310_ms": 120,
        "q_in_db": -6.0,
        "q_out_db": -8.0,
        "l1_q_in_filter_len_ms": 40,
        "l1_q_out_filter_len_ms": 40,
        "l3_filter_coef": 0,
        "t_ho_prep_ms_simulated": 120,
        "t_ho_exec_ms_simulated": 120,
        "t_rlfr_ms_simulated": 200,
        "t_mts_ms": 1000,
        "ho_event_config": {"a3": {"ttt_ms": 0, "hys": 0.0, "offset": 0.0}},
    }
    fields.update(overrides)
    return RRCConfig(**fields)  # type: ignore[arg-type]


def drive(sinr_db: np.ndarray, config: RRCConfig | None = None) -> dict[str, np.ndarray]:
    """Step a UE through a per-step ``(n_steps, n_cells)`` SINR trace.

    Q-in/Q-out are supplied explicitly so a test controls the sync state
    directly instead of having to reverse-engineer the L1 filter.
    """
    config = config or rrc_config()
    ue = RRCStepUE(ue_id=0, rrc_config=config)
    ue.set_learning_mode(enable_agent_mode=False, enable_eval=True)

    recorded: dict[str, list[int]] = {
        name: [] for name in (*RRCFlags.field_names(), "pcell")
    }
    for step in range(sinr_db.shape[0]):
        row = sinr_db[step].astype(np.float32)
        ue.step(
            time_ms=step * config.t_res_ms,
            rsrp_dbm=row - 70.0,
            sinr_db=row,
            q_in=(row >= config.q_in_db).astype(int),
            q_out=(row <= config.q_out_db).astype(int),
            action=None,
        )
        for name in RRCFlags.field_names():
            recorded[name].append(int(getattr(ue.flags_now, name)))
        recorded["pcell"].append(int(ue.mob.get_pcell_if_available() or -1))
    return {name: np.asarray(values) for name, values in recorded.items()}


def assert_causes_partition_rlf_starts(flags: dict[str, np.ndarray]) -> None:
    """Exactly one cause per RLF start, and no cause without one."""
    total = sum(flags[name] for name in LINK_FAILURE_CAUSES)
    assert np.array_equal(total, flags["rlf_start"]), (
        "link-failure causes must partition rlf_start exactly; got "
        f"causes={total.tolist()} vs rlf_start={flags['rlf_start'].tolist()}"
    )


def test_a_plain_link_failure_is_not_counted_as_a_handover_failure() -> None:
    """Serving cell fades with no usable neighbour: nothing to hand over to."""
    n_steps = 30
    sinr = np.full((n_steps, N_CELLS), BAD_SINR, dtype=np.float32)
    sinr[:, 0] = GOOD_SINR
    sinr[8:, 0] = BAD_SINR  # every cell now unusable

    flags = drive(sinr)

    assert flags["rlf_start"].sum() >= 1
    assert flags["rlf_no_ho"].sum() >= 1
    assert flags["ho_failure"].sum() == 0
    for cause in HANDOVER_FAILURE_CAUSES:
        assert flags[cause].sum() == 0, cause
    assert_causes_partition_rlf_starts(flags)


def test_a_failure_during_preparation_is_attributed_to_the_prep_phase() -> None:
    """T310 *expires* while preparation is running: the link died mid-prep.

    Distinct from hof_cmd, where preparation completed and the source was
    merely out of sync when the command came due. Separating the two is
    the point: one says the source collapsed after the decision, the
    other says the decision was made on a link already too poor to carry
    the command.
    """
    n_steps = 60
    sinr = np.full((n_steps, N_CELLS), BAD_SINR, dtype=np.float32)
    sinr[:, 0] = GOOD_SINR
    sinr[:, 1] = BAD_SINR
    sinr[5:, 1] = GOOD_SINR + 10.0  # neighbour becomes clearly better -> A3
    sinr[6:, 0] = BAD_SINR  # source collapses right after the trigger

    # T310 must be able to run out inside the preparation window, which is
    # what makes this a mid-preparation loss rather than a late command.
    config = rrc_config(t310_ms=80, t_ho_prep_ms_simulated=400)
    flags = drive(sinr, config)

    assert flags["prep_start"].sum() >= 1, "test trace must start a preparation"
    assert flags["hof_prep"].sum() >= 1
    assert flags["rlf_no_ho"].sum() == 0
    assert_causes_partition_rlf_starts(flags)


def test_a_failure_at_command_reception_is_attributed_to_the_command() -> None:
    """Preparation completes while T310 still runs: the command cannot land.

    Physically this is the "too late handover" of TR 36.839: the UE
    decided correctly but its source link was already out of sync when
    the network tried to deliver the HO command.

    The condition is checked at the preparation-expiry instant, so it is
    classified there. A second, identical check exists one step later in
    ``_after_ho_exec_start_failed_to_phy_problem`` and is unreachable --
    see the note on that method. The failure *mode* is modelled either
    way; only the duplicate branch is dead.
    """
    n_steps = 60
    sinr = np.full((n_steps, N_CELLS), BAD_SINR, dtype=np.float32)
    sinr[:, 0] = GOOD_SINR
    sinr[:, 1] = BAD_SINR
    sinr[5:, 1] = GOOD_SINR + 10.0
    sinr[6:, 0] = BAD_SINR

    # Long T310 relative to preparation: the source is out of sync when
    # the command is due, but T310 has not expired yet.
    config = rrc_config(t310_ms=2000, t_ho_prep_ms_simulated=120)
    flags = drive(sinr, config)

    assert flags["prep_start"].sum() >= 1
    assert flags["hof_cmd"].sum() >= 1
    assert flags["hof_prep"].sum() == 0
    assert_causes_partition_rlf_starts(flags)


def test_a_failure_at_the_execution_end_is_attributed_to_the_target() -> None:
    """The target cell is out of sync when the UE arrives.

    Distinct from a preparation failure: the command was delivered and
    the execution ran; the handover was simply aimed at a cell that had
    gone bad by the time it completed -- the signature of a too-late or
    badly-chosen handover.
    """
    n_steps = 40
    sinr = np.full((n_steps, N_CELLS), BAD_SINR, dtype=np.float32)
    sinr[:, 0] = GOOD_SINR
    sinr[3:, 1] = GOOD_SINR + 10.0  # trigger a handover to cell 1
    # Source stays fine (so prep/cmd succeed) while the target collapses
    # during execution.
    sinr[7:, 1] = BAD_SINR

    flags = drive(sinr)

    assert flags["exec_start"].sum() >= 1, "test trace must reach execution"
    assert flags["hof_exec"].sum() >= 1
    assert flags["hof_prep"].sum() == 0
    assert flags["rlf_no_ho"].sum() == 0
    assert_causes_partition_rlf_starts(flags)


def test_causes_partition_rlf_starts_on_a_randomized_sweep() -> None:
    """The invariant holds on traces nobody hand-designed.

    Hand-built traces only exercise the failure modes their author
    thought of; this sweeps arbitrary fading patterns to catch a path
    that reaches RLF recovery without classifying itself. That path would
    raise, so reaching the assertion at all is already meaningful.
    """
    rng = np.random.default_rng(0)
    for trial in range(40):
        n_steps = int(rng.integers(20, 60))
        sinr = rng.uniform(-25.0, 25.0, size=(n_steps, N_CELLS)).astype(np.float32)
        flags = drive(sinr)
        assert_causes_partition_rlf_starts(flags)
        # Whenever a handover failed, exactly one hof_* cause explains it.
        hof_total = sum(flags[name] for name in HANDOVER_FAILURE_CAUSES)
        assert np.array_equal(hof_total, flags["ho_failure"]), f"trial {trial}"


def test_handover_failures_and_plain_failures_sum_to_the_rlf_count() -> None:
    """The reporting identity downstream metrics rely on."""
    rng = np.random.default_rng(7)
    sinr = rng.uniform(-25.0, 25.0, size=(80, N_CELLS)).astype(np.float32)
    flags = drive(sinr)

    num_rlf = int(flags["rlf_start"].sum())
    num_hof = int(sum(flags[name] for name in HANDOVER_FAILURE_CAUSES).sum())
    num_plain = int(flags["rlf_no_ho"].sum())

    assert num_hof + num_plain == num_rlf


def test_new_cause_flags_do_not_disturb_the_existing_flag_order() -> None:
    """SampleHistory serializes by field order, so old columns must not move."""
    names = RRCFlags.field_names()
    assert names[:17] == [
        "rlf_start",
        "rlf_expire",
        "t310_start",
        "t310_cancel",
        "t310_expire",
        "ttt_start",
        "ttt_cancel",
        "ttt_expire",
        "prep_start",
        "prep_cancel",
        "prep_expire",
        "exec_start",
        "exec_cancel",
        "exec_expire",
        "ho_success",
        "ho_failure",
        "pp_detected",
    ]
    assert names[17:] == list(LINK_FAILURE_CAUSES)


def test_all_three_handover_failure_causes_occur_in_practice() -> None:
    """None of the categories is a label nothing ever produces.

    A taxonomy with a permanently empty bucket would suggest a failure
    mode is impossible when it is merely mis-assigned -- which is exactly
    what an earlier version of this mapping did, folding every
    command-reception failure into hof_prep.
    """
    rng = np.random.default_rng(11)
    totals = {cause: 0 for cause in HANDOVER_FAILURE_CAUSES}
    for _ in range(60):
        sinr = rng.uniform(-25.0, 25.0, size=(120, N_CELLS)).astype(np.float32)
        flags = drive(sinr)
        for cause in HANDOVER_FAILURE_CAUSES:
            totals[cause] += int(flags[cause].sum())

    empty = [cause for cause, count in totals.items() if count == 0]
    assert not empty, f"handover failure causes never observed: {empty} ({totals})"


def test_causes_reset_between_samples() -> None:
    """A cause must describe its own sample, not linger into the next."""
    flags = RRCFlags()
    flags.hof_exec = 1
    flags.rlf_start = 1
    flags.reset()
    assert all(getattr(flags, name) == 0 for name in LINK_FAILURE_CAUSES)

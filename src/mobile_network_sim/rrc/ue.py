"""UE-side RRC logic for step-based simulation."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .flags import LINK_FAILURE_CAUSES, RRCFlags, SampleHistory
from .measurement_handler import A3Action, MeasurementHandler, MobilityContext
from .snapshot import RRCUESnapshotBuilder
from .states import RRCState
from .timers import StepTimers

if TYPE_CHECKING:
    from .config import RRCConfig

DEBUG_LVL = logging.WARNING
logging.basicConfig(level=DEBUG_LVL, format="%(message)s")


_RRC_TRANSITIONS: dict[tuple[RRCState, str], RRCState] = {
    (RRCState.IDLE, "attach"): RRCState.CONNECTED_NORMAL,
    (RRCState.CONNECTED_NORMAL, "ttt_expired"): RRCState.CONNECTED_HO_PREP,
    (RRCState.CONNECTED_HO_PREP, "prep_expired"): RRCState.CONNECTED_HO_EXEC,
    (RRCState.CONNECTED_NORMAL, "t310_expired"): RRCState.CONNECTED_PHY_PROBLEM,
    (RRCState.CONNECTED_HO_PREP, "t310_expired"): RRCState.CONNECTED_PHY_PROBLEM,
    (RRCState.CONNECTED_HO_PREP, "prep_failed"): RRCState.CONNECTED_PHY_PROBLEM,
    (RRCState.CONNECTED_HO_EXEC, "exec_failed"): RRCState.CONNECTED_PHY_PROBLEM,
    (RRCState.CONNECTED_HO_EXEC, "exec_expired"): RRCState.CONNECTED_NORMAL,
    (RRCState.CONNECTED_PHY_PROBLEM, "rlf_recovery_start"): RRCState.RLF_RECOVERY,
    (RRCState.RLF_RECOVERY, "rlf_recovery_expired"): RRCState.IDLE,
}


@dataclass(slots=True)
class StepTiming:
    """Timer durations in integer steps."""

    t310_steps: int
    ttt_steps: int
    prep_steps: int
    exec_steps: int
    rlf_steps: int


class RRCStepUE:
    """Simulate one UE with an explicitly stepped RRC state machine.

    Parameters
    ----------
    ue_id : int
        Stable UE identifier.
    rrc_config : RRCConfig
        Timer, filtering, measurement-event, and procedure configuration.

    Attributes
    ----------
    state : RRCState
        Current state-machine state.
    current_time_ms : int
        Time assigned to the current sample in milliseconds.
    flags_now : RRCFlags
        Transient events raised in the current sample.

    Notes
    -----
    Procedure durations are converted to integer step counters during
    initialization. All configured durations used by those counters must be
    integer multiples of ``rrc_config.t_res_ms``.
    """

    def __init__(self, ue_id: int, rrc_config: "RRCConfig") -> None:
        self.ue_id = ue_id
        self.node_id = ue_id
        self.config = rrc_config
        # model_copy(update=...) intentionally bypasses Pydantic validation.
        # Enforce the physical contract again before any simulator state exists.
        self.config._validate_handover_mode()

        self.enable_ttt_abort = bool(getattr(self.config, "enable_ttt_abort", False))
        self.learning_mode = False
        self.eval_mode = False
        self.extend_rlf_recovery_until_target = False
        self.preserve_pcell_on_rlf_recovery = False

        self.state = RRCState.IDLE
        self._initial_access_done = False
        self._sync_indicator_cnt = 0
        self.rlm_evaluated_now = False
        self.current_time_ms = 0
        self._decision_pending = False

        self.msr_handler = MeasurementHandler(ue_id=ue_id, rrc_config=self.config)
        self.mob = MobilityContext(ue_id=ue_id, t_mts_ms=self.config.t_mts_ms)
        self.flags_now = RRCFlags()
        self.flags_hist = SampleHistory()
        self.timers = StepTimers.create(step_ms=self.config.t_res_ms)

        self._validate_time_alignment()
        self._timing = StepTiming(
            t310_steps=self.config.t310_ms // self.config.t_res_ms,
            ttt_steps=self.msr_handler.a3_config.ttt_ms // self.config.t_res_ms,
            prep_steps=self.config.t_ho_prep_ms_simulated // self.config.t_res_ms,
            exec_steps=self.config.t_ho_exec_ms_simulated // self.config.t_res_ms,
            rlf_steps=self.config.t_rlfr_ms_simulated // self.config.t_res_ms,
        )
        if self._timing.prep_steps == 0:
            warnings.warn(
                "CHO_IDEALIZED: T_prep=0 ms changes standard HO semantics. "
                "Every requested target and its UE command are assumed prepared "
                "in advance; preparation cost, candidate admission and validity "
                "are not simulated. This is not a full Conditional Handover model; "
                "do not compare against standard-HO dataset results as identical physics.",
                RuntimeWarning,
                stacklevel=2,
            )

    def set_learning_mode(self, enable_agent_mode: bool, enable_eval: bool) -> None:
        """Configure external policy control.

        Parameters
        ----------
        enable_agent_mode : bool
            Disable autonomous measurement-event handover triggering when
            ``True``.
        enable_eval : bool
            Record whether the external policy is being evaluated.
        """
        self.learning_mode = bool(enable_agent_mode)
        self.eval_mode = bool(enable_eval)

    def set_milp_replay_alignment(
        self,
        *,
        extend_rlf_recovery: bool,
        preserve_pcell_on_rlf_recovery: bool = False,
    ) -> None:
        """Configure replay-specific radio-link-failure behavior.

        Parameters
        ----------
        extend_rlf_recovery : bool
            Extend recovery in one-step increments until replay permits expiry.
        preserve_pcell_on_rlf_recovery : bool, optional
            Retain the serving-cell association while recovery is active.
        """
        self.extend_rlf_recovery_until_target = bool(extend_rlf_recovery)
        self.preserve_pcell_on_rlf_recovery = bool(preserve_pcell_on_rlf_recovery)

    def _validate_time_alignment(self) -> None:
        grid = self.config.t_res_ms
        durations = {
            "t310_ms": self.config.t310_ms,
            "a3_ttt_ms": self.msr_handler.a3_config.ttt_ms,
            "t_ho_prep_ms": self.config.t_ho_prep_ms_simulated,
            "t_ho_exec_ms": self.config.t_ho_exec_ms_simulated,
            "t_rlfr_ms_simulated": self.config.t_rlfr_ms_simulated,
            "t_mts_ms": self.config.t_mts_ms,
        }
        for name, value in durations.items():
            if value % grid != 0:
                raise ValueError(
                    f"{name}={value} ms is not a multiple of t_res_ms={grid} ms"
                )

    def _prepare_new_sample(self) -> None:
        current_sample_len = self.current_time_ms // self.config.t_res_ms + 1
        self.flags_now.reset()
        self.rlm_evaluated_now = False
        self.flags_hist.ensure_length(current_sample_len)
        self._commit_flags()

    def _commit_flags(self) -> None:
        self.flags_hist.write_current_sample(self.flags_now)

    def _log(self, msg: str, lvl: int = logging.INFO) -> None:
        if DEBUG_LVL <= lvl:
            if lvl == logging.DEBUG:
                logging.debug(
                    "[RRC-STEP] t=%d UE%s %s %s",
                    self.current_time_ms,
                    self.ue_id,
                    self.state,
                    msg,
                )
            elif lvl == logging.INFO:
                logging.info(
                    "[RRC-STEP] t=%d UE%s %s", self.current_time_ms, self.ue_id, msg
                )

    def begin_sample(self, time_ms: int) -> None:
        """Start a sample and clear its transient event flags.

        Parameters
        ----------
        time_ms : int
            Current simulation time in milliseconds.
        """
        self.current_time_ms = int(time_ms)
        self._prepare_new_sample()

    def receive_phy_update(
        self,
        rsrp_dbm: np.ndarray,
        sinr_db: np.ndarray,
        q_in: np.ndarray | None = None,
        q_out: np.ndarray | None = None,
    ) -> None:
        """Update measurements and optional precomputed sync indications.

        Parameters
        ----------
        rsrp_dbm : numpy.ndarray
            Per-cell RSRP values in dBm, with shape ``(n_cells,)``.
        sinr_db : numpy.ndarray
            Per-cell SINR values in dB, with shape ``(n_cells,)``.
        q_in, q_out : numpy.ndarray, optional
            Precomputed per-cell in-sync and out-of-sync Boolean indications.
        """
        self.msr_handler.update_measurements(
            rsrp_dbm=rsrp_dbm, sinr_db=sinr_db, q_in=q_in, q_out=q_out
        )

    def initial_access(self, cell_idx: int | None = None) -> None:
        """
        Perform initial access to the network.

        Parameters
        ----------
        cell_idx : int or None
            The index of the cell to attach to. If None, the UE will attach to
            the best cell according to the measurement handler.

        Returns
        -------
        bool
            ``True`` exactly when the external learning action was accepted
            and started handover preparation.  ``False`` includes action zero,
            non-learning operation, and positive requests rejected by the
            current RRC state or an invalid target.

        Raises
        ------
        RuntimeError
            If initial access has already been completed.
        """
        if self._initial_access_done:
            raise RuntimeError(f"UE{self.ue_id}: RRC already initialized")
        self._initial_access_done = True
        if cell_idx is None:
            self._apply_event("attach")
        else:
            self._apply_state_transition(RRCState.CONNECTED_NORMAL)
            self._after_attach(cell_idx=cell_idx)

    def _apply_event(self, event_name: str) -> None:
        new_state = self._resolve_next_state(event_name)
        self._apply_state_transition(new_state)
        self._apply_state_side_effects(event_name)

    def _resolve_next_state(self, event_name: str) -> RRCState:
        key = (self.state, event_name)
        if key not in _RRC_TRANSITIONS:
            raise RuntimeError(f"Invalid transition: ({self.state}, {event_name})")
        return _RRC_TRANSITIONS[key]

    def _apply_state_transition(self, new_state: RRCState) -> None:
        self.state = new_state

    def _apply_state_side_effects(self, event_name: str) -> None:
        if event_name == "attach":
            self._after_attach()
        elif event_name == "ttt_expired":
            self._after_ttt_expired_to_ho_prep()
        elif event_name == "prep_expired":
            self._after_prep_expired_to_ho_exec()
        elif event_name == "exec_expired":
            self._after_exec_expired_to_normal()
        elif event_name == "t310_expired":
            self._after_t310_expired_to_phy_problem()
        elif event_name == "prep_failed":
            self._after_prep_failed_to_phy_problem()
        elif event_name == "exec_failed":
            self._after_exec_failed_to_phy_problem()
        elif event_name == "rlf_recovery_start":
            self._after_rlf_recovery_start()
        elif event_name == "rlf_recovery_expired":
            self._after_rlf_recovery_expired_to_idle_then_attach()
        else:
            raise RuntimeError(
                f"Unhandled transition side effects for event {event_name}"
            )

    def _after_attach(self, cell_idx: int | None = None) -> None:
        self._log("Attaching")
        if cell_idx is None:
            cell_idx = self.msr_handler.get_best_cell_index(msr_obj="rsrp")
        self.mob.attach(int(cell_idx))

    def _after_ttt_expired_to_ho_prep(self) -> None:
        self._log("HO preparation started")
        self.flags_now.prep_start = 1
        self._commit_flags()
        self.timers.ttt.clear()
        if self._timing.prep_steps == 0:
            # A zero timer is inactive and will never tick. Complete this
            # virtual phase now; execution still starts with its full duration.
            self._on_prep_expiry()
        else:
            self.timers.prep.start(self._timing.prep_steps)

    def _after_prep_expired_to_ho_exec(self) -> None:
        if self.mob.get_target_cell_if_available() is None:
            raise RuntimeError("No reporting cell for HO execution")
        self._log("HO execution started")
        self._sync_indicator_cnt = 0
        self.flags_now.exec_start = 1
        self._commit_flags()
        self.timers.prep.clear()
        self.timers.exec.start(self._timing.exec_steps)
        if self.timers.t310.active:
            # Unreachable by construction, enforced rather than assumed.
            #
            # This method is only entered via the "prep_expired" event,
            # which _on_prep_expiry raises exclusively in the branch where
            # T310 is inactive -- the active case is already classified
            # there as hof_cmd. Nothing in between can start T310:
            # _t310_start is reached only from _out_of_sync, only from
            # update_sync_state, which step() runs *before* the timer tick
            # that produced this expiry.
            #
            # The check stays as an invariant so that reordering step()
            # fails loudly here instead of silently creating an
            # unclassified path into RLF recovery.
            raise RuntimeError(
                f"UE{self.ue_id}: T310 active at HO execution start; this "
                "contradicts _on_prep_expiry's branch condition and would "
                "leave a link failure unclassified"
            )

    def _after_exec_expired_to_normal(self) -> None:
        self._log("HO execution finished")
        self.mob.perform_handover(self.current_time_ms)
        self.flags_now.pp_detected = int(self.mob.pp_detected)
        self.flags_now.ho_success = 1
        self._commit_flags()

    def _after_t310_expired_to_phy_problem(self) -> None:
        if self.timers.prep.active:
            self.flags_now.prep_cancel = 1
            self.flags_now.ho_failure = 1
            # T310 *expired* while preparation was running: the link was
            # lost during the preparation phase, before the HO command
            # was ever due. Distinct from hof_cmd, where preparation
            # completed but the source was out of sync at delivery.
            self.flags_now.hof_prep = 1
            self._commit_flags()
            self._log("HOF due to T310 expiry, cancel HO.")
            self.timers.prep.cancel()
            self.mob.reset_target_cell()
        else:
            self._log("Declaring PHY problem due to T310 expiry")
            # No handover was in progress: a link failure in its own right.
            self.flags_now.rlf_no_ho = 1
            self._ttt_cancel()
        self._start_rlf_recovery()

    def _after_prep_failed_to_phy_problem(self) -> None:
        self._log("HOF due to concurrent T310, cancel HO.")
        self.flags_now.prep_cancel = 1
        self.flags_now.ho_failure = 1
        # Reached from _on_prep_expiry when the preparation timer runs out
        # while T310 is still active: preparation *completed*, but the
        # source link is out of sync exactly when the HO command would be
        # delivered, so the UE cannot receive it. That is the
        # command-reception failure, not a failure during preparation --
        # the link was not lost mid-preparation, it was already bad when
        # the command came due.
        self.flags_now.hof_cmd = 1
        self._commit_flags()
        self.timers.prep.cancel_if_active()
        self.mob.reset_target_cell()
        self._t310_cancel()
        self._start_rlf_recovery()

    def _after_exec_failed_to_phy_problem(self) -> None:
        self._log("Target cell OOS at HO exec end, declare HOF.")
        self.flags_now.exec_cancel = 1
        self.flags_now.ho_failure = 1
        # The handover was commanded and executed, but the *target* was
        # unusable on arrival -- a too-late or badly-chosen handover.
        self.flags_now.hof_exec = 1
        self._commit_flags()
        self.mob.reset_target_cell()
        self._start_rlf_recovery()

    def _after_rlf_recovery_start(self) -> None:
        self._log("Starting RLF recovery.")

    def _after_rlf_recovery_expired_to_idle_then_attach(self) -> None:
        self._log("RLF recovery finished, re-establishing connection")
        target_cell = self.msr_handler.get_best_cell_index(msr_obj="rsrp")
        self.mob.detach()
        self._apply_state_transition(RRCState.CONNECTED_NORMAL)
        self._after_attach(cell_idx=target_cell)

    def update_sync_state(self) -> None:
        """Update sync counters and radio-link-failure detection state."""
        if not self._initial_access_done:
            raise RuntimeError("RRC UE not initialized with PHY measurements")
        if self.state not in {RRCState.CONNECTED_NORMAL, RRCState.CONNECTED_HO_PREP}:
            return
        if self.timers.rlf_recovery.active:
            return
        if self.timers.exec.active:
            return
        # MILP alignment: on the sample where HO preparation expires, MILP
        # already marks HO execution as running. Old-PCell q-out must not open
        # a new T310/HOF path on that boundary sample.
        if (
            self.state == RRCState.CONNECTED_HO_PREP
            and self.timers.prep.active
            and self.timers.prep.remaining_steps <= 1
        ):
            return

        self.rlm_evaluated_now = True

        pcell = self.mob.pcell
        if self.msr_handler.q_in[pcell] and self.timers.t310.active:
            self._in_sync()
        elif self.msr_handler.q_out[pcell] and not self.timers.t310.active:
            self._out_of_sync()

    def _in_sync(self) -> None:
        self._sync_indicator_cnt += 1
        self._log(f"in_sync cnt={self._sync_indicator_cnt}")
        if self._sync_indicator_cnt == self.config.n311:
            self._log("N311 reached maximum")
            self._sync_indicator_cnt = 0
            self._t310_cancel()

    def _out_of_sync(self) -> None:
        self._sync_indicator_cnt += 1
        self._log(f"out_of_sync cnt={self._sync_indicator_cnt}")
        if self._sync_indicator_cnt == self.config.n310:
            self._log("N310 reached maximum")
            self._sync_indicator_cnt = 0
            self._t310_start()

    def _t310_start(self) -> None:
        if self._timing.t310_steps <= 0:
            self.flags_now.t310_start = 1
            self.flags_now.t310_expire = 1
            self._commit_flags()
            self._apply_event("t310_expired")
            return
        self._log("T310 started")
        self._sync_indicator_cnt = 0
        self.flags_now.t310_start = 1
        self._commit_flags()
        if not self.timers.t310.active:
            self.timers.t310.start(self._timing.t310_steps)

    def _t310_cancel(self) -> None:
        if not self.timers.t310.active:
            return
        self._log("T310 stopped")
        self.flags_now.t310_cancel = 1
        self._commit_flags()
        self.timers.t310.cancel()
        self._sync_indicator_cnt = 0

    def _ttt_start(self) -> None:
        if self.timers.ttt.active:
            return
        if self._timing.ttt_steps <= 0:
            self.flags_now.ttt_start = 1
            self.flags_now.ttt_expire = 1
            self._commit_flags()
            self._apply_event("ttt_expired")
            return
        self._log("TTT started")
        self.flags_now.ttt_start = 1
        self._commit_flags()
        self.timers.ttt.start(self._timing.ttt_steps)

    def _ttt_cancel(self) -> None:
        if not self.timers.ttt.active:
            return
        self._log("TTT cancelled")
        self.flags_now.ttt_cancel = 1
        self._commit_flags()
        self.timers.ttt.cancel()
        self.mob.reset_target_cell()

    def _start_rlf_recovery(self) -> None:
        if self.state != RRCState.CONNECTED_PHY_PROBLEM:
            raise RuntimeError(
                "RLF recovery timer can only start from CONNECTED_PHY_PROBLEM"
            )
        self._log("RLF recovery timer started")
        # Every path into RLF recovery must have classified itself first,
        # otherwise the failure-cause breakdown silently loses events and
        # the per-cause counts stop summing to num_rlf.
        causes = [name for name in LINK_FAILURE_CAUSES if getattr(self.flags_now, name)]
        if len(causes) != 1:
            raise RuntimeError(
                f"UE{self.ue_id}: RLF recovery started with {len(causes)} link "
                f"failure causes set ({causes}); exactly one is required"
            )
        self.flags_now.rlf_start = 1
        self._commit_flags()
        if self._timing.rlf_steps <= 0:
            self.flags_now.rlf_expire = 1
            self._commit_flags()
            self._apply_event("rlf_recovery_expired")
            return
        # MILP marks RLF recovery as running for the start sample plus the
        # configured recovery window. Since timers tick at the beginning of the
        # next sample, add one step so expiry/re-establishment occurs on the
        # first sample where MILP has i_rlf_running == 0.
        self.timers.rlf_recovery.start(self._timing.rlf_steps + 1)
        self._apply_event("rlf_recovery_start")

    def _on_t310_expiry(self) -> None:
        self._log("T310 expired")
        self.flags_now.t310_expire = 1
        self._commit_flags()
        self.timers.t310.clear()
        self._apply_event("t310_expired")

    def _on_ttt_expiry(self) -> None:
        self._log("TTT expired")
        self.flags_now.ttt_expire = 1
        self._commit_flags()
        self.timers.ttt.clear()
        self._apply_event("ttt_expired")

    def _on_prep_expiry(self) -> None:
        if self.timers.rlf_recovery.active:
            return
        if self.mob.get_target_cell_if_available() is None:
            raise RuntimeError("No reporting cell for HO preparation")
        self._log("HO preparation expired")
        self.flags_now.prep_expire = 1
        self._commit_flags()
        if self._timing.prep_steps == 0:
            # Explicit CHO idealization: the UE already holds the command, so
            # an active source T310 cannot cause a new command-delivery failure.
            # T310 expiry earlier in prepare_sample still enters RLF and blocks
            # the request; this cannot revive an already failed connection.
            self._t310_cancel()
        if self.timers.t310.active:
            self._apply_event("prep_failed")
        else:
            self._apply_event("prep_expired")

    def _on_exec_expiry(self) -> None:
        self._log("HO execution expired")
        self.flags_now.exec_expire = 1
        self._commit_flags()
        self.timers.exec.clear()
        target_cell = self.mob.target_cell
        if self.msr_handler.q_out[target_cell]:
            self._apply_event("exec_failed")
        else:
            self._apply_event("exec_expired")

    def _on_rlf_recovery_expiry(self) -> None:
        self._log("RLF recovery timer expired")
        self.flags_now.rlf_expire = 1
        self._commit_flags()
        self.timers.rlf_recovery.clear()
        self._apply_event("rlf_recovery_expired")

    def process_events(self) -> None:
        """Process autonomous measurement events for the current sample.

        Notes
        -----
        Event processing is skipped during external policy control and outside
        the normal connected state.
        """
        if self.learning_mode:
            return
        if self.state != RRCState.CONNECTED_NORMAL:
            return

        a3 = self.msr_handler.evaluate_a3(
            pcell_idx=self.mob.pcell,
            target_cell=self.mob.get_target_cell_if_available(),
        )

        if a3.action == A3Action.NONE:
            return
        if a3.action == A3Action.ENTER:
            if a3.target_cell is None:
                raise RuntimeError("A3 entry condition met but no best neighbor")
            self.mob.set_target_cell(a3.target_cell)
            self._ttt_start()
        elif a3.action == A3Action.STAY:
            return
        elif a3.action == A3Action.LEAVE:
            self._ttt_cancel()
        elif a3.action == A3Action.SWITCH:
            if a3.target_cell is None:
                raise RuntimeError("A3 switch condition met but no new target cell")
            self._ttt_cancel()
            if self.enable_ttt_abort:
                self._log(f"Cell changed target -> {a3.target_cell}")
                self.mob.set_target_cell(a3.target_cell)
                self._ttt_start()
        elif a3.action == A3Action.RETARGET:
            if a3.target_cell is None:
                raise RuntimeError("A3 retarget condition met but no new target cell")
            if self.enable_ttt_abort:
                self._log(f"Cell changed {self.mob.target_cell} -> {a3.target_cell}")
                self._ttt_cancel()
                self.mob.change_target_cell(a3.target_cell)
                self._ttt_start()
        else:
            raise RuntimeError(f"Unknown A3 action {a3.action}")

    def request_handover(self, target_cell: int) -> bool:
        """Request a handover directly from the policy.

        In learning mode, this replaces A3/TTT triggering. The target cell is
        set immediately and HO preparation starts in the current sample.

        Parameters
        ----------
        target_cell : int
            Candidate target-cell index.

        Returns
        -------
        bool
            ``True`` when handover preparation was started; ``False`` when the
            request was invalid for the current state or cell association.
        """
        if self.state != RRCState.CONNECTED_NORMAL:
            return False

        pcell = self.mob.get_pcell_if_available()
        if pcell is None or target_cell == pcell:
            return False

        n_cells = self.msr_handler.rsrp_dbm.size
        if not 0 <= int(target_cell) < n_cells:
            return False

        if self.mob.get_target_cell_if_available() is not None:
            self.mob.reset_target_cell()
        self.mob.set_target_cell(int(target_cell))
        self.flags_now.ttt_start = 1
        self.flags_now.ttt_expire = 1
        self._commit_flags()
        self._apply_event("ttt_expired")
        return True

    def request_measurement_report_to_best_neighbor(self) -> bool:
        """Request HO preparation to the strongest non-serving RSRP cell.

        The learning policy models the UE sending a measurement report. The
        network side then selects the target cell. For now this target is the
        best filtered RSRP neighbor, excluding the current PCell.

        Returns
        -------
        bool
            ``True`` when a finite non-serving candidate starts handover
            preparation; otherwise ``False``.
        """
        pcell = self.mob.get_pcell_if_available()
        if pcell is None:
            return False

        rsrp = np.asarray(self.msr_handler.rsrp_dbm, dtype=np.float32)
        if rsrp.ndim != 1 or rsrp.size < 2:
            return False

        candidate_rsrp = rsrp.copy()
        candidate_rsrp[int(pcell)] = -np.inf
        target_cell = int(np.argmax(candidate_rsrp))
        if not np.isfinite(candidate_rsrp[target_cell]):
            return False

        return self.request_handover(target_cell)

    def _tick_failure_timers(self) -> None:
        if self.timers.t310.tick():
            self._on_t310_expiry()

    def _tick_mobility_timers(self) -> None:
        ttt_active = self.timers.ttt.active
        prep_active = self.timers.prep.active
        exec_active = self.timers.exec.active
        rlf_active = self.timers.rlf_recovery.active

        if ttt_active and self.timers.ttt.tick():
            self._on_ttt_expiry()
        if prep_active and self.timers.prep.tick():
            self._on_prep_expiry()
        if exec_active and self.timers.exec.tick():
            self._on_exec_expiry()
        if rlf_active and self.timers.rlf_recovery.tick():
            if self.extend_rlf_recovery_until_target:
                self.timers.rlf_recovery.start(1)
            else:
                self._on_rlf_recovery_expiry()

    def step(
        self,
        *,
        time_ms: int,
        rsrp_dbm: np.ndarray,
        sinr_db: np.ndarray,
        q_in: np.ndarray | None = None,
        q_out: np.ndarray | None = None,
        action: int | None = None,
        action_mode: str = "binary_report",
        initial_pcell: int | None = None,
    ) -> bool:
        """Advance the state machine by one measurement sample.

        Parameters
        ----------
        time_ms : int
            Current simulation time in milliseconds.
        rsrp_dbm, sinr_db : numpy.ndarray
            Per-cell RSRP in dBm and SINR in dB.
        q_in, q_out : numpy.ndarray, optional
            Optional precomputed per-cell sync indications.
        action : int, optional
            External policy action. Positive values request a handover when
            learning mode is enabled.
        action_mode : {"binary_report", "target_cell"}, optional
            Interpretation of a positive external action.
        initial_pcell : int, optional
            Serving-cell index used for first access. If omitted, the strongest
            measured cell is selected.

        Returns
        -------
        bool
            ``True`` exactly when the external learning action was accepted
            and started handover preparation.  ``False`` includes action zero,
            non-learning operation, and positive requests rejected by the
            current RRC state or an invalid target.

        Raises
        ------
        ValueError
            If ``action_mode`` is unknown when a positive action is processed.
        """
        self.prepare_sample(
            time_ms=time_ms,
            rsrp_dbm=rsrp_dbm,
            sinr_db=sinr_db,
            q_in=q_in,
            q_out=q_out,
            initial_pcell=initial_pcell,
        )
        return self.finish_sample(action=action, action_mode=action_mode)

    def prepare_sample(
        self,
        *,
        time_ms: int,
        rsrp_dbm: np.ndarray,
        sinr_db: np.ndarray,
        q_in: np.ndarray | None = None,
        q_out: np.ndarray | None = None,
        initial_pcell: int | None = None,
    ) -> None:
        """Process measurements and timers, stopping immediately before A3/action.

        The resulting state is causal at this sample, including a handover
        that has just completed. Both native A3 and external policies decide
        here; no extra physical tick is inserted for an external policy.
        """
        if self._decision_pending:
            raise RuntimeError("finish_sample must consume the pending decision first")
        self.begin_sample(time_ms=time_ms)
        self._tick_failure_timers()
        self.receive_phy_update(
            rsrp_dbm=rsrp_dbm, sinr_db=sinr_db, q_in=q_in, q_out=q_out
        )

        if not self.initial_access_done:
            self.initial_access(cell_idx=initial_pcell)

        self.update_sync_state()
        self._tick_mobility_timers()
        self._decision_pending = True

    def finish_sample(
        self,
        *,
        action: int | None = None,
        action_mode: str = "binary_report",
    ) -> bool:
        """Apply the decision without advancing measurements, timers or time."""
        if not self._decision_pending:
            raise RuntimeError("prepare_sample must precede finish_sample")
        self._decision_pending = False
        self.process_events()

        action_accepted = False
        if self.learning_mode and action is not None and int(action) > 0:
            if action_mode == "binary_report":
                action_accepted = self.request_measurement_report_to_best_neighbor()
            elif action_mode == "target_cell":
                target_cell = int(action) - 1
                action_accepted = self.request_handover(target_cell)
            else:
                raise ValueError(f"Unknown action_mode: {action_mode}")
        return bool(action_accepted)

    def get_flags(self) -> dict[str, np.ndarray]:
        """Return event-flag history.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary where keys are flag names and values are numpy arrays
            containing the historical values of the flags.
        """
        return {
            name: np.asarray(values)
            for name, values in self.flags_hist.as_dict().items()
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the current state and measurements as a flat snapshot.

        Returns
        -------
        dict of str to object
            Snapshot fields produced by :class:`RRCUESnapshotBuilder`.
        """
        return RRCUESnapshotBuilder.snapshot(self)

    def get_snapshot_fields(self) -> list[str]:
        """Return the ordered snapshot field names.

        Returns
        -------
        list of str
            Field names produced by :meth:`snapshot`.
        """
        return RRCUESnapshotBuilder.fields()

    def get_pcell_sinr(self) -> float:
        """Return the L3-filtered serving-cell SINR.

        This is the *measurement* value, smoothed per TS 38.331 clause
        5.5.3.2. For achievable rate use :meth:`get_pcell_raw_sinr`
        instead -- see ``analysis/report/analysis-v2.tex``.

        Returns
        -------
        float
            Serving-cell SINR in dB, or negative infinity if the UE has no
            serving cell.
        """
        pcell = self.mob.get_pcell_if_available()
        if pcell is None:
            return -np.inf
        return float(self.msr_handler.sinr_db[pcell])

    def get_pcell_raw_sinr(self) -> float:
        """Return the unfiltered serving-cell SINR.

        The achievable rate is a property of the channel, not of the
        measurement procedure, so it must be computed from this rather
        than from the smoothed reporting value.

        Returns
        -------
        float
            Serving-cell SINR in dB, or negative infinity if the UE has no
            serving cell.
        """
        pcell = self.mob.get_pcell_if_available()
        if pcell is None:
            return -np.inf
        return float(self.msr_handler.raw_sinr_db[pcell])

    @property
    def in_sync(self) -> bool | None:
        """
        Indicate whether the UE is currently in sync with the serving cell.

        Returns
        -------
        bool | None
            True if the UE is in sync, False if out of sync, None if no serving cell is available.
        """
        pcell = self.mob.get_pcell_if_available()
        if pcell is None:
            return None
        return bool(self.msr_handler.q_in[pcell])

    @property
    def out_of_sync(self) -> bool | None:
        """
        Indicate whether the UE is currently out of sync with the serving cell.

        Returns
        -------
        bool | None
            True if the UE is out-of-sync, False if in sync, None if no serving cell is available.
        """
        pcell = self.mob.get_pcell_if_available()
        if pcell is None:
            return None
        return bool(self.msr_handler.q_out[pcell])

    @property
    def up_connected(self) -> bool:
        """Whether the user plane is available.

        The user plane remains available during handover preparation and is
        interrupted when handover execution starts.
        """
        return self.state in {RRCState.CONNECTED_NORMAL, RRCState.CONNECTED_HO_PREP}

    @property
    def up_interrupted(self) -> bool:
        """Whether the user plane is unavailable.

        Handover preparation does not interrupt the user plane. Handover
        execution, physical-layer failure handling, RLF recovery, and idle
        state do interrupt it.
        """
        return not self.up_connected

    @property
    def n310_count(self) -> int:
        """Return the current N310 counter value if T310 is inactive, otherwise 0."""
        return self._sync_indicator_cnt if not self.timers.t310.active else 0

    @property
    def n311_count(self) -> int:
        """Return the current N311 counter value if T310 is active, otherwise 0."""
        return self._sync_indicator_cnt if self.timers.t310.active else 0

    @property
    def initial_access_done(self) -> bool:
        """Flag indicating whether the UE has performed initial access to the network."""
        return self._initial_access_done

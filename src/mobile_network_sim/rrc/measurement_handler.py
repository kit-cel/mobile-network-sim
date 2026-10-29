"""Measurement handler for RRC Lite."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from .rrc_utils import BoxcarFilter, l3_filtering

if TYPE_CHECKING:
    from .config import RRCConfig


class MobilityContext:
    """Track serving, previous, and handover-target cells for one UE.

    Parameters
    ----------
    ue_id : int
        UE identifier used in validation errors.
    t_mts_ms : int
        Maximum interval, in milliseconds, used by the implemented ping-pong
        detection rule.
    """

    def __init__(self, ue_id: int, t_mts_ms: int) -> None:
        self.ue_id = ue_id
        self.t_mts_ms = t_mts_ms

        self._pcell: int | None = None
        self._prev_pcell: int | None = None
        self._target_cell: int | None = None

        self._last_ho_t_ms: int | None = None
        self._pp_detected: bool = False

    def attach(self, cell_idx: int) -> None:
        """Attach the UE to a serving cell.

        Parameters
        ----------
        cell_idx : int
            Index of the new primary cell.

        Raises
        ------
        RuntimeError
            If a primary cell is already set.
        """
        if self._pcell is not None:
            raise RuntimeError(f"UE{self.ue_id} already has a primary cell")
        self._prev_pcell = None
        self._pcell = cell_idx
        self._target_cell = None

    def detach(self) -> None:
        """Detach from the current cell, resetting the mobility context."""
        if self._pcell is None:
            raise RuntimeError(f"UE{self.ue_id} has no primary cell to detach from")
        self._prev_pcell = self._pcell
        self._pcell = None
        self._target_cell = None

        self._pp_detected = False

    def set_target_cell(self, cell_idx: int) -> None:
        """Set the target cell used during handover preparation.

        Parameters
        ----------
        cell_idx : int
            Target-cell index.

        Raises
        ------
        RuntimeError
            If the UE has no primary cell or already has a target cell.
        """
        if self._pcell is None:
            raise RuntimeError(
                f"UE{self.ue_id} has no primary cell for setting target cell"
            )
        if self._target_cell is not None:
            raise RuntimeError(f"UE{self.ue_id} target cell already set")
        self._target_cell = cell_idx

    def reset_target_cell(self) -> None:
        """Reset the target cell, e.g., after a handover failure."""
        if self._target_cell is None:
            raise RuntimeError(f"UE{self.ue_id} has no target cell to reset")
        self._target_cell = None

    def change_target_cell(self, cell_idx: int) -> None:
        """Replace the current handover target.

        Parameters
        ----------
        cell_idx : int
            New target-cell index.

        Raises
        ------
        RuntimeError
            If the UE has no primary cell or no current target cell.
        """
        if self._pcell is None:
            raise RuntimeError(
                f"UE{self.ue_id} has no primary cell for changing target cell"
            )
        if self._target_cell is None:
            raise RuntimeError(f"UE{self.ue_id} has no target cell to change")
        self._target_cell = cell_idx

    def perform_handover(self, current_time_ms: int) -> None:
        """Promote the target cell to the primary cell.

        Parameters
        ----------
        current_time_ms : int
            Current simulation time in milliseconds.

        Raises
        ------
        RuntimeError
            If either the primary cell or target cell is unavailable.
        """
        if self._pcell is None:
            raise RuntimeError(f"UE{self.ue_id} has no primary cell for handover")
        if self._target_cell is None:
            raise RuntimeError(f"UE{self.ue_id} has no target cell for handover")
        # Ping-pong (A -> B -> A) compares the new pcell against the cell held
        # two handovers back, i.e. _prev_pcell as of *before* this handover's
        # update below -- captured first since it is about to be overwritten
        # to one-hop-back (B, the cell this handover departs from), which by
        # construction never equals the new pcell.
        cell_two_hops_back = self._prev_pcell
        self._prev_pcell = self._pcell
        self._pcell = self._target_cell
        self._target_cell = None

        self._pp_detected = self._detect_ping_pong(current_time_ms, cell_two_hops_back)
        self._last_ho_t_ms = current_time_ms

    def get_pcell_if_available(self) -> int | None:
        """Return the current primary-cell index, if available.

        Returns
        -------
        int or None
            Primary-cell index, or ``None`` while detached.
        """
        return self._pcell

    def get_prev_pcell_if_available(self) -> int | None:
        """Return the previous primary-cell index, if available.

        Returns
        -------
        int or None
            Previous primary-cell index, or ``None`` if none has been stored.
        """
        return self._prev_pcell

    def get_target_cell_if_available(self) -> int | None:
        """Return the target-cell index, if available.

        Returns
        -------
        int or None
            Target-cell index, or ``None`` outside handover preparation.
        """
        return self._target_cell

    def _detect_ping_pong(
        self, current_time_ms: int, cell_two_hops_back: int | None
    ) -> bool:
        """Detect if this handover returned to the cell held before the last one.

        Parameters
        ----------
        current_time_ms : int
            Current simulation time in milliseconds.
        cell_two_hops_back : int or None
            Serving-cell index held immediately before the *previous*
            handover, i.e. two hops back relative to the pcell this
            handover just moved to. ``None`` if fewer than two handovers
            have occurred yet.
        """
        if self._last_ho_t_ms is None or cell_two_hops_back is None:
            return False

        if self._pcell is None:
            raise RuntimeError(f"UE{self.ue_id} has insufficient mobility context.")
        if (
            cell_two_hops_back == self._pcell
            and current_time_ms - self._last_ho_t_ms <= self.t_mts_ms
        ):
            return True
        return False

    @property
    def pcell(self) -> int:
        """Current primary cell index."""
        if self._pcell is None:
            raise RuntimeError(f"UE{self.ue_id} has no primary cell set")
        return self._pcell

    @property
    def prev_pcell(self) -> int:
        """Previous primary cell index."""
        if self._prev_pcell is None:
            raise RuntimeError(f"UE{self.ue_id} has no previous primary cell set")
        return self._prev_pcell

    @property
    def target_cell(self) -> int:
        """Current target cell index."""
        if self._target_cell is None:
            raise RuntimeError(f"UE{self.ue_id} has no target cell set")
        return self._target_cell

    @property
    def pp_detected(self) -> bool:
        """Whether a ping-pong handover was detected in the last handover."""
        return self._pp_detected


@dataclass(slots=True)
class A3Config:
    """Configuration for Event A3."""

    ttt_ms: int
    hys_db: float
    off_db: float


def a3_entry_condition(
    serving_dbm: float,
    neighbor_dbm: float,
    offset_db: float,
    hysteresis_db: float = 0.0,
) -> bool:
    """Strict A3 entry comparison, shared by native and offset-controlled HO."""
    return bool(neighbor_dbm - hysteresis_db > serving_dbm + offset_db)


class A3Action(Enum):
    """Represent the action selected by A3 evaluation.

    ``NONE`` means that no event was triggered. ``ENTER`` starts an A3
    condition, ``LEAVE`` ends it, and ``STAY`` retains its target. ``SWITCH``
    leaves the current condition and enters another, while ``RETARGET`` keeps
    the condition active but selects a different neighbor.
    """

    NONE = auto()
    ENTER = auto()
    STAY = auto()
    LEAVE = auto()
    SWITCH = auto()
    RETARGET = auto()


@dataclass(slots=True)
class A3Decision:
    """
    Result of A3 evaluation, indicating the action to take and the target cell
    if set.
    """

    action: A3Action = A3Action.NONE
    target_cell: int | None = None


class MeasurementHandler:
    """Store PHY measurements and evaluate the implemented A3 decision rule.

    Parameters
    ----------
    ue_id : int
        UE identifier used in validation errors.
    rrc_config : RRCConfig
        RRC thresholds and A3 configuration.

    Notes
    -----
    RSRP and SINR are stored L3-filtered (a 3GPP-style exponential moving
    average, see :func:`~mobile_network_sim.rrc.rrc_utils.l3_filtering`),
    applied per cell across successive :meth:`update_measurements` calls, not
    as raw instantaneous samples. This is what ``evaluate_a3``,
    ``get_best_cell_index``, and callers reading :attr:`rsrp_dbm`/
    :attr:`sinr_db` all observe. Achievable-capacity accounting must use
    :attr:`raw_sinr_db`/:meth:`RRCStepUE.get_pcell_raw_sinr`; L3 is a
    reporting filter, not a channel-rate filter. ``q_in``/``q_out`` are
    derived separately,
    from a plain moving average (not the L3 exponential filter) of *raw*
    SINR with independent Q-in/Q-out window lengths, per
    ``rrc_config.l1_q_in_filter_len_ms``/``l1_q_out_filter_len_ms`` -- unless
    explicitly overridden via :meth:`update_measurements`.

    The L1/L3 filter sample interval is fixed to ``rrc_config.t_res_ms``
    (one measurement per RRC step); ``rrc_config.l1_filter_sample_interval_ms``
    is not used, matching the reference preprocessing implementation, which
    only ever decouples the two once a PHY update cadence faster than the
    RRC step cadence exists.
    """

    def __init__(self, ue_id: int, rrc_config: RRCConfig) -> None:
        self.ue_id = ue_id
        self.q_in_db = rrc_config.q_in_db
        self.q_out_db = rrc_config.q_out_db

        self.a3_config = A3Config(
            ttt_ms=int(rrc_config.ho_event_config["a3"]["ttt_ms"]),
            hys_db=float(rrc_config.ho_event_config["a3"]["hys"]),
            off_db=float(rrc_config.ho_event_config["a3"]["offset"]),
        )

        self._l3_k = rrc_config.l3_filter_coef
        self._l3_t_ref_ms = rrc_config.l3_filter_sample_reference_interval_ms
        self._t_sample_ms = rrc_config.t_res_ms
        self._l3_rsrp_state: np.ndarray | None = None
        self._l3_sinr_state: np.ndarray | None = None
        qin_window = max(1, rrc_config.l1_q_in_filter_len_ms // rrc_config.t_res_ms)
        qout_window = max(1, rrc_config.l1_q_out_filter_len_ms // rrc_config.t_res_ms)
        self._l1_qin_filter = BoxcarFilter(qin_window)
        self._l1_qout_filter = BoxcarFilter(qout_window)

        self._rsrp: np.ndarray | None = None
        self._sinr: np.ndarray | None = None
        # The unfiltered sample, kept alongside the L3 state. Achievable
        # rate is a property of the channel, not a measurement report, so
        # it must be computed from this rather than from the smoothed
        # value used for mobility decisions -- see
        # analysis/report/analysis-v2.tex.
        self._raw_rsrp: np.ndarray | None = None
        self._raw_sinr: np.ndarray | None = None
        self._q_in: np.ndarray | None = None
        self._q_out: np.ndarray | None = None
        self._l1_sinr_qin: np.ndarray | None = None
        self._l1_sinr_qout: np.ndarray | None = None

    def update_measurements(
        self,
        rsrp_dbm: np.ndarray,
        sinr_db: np.ndarray,
        q_in: np.ndarray | None = None,
        q_out: np.ndarray | None = None,
    ) -> None:
        """Update measurements for the current sample.

        Parameters
        ----------
        rsrp_dbm : numpy.ndarray
            Per-cell raw (unfiltered) RSRP values in dBm for this sample.
            L3-filtered internally before use; see class docstring.
        sinr_db : numpy.ndarray
            Per-cell raw (unfiltered) SINR values in dB for this sample,
            with the same shape as ``rsrp_dbm``. L3-filtered internally
            before exposure via :attr:`sinr_db`; separately L1-filtered
            (unless ``q_in``/``q_out`` are given) to derive ``q_in``/``q_out``.
        q_in, q_out : numpy.ndarray or None, optional
            Optional per-cell Boolean synchronization indications, overriding
            this handler's own L1-filtered-SINR threshold comparison.

        Raises
        ------
        ValueError
            If the flattened inputs have inconsistent sizes or contain no
            cells.
        """
        rsrp_dbm = rsrp_dbm.flatten()
        sinr_db = sinr_db.flatten()
        q_in_arr = None if q_in is None else np.asarray(q_in).flatten()
        q_out_arr = None if q_out is None else np.asarray(q_out).flatten()
        n_bs = rsrp_dbm.shape[0]
        if rsrp_dbm.ndim != 1 or sinr_db.ndim != 1:
            raise ValueError(f"UE{self.ue_id}: rsrp_dbm and sinr_db must be 1D arrays")
        if not (rsrp_dbm.shape == sinr_db.shape and n_bs >= 1):
            raise ValueError(f"UE{self.ue_id}: Inconsistent input dimensions")
        if q_in_arr is not None and q_in_arr.shape != rsrp_dbm.shape:
            raise ValueError(f"UE{self.ue_id}: q_in shape must match measurements")
        if q_out_arr is not None and q_out_arr.shape != rsrp_dbm.shape:
            raise ValueError(f"UE{self.ue_id}: q_out shape must match measurements")

        if self._l3_rsrp_state is None:
            self._l3_rsrp_state = rsrp_dbm.astype(np.float64, copy=True)
            self._l3_sinr_state = sinr_db.astype(np.float64, copy=True)
        else:
            self._l3_rsrp_state = l3_filtering(
                self._l3_rsrp_state,
                rsrp_dbm,
                k=self._l3_k,
                t_sample=self._t_sample_ms,
                t_ref=self._l3_t_ref_ms,
            )
            self._l3_sinr_state = l3_filtering(
                self._l3_sinr_state,
                sinr_db,
                k=self._l3_k,
                t_sample=self._t_sample_ms,
                t_ref=self._l3_t_ref_ms,
            )
        self._rsrp = self._l3_rsrp_state
        self._sinr = self._l3_sinr_state
        self._raw_rsrp = rsrp_dbm
        self._raw_sinr = sinr_db

        # Retain both continuous L1 streams even when a caller supplies the
        # already-thresholded indicators.  Dataset producers need the exact
        # quantities Qin/Qout were compared against; recomputing them later
        # would make event diagnostics depend on a second filtering path.
        self._l1_sinr_qin = self._l1_qin_filter.step(sinr_db)
        self._l1_sinr_qout = self._l1_qout_filter.step(sinr_db)
        if q_in_arr is not None:
            self._q_in = q_in_arr.astype(bool, copy=False)
        else:
            self._q_in = self._l1_sinr_qin >= self.q_in_db
        if q_out_arr is not None:
            self._q_out = q_out_arr.astype(bool, copy=False)
        else:
            self._q_out = self._l1_sinr_qout <= self.q_out_db

    def evaluate_a3(self, pcell_idx: int, target_cell: int | None) -> A3Decision:
        """Evaluate the configured A3 decision rule.

        Parameters
        ----------
        pcell_idx : int
            Current primary-cell index.
        target_cell : int or None
            Current handover target, or ``None`` if no A3 event is active.

        Returns
        -------
        A3Decision
            Action and optional neighbor-cell target selected for this sample.

        Raises
        ------
        RuntimeError
            If PHY measurements have not been provided.
        """
        self._require_phy()
        if self.rsrp_dbm.size < 2:
            return A3Decision()

        neighbor_cell_idxs = np.delete(np.arange(self.rsrp_dbm.size), pcell_idx)
        rsrp_neighbor_cells = np.delete(self.rsrp_dbm, pcell_idx)
        best_ncell = int(neighbor_cell_idxs[np.argmax(rsrp_neighbor_cells)])

        mp = self.rsrp_dbm[pcell_idx]
        mc = self.rsrp_dbm[best_ncell]

        best_ncell_meets_a3_thold = a3_entry_condition(
            mp, mc, self.a3_config.off_db, self.a3_config.hys_db
        )

        if target_cell is None:
            # No current target cell, so only check for A3 entry
            if best_ncell_meets_a3_thold:
                # A3 entry condition met
                decision = A3Decision(A3Action.ENTER, target_cell=best_ncell)
            else:
                # No A3 event triggered
                decision = A3Decision(action=A3Action.NONE)
        else:
            # Current target cell exists, so check for A3 leave
            mn = self.rsrp_dbm[target_cell]
            still_valid = mn + self.a3_config.hys_db >= mp + self.a3_config.off_db
            if not still_valid:
                # A3 leave condition met
                if best_ncell_meets_a3_thold:
                    # Another cell meets A3 entry condition, so leave with new target
                    decision = A3Decision(
                        action=A3Action.SWITCH, target_cell=best_ncell
                    )
                else:
                    # No cell meets A3 entry condition, so leave without new target
                    decision = A3Decision(action=A3Action.LEAVE)
            else:
                if target_cell == best_ncell:
                    # Target cell is still the best cell, stay in A3 condition
                    decision = A3Decision(action=A3Action.STAY, target_cell=target_cell)
                else:
                    # Remains in A3 condition, possible change of best neighbor
                    decision = A3Decision(
                        action=A3Action.RETARGET, target_cell=best_ncell
                    )

        return decision

    def get_best_cell_index(self, msr_obj: str) -> int:
        """Return the cell with the largest selected measurement.

        Parameters
        ----------
        msr_obj : {``"rsrp"``, ``"sinr"``}
            Measurement used for cell selection.

        Returns
        -------
        int
            Index of the cell with the maximum measurement.

        Raises
        ------
        RuntimeError
            If PHY measurements have not been provided.
        ValueError
            If ``msr_obj`` is unsupported.
        """
        self._require_phy()
        if msr_obj == "rsrp":
            return int(np.argmax(self.rsrp_dbm))
        if msr_obj == "sinr":
            return int(np.argmax(self.sinr_db))
        raise ValueError(f"UE{self.ue_id}: Invalid measurement object")

    def _require_phy(self) -> None:
        if self._rsrp is None or self._sinr is None:
            raise RuntimeError(
                f"UE{self.ue_id}: RRC UE not initialized with PHY measurements"
            )

    @property
    def rsrp_dbm(self) -> np.ndarray:
        """Per-cell RSRP measurements for the current sample, in dBm."""
        if self._rsrp is None:
            raise RuntimeError(f"UE{self.ue_id}: Rsrp measurements not set")
        return self._rsrp

    @property
    def sinr_db(self) -> np.ndarray:
        """Per-cell SINR measurements for the current sample, in dB."""
        if self._sinr is None:
            raise RuntimeError(f"UE{self.ue_id}: Sinr measurements not set")
        return self._sinr

    @property
    def raw_rsrp_dbm(self) -> np.ndarray:
        """Per-cell RSRP of the current sample, before L3 filtering.

        Use this for anything that is a property of the channel rather
        than of the measurement procedure. :attr:`rsrp_dbm` is the
        L3-filtered value the A3 event is evaluated on, per
        TS 38.331 clause 5.5.3.2.
        """
        if self._raw_rsrp is None:
            raise RuntimeError(f"UE{self.ue_id}: Rsrp measurements not set")
        return self._raw_rsrp

    @property
    def raw_sinr_db(self) -> np.ndarray:
        """Per-cell SINR of the current sample, before L3 filtering.

        This is the quantity the achievable rate must be computed from:
        L3 filtering is defined for measurement reporting, not for the
        channel a transmission actually experiences.
        """
        if self._raw_sinr is None:
            raise RuntimeError(f"UE{self.ue_id}: Sinr measurements not set")
        return self._raw_sinr

    @property
    def q_in(self) -> np.ndarray:
        """Per-cell in-sync indications for the current sample.

        L1-filtered-SINR threshold comparison (or the caller-supplied
        override), computed once per :meth:`update_measurements` call -- see
        the class docstring.
        """
        if self._q_in is None:
            raise RuntimeError(f"UE{self.ue_id}: q_in not set")
        return self._q_in

    @property
    def q_out(self) -> np.ndarray:
        """Per-cell out-of-sync indications for the current sample.

        L1-filtered-SINR threshold comparison (or the caller-supplied
        override), computed once per :meth:`update_measurements` call -- see
        the class docstring.
        """
        if self._q_out is None:
            raise RuntimeError(f"UE{self.ue_id}: q_out not set")
        return self._q_out

    @property
    def l1_sinr_qin_db(self) -> np.ndarray:
        """Continuous per-cell L1 SINR compared with Qin."""
        if self._l1_sinr_qin is None:
            raise RuntimeError(f"UE{self.ue_id}: L1 Qin SINR not set")
        return self._l1_sinr_qin

    @property
    def l1_sinr_qout_db(self) -> np.ndarray:
        """Continuous per-cell L1 SINR compared with Qout."""
        if self._l1_sinr_qout is None:
            raise RuntimeError(f"UE{self.ue_id}: L1 Qout SINR not set")
        return self._l1_sinr_qout

"""Snapshot helpers for RRC UEs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from .flags import RRCFlags

if TYPE_CHECKING:
    from .ue import RRCStepUE


@dataclass(frozen=True, slots=True)
class SnapshotCore:
    """
    Core snapshot data for an RRC UE.

    Attributes
    ----------
    time_ms: int
        The current time in milliseconds.
    imsi: int
        The International Mobile Subscriber Identity (IMSI) of the UE
        (just used as a unique identifier in the simulation, not a real IMSI).
    state: int
        The current RRC state. See RRCState for possible values.
    in_sync: bool | None
        Whether the UE is in sync with the network. None if no serving cell is available.
    out_of_sync: bool | None
        Whether the UE is out of sync with the network. None if no serving cell is available.
    up_connected: bool
        Whether the user plane is available. It remains available in
        ``CONNECTED_NORMAL`` and ``CONNECTED_HO_PREP``.
    n310_count: int
        The current count of the N310 timer.
    n311_count: int
        The current count of the N311 timer.
    t310_remaining_ms: int | None
        The remaining time for the T310 timer in milliseconds; None if inactive.
    t_rlf_remaining_ms: int | None
        The remaining time for the (simulated) RLF recovery timer in ms, None if inactive.
    pcell: int | None
        The ID of the current primary cell (PCell) if available, None otherwise.
    prev_pcell: int | None
        The ID of the previous primary cell (PCell) if available, None otherwise.
    target_cell: int | None
        The ID of the target cell for handover if available, None otherwise.
    ttt_active: bool
        Whether the TTT timer is currently active.
    ho_prep_ongoing: bool
        Whether a handover preparation is currently ongoing.
    ho_exec_ongoing: bool
        Whether a handover execution is currently ongoing.
    t310_active: bool
        Whether the T310 timer is currently active.
    rlf_timer_active: bool
        Whether the RLF recovery timer is currently active.
    rlm_evaluated: bool
        Whether RLM updated N310/N311/T310 on this sample.
    """

    time_ms: int
    imsi: int
    state: int

    in_sync: bool | None
    out_of_sync: bool | None
    up_connected: bool

    n310_count: int
    n311_count: int

    t310_remaining_ms: int | None
    t_rlf_remaining_ms: int | None

    pcell: int | None
    prev_pcell: int | None
    target_cell: int | None

    ttt_active: bool
    ho_prep_ongoing: bool
    ho_exec_ongoing: bool
    t310_active: bool
    rlf_timer_active: bool
    rlm_evaluated: bool


class RRCUESnapshotBuilder:
    """Build snapshots and field lists for an RRC UE."""

    @staticmethod
    def _build_core(ue: RRCStepUE) -> SnapshotCore:
        return SnapshotCore(**RRCUESnapshotBuilder._core_values(ue))

    @staticmethod
    def _core_values(ue: RRCStepUE) -> dict[str, Any]:
        if not ue.initial_access_done:
            raise RuntimeError("Initial access not done, snapshot not available")

        return {
            "time_ms": ue.current_time_ms,
            "imsi": ue.ue_id,
            "state": int(ue.state),
            "in_sync": ue.in_sync,
            "out_of_sync": ue.out_of_sync,
            "up_connected": ue.up_connected,
            "n310_count": ue.n310_count,
            "n311_count": ue.n311_count,
            "t310_remaining_ms": ue.timers.t310.remaining_ms,
            "t_rlf_remaining_ms": ue.timers.rlf_recovery.remaining_ms,
            "pcell": ue.mob.get_pcell_if_available(),
            "prev_pcell": ue.mob.get_prev_pcell_if_available(),
            "target_cell": ue.mob.get_target_cell_if_available(),
            "ttt_active": ue.timers.ttt.active,
            "ho_prep_ongoing": ue.timers.prep.active,
            "ho_exec_ongoing": ue.timers.exec.active,
            "t310_active": ue.timers.t310.active,
            "rlf_timer_active": ue.timers.rlf_recovery.active,
            "rlm_evaluated": ue.rlm_evaluated_now,
        }

    @staticmethod
    def snapshot(ue: RRCStepUE) -> dict[str, Any]:
        """Build a snapshot of an RRC UE.

        Parameters
        ----------
        ue : RRCStepUE
            UE whose current RRC state is serialized.

        Returns
        -------
        dict of str to Any
            Core state and current event flags.

        Raises
        ------
        RuntimeError
            If initial access has not completed.
        """
        out = RRCUESnapshotBuilder._core_values(ue)

        for flag_name, key in _FLAG_FIELDS:
            out[key] = getattr(ue.flags_now, flag_name)

        return out

    @staticmethod
    def fields() -> list[str]:
        """Return the snapshot fields in serialization order.

        Returns
        -------
        list of str
            Core field names followed by ``flag_``-prefixed event fields.
        """
        return list(_SNAPSHOT_FIELDS)


_FLAG_FIELDS = tuple((name, f"flag_{name}") for name in RRCFlags.field_names())
_SNAPSHOT_FIELDS = tuple(field.name for field in fields(SnapshotCore)) + tuple(
    key for _, key in _FLAG_FIELDS
)

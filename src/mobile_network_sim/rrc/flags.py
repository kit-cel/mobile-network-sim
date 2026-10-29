"""RRC flag containers."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

#: Mutually exclusive causes of a link failure, in reporting order.
#:
#: Exactly one is set on every sample that starts RLF recovery. The three
#: ``hof_*`` entries are the handover-failure causes, split by the phase
#: the handover was in; ``rlf_no_ho`` is a link failure with no handover
#: involved at all.
LINK_FAILURE_CAUSES = ("rlf_no_ho", "hof_prep", "hof_cmd", "hof_exec")

#: Handover-failure causes, i.e. every link failure a handover caused.
HANDOVER_FAILURE_CAUSES = ("hof_prep", "hof_cmd", "hof_exec")


_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def _field_names(cls: type) -> tuple[str, ...]:
    """Cache schema names, including distinct dataclass subclasses."""
    names = _FIELD_NAMES.get(cls)
    if names is None:
        names = tuple(field.name for field in fields(cls))
        if len(_FIELD_NAMES) < 32:
            _FIELD_NAMES[cls] = names
    return names


@dataclass(slots=True)
class RRCFlags:
    """Per-sample RRC event flags."""

    rlf_start: int = 0
    rlf_expire: int = 0

    t310_start: int = 0
    t310_cancel: int = 0
    t310_expire: int = 0

    ttt_start: int = 0
    ttt_cancel: int = 0
    ttt_expire: int = 0

    prep_start: int = 0
    prep_cancel: int = 0
    prep_expire: int = 0

    exec_start: int = 0
    exec_cancel: int = 0
    exec_expire: int = 0

    ho_success: int = 0
    ho_failure: int = 0

    pp_detected: int = 0

    # Cause of the link failure that started RLF recovery. Exactly one is
    # set on any sample where ``rlf_start`` is set, and none otherwise --
    # see :meth:`link_failure_cause`.
    #
    # These exist because ``rlf_start`` alone conflates two very different
    # events: a link lost while simply connected, and a link lost *because*
    # a handover went wrong. Comparing a learned policy against A3 needs
    # them separated -- a policy that hands over too eagerly and one that
    # hands over too late both show up as "more RLFs" otherwise, while
    # their actual failure modes are opposites.
    #
    # Appended after the existing fields on purpose: :meth:`field_names`
    # order is the serialization order of :class:`SampleHistory`.
    rlf_no_ho: int = 0
    """T310 expired with no handover in progress: a 'pure' radio link failure."""

    hof_prep: int = 0
    """Failure during handover preparation, before the HO command is due."""

    hof_cmd: int = 0
    """Failure at HO command reception (preparation end / execution start)."""

    hof_exec: int = 0
    """Failure at execution end: the target cell was out of sync on arrival."""

    def reset(self) -> None:
        """Reset all flags to zero."""
        for name in _field_names(type(self)):
            setattr(self, name, 0)

    @classmethod
    def field_names(cls) -> list[str]:
        """Return the flag names in their serialized order.

        Returns
        -------
        list of str
            Dataclass field names in definition order.
        """
        return list(_field_names(cls))

    def to_array(self) -> np.ndarray:
        """Return the current flags as a one-dimensional array.

        Returns
        -------
        numpy.ndarray
            Integer flag array with shape ``(n_flags,)`` and dtype
            :class:`numpy.int8`.
        """
        return np.asarray(
            [getattr(self, name) for name in _field_names(type(self))],
            dtype=np.int8,
        )

    @property
    def ho_detected(self) -> bool:
        """Whether handover preparation started in the current sample."""
        return bool(self.prep_start)

    @property
    def hof_detected(self) -> bool:
        """Whether a handover failure occurred in the current sample."""
        return bool(self.ho_failure)

    @property
    def rlf_detected(self) -> bool:
        """Whether radio-link-failure handling started in the current sample."""
        return bool(self.rlf_start)


class SampleHistory:
    """Store sampled flags using the :class:`RRCFlags` schema.

    Attributes
    ----------
    _data : numpy.ndarray
        Internal array with shape ``(n_samples, n_flags)`` and dtype
        :class:`numpy.int8`.
    """

    def __init__(self) -> None:
        self._field_names = RRCFlags.field_names()
        self._field_to_idx = {name: idx for idx, name in enumerate(self._field_names)}
        self._data = np.zeros((0, len(self._field_names)), dtype=np.int8)

    def ensure_length(self, length: int) -> None:
        """Extend the history to contain at least ``length`` samples.

        Parameters
        ----------
        length : int
            Required number of sample rows. Existing rows are preserved.
        """
        if length <= self._data.shape[0]:
            return

        add_rows = length - self._data.shape[0]
        self._data = np.vstack(
            [
                self._data,
                np.zeros((add_rows, self._data.shape[1]), dtype=np.int8),
            ]
        )

    def write_current_sample(self, flags: RRCFlags) -> None:
        """Write flags to the last allocated sample row.

        Parameters
        ----------
        flags : RRCFlags
            Flag values for the current sample.

        Raises
        ------
        RuntimeError
            If no sample row has been allocated.
        """
        if self._data.shape[0] == 0:
            raise RuntimeError("Flag history is empty. Call ensure_length() first.")
        self._data[-1, :] = flags.to_array()

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a separate history array for each flag.

        Returns
        -------
        dict of str to numpy.ndarray
            Copies of the flag columns, each with shape ``(n_samples,)``.
        """
        return {
            name: self._data[:, idx].copy()
            for idx, name in enumerate(self._field_names)
        }

    def __getattr__(self, name: str) -> np.ndarray:
        """Allow attribute-style access, e.g. history.ho_success."""
        # ``copy.deepcopy`` probes special methods while reconstructing an
        # instance, before ``_field_to_idx`` has been restored.  Looking that
        # attribute up through ``self._field_to_idx`` recursively re-enters
        # ``__getattr__`` and eventually raises ``RecursionError``.  Reading
        # the partially constructed object's dictionary directly keeps normal
        # attribute access unchanged and makes complete RRC state snapshots
        # safely branchable for short local simulations.
        field_to_idx = self.__dict__.get("_field_to_idx")
        if field_to_idx is not None and name in field_to_idx:
            return self._data[:, field_to_idx[name]]
        raise AttributeError(name)

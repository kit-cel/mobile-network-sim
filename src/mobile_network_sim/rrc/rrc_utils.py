"""Utility functions for RRC (Radio Resource Control) operations in 5G simulations."""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

_L3_FILTER_COEFFICIENTS = frozenset((*range(10), 11, 13, 15, 17, 19))


@lru_cache(maxsize=128)
def _l3_weight(k: int, t_sample: int, t_ref: int) -> float:
    """Cache configuration-only arithmetic, never measurement state."""
    if k not in _L3_FILTER_COEFFICIENTS:
        raise ValueError("k must be one of 0-9, 11, 13, 15, 17, or 19.")
    a = 1 / (2 ** (k / 4))
    return 1 - (1 - a) ** (t_sample / t_ref)


def l3_filtering(
    f_old: NDArray[np.floating],
    m_new: NDArray[np.floating],
    k: int,
    t_sample: int,
    t_ref: int = 200,
) -> NDArray[np.floating]:
    r"""Apply Layer-3 measurement filtering.

    For a measurement :math:`M_n` and previous filtered value
    :math:`F_{n-1}`, the nominal filter is

    .. math::

       F_n = (1-a)F_{n-1} + aM_n, \qquad a=2^{-k/4}.

    When the simulator sampling interval :math:`T_s` differs from the
    reference interval :math:`T_{\mathrm{ref}}`, it uses

    .. math::

       a_s = 1-(1-a)^{T_s/T_{\mathrm{ref}}}

    so repeated updates over the same elapsed time have the same exponential
    decay.

    Parameters
    ----------
    f_old :  NDArray[np.floating]
        The previous filtered RSRP/SINR values.
    m_new :  NDArray[np.floating]
        The new RSRP/SINR measurements.
    k : int
        The L3 filter coefficient. Accepted values are 0 through 9, 11, 13,
        15, 17, and 19.
    t_sample : int
        The sampling period of the measurements in milliseconds.
    t_ref : int, optional
        Reference period in milliseconds. The default, 200 ms, is the E-UTRA
        reference period. For native NR measurements, pass the applicable
        period :math:`X` from 3GPP TS 38.133 as directed by TS 38.331.

    Returns
    -------
    NDArray[np.floating]
        The filtered RSRP/SINR values after applying L3 filtering.

    Raises
    ------
    ValueError
        If ``k`` is not a supported filter coefficient.

    Notes
    -----
    The accepted values correspond to the ``FilterCoefficient`` information
    element: 0 through 9, 11, 13, 15, 17, and 19. For ``k=0``, :math:`a=1`
    and no history is retained. Filtering is performed in the same domain as
    the reporting criterion, for example RSRP in dBm or RSRQ and SINR in dB.

    References
    ----------
    .. [1] ETSI, *5G; NR; Radio Resource Control (RRC); Protocol
       specification (3GPP TS 38.331 version 18.8.0 Release 18)*,
       ETSI TS 138 331 V18.8.0, clause 5.5.3.2 and the
       ``FilterCoefficient`` information element, February 2026.
       https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/
       18.08.00_60/ts_138331v180800p.pdf
    .. [2] ETSI, *LTE; Evolved Universal Terrestrial Radio Access (E-UTRA);
       Radio Resource Control (RRC); Protocol specification (3GPP TS 36.331
       version 18.8.0 Release 18)*, ETSI TS 136 331 V18.8.0,
       clause 5.5.3.2, February 2026.
       https://www.etsi.org/deliver/etsi_ts/136300_136399/136331/
       18.08.00_60/ts_136331v180800p.pdf
    """
    a_new = _l3_weight(k, t_sample, t_ref)
    return (1 - a_new) * f_old + a_new * m_new


def count_ping_pongs(
    pcell: NDArray[np.integer],
    ho_completed: NDArray[np.bool_] | NDArray[np.integer],
    t_mts_steps: int,
) -> int:
    """Count ping-pong handovers in a recorded single-UE trajectory.

    A ping-pong is a handover that returns to the cell the UE served
    *before* the previous handover (A -> B -> A), with the two handovers
    no more than ``t_mts_steps`` apart.

    This is the offline counterpart of
    :attr:`MobilityContext.pp_detected`, which applies the identical rule
    online during a live simulation. Both exist because some consumers
    have a running state machine to query and others only have stored
    arrays (a solved MILP assignment has no state machine at all). They
    must agree, so the rule lives here once instead of being written out
    at each call site -- a divergence between two copies of exactly this
    comparison is what previously made ping-pong detection report zero on
    every trajectory.

    Parameters
    ----------
    pcell : NDArray[np.integer]
        Serving-cell index per time step, shape ``(n_steps,)``.
    ho_completed : NDArray
        Per-step flag marking the steps at which a handover execution
        completed, shape ``(n_steps,)``. Truthy values are the events;
        RLF re-establishments must not be flagged here, since
        reattaching after a failure is not a handover.
    t_mts_steps : int
        Minimum-time-of-stay window in time steps. The comparison is
        inclusive (``<=``), matching the online rule.

    Returns
    -------
    int
        Number of ping-pong handovers.

    Raises
    ------
    ValueError
        If the inputs are not 1-D and of equal length, or if
        ``t_mts_steps`` is negative.
    """
    pcell = np.asarray(pcell)
    completions = np.asarray(ho_completed)
    if pcell.ndim != 1 or completions.ndim != 1:
        raise ValueError(
            "pcell and ho_completed must be 1-D single-UE arrays; got shapes "
            f"{pcell.shape} and {completions.shape}"
        )
    if pcell.shape != completions.shape:
        raise ValueError(
            f"pcell {pcell.shape} and ho_completed {completions.shape} must match"
        )
    if t_mts_steps < 0:
        raise ValueError("t_mts_steps must be non-negative")

    event_steps = np.flatnonzero(completions)
    num_pp = 0
    for first, second in zip(event_steps[:-1], event_steps[1:]):
        if second - first > t_mts_steps:
            continue
        # The cell held before the earlier handover completed is the
        # "two hops back" cell. A completion at step 0 has no such
        # predecessor, so it can never form a ping-pong pair.
        if first == 0:
            continue
        if pcell[first - 1] == pcell[second]:
            num_pp += 1
    return num_pp


class BoxcarFilter:
    """Fixed-length rolling arithmetic mean, used for L1 RLM filtering.

    Used to average SINR over a short window before comparing it against the
    Q-in/Q-out radio-link-monitoring thresholds, per the L1 measurement
    filtering described in TR 36.839. Unlike :func:`l3_filtering`'s
    exponential weighting, L1 filtering is a plain, equally-weighted moving
    average over the last ``window`` samples.

    Parameters
    ----------
    window : int
        Number of most recent samples to average over. Must be positive.
    """

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self._buf: NDArray[np.floating] | None = None
        self._sum: NDArray[np.floating] | None = None
        self._head = 0
        self._count = 0

    def step(self, value: NDArray[np.floating]) -> NDArray[np.floating]:
        """Push one new sample and return the updated moving average.

        Parameters
        ----------
        value : NDArray[np.floating]
            New sample, of arbitrary (but call-to-call consistent) shape.

        Returns
        -------
        NDArray[np.floating]
            Mean over the last ``min(count_seen, window)`` samples, i.e. the
            plain value itself on the very first call.
        """
        if self._buf is None:
            self._buf = np.zeros((self.window, *value.shape), dtype=value.dtype)
            self._sum = np.zeros_like(value)
        assert self._sum is not None
        if self._count >= self.window:
            self._sum -= self._buf[self._head]
        self._buf[self._head] = value
        self._sum += value
        self._head = (self._head + 1) % self.window
        self._count += 1
        return self._sum / min(self._count, self.window)

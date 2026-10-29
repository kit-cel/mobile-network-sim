"""Lightweight metrics for StepResult-based simulator evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NetworkMetrics:
    """Store aggregate network metrics for an episode.

    Attributes
    ----------
    ep_mean_rate : float
        Mean spectral-efficiency proxy over logged steps, in bit/s/Hz.
    ep_mean_connected_time : float
        Fraction of the episode treated as connected.
    ep_mean_sinr_db : float
        Mean selected-cell SINR over logged steps, in decibels.
    n_steps : int
        Number of radio steps included in the aggregate.
    """

    ep_mean_rate: float = 0.0
    ep_mean_connected_time: float = 1.0
    ep_mean_sinr_db: float = 0.0
    n_steps: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert the metrics to a dictionary.

        Returns
        -------
        dict of str to float or int
            Mapping from dataclass field name to its current value.
        """
        return asdict(self)

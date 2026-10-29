"""StepResult loggers compatible with PPO evaluation code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .network_metrics import NetworkMetrics
from .result_loggers import HDF5ResultsLogger as _StepHDF5ResultsLogger

if TYPE_CHECKING:
    from ..simulation.simulation_env import RadioEnvironment, RadioStepResult


class BaseResultsLogger(ABC):
    """Base class for logging StepResult samples."""

    @abstractmethod
    def new_episode(self) -> None:
        """Reset any episode-specific state at the start of a new episode."""

    @abstractmethod
    def log_step(self, step_result: "RadioStepResult") -> None:
        """Log a single radio-step sample.

        Parameters
        ----------
        step_result : RadioStepResult
            Result produced by one environment step.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release any resources."""


class DefaultResultsLogger(BaseResultsLogger):
    """Aggregate basic radio metrics from radio-step results.

    Parameters
    ----------
    env : RadioEnvironment
        Radio environment associated with the logged run. The compatibility
        logger does not currently inspect the environment.
    n_ep : int
        Number of episodes expected by the caller. The compatibility logger
        aggregates one episode at a time and does not retain this value.

    Attributes
    ----------
    metrics : NetworkMetrics
        Running aggregate for the current episode.
    """

    def __init__(self, env: "RadioEnvironment", n_ep: int) -> None:
        del env, n_ep
        self.metrics = NetworkMetrics()
        self._sum_rate = 0.0
        self._sum_sinr = 0.0
        self._n = 0

    def new_episode(self) -> None:
        """Reset all aggregates for a new episode."""
        self._sum_rate = 0.0
        self._sum_sinr = 0.0
        self._n = 0
        self.metrics = NetworkMetrics()

    def log_step(self, step_result: "RadioStepResult") -> None:
        """Include one radio step in the current episode aggregate.

        Parameters
        ----------
        step_result : RadioStepResult
            Per-cell radio measurements for one simulation step. The logger
            selects the highest SINR for each UE before averaging.
        """
        best_sinr = np.max(np.asarray(step_result.sinr_db, dtype=float), axis=-1)
        rate = np.log2(1.0 + np.power(10.0, best_sinr / 10.0))
        self._sum_rate += float(np.mean(rate))
        self._sum_sinr += float(np.mean(best_sinr))
        self._n += 1
        self.metrics = NetworkMetrics(
            ep_mean_rate=self._sum_rate / self._n,
            ep_mean_connected_time=1.0,
            ep_mean_sinr_db=self._sum_sinr / self._n,
            n_steps=self._n,
        )

    def close(self) -> None:
        """Close the logger.

        Notes
        -----
        This in-memory logger owns no external resources, so closing it has no
        effect.
        """


class HDF5ResultsLogger(_StepHDF5ResultsLogger):
    """Compatibility wrapper for the radio-step HDF5 logger.

    Parameters
    ----------
    env : RadioEnvironment
        Radio environment whose result schema is logged.
    log_path : str
        Output HDF5 file path.
    n_ep : int
        Number of episodes allocated in the output file.
    """

    def __init__(self, env: "RadioEnvironment", log_path: str, n_ep: int) -> None:
        n_steps = int(getattr(env.config, "num_steps", 1000))
        super().__init__(env=env, log_path=log_path, n_ep=n_ep, n_steps=n_steps)

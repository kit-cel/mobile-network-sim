"""Network simulation module."""

import os
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from mobile_network_sim.seeding import RadioEpisodeSeeds
from mobile_network_sim.seeding import episode_seed as derive_episode_seed

from ..result_handler.result_loggers import HDF5ResultsLogger
from .simulation_env import RadioEnvironment

if TYPE_CHECKING:
    from ..config.simulation_config import SimulationConfig
    from ..lsf.lsf_map import CoverageMaps
    from ..result_handler.result_loggers import BaseResultsLogger


class RadioSimulationRunner:
    """Run episodes using a configured radio simulation.

    Parameters
    ----------
    config : SimulationConfig
        Complete radio simulation configuration.

    Attributes
    ----------
    config : SimulationConfig
        Active simulation configuration.
    env : RadioEnvironment
        Environment advanced by :meth:`run`.
    lsf_maps : CoverageMaps or None
        Precomputed coverage maps when map creation is enabled.

    Notes
    -----
    Each episode resets the environment. When result persistence is enabled,
    samples are written through an HDF5 results logger and the resulting path
    is returned.
    """

    _simulation_id: str
    n_ep: int
    n_steps: int

    def __init__(self, config: "SimulationConfig") -> None:
        """Initialize the simulation environment."""
        self.config = config
        self.t_res_ms = config.time_resolution_ms
        self.storage_dtype = np.dtype(config.dtypes.storage)
        self.internal_dtype = np.dtype(config.dtypes.internal)
        self.map_dtype = np.dtype(config.dtypes.maps)

        # Network environment
        self.env = RadioEnvironment(
            config.radio,
            t_res_ms=self.t_res_ms,
            internal_dtype=self.internal_dtype,
            map_dtype=self.map_dtype,
            beam_dtype=config.dtypes.beam,
            grf_dtype=config.dtypes.grf,
            realize_lsf=False,
        )

        # A coverage map belongs to a concrete episode realization. It is
        # therefore created after that episode's reset in ``run()``, not here
        # against a different initialization seed.
        self.lsf_maps: CoverageMaps | None = None

    def episode_seed(self, episode_index: int) -> int:
        """Return the reset seed for one episode of this run.

        Episodes must differ from each other but the *run* must be
        reproducible from :attr:`SimulationConfig.seed` alone, so seeds
        are derived rather than either shared or drawn fresh.

        Derivation goes through
        :func:`mobile_network_sim.seeding.episode_seed`, which mixes a
        structured path rather than offsetting an integer. That matters
        because each UE's own seed is derived from this one: an
        arithmetic scheme made neighbouring episodes share UE streams,
        and the resulting duplicate trajectories were invisible to a
        per-episode uniqueness check.

        Parameters
        ----------
        episode_index : int
            Zero-based episode index within this run.

        Returns
        -------
        int
            Seed for :meth:`RadioEnvironment.reset`.
        """
        return derive_episode_seed(self.config.seed, episode_index)

    def run(
        self,
        n_ep: int,
        n_steps: int,
        save_results: bool = False,
        output_dir: str | os.PathLike[str] | None = None,
        episode_seeds: Sequence[RadioEpisodeSeeds] | None = None,
    ) -> str | None:
        """Run one or more simulation episodes.

        The run is reproducible: episode ``i`` is reset with
        :meth:`episode_seed`, derived from ``config.seed``. Repeating a
        run with the same configuration reproduces the same trajectories.

        Parameters
        ----------
        n_ep : int
            Number of episodes.
        n_steps : int
            Number of simulation steps per episode.
        save_results : bool, optional
            Write step results to HDF5 when ``True``.
        output_dir : str or os.PathLike, optional
            Destination directory for persisted results. If omitted, results
            are written below ``outputs/results/h5``.
        episode_seeds : sequence of RadioEpisodeSeeds or None, optional
            Fully resolved component streams. Dataset generation supplies
            these to execute exactly the seeds written to provenance. Normal
            multi-episode runs derive one plan per episode from ``config.seed``.

        Returns
        -------
        str or None
            HDF5 path when ``save_results`` is enabled, otherwise ``None``.
        """
        self._simulation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")  # noqa: DTZ005 - legacy local-time run ID
        self.n_ep = n_ep
        self.n_steps = n_steps
        if episode_seeds is not None and len(episode_seeds) != n_ep:
            raise ValueError(
                "episode_seeds must contain exactly one plan per episode "
                f"({len(episode_seeds)} != {n_ep})"
            )

        pbar_step = 0
        pbar_update_interval = 1_000 if n_ep * n_steps >= 10_000 else 10
        pbar = tqdm(
            total=n_ep * n_steps,
            desc="Evaluating",
            # disable=not show_pbar,
            unit_scale=True,
            unit="step",
            smoothing=0.1,
            ncols=100,
            dynamic_ncols=True,
        )

        loggers: list[BaseResultsLogger] = []
        if save_results:
            result_dir = (
                os.fspath(output_dir)
                if output_dir is not None
                else os.path.join("outputs", "results", "h5")
            )
            log_path = os.path.join(
                result_dir,
                f"sim_seed{self.config.seed}_n_ue{self.config.radio.num_ues}_"
                f"{self._simulation_id}_3gpp_results.h5",
            )
            loggers.append(
                HDF5ResultsLogger(
                    self.env,
                    log_path,
                    n_ep,
                    n_steps,
                    storage_dtype=self.storage_dtype,
                )
            )
        else:
            log_path = None

        for i in range(n_ep):  #  Episode loop
            pbar.set_postfix(ep=f"{i + 1}/{n_ep}", refresh=False)
            seed_plan = (
                episode_seeds[i]
                if episode_seeds is not None
                else RadioEpisodeSeeds.from_root(self.episode_seed(i))
            )
            self.env.reset(seed=seed_plan)
            self.lsf_maps = (
                self.env.get_lsf_maps()
                if self.config.radio.channel.create_lsf_maps
                else None
            )
            for logger in loggers:
                logger.new_episode()

            for _ in range(n_steps):  #  Step loop
                step_result = self.env.step()
                for logger in loggers:
                    logger.log_step(step_result)
                pbar_step += 1
                if pbar_step % pbar_update_interval == 0:
                    pbar.update(pbar_update_interval)

        remaining = pbar_step % pbar_update_interval
        if remaining > 0:
            pbar.update(remaining)
        pbar.close()
        for logger in loggers:
            logger.close()

        return log_path

    @property
    def simulation_id(self) -> str:
        """Get the unique simulation ID."""
        if not hasattr(self, "_simulation_id"):
            raise AttributeError("Simulation ID is not set. Run the simulation first.")
        return self._simulation_id

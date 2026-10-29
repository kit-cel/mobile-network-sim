"""Runner that composes radio measurements with the step-based RRC model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ..radio import RadioEnvironment, RadioStepResult, SimulationConfig
from ..radio.simulation.sample_source import RadioSampleSource
from ..rrc import RRCConfig, RRCStepUE
from ..seeding import RadioEpisodeSeeds

ActionMode = Literal["binary_report", "target_cell"]


class RadioRRCSimulationConfig(SimulationConfig):
    """Configure a synchronized radio and RRC simulation.

    Attributes
    ----------
    rrc : RRCConfig
        RRC state-machine and timer configuration. Its ``t_res_ms`` value
        must equal :attr:`SimulationConfig.time_resolution_ms`.
    """

    rrc: RRCConfig
    radio_source_mode: Literal["block", "thread"] = "thread"
    radio_source_capacity: int = Field(
        default=1500,
        ge=1,
        le=1500,
        description="Maximum buffered physical samples; bounded by num_steps at runtime.",
    )

    @model_validator(mode="after")
    def _validate_time_grid(self) -> RadioRRCSimulationConfig:
        if self.time_resolution_ms != self.rrc.t_res_ms:
            raise ValueError(
                "radio time_resolution_ms must equal rrc.t_res_ms "
                f"({self.time_resolution_ms} != {self.rrc.t_res_ms})"
            )
        return self


@dataclass(frozen=True, slots=True)
class RadioRRCStepResult:
    """Store one synchronized radio and RRC sample.

    Attributes
    ----------
    radio : RadioStepResult
        Radio measurements produced for the current step.
    states : numpy.ndarray
        Integer RRC state for each UE, with shape ``(n_ues,)``.
    serving_cells : numpy.ndarray
        Serving-cell index for each UE, with shape ``(n_ues,)``. A value of
        ``-1`` indicates that no serving cell is available.
    flags : numpy.ndarray
        RRC event flags stacked along the UE axis.
    snapshots : tuple of dict
        Per-UE state snapshots after processing the current measurements.
    """

    radio: RadioStepResult
    states: np.ndarray
    serving_cells: np.ndarray
    flags: np.ndarray
    snapshots: tuple[dict[str, object], ...]


class RadioRRCSimulationRunner:
    """Advance radio and one RRC state machine per UE in lockstep.

    Parameters
    ----------
    config : RadioRRCSimulationConfig
        Combined radio and RRC configuration.
    action_mode : {"binary_report", "target_cell"}, optional
        External-control interpretation. If omitted, the runner operates
        autonomously. Controlled runners require one action per UE on every
        call to :meth:`step`.
    auto_reset : bool, optional
        Reset immediately with ``config.seed``. The default preserves direct
        runner and CLI behavior. Set to ``False`` when the caller will supply
        the actual episode seed through an explicit :meth:`reset`.

    Attributes
    ----------
    config : RadioRRCSimulationConfig
        Active simulation configuration.
    env : RadioEnvironment
        Radio environment advanced before the RRC state machines.
    rrc_ues : list of RRCStepUE
        RRC state machines in UE-index order.

    Notes
    -----
    The radio result for a step is calculated first. Its RSRP and SINR rows are
    then passed to the corresponding RRC UE at the current discrete simulation
    time.
    """

    def __init__(
        self,
        config: RadioRRCSimulationConfig,
        *,
        action_mode: ActionMode | None = None,
        auto_reset: bool = True,
        static_resources=None,
    ) -> None:
        self.config = config
        self.action_mode = action_mode
        self.env = RadioEnvironment(
            config.radio,
            t_res_ms=config.time_resolution_ms,
            internal_dtype=config.dtypes.internal,
            map_dtype=config.dtypes.maps,
            beam_dtype=config.dtypes.beam,
            grf_dtype=config.dtypes.grf,
            realize_lsf=False,
            static_resources=static_resources,
        )
        self._source = RadioSampleSource(
            self.env,
            mode=config.radio_source_mode,
            capacity=min(config.radio_source_capacity, config.num_steps),
            horizon=config.num_steps,
        )
        self._step_index = 0
        self._pending_radio: RadioStepResult | None = None
        self.rrc_ues: list[RRCStepUE] = []
        self._is_reset = False
        if auto_reset:
            self.reset(seed=config.seed)

    def reset(
        self, seed: int | RadioEpisodeSeeds | None = None, **radio_reset_options
    ) -> None:
        """Reset radio state, time, and all RRC state machines.

        Parameters
        ----------
        seed : int, RadioEpisodeSeeds, or None, optional
            Episode root or resolved component streams passed to the radio
            environment. If omitted, the configured simulation seed is used.

        Additional reset options are forwarded after joining the producer, so
        UE replacement and optional spatial-field reuse cannot race with it.
        """
        self._is_reset = False
        self._source.reset(
            self.config.seed if seed is None else seed, **radio_reset_options
        )
        self._step_index = 0
        self._pending_radio = None
        self.rrc_ues = [
            RRCStepUE(ue_id=index, rrc_config=self.config.rrc)
            for index in range(self.config.radio.num_ues)
        ]
        if self.action_mode is not None:
            for ue in self.rrc_ues:
                ue.set_learning_mode(enable_agent_mode=True, enable_eval=True)
        self._is_reset = True

    def step(
        self, actions: Sequence[int] | np.ndarray | None = None
    ) -> RadioRRCStepResult:
        """Advance the synchronized simulation by one step.

        Parameters
        ----------
        actions : sequence of int or numpy.ndarray, optional
            One control action per UE. Actions are forbidden in autonomous
            mode and required when ``action_mode`` is configured.

        Returns
        -------
        RadioRRCStepResult
            Radio measurements and RRC state recorded after the step.

        Raises
        ------
        ValueError
            If actions are inconsistent with the configured control mode or
            their count does not equal the number of UEs.
        """
        self._validate_actions(actions)
        if self._pending_radio is not None:
            raise RuntimeError("finish_step must consume the pending sample first")
        radio_result = self._source.step()
        return self._consume_radio_result(radio_result, actions)

    def prepare_step(self) -> RadioRRCStepResult:
        """Expose current measurements and timer outcomes before any decision."""
        if not self._is_reset or self._pending_radio is not None:
            raise RuntimeError("prepare_step requires reset and no pending sample")
        radio_result = self._source.step()
        time_ms = self._step_index * self.config.time_resolution_ms
        for index, ue in enumerate(self.rrc_ues):
            ue.prepare_sample(
                time_ms=time_ms,
                rsrp_dbm=np.asarray(radio_result.rsrp_dbm[index]),
                sinr_db=np.asarray(radio_result.sinr_db[index]),
            )
        self._pending_radio = radio_result
        return self._result(radio_result)

    def finish_step(self, actions=None) -> RadioRRCStepResult:
        """Complete a prepared sample without drawing another radio sample."""
        self._validate_actions(actions)
        if self._pending_radio is None:
            raise RuntimeError("prepare_step must precede finish_step")
        for index, ue in enumerate(self.rrc_ues):
            ue.finish_sample(
                action=None if actions is None else int(actions[index]),
                action_mode=self.action_mode or "binary_report",
            )
        radio_result = self._pending_radio
        self._pending_radio = None
        self._step_index += 1
        return self._result(radio_result)

    def _validate_actions(self, actions):
        if not self._is_reset:
            raise RuntimeError(
                "RadioRRCSimulationRunner.reset() must be called before step() "
                "when auto_reset=False"
            )
        if self.action_mode is None and actions is not None:
            raise ValueError("actions require an action_mode")
        if self.action_mode is not None and actions is None:
            raise ValueError("actions are required when action_mode is configured")
        if actions is not None and len(actions) != len(self.rrc_ues):
            raise ValueError("one action is required per UE")

    def _consume_radio_result(self, radio_result, actions):
        """Advance only RRC after consuming one ordered, already computed sample."""
        self._validate_actions(actions)
        if self._pending_radio is not None:
            raise RuntimeError("finish_step must consume the pending sample first")
        time_ms = self._step_index * self.config.time_resolution_ms
        for index, ue in enumerate(self.rrc_ues):
            ue.step(
                time_ms=time_ms,
                rsrp_dbm=np.asarray(radio_result.rsrp_dbm[index]),
                sinr_db=np.asarray(radio_result.sinr_db[index]),
                action=None if actions is None else int(actions[index]),
                action_mode=self.action_mode or "binary_report",
            )
        self._step_index += 1

        return self._result(radio_result)

    def _result(self, radio_result) -> RadioRRCStepResult:
        """Snapshot either side of the decision without changing simulator state."""

        serving_cells = np.asarray(
            [
                (
                    -1
                    if ue.mob.get_pcell_if_available() is None
                    else ue.mob.get_pcell_if_available()
                )
                for ue in self.rrc_ues
            ],
            dtype=np.int32,
        )
        return RadioRRCStepResult(
            radio=radio_result,
            states=np.asarray([int(ue.state) for ue in self.rrc_ues], dtype=np.int8),
            serving_cells=serving_cells,
            flags=np.stack([ue.flags_now.to_array() for ue in self.rrc_ues]),
            snapshots=tuple(ue.snapshot() for ue in self.rrc_ues),
        )

    def pause_radio(self) -> None:
        """Park radio prefetch without resetting the episode."""
        self._source.pause()

    def resume_radio(self) -> None:
        """Resume an evaluation-paused radio producer."""
        self._source.resume()

    def close(self) -> None:
        """Join the optional radio producer and release buffered samples."""
        self._source.close()
        if self.config.radio.resource_storage == "shared_mmap":
            for model in (self.env.lsf_model.sfm, self.env.lsf_model.los_model):
                if model is not None and hasattr(model, "release_shared_resources"):
                    model.release_shared_resources()

    def run(self, num_steps: int | None = None) -> list[RadioRRCStepResult]:
        """Run an autonomous simulation for a fixed number of steps.

        Parameters
        ----------
        num_steps : int, optional
            Number of steps to execute. The configured value is used when this
            argument is omitted.

        Returns
        -------
        list of RadioRRCStepResult
            Step results in chronological order.

        Raises
        ------
        ValueError
            If the runner is externally controlled or ``num_steps`` is not
            positive.
        """
        if self.action_mode is not None:
            raise ValueError(
                "run() is autonomous; controlled simulations must call step(actions)"
            )
        steps = self.config.num_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        return [self.step() for _ in range(steps)]

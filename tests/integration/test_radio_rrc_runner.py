from dataclasses import fields

import numpy as np
import pytest
from pydantic import ValidationError

from mobile_network_sim.runners import (
    RadioRRCSimulationConfig,
    RadioRRCSimulationRunner,
)


def test_combined_runner_aligns_ues_and_measurements() -> None:
    config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml")
    runner = RadioRRCSimulationRunner(config)
    result = runner.step()

    assert result.radio.rsrp_dbm.shape[0] == config.radio.num_ues
    assert result.states.shape == (config.radio.num_ues,)
    assert result.serving_cells.shape == (config.radio.num_ues,)
    assert result.flags.shape[0] == config.radio.num_ues
    assert np.all(result.serving_cells >= 0)


def test_controlled_runner_requires_one_action_per_ue() -> None:
    config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml")
    runner = RadioRRCSimulationRunner(config, action_mode="target_cell")
    with pytest.raises(ValueError, match="one action"):
        runner.step([0])


def test_lazy_runner_requires_reset_and_exposes_static_topology() -> None:
    config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml")
    runner = RadioRRCSimulationRunner(config, auto_reset=False)

    assert runner.rrc_ues == []
    assert runner.env.bs_set.n_cells > 0
    with pytest.raises(RuntimeError, match=r"reset\(\).*before step"):
        runner.step()

    runner.reset(seed=123)
    result = runner.step()
    assert result.radio.rsrp_dbm.shape == (
        config.radio.num_ues,
        runner.env.bs_set.n_cells,
    )


def test_eager_and_lazy_runner_are_bit_identical_after_same_reset() -> None:
    config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml")
    channel = config.radio.channel.model_copy(
        update={
            "los_model": "UMaLOSModelSoft",
            "soft_los_map_grid_spacing_m": 100.0,
            "shadowing_enabled": True,
            "shadowing_map_grid_spacing_m": 100.0,
        }
    )
    radio = config.radio.model_copy(update={"channel": channel})
    config = config.model_copy(update={"radio": radio})
    eager = RadioRRCSimulationRunner(config, action_mode="target_cell")
    lazy = RadioRRCSimulationRunner(
        config,
        action_mode="target_cell",
        auto_reset=False,
    )
    eager.reset(seed=456)
    lazy.reset(seed=456)

    action_rows = ([0, 0], [1, 1], [0, 0], [2, 2], [0, 0])
    for actions in action_rows:
        first = eager.step(actions)
        second = lazy.step(actions)
        for field in fields(type(first.radio)):
            np.testing.assert_array_equal(
                getattr(first.radio, field.name),
                getattr(second.radio, field.name),
            )
        np.testing.assert_array_equal(first.states, second.states)
        np.testing.assert_array_equal(first.serving_cells, second.serving_cells)
        np.testing.assert_array_equal(first.flags, second.flags)
        assert first.snapshots == second.snapshots


def test_combined_config_rejects_different_time_grids() -> None:
    config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml").to_dict()
    config["rrc"]["t_res_ms"] = 20
    config["rrc"]["phy_update_interval_ms"] = 20
    config["rrc"]["msr_interval_ms"] = 20
    with pytest.raises(ValidationError, match="must equal"):
        RadioRRCSimulationConfig.model_validate(config)

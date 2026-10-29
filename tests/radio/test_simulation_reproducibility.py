"""Regression tests for reproducible episode generation.

``RadioSimulationRunner.run`` used to reset each episode without a seed,
discarding the seeded reset done just before the loop. Two runs of the
same configuration therefore produced entirely different trajectories,
while the config, the recorded seed, and every metric looked perfectly
normal -- so nothing downstream could notice.

That matters beyond tidiness: a released dataset's central claim is that
its episodes can be regenerated from its recorded seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from mobile_network_sim.radio.config import SimulationConfig
from mobile_network_sim.radio.simulation import RadioSimulationRunner

N_STEPS = 12


@pytest.fixture(scope="module")
def config() -> SimulationConfig:
    """A small but real radio configuration.

    Loaded from the shipped example rather than hand-built so the test
    exercises a configuration shape the project actually uses.
    """
    payload = yaml.safe_load(open("examples/radio_rrc.yaml", encoding="utf-8"))
    payload.pop("rrc", None)
    payload["num_steps"] = N_STEPS
    payload["seed"] = 1234
    payload["radio"]["num_ues"] = 3
    return SimulationConfig.model_validate(payload)


def _run_positions(config: SimulationConfig, n_ep: int = 1) -> np.ndarray:
    """Run and return the UE positions of every step, without file I/O."""
    runner = RadioSimulationRunner(config)
    positions: list[np.ndarray] = []
    runner.env.reset(seed=config.seed)
    for episode_index in range(n_ep):
        runner.env.reset(seed=runner.episode_seed(episode_index))
        for _ in range(N_STEPS):
            runner.env.step()
            positions.append(np.array(runner.env.ue_set.xy, dtype=float))
    return np.stack(positions)


def test_episodes_never_share_a_ue_mobility_stream(
    config: SimulationConfig,
) -> None:
    """The property that actually matters, checked on realized seeds.

    Episode seeds feed a per-UE derivation, so two episodes sharing even
    one UE seed means two UEs replay identical trajectories. The old
    arithmetic derivation made that the *normal* case for neighbouring
    episodes while a per-episode uniqueness check still passed.
    """
    from mobile_network_sim.seeding import ue_mobility_seed

    runner = RadioSimulationRunner(config)
    n_ue = config.radio.num_ues
    realized: set[int] = set()
    for episode_index in range(8):
        base = runner.episode_seed(episode_index)
        seeds = {ue_mobility_seed(base, ue) for ue in range(n_ue)}
        assert len(seeds) == n_ue, "UEs within an episode share a stream"
        assert not (seeds & realized), (
            f"episode {episode_index} reuses an earlier episode's UE stream"
        )
        realized |= seeds


def test_neighbouring_base_seeds_do_not_alias(config: SimulationConfig) -> None:
    """The exact failure the arithmetic derivation produced.

    Under ``base + index + 1``, base seeds 100 and 101 shared all but one
    UE stream. The structured derivation must share none.
    """
    from mobile_network_sim.seeding import ue_mobility_seed

    n_ue = 25
    first = {ue_mobility_seed(100, ue) for ue in range(n_ue)}
    second = {ue_mobility_seed(101, ue) for ue in range(n_ue)}
    assert not (first & second)

    arithmetic_first = {100 + ue + 1 for ue in range(n_ue)}
    arithmetic_second = {101 + ue + 1 for ue in range(n_ue)}
    assert len(arithmetic_first & arithmetic_second) == n_ue - 1


def test_episode_seed_stays_in_the_32_bit_range(config: SimulationConfig) -> None:
    """Late episodes of a long run must not overflow the seed domain."""
    high_seed = config.model_copy(update={"seed": 2**32 - 10})
    runner = RadioSimulationRunner(high_seed)
    for index in range(5):
        assert 0 <= runner.episode_seed(index) < 2**32


def test_two_runs_of_the_same_config_produce_identical_trajectories(
    config: SimulationConfig,
) -> None:
    """The regression itself."""
    first = _run_positions(config)
    second = _run_positions(config)

    assert first.shape == second.shape
    assert np.array_equal(first, second), (
        "identical configuration produced different UE trajectories; episode "
        "reset is not seeded"
    )


def test_a_different_seed_produces_different_trajectories(
    config: SimulationConfig,
) -> None:
    """Reproducibility must not have been achieved by ignoring the seed."""
    first = _run_positions(config)
    second = _run_positions(config.model_copy(update={"seed": config.seed + 99_999}))

    assert not np.array_equal(first, second)


def test_successive_episodes_of_one_run_differ(config: SimulationConfig) -> None:
    """Episodes are still varied; only the run as a whole is deterministic."""
    positions = _run_positions(config, n_ep=2)
    first_episode = positions[:N_STEPS]
    second_episode = positions[N_STEPS:]

    assert not np.array_equal(first_episode, second_episode)

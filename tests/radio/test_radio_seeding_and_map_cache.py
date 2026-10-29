"""Episode-level radio seeding and cache-correctness regressions."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from mobile_network_sim.radio.config import SimulationConfig
from mobile_network_sim.radio.config.layout_config import NetworkLayoutConfig
from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField
from mobile_network_sim.radio.lsf.los_models.uma_soft_los import UMaLOSModelSoft
from mobile_network_sim.radio.lsf.lsf_map import CoverageMaps, MapGenerator
from mobile_network_sim.radio.lsf.shadowing.sfm_cholesky import (
    ShadowFadingCholesky,
)
from mobile_network_sim.radio.geometry.simulation_geometry import SimulationGeometry
from mobile_network_sim.radio.simulation.simulation_env import (
    RadioEnvironment,
    RadioStepResult,
)
from mobile_network_sim.radio.simulation import RadioSimulationRunner
from mobile_network_sim.seeding import RadioEpisodeSeeds


def _config(*, soft_fields: bool) -> SimulationConfig:
    payload = yaml.safe_load(open("examples/radio_rrc.yaml", encoding="utf-8"))
    payload.pop("rrc", None)
    channel = payload["radio"]["channel"]
    channel.update(
        {
            "los_model": "UMaLOSModelSoft" if soft_fields else "UMaLOSModel",
            "soft_los_map_grid_spacing_m": 100.0,
            "soft_los_d_decorr_m": 50.0,
            "shadowing_enabled": soft_fields,
            "shadowing_map_grid_spacing_m": 100.0,
            "shadowing_d_decorr_m": 50.0,
            "create_lsf_maps": True,
            "lsf_maps_grid_res_m": 100.0,
        }
    )
    if not soft_fields:
        payload["radio"]["ues"]["height_m"] = 15.0
    return SimulationConfig.model_validate(payload)


def _field_realizations(env: RadioEnvironment) -> tuple[np.ndarray, np.ndarray]:
    los = np.asarray(env.lsf_model.los_model.los_grf).copy()
    shadow = np.asarray(env.lsf_model.sfm.shadowing_z_maps).copy()
    return los, shadow


def _assert_step_equal(first: RadioStepResult, second: RadioStepResult) -> None:
    for field in fields(RadioStepResult):
        left = np.asarray(getattr(first, field.name))
        right = np.asarray(getattr(second, field.name))
        assert np.array_equal(left, right, equal_nan=True), field.name


def test_deferred_environment_realizes_fields_once_per_explicit_reset(
    monkeypatch,
) -> None:
    config = _config(soft_fields=True)
    calls = {"los": 0, "shadowing": 0}
    original_los = UMaLOSModelSoft.generate_los_maps
    original_shadowing = ShadowFadingCholesky.generate_shadow_fading_maps

    def generate_los(model):
        calls["los"] += 1
        return original_los(model)

    def generate_shadowing(model):
        calls["shadowing"] += 1
        return original_shadowing(model)

    monkeypatch.setattr(UMaLOSModelSoft, "generate_los_maps", generate_los)
    monkeypatch.setattr(
        ShadowFadingCholesky,
        "generate_shadow_fading_maps",
        generate_shadowing,
    )

    env = RadioEnvironment(
        config.radio,
        config.time_resolution_ms,
        realize_lsf=False,
    )
    assert calls == {"los": 0, "shadowing": 0}
    assert env.bs_set.n_cells > 0
    assert env.get_dummy_step_result().rsrp_dbm.shape == (
        config.radio.num_ues,
        env.bs_set.n_cells,
    )
    assert calls == {"los": 0, "shadowing": 0}
    with np.testing.assert_raises_regex(RuntimeError, "reset.*physical simulation"):
        env.step()
    with np.testing.assert_raises_regex(RuntimeError, "reset.*physical"):
        env.get_lsf_maps()

    env.reset(RadioEpisodeSeeds.from_root(100))
    assert calls == {"los": 1, "shadowing": 1}
    env.reset(RadioEpisodeSeeds.from_root(101))
    assert calls == {"los": 2, "shadowing": 2}


def test_direct_environment_default_remains_eager() -> None:
    config = _config(soft_fields=True)
    env = RadioEnvironment(config.radio, config.time_resolution_ms)

    assert env.lsf_model.los_model.los_grf is not None
    assert env.lsf_model.sfm is not None
    assert env.lsf_model.sfm.shadowing_z_maps is not None
    env.step()


def test_radio_simulation_runner_realizes_once_for_first_episode(monkeypatch) -> None:
    config = _config(soft_fields=True)
    config = config.model_copy(
        update={
            "radio": config.radio.model_copy(
                update={
                    "channel": config.radio.channel.model_copy(
                        update={"create_lsf_maps": False}
                    )
                }
            )
        }
    )
    calls = {"los": 0, "shadowing": 0}
    original_los = UMaLOSModelSoft.generate_los_maps
    original_shadowing = ShadowFadingCholesky.generate_shadow_fading_maps

    def generate_los(model):
        calls["los"] += 1
        return original_los(model)

    def generate_shadowing(model):
        calls["shadowing"] += 1
        return original_shadowing(model)

    monkeypatch.setattr(UMaLOSModelSoft, "generate_los_maps", generate_los)
    monkeypatch.setattr(
        ShadowFadingCholesky,
        "generate_shadow_fading_maps",
        generate_shadowing,
    )

    runner = RadioSimulationRunner(config)
    assert calls == {"los": 0, "shadowing": 0}
    runner.run(n_ep=1, n_steps=1)
    assert calls == {"los": 1, "shadowing": 1}


def test_deferred_radio_runner_can_initialize_hdf5_logger(tmp_path: Path) -> None:
    config = _config(soft_fields=False)
    runner = RadioSimulationRunner(config)

    output = runner.run(n_ep=1, n_steps=2, save_results=True, output_dir=tmp_path)

    assert output is not None
    assert Path(output).is_file()


def test_episode_streams_reproduce_and_radio_map_is_independent() -> None:
    config = _config(soft_fields=True)
    env = RadioEnvironment(config.radio, config.time_resolution_ms)
    first = RadioEpisodeSeeds.from_root(100)

    env.reset(first)
    first_positions = env.ue_set.xy.copy()
    first_angles = env.ue_set.angles_rad.copy()
    first_fields = _field_realizations(env)

    env.reset(first)
    assert np.array_equal(env.ue_set.xy, first_positions)
    assert np.array_equal(env.ue_set.angles_rad, first_angles)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(_field_realizations(env), first_fields)
    )

    # Changing only the radio-map stream must not move initial positions or
    # mobility RNGs, but must produce another LOS/shadowing realization.
    changed_map = replace(first, radio_map_seed=first.radio_map_seed + 1)
    env.reset(changed_map)
    assert np.array_equal(env.ue_set.xy, first_positions)
    assert np.array_equal(env.ue_set.angles_rad, first_angles)
    assert all(
        not np.array_equal(actual, expected)
        for actual, expected in zip(_field_realizations(env), first_fields)
    )


def test_seeded_partial_ue_reset_is_independent_of_reset_order() -> None:
    config = _config(soft_fields=True)
    plan = RadioEpisodeSeeds.from_root(100)
    env = RadioEnvironment(config.radio, config.time_resolution_ms)

    env.ue_set.reset_ue([0, 1], seed=plan)
    together = env.ue_set.xy.copy()
    env.ue_set.reset_ue(1, seed=plan)
    second_alone = env.ue_set.xy[1].copy()
    env.ue_set.reset_ue(0, seed=plan)
    first_alone = env.ue_set.xy[0].copy()

    assert np.array_equal(first_alone, together[0])
    assert np.array_equal(second_alone, together[1])


def test_random_layout_seed_is_serialized_and_fingerprinted() -> None:
    first = NetworkLayoutConfig(
        layout="HetNetRandomLayout", isd_m=200.0, n_tiers=1, seed=10
    )
    second = first.model_copy(update={"seed": 11})

    assert first.to_dict()["seed"] == 10
    assert second.to_dict()["seed"] == 11
    assert first.fingerprint != second.fingerprint
    geometry = _config(soft_fields=True).radio.geometry.model_copy(
        update={"layout": first}
    )
    assert geometry.to_dict()["layout"]["seed"] == 10


def test_random_layout_changes_bs_correlation_but_not_grf_grid_cache_key() -> None:
    base_geometry = _config(soft_fields=True).radio.geometry

    def geometry(seed: int) -> SimulationGeometry:
        layout = base_geometry.layout.model_copy(
            update={"layout": "HetNetRandomLayout", "seed": seed}
        )
        return SimulationGeometry(base_geometry.model_copy(update={"layout": layout}))

    first = geometry(10)
    second = geometry(11)
    assert not np.array_equal(first.layout.bs_xy, second.layout.bs_xy)
    assert first.roi.fingerprint == second.roi.fingerprint
    assert np.array_equal(
        first.layout.wraparound_shifts, second.layout.wraparound_shifts
    )

    def grf_for(geo: SimulationGeometry) -> GaussianRandomField:
        value = object.__new__(GaussianRandomField)
        value.roi = geo.roi
        value.grid_spacing = 5.0
        value.d_decorr = 50.0
        value.roi_extension_factor = 1.0
        value.progress_bar = False
        return value

    first_grf = grf_for(first)
    second_grf = grf_for(second)
    # The persistent GRF cache describes spatial correlation between ROI-grid
    # points, not correlation between BS-specific fields.
    assert first_grf._cache_filename == second_grf._cache_filename  # noqa: SLF001

    first_bs_distances = first_grf._compute_pairwise_distance_matrix(  # noqa: SLF001
        first.layout.bs_xy
    )
    second_bs_distances = second_grf._compute_pairwise_distance_matrix(  # noqa: SLF001
        second.layout.bs_xy
    )
    assert not np.array_equal(first_bs_distances, second_bs_distances)


def test_cold_and_warm_coverage_cache_produce_identical_live_step(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import mobile_network_sim.radio.lsf.grf.grf as grf_module
    import mobile_network_sim.radio.lsf.lsf_map as map_module

    grf_root = tmp_path / "grf"
    map_root = tmp_path / "coverage"
    monkeypatch.setattr(grf_module, "cache_root", lambda: grf_root)
    monkeypatch.setattr(map_module, "cache_root", lambda: map_root)

    config = _config(soft_fields=False)
    plan = RadioEpisodeSeeds.from_root(321)

    cold_env = RadioEnvironment(
        config.radio, config.time_resolution_ms, realize_lsf=False
    )
    cold_env.reset(plan)
    cold_maps = cold_env.get_lsf_maps()
    cold_step = cold_env.step()
    cold_log = capsys.readouterr().out

    warm_env = RadioEnvironment(
        config.radio, config.time_resolution_ms, realize_lsf=False
    )
    warm_env.reset(plan)
    warm_maps = warm_env.get_lsf_maps()
    warm_step = warm_env.step()
    warm_log = capsys.readouterr().out

    for name in CoverageMaps.map_names():
        assert np.array_equal(
            np.asarray(getattr(cold_maps, name)),
            np.asarray(getattr(warm_maps, name)),
            equal_nan=True,
        )
    _assert_step_equal(cold_step, warm_step)

    assert "Saved coverage-map realization to cache" in cold_log
    assert "Loaded cached coverage-map realization" in warm_log
    assert "maps=['pl_db', 'cg_db', 'rsrp_dbm', 'sinr_db', 'los', 'sf_db']" in warm_log
    assert "radio_map_seed=" in warm_log


def test_truncated_coverage_cache_is_regenerated(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import mobile_network_sim.radio.lsf.grf.grf as grf_module
    import mobile_network_sim.radio.lsf.lsf_map as map_module

    monkeypatch.setattr(grf_module, "cache_root", lambda: tmp_path / "grf")
    monkeypatch.setattr(map_module, "cache_root", lambda: tmp_path / "coverage")
    config = _config(soft_fields=False)
    plan = RadioEpisodeSeeds.from_root(654)
    env = RadioEnvironment(config.radio, config.time_resolution_ms, realize_lsf=False)
    env.reset(plan)
    expected_pl_db = np.asarray(env.get_lsf_maps().pl_db).copy()
    capsys.readouterr()

    damaged = Path(env.lsf_map_generator._map_file_path("sinr_db"))  # noqa: SLF001
    with damaged.open("r+b") as stream:
        stream.truncate(16)

    env.reset(plan)
    repaired = env.get_lsf_maps()
    log = capsys.readouterr().out
    assert "Cached map file has wrong size" in log
    assert "Saved coverage-map realization to cache" in log
    assert np.array_equal(expected_pl_db, repaired.pl_db, equal_nan=True)


def test_runner_executes_explicit_episode_plans_without_rederiving() -> None:
    base = _config(soft_fields=True)
    config = base.model_copy(
        update={
            "radio": base.radio.model_copy(
                update={
                    "channel": base.radio.channel.model_copy(
                        update={"create_lsf_maps": False}
                    )
                }
            )
        }
    )
    runner = RadioSimulationRunner(config)
    plans = [RadioEpisodeSeeds.from_root(111), RadioEpisodeSeeds.from_root(222)]
    seen: list[RadioEpisodeSeeds] = []
    original_reset = runner.env.reset

    def recording_reset(seed=None):
        seen.append(seed)
        original_reset(seed)

    runner.env.reset = recording_reset  # type: ignore[method-assign]
    runner.run(n_ep=2, n_steps=1, episode_seeds=plans)

    assert seen == plans
    assert runner.env.lsf_model.seed == plans[-1].radio_map_seed
    assert runner.env.lsf_model.los_model.seed == plans[-1].lsf_los_seed
    assert runner.env.lsf_model.sfm.seed == plans[-1].lsf_shadowing_seed
    assert runner.env.lsf_model.channel_model.seed == plans[-1].lsf_channel_seed
    assert [ue.mobility_model.seed for ue in runner.env.ue_set.ue_list] == [
        plans[-1].ue_mobility_seed(i) for i in range(config.radio.num_ues)
    ]


def test_realized_map_cache_key_includes_seed_and_actual_layout() -> None:
    class Config:
        fingerprint = "same"

        def to_dict(self):
            return {"same": True}

    def generator(seed: int, bs_xy: np.ndarray) -> MapGenerator:
        ran = Config()
        lsf = SimpleNamespace(
            seed=seed,
            config=Config(),
            internal_dtype=np.dtype("float64"),
            p_noise_ue_dbm=np.array([-90.0]),
            bs_set=SimpleNamespace(ran_config=ran),
        )
        geo = SimpleNamespace(
            config=Config(),
            roi=SimpleNamespace(fingerprint="roi"),
            layout=SimpleNamespace(
                bs_xy=np.asarray(bs_xy, dtype=np.float64),
                wraparound_shifts=np.zeros((1, 2), dtype=np.float64),
            ),
        )
        return MapGenerator(lsf, geo)

    base = generator(10, np.array([[0.0, 0.0]]))._cache_base_name
    assert "seed10" in base
    assert generator(11, np.array([[0.0, 0.0]]))._cache_base_name != base
    assert generator(10, np.array([[1.0, 0.0]]))._cache_base_name != base


def test_grf_matrix_key_uses_exact_inputs_but_not_realization_seed() -> None:
    roi = SimpleNamespace(
        fingerprint="roi",
        wraparound_shifts=np.zeros((1, 2), dtype=np.float64),
    )

    def grf(spacing: float, seed: int) -> GaussianRandomField:
        value = object.__new__(GaussianRandomField)
        value.roi = roi
        value.grid_spacing = spacing
        value.d_decorr = 50.0
        value.roi_extension_factor = 1.0
        value.seed = seed
        return value

    assert grf(5.001, 1)._cache_filename == grf(5.001, 2)._cache_filename
    assert grf(5.001, 1)._cache_filename != grf(5.004, 1)._cache_filename

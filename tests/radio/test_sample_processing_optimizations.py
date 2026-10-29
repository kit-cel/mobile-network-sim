"""Independent references for sample-local optimization and ownership."""

from dataclasses import asdict, fields
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from mobile_network_sim.radio.geometry.grid.grid import Grid
from mobile_network_sim.radio.lsf.channels.fiveg_uma import UrbanMacro
from mobile_network_sim.radio.result_handler.result_loggers import HDF5ResultsLogger
from mobile_network_sim.radio.simulation.sample_source import freeze_result
from mobile_network_sim.rrc.snapshot import RRCUESnapshotBuilder
from mobile_network_sim.runners.radio_rrc import RadioRRCSimulationRunner
from tests.radio.test_runtime_optimizations import assert_radio_equal, runtime_config


@pytest.mark.parametrize("height", [1.5, 12.9, 13.0, 18.0, 22.5])
def test_combined_pathloss_preserves_two_draws_and_general_heights(height):
    baseline, optimized = UrbanMacro(28, seed=43), UrbanMacro(28, seed=43)
    d2 = np.array([[10.0, 18.0, 100.0, 1000.0, 5000.0]])
    d3 = np.sqrt(d2**2 + (25 - height) ** 2)
    inputs = {"d_2d": d2, "d_3d": d3, "h_bs": 25.0, "h_ue": height}
    for _ in range(4):
        expected = baseline.get_pl_los(**inputs), baseline.get_pl_nlos(**inputs)
        actual = optimized.get_pl_los_nlos(**inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        assert baseline.rng1.bit_generator.state == optimized.rng1.bit_generator.state
        assert baseline.rng2.bit_generator.state == optimized.rng2.bit_generator.state


def test_low_height_rng_matches_original_general_algorithm():
    model = UrbanMacro(28, seed=43)
    rng1 = np.random.default_rng()
    rng1.bit_generator.state = model.rng1.bit_generator.state
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = model.rng2.bit_generator.state
    distances = np.ones((2, 7)) * 100
    for _ in range(10):
        c = UrbanMacro._calculate_c(d_2d=distances, h_ue=np.ones((2, 1)) * 1.5)
        draws = rng1.binomial(1, 1 / (1 + c), size=distances.shape)
        rng2.choice([12, 15, 18, 21], size=np.where(draws == 0)[0].shape)
        model._get_effective_env_height(d_2d=distances, h_ue=np.ones((2, 1)) * 1.5)
    assert rng1.bit_generator.state == model.rng1.bit_generator.state
    assert rng2.bit_generator.state == model.rng2.bit_generator.state


def test_prepared_interpolation_matches_original_nan_normalization():
    grid = Grid(20, 20, 1)
    maps = np.random.default_rng(3).normal(size=(3, *grid.shape))
    maps[0, 8:10, 8:10] = np.nan
    coords = np.array([[0.1, 0.2], [-9.0, -9.0], [8.9, 8.7], [0.0, 0.0]])
    values, x, y, origins = grid._get_surrounding_point_data(maps, coords)
    xf, yf = (
        ((x - origins[:, 0]) / grid.spacing)[None],
        ((y - origins[:, 1]) / grid.spacing)[None],
    )
    weights = np.array([(1 - xf) * (1 - yf), xf * (1 - yf), (1 - xf) * yf, xf * yf])
    weights = np.where(~np.isnan(values), weights, 0)
    total = weights.sum(axis=0, keepdims=True)
    norm = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    expected = np.sum(np.nan_to_num(values) * norm, axis=0)
    prepared = grid.prepare_interpolation(coords)
    np.testing.assert_array_equal(grid.interpolate_prepared(maps, prepared), expected)
    with pytest.raises(ValueError, match="different grid"):
        Grid(20, 20, 2).interpolate_prepared(maps, prepared)


@pytest.mark.parametrize("coupling", ["nearest_image", "max_coupling_over_images"])
def test_old_samples_snapshots_and_rng_are_not_mutated(coupling):
    runner = RadioRRCSimulationRunner(
        runtime_config(soft=False, coupling=coupling), action_mode="binary_report"
    )
    try:
        first = runner.step([0, 0])
        saved = freeze_result(first.radio)
        snapshots = [dict(item) for item in first.snapshots]
        for _ in range(30):
            runner.step([1, 0])
        assert_radio_equal(first.radio, saved)
        assert list(first.snapshots) == snapshots
        assert all(
            not getattr(first.radio, f.name).flags.writeable
            for f in fields(first.radio)
        )
        for ue in runner.rrc_ues:
            core = asdict(RRCUESnapshotBuilder._build_core(ue))
            expected = core | {
                f"flag_{key}": getattr(ue.flags_now, key)
                for key in ue.flags_now.field_names()
            }
            assert ue.snapshot() == expected
    finally:
        runner.close()


def test_owned_freeze_reuses_only_fresh_arrays():
    runner = RadioRRCSimulationRunner(runtime_config(soft=False), auto_reset=False)
    try:
        runner.env.reset(2)
        raw = runner.env.step()
        frozen = freeze_result(raw, owned=True)
        assert frozen.rsrp_dbm is raw.rsrp_dbm
        assert not np.shares_memory(frozen.ue_pos_xy, runner.env.ue_set.xy)
        public = freeze_result(raw)
        assert not np.shares_memory(public.rsrp_dbm, raw.rsrp_dbm)
    finally:
        runner.close()


def test_equal_grids_share_preparation_but_not_field_realizations(
    tmp_path, monkeypatch
):
    from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField

    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(tmp_path))
    runner = RadioRRCSimulationRunner(runtime_config(soft=True))
    calls = []
    original = Grid.prepare_interpolation

    def prepare(grid, *args, **kwargs):
        calls.append(grid)
        return original(grid, *args, **kwargs)

    monkeypatch.setattr(Grid, "prepare_interpolation", prepare)
    try:
        runner.step()
        assert len(calls) == 1
        lsf = runner.env.lsf_model
        assert not np.shares_memory(lsf.los_model.los_grf, lsf.sfm.shadowing_z_maps)
    finally:
        runner.close()


@pytest.mark.parametrize("clipping", [False, True])
def test_rsrp_sinr_against_explicit_interference_sum(clipping):
    runner = RadioRRCSimulationRunner(runtime_config(soft=False), auto_reset=False)
    try:
        lsf = runner.env.lsf_model
        lsf.rat_config.rsrp_clipping_enabled = clipping
        n = lsf.bs_set.n_cells
        cg = np.linspace(-120, -50, 2 * n).reshape(2, n)
        rsrp = lsf._calculate_rsrp(cg)
        expected_rsrp = cg + lsf.p_tx_sc_bs_dbm
        if clipping:
            expected_rsrp = np.clip(
                expected_rsrp, lsf.rat_config.rsrp_dbm_min, lsf.rat_config.rsrp_dbm_max
            )
        np.testing.assert_allclose(rsrp, expected_rsrp, rtol=1e-12, atol=1e-10)
        # Include a 100 dB strongest/weakest-power contrast to exercise
        # cancellation in total-minus-own-power without changing the model.
        for span in (30, 100):
            mean = np.tile(np.linspace(-100, -100 + span, n), (2, 1))
            signal = mean + 3
            power = 10 ** ((mean - 30) / 10)
            interference = np.array(
                [
                    [sum(row[k] for k in range(n) if k != cell) for cell in range(n)]
                    for row in power
                ]
            )
            noise = lsf._get_noise_w_for_shape(signal.shape, per_subcarrier=True)
            expected = 10 * np.log10(
                10 ** ((signal - 30) / 10) / (interference + noise)
            )
            np.testing.assert_allclose(
                lsf._calculate_mean_sinr(signal, mean), expected, rtol=1e-12, atol=1e-10
            )
    finally:
        runner.close()


def test_block_logging_matches_single_sample_writes_and_last_partial_block(tmp_path):
    class Result:
        def __init__(self, value):
            self.value = np.full((2, 3), value, dtype=np.float64)

        def to_dict(self):
            return {"signal": self.value}

    env = SimpleNamespace(
        config=SimpleNamespace(), get_dummy_step_result=lambda: Result(0)
    )
    for block in (1, 4):
        logger = HDF5ResultsLogger(
            env, str(tmp_path / f"block{block}.h5"), 2, 7, buffer_size=block
        )
        for episode in range(2):
            logger.new_episode()
            for index in range(7 if episode == 0 else 5):
                result = Result(episode * 100 + index)
                logger.log_step(result)
                result.value[:] = -999
        logger.close()
        logger.close()
    with h5py.File(tmp_path / "block1.h5") as a, h5py.File(tmp_path / "block4.h5") as b:
        for episode in range(2):
            key = f"ep_{episode:04d}/signal"
            np.testing.assert_array_equal(a[key][:], b[key][:])
            assert a[key].dtype == b[key].dtype == np.dtype("float32")

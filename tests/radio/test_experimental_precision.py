"""Opt-in precision is isolated from RNG streams and historical caches."""

import hashlib

import numpy as np
import pytest

from mobile_network_sim.radio.components.antennas.ant_array import PanelArrayAnt
from mobile_network_sim.radio.config.simulation_config import SimulationDTypeConfig
from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField
from mobile_network_sim.runners.radio_rrc import RadioRRCSimulationRunner
from tests.radio.test_runtime_optimizations import runtime_config


def test_precision_defaults_and_validation():
    config = SimulationDTypeConfig()
    assert config.beam == config.grf == config.internal == "float64"
    assert config.maps == config.storage == "float32"
    for name in ("beam", "grf"):
        with pytest.raises(ValueError):
            SimulationDTypeConfig.model_validate({name: "float16"})
        assert (
            getattr(SimulationDTypeConfig.model_validate({name: "float32"}), name)
            == "float32"
        )


def test_beam_precision_uses_distinct_shared_resources():
    runner = RadioRRCSimulationRunner(runtime_config(soft=False), auto_reset=False)
    try:
        source = runner.env.bs_set.bs_list[0]._sector_ant_list[0]
        ants = [
            PanelArrayAnt(
                0,
                source.config,
                source.fc_ghz,
                resource_storage="shared_mmap",
                beam_dtype=dtype,
            )
            for dtype in ("float64", "float32", "float32")
        ]
        assert ants[0].codebook_weights.dtype == np.dtype("complex128")
        assert ants[1].codebook_weights.dtype == np.dtype("complex64")
        assert ants[1].codebook_weights is ants[2].codebook_weights
        assert ants[0]._shared_resource is not ants[1]._shared_resource
        assert ants[1].codebook_weights.nbytes * 2 == ants[0].codebook_weights.nbytes
        phi, theta = np.array([0.0, 1.0, 5.0]), np.array([100.0, 101.0, 102.0])
        assert ants[1].steering_vector(phi, theta).dtype == np.dtype("complex64")
        np.testing.assert_allclose(
            ants[1].get_max_mean_beam_gain_db(phi, theta),
            ants[0].get_max_mean_beam_gain_db(phi, theta),
            rtol=1e-6,
            atol=2e-5,
        )
    finally:
        runner.close()


@pytest.mark.parametrize("storage", ["legacy", "shared_mmap"])
def test_grf_precision_cache_copy_reset_and_rng(tmp_path, monkeypatch, storage):
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(tmp_path))
    config = runtime_config(soft=True)
    config.radio.resource_storage = storage
    first = RadioRRCSimulationRunner(config, auto_reset=False)
    original = first.env.lsf_model.los_model
    old_files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    config32 = config.model_copy(deep=True)
    config32.dtypes.grf = "float32"
    second = RadioRRCSimulationRunner(config32, auto_reset=False)
    reduced = second.env.lsf_model.los_model
    try:
        assert original.cholesky_mat.dtype == np.dtype("float64")
        assert reduced.cholesky_mat.dtype == np.dtype("float32")
        assert original._cache_filename != reduced._cache_filename
        assert "factor_dtype" not in original._matrix_cache_identity
        assert reduced.cholesky_mat.nbytes * 2 == original.cholesky_mat.nbytes
        np.testing.assert_array_equal(
            reduced.cholesky_mat, original.cholesky_mat.astype("float32")
        )
        copied = GaussianRandomField.copy(reduced, seed=3)
        assert copied.factor_dtype == np.dtype("float32")
        initial = None
        for seed in (13, 27, 13):
            first.reset(seed)
            second.reset(seed)
            assert (
                original.np_rng.bit_generator.state
                == reduced.np_rng.bit_generator.state
            )
            if seed == 13:
                if initial is None:
                    initial = reduced.los_grf.copy()
                else:
                    np.testing.assert_array_equal(initial, reduced.los_grf)
        for path, digest in old_files.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        if storage == "shared_mmap":
            assert isinstance(reduced.cholesky_mat, np.memmap)
            assert not reduced.cholesky_mat.flags.writeable
    finally:
        first.close()
        second.close()


def test_beam_precision_invalidates_runner_static_resources():
    config = runtime_config(soft=False)
    first = RadioRRCSimulationRunner(config, auto_reset=False)
    config32 = config.model_copy(deep=True)
    config32.dtypes.beam = "float32"
    second = RadioRRCSimulationRunner(
        config32, auto_reset=False, static_resources=first.env.static_resources
    )
    try:
        assert second.env.bs_set is not first.env.bs_set
        assert second.env.bs_set.bs_list[0]._sector_ant_list[
            0
        ].codebook_weights.dtype == np.dtype("complex64")
    finally:
        first.close()
        second.close()

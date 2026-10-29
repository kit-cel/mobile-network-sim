"""Sharing must change ownership, never random fields or physical trajectories."""

import gc
import multiprocessing as mp
import weakref
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import pytest
from test_runtime_optimizations import assert_radio_equal, runtime_config

from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField
from mobile_network_sim.runners.radio_rrc import RadioRRCSimulationRunner


def make_runner(storage):
    config = runtime_config()
    config.radio.resource_storage = storage
    return RadioRRCSimulationRunner(
        config, action_mode="binary_report", auto_reset=False
    )


def test_shared_identity_copy_readonly_and_lifetime(tmp_path, monkeypatch):
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(tmp_path))
    a, b = make_runner("shared_mmap"), make_runner("shared_mmap")
    x, y = a.env.lsf_model.sfm, b.env.lsf_model.sfm
    assert x.cholesky_mat is y.cholesky_mat
    assert isinstance(x.cholesky_mat, np.memmap)
    copied = GaussianRandomField.copy(x, seed=42)
    assert copied.cholesky_mat is x.cholesky_mat
    assert copied.grid is not x.grid
    assert copied.grid.xy is x.grid.xy
    assert copied.np_rng is not x.np_rng
    with pytest.raises(ValueError):
        x.cholesky_mat[0, 0] = 99
    ant = a.env.bs_set.bs_list[0]._sector_ant_list[0]
    assert (
        ant.codebook_weights
        is b.env.bs_set.bs_list[0]._sector_ant_list[0].codebook_weights
    )
    assert deepcopy(ant).codebook_weights is ant.codebook_weights
    ref = weakref.ref(x._shared_resource)
    a.close()
    b.close()
    copied.release_shared_resources()
    gc.collect()
    assert ref() is None
    a.reset(13)
    assert isinstance(a.env.lsf_model.sfm.cholesky_mat, np.memmap)
    a.close()
    del a, b, x, y, copied
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("coupling", ["nearest_image", "max_coupling_over_images"])
def test_legacy_shared_replay_and_private_fields(tmp_path, monkeypatch, coupling):
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(tmp_path))
    config = runtime_config(coupling=coupling)
    config.radio.resource_storage = "legacy"
    reference = RadioRRCSimulationRunner(
        config, action_mode="binary_report", auto_reset=False
    )
    config = config.model_copy(deep=True)
    config.radio.resource_storage = "shared_mmap"
    shared = RadioRRCSimulationRunner(
        config, action_mode="binary_report", auto_reset=False
    )
    for seed in (13, 27, 13):
        reference.reset(seed)
        shared.reset(seed)
        assert not np.shares_memory(
            reference.env.lsf_model.sfm.shadowing_z_maps,
            shared.env.lsf_model.sfm.shadowing_z_maps,
        )
        for step in range(12):
            actions = [int(step % 5 == 0), int(step % 7 == 0)]
            a, b = reference.step(actions), shared.step(actions)
            assert_radio_equal(a.radio, b.radio)
            np.testing.assert_array_equal(a.flags, b.flags)
            np.testing.assert_array_equal(a.serving_cells, b.serving_cells)
            for x, y in zip(
                reference.env.lsf_model._random_generators(),
                shared.env.lsf_model._random_generators(),
                strict=True,
            ):
                assert x.bit_generator.state == y.bit_generator.state
    reference.close()
    shared.close()


def spawned_shared(cache):
    GaussianRandomField.CACHE_DIR = cache
    runner = make_runner("shared_mmap")
    runner.reset(13)
    factor = runner.env.lsf_model.sfm.cholesky_mat
    result = (str(factor.filename), runner.step([0, 0]).radio.rsrp_dbm)
    runner.close()
    return result


def test_spawned_users_map_same_files(tmp_path):
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        a, b = list(pool.map(spawned_shared, [str(tmp_path)] * 2))
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


def test_disabled_cache_and_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", None)
    first = make_runner("shared_mmap")
    assert first.env.lsf_model.sfm.resource_status == "memory_only"
    first.close()
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(blocked))
    with pytest.warns(UserWarning, match="private arrays"):
        fallback = make_runner("shared_mmap")
    fallback.reset(13)
    assert fallback.env.lsf_model.sfm.resource_status == "private_fallback"
    fallback.close()

"""Spawned cold-cache users must agree and may publish the same cache safely."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from test_runtime_optimizations import runtime_config

from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField
from mobile_network_sim.runners.radio_rrc import RadioRRCSimulationRunner


def _cold_worker(cache):
    GaussianRandomField.CACHE_DIR = cache
    runner = RadioRRCSimulationRunner(runtime_config(), auto_reset=False)
    runner.reset(61)
    return np.stack([runner.step().radio.rsrp_dbm for _ in range(4)])


def test_concurrent_cache_population_and_new_process_warm_start(tmp_path):
    cache = str(tmp_path / "initially_absent")
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        cold = list(pool.map(_cold_worker, [cache, cache]))
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        warm = pool.submit(_cold_worker, cache).result()
    np.testing.assert_array_equal(cold[0], cold[1])
    np.testing.assert_array_equal(cold[0], warm)

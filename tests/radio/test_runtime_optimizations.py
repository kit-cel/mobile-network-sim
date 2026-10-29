"""Runtime changes must preserve radio values, event timing and RNG streams."""

from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest
import yaml

from mobile_network_sim.radio.components.bs.bs_collection import BaseStationCollection
from mobile_network_sim.radio.geometry.grid.grid import Grid
from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField
from mobile_network_sim.runners.radio_rrc import (
    RadioRRCSimulationConfig,
    RadioRRCSimulationRunner,
)


def runtime_config(*, soft=True, coupling="nearest_image", dtype="float64"):
    with open("examples/radio_rrc.yaml", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    data["num_steps"] = 80
    data["radio_source_mode"] = "block"
    data["radio_source_capacity"] = 1
    data["dtypes"]["internal"] = dtype
    data["radio"]["channel"].update(
        los_model="UMaLOSModelSoft" if soft else "ConstLOS",
        shadowing_enabled=soft,
        soft_los_map_grid_spacing_m=100.0,
        shadowing_map_grid_spacing_m=100.0,
        wraparound_coupling_method=coupling,
    )
    data["radio"]["network"]["macro_bs"]["ant"].update(
        n_element_h=4,
        m_element_v=4,
        beamforming={"az_list_deg": [-45.0, 0.0, 45.0], "el_list_deg": [-10.0] * 3},
    )
    return RadioRRCSimulationConfig.model_validate(data)


def assert_radio_equal(left, right, dtype="float64"):
    rtol, atol = (1e-12, 1e-10) if dtype == "float64" else (1e-6, 2e-5)
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(np.isinf(a), np.isinf(b))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=field.name)


@pytest.mark.parametrize("coupling", ["nearest_image", "max_coupling_over_images"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("soft", [False, True])
def test_radio_events_seeds_and_reset_match_reference(
    tmp_path, monkeypatch, coupling, dtype, soft
):
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(tmp_path))
    config = runtime_config(soft=soft, coupling=coupling, dtype=dtype)
    reference = RadioRRCSimulationRunner(
        config, action_mode="binary_report", auto_reset=False
    )
    optimized = RadioRRCSimulationRunner(
        config, action_mode="binary_report", auto_reset=False
    )
    first = None
    for seed in [13, 27, 13]:
        reference.reset(seed)
        optimized.reset(seed)
        for sample in range(40):
            actions = [int(sample % 11 == 0), int(sample % 7 == 0)]
            monkeypatch.setattr(BaseStationCollection, "COMBINED_BEAM_GAINS", False)
            monkeypatch.setattr(Grid, "CACHE_CORNERS", False)
            expected = reference.step(actions)
            monkeypatch.setattr(BaseStationCollection, "COMBINED_BEAM_GAINS", True)
            monkeypatch.setattr(Grid, "CACHE_CORNERS", True)
            actual = optimized.step(actions)
            assert_radio_equal(expected.radio, actual.radio, dtype)
            np.testing.assert_array_equal(expected.states, actual.states)
            np.testing.assert_array_equal(expected.serving_cells, actual.serving_cells)
            np.testing.assert_array_equal(expected.flags, actual.flags)
            for a, b in zip(
                reference.env.lsf_model._random_generators(),
                optimized.env.lsf_model._random_generators(),
                strict=True,
            ):
                assert a.bit_generator.state == b.bit_generator.state
            if seed == 13 and sample == 0:
                if first is None:
                    first = deepcopy(actual.radio)
                else:
                    assert_radio_equal(first, actual.radio, dtype)


def test_combined_beams_match_explicit_element_sum():
    runner = RadioRRCSimulationRunner(runtime_config(soft=False), auto_reset=False)
    ant = runner.env.bs_set.bs_list[0]._sector_ant_list[0]
    phi = np.array([-180.0, -45.0, 0.0, 45.0, 179.0, 0.0])
    theta = np.array([90.0, 100.0, 100.0, 100.0, 90.0, 0.0])
    # Independent scalar element accumulation; no BLAS reduction or max/mean API.
    from mobile_network_sim.radio.geometry.geo_utils import gcs_to_lcs

    local_phi, local_theta = gcs_to_lcs(
        phi,
        theta,
        bearing=ant.sector_azimuth_deg,
        downtilt=ant.mechanical_downtilt_deg,
        slant=0.0,
    )
    steering = ant.steering_vector(local_phi, local_theta)
    element = ant.element_gain_db(local_phi, local_theta)
    gains = np.array(
        [
            element
            + 10
            * np.log10(
                np.maximum(
                    np.abs(
                        sum(
                            np.conj(weight) * steering[index]
                            for index, weight in enumerate(beam)
                        )
                    )
                    ** 2,
                    1e-12,
                )
            )
            for beam in ant.codebook_weights
        ]
    )
    best, mean = ant.get_max_mean_beam_gain_db(phi, theta)
    np.testing.assert_allclose(best, gains.max(axis=0), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        mean, 10 * np.log10(np.mean(10 ** (gains / 10), axis=0)), rtol=1e-12, atol=1e-10
    )
    # Exact ties select the first beam.
    from mobile_network_sim.radio.components.antennas.ant_array import PanelArrayAnt

    config = ant.config.model_copy(deep=True)
    from mobile_network_sim.radio.config.beamforming_config import BeamformingConfig

    config.beamforming = BeamformingConfig(
        az_list_deg=[0.0, 0.0], el_list_deg=[0.0, 0.0]
    )
    duplicate = PanelArrayAnt(0, config, ant.fc_ghz)
    tied = duplicate._compute_max_beam_gain_and_beam_idx(
        np.array([0.0]), np.array([90.0])
    )[1]
    assert tied[0] == 0


def test_grid_cache_retains_nan_boundary_and_weight_semantics(monkeypatch):
    grid = Grid(20.0, 20.0, 1.0)
    values = grid.xy[..., 0] + 2 * grid.xy[..., 1]
    coordinates = np.array([[-5.2, -4.6], [0.0, 0.0], [0.1, 0.2], [5.8, 5.9]])
    np.testing.assert_allclose(
        grid.get_interpolated_values(values, coordinates),
        (coordinates[:, 0] + 2 * coordinates[:, 1])[None],
    )
    values[3, 3] = np.nan
    for points in (coordinates, coordinates + 0.01):
        monkeypatch.setattr(Grid, "CACHE_CORNERS", False)
        expected = grid.get_interpolated_values(values, points)
        monkeypatch.setattr(Grid, "CACHE_CORNERS", True)
        np.testing.assert_array_equal(
            grid.get_interpolated_values(values, points), expected
        )
    monkeypatch.setattr(Grid, "MAX_CACHED_CELLS", 2)
    grid.get_interpolated_values(values, coordinates + 0.3)
    assert len(grid._corner_cache) <= 2


def test_process_cache_shares_only_readonly_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(GaussianRandomField, "CACHE_DIR", str(tmp_path))
    GaussianRandomField.clear_process_cache()
    config = runtime_config()
    first = RadioRRCSimulationRunner(config, auto_reset=False)
    second = RadioRRCSimulationRunner(config, auto_reset=False)
    a, b = first.env.lsf_model.los_model, second.env.lsf_model.los_model
    assert a.cholesky_mat is b.cholesky_mat
    assert a.grid is not b.grid
    assert not a.cholesky_mat.flags.writeable
    first.reset(3)
    second.reset(4)
    assert not np.shares_memory(a.los_grf, b.los_grf)
    second.reset(3)
    np.testing.assert_array_equal(a.los_grf, b.los_grf)
    monkeypatch.setattr(GaussianRandomField, "USE_PROCESS_CACHE", False)
    third = RadioRRCSimulationRunner(config, auto_reset=False)
    third.reset(3)
    assert_radio_equal(first.step().radio, third.step().radio)

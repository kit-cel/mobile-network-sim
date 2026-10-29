import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erfinv

from mobile_network_sim.radio.config.lsf_config import LargeScaleFadingConfig
from mobile_network_sim.radio.lsf.channels.fiveg_rma import RuralMacro
from mobile_network_sim.radio.lsf.channels.fiveg_uma import UrbanMacro
from mobile_network_sim.radio.lsf.grf.grf import GaussianRandomField
from mobile_network_sim.radio.lsf.los_models.rma_los import RMaLOSModel
from mobile_network_sim.radio.lsf.los_models.uma_soft_los import UMaLOSModelSoft


def test_uma_get_pl_nlos_applies_los_lower_bound() -> None:
    model = UrbanMacro(fc_ghz=4.0, seed=7)
    d_2d = np.asarray([[10.0, 20.0, 50.0]], dtype=np.float64)
    d_3d = d_2d.copy()
    h_bs = np.asarray([25.0, 25.0, 25.0], dtype=np.float64)
    h_ue = np.asarray([22.0], dtype=np.float64)

    pl_los = model.get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
    pl_nlos_formula = model._get_pl_nlos_formula(d_3d=d_3d, h_ue=np.asarray([[22.0]]))
    pl_nlos = model.get_pl_nlos(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)

    np.testing.assert_allclose(pl_nlos, np.maximum(pl_los, pl_nlos_formula))
    assert np.any(pl_nlos_formula < pl_los)


def test_uma_get_pl_validates_distance_shapes() -> None:
    model = UrbanMacro(fc_ghz=4.0, seed=7)
    d_2d = np.asarray([[50.0, 60.0]], dtype=np.float64)
    d_3d = np.asarray([[50.0], [60.0]], dtype=np.float64)
    is_los = np.ones_like(d_2d, dtype=bool)

    with pytest.raises(ValueError, match="d_3d shape must match"):
        model.get_pl(
            d_2d=d_2d,
            d_3d=d_3d,
            is_los=is_los,
            h_bs=np.asarray([25.0, 25.0], dtype=np.float64),
            h_ue=np.asarray([1.5], dtype=np.float64),
        )


def test_uma_get_pl_validates_los_shape() -> None:
    model = UrbanMacro(fc_ghz=4.0, seed=7)
    d_2d = np.asarray([[50.0, 60.0]], dtype=np.float64)
    d_3d = d_2d.copy()
    is_los = np.ones((2, 1), dtype=bool)

    with pytest.raises(ValueError, match="is_los shape must match"):
        model.get_pl(
            d_2d=d_2d,
            d_3d=d_3d,
            is_los=is_los,
            h_bs=np.asarray([25.0, 25.0], dtype=np.float64),
            h_ue=np.asarray([1.5], dtype=np.float64),
        )


def test_rural_macro_is_selectable() -> None:
    config = LargeScaleFadingConfig(
        channel_model="RuralMacro",
        los_model="RMaLOSModel",
    )

    assert config.channel_model == "RuralMacro"
    assert config.los_model == "RMaLOSModel"


def test_rma_los_path_loss_matches_spec_formula() -> None:
    model = RuralMacro(fc_ghz=4.0, seed=7)
    d_2d = np.asarray([[1000.0, 6000.0]], dtype=np.float64)
    d_3d = d_2d.copy()
    h_bs = np.asarray([35.0, 35.0], dtype=np.float64)
    h_ue = np.asarray([1.5], dtype=np.float64)

    pl_los = model.get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
    d_break = 2.0 * np.pi * 35.0 * 1.5 * 4.0e9 / 3.0e8
    pl_1 = (
        20.0 * np.log10(40.0 * np.pi * d_3d * 4.0 / 3.0)
        + min(0.03 * 5.0**1.72, 10.0) * np.log10(d_3d)
        - min(0.044 * 5.0**1.72, 14.77)
        + 0.002 * np.log10(5.0) * d_3d
    )
    pl_1_at_break = (
        20.0 * np.log10(40.0 * np.pi * d_break * 4.0 / 3.0)
        + min(0.03 * 5.0**1.72, 10.0) * np.log10(d_break)
        - min(0.044 * 5.0**1.72, 14.77)
        + 0.002 * np.log10(5.0) * d_break
    )
    pl_2 = pl_1_at_break + 40.0 * np.log10(d_3d / d_break)
    expected = np.where(d_2d < d_break, pl_1, pl_2)

    np.testing.assert_allclose(pl_los, expected)


def test_rma_get_pl_nlos_applies_los_lower_bound() -> None:
    model = RuralMacro(fc_ghz=4.0, seed=7)
    d_2d = np.asarray([[10.0, 20.0, 50.0]], dtype=np.float64)
    d_3d = d_2d.copy()
    h_bs = np.asarray([35.0, 35.0, 35.0], dtype=np.float64)
    h_ue = np.asarray([1.5], dtype=np.float64)

    pl_los = model.get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
    pl_nlos_formula = model._pl_nlos_formula(
        d_3d=d_3d,
        h_bs=np.asarray([[35.0, 35.0, 35.0]], dtype=np.float64),
        h_ue=np.asarray([[1.5]], dtype=np.float64),
    )
    pl_nlos = model.get_pl_nlos(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)

    np.testing.assert_allclose(pl_nlos, np.maximum(pl_los, pl_nlos_formula))
    assert np.any(pl_nlos_formula < pl_los)


def test_rma_enforces_state_specific_distance_limits() -> None:
    model = RuralMacro(fc_ghz=4.0, seed=7)
    d_2d = np.asarray([[6000.0]], dtype=np.float64)
    d_3d = np.sqrt(d_2d**2 + (35.0 - 1.5) ** 2)

    model.get_pl(
        d_2d=d_2d,
        d_3d=d_3d,
        is_los=np.asarray([[True]]),
        h_bs=35.0,
        h_ue=1.5,
    )

    with pytest.raises(ValueError, match="5 km for NLOS"):
        model.get_pl(
            d_2d=d_2d,
            d_3d=d_3d,
            is_los=np.asarray([[False]]),
            h_bs=35.0,
            h_ue=1.5,
        )


def test_rma_clipping_preserves_vertical_distance() -> None:
    model = RuralMacro(fc_ghz=4.0, seed=7, clip=True)
    d_2d = np.asarray([[6000.0]], dtype=np.float64)
    vertical_distance_m = 35.0 - 1.5
    d_3d = np.sqrt(d_2d**2 + vertical_distance_m**2)

    actual = model.get_pl_nlos(
        d_2d=d_2d,
        d_3d=d_3d,
        h_bs=35.0,
        h_ue=1.5,
    )
    clipped_d_2d = np.asarray([[5000.0]], dtype=np.float64)
    clipped_d_3d = np.sqrt(clipped_d_2d**2 + vertical_distance_m**2)
    expected = model.get_pl_nlos(
        d_2d=clipped_d_2d,
        d_3d=clipped_d_3d,
        h_bs=35.0,
        h_ue=1.5,
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("fc_ghz", [0.5, 30.0])
def test_rma_rejects_frequency_range_boundaries(fc_ghz: float) -> None:
    with pytest.raises(ValueError, match="fc_ghz must be in"):
        RuralMacro(fc_ghz=fc_ghz)


def test_rma_los_probability_matches_spec_formula() -> None:
    model = RMaLOSModel(seed=7)
    d_2d = np.asarray([[5.0, 10.0, 1010.0]], dtype=np.float64)

    p_los = model.get_los_probs(
        d_2d=d_2d,
        heights=np.asarray([1.5], dtype=np.float64),
        pos_2d=np.zeros((1, 2), dtype=np.float64),
    )

    expected = np.asarray([[1.0, 1.0, np.exp(-1.0)]], dtype=np.float64)
    np.testing.assert_allclose(p_los, expected)


def test_uma_soft_los_uses_standard_correlation_distance_by_default() -> None:
    parameter = inspect.signature(UMaLOSModelSoft).parameters["d_decorr"]

    assert parameter.default == 50.0


def test_grf_uses_3gpp_one_over_e_correlation_distance() -> None:
    model = object.__new__(GaussianRandomField)
    model.d_decorr = 20.0
    model.alpha = 1.0 / model.d_decorr

    distance_matrix = np.asarray(
        [[0.0, model.d_decorr], [model.d_decorr, 0.0]], dtype=np.float64
    )
    cholesky_matrix = model._compute_cholesky_matrix(distance_matrix)
    correlation_matrix = cholesky_matrix @ cholesky_matrix.T

    np.testing.assert_allclose(correlation_matrix[0, 1], np.exp(-1.0))


def test_grf_cache_key_includes_new_correlation_schema() -> None:
    grf = object.__new__(GaussianRandomField)
    grf.roi = SimpleNamespace(
        fingerprint="roi", wraparound_shifts=np.zeros((1, 2), dtype=float)
    )
    grf.grid_spacing = 10.0
    grf.d_decorr = 20.0
    grf.mu = 0.0
    grf.sigma = 1.0
    grf.roi_extension_factor = 1.0
    grf.map_dtype = np.dtype("float32")

    assert GaussianRandomField.CACHE_SCHEMA in grf._cache_filename


def test_uma_soft_los_matches_spec_mapping() -> None:
    model = object.__new__(UMaLOSModelSoft)
    model.los_grf = np.ones((1, 1, 1), dtype=np.float64)
    model.lambda_fc = 0.1
    gaussian_value = 0.25
    los_probability = 0.8
    model.get_grf_values = lambda fields, positions: np.asarray(  # type: ignore[method-assign]
        [[gaussian_value]], dtype=np.float64
    )
    model._uma_los_model = SimpleNamespace(
        get_los_probs=lambda distances, heights, positions: np.asarray(
            [[los_probability]], dtype=np.float64
        )
    )

    actual = model.get_los_probs(
        d_2d=np.asarray([[100.0]], dtype=np.float64),
        heights=np.asarray([1.5], dtype=np.float64),
        pos_2d=np.asarray([[0.0, 0.0]], dtype=np.float64),
    )
    threshold = np.sqrt(2.0) * erfinv(2.0 * los_probability - 1.0)
    expected = (
        0.5
        + np.arctan(np.sqrt(20.0 / model.lambda_fc) * (gaussian_value + threshold))
        / np.pi
    )

    np.testing.assert_allclose(actual, [[expected]])

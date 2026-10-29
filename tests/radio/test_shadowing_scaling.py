import numpy as np
import pytest
from pydantic import ValidationError

from mobile_network_sim.radio.config.lsf_config import LargeScaleFadingConfig
from mobile_network_sim.radio.config.simulation_config import SimulationConfig
from mobile_network_sim.radio.lsf.lsf import LSFModel


class _DummySfm:
    def __init__(self, z_values: np.ndarray) -> None:
        self.z_values = z_values

    def get_sf_values(self, positions: np.ndarray) -> np.ndarray:
        return self.z_values[: positions.shape[0]]


class _DummyBsSet:
    n_bs = 2
    n_cells_per_bs = 3


def _lsf_for_shadowing(
    z_values: np.ndarray | None,
    *,
    sigma_los: float = 4.0,
    sigma_nlos: float = 6.0,
) -> LSFModel:
    lsf = LSFModel.__new__(LSFModel)
    lsf.config = LargeScaleFadingConfig(
        shadowing_std_los_db=sigma_los,
        shadowing_std_nlos_db=sigma_nlos,
    )
    lsf.internal_dtype = np.dtype("float64")
    lsf.bs_set = _DummyBsSet()
    lsf.sfm = None if z_values is None else _DummySfm(z_values)
    return lsf


def test_shadowing_scaling_uses_los_std_for_hard_los() -> None:
    z_values = np.asarray([[1.0, -2.0], [0.5, 1.5]], dtype=np.float64)
    lsf = _lsf_for_shadowing(z_values, sigma_los=4.0, sigma_nlos=6.0)
    pos_2d = np.zeros((2, 2), dtype=np.float64)
    los = np.ones((2, 6), dtype=np.float64)

    sf_db = lsf._calculate_shadow_fading_db(
        pos_2d=pos_2d,
        los=los,
        target_shape=los.shape,
    )

    expected = np.repeat(z_values, 3, axis=-1) * 4.0
    np.testing.assert_allclose(sf_db, expected)


def test_shadowing_scaling_uses_nlos_std_for_hard_nlos() -> None:
    z_values = np.asarray([[1.0, -2.0], [0.5, 1.5]], dtype=np.float64)
    lsf = _lsf_for_shadowing(z_values, sigma_los=4.0, sigma_nlos=6.0)
    pos_2d = np.zeros((2, 2), dtype=np.float64)
    los = np.zeros((2, 6), dtype=np.float64)

    sf_db = lsf._calculate_shadow_fading_db(
        pos_2d=pos_2d,
        los=los,
        target_shape=los.shape,
    )

    expected = np.repeat(z_values, 3, axis=-1) * 6.0
    np.testing.assert_allclose(sf_db, expected)


def test_shadowing_scaling_uses_soft_los_variance_blend() -> None:
    z_values = np.asarray([[1.0, -2.0]], dtype=np.float64)
    lsf = _lsf_for_shadowing(z_values, sigma_los=4.0, sigma_nlos=6.0)
    pos_2d = np.zeros((1, 2), dtype=np.float64)
    los = np.asarray([[0.0, 0.25, 0.5, 0.75, 1.0, 0.5]], dtype=np.float64)

    sf_db = lsf._calculate_shadow_fading_db(
        pos_2d=pos_2d,
        los=los,
        target_shape=los.shape,
    )

    z_sf = np.repeat(z_values, 3, axis=-1)
    sigma_eff = np.sqrt(los * 4.0**2 + (1.0 - los) * 6.0**2)
    np.testing.assert_allclose(sf_db, z_sf * sigma_eff)


def test_shadowing_disabled_returns_zero_shadowing() -> None:
    lsf = _lsf_for_shadowing(None)
    pos_2d = np.zeros((2, 2), dtype=np.float64)
    los = np.ones((2, 6), dtype=np.float64)

    sf_db = lsf._calculate_shadow_fading_db(
        pos_2d=pos_2d,
        los=los,
        target_shape=los.shape,
    )

    np.testing.assert_array_equal(sf_db, np.zeros_like(los))


def test_old_shadowing_std_db_config_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LargeScaleFadingConfig(shadowing_std_db=6.0)


def test_example_config_uses_los_nlos_shadowing_std_keys() -> None:
    config = SimulationConfig.from_yaml("examples/radio.yaml")

    assert config.radio.channel.shadowing_std_los_db == 4.0
    assert config.radio.channel.shadowing_std_nlos_db == 6.0

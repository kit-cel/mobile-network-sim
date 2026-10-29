import numpy as np
import pytest
from pydantic import ValidationError

from mobile_network_sim.radio.config.lsf_config import LargeScaleFadingConfig
from mobile_network_sim.radio.geometry.topology.euclidean_radio_topology import (
    EuclideanRadioTopology,
)
from mobile_network_sim.radio.geometry.topology.wraparound_topology import (
    WraparoundRadioTopology,
)
from mobile_network_sim.radio.lsf.los_models.const_los_nlos import ConstLOS
from mobile_network_sim.radio.lsf.lsf import LSFModel


class _DistancePathLoss:
    def get_pl(self, *, d_2d, d_3d, is_los, h_bs, h_ue):
        return d_2d


class _SingleCellBsSet:
    n_bs = 1
    n_cells = 1
    n_cells_per_bs = [1]
    bs_heights = np.asarray([25.0], dtype=np.float64)

    def get_max_ant_gains(self, phi_az, theta_el):
        return phi_az[..., 0][..., None]

    def get_mean_ant_gains(self, phi_az, theta_el):
        return np.zeros((*phi_az.shape[:-1], 1), dtype=np.float64)

    def get_max_mean_ant_gains(self, phi_az, theta_el):
        return self.get_max_ant_gains(phi_az, theta_el), self.get_mean_ant_gains(
            phi_az, theta_el
        )


class _RatConfig:
    rsrp_clipping_enabled = False


def _synthetic_lsf() -> LSFModel:
    lsf = LSFModel.__new__(LSFModel)
    lsf.config = LargeScaleFadingConfig(
        wraparound_coupling_method="max_coupling_over_images",
        shadowing_enabled=False,
    )
    lsf.internal_dtype = np.dtype("float64")
    lsf.bs_set = _SingleCellBsSet()
    lsf.rat_config = _RatConfig()
    lsf.los_model = ConstLOS()
    lsf.channel_model = _DistancePathLoss()
    lsf.sfm = None
    lsf.p_tx_sc_bs_dbm = np.asarray([0.0], dtype=np.float64)
    lsf.p_noise_sc_ue_w = np.asarray([[1e-9]], dtype=np.float64)
    return lsf


def test_lsf_config_accepts_wraparound_coupling_modes() -> None:
    nearest = LargeScaleFadingConfig(wraparound_coupling_method="nearest_image")
    max_coupling = LargeScaleFadingConfig(
        wraparound_coupling_method="max_coupling_over_images"
    )

    assert nearest.wraparound_coupling_method == "nearest_image"
    assert max_coupling.wraparound_coupling_method == "max_coupling_over_images"


def test_lsf_config_rejects_unknown_wraparound_coupling_mode() -> None:
    with pytest.raises(ValidationError):
        LargeScaleFadingConfig(wraparound_coupling_method="invalid")


def test_euclidean_candidate_geometry_has_single_image_axis() -> None:
    topology = EuclideanRadioTopology(bs_xy=np.asarray([[0.0, 0.0]]))

    d_2d, d_3d, phi, theta = topology.candidate_distances_and_angles(
        ue_xy=np.asarray([[3.0, 4.0]]),
        ue_heights=np.asarray([1.5]),
        bs_heights=np.asarray([25.0]),
    )

    assert d_2d.shape == (1, 1, 1)
    assert d_3d.shape == (1, 1, 1)
    assert phi.shape == (1, 1, 1)
    assert theta.shape == (1, 1, 1)
    np.testing.assert_allclose(d_2d[..., 0, :], [[5.0]])


def test_wraparound_candidate_geometry_exposes_all_images() -> None:
    topology = WraparoundRadioTopology(
        bs_xy=np.asarray([[0.0, 0.0]]),
        wraparound_shifts=np.asarray([[0.0, 0.0], [100.0, 0.0]]),
    )

    disp = topology.candidate_displacements_2d(np.asarray([[90.0, 0.0]]))
    d_2d, d_3d, phi, theta = topology.candidate_distances_and_angles(
        ue_xy=np.asarray([[90.0, 0.0]]),
        ue_heights=np.asarray([1.5]),
        bs_heights=np.asarray([25.0]),
    )
    nearest = topology.displacement_2d(np.asarray([[90.0, 0.0]]))

    np.testing.assert_allclose(disp[0, :, 0, 0], [90.0, -10.0])
    np.testing.assert_allclose(d_2d[0, :, 0], [90.0, 10.0])
    assert d_3d.shape == (1, 2, 1)
    assert phi.shape == (1, 2, 1)
    assert theta.shape == (1, 2, 1)
    np.testing.assert_allclose(nearest[0, 0], [-10.0, 0.0])


def test_max_coupling_over_images_selects_strongest_image_consistently() -> None:
    lsf = _synthetic_lsf()
    pos_3d = np.asarray([[0.0, 0.0, 1.5]], dtype=np.float64)

    metrics = lsf.calculate_lsf_metrics(
        pos_3d=pos_3d,
        d_2d=np.asarray([[10.0]], dtype=np.float64),
        d_3d=np.asarray([[10.0]], dtype=np.float64),
        phi_az_deg=np.asarray([[0.0]], dtype=np.float64),
        theta_el_deg=np.asarray([[90.0]], dtype=np.float64),
        candidate_d_2d=np.asarray([[[10.0], [100.0]]], dtype=np.float64),
        candidate_d_3d=np.asarray([[[10.0], [100.0]]], dtype=np.float64),
        candidate_phi_az_deg=np.asarray([[[0.0], [150.0]]], dtype=np.float64),
        candidate_theta_el_deg=np.asarray([[[90.0], [90.0]]], dtype=np.float64),
    )

    np.testing.assert_allclose(metrics.pl_db, [[100.0]])
    np.testing.assert_allclose(metrics.cg_db, [[50.0]])
    np.testing.assert_allclose(metrics.rsrp_dbm, [[50.0]])
    np.testing.assert_allclose(metrics.los, [[1.0]])
    np.testing.assert_allclose(metrics.sf_db, [[0.0]])

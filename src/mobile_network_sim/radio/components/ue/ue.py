"""User equipment entity."""

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from . import mobility
from .mobility.mobility_state import UEMobilityState
from ...utils.types import FloatArr
from ...utils import commuinations as utc

if TYPE_CHECKING:
    from .mobility.abstract_mobility_model import AbstractMobilityModel
    from ...config.ue_config import UserEquipmentConfig


@dataclass(slots=True)
class Distances:
    """
    Data class for UE trajectory information.

    Attributes
    ----------
    d_2d_m : FloatArr
        2D distances to all BSs in meters.
    d_3d_m : FloatArr
        3D distances to all BSs in meters.
    """

    d_2d_m: FloatArr = field(default_factory=lambda: np.full((0,), np.nan, dtype=float))
    d_3d_m: FloatArr = field(default_factory=lambda: np.full((0,), np.nan, dtype=float))

    def update(self, d_2d: FloatArr, d_3d: FloatArr) -> None:
        """Update the link-distance arrays.

        Parameters
        ----------
        d_2d : FloatArr
            Two-dimensional link distances in metres.
        d_3d : FloatArr
            Three-dimensional link distances in metres.
        """
        self.d_2d_m = d_2d
        self.d_3d_m = d_3d

    @classmethod
    def reset(cls) -> "Distances":
        """Create an empty distance container.

        Returns
        -------
        Distances
            New container whose distance arrays are empty.
        """
        return cls()


@dataclass(slots=True)
class UESignalMetrics:
    """
    Data class for UE signal conditions.

    Attributes
    ----------
    pl_db : FloatArr
        Path loss in dB.
    cg_db : FloatArr
        Coupling gain in dB.
    rsrp_dbm : FloatArr
        Reference Signal Received Power (RSRP) in dBm.
    sinr_db : FloatArr
        Signal-to-Interference-plus-Noise Ratio (SINR) in dB.
    """

    pl_db: FloatArr = field(default_factory=lambda: np.empty((0,), dtype=float))
    cg_db: FloatArr = field(default_factory=lambda: np.empty((0,), dtype=float))
    rsrp_dbm: FloatArr = field(default_factory=lambda: np.empty((0,), dtype=float))
    sinr_db: FloatArr = field(default_factory=lambda: np.empty((0,), dtype=float))

    def update(
        self,
        pl_db: FloatArr,
        cg_db: FloatArr,
        rsrp_dbm: FloatArr,
        sinr_db: FloatArr,
    ) -> None:
        """Update all per-link signal metrics.

        Parameters
        ----------
        pl_db : FloatArr
            Path loss in dB.
        cg_db : FloatArr
            Coupling gain in dB.
        rsrp_dbm : FloatArr
            Reference signal received power in dBm.
        sinr_db : FloatArr
            Signal-to-interference-plus-noise ratio in dB.
        """
        self.pl_db = pl_db
        self.cg_db = cg_db
        self.rsrp_dbm = rsrp_dbm
        self.sinr_db = sinr_db

    @classmethod
    def reset(cls) -> "UESignalMetrics":
        """Create an empty signal-metrics container.

        Returns
        -------
        UESignalMetrics
            New container whose metric arrays are empty.
        """
        return cls()


class UserEquipment:
    """
    User equipment (UE) entity.

    Parameters
    ----------
    ue_id : int
        Unique ID of this UE.
    delta_t : float
        Time step in seconds.
    pos_xy : np.ndarray
        Initial 2D position of the UE.
    height : float
        Antenna height of the UE.
    mobility_model : str
        Mobility model of the UE.
    seed : int
        Seed for initalization of random number generator.
    debug : bool
        Enable debug mode. Default is False.
    log_level : int
        Log/Verbosity level. Default is logging.INFO.
    """

    def __init__(
        self,
        ue_id: int,
        config: "UserEquipmentConfig",
        sample_waypoint_fn: Callable[[np.random.Generator], np.ndarray],
    ):
        self.ue_id = ue_id
        self.config = config
        self.height = config.height_m
        self.sample_waypoint_fn = sample_waypoint_fn

        self.p_tx_w = utc.dbm_to_w(config.power_class_dbm)  # type: ignore
        self.rx_nf_db = config.rx_noise_figure_db

        self.mobility_state = UEMobilityState()
        self.mobility_model = self._get_mobility_model()

        self.distances = Distances()
        self.signal_metrics = UESignalMetrics()

        # Initialize random number generator(s)
        self.set_seed(config.seed)

    def _get_mobility_model(self) -> "AbstractMobilityModel":
        """Get the mobility model based on the configuration."""
        if self.config.mobility_model == mobility.ConstPosition.NAME:
            return mobility.ConstPosition(seed=self.config.seed)

        if self.config.mobility_model == mobility.ConstSpeedCorrRandomWalk.NAME:
            return mobility.ConstSpeedCorrRandomWalk(
                speed_kph=self.config.speed_kph,
                max_turn_angle_deg=self.config.max_turn_angle_deg,
                seed=self.config.seed,
            )

        if self.config.mobility_model == mobility.GaussMarkovMobility.NAME:
            return mobility.GaussMarkovMobility(
                mean_speed_kph=self.config.gauss_markov_mean_speed_kph,
                alpha=self.config.gauss_markov_alpha,
                speed_std_kph=self.config.gauss_markov_speed_std_kph,
                angle_std_deg=self.config.gauss_markov_angle_std_deg,
                min_speed_kph=self.config.gauss_markov_min_speed_kph,
                max_speed_kph=self.config.gauss_markov_max_speed_kph,
                seed=self.config.seed,
            )

        if self.config.mobility_model == mobility.RandomSpeedRandomWalk.NAME:
            return mobility.RandomSpeedRandomWalk(
                speed_kph=self.config.speed_kph,
                random_speed_corr=self.config.random_speed_corr,
                random_speed_lim_kph=self.config.random_speed_lim_kph,
                max_turn_angle_deg=self.config.max_turn_angle_deg,
                seed=self.config.seed,
            )

        if self.config.mobility_model == mobility.RandomWaypoint.NAME:
            return mobility.RandomWaypoint(
                sample_waypoint_fn=self.sample_waypoint_fn,
                waypoint_speed_lim_kph=self.config.waypoint_speed_lim_kph,
                waypoint_pause_time_lim_s=self.config.waypoint_pause_time_lim_s,
                waypoint_tolerance_m=self.config.waypoint_tolerance_m,
                seed=self.config.seed,
            )

        raise ValueError(f"Unsupported mobility model: {self.config.mobility_model}")

    def set_initial_position(self, initial_xy: FloatArr) -> None:
        """
        Set the initial position of the UE.

        Parameters
        ----------
        initial_xy : FloatArr
            Initial 2D position of the UE in meters. Shape: (2,).
        """
        initial_angle_rad = self.mobility_model.get_random_angle_rad()
        self.mobility_state.set_initial_position(initial_xy, initial_angle_rad)

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the UE.

        Parameters
        ----------
        seed : int | None
            Seed used to reset the random number generator. If ``None``, a
            new random seed is generated.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
        self.seed = int(seed)

        self.mobility_state = UEMobilityState.reset()
        self.distances.reset()
        self.signal_metrics.reset()

        self.mobility_model.reset(seed=self.seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the UE random seed and reset its state.

        Parameters
        ----------
        seed : int or None
            Random seed. If ``None``, generate a seed from a new NumPy random
            number generator.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        self.seed = seed
        self.reset(seed=seed)

    def update_position(self, time_increment_s: float) -> None:
        """
        Calculate the next UE position and keep the UE in the ROI.
        The UE position is updated based on the mobility model.

        Parameters
        ----------
        time_increment_s : float
            Time increment in seconds for the position update.

        Notes
        -----
        The updated position is not guaranteed to be within the ROI.
        After calling this method, the caller should enforce the ROI constraints
        on the new position.
        """
        self.mobility_model.update_ue_position(
            mobility_state=self.mobility_state, time_increment_s=time_increment_s
        )

    def update_signal_metrics(
        self,
        path_loss_db: FloatArr,
        coupling_gain_db: FloatArr,
        rsrp_dbm: FloatArr,
        sinr_db: FloatArr,
    ) -> None:
        """
        Update signal metrics.

        Parameters
        ----------
        path_loss_db : FloatArr
            Path loss in dB.
        coupling_gain_db : FloatArr
            Coupling gain in dB.
        rsrp_dbm : FloatArr
            RSRP in dBm.
        sinr_db : FloatArr
            SINR in dB.
        """
        self.signal_metrics.update(
            pl_db=path_loss_db,
            cg_db=coupling_gain_db,
            rsrp_dbm=rsrp_dbm,
            sinr_db=sinr_db,
        )

    def update_distances(self, d_2d: FloatArr, d_3d: FloatArr) -> None:
        """
        Update distances.

        Parameters
        ----------
        d_2d : FloatArr
            2D distances in meters. Shape: (n_bs,).
        d_3d : FloatArr
            3D distances in meters. Shape: (n_bs,).
        """
        self.distances.update(d_2d=d_2d, d_3d=d_3d)

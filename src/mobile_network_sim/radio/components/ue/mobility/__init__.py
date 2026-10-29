"""Mobility models for UEs."""

from .abstract_mobility_model import AbstractMobilityModel
from .const_const_position import ConstPosition
from .const_speed_corr_random_walk import ConstSpeedCorrRandomWalk
from .gauss_markov_mobility_model import GaussMarkovMobility
from .random_speed_random_walk import RandomSpeedRandomWalk
from .random_waypoint import RandomWaypoint

__all__ = [
    "AbstractMobilityModel",
    "ConstPosition",
    "ConstSpeedCorrRandomWalk",
    "GaussMarkovMobility",
    "RandomSpeedRandomWalk",
    "RandomWaypoint",
]

mobility_models = (
    ConstPosition.NAME,
    ConstSpeedCorrRandomWalk.NAME,
    GaussMarkovMobility.NAME,
    RandomSpeedRandomWalk.NAME,
    RandomWaypoint.NAME,
)

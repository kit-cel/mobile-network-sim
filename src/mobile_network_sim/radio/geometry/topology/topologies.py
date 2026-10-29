"""Module for radio topologies."""

from .abstract_topology import AbstractTopology
from .euclidean_radio_topology import EuclideanRadioTopology
from .wraparound_topology import WraparoundRadioTopology

topology_dict = {
    AbstractTopology.__name__: AbstractTopology,
    EuclideanRadioTopology.__name__: EuclideanRadioTopology,
    WraparoundRadioTopology.__name__: WraparoundRadioTopology,
}

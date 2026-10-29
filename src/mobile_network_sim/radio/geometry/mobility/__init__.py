"""Mobility handlers for user equipment (UE) in the simulation."""

from .reflect_boundary_handler import ReflectiveBoundary
from .wrap_boundary_handler import WraparoundBoundary

__all__ = [
    "ReflectiveBoundary",
    "WraparoundBoundary",
]

"""
Particle Simulation Package

A comprehensive simulation comparing classical and quantum approaches for:
1. Three-electron system
2. Electron-proton-neutron system (hydrogen-like atom)

This package provides mathematically rigorous implementations of both
classical (Newtonian mechanics) and quantum mechanical simulations.
"""

from .constants import PHYSICAL_CONSTANTS
from .benchmark import SimulationBenchmark

__version__ = "1.0.0"
__all__ = ['PHYSICAL_CONSTANTS', 'SimulationBenchmark']

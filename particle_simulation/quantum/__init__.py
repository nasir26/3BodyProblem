"""
Quantum Simulation Module

Implements quantum mechanical simulations using:
- Time-dependent Schrödinger equation
- Variational methods
- Exact diagonalization
- Hartree-Fock for multi-electron systems
"""

from .three_electron import ThreeElectronQuantum
from .hydrogen_like import HydrogenLikeQuantum

__all__ = ['ThreeElectronQuantum', 'HydrogenLikeQuantum']

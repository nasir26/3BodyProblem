"""
Classical Simulation Module

Implements Newtonian mechanics for particle systems using:
- Coulomb's law for electrostatic interactions
- Classical equations of motion (F = ma)
- Numerical integration (Verlet, RK4)
"""

from .three_electron import ThreeElectronClassical
from .hydrogen_like import HydrogenLikeClassical

__all__ = ['ThreeElectronClassical', 'HydrogenLikeClassical']

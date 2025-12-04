"""Physical constants expressed in atomic units.

The project keeps all numerical work in atomic units (Hartree energy,
Bohr radius, electron mass, elementary charge, and reduced Planck's
constant all equal to 1).  Conversion helpers are provided for
reporting results in SI-friendly units such as electron-volts.
"""
from __future__ import annotations

from dataclasses import dataclass

# Fundamental constants in atomic units (a.u.)
ELECTRON_MASS = 1.0
PROTON_MASS = 1836.15267343
NEUTRON_MASS = 1838.68366158
ELEMENTARY_CHARGE = 1.0  # magnitude; electrons carry -1
COULOMB_COEFFICIENT = 1.0  # 1 / (4 * pi * epsilon_0) in a.u.
HBAR = 1.0

# Conversion helpers
def hartree_to_ev(energy_hartree: float) -> float:
    """Convert Hartree energy to electron-volts."""

    return energy_hartree * 27.211386245988


def ev_to_hartree(energy_ev: float) -> float:
    """Convert electron-volts to Hartree."""

    return energy_ev / 27.211386245988


@dataclass(frozen=True)
class RuntimeMetrics:
    """Timing helper used by the simulators."""

    wall_time_s: float

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return f"{self.wall_time_s:.3f} s"

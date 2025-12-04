"""
Physical Constants Module
=========================
This module provides fundamental physical constants in both SI and atomic units.

Atomic Units (a.u.):
- ℏ = 1 (reduced Planck constant)
- m_e = 1 (electron mass)
- e = 1 (elementary charge)
- 4πε₀ = 1 (permittivity factor)
- a₀ = 1 (Bohr radius)
- E_h = 1 (Hartree energy)

These are the natural units for atomic and molecular physics calculations.
"""

import numpy as np

# ============================================================================
# SI Constants (CODATA 2018 values)
# ============================================================================

# Planck constant
HBAR_SI = 1.054571817e-34  # J·s (reduced Planck constant)
H_SI = 6.62607015e-34  # J·s (Planck constant)

# Elementary charge
E_SI = 1.602176634e-19  # C

# Masses
M_ELECTRON_SI = 9.1093837015e-31  # kg
M_PROTON_SI = 1.67262192369e-27  # kg
M_NEUTRON_SI = 1.67492749804e-27  # kg

# Permittivity
EPSILON_0_SI = 8.8541878128e-12  # F/m

# Coulomb constant
K_COULOMB_SI = 1 / (4 * np.pi * EPSILON_0_SI)  # N·m²/C²

# Bohr radius
A0_SI = 5.29177210903e-11  # m

# Hartree energy
E_HARTREE_SI = 4.3597447222071e-18  # J
E_HARTREE_EV = 27.211386245988  # eV

# Speed of light
C_SI = 299792458  # m/s

# Boltzmann constant
KB_SI = 1.380649e-23  # J/K

# ============================================================================
# Atomic Units (Natural units for atomic physics)
# ============================================================================

# In atomic units, these are all 1
HBAR_AU = 1.0
M_ELECTRON_AU = 1.0
E_AU = 1.0
K_COULOMB_AU = 1.0  # 1/(4πε₀) = 1
A0_AU = 1.0
E_HARTREE_AU = 1.0

# Mass ratios in atomic units
M_PROTON_AU = M_PROTON_SI / M_ELECTRON_SI  # ≈ 1836
M_NEUTRON_AU = M_NEUTRON_SI / M_ELECTRON_SI  # ≈ 1839

# ============================================================================
# Conversion factors
# ============================================================================

# Length conversions
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM

# Energy conversions
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1 / HARTREE_TO_EV
HARTREE_TO_KCAL = 627.509474
HARTREE_TO_KJ = 2625.49964

# Time conversions
AU_TIME_TO_FEMTOSECONDS = 0.02418884326509  # 1 a.u. of time ≈ 24.19 as (attoseconds)

# ============================================================================
# Useful derived quantities
# ============================================================================

# Fine structure constant
ALPHA = 1 / 137.035999084

# Rydberg constant (in terms of Hartree)
RYDBERG = 0.5  # Hartree

# Ionization energy of hydrogen (in Hartree)
E_IONIZATION_H = 0.5  # Hartree = 13.6 eV


def print_constants():
    """Print all major physical constants."""
    print("=" * 60)
    print("PHYSICAL CONSTANTS")
    print("=" * 60)
    print("\nSI Units:")
    print(f"  ℏ = {HBAR_SI:.6e} J·s")
    print(f"  m_e = {M_ELECTRON_SI:.6e} kg")
    print(f"  m_p = {M_PROTON_SI:.6e} kg")
    print(f"  m_n = {M_NEUTRON_SI:.6e} kg")
    print(f"  e = {E_SI:.6e} C")
    print(f"  a₀ = {A0_SI:.6e} m")
    print(f"  E_H = {E_HARTREE_SI:.6e} J = {E_HARTREE_EV:.4f} eV")
    print("\nAtomic Units:")
    print(f"  m_p/m_e = {M_PROTON_AU:.4f}")
    print(f"  m_n/m_e = {M_NEUTRON_AU:.4f}")
    print(f"  α (fine structure) = {ALPHA:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    print_constants()

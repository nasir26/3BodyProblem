"""
Physical constants in SI and atomic units for quantum/classical simulations.
"""

import numpy as np

# SI Units
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
ME = 9.1093837015e-31   # Electron mass (kg)
MP = 1.67262192369e-27  # Proton mass (kg)
MN = 1.67492749804e-27  # Neutron mass (kg)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
K_E = 8.9875517923e9    # Coulomb constant (N·m²/C²)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
A_0 = 5.29177210903e-11  # Bohr radius (m)
E_H = 4.3597447222071e-18  # Hartree energy (J)

# Atomic Units (a.u.) - Natural units for quantum mechanics
# In atomic units: ℏ = mₑ = e = 4πε₀ = 1
AU_HBAR = 1.0
AU_ME = 1.0
AU_E = 1.0
AU_KE = 1.0
AU_A0 = 1.0
AU_EH = 1.0

# Mass ratios in atomic units
AU_MP = MP / ME  # ~1836.15
AU_MN = MN / ME  # ~1838.68

# Conversion factors
EV_TO_HARTREE = 0.0367493
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.8897259886

# Time conversion
AU_TIME_TO_FS = 0.02418884254  # atomic time unit to femtoseconds


class AtomicUnits:
    """Class to work with atomic units for cleaner calculations."""
    
    hbar = 1.0
    m_e = 1.0
    e = 1.0
    k_e = 1.0
    a_0 = 1.0
    E_h = 1.0
    m_p = AU_MP
    m_n = AU_MN
    
    @staticmethod
    def energy_to_eV(E_hartree):
        """Convert energy from Hartree to eV."""
        return E_hartree * HARTREE_TO_EV
    
    @staticmethod
    def length_to_angstrom(r_bohr):
        """Convert length from Bohr to Angstrom."""
        return r_bohr * BOHR_TO_ANGSTROM

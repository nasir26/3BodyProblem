"""
Physical Constants Module

Contains all fundamental physical constants used in the simulations.
All values are in SI units unless otherwise specified.
"""

import numpy as np

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (SI Units)
# =============================================================================

PHYSICAL_CONSTANTS = {
    # Planck's constant
    'h': 6.62607015e-34,        # J⋅s (Planck constant)
    'hbar': 1.054571817e-34,    # J⋅s (reduced Planck constant, h/2π)
    
    # Electromagnetic constants
    'e': 1.602176634e-19,       # C (elementary charge)
    'epsilon_0': 8.8541878128e-12,  # F/m (vacuum permittivity)
    'k_e': 8.9875517923e9,      # N⋅m²/C² (Coulomb constant, 1/(4πε₀))
    'c': 299792458,             # m/s (speed of light)
    
    # Particle masses
    'm_e': 9.1093837015e-31,    # kg (electron mass)
    'm_p': 1.67262192369e-27,   # kg (proton mass)
    'm_n': 1.67492749804e-27,   # kg (neutron mass)
    
    # Bohr model constants
    'a_0': 5.29177210903e-11,   # m (Bohr radius)
    'E_h': 4.3597447222071e-18, # J (Hartree energy)
    
    # Fine structure constant
    'alpha': 7.2973525693e-3,   # dimensionless
    
    # Boltzmann constant
    'k_B': 1.380649e-23,        # J/K
}

# =============================================================================
# ATOMIC UNITS (for quantum calculations)
# =============================================================================
# In atomic units: ℏ = m_e = e = k_e = 1

ATOMIC_UNITS = {
    'hbar': 1.0,
    'm_e': 1.0,
    'e': 1.0,
    'k_e': 1.0,
    'a_0': 1.0,     # Length unit (Bohr radius)
    'E_h': 1.0,     # Energy unit (Hartree)
    't_au': 2.4188843265857e-17,  # Time unit in seconds
}

# Conversion factors
def si_to_atomic(value, unit_type):
    """
    Convert SI units to atomic units.
    
    Parameters:
    -----------
    value : float
        Value in SI units
    unit_type : str
        Type of unit: 'length', 'energy', 'time', 'mass'
    
    Returns:
    --------
    float
        Value in atomic units
    """
    conversions = {
        'length': 1 / PHYSICAL_CONSTANTS['a_0'],
        'energy': 1 / PHYSICAL_CONSTANTS['E_h'],
        'time': 1 / ATOMIC_UNITS['t_au'],
        'mass': 1 / PHYSICAL_CONSTANTS['m_e'],
    }
    return value * conversions.get(unit_type, 1.0)

def atomic_to_si(value, unit_type):
    """
    Convert atomic units to SI units.
    
    Parameters:
    -----------
    value : float
        Value in atomic units
    unit_type : str
        Type of unit: 'length', 'energy', 'time', 'mass'
    
    Returns:
    --------
    float
        Value in SI units
    """
    conversions = {
        'length': PHYSICAL_CONSTANTS['a_0'],
        'energy': PHYSICAL_CONSTANTS['E_h'],
        'time': ATOMIC_UNITS['t_au'],
        'mass': PHYSICAL_CONSTANTS['m_e'],
    }
    return value * conversions.get(unit_type, 1.0)


# =============================================================================
# PARTICLE PROPERTIES
# =============================================================================

class Particle:
    """Represents a particle with its physical properties."""
    
    def __init__(self, name, mass, charge, spin=0.5):
        """
        Initialize a particle.
        
        Parameters:
        -----------
        name : str
            Particle name
        mass : float
            Mass in kg
        charge : float
            Charge in Coulombs
        spin : float
            Spin quantum number (default 0.5 for fermions)
        """
        self.name = name
        self.mass = mass
        self.charge = charge
        self.spin = spin
        
        # Mass in atomic units
        self.mass_au = mass / PHYSICAL_CONSTANTS['m_e']
        
        # Charge in atomic units (units of e)
        self.charge_au = charge / PHYSICAL_CONSTANTS['e']
    
    def __repr__(self):
        return f"Particle({self.name}, m={self.mass:.4e} kg, q={self.charge:.4e} C)"


# Pre-defined particles
ELECTRON = Particle('electron', PHYSICAL_CONSTANTS['m_e'], -PHYSICAL_CONSTANTS['e'], spin=0.5)
PROTON = Particle('proton', PHYSICAL_CONSTANTS['m_p'], PHYSICAL_CONSTANTS['e'], spin=0.5)
NEUTRON = Particle('neutron', PHYSICAL_CONSTANTS['m_n'], 0, spin=0.5)


if __name__ == "__main__":
    # Print constants for verification
    print("Physical Constants (SI):")
    for name, value in PHYSICAL_CONSTANTS.items():
        print(f"  {name}: {value:.6e}")
    
    print("\nPre-defined particles:")
    print(f"  {ELECTRON}")
    print(f"  {PROTON}")
    print(f"  {NEUTRON}")

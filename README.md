# Classical vs Quantum Simulation: Particle Systems

A comprehensive physics simulation comparing classical and quantum mechanical approaches for modeling atomic-scale systems.

## Overview

This project implements mathematically rigorous simulations of two physical systems:

### System 1: Three-Electron System
A quantum dot model with three electrons confined by a harmonic potential.

### System 2: Electron-Proton-Neutron System  
A simplified deuterium-like system with hydrogen-like electron and nuclear binding.

## Physics Background

### Classical Mechanics Approach
- Particles as point masses with definite positions and momenta
- Forces from potential gradients (Coulomb, Yukawa)
- Newton's equations integrated via velocity Verlet (symplectic)
- Energy conservation as numerical check

### Quantum Mechanics Approach
- Wave functions with probability interpretation
- Schrödinger equation (exact for hydrogen)
- Variational Monte Carlo for many-electron systems
- Exchange and correlation effects included

## Key Results

| Criterion | Classical | Quantum | Winner |
|-----------|-----------|---------|--------|
| **Speed** | ~5-100x faster | Slower | Classical |
| **Accuracy** | ~10-50% error | Exact or <1% | Quantum |
| **Physics** | Missing exchange/correlation | Complete | Quantum |
| **Stability** | Electron radiates (unphysical) | Stable ground state | Quantum |

### Critical Finding
**For atomic-scale systems, quantum simulation is essential.** Classical mechanics:
- Cannot explain atomic stability (electron would spiral into nucleus)
- Misses exchange energy (Pauli exclusion principle)
- Misses correlation energy (electron entanglement)
- Cannot predict discrete energy levels

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0 (optional, for visualization)

## Usage

### Full Comparison
```bash
python main.py --full
```

### Individual Simulations with Details
```bash
python main.py --individual
```

### Mathematical Formulations
```bash
python main.py --math
```

### Quick Test
```bash
python main.py --quick
```

### Generate Visualization
```bash
python main.py --full --visualize
```

## File Structure

```
├── main.py                  # Main entry point
├── constants.py             # Physical constants (SI and atomic units)
├── classical_simulation.py  # Classical mechanics implementation
├── quantum_simulation.py    # Quantum mechanics implementation
├── comparison.py            # Comparison framework and analysis
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Mathematical Details

### Atomic Units (Used Throughout)
- ℏ = mₑ = e = 4πε₀ = 1
- Length: 1 Bohr = 0.529 Å
- Energy: 1 Hartree = 27.21 eV

### Classical Hamiltonian
```
H = Σᵢ pᵢ²/(2mᵢ) + V(r₁, r₂, ...)

V_Coulomb = qᵢqⱼ/rᵢⱼ
V_Yukawa = -V₀ exp(-r/a)/(r/a)
```

### Quantum Hamiltonian
```
Ĥ = -½∇² + V(r)    (atomic units)

Ψ_hydrogen = (1/√π) exp(-r)
E_hydrogen = -0.5 Hartree = -13.6 eV (exact)
```

### Variational Monte Carlo
```
E ≈ ⟨E_local⟩ = ⟨(ĤΨ)/Ψ⟩

Ψ_trial = det|φᵢ(rⱼ)| × exp(Σᵢ<ⱼ u(rᵢⱼ))
        = Slater × Jastrow
```

## Physical Interpretation

### Three-Electron System
- **Classical**: Electrons oscillate in harmonic trap, repel via Coulomb
- **Quantum**: Ground state with Fermi statistics (antisymmetric Ψ)
- **Key difference**: Quantum includes exchange energy (~0.5 Hartree)

### Electron-Proton-Neutron System  
- **Classical electron**: Kepler orbits, but would radiate and collapse
- **Quantum electron**: Stable 1s orbital, exact -13.6 eV
- **Nuclear binding**: Requires quantum (strong force = QCD)

## Verification

### Hydrogen (Quantum)
- Energy: -0.5 Hartree (exact, verified to machine precision)
- ⟨r⟩: 1.5 Bohr (exact)
- Heisenberg: ΔxΔp > 0.5 (verified)

### Classical Energy Conservation
- Symplectic integrator: <10⁻⁶ relative drift
- Verifies numerical correctness

## Limitations

1. **Classical**: Fundamentally wrong for ground states
2. **VMC**: Statistical errors, limited by trial wave function
3. **Nuclear**: Simplified Yukawa potential (not full QCD)
4. **Relativity**: Not included (valid for light atoms)

## References

1. Helgaker, T., Jørgensen, P., & Olsen, J. (2014). *Molecular Electronic-Structure Theory*
2. Foulkes, W. M. C., et al. (2001). "Quantum Monte Carlo simulations of solids", *Rev. Mod. Phys.*
3. Griffiths, D. J. (2018). *Introduction to Quantum Mechanics*

## License

MIT License - See LICENSE file

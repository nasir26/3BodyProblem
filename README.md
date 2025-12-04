# Classical vs Quantum Simulation Comparison

A complete, mathematically rigorous simulation framework comparing classical and quantum mechanical approaches for atomic systems.

## Overview

This project implements and compares **Classical** and **Quantum** simulations for two distinct atomic systems:

### System 1: Three-Electron System (Lithium-like, Z=3)

| Approach | Method | Energy | Accuracy |
|----------|--------|--------|----------|
| **Classical** | Molecular Dynamics (Coulomb) | ~-15.4 Ha | 106% error |
| **Quantum** | Variational (2-parameter) | ~-10.4 Ha | 39% error |
| **Exact** | Full CI (reference) | -7.478 Ha | — |

### System 2: Electron-Proton-Neutron System (Hydrogen-like)

| Approach | Method | Energy | Accuracy |
|----------|--------|--------|----------|
| **Classical** | Bohr Model Dynamics | -0.500 Ha | Exact! |
| **Quantum** | Schrödinger Equation | -0.500 Ha | Exact |

## Physical Models

### Classical Mechanics

The classical simulation treats particles as point charges following Newton's laws:

```
F = m·a = -∇V(r)

V(r) = Σᵢ<ⱼ qᵢqⱼ/(4πε₀|rᵢ - rⱼ|) - Σᵢ Zqᵢ/(4πε₀|rᵢ|)
```

**Limitations:**
- No wave-particle duality
- No quantized energy levels
- No Pauli exclusion (exchange)
- No Heisenberg uncertainty
- No zero-point energy

### Quantum Mechanics

The quantum simulation solves the Schrödinger equation:

```
ĤΨ = EΨ

Ĥ = -ℏ²/2m Σᵢ∇ᵢ² + V(r₁, r₂, ...)
```

**Methods implemented:**
- **Variational principle**: Minimize ⟨Ψ|Ĥ|Ψ⟩/⟨Ψ|Ψ⟩
- **Hartree-Fock**: Self-consistent field with exchange
- **Perturbation theory**: First-order corrections
- **Analytical solutions**: Exact for hydrogen

## Mathematical Framework

### Atomic Units

All calculations use atomic units (a.u.) where:
- ℏ = 1 (reduced Planck constant)
- mₑ = 1 (electron mass)  
- e = 1 (elementary charge)
- 4πε₀ = 1 (Coulomb constant)
- a₀ = 1 Bohr radius ≈ 0.529 Å
- Eₕ = 1 Hartree ≈ 27.21 eV

### Three-Electron System

**Classical Hamiltonian:**
```
H = Σᵢ pᵢ²/2mₑ + Σᵢ<ⱼ e²/rᵢⱼ - Σᵢ Ze²/rᵢ
```

**Quantum Variational Wavefunction:**
```
Ψ = |1s(ζ₁)↑, 1s(ζ₁)↓, 2s(ζ₂)↑⟩  (Slater determinant)

φ₁ₛ(r) = (ζ³/π)^(1/2) exp(-ζr)
φ₂ₛ(r) = (ζ³/32π)^(1/2) (2-ζr) exp(-ζr/2)
```

**Variational Energy Functional:**
```
E[ζ₁, ζ₂] = 2·E₁ₛ(ζ₁) + E₂ₛ(ζ₂) + J₁₁ + 2J₁₂ - K₁₂

where:
- E₁ₛ = ζ₁²/2 - Zζ₁ (1s one-electron energy)
- E₂ₛ = ζ₂²/8 - Zζ₂/2 (2s one-electron energy)  
- J₁₁ = 5ζ₁/8 (1s-1s Coulomb)
- J₁₂ = 17ζₐᵥ/81 (1s-2s Coulomb)
- K₁₂ = exchange integral
```

### Hydrogen Atom (Electron-Proton-Neutron)

**Exact Quantum Solution:**
```
Eₙ = -Z²μ/2n² Hartree

where μ = mₑM/(mₑ + M) is the reduced mass
```

**Radial Wavefunctions:**
```
Rₙₗ(r) = Nₙₗ (2Zr/na₀)^l Lₙ₋ₗ₋₁^(2l+1)(2Zr/na₀) exp(-Zr/na₀)
```

**Classical Bohr Model:**
```
E = -Z²mₑe⁴/(2ℏ²n²) = -Z²/2n² Hartree

rₙ = n²a₀/Z
vₙ = Z/(na₀)
```

## Results Summary

### Speed Comparison

| System | Classical | Quantum | Ratio |
|--------|-----------|---------|-------|
| 3-electron | ~0.6 s | ~0.001 s | Quantum 500x faster* |
| e-p-n | ~0.3 s | ~0.06 s | Quantum 5x faster |

*Using analytical integrals vs. numerical integration

### Accuracy Comparison

| System | Classical Error | Quantum Error | Winner |
|--------|-----------------|---------------|--------|
| 3-electron | 106% | 39% | **Quantum** |
| e-p-n | 0.03% | 0.03% | Tie |

### Physical Correctness

| Feature | Classical | Quantum |
|---------|-----------|---------|
| Discrete energy levels | ❌ | ✅ |
| Heisenberg uncertainty | ❌ | ✅ |
| Pauli exclusion | ❌ | ✅ |
| Probability distributions | ❌ | ✅ |
| Zero-point energy | ❌ | ✅ |
| Exchange interaction | ❌ | ✅ |
| Stable atoms | ❌ | ✅ |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantum-classical-simulation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Comparison

```bash
python3 run_simulation.py
```

This will:
1. Run both classical and quantum simulations for both systems
2. Generate comparison analysis
3. Save results to `simulation_results.json`
4. Generate visualization figures (PNG files)

### Run Individual Systems

```python
# Three-electron system
from three_electron_system import compare_classical_quantum
results_3e = compare_classical_quantum(nuclear_charge=3.0)

# Electron-proton-neutron system
from epn_system import compare_classical_quantum_epn
results_epn = compare_classical_quantum_epn()
```

### Examine Physical Constants

```python
from physics_constants import print_constants
print_constants()
```

## File Structure

```
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── physics_constants.py       # Physical constants in SI and atomic units
├── three_electron_system.py   # 3-electron simulation (classical + quantum)
├── epn_system.py              # e-p-n simulation (classical + quantum)
├── visualization.py           # Plotting and visualization
├── run_simulation.py          # Main entry point
└── simulation_results.json    # Output results (generated)
```

## Key Insights

### Why Quantum Mechanics Wins for Multi-Electron Systems

1. **Exchange Antisymmetry**: Fermions (electrons) require antisymmetric wavefunctions. The Pauli exclusion principle emerges naturally from this symmetry.

2. **Correlation Effects**: Electrons avoid each other due to Coulomb repulsion. Quantum mechanics captures this through correlated wavefunctions.

3. **Variational Principle**: The true ground state energy is a lower bound, allowing systematic improvement.

### Why Bohr Model Works for Hydrogen

The Bohr model gives the **exact ground state energy** for hydrogen because:
- Circular orbits happen to match the n=1 quantum state energy
- Angular momentum quantization (L = nℏ) is built in
- For a single electron, no exchange or correlation issues

However, Bohr model fails to predict:
- Orbital shapes (s, p, d, f)
- Selection rules for transitions
- Fine structure and spin-orbit coupling
- Probability distributions

## Conclusions

### When to Use Classical Mechanics

- Fast, qualitative dynamics
- Large systems (molecular dynamics)
- High-temperature limits
- When quantum effects average out

### When to Use Quantum Mechanics

- Accurate energies and spectra
- Chemistry and bonding
- Electronic structure
- Low temperatures
- Nanoscale systems

## References

1. Griffiths, D.J. "Introduction to Quantum Mechanics" (Cambridge, 2018)
2. Szabo & Ostlund "Modern Quantum Chemistry" (Dover, 1996)
3. NIST Atomic Spectra Database: https://physics.nist.gov/asd

## License

MIT License - See LICENSE file for details.

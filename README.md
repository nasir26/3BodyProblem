# Classical vs Quantum Simulation Comparison

A comprehensive simulation framework comparing classical (Newtonian) and quantum (Schrödinger equation) mechanics for two different physical systems:

1. **3-Electron System**: Three identical electrons interacting via Coulomb forces
2. **Electron-Proton-Neutron System**: A mixed system with one electron, one proton, and one neutron

## Overview

This project implements mathematically correct simulations using:
- **Classical Mechanics**: Newton's laws with Coulomb force interactions, solved using Runge-Kutta integration
- **Quantum Mechanics**: Time-dependent Schrödinger equation with Hartree-Fock approximation

The framework compares both approaches in terms of:
- **Speed**: Computation time for each method
- **Accuracy**: Energy conservation and trajectory stability
- **Physical Validity**: Which approach better describes the system behavior

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation:

```bash
python main_simulation.py
```

This will:
1. Simulate both systems using classical mechanics
2. Simulate both systems using quantum mechanics
3. Compare results in terms of speed and accuracy
4. Generate visualization plots

## Project Structure

- `classical_simulation.py`: Classical mechanics implementation using Newton's laws
- `quantum_simulation.py`: Quantum mechanics implementation using Schrödinger equation
- `comparison_framework.py`: Comparison metrics and analysis
- `main_simulation.py`: Main runner script with visualization

## Mathematical Foundations

### Classical Mechanics
- Uses Newton's second law: **F = ma**
- Coulomb force: **F = k·q₁·q₂/r²**
- Solved using Runge-Kutta 4th/5th order method
- Energy: **E = T + V** (kinetic + potential)

### Quantum Mechanics
- Time-dependent Schrödinger equation: **iℏ ∂ψ/∂t = Hψ**
- Hamiltonian: **H = T + V** (kinetic + potential operators)
- Uses finite difference method for spatial discretization
- Hartree-Fock approximation for multi-particle systems

## Results

The simulation provides detailed comparisons including:
- Computation time for each method
- Energy conservation error (should be zero for isolated systems)
- Trajectory stability and boundedness
- Visual plots of energy evolution and particle trajectories

## Physical Constants

All simulations use accurate physical constants:
- Coulomb constant: k = 8.9875517923×10⁹ N⋅m²/C²
- Elementary charge: e = 1.602176634×10⁻¹⁹ C
- Electron mass: mₑ = 9.1093837015×10⁻³¹ kg
- Proton mass: mₚ = 1.67262192369×10⁻²⁷ kg
- Neutron mass: mₙ = 1.67492749804×10⁻²⁷ kg
- Reduced Planck constant: ℏ = 1.054571817×10⁻³⁴ J⋅s

## Notes

- Quantum simulations use 1D approximation for computational feasibility (full 3D would require grid_size³ points)
- Classical simulations are fully 3D
- Both methods use adaptive time-stepping for numerical accuracy
- Energy conservation is a key metric for accuracy assessment

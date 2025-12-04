# Classical vs Quantum Simulation Comparison

A comprehensive simulation framework comparing classical mechanics and quantum mechanics approaches for atomic-scale multi-particle systems.

## Overview

This project implements and compares two different simulation approaches:

1. **Classical Mechanics**: Uses Newton's laws and Coulomb's law to simulate particle dynamics
2. **Quantum Mechanics**: Uses the time-dependent Schrödinger equation to simulate quantum wave function evolution

## Systems Simulated

### 1. Three-Electron System
- Three identical electrons with negative charge
- Initial configuration: Equilateral triangle arrangement
- Tests quantum effects in identical fermion systems

### 2. Electron-Proton-Neutron System
- One electron (negative charge)
- One proton (positive charge)
- One neutron (neutral)
- Initial configuration: Linear arrangement
- Tests quantum effects in mixed particle systems

## Features

- **Mathematically Correct Models**: 
  - Classical: Newtonian dynamics with Coulomb forces
  - Quantum: Time-dependent Schrödinger equation with proper Hamiltonian
  
- **Performance Comparison**:
  - Computation time measurement
  - Speed comparison between methods
  
- **Accuracy Analysis**:
  - Energy conservation (classical)
  - Energy stability (quantum)
  - Position evolution tracking

- **Visualization**:
  - Energy evolution plots
  - Position evolution plots
  - Speed comparison charts
  - Summary comparison plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete simulation comparison:

```bash
python main.py
```

This will:
1. Run classical and quantum simulations for both systems
2. Compare speed and accuracy
3. Generate detailed text report (`simulation_report.txt`)
4. Create visualization plots:
   - `comparison_3electron.png`
   - `comparison_electron_proton_neutron.png`
   - `summary_comparison.png`

## Project Structure

- `classical_simulation.py`: Classical mechanics implementation
- `quantum_simulation.py`: Quantum mechanics implementation
- `comparison_framework.py`: Comparison and benchmarking tools
- `main.py`: Main script to run all simulations
- `requirements.txt`: Python dependencies

## Key Findings

The simulations demonstrate:

1. **Speed**: Classical mechanics is typically faster for simple systems due to fewer computational operations
2. **Accuracy**: Quantum mechanics is more accurate for atomic-scale systems where quantum effects are significant
3. **Energy Conservation**: Both methods maintain energy conservation within numerical precision
4. **System-Dependent**: The choice between classical and quantum depends on:
   - Required accuracy
   - Computational resources
   - Scale of the system (quantum effects more important for light particles)

## Mathematical Models

### Classical Mechanics
- **Equations**: Newton's second law: F = ma
- **Force**: Coulomb's law: F = k·q₁·q₂/r²
- **Integration**: Runge-Kutta 4th/5th order (RK45)

### Quantum Mechanics
- **Equation**: Time-dependent Schrödinger equation: iħ ∂ψ/∂t = Hψ
- **Hamiltonian**: H = -ħ²/(2m) ∇² + V(x)
- **Potential**: Coulomb potential: V = k·q₁·q₂/r
- **Integration**: Runge-Kutta 4th/5th order (RK45)

## Physical Constants Used

- Coulomb constant: k = 8.99×10⁹ N⋅m²/C²
- Elementary charge: e = 1.602×10⁻¹⁹ C
- Reduced Planck constant: ħ = 1.055×10⁻³⁴ J⋅s
- Electron mass: mₑ = 9.109×10⁻³¹ kg
- Proton mass: mₚ = 1.673×10⁻²⁷ kg
- Neutron mass: mₙ = 1.675×10⁻²⁷ kg

## Results Interpretation

The comparison report provides:
- Computation time for each method
- Energy conservation/stability metrics
- Position evolution tracking
- Recommendations for which method to use based on the system

## Notes

- The quantum simulation uses a 1D approximation for computational efficiency
- Full 3D quantum simulations would require significantly more computational resources
- Both simulations use appropriate time scales (femtoseconds) for atomic dynamics
- Energy conservation is a key metric for accuracy assessment

## License

This project is provided for educational and research purposes.

# Classical vs Quantum Simulation Comparison

A comprehensive simulation framework comparing classical mechanics and quantum mechanics approaches for two different multi-particle systems.

## Systems Simulated

1. **3-Electron System**: Three electrons interacting via Coulomb forces
2. **Electron-Proton-Neutron System**: An atomic-like system with one electron, one proton, and one neutron

## Features

- **Classical Mechanics Simulation**: Uses Newtonian mechanics with Coulomb force calculations
- **Quantum Mechanics Simulation**: 
  - Hartree-Fock method for many-electron systems
  - Variational Monte Carlo for correlated systems
  - Exact Schrödinger equation solutions for hydrogen-like systems
- **Comprehensive Comparison**: 
  - Speed analysis (computation time)
  - Accuracy analysis (energy values, conservation)
  - Physical correctness evaluation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation script:

```bash
python main.py
```

This will:
1. Run classical simulations for both systems
2. Run quantum simulations for both systems
3. Compare speed and accuracy
4. Generate a comprehensive report
5. Save results to `comparison_report.txt` and `results.json`
6. Generate visualization plots (if matplotlib is available)

## Project Structure

```
.
├── classical_simulation.py    # Classical mechanics implementation
├── quantum_simulation.py      # Quantum mechanics implementation
├── comparison.py              # Comparison framework
├── main.py                    # Main script to run all simulations
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Mathematical Foundations

### Classical Mechanics

The classical simulation uses Newton's laws of motion:

```
F = ma
F = k * q1 * q2 / r²
```

Where:
- `F` is the Coulomb force
- `k` is the Coulomb constant
- `q1`, `q2` are particle charges
- `r` is the distance between particles

The equations of motion are integrated using:
- `scipy.integrate.odeint` (Runge-Kutta method)
- Alternative Verlet integration method

### Quantum Mechanics

#### For 3-Electron System:
- **Hartree-Fock Method**: Self-consistent field approximation
  - Fock operator: `F = H + Σ(2J_j - K_j)`
  - J: Coulomb operator
  - K: Exchange operator
- **Variational Monte Carlo**: Correlated wavefunction approach
  - Trial wavefunction with Jastrow factor for electron correlation

#### For Electron-Proton-Neutron System:
- **Schrödinger Equation**: Time-independent form
  ```
  Hψ = Eψ
  H = -ℏ²/(2m)∇² + V(r)
  ```
- **Born-Oppenheimer Approximation**: Heavy particles (proton, neutron) treated as stationary
- **Analytical Comparison**: Hydrogen atom ground state energy `E = -0.5 * μ` (in atomic units)

## Key Findings

### Speed Comparison
- Classical mechanics is typically faster for simple time evolution
- Quantum mechanics requires iterative/self-consistent methods (slower)
- Speed depends on system complexity and desired accuracy

### Accuracy Comparison
- **Quantum mechanics is physically correct** for atomic/molecular systems
- **Classical mechanics fails** for quantum systems because it:
  - Violates Pauli exclusion principle (for electrons)
  - Predicts unphysical collapse of atoms
  - Cannot describe bound states properly
  - Misses quantum effects (tunneling, discrete energy levels, etc.)

### Energy Values
- Classical and quantum energies differ significantly
- Quantum ground state energy is typically lower (more stable)
- Classical systems have continuous energy (not quantized)

## Physical Units

All calculations use **atomic units**:
- Length: Bohr radius (a₀ = 5.29 × 10⁻¹¹ m)
- Energy: Hartree (E_h = 4.36 × 10⁻¹⁸ J)
- Mass: Electron mass (m_e = 9.11 × 10⁻³¹ kg)
- Charge: Elementary charge (e = 1.60 × 10⁻¹⁹ C)
- Time: Atomic time unit (2.42 × 10⁻¹⁷ s)

## Output Files

- `comparison_report.txt`: Detailed text report with all results
- `results.json`: Structured JSON data with all simulation results
- `plots/comparison_plots.png`: Visualization plots (if generated)

## Limitations and Approximations

1. **1D Approximation**: Full 3D calculations are computationally expensive, so simplified 1D grids are used
2. **Finite Grid**: Spatial discretization limits accuracy
3. **Hartree-Fock**: Mean-field approximation, misses some electron correlation
4. **Classical Simulation**: Does not account for quantum effects at all

## Extending the Code

To add new systems:
1. Create particle configurations in `classical_simulation.py`
2. Add corresponding quantum system class in `quantum_simulation.py`
3. Add comparison method in `comparison.py`
4. Update `main.py` to include new system

## References

- Quantum Mechanics: Griffiths, "Introduction to Quantum Mechanics"
- Computational Methods: Szabo & Ostlund, "Modern Quantum Chemistry"
- Classical Mechanics: Goldstein, "Classical Mechanics"
- Numerical Methods: Press et al., "Numerical Recipes"

## License

This project is provided for educational and research purposes.

## Author

Simulation framework for comparing classical and quantum mechanical approaches to multi-particle systems.

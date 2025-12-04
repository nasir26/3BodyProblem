# Particle Simulation: Classical vs Quantum Mechanics Comparison

A comprehensive physics simulation framework comparing classical Newtonian mechanics with quantum mechanics for atomic-scale particle systems.

## 🎯 Overview

This project implements mathematically rigorous simulations of two particle systems:

1. **Three-Electron System**: Three interacting electrons with mutual Coulomb repulsion
2. **Hydrogen-Like Atom**: An electron, proton, and neutron system (similar to deuterium)

Both classical and quantum mechanical approaches are implemented, with detailed comparisons of speed and accuracy.

## 🔬 Physical Systems

### System 1: Three Interacting Electrons

**Classical Model:**
- Electrons as point charges with mass m_e = 9.109 × 10⁻³¹ kg
- Coulomb repulsion: F = k_e × e² / r²
- Equations of motion: m × d²r/dt² = ΣF
- Integration: Velocity Verlet algorithm (symplectic, O(dt²))

**Quantum Model:**
- Hamiltonian: Ĥ = Σᵢ T̂ᵢ + Σᵢ<ⱼ Vᵢⱼ
- Variational Monte Carlo with Gaussian/Slater-Jastrow trial wavefunctions
- Hartree-Fock approximation for mean-field treatment

### System 2: Hydrogen-Like Atom (e-p-n)

**Classical Model:**
- Bohr-like orbital mechanics
- Electron orbiting nucleus (proton + neutron)
- Energy: E = -k_e × e² / (2r)

**Quantum Model:**
- Exact analytical solutions: ψ_nlm(r,θ,φ) = R_nl(r) × Y_lm(θ,φ)
- Energy levels: E_n = -13.6 eV / n²
- Numerical matrix diagonalization verification

## 📊 Key Results

### Speed Comparison

| System | Classical | Quantum |
|--------|-----------|---------|
| Three-Electron | ~0.05 s | ~2-5 s (VMC) |
| Hydrogen-Like | ~0.1 s | <0.001 s (analytical!) |

### Accuracy Comparison

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Energy Quantization | ❌ Missing | ✅ Exact |
| Uncertainty Principle | ❌ Missing | ✅ Included |
| Exchange Interaction | ❌ Missing | ✅ Included |
| Atomic Stability | ❌ Cannot explain | ✅ Explains |
| Experimental Agreement | ❌ Wrong | ✅ Perfect |

### Verdict

> **For atomic-scale systems: ALWAYS use quantum mechanics.**
> 
> Classical mechanics is fundamentally incorrect at this scale. Speed advantages are meaningless when results are wrong.

## 📁 Project Structure

```
particle_simulation/
├── __init__.py              # Package initialization
├── constants.py             # Physical constants (SI & atomic units)
├── benchmark.py             # Comparison framework
├── visualization.py         # Plotting utilities
├── classical/
│   ├── __init__.py
│   ├── three_electron.py    # Classical 3e⁻ simulation
│   └── hydrogen_like.py     # Classical H-like simulation
└── quantum/
    ├── __init__.py
    ├── three_electron.py    # Quantum 3e⁻ simulation (VMC, HF)
    └── hydrogen_like.py     # Quantum H-like (exact + numerical)

main.py                      # Main simulation runner
requirements.txt             # Dependencies
README.md                    # This file
```

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd particle_simulation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Run Full Comparison

```bash
python main.py
```

### Run Specific System

```bash
# Three-electron system only
python main.py --system 3e

# Hydrogen-like atom only
python main.py --system hydrogen
```

### Generate Visualizations

```bash
python main.py --visualize
```

### Quick Run (fewer Monte Carlo samples)

```bash
python main.py --quick
```

### Full Benchmark Suite

```bash
python main.py --benchmark
```

## 🧮 Mathematical Details

### Classical Mechanics

**Newton's Second Law:**
```
m_i × (d²r_i/dt²) = Σⱼ F_ij
```

**Coulomb Force:**
```
F_ij = k_e × q_i × q_j / |r_i - r_j|² × r̂_ij
```

**Velocity Verlet Integration:**
```
v(t + dt/2) = v(t) + (dt/2) × a(t)
r(t + dt) = r(t) + dt × v(t + dt/2)
a(t + dt) = F(r(t + dt)) / m
v(t + dt) = v(t + dt/2) + (dt/2) × a(t + dt)
```

### Quantum Mechanics

**Time-Independent Schrödinger Equation:**
```
Ĥψ = Eψ
```

**Hydrogen Atom Hamiltonian:**
```
Ĥ = -ℏ²/(2μ)∇² - k_e×e²/r
```

**Energy Levels:**
```
E_n = -μ Z² e⁴ / (2ℏ² n²) = -13.6 eV / n²  (for hydrogen)
```

**Variational Principle:**
```
E₀ ≤ ⟨ψ_trial|Ĥ|ψ_trial⟩ / ⟨ψ_trial|ψ_trial⟩
```

### Atomic Units

For numerical stability, calculations use atomic units:
- ℏ = 1 (reduced Planck constant)
- m_e = 1 (electron mass)
- e = 1 (elementary charge)
- k_e = 1 (Coulomb constant)
- a₀ = 1 (Bohr radius = 5.29 × 10⁻¹¹ m)
- E_h = 1 (Hartree = 27.2 eV)

## 🔍 Physical Insights

### Why Classical Mechanics Fails

1. **Ultraviolet Catastrophe**: Classically, accelerating charges radiate energy. An orbiting electron should spiral into the nucleus in ~10⁻¹¹ seconds.

2. **No Quantization**: Classical mechanics allows any energy, but atoms only emit/absorb specific wavelengths (spectral lines).

3. **No Uncertainty**: Classical positions and momenta are exact, but Heisenberg's principle says Δx × Δp ≥ ℏ/2.

4. **No Exchange**: Identical particles in classical mechanics are distinguishable, but electrons obey Fermi-Dirac statistics.

### Why Quantum Mechanics Succeeds

1. **Stability**: Wave function cannot collapse to a point (kinetic energy increases).

2. **Quantization**: Boundary conditions on wave functions → discrete energy levels.

3. **Uncertainty**: Wave nature naturally includes position/momentum uncertainty.

4. **Exchange**: Antisymmetric wave functions automatically include Pauli exclusion.

## 📈 Output Examples

### Three-Electron Classical Trajectory
The simulation tracks three electrons repelling each other:
```
Initial Energy: 0.577350 Hartree
Final Energy: 0.577351 Hartree
Energy Conservation Error: 2.5e-06
```

### Hydrogen Atom Quantum Levels
```
n=1: E = -0.499863 Hartree = -13.599 eV
n=2: E = -0.124966 Hartree = -3.400 eV
n=3: E = -0.055540 Hartree = -1.511 eV
```

## 🛠️ Extending the Code

### Adding New Trial Wavefunctions

```python
def my_trial_wavefunction(r: np.ndarray, alpha: float, beta: float) -> float:
    """Custom trial wavefunction for VMC."""
    # Your implementation here
    return psi_value

# Use it:
result = sim.variational_monte_carlo(my_trial_wavefunction, (0.5, 0.3))
```

### Adding New Physical Systems

1. Create new module in `classical/` or `quantum/`
2. Implement required methods: `initialize_state()`, `run_simulation()`
3. Add to `benchmark.py` for comparisons

## 📚 References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*. Cambridge University Press.
2. Thijssen, J.M. (2007). *Computational Physics*. Cambridge University Press.
3. Foulkes, W.M.C. et al. (2001). "Quantum Monte Carlo simulations of solids". *Rev. Mod. Phys.* 73, 33.
4. NIST Physical Constants: https://physics.nist.gov/cuu/Constants/

## 📄 License

MIT License - feel free to use and modify for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More sophisticated quantum methods (DMQMC, FCIQMC)
- Relativistic corrections
- More particle systems
- Interactive visualizations

---

*"Anyone who is not shocked by quantum theory has not understood it."* — Niels Bohr

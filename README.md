# Classical vs Quantum Simulations

This project implements end-to-end simulations for two small physical
systems so that we can compare how classical and quantum modelling
techniques perform in terms of numerical accuracy and runtime speed.
The systems are:

1. **Penning-like three-electron trap** – three electrons confined by a
   weak harmonic potential and mutual Coulomb repulsion.
2. **Electron–proton–neutron triple** – a hydrogenic electron–proton pair
   plus a spectator neutron that couples to the proton via a
   phenomenological Yukawa interaction.

For each system we run:

- A **classical** velocity-Verlet integration in atomic units.
- A **quantum** solver (self-consistent Hartree for the trap, radial
  Schrödinger equations for the e–p–n triple).
- A **reference** high-resolution quantum solve that acts as our ground
  truth so that we can score the other two simulations.

The driver script (`python3 -m src.comparison`) prints a table that
includes absolute energies, absolute errors (vs. the reference), and
wall-clock runtimes. The final lines declare the winner for accuracy and
speed in each system.

## Repository layout

```
README.md
requirements.txt
src/
  classical.py        # Deterministic velocity-Verlet engine
  comparison.py       # CLI entry point and reporting utilities
  constants.py        # Atomic-unit constants + unit conversions
  quantum.py          # Hartree and radial Schrödinger solvers
  systems.py          # System specifications for both targets
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.comparison
```

> The project keeps everything in atomic units (Hartree energy, Bohr
> radius, electron mass, elementary charge, and reduced Planck's
> constant set to 1). Results are converted to electron-volts for the
> final report.

## Modelling notes

*Classical*: Each particle is advanced with a velocity-Verlet integrator.
Forces include softened Coulomb terms, optional harmonic confinement, and
(optional) Yukawa attractions meant to mimic the short-range nuclear
force. Energies are averaged over the second half of the trajectory so
that transient start-up effects are discarded.

*Quantum*: The three-electron trap uses a self-consistent Hartree method
on a 1D grid. Occupation numbers (2, 1) enforce the Pauli principle. The
combined electron–proton–neutron system solves two independent radial
Schrödinger problems: hydrogenic Coulomb binding with the proper reduced
mass, and a proton–neutron Yukawa well for the nuclear sub-problem. The
sum of those bound-state energies is compared against the coarse
simulators.

The high-resolution quantum runs use denser grids and tighter convergence
settings and therefore act as the "true" values during comparison. This
keeps the decision about accuracy entirely data-driven.

## Interpreting the results

- **Accuracy**: The smaller absolute error (relative to the high
  resolution reference) wins. Quantum methods are expected to dominate
  because the targets are inherently quantum mechanical.
- **Speed**: Wall-clock time measured with `perf_counter`. Classical
  integrators usually win here because they avoid global linear-algebra
  solves.
- **Energy scale**: Note that the electron–proton–neutron energies are
  dominated by the nuclear binding term, which sits in the MeV regime.

Both simulation stacks are built so that additional systems or solver
parameters can be plugged in with minimal changes: add another entry to
`systems.py` and re-run `python -m src.comparison`.

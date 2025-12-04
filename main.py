#!/usr/bin/env python3
"""
Classical vs Quantum Simulation: Complete Analysis

This program simulates two physical systems using both classical and quantum mechanics:

SYSTEM 1: THREE ELECTRONS
=========================
A system of three electrons confined by a harmonic potential (quantum dot model).

Classical Model:
- Point particles with Coulomb repulsion
- Newton's equations of motion (F = ma)
- Velocity Verlet integration
- Harmonic confinement prevents dispersion

Quantum Model:
- Wave function with Slater determinant (antisymmetry)
- Jastrow correlation factor
- Variational Monte Carlo sampling
- Ground state energy calculation

SYSTEM 2: ELECTRON + PROTON + NEUTRON
=====================================
A hydrogen-like system with additional neutron (simplified deuterium model).

Classical Model:
- Electron orbits proton (Kepler-like motion)
- Proton-neutron bound by Yukawa potential (nuclear force)
- Full 3D dynamics

Quantum Model:
- Electron: Exact hydrogen solution (Schrödinger equation)
- Nuclear: Finite-difference solution of radial equation
- Born-Oppenheimer approximation

COMPARISON CRITERIA:
===================
1. Speed: Computation time for equivalent accuracy
2. Accuracy: Deviation from known exact results
3. Physical validity: Correct prediction of observable quantities

Author: Physics Simulation Framework
"""

import numpy as np
import sys
import time
import argparse
from typing import Dict, Any

# Import simulation modules
from constants import AtomicUnits, HARTREE_TO_EV, BOHR_TO_ANGSTROM
from classical_simulation import ClassicalThreeElectronSystem, ClassicalEPNSystem
from quantum_simulation import QuantumThreeElectronSystem, QuantumEPNSystem
from comparison import run_full_comparison, compare_three_electron_system, compare_epn_system


def print_header():
    """Print program header."""
    header = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ██████╗██╗      █████╗ ███████╗███████╗██╗ ██████╗ █████╗ ██╗          ║
║  ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██╔══██╗██║          ║
║  ██║     ██║     ███████║███████╗███████╗██║██║     ███████║██║          ║
║  ██║     ██║     ██╔══██║╚════██║╚════██║██║██║     ██╔══██║██║          ║
║  ╚██████╗███████╗██║  ██║███████║███████║██║╚██████╗██║  ██║███████╗     ║
║   ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝     ║
║                           VS                                              ║
║     ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗     ║
║    ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║     ║
║    ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║     ║
║    ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║     ║
║    ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║     ║
║     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝     ║
║                                                                          ║
║                    PARTICLE PHYSICS SIMULATION                           ║
║                                                                          ║
║     Comparing Classical and Quantum Approaches for Atomic Systems        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(header)


def run_individual_simulations():
    """Run each simulation individually with detailed output."""
    results = {}
    
    print("\n" + "=" * 74)
    print("  PART 1: THREE-ELECTRON SYSTEM SIMULATIONS")
    print("=" * 74)
    
    # ========== Classical 3-electron ==========
    print("\n" + "-" * 74)
    print("  1.1 Classical Three-Electron System")
    print("-" * 74)
    
    print("\n  Physical Model:")
    print("  • Three electrons as point particles")
    print("  • Coulomb repulsion: V = 1/r₁₂ + 1/r₁₃ + 1/r₂₃")
    print("  • Harmonic confinement: V = ½mω²r²")
    print("  • Integration: Velocity Verlet (symplectic)")
    
    omega = 0.25
    classical_3e = ClassicalThreeElectronSystem(omega=omega)
    
    print(f"\n  Parameters:")
    print(f"  • Confinement ω = {omega} (a.u.)")
    print(f"  • Initial config: Equilateral triangle")
    print(f"  • Total time: 20 a.u.")
    print(f"  • Time step: 0.01 a.u.")
    
    print("\n  Running simulation...", end=" ", flush=True)
    results_c3e = classical_3e.run_simulation(t_total=20.0, dt=0.01)
    props_c3e = classical_3e.get_average_properties()
    print("Done!")
    
    print(f"\n  Results:")
    print(f"  • Computation time: {results_c3e['computation_time']:.4f} seconds")
    print(f"  • Number of steps: {results_c3e['n_steps']}")
    print(f"  • Mean total energy: {props_c3e['mean_energy']:.6f} Hartree")
    print(f"  • Energy fluctuation (σ): {props_c3e['energy_std']:.6f} Hartree")
    print(f"  • Energy conservation: {props_c3e['energy_conservation']:.2e} (relative drift)")
    print(f"  • Mean inter-electron distance: {props_c3e['mean_inter_electron_distance']:.4f} Bohr")
    
    results['classical_3e'] = {'results': results_c3e, 'props': props_c3e}
    
    # ========== Quantum 3-electron ==========
    print("\n" + "-" * 74)
    print("  1.2 Quantum Three-Electron System (Variational Monte Carlo)")
    print("-" * 74)
    
    print("\n  Physical Model:")
    print("  • Trial wave function: Ψ = Φ_Slater × J_Jastrow")
    print("  • Slater determinant: Antisymmetric (Pauli principle)")
    print("  • Jastrow factor: exp(Σ rᵢⱼ/(1+βrᵢⱼ)) (correlation)")
    print("  • Energy: E = ⟨Ψ|Ĥ|Ψ⟩/⟨Ψ|Ψ⟩ (variational)")
    
    quantum_3e = QuantumThreeElectronSystem(omega=omega)
    quantum_3e.n_walkers = 500
    quantum_3e.n_steps = 2000
    quantum_3e.n_equilibration = 500
    
    print(f"\n  Parameters:")
    print(f"  • Confinement ω = {omega} (a.u.)")
    print(f"  • VMC walkers: {quantum_3e.n_walkers}")
    print(f"  • Production steps: {quantum_3e.n_steps}")
    print(f"  • Equilibration steps: {quantum_3e.n_equilibration}")
    
    print("\n  Running VMC simulation...", end=" ", flush=True)
    results_q3e = quantum_3e.run_simulation(optimize=False)
    print("Done!")
    
    print(f"\n  Results:")
    print(f"  • Computation time: {results_q3e['computation_time']:.4f} seconds")
    print(f"  • Ground state energy: {results_q3e['ground_state_energy']:.6f} ± "
          f"{results_q3e['energy_error']:.6f} Hartree")
    print(f"  • Non-interacting reference: {results_q3e['non_interacting_energy']:.6f} Hartree")
    print(f"  • Correlation energy: {results_q3e['correlation_energy']:.6f} Hartree")
    print(f"  • Variational parameters: α = {results_q3e['variational_params']['alpha']:.4f}, "
          f"β = {results_q3e['variational_params']['beta']:.4f}")
    
    results['quantum_3e'] = results_q3e
    
    print("\n" + "=" * 74)
    print("  PART 2: ELECTRON-PROTON-NEUTRON SYSTEM SIMULATIONS")
    print("=" * 74)
    
    # ========== Classical E-P-N ==========
    print("\n" + "-" * 74)
    print("  2.1 Classical Electron-Proton-Neutron System")
    print("-" * 74)
    
    print("\n  Physical Model:")
    print("  • Electron (m=mₑ, q=-e): Orbits proton")
    print("  • Proton (m=1836mₑ, q=+e): Coulomb attraction to electron")
    print("  • Neutron (m=1839mₑ, q=0): Bound to proton by Yukawa potential")
    print("  • Yukawa nuclear force: V = -V₀ exp(-r/a)/(r/a)")
    
    classical_epn = ClassicalEPNSystem()
    
    print(f"\n  Parameters:")
    print(f"  • Nuclear potential depth: {classical_epn.V0_nuclear} Hartree")
    print(f"  • Nuclear range: {classical_epn.a_nuclear} Bohr")
    print(f"  • Initial electron radius: 2.0 Bohr")
    
    print("\n  Running simulation...", end=" ", flush=True)
    results_cepn = classical_epn.run_simulation(t_total=30.0, dt=0.0005)
    props_cepn = classical_epn.get_average_properties()
    print("Done!")
    
    print(f"\n  Results:")
    print(f"  • Computation time: {results_cepn['computation_time']:.4f} seconds")
    print(f"  • Number of steps: {results_cepn['n_steps']}")
    print(f"  • Mean total energy: {props_cepn['mean_energy']:.4f} Hartree")
    print(f"  • Energy conservation: {props_cepn['energy_conservation']:.2e}")
    print(f"  • Mean e-p distance: {props_cepn['mean_electron_proton_distance']:.4f} Bohr")
    print(f"  • Mean p-n distance: {props_cepn['mean_proton_neutron_distance']:.6f} Bohr")
    
    results['classical_epn'] = {'results': results_cepn, 'props': props_cepn}
    
    # ========== Quantum E-P-N ==========
    print("\n" + "-" * 74)
    print("  2.2 Quantum Electron-Proton-Neutron System")
    print("-" * 74)
    
    print("\n  Physical Model:")
    print("  • Born-Oppenheimer approximation (separate nuclear/electronic)")
    print("  • Electron: Exact hydrogen solution")
    print("    - Ψ₁ₛ = (1/√π) exp(-r)")
    print("    - E = -0.5 Hartree = -13.6 eV (exact)")
    print("  • Nuclear: Finite-difference Schrödinger equation")
    
    quantum_epn = QuantumEPNSystem()
    
    print("\n  Running simulation...", end=" ", flush=True)
    results_qepn = quantum_epn.run_simulation()
    print("Done!")
    
    print(f"\n  Results:")
    print(f"  • Computation time: {results_qepn['computation_time']:.4f} seconds")
    print(f"\n  Electron (Hydrogen-like):")
    print(f"  • Ground state energy: {results_qepn['electron_energy']:.10f} Hartree")
    print(f"  • Energy in eV: {results_qepn['electron_energy_eV']:.6f} eV")
    print(f"  • EXACT value: -0.5 Hartree = -13.6057 eV")
    print(f"  • Error: {abs(results_qepn['electron_energy'] + 0.5):.2e} Hartree")
    print(f"  • Average radius ⟨r⟩: {results_qepn['electron_avg_radius']:.4f} Bohr")
    print(f"  • EXACT ⟨r⟩: 1.5 Bohr")
    
    print(f"\n  Nuclear (Deuteron-like):")
    print(f"  • Binding energy: {results_qepn['nuclear_binding_energy']:.4f} Hartree")
    
    print(f"\n  Uncertainty Principle Verification:")
    print(f"  • ΔxΔp = {results_qepn['heisenberg_product']:.4f}")
    print(f"  • Required: ΔxΔp ≥ ℏ/2 = 0.5")
    print(f"  • Satisfied: {'✓ YES' if results_qepn['heisenberg_product'] >= 0.5 else '✗ NO'}")
    
    results['quantum_epn'] = results_qepn
    
    return results


def print_mathematical_details():
    """Print the mathematical formulations used."""
    print("\n" + "=" * 74)
    print("  MATHEMATICAL FORMULATIONS")
    print("=" * 74)
    
    math_text = """
  ┌────────────────────────────────────────────────────────────────────────┐
  │                     CLASSICAL MECHANICS                                 │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Newton's Second Law:                                                  │
  │      F = ma = m(d²r/dt²)                                               │
  │                                                                        │
  │  Hamiltonian:                                                          │
  │      H = T + V = Σᵢ pᵢ²/(2mᵢ) + V(r₁, r₂, ...)                        │
  │                                                                        │
  │  Coulomb Potential:                                                    │
  │      V_Coulomb = kₑ × q₁q₂/r = q₁q₂/(4πε₀r)                           │
  │      In atomic units: V = q₁q₂/r                                       │
  │                                                                        │
  │  Yukawa (Nuclear) Potential:                                           │
  │      V_Yukawa = -V₀ × exp(-r/a)/(r/a)                                  │
  │                                                                        │
  │  Velocity Verlet Integration:                                          │
  │      v(t+dt/2) = v(t) + a(t)×dt/2                                     │
  │      r(t+dt) = r(t) + v(t+dt/2)×dt                                    │
  │      a(t+dt) = F(r(t+dt))/m                                            │
  │      v(t+dt) = v(t+dt/2) + a(t+dt)×dt/2                               │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────────────┐
  │                     QUANTUM MECHANICS                                   │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Schrödinger Equation:                                                 │
  │      ĤΨ = EΨ                                                           │
  │      iℏ ∂Ψ/∂t = ĤΨ (time-dependent)                                   │
  │                                                                        │
  │  Hamiltonian Operator:                                                 │
  │      Ĥ = T̂ + V̂ = -ℏ²/(2m)∇² + V(r)                                   │
  │      In atomic units: Ĥ = -½∇² + V(r)                                 │
  │                                                                        │
  │  Hydrogen Ground State (exact):                                        │
  │      Ψ₁ₛ(r) = (1/√π) × exp(-r)                                        │
  │      E₁ₛ = -½ Hartree = -13.6 eV                                      │
  │      ⟨r⟩ = 3/2 Bohr = 0.79 Å                                          │
  │                                                                        │
  │  Variational Principle:                                                │
  │      E₀ ≤ ⟨Ψ_trial|Ĥ|Ψ_trial⟩/⟨Ψ_trial|Ψ_trial⟩                      │
  │      (Any trial gives upper bound to ground state)                     │
  │                                                                        │
  │  Trial Wave Function (3 electrons):                                    │
  │      Ψ = det|φᵢ(rⱼ)| × exp(Σᵢ<ⱼ rᵢⱼ/(1+βrᵢⱼ))                        │
  │           ↑ Slater det     ↑ Jastrow factor                           │
  │         (antisymmetry)   (electron correlation)                       │
  │                                                                        │
  │  Heisenberg Uncertainty Principle:                                     │
  │      ΔxΔp ≥ ℏ/2 = 0.5 (atomic units)                                  │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────────────┐
  │                     ATOMIC UNITS                                        │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Base Units (set to 1):                                                │
  │      ℏ = mₑ = e = 4πε₀ = 1                                            │
  │                                                                        │
  │  Derived Units:                                                        │
  │      Length: a₀ = 0.529 Å (Bohr radius)                               │
  │      Energy: Eₕ = 27.21 eV (Hartree)                                  │
  │      Time: ℏ/Eₕ = 24.2 attoseconds                                    │
  │                                                                        │
  │  Mass Ratios:                                                          │
  │      mₚ/mₑ ≈ 1836                                                     │
  │      mₙ/mₑ ≈ 1839                                                     │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
"""
    print(math_text)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Classical vs Quantum Simulation Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full          Run full comparison with all details
  python main.py --quick         Quick comparison (reduced sampling)
  python main.py --math          Show mathematical formulations
  python main.py --individual    Run each simulation separately
        """
    )
    
    parser.add_argument('--full', action='store_true', 
                       help='Run full comparison analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Quick comparison (faster, less accurate)')
    parser.add_argument('--math', action='store_true',
                       help='Display mathematical formulations')
    parser.add_argument('--individual', action='store_true',
                       help='Run individual simulations with detailed output')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Default: run full comparison if no arguments
    if not any([args.full, args.quick, args.math, args.individual]):
        args.full = True
    
    print_header()
    
    if args.math:
        print_mathematical_details()
    
    if args.individual:
        results = run_individual_simulations()
    
    if args.full or args.quick:
        # Run comparison
        metrics_3e, metrics_epn = run_full_comparison()
        
        if args.visualize:
            try:
                from comparison import create_visualization
                create_visualization(metrics_3e, metrics_epn)
            except Exception as e:
                print(f"\nVisualization skipped: {e}")
    
    print("\n" + "=" * 74)
    print("  SIMULATION COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()

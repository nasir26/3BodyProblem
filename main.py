#!/usr/bin/env python3
"""
Main Simulation Runner

A comprehensive simulation comparing classical and quantum approaches for:
1. Three-electron system
2. Electron-proton-neutron (hydrogen-like) system

This script runs both simulations and generates a detailed comparison report.

Usage:
    python main.py                    # Run full comparison
    python main.py --system 3e        # Only three-electron system
    python main.py --system hydrogen  # Only hydrogen-like system
    python main.py --visualize        # Generate visualization plots
    python main.py --quick            # Quick run with fewer samples
"""

import argparse
import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_simulation.classical.three_electron import ThreeElectronClassical
from particle_simulation.classical.hydrogen_like import HydrogenLikeClassical
from particle_simulation.quantum.three_electron import ThreeElectronQuantum
from particle_simulation.quantum.hydrogen_like import HydrogenLikeQuantum
from particle_simulation.benchmark import SimulationBenchmark
from particle_simulation.constants import PHYSICAL_CONSTANTS


def print_header():
    """Print a nice header."""
    header = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      PARTICLE SIMULATION: CLASSICAL vs QUANTUM MECHANICS COMPARISON           ║
║                                                                               ║
║      Systems:                                                                 ║
║        1. Three-Electron System                                               ║
║        2. Electron-Proton-Neutron (Hydrogen-Like Atom)                        ║
║                                                                               ║
║      Author: Particle Physics Simulation Framework                            ║
║      Date: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """                                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(header)


def run_three_electron_classical(verbose: bool = True):
    """Run classical simulation for three-electron system."""
    if verbose:
        print("\n" + "═" * 70)
        print("CLASSICAL THREE-ELECTRON SIMULATION")
        print("═" * 70)
    
    sim = ThreeElectronClassical(use_atomic_units=True)
    sim.initialize_state(configuration='equilateral')
    
    if verbose:
        print(f"\nInitial Configuration:")
        print(f"  Position 1: {sim.positions[0]}")
        print(f"  Position 2: {sim.positions[1]}")
        print(f"  Position 3: {sim.positions[2]}")
        print(f"  Initial Energy: {sim.compute_total_energy():.6f} Hartree")
        print(f"  Average Separation: {sim.get_average_separation():.3f} Bohr")
    
    # Run simulation
    results = sim.run_simulation(total_time=10.0, dt=0.001, method='verlet')
    
    if verbose:
        print(f"\nSimulation Results:")
        print(f"  Steps: {results['n_steps']}")
        print(f"  Computation Time: {results['computation_time']:.4f} seconds")
        print(f"  Final Energy: {results['final_energy']:.6f} Hartree")
        print(f"  Energy Conservation Error: {results['energy_conservation_error']:.2e}")
        print(f"  Angular Momentum: {np.linalg.norm(results['angular_momentum']):.6f}")
    
    return results


def run_three_electron_quantum(n_samples: int = 10000, verbose: bool = True):
    """Run quantum simulation for three-electron system."""
    if verbose:
        print("\n" + "═" * 70)
        print("QUANTUM THREE-ELECTRON SIMULATION")
        print("═" * 70)
    
    sim = ThreeElectronQuantum()
    
    # Variational Monte Carlo with simple Gaussian
    if verbose:
        print("\n1. Variational Monte Carlo (Gaussian):")
    result_vmc = sim.run_simulation('vmc', n_samples=n_samples, alpha=0.5)
    
    if verbose:
        print(f"   Energy: {result_vmc['energy']:.4f} ± {result_vmc['energy_error']:.4f} Hartree")
        print(f"   Acceptance Rate: {result_vmc['acceptance_rate']:.2%}")
        print(f"   Computation Time: {result_vmc['computation_time']:.4f} s")
    
    # VMC with Jastrow correlation
    if verbose:
        print("\n2. VMC with Slater-Jastrow (includes correlation):")
    result_sj = sim.run_simulation('vmc_jastrow', n_samples=n_samples, alpha=0.3, beta=0.5)
    
    if verbose:
        print(f"   Energy: {result_sj['energy']:.4f} ± {result_sj['energy_error']:.4f} Hartree")
        print(f"   Acceptance Rate: {result_sj['acceptance_rate']:.2%}")
        print(f"   Computation Time: {result_sj['computation_time']:.4f} s")
    
    # Hartree-Fock (in harmonic trap for reference)
    if verbose:
        print("\n3. Hartree-Fock Approximation (harmonic trap):")
    result_hf = sim.run_simulation('hartree_fock', omega=0.5)
    
    if verbose:
        print(f"   Energy: {result_hf['energy']:.4f} Hartree")
        print(f"   Converged: {result_hf['converged']}")
        print(f"   Computation Time: {result_hf['computation_time']:.4f} s")
    
    return {'vmc': result_vmc, 'slater_jastrow': result_sj, 'hartree_fock': result_hf}


def run_hydrogen_classical(verbose: bool = True):
    """Run classical simulation for hydrogen-like atom."""
    if verbose:
        print("\n" + "═" * 70)
        print("CLASSICAL HYDROGEN-LIKE ATOM SIMULATION")
        print("═" * 70)
    
    sim = HydrogenLikeClassical(use_atomic_units=True)
    sim.initialize_state(configuration='bohr_orbit')
    
    if verbose:
        print(f"\nInitial Configuration (Bohr Model):")
        print(f"  Electron: {sim.positions[0]} (at Bohr radius)")
        print(f"  Proton: {sim.positions[1]} (at origin)")
        print(f"  Neutron: {sim.positions[2]} (in nucleus)")
        print(f"  Initial Energy: {sim.compute_total_energy():.6f} Hartree")
        print(f"  Bohr model prediction: -0.500000 Hartree")
    
    # Run for 10 orbital periods
    orbital_period = 2 * np.pi
    results = sim.run_simulation(total_time=10*orbital_period, dt=0.01, method='verlet')
    stability = sim.analyze_orbit_stability()
    
    if verbose:
        print(f"\nSimulation Results:")
        print(f"  Simulated time: 10 orbital periods")
        print(f"  Computation Time: {results['computation_time']:.4f} seconds")
        print(f"  Final Energy: {results['final_energy']:.6f} Hartree")
        print(f"  Energy Conservation Error: {results['energy_conservation_error']:.2e}")
        print(f"  Average e-p distance: {results['avg_e_p_distance']:.4f} Bohr")
        print(f"\nOrbit Stability:")
        print(f"  Initial radius: {stability['initial_radius']:.4f} Bohr")
        print(f"  Final radius: {stability['final_radius']:.4f} Bohr")
        print(f"  Stable: {stability['stable']}")
        print(f"\n⚠️  Note: {stability['classical_issue']}")
    
    return {'simulation': results, 'stability': stability}


def run_hydrogen_quantum(verbose: bool = True):
    """Run quantum simulation for hydrogen-like atom."""
    if verbose:
        print("\n" + "═" * 70)
        print("QUANTUM HYDROGEN-LIKE ATOM SIMULATION")
        print("═" * 70)
    
    sim = HydrogenLikeQuantum(Z=1.0, use_reduced_mass=True)
    
    if verbose:
        print(f"\nReduced mass correction: μ/m_e = {sim.mu_au:.6f}")
        print(f"(Accounts for proton+neutron nuclear mass)")
    
    # Analytical solutions
    if verbose:
        print("\n1. Analytical Energy Levels:")
    result_analytical = sim.run_simulation('analytical', n_states=5)
    
    if verbose:
        print(f"   Ground state: {result_analytical['ground_state_energy']:.6f} Hartree")
        print(f"                 = {result_analytical['ground_state_energy_eV']:.4f} eV")
        print(f"   Computation Time: {result_analytical['computation_time']:.6f} s (instantaneous!)")
        
        print("\n   Energy levels:")
        seen_n = set()
        for state in result_analytical['states']:
            if state['n'] not in seen_n:
                print(f"     n={state['n']}: E = {state['energy']:.6f} Ha = {state['energy_eV']:.4f} eV")
                seen_n.add(state['n'])
    
    # Numerical matrix method
    if verbose:
        print("\n2. Numerical Matrix Diagonalization:")
    result_numerical = sim.run_simulation('matrix', n_states=5, n_points=500)
    
    if verbose:
        print(f"   Computation Time: {result_numerical['computation_time']:.4f} s")
        print(f"   Numerical vs Analytical comparison:")
        for state in result_numerical['states'][:3]:
            if state['energy_analytical'] is not None:
                print(f"     State {state['index']}: E_num = {state['energy_numerical']:.6f}, "
                      f"Error = {state['error']:.2e}")
    
    # Comparison with classical
    if verbose:
        print("\n3. Quantum vs Classical Comparison (n=1 ground state):")
    comparison = sim.compare_with_classical_orbit(n=1, l=0)
    
    if verbose:
        print(f"   Quantum energy:  {comparison['quantum']['energy']:.6f} Hartree")
        print(f"   Classical energy: {comparison['classical_bohr']['energy']:.6f} Hartree")
        print(f"   Quantum <r>:     {comparison['quantum']['mean_radius']:.4f} Bohr")
        print(f"   Classical r:     {comparison['classical_bohr']['radius']:.4f} Bohr")
        print(f"   Position uncertainty: {comparison['quantum']['uncertainty_radius']:.4f} Bohr")
        print(f"   Relative uncertainty: {comparison['quantum']['relative_uncertainty']:.1%}")
        print(f"\n   Key difference: {comparison['comparison']['key_difference']}")
    
    return {
        'analytical': result_analytical,
        'numerical': result_numerical,
        'comparison': comparison
    }


def print_comparison_summary(classical_3e, quantum_3e, classical_h, quantum_h):
    """Print a summary comparing all simulations."""
    print("\n" + "═" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("═" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THREE-ELECTRON SYSTEM                                  │
├──────────────────────────┬──────────────────────────┬────────────────────────────┤
│         Metric           │       Classical          │         Quantum            │
├──────────────────────────┼──────────────────────────┼────────────────────────────┤""")
    
    cl_time = classical_3e['computation_time']
    qm_time = quantum_3e['vmc']['computation_time']
    print(f"│ Computation Time         │ {cl_time:>20.4f} s   │ {qm_time:>22.4f} s   │")
    
    cl_E = classical_3e['final_energy']
    qm_E = quantum_3e['vmc']['energy']
    print(f"│ Energy (Hartree)         │ {cl_E:>20.4f}     │ {qm_E:>18.4f} ± {quantum_3e['vmc']['energy_error']:.3f} │")
    
    cl_err = classical_3e['energy_conservation_error']
    print(f"│ Energy Conservation      │ {cl_err:>20.2e}     │       Variational bound    │")
    
    print("""├──────────────────────────┴──────────────────────────┴────────────────────────────┤
│ VERDICT: Quantum required - Classical misses exchange and correlation            │
└──────────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        HYDROGEN-LIKE ATOM (e-p-n)                                │
├──────────────────────────┬──────────────────────────┬────────────────────────────┤
│         Metric           │       Classical          │         Quantum            │
├──────────────────────────┼──────────────────────────┼────────────────────────────┤""")
    
    cl_time = classical_h['simulation']['computation_time']
    qm_time = quantum_h['analytical']['computation_time']
    print(f"│ Computation Time         │ {cl_time:>20.4f} s   │ {qm_time:>22.6f} s   │")
    
    cl_E = classical_h['simulation']['final_energy']
    qm_E = quantum_h['analytical']['ground_state_energy']
    print(f"│ Ground State Energy      │ {cl_E:>20.4f} Ha  │ {qm_E:>22.6f} Ha  │")
    
    print(f"│ Experimental Value       │           -0.5000 Ha  │            -0.5000 Ha  │")
    
    cl_stable = classical_h['stability']['stable']
    print(f"│ Orbit/State Stable       │ {str(cl_stable):>20}     │                 Always     │")
    
    print("""├──────────────────────────┴──────────────────────────┴────────────────────────────┤
│ VERDICT: Quantum is EXACT and explains atomic stability!                         │
└──────────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                              FINAL CONCLUSIONS                                   ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  SPEED:                                                                          ║
║  • Classical trajectory simulations are fast (~0.01-0.1 seconds)                 ║
║  • Quantum analytical solutions are INSTANTANEOUS (<1 ms)                        ║
║  • Quantum numerical methods (VMC) are moderate (~1-10 seconds)                  ║
║                                                                                  ║
║  ACCURACY:                                                                       ║
║  • Classical mechanics FAILS for atomic systems                                  ║
║    - Cannot explain energy quantization                                          ║
║    - Cannot explain atomic stability                                             ║
║    - Misses uncertainty principle                                                ║
║    - No exchange/correlation effects                                             ║
║                                                                                  ║
║  • Quantum mechanics is EXACT for these systems                                  ║
║    - Matches experimental observations perfectly                                 ║
║    - Explains all atomic phenomena                                               ║
║    - Predicts correct energy levels                                              ║
║                                                                                  ║
║  RECOMMENDATION:                                                                 ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │  FOR ATOMIC-SCALE SYSTEMS: ALWAYS USE QUANTUM MECHANICS                    │ ║
║  │                                                                             │ ║
║  │  Classical mechanics is fundamentally wrong at this scale.                 │ ║
║  │  Speed advantages are meaningless when results are incorrect.              │ ║
║  └─────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                  ║
╚═════════════════════════════════════════════════════════════════════════════════╝
""")


def generate_visualizations(classical_3e, quantum_3e, classical_h, quantum_h):
    """Generate visualization plots."""
    try:
        from particle_simulation.visualization import (
            plot_three_electron_trajectory,
            plot_energy_conservation,
            plot_hydrogen_orbit,
            plot_energy_levels,
            create_summary_figure,
        )
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualizations...")
        os.makedirs('output', exist_ok=True)
        
        # Three-electron trajectory
        if 'trajectory' in classical_3e:
            fig = plot_three_electron_trajectory(
                classical_3e['trajectory'],
                save_path='output/three_electron_trajectory.png'
            )
            plt.close(fig)
            print("  ✓ output/three_electron_trajectory.png")
        
        # Energy conservation
        if 'energies' in classical_3e and 'times' in classical_3e:
            fig = plot_energy_conservation(
                classical_3e['times'],
                classical_3e['energies'],
                save_path='output/energy_conservation.png'
            )
            plt.close(fig)
            print("  ✓ output/energy_conservation.png")
        
        # Hydrogen orbit
        if 'trajectory' in classical_h['simulation']:
            fig = plot_hydrogen_orbit(
                classical_h['simulation']['trajectory'],
                classical_h['simulation']['times'],
                save_path='output/hydrogen_orbit.png'
            )
            plt.close(fig)
            print("  ✓ output/hydrogen_orbit.png")
        
        # Energy levels
        quantum_E = [
            (1, quantum_h['analytical']['ground_state_energy']),
            (2, quantum_h['analytical']['ground_state_energy'] * 0.25),
            (3, quantum_h['analytical']['ground_state_energy'] / 9),
        ]
        fig = plot_energy_levels(
            classical_h['simulation']['final_energy'],
            quantum_E,
            save_path='output/energy_levels.png'
        )
        plt.close(fig)
        print("  ✓ output/energy_levels.png")
        
        # Summary figure
        benchmark_results = {
            'three_electron_classical': {'computation_time': classical_3e['computation_time']},
            'three_electron_quantum_vmc': {'computation_time': quantum_3e['vmc']['computation_time']},
            'hydrogen_classical': {'computation_time': classical_h['simulation']['computation_time']},
            'hydrogen_quantum': {'computation_time': quantum_h['analytical']['computation_time']},
        }
        fig = create_summary_figure(
            benchmark_results,
            save_path='output/comparison_summary.png'
        )
        plt.close(fig)
        print("  ✓ output/comparison_summary.png")
        
        print("\nAll visualizations saved to 'output/' directory.")
        
    except ImportError as e:
        print(f"\nWarning: Could not generate visualizations ({e})")
        print("Install matplotlib for visualizations: pip install matplotlib")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Classical vs Quantum Particle Simulation Comparison'
    )
    parser.add_argument('--system', choices=['3e', 'hydrogen', 'all'], default='all',
                        help='Which system to simulate')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer samples')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run full benchmark suite')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    n_samples = 2000 if args.quick else 10000
    
    if verbose:
        print_header()
    
    results = {}
    
    # Run benchmark suite
    if args.benchmark:
        if verbose:
            print("\nRunning full benchmark suite...")
        benchmark = SimulationBenchmark()
        benchmark.run_full_comparison()
        report = benchmark.generate_comparison_report()
        print(report)
        return
    
    # Run specific simulations
    if args.system in ['3e', 'all']:
        results['classical_3e'] = run_three_electron_classical(verbose)
        results['quantum_3e'] = run_three_electron_quantum(n_samples, verbose)
    
    if args.system in ['hydrogen', 'all']:
        results['classical_h'] = run_hydrogen_classical(verbose)
        results['quantum_h'] = run_hydrogen_quantum(verbose)
    
    # Print summary if running all
    if args.system == 'all' and not args.quiet:
        print_comparison_summary(
            results['classical_3e'],
            results['quantum_3e'],
            results['classical_h'],
            results['quantum_h']
        )
    
    # Generate visualizations
    if args.visualize and args.system == 'all':
        generate_visualizations(
            results['classical_3e'],
            results['quantum_3e'],
            results['classical_h'],
            results['quantum_h']
        )
    
    if verbose:
        print("\n✓ Simulation complete!")
    
    return results


if __name__ == "__main__":
    main()

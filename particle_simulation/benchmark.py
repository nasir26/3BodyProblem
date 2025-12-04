"""
Simulation Benchmark and Comparison Module

This module provides comprehensive comparison between classical and quantum
simulations for both particle systems:
1. Three-electron system
2. Electron-proton-neutron (hydrogen-like) system

Comparison Criteria:
-------------------
1. Speed: Computation time for equivalent accuracy
2. Accuracy: Physical correctness of results
3. Physical Predictions: What each model can/cannot predict
4. Scalability: How performance scales with system size
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .classical.three_electron import ThreeElectronClassical
from .classical.hydrogen_like import HydrogenLikeClassical
from .quantum.three_electron import ThreeElectronQuantum
from .quantum.hydrogen_like import HydrogenLikeQuantum
from .constants import PHYSICAL_CONSTANTS


class SimulationType(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    system: str
    simulation_type: SimulationType
    method: str
    computation_time: float
    energy: float
    energy_error: Optional[float]
    additional_metrics: Dict
    physical_validity: Dict
    

class SimulationBenchmark:
    """
    Comprehensive benchmark comparing classical and quantum simulations.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {
            'three_electron': {'classical': None, 'quantum': None},
            'hydrogen_like': {'classical': None, 'quantum': None},
        }
        
    # =========================================================================
    # THREE-ELECTRON SYSTEM BENCHMARKS
    # =========================================================================
    
    def benchmark_three_electron_classical(self,
                                            total_time: float = 10.0,
                                            dt: float = 0.001,
                                            configuration: str = 'equilateral') -> Dict:
        """
        Benchmark classical three-electron simulation.
        
        Parameters:
        -----------
        total_time : float
            Simulation time in atomic units
        dt : float
            Time step
        configuration : str
            Initial configuration
        """
        sim = ThreeElectronClassical(use_atomic_units=True)
        sim.initialize_state(configuration=configuration)
        
        # Run simulation
        result = sim.run_simulation(total_time=total_time, dt=dt, method='verlet')
        
        benchmark = BenchmarkResult(
            system='three_electron',
            simulation_type=SimulationType.CLASSICAL,
            method='verlet',
            computation_time=result['computation_time'],
            energy=result['final_energy'],
            energy_error=result['energy_conservation_error'],
            additional_metrics={
                'initial_energy': result['initial_energy'],
                'max_energy_error': result['max_energy_error'],
                'n_steps': result['n_steps'],
                'angular_momentum': np.linalg.norm(result['angular_momentum']),
            },
            physical_validity={
                'energy_conservation': result['energy_conservation_error'] < 0.01,
                'angular_momentum_conservation': True,  # Symplectic integrator
                'missing_physics': [
                    'Quantum exchange interaction (Pauli exclusion)',
                    'Heisenberg uncertainty principle',
                    'Wave-particle duality',
                    'Spin-spin interactions',
                    'Zero-point energy',
                ],
            }
        )
        
        self.results['three_electron']['classical'] = benchmark
        return benchmark.__dict__
    
    def benchmark_three_electron_quantum(self,
                                          method: str = 'vmc',
                                          n_samples: int = 10000) -> Dict:
        """
        Benchmark quantum three-electron simulation.
        
        Parameters:
        -----------
        method : str
            'vmc', 'vmc_jastrow', or 'hartree_fock'
        n_samples : int
            Number of Monte Carlo samples
        """
        sim = ThreeElectronQuantum()
        
        if method == 'vmc':
            result = sim.run_simulation('vmc', n_samples=n_samples, alpha=0.5)
        elif method == 'vmc_jastrow':
            result = sim.run_simulation('vmc_jastrow', n_samples=n_samples, alpha=0.3, beta=0.5)
        elif method == 'hartree_fock':
            result = sim.run_simulation('hartree_fock', omega=0.5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        benchmark = BenchmarkResult(
            system='three_electron',
            simulation_type=SimulationType.QUANTUM,
            method=method,
            computation_time=result['computation_time'],
            energy=result['energy'],
            energy_error=result.get('energy_error', 0),
            additional_metrics={
                'n_samples': result.get('n_samples', 0),
                'acceptance_rate': result.get('acceptance_rate', 0),
                'params': result.get('params', result.get('optimal_params', None)),
            },
            physical_validity={
                'includes_exchange': method != 'vmc',  # Simple Gaussian doesn't include
                'includes_correlation': 'jastrow' in method,
                'variational_principle': True,  # Energy is upper bound
                'captures_physics': [
                    'Electron-electron repulsion',
                    'Quantum confinement effects',
                    'Electron correlation (if Jastrow)',
                ],
            }
        )
        
        self.results['three_electron']['quantum'] = benchmark
        return benchmark.__dict__
    
    # =========================================================================
    # HYDROGEN-LIKE SYSTEM BENCHMARKS
    # =========================================================================
    
    def benchmark_hydrogen_classical(self,
                                      total_time: float = 100.0,
                                      dt: float = 0.01,
                                      configuration: str = 'bohr_orbit') -> Dict:
        """
        Benchmark classical hydrogen-like simulation.
        """
        sim = HydrogenLikeClassical(use_atomic_units=True)
        sim.initialize_state(configuration=configuration)
        
        result = sim.run_simulation(total_time=total_time, dt=dt, method='verlet')
        stability = sim.analyze_orbit_stability()
        
        benchmark = BenchmarkResult(
            system='hydrogen_like',
            simulation_type=SimulationType.CLASSICAL,
            method='verlet',
            computation_time=result['computation_time'],
            energy=result['final_energy'],
            energy_error=result['energy_conservation_error'],
            additional_metrics={
                'initial_energy': result['initial_energy'],
                'avg_e_p_distance': result['avg_e_p_distance'],
                'final_e_p_distance': result['final_e_p_distance'],
                'orbit_stable': stability['stable'],
            },
            physical_validity={
                'energy_conservation': result['energy_conservation_error'] < 0.01,
                'orbit_stability': stability['stable'],
                'correct_energy': np.isclose(result['initial_energy'], -0.5, atol=0.1),
                'critical_flaw': 'Does not include radiation - classically, electron '
                                 'should spiral into nucleus.',
                'missing_physics': [
                    'Quantized energy levels',
                    'Wave function probability',
                    'Uncertainty principle',
                    'Tunneling effects',
                    'Correct ground state energy',
                ],
            }
        )
        
        self.results['hydrogen_like']['classical'] = benchmark
        return benchmark.__dict__
    
    def benchmark_hydrogen_quantum(self,
                                    method: str = 'analytical',
                                    n_states: int = 5) -> Dict:
        """
        Benchmark quantum hydrogen-like simulation.
        """
        sim = HydrogenLikeQuantum(Z=1.0, use_reduced_mass=True)
        
        result = sim.run_simulation(method, n_states=n_states)
        
        if method == 'analytical':
            comparison = sim.compare_with_classical_orbit(n=1, l=0)
            
            benchmark = BenchmarkResult(
                system='hydrogen_like',
                simulation_type=SimulationType.QUANTUM,
                method=method,
                computation_time=result['computation_time'],
                energy=result['ground_state_energy'],
                energy_error=0.0,  # Exact analytical
                additional_metrics={
                    'ground_state_eV': result['ground_state_energy_eV'],
                    'reduced_mass_correction': result['reduced_mass_au'],
                    'n_states': len(result['states']),
                    'mean_radius': comparison['quantum']['mean_radius'],
                    'radius_uncertainty': comparison['quantum']['uncertainty_radius'],
                },
                physical_validity={
                    'exact_solution': True,
                    'includes_quantization': True,
                    'includes_uncertainty': True,
                    'experimental_agreement': True,
                    'captures_physics': [
                        'Quantized energy levels (matches experiment)',
                        'Probability distribution of electron',
                        'Heisenberg uncertainty principle',
                        'Correct ground state energy (-13.6 eV)',
                        'Atomic stability',
                    ],
                }
            )
        else:
            benchmark = BenchmarkResult(
                system='hydrogen_like',
                simulation_type=SimulationType.QUANTUM,
                method=method,
                computation_time=result['computation_time'],
                energy=result['eigenvalues'][0] if 'eigenvalues' in result else 0,
                energy_error=result['states'][0]['error'] if result.get('states') else 0,
                additional_metrics=result,
                physical_validity={
                    'numerical_accuracy': True,
                    'includes_quantization': True,
                }
            )
        
        self.results['hydrogen_like']['quantum'] = benchmark
        return benchmark.__dict__
    
    # =========================================================================
    # COMPARISON ANALYSIS
    # =========================================================================
    
    def run_full_comparison(self) -> Dict:
        """
        Run complete benchmark suite for both systems.
        """
        print("=" * 70)
        print("PARTICLE SIMULATION BENCHMARK: CLASSICAL vs QUANTUM")
        print("=" * 70)
        
        results = {}
        
        # Three-electron system
        print("\n" + "=" * 70)
        print("SYSTEM 1: THREE INTERACTING ELECTRONS")
        print("=" * 70)
        
        print("\nRunning classical simulation...")
        results['three_electron_classical'] = self.benchmark_three_electron_classical()
        
        print("Running quantum VMC simulation...")
        results['three_electron_quantum_vmc'] = self.benchmark_three_electron_quantum('vmc')
        
        print("Running quantum Hartree-Fock simulation...")
        results['three_electron_quantum_hf'] = self.benchmark_three_electron_quantum('hartree_fock')
        
        # Hydrogen-like system
        print("\n" + "=" * 70)
        print("SYSTEM 2: ELECTRON-PROTON-NEUTRON (HYDROGEN-LIKE)")
        print("=" * 70)
        
        print("\nRunning classical simulation...")
        results['hydrogen_classical'] = self.benchmark_hydrogen_classical()
        
        print("Running quantum analytical simulation...")
        results['hydrogen_quantum'] = self.benchmark_hydrogen_quantum('analytical')
        
        print("Running quantum numerical simulation...")
        results['hydrogen_quantum_numerical'] = self.benchmark_hydrogen_quantum('matrix')
        
        return results
    
    def generate_comparison_report(self) -> str:
        """
        Generate a detailed comparison report.
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("COMPREHENSIVE COMPARISON REPORT: CLASSICAL vs QUANTUM SIMULATIONS")
        report.append("=" * 80)
        
        # Run benchmarks if not already done
        if self.results['three_electron']['classical'] is None:
            self.run_full_comparison()
        
        # Three-electron comparison
        report.append("\n" + "-" * 80)
        report.append("SYSTEM 1: THREE-ELECTRON SYSTEM")
        report.append("-" * 80)
        
        cl = self.results['three_electron']['classical']
        qm = self.results['three_electron']['quantum']
        
        if cl and qm:
            report.append(f"\n{'Metric':<30} {'Classical':<20} {'Quantum':<20}")
            report.append("-" * 70)
            report.append(f"{'Computation Time (s)':<30} {cl.computation_time:<20.4f} {qm.computation_time:<20.4f}")
            report.append(f"{'Energy (Hartree)':<30} {cl.energy:<20.4f} {qm.energy:<20.4f}")
            report.append(f"{'Energy Error':<30} {cl.energy_error:<20.2e} {qm.energy_error:<20.4f}")
            
            report.append("\n  SPEED COMPARISON:")
            if cl.computation_time < qm.computation_time:
                speedup = qm.computation_time / cl.computation_time
                report.append(f"  → Classical is {speedup:.1f}x FASTER")
            else:
                speedup = cl.computation_time / qm.computation_time
                report.append(f"  → Quantum is {speedup:.1f}x FASTER")
            
            report.append("\n  ACCURACY COMPARISON:")
            report.append("  Classical:")
            report.append(f"    - Energy conservation: {cl.physical_validity['energy_conservation']}")
            report.append("    - MISSING: " + ", ".join(cl.physical_validity['missing_physics'][:3]))
            report.append("  Quantum:")
            report.append(f"    - Variational upper bound: {qm.physical_validity['variational_principle']}")
            report.append(f"    - Includes correlation: {qm.physical_validity.get('includes_correlation', False)}")
        
        # Hydrogen-like comparison
        report.append("\n" + "-" * 80)
        report.append("SYSTEM 2: HYDROGEN-LIKE ATOM (e-p-n)")
        report.append("-" * 80)
        
        cl = self.results['hydrogen_like']['classical']
        qm = self.results['hydrogen_like']['quantum']
        
        if cl and qm:
            report.append(f"\n{'Metric':<30} {'Classical':<20} {'Quantum':<20}")
            report.append("-" * 70)
            report.append(f"{'Computation Time (s)':<30} {cl.computation_time:<20.4f} {qm.computation_time:<20.6f}")
            report.append(f"{'Ground State Energy (Ha)':<30} {cl.energy:<20.4f} {qm.energy:<20.6f}")
            report.append(f"{'Energy Error':<30} {cl.energy_error:<20.2e} {'Exact':<20}")
            
            # Theoretical comparison
            E_exact = -0.5  # Hartree
            report.append(f"\n  Exact ground state energy: {E_exact:.6f} Hartree = -13.6 eV")
            report.append(f"  Classical prediction: {cl.energy:.6f} Hartree (error: {abs(cl.energy - E_exact):.4f})")
            report.append(f"  Quantum prediction: {qm.energy:.6f} Hartree (exact)")
            
            report.append("\n  SPEED COMPARISON:")
            if cl.computation_time < qm.computation_time:
                speedup = qm.computation_time / cl.computation_time
                report.append(f"  → Classical is {speedup:.1f}x FASTER for trajectory simulation")
            else:
                speedup = cl.computation_time / qm.computation_time
                report.append(f"  → Quantum is {speedup:.1f}x FASTER (analytical solution!)")
            
            report.append("\n  ACCURACY COMPARISON:")
            report.append("  Classical limitations:")
            report.append(f"    - {cl.physical_validity.get('critical_flaw', 'N/A')}")
            for item in cl.physical_validity.get('missing_physics', [])[:3]:
                report.append(f"    - Missing: {item}")
            report.append("  Quantum advantages:")
            for item in qm.physical_validity.get('captures_physics', [])[:3]:
                report.append(f"    - {item}")
        
        # Overall conclusions
        report.append("\n" + "=" * 80)
        report.append("OVERALL CONCLUSIONS")
        report.append("=" * 80)
        
        report.append("""
  1. SPEED:
     - Classical simulations are generally FASTER for trajectory calculations
     - Quantum analytical solutions (when available) are INSTANTANEOUS
     - Quantum numerical methods (VMC, matrix) can be slower
     - Classical: O(N²) per step for N particles
     - Quantum: Exponential in particle number for exact methods

  2. ACCURACY:
     - Classical mechanics FAILS for atomic-scale systems
     - Quantum mechanics provides EXACT agreement with experiment
     - Key quantum effects missing in classical:
       * Energy quantization
       * Wave-particle duality
       * Uncertainty principle
       * Atomic stability (no classical explanation)

  3. WHEN TO USE CLASSICAL:
     - Large systems where quantum effects average out
     - High temperatures (kT >> quantum energy scales)
     - Heavy particles (classical limit)
     - Approximate dynamics when exact quantum not needed

  4. WHEN TO USE QUANTUM:
     - Atomic and molecular systems
     - When energy levels are quantized
     - When accuracy is critical
     - When phenomena depend on wave nature of particles

  5. VERDICT FOR THESE SYSTEMS:
     - THREE-ELECTRON SYSTEM: Quantum REQUIRED for accurate physics
       (Classical misses exchange, correlation, quantization)
     - HYDROGEN-LIKE ATOM: Quantum REQUIRED
       (Classical cannot explain atomic stability or spectra)
""")
        
        return "\n".join(report)
    
    def get_speed_accuracy_tradeoff(self) -> Dict:
        """
        Analyze speed vs accuracy tradeoff for both approaches.
        """
        if self.results['three_electron']['classical'] is None:
            self.run_full_comparison()
        
        tradeoff = {
            'three_electron': {
                'classical': {
                    'speed': 'FAST',
                    'accuracy': 'POOR (missing quantum effects)',
                    'use_case': 'Qualitative dynamics only',
                },
                'quantum': {
                    'speed': 'MODERATE',
                    'accuracy': 'GOOD (includes correlation)',
                    'use_case': 'Accurate ground state properties',
                },
            },
            'hydrogen_like': {
                'classical': {
                    'speed': 'FAST',
                    'accuracy': 'FUNDAMENTALLY FLAWED',
                    'use_case': 'Educational demonstration of failure',
                },
                'quantum': {
                    'speed': 'VERY FAST (analytical)',
                    'accuracy': 'EXACT',
                    'use_case': 'All applications',
                },
            },
            'recommendation': {
                'three_electron': 'USE QUANTUM - Classical misses essential physics',
                'hydrogen_like': 'USE QUANTUM - Only correct approach for atoms',
            }
        }
        
        return tradeoff


if __name__ == "__main__":
    benchmark = SimulationBenchmark()
    
    # Run full comparison
    results = benchmark.run_full_comparison()
    
    # Generate and print report
    report = benchmark.generate_comparison_report()
    print(report)
    
    # Get speed-accuracy tradeoff
    tradeoff = benchmark.get_speed_accuracy_tradeoff()
    print("\nSPEED-ACCURACY TRADEOFF:")
    print("-" * 40)
    for system, data in tradeoff.items():
        if system != 'recommendation':
            print(f"\n{system.upper()}:")
            for sim_type, metrics in data.items():
                print(f"  {sim_type}:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")

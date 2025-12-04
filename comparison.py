"""
Comparison Framework
Compares classical and quantum simulations in terms of speed and accuracy.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from classical_simulation import (
    ClassicalSystem, 
    create_three_electron_system, 
    create_electron_proton_neutron_system
)
from quantum_simulation import (
    ThreeElectronQuantumSystem,
    ElectronProtonNeutronQuantumSystem
)


class SimulationComparison:
    """Compare classical vs quantum simulations."""
    
    def __init__(self):
        """Initialize comparison framework."""
        self.results = {}
    
    def compare_three_electron_system(self, t_max: float = 1.0, dt: float = 0.01) -> Dict:
        """
        Compare classical and quantum simulations for 3-electron system.
        
        Args:
            t_max: Maximum time for classical simulation
            dt: Time step for classical simulation
            
        Returns:
            Dictionary with comparison results
        """
        print("=" * 60)
        print("Comparing 3-Electron System")
        print("=" * 60)
        
        results = {
            'system': '3-electron',
            'classical': {},
            'quantum': {},
            'comparison': {}
        }
        
        # Classical simulation
        print("\nRunning Classical Simulation...")
        classical_sys = create_three_electron_system()
        classical_result = classical_sys.simulate(t_max=t_max, dt=dt)
        
        # Extract final energy
        final_energy = classical_result['energies'][-1]['total']
        initial_energy = classical_result['energies'][0]['total']
        energy_conservation_error = abs(final_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0
        
        results['classical'] = {
            'computation_time': classical_result['computation_time'],
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_conservation_error': energy_conservation_error,
            'n_steps': classical_result['n_steps'],
            'average_kinetic': np.mean([e['kinetic'] for e in classical_result['energies']]),
            'average_potential': np.mean([e['potential'] for e in classical_result['energies']])
        }
        
        print(f"  Computation time: {classical_result['computation_time']:.4f} seconds")
        print(f"  Initial energy: {initial_energy:.6f} a.u.")
        print(f"  Final energy: {final_energy:.6f} a.u.")
        print(f"  Energy conservation error: {energy_conservation_error:.2e}")
        
        # Quantum simulation - Hartree-Fock
        print("\nRunning Quantum Simulation (Hartree-Fock)...")
        quantum_sys = ThreeElectronQuantumSystem()
        quantum_result_hf = quantum_sys.hartree_fock_solve(grid_size=100, grid_range=(-5, 5))
        
        results['quantum']['hartree_fock'] = {
            'computation_time': quantum_result_hf['computation_time'],
            'ground_state_energy': quantum_result_hf['ground_state_energy'],
            'n_iterations': quantum_result_hf['n_iterations']
        }
        
        print(f"  Computation time: {quantum_result_hf['computation_time']:.4f} seconds")
        print(f"  Ground state energy: {quantum_result_hf['ground_state_energy']:.6f} a.u.")
        print(f"  Converged in {quantum_result_hf['n_iterations']} iterations")
        
        # Quantum simulation - Variational (more accurate)
        print("\nRunning Quantum Simulation (Variational)...")
        quantum_result_var = quantum_sys.exact_3body_quantum(grid_size=50)
        
        results['quantum']['variational'] = {
            'computation_time': quantum_result_var['computation_time'],
            'ground_state_energy': quantum_result_var['ground_state_energy']
        }
        
        print(f"  Computation time: {quantum_result_var['computation_time']:.4f} seconds")
        print(f"  Ground state energy: {quantum_result_var['ground_state_energy']:.6f} a.u.")
        
        # Comparison metrics
        print("\n" + "-" * 60)
        print("COMPARISON RESULTS")
        print("-" * 60)
        
        # Speed comparison
        classical_time = results['classical']['computation_time']
        quantum_hf_time = results['quantum']['hartree_fock']['computation_time']
        quantum_var_time = results['quantum']['variational']['computation_time']
        
        speed_ratio_hf = classical_time / quantum_hf_time if quantum_hf_time > 0 else float('inf')
        speed_ratio_var = classical_time / quantum_var_time if quantum_var_time > 0 else float('inf')
        
        results['comparison']['speed'] = {
            'classical_time': classical_time,
            'quantum_hf_time': quantum_hf_time,
            'quantum_var_time': quantum_var_time,
            'classical_vs_hf_ratio': speed_ratio_hf,
            'classical_vs_var_ratio': speed_ratio_var,
            'faster_method': 'classical' if classical_time < min(quantum_hf_time, quantum_var_time) else 'quantum'
        }
        
        print(f"\nSpeed Comparison:")
        print(f"  Classical: {classical_time:.4f} s")
        print(f"  Quantum (HF): {quantum_hf_time:.4f} s ({speed_ratio_hf:.2f}x {'faster' if speed_ratio_hf > 1 else 'slower'})")
        print(f"  Quantum (Var): {quantum_var_time:.4f} s ({speed_ratio_var:.2f}x {'faster' if speed_ratio_var > 1 else 'slower'})")
        
        # Accuracy comparison
        # Classical gives time-dependent energy, quantum gives ground state
        # For accuracy, we compare energy values and physical correctness
        classical_avg_energy = results['classical']['average_kinetic'] + results['classical']['average_potential']
        quantum_ground_energy = results['quantum']['variational']['ground_state_energy']
        
        # Classical should have higher energy (not in ground state)
        energy_difference = abs(classical_avg_energy - quantum_ground_energy)
        
        results['comparison']['accuracy'] = {
            'classical_avg_energy': classical_avg_energy,
            'quantum_ground_energy': quantum_ground_energy,
            'energy_difference': energy_difference,
            'classical_energy_conservation': energy_conservation_error,
            'quantum_method': 'variational'
        }
        
        print(f"\nAccuracy Comparison:")
        print(f"  Classical average energy: {classical_avg_energy:.6f} a.u.")
        print(f"  Quantum ground state: {quantum_ground_energy:.6f} a.u.")
        print(f"  Energy difference: {energy_difference:.6f} a.u.")
        print(f"  Classical energy conservation error: {energy_conservation_error:.2e}")
        
        # Physical correctness
        print(f"\nPhysical Correctness:")
        print(f"  Classical: Describes time evolution, but electrons violate Pauli exclusion")
        print(f"  Quantum: Properly accounts for electron spin, antisymmetry, and correlation")
        
        results['comparison']['conclusion'] = self._analyze_three_electron(results)
        
        return results
    
    def compare_electron_proton_neutron_system(self, t_max: float = 1.0, dt: float = 0.01) -> Dict:
        """
        Compare classical and quantum simulations for electron-proton-neutron system.
        
        Args:
            t_max: Maximum time for classical simulation
            dt: Time step for classical simulation
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 60)
        print("Comparing Electron-Proton-Neutron System")
        print("=" * 60)
        
        results = {
            'system': 'electron-proton-neutron',
            'classical': {},
            'quantum': {},
            'comparison': {}
        }
        
        # Classical simulation
        print("\nRunning Classical Simulation...")
        classical_sys = create_electron_proton_neutron_system()
        classical_result = classical_sys.simulate(t_max=t_max, dt=dt)
        
        final_energy = classical_result['energies'][-1]['total']
        initial_energy = classical_result['energies'][0]['total']
        energy_conservation_error = abs(final_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0
        
        results['classical'] = {
            'computation_time': classical_result['computation_time'],
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_conservation_error': energy_conservation_error,
            'n_steps': classical_result['n_steps'],
            'average_kinetic': np.mean([e['kinetic'] for e in classical_result['energies']]),
            'average_potential': np.mean([e['potential'] for e in classical_result['energies']])
        }
        
        print(f"  Computation time: {classical_result['computation_time']:.4f} seconds")
        print(f"  Initial energy: {initial_energy:.6f} a.u.")
        print(f"  Final energy: {final_energy:.6f} a.u.")
        print(f"  Energy conservation error: {energy_conservation_error:.2e}")
        
        # Quantum simulation - Hydrogen-like
        print("\nRunning Quantum Simulation (Hydrogen-like)...")
        quantum_sys = ElectronProtonNeutronQuantumSystem()
        quantum_result = quantum_sys.solve_hydrogen_like(grid_size=200, grid_range=(-10, 10))
        
        results['quantum'] = {
            'computation_time': quantum_result['computation_time'],
            'ground_state_energy': quantum_result['ground_state_energy'],
            'analytical_energy': quantum_result['analytical_hydrogen_energy'],
            'energy_error': abs(quantum_result['ground_state_energy'] - quantum_result['analytical_hydrogen_energy'])
        }
        
        print(f"  Computation time: {quantum_result['computation_time']:.4f} seconds")
        print(f"  Ground state energy: {quantum_result['ground_state_energy']:.6f} a.u.")
        print(f"  Analytical (hydrogen): {quantum_result['analytical_hydrogen_energy']:.6f} a.u.")
        print(f"  Numerical error: {results['quantum']['energy_error']:.2e}")
        
        # Full 3-body quantum
        print("\nRunning Quantum Simulation (Full 3-body)...")
        quantum_result_full = quantum_sys.solve_full_3body_quantum(grid_size=50)
        
        results['quantum']['full_3body'] = {
            'computation_time': quantum_result_full['computation_time'],
            'ground_state_energy': quantum_result_full['ground_state_energy']
        }
        
        print(f"  Computation time: {quantum_result_full['computation_time']:.4f} seconds")
        print(f"  Ground state energy: {quantum_result_full['ground_state_energy']:.6f} a.u.")
        
        # Comparison metrics
        print("\n" + "-" * 60)
        print("COMPARISON RESULTS")
        print("-" * 60)
        
        classical_time = results['classical']['computation_time']
        quantum_time = results['quantum']['computation_time']
        quantum_full_time = results['quantum']['full_3body']['computation_time']
        
        speed_ratio = classical_time / quantum_time if quantum_time > 0 else float('inf')
        speed_ratio_full = classical_time / quantum_full_time if quantum_full_time > 0 else float('inf')
        
        results['comparison']['speed'] = {
            'classical_time': classical_time,
            'quantum_time': quantum_time,
            'quantum_full_time': quantum_full_time,
            'classical_vs_quantum_ratio': speed_ratio,
            'classical_vs_full_ratio': speed_ratio_full,
            'faster_method': 'classical' if classical_time < min(quantum_time, quantum_full_time) else 'quantum'
        }
        
        print(f"\nSpeed Comparison:")
        print(f"  Classical: {classical_time:.4f} s")
        print(f"  Quantum (H-like): {quantum_time:.4f} s ({speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'})")
        print(f"  Quantum (Full): {quantum_full_time:.4f} s ({speed_ratio_full:.2f}x {'faster' if speed_ratio_full > 1 else 'slower'})")
        
        classical_avg_energy = results['classical']['average_kinetic'] + results['classical']['average_potential']
        quantum_ground_energy = results['quantum']['ground_state_energy']
        
        energy_difference = abs(classical_avg_energy - quantum_ground_energy)
        
        results['comparison']['accuracy'] = {
            'classical_avg_energy': classical_avg_energy,
            'quantum_ground_energy': quantum_ground_energy,
            'energy_difference': energy_difference,
            'classical_energy_conservation': energy_conservation_error,
            'quantum_numerical_error': results['quantum']['energy_error']
        }
        
        print(f"\nAccuracy Comparison:")
        print(f"  Classical average energy: {classical_avg_energy:.6f} a.u.")
        print(f"  Quantum ground state: {quantum_ground_energy:.6f} a.u.")
        print(f"  Energy difference: {energy_difference:.6f} a.u.")
        print(f"  Classical energy conservation error: {energy_conservation_error:.2e}")
        print(f"  Quantum numerical error: {results['quantum']['energy_error']:.2e}")
        
        print(f"\nPhysical Correctness:")
        print(f"  Classical: Cannot describe bound states, electron would spiral into nucleus")
        print(f"  Quantum: Correctly describes stable atomic structure with discrete energy levels")
        
        results['comparison']['conclusion'] = self._analyze_electron_proton_neutron(results)
        
        return results
    
    def _analyze_three_electron(self, results: Dict) -> str:
        """Analyze and provide conclusion for 3-electron system."""
        speed = results['comparison']['speed']
        accuracy = results['comparison']['accuracy']
        
        conclusion = []
        conclusion.append("For 3-electron system:")
        
        if speed['classical_time'] < speed['quantum_hf_time']:
            conclusion.append(f"- Classical is {speed['classical_vs_hf_ratio']:.2f}x faster than quantum (HF)")
        else:
            conclusion.append(f"- Quantum (HF) is {1/speed['classical_vs_hf_ratio']:.2f}x faster than classical")
        
        conclusion.append("- Quantum mechanics is physically correct (Pauli exclusion, electron correlation)")
        conclusion.append("- Classical mechanics violates quantum principles for electrons")
        conclusion.append(f"- Energy difference: {accuracy['energy_difference']:.6f} a.u.")
        
        if accuracy['classical_energy_conservation'] < 1e-3:
            conclusion.append("- Classical maintains good energy conservation")
        
        return "\n".join(conclusion)
    
    def _analyze_electron_proton_neutron(self, results: Dict) -> str:
        """Analyze and provide conclusion for electron-proton-neutron system."""
        speed = results['comparison']['speed']
        accuracy = results['comparison']['accuracy']
        
        conclusion = []
        conclusion.append("For electron-proton-neutron system:")
        
        if speed['classical_time'] < speed['quantum_time']:
            conclusion.append(f"- Classical is {speed['classical_vs_quantum_ratio']:.2f}x faster than quantum")
        else:
            conclusion.append(f"- Quantum is {1/speed['classical_vs_quantum_ratio']:.2f}x faster than classical")
        
        conclusion.append("- Quantum mechanics is essential for atomic structure (prevents electron collapse)")
        conclusion.append("- Classical mechanics predicts unphysical collapse of atom")
        conclusion.append(f"- Quantum ground state: {accuracy['quantum_ground_energy']:.6f} a.u.")
        conclusion.append(f"- Classical average: {accuracy['classical_avg_energy']:.6f} a.u.")
        
        return "\n".join(conclusion)
    
    def generate_report(self, results_3e: Dict, results_epn: Dict) -> str:
        """Generate comprehensive comparison report."""
        report = []
        report.append("=" * 80)
        report.append("CLASSICAL vs QUANTUM SIMULATION COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # 3-electron system
        report.append("SYSTEM 1: 3-ELECTRON SYSTEM")
        report.append("-" * 80)
        report.append(f"Classical Simulation:")
        report.append(f"  Computation time: {results_3e['classical']['computation_time']:.4f} s")
        report.append(f"  Energy conservation error: {results_3e['classical']['energy_conservation_error']:.2e}")
        report.append(f"  Average energy: {results_3e['classical']['average_kinetic'] + results_3e['classical']['average_potential']:.6f} a.u.")
        report.append("")
        report.append(f"Quantum Simulation (Hartree-Fock):")
        report.append(f"  Computation time: {results_3e['quantum']['hartree_fock']['computation_time']:.4f} s")
        report.append(f"  Ground state energy: {results_3e['quantum']['hartree_fock']['ground_state_energy']:.6f} a.u.")
        report.append("")
        report.append(f"Quantum Simulation (Variational):")
        report.append(f"  Computation time: {results_3e['quantum']['variational']['computation_time']:.4f} s")
        report.append(f"  Ground state energy: {results_3e['quantum']['variational']['ground_state_energy']:.6f} a.u.")
        report.append("")
        report.append(f"Conclusion: {results_3e['comparison']['conclusion']}")
        report.append("")
        
        # Electron-proton-neutron system
        report.append("SYSTEM 2: ELECTRON-PROTON-NEUTRON SYSTEM")
        report.append("-" * 80)
        report.append(f"Classical Simulation:")
        report.append(f"  Computation time: {results_epn['classical']['computation_time']:.4f} s")
        report.append(f"  Energy conservation error: {results_epn['classical']['energy_conservation_error']:.2e}")
        report.append(f"  Average energy: {results_epn['classical']['average_kinetic'] + results_epn['classical']['average_potential']:.6f} a.u.")
        report.append("")
        report.append(f"Quantum Simulation:")
        report.append(f"  Computation time: {results_epn['quantum']['computation_time']:.4f} s")
        report.append(f"  Ground state energy: {results_epn['quantum']['ground_state_energy']:.6f} a.u.")
        report.append(f"  Numerical error: {results_epn['quantum']['energy_error']:.2e}")
        report.append("")
        report.append(f"Conclusion: {results_epn['comparison']['conclusion']}")
        report.append("")
        
        # Overall summary
        report.append("=" * 80)
        report.append("OVERALL SUMMARY")
        report.append("=" * 80)
        report.append("")
        report.append("SPEED:")
        report.append("  - Classical mechanics is generally faster for simple time evolution")
        report.append("  - Quantum mechanics requires iterative/self-consistent methods")
        report.append("  - Speed depends on system complexity and desired accuracy")
        report.append("")
        report.append("ACCURACY:")
        report.append("  - Quantum mechanics is physically correct for atomic/molecular systems")
        report.append("  - Classical mechanics fails for quantum systems (violates Pauli exclusion,")
        report.append("    predicts unphysical collapse, cannot describe bound states)")
        report.append("  - Energy values differ significantly between classical and quantum")
        report.append("")
        report.append("RECOMMENDATION:")
        report.append("  - Use QUANTUM mechanics for electrons, atoms, and molecules")
        report.append("  - Use CLASSICAL mechanics for macroscopic objects or when quantum")
        report.append("    effects are negligible")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

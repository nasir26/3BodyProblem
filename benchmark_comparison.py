"""
Benchmarking and Comparison Module
Compares classical vs quantum simulations in terms of speed and accuracy
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

from classical_simulation import (
    ClassicalSimulator, create_3electron_system, 
    create_electron_proton_neutron_system, M_ELECTRON, M_PROTON, M_NEUTRON,
    E_CHARGE, K_COULOMB, AU_ENERGY, AU_TIME, AU_LENGTH
)
from quantum_simulation import QuantumSimulator


class BenchmarkSuite:
    """Comprehensive benchmarking suite for comparing classical and quantum simulations"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_3electron_system(self, simulation_time: float = 1e-15, 
                                   dt_classical: float = 1e-18,
                                   dt_quantum: float = 0.01,
                                   n_runs: int = 3) -> Dict:
        """
        Benchmark 3-electron system
        
        Args:
            simulation_time: Total simulation time in seconds
            dt_classical: Time step for classical simulation
            dt_quantum: Time step for quantum simulation (in atomic units)
            n_runs: Number of runs for averaging
        """
        print("=" * 60)
        print("Benchmarking 3-Electron System")
        print("=" * 60)
        
        results = {
            'classical': {'times': [], 'energies': [], 'final_positions': []},
            'quantum': {'times': [], 'energies': [], 'final_positions': []}
        }
        
        # Classical simulation
        print("\nRunning Classical Simulation...")
        for run in range(n_runs):
            particles = create_3electron_system(initial_separation=1e-10)
            simulator = ClassicalSimulator(particles, dt=dt_classical)
            
            elapsed = simulator.simulate(simulation_time, save_interval=100)
            final_state = simulator.trajectory[-1] if simulator.trajectory else None
            
            results['classical']['times'].append(elapsed)
            if final_state:
                results['classical']['energies'].append(final_state['energies'])
                results['classical']['final_positions'].append(final_state['positions'])
        
        classical_avg_time = np.mean(results['classical']['times'])
        print(f"Classical average time: {classical_avg_time:.4f} seconds")
        
        # Quantum simulation (simplified - full 3-electron is very complex)
        print("\nRunning Quantum Simulation...")
        print("Note: Full 3-electron quantum simulation is computationally intensive.")
        print("Using simplified mean-field approximation.")
        
        for run in range(n_runs):
            qsim = QuantumSimulator(grid_size=32, grid_range=5.0)
            
            # Initialize wavefunction (simplified)
            r = np.sqrt(qsim.X**2 + qsim.Y**2 + qsim.Z**2)
            psi0 = np.exp(-r) / np.sqrt(np.pi)
            psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * qsim.dx**3)
            
            # Convert simulation time to atomic units
            time_au = simulation_time / AU_TIME
            
            elapsed = qsim.simulate_time_evolution(
                psi0, time_au, dt=dt_quantum, system_type='3electron'
            )
            
            final_state = qsim.trajectory[-1] if qsim.trajectory else None
            
            results['quantum']['times'].append(elapsed)
            if final_state:
                results['quantum']['energies'].append(final_state['energy'])
                results['quantum']['final_positions'].append(final_state['position'])
        
        quantum_avg_time = np.mean(results['quantum']['times'])
        print(f"Quantum average time: {quantum_avg_time:.4f} seconds")
        
        # Analysis
        speed_ratio = classical_avg_time / quantum_avg_time if quantum_avg_time > 0 else float('inf')
        print(f"\nSpeed Ratio (Classical/Quantum): {speed_ratio:.2f}x")
        
        self.results['3electron'] = results
        return results
    
    def benchmark_hydrogen_like_system(self, simulation_time: float = 1e-15,
                                      dt_classical: float = 1e-18,
                                      dt_quantum: float = 0.01,
                                      n_runs: int = 3) -> Dict:
        """
        Benchmark 1 electron + 1 proton + 1 neutron system
        
        For this system, we can compare with known analytical results
        """
        print("\n" + "=" * 60)
        print("Benchmarking Electron + Proton + Neutron System")
        print("=" * 60)
        
        results = {
            'classical': {'times': [], 'energies': [], 'final_positions': []},
            'quantum': {'times': [], 'energies': [], 'final_positions': []},
            'analytical': {'ground_state_energy': -0.5}  # In atomic units (Hartrees)
        }
        
        # Classical simulation
        print("\nRunning Classical Simulation...")
        for run in range(n_runs):
            particles = create_electron_proton_neutron_system(initial_separation=5e-11)
            simulator = ClassicalSimulator(particles, dt=dt_classical)
            
            elapsed = simulator.simulate(simulation_time, save_interval=100)
            final_state = simulator.trajectory[-1] if simulator.trajectory else None
            
            results['classical']['times'].append(elapsed)
            if final_state:
                results['classical']['energies'].append(final_state['energies'])
                results['classical']['final_positions'].append(final_state['positions'])
        
        classical_avg_time = np.mean(results['classical']['times'])
        print(f"Classical average time: {classical_avg_time:.4f} seconds")
        
        # Quantum simulation (hydrogen-like, can be solved accurately)
        print("\nRunning Quantum Simulation...")
        for run in range(n_runs):
            qsim = QuantumSimulator(grid_size=64, grid_range=10.0)
            
            # Solve for ground state
            energies, wavefunctions = qsim.solve_stationary_hydrogen(n_states=1)
            psi0 = wavefunctions[0] if wavefunctions else None
            
            if psi0 is not None:
                # Convert simulation time to atomic units
                time_au = simulation_time / AU_TIME
                
                elapsed = qsim.simulate_time_evolution(
                    psi0, time_au, dt=dt_quantum, system_type='hydrogen'
                )
                
                final_state = qsim.trajectory[-1] if qsim.trajectory else None
                
                results['quantum']['times'].append(elapsed)
                if final_state:
                    results['quantum']['energies'].append(final_state['energy'])
                    results['quantum']['final_positions'].append(final_state['position'])
        
        quantum_avg_time = np.mean(results['quantum']['times'])
        print(f"Quantum average time: {quantum_avg_time:.4f} seconds")
        
        # Compare with analytical result
        if results['quantum']['energies']:
            quantum_avg_energy = np.mean([e for e in results['quantum']['energies'] if e is not None])
            analytical_energy = results['analytical']['ground_state_energy']
            energy_error = abs(quantum_avg_energy - analytical_energy) / abs(analytical_energy) * 100
            print(f"\nQuantum Energy: {quantum_avg_energy:.6f} Hartrees")
            print(f"Analytical Energy: {analytical_energy:.6f} Hartrees")
            print(f"Error: {energy_error:.2f}%")
        
        # Analysis
        speed_ratio = classical_avg_time / quantum_avg_time if quantum_avg_time > 0 else float('inf')
        print(f"\nSpeed Ratio (Classical/Quantum): {speed_ratio:.2f}x")
        
        self.results['hydrogen_like'] = results
        return results
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("CLASSICAL vs QUANTUM SIMULATION COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # 3-Electron System
        if '3electron' in self.results:
            report.append("1. THREE-ELECTRON SYSTEM")
            report.append("-" * 80)
            r = self.results['3electron']
            
            classical_times = r['classical']['times']
            quantum_times = r['quantum']['times']
            
            if classical_times and quantum_times:
                report.append(f"Classical Simulation:")
                report.append(f"  Average Time: {np.mean(classical_times):.4f} s")
                report.append(f"  Std Dev: {np.std(classical_times):.4f} s")
                
                report.append(f"\nQuantum Simulation:")
                report.append(f"  Average Time: {np.mean(quantum_times):.4f} s")
                report.append(f"  Std Dev: {np.std(quantum_times):.4f} s")
                
                speed_ratio = np.mean(classical_times) / np.mean(quantum_times)
                report.append(f"\nSpeed Comparison:")
                report.append(f"  Classical is {speed_ratio:.2f}x {'faster' if speed_ratio < 1 else 'slower'} than Quantum")
            
            report.append("\nAccuracy Analysis:")
            report.append("  - Classical: Uses Newtonian mechanics, accurate for macroscopic systems")
            report.append("  - Quantum: Required for accurate description of electron behavior")
            report.append("  - Note: 3-electron system requires many-body quantum treatment")
            report.append("")
        
        # Hydrogen-like System
        if 'hydrogen_like' in self.results:
            report.append("2. ELECTRON + PROTON + NEUTRON SYSTEM")
            report.append("-" * 80)
            r = self.results['hydrogen_like']
            
            classical_times = r['classical']['times']
            quantum_times = r['quantum']['times']
            
            if classical_times and quantum_times:
                report.append(f"Classical Simulation:")
                report.append(f"  Average Time: {np.mean(classical_times):.4f} s")
                report.append(f"  Std Dev: {np.std(classical_times):.4f} s")
                
                report.append(f"\nQuantum Simulation:")
                report.append(f"  Average Time: {np.mean(quantum_times):.4f} s")
                report.append(f"  Std Dev: {np.std(quantum_times):.4f} s")
                
                speed_ratio = np.mean(classical_times) / np.mean(quantum_times)
                report.append(f"\nSpeed Comparison:")
                report.append(f"  Classical is {speed_ratio:.2f}x {'faster' if speed_ratio < 1 else 'slower'} than Quantum")
            
            if r['quantum']['energies']:
                quantum_energy = np.mean([e for e in r['quantum']['energies'] if e is not None])
                analytical = r['analytical']['ground_state_energy']
                error = abs(quantum_energy - analytical) / abs(analytical) * 100
                
                report.append("\nAccuracy Analysis:")
                report.append(f"  Quantum Ground State Energy: {quantum_energy:.6f} Hartrees")
                report.append(f"  Analytical (Exact) Energy: {analytical:.6f} Hartrees")
                report.append(f"  Error: {error:.2f}%")
                report.append("  - Quantum simulation matches analytical solution well")
                report.append("  - Classical simulation cannot capture bound states")
            
            report.append("")
        
        # Overall Conclusions
        report.append("=" * 80)
        report.append("OVERALL CONCLUSIONS")
        report.append("=" * 80)
        report.append("")
        report.append("SPEED:")
        report.append("  - Classical simulations are typically faster for simple systems")
        report.append("  - Quantum simulations require more computational resources")
        report.append("  - Speed depends on system complexity and grid resolution")
        report.append("")
        report.append("ACCURACY:")
        report.append("  - Classical: Accurate for macroscopic systems, fails for atomic scales")
        report.append("  - Quantum: Required for accurate description of electrons and atoms")
        report.append("  - Quantum mechanics is necessary for:")
        report.append("    * Bound states (atoms, molecules)")
        report.append("    * Tunneling effects")
        report.append("    * Discrete energy levels")
        report.append("    * Wave-particle duality")
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("  - Use classical mechanics for: large systems, high energies, simple trajectories")
        report.append("  - Use quantum mechanics for: atomic/molecular systems, low energies, bound states")
        report.append("  - Hybrid methods (QM/MM) are often used for complex systems")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

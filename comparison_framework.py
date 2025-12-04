"""
Comparison Framework for Classical vs Quantum Simulations
Compares speed and accuracy of both approaches
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from classical_simulation import (
    ClassicalSimulator, 
    create_3electron_system,
    create_electron_proton_neutron_system
)
from quantum_simulation import (
    QuantumSimulator,
    create_3electron_quantum_system,
    create_electron_proton_neutron_quantum_system
)


class SimulationComparator:
    """Compare classical and quantum simulations"""
    
    def __init__(self):
        self.results = {}
    
    def run_comparison(self, system_name: str, t_span: Tuple[float, float], 
                      n_points: int = 100) -> Dict:
        """
        Run both classical and quantum simulations and compare
        
        Args:
            system_name: '3electron' or 'electron_proton_neutron'
            t_span: Time span for simulation (seconds)
            n_points: Number of time points to evaluate
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*60}")
        print(f"Comparing simulations for: {system_name}")
        print(f"{'='*60}\n")
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Create systems
        if system_name == '3electron':
            classical_particles = create_3electron_system()
            quantum_particles = create_3electron_quantum_system()
        elif system_name == 'electron_proton_neutron':
            classical_particles = create_electron_proton_neutron_system()
            quantum_particles = create_electron_proton_neutron_quantum_system()
        else:
            raise ValueError(f"Unknown system: {system_name}")
        
        # Run classical simulation
        print("Running classical simulation...")
        classical_sim = ClassicalSimulator(classical_particles)
        classical_results = classical_sim.simulate(t_span, t_eval)
        
        # Run quantum simulation
        print("Running quantum simulation...")
        quantum_sim = QuantumSimulator(quantum_particles)
        quantum_results = quantum_sim.simulate(t_span, t_eval)
        
        # Compare results
        comparison = self._compare_results(
            classical_results, quantum_results, system_name
        )
        
        self.results[system_name] = {
            'classical': classical_results,
            'quantum': quantum_results,
            'comparison': comparison
        }
        
        return self.results[system_name]
    
    def _compare_results(self, classical: Dict, quantum: Dict, 
                        system_name: str) -> Dict:
        """
        Compare classical and quantum results
        
        Args:
            classical: Classical simulation results
            quantum: Quantum simulation results
            system_name: Name of the system
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        # Speed comparison
        comparison['speed'] = {
            'classical_time': classical['computation_time'],
            'quantum_time': quantum['computation_time'],
            'speedup': classical['computation_time'] / quantum['computation_time'] 
                       if quantum['computation_time'] > 0 else np.inf,
            'faster': 'classical' if classical['computation_time'] < quantum['computation_time'] 
                     else 'quantum'
        }
        
        # Energy conservation (accuracy metric for classical)
        if len(classical['total_energy']) > 1:
            energy_variation_classical = np.std(classical['total_energy']) / np.abs(np.mean(classical['total_energy']))
            comparison['energy_conservation_classical'] = {
                'mean_energy': np.mean(classical['total_energy']),
                'std_energy': np.std(classical['total_energy']),
                'relative_variation': energy_variation_classical,
                'conserved': energy_variation_classical < 0.01  # 1% threshold
            }
        
        # Energy stability (accuracy metric for quantum)
        if len(quantum['energy']) > 1:
            energy_variation_quantum = np.std(quantum['energy']) / np.abs(np.mean(quantum['energy']))
            comparison['energy_stability_quantum'] = {
                'mean_energy': np.mean(quantum['energy']),
                'std_energy': np.std(quantum['energy']),
                'relative_variation': energy_variation_quantum,
                'stable': energy_variation_quantum < 0.01  # 1% threshold
            }
        
        # Position tracking (if comparable)
        # Classical: center of mass
        if 'positions' in classical:
            classical_com = np.mean(classical['positions'], axis=1)
            comparison['classical_com_motion'] = {
                'initial_position': classical_com[0],
                'final_position': classical_com[-1],
                'displacement': np.linalg.norm(classical_com[-1] - classical_com[0])
            }
        
        # Quantum: expectation value of position
        if 'position' in quantum:
            comparison['quantum_position_evolution'] = {
                'initial_position': quantum['position'][0],
                'final_position': quantum['position'][-1],
                'displacement': np.abs(quantum['position'][-1] - quantum['position'][0])
            }
        
        # Overall accuracy assessment
        # Classical: Good for macroscopic systems, exact for point particles
        # Quantum: More accurate for atomic-scale systems
        comparison['accuracy_assessment'] = {
            'classical_appropriate': system_name == 'electron_proton_neutron',  # Better for heavier particles
            'quantum_appropriate': True,  # Always more accurate at atomic scale
            'recommendation': self._get_recommendation(comparison, system_name)
        }
        
        return comparison
    
    def _get_recommendation(self, comparison: Dict, system_name: str) -> str:
        """Get recommendation on which method to use"""
        speed_ratio = comparison['speed']['speedup']
        
        if system_name == '3electron':
            # For light particles, quantum is more accurate
            if speed_ratio > 2:
                return "Classical is faster but quantum is more accurate for electrons. Use quantum for accuracy."
            else:
                return "Quantum is recommended for electron systems due to quantum effects."
        else:
            # For heavier particles, classical might be acceptable
            if speed_ratio > 5:
                return "Classical is much faster. Use classical for speed, quantum for accuracy."
            else:
                return "Quantum is recommended for accurate atomic-scale simulations."
    
    def generate_report(self) -> str:
        """Generate text report of all comparisons"""
        report = []
        report.append("="*80)
        report.append("SIMULATION COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        for system_name, results in self.results.items():
            report.append(f"\n{'='*80}")
            report.append(f"SYSTEM: {system_name.upper()}")
            report.append(f"{'='*80}\n")
            
            comp = results['comparison']
            
            # Speed
            report.append("SPEED COMPARISON:")
            report.append(f"  Classical computation time: {comp['speed']['classical_time']:.6f} s")
            report.append(f"  Quantum computation time:  {comp['speed']['quantum_time']:.6f} s")
            report.append(f"  Speedup: {comp['speed']['speedup']:.2f}x")
            report.append(f"  Faster method: {comp['speed']['faster'].upper()}")
            report.append("")
            
            # Energy conservation
            if 'energy_conservation_classical' in comp:
                ec = comp['energy_conservation_classical']
                report.append("CLASSICAL ENERGY CONSERVATION:")
                report.append(f"  Mean energy: {ec['mean_energy']:.6e} J")
                report.append(f"  Energy variation: {ec['relative_variation']*100:.4f}%")
                report.append(f"  Energy conserved: {'Yes' if ec['conserved'] else 'No'}")
                report.append("")
            
            if 'energy_stability_quantum' in comp:
                es = comp['energy_stability_quantum']
                report.append("QUANTUM ENERGY STABILITY:")
                report.append(f"  Mean energy: {es['mean_energy']:.6e} J")
                report.append(f"  Energy variation: {es['relative_variation']*100:.4f}%")
                report.append(f"  Energy stable: {'Yes' if es['stable'] else 'No'}")
                report.append("")
            
            # Recommendation
            if 'accuracy_assessment' in comp:
                rec = comp['accuracy_assessment']
                report.append("RECOMMENDATION:")
                report.append(f"  {rec['recommendation']}")
                report.append("")
        
        report.append("="*80)
        report.append("SUMMARY")
        report.append("="*80)
        report.append("\nKey Findings:")
        report.append("1. Classical mechanics is typically faster for simple systems")
        report.append("2. Quantum mechanics is more accurate for atomic-scale systems")
        report.append("3. Choice depends on required accuracy vs computational cost")
        report.append("4. For electrons, quantum effects are significant and should be included")
        report.append("5. For heavier particles, classical may be acceptable approximation")
        
        return "\n".join(report)
    
    def plot_comparison(self, system_name: str, save_path: str = None):
        """Create visualization comparing classical and quantum results"""
        if system_name not in self.results:
            raise ValueError(f"No results for system: {system_name}")
        
        results = self.results[system_name]
        classical = results['classical']
        quantum = results['quantum']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Classical vs Quantum Simulation: {system_name}', 
                     fontsize=16, fontweight='bold')
        
        # Energy comparison
        ax1 = axes[0, 0]
        ax1.plot(classical['time'] * 1e18, classical['total_energy'] * 1e18, 
                'b-', label='Classical Total Energy', linewidth=2)
        ax1.plot(quantum['time'] * 1e18, quantum['energy'].real * 1e18, 
                'r--', label='Quantum Energy', linewidth=2)
        ax1.set_xlabel('Time (as)')
        ax1.set_ylabel('Energy (aJ)')
        ax1.set_title('Energy Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Position evolution
        ax2 = axes[0, 1]
        if 'positions' in classical:
            com = np.mean(classical['positions'], axis=1)
            ax2.plot(classical['time'] * 1e18, com[:, 0] * 1e12, 
                    'b-', label='Classical COM (x)', linewidth=2)
        if 'position' in quantum:
            ax2.plot(quantum['time'] * 1e18, quantum['position'] * 1e12, 
                    'r--', label='Quantum <x>', linewidth=2)
        ax2.set_xlabel('Time (as)')
        ax2.set_ylabel('Position (pm)')
        ax2.set_title('Position Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Energy conservation (classical)
        ax3 = axes[1, 0]
        if len(classical['total_energy']) > 1:
            energy_error = (classical['total_energy'] - classical['total_energy'][0]) / classical['total_energy'][0] * 100
            ax3.plot(classical['time'] * 1e18, energy_error, 
                    'b-', linewidth=2)
            ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Time (as)')
            ax3.set_ylabel('Energy Error (%)')
            ax3.set_title('Classical Energy Conservation')
            ax3.grid(True, alpha=0.3)
        
        # Computation time comparison
        ax4 = axes[1, 1]
        methods = ['Classical', 'Quantum']
        times = [results['comparison']['speed']['classical_time'],
                results['comparison']['speed']['quantum_time']]
        colors = ['blue', 'red']
        bars = ax4.bar(methods, times, color=colors, alpha=0.7)
        ax4.set_ylabel('Computation Time (s)')
        ax4.set_title('Speed Comparison')
        ax4.set_yscale('log')
        for i, (bar, time_val) in enumerate(zip(bars, times)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.4f}s',
                    ha='center', va='bottom')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig

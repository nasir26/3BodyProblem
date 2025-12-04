"""
Comparison Framework for Classical vs Quantum Simulations
Measures speed, accuracy, and energy conservation
"""

import numpy as np
from typing import Dict, List, Tuple
import time
from classical_simulation import (
    create_three_electron_system,
    create_electron_proton_neutron_system
)
from quantum_simulation import (
    create_three_electron_quantum_system,
    create_electron_proton_neutron_quantum_system
)


class SimulationComparison:
    """Compare classical and quantum simulations"""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.results = {}
    
    def compute_energy_conservation_error(self, energies: np.ndarray) -> Dict:
        """Measure energy conservation (should be constant in isolated system)"""
        if len(energies) < 2:
            return {'mean_error': 0.0, 'max_error': 0.0, 'std_error': 0.0}
        
        initial_energy = energies[0]
        relative_errors = np.abs((energies - initial_energy) / initial_energy)
        
        return {
            'mean_error': np.mean(relative_errors),
            'max_error': np.max(relative_errors),
            'std_error': np.std(relative_errors),
            'energy_drift': energies[-1] - energies[0]
        }
    
    def compute_trajectory_stability(self, positions: np.ndarray) -> Dict:
        """Measure trajectory stability and boundedness"""
        if positions.ndim < 2:
            return {'max_displacement': 0.0, 'trajectory_variance': 0.0}
        
        # Maximum displacement from initial position
        initial_pos = positions[0]
        displacements = np.linalg.norm(positions - initial_pos, axis=-1)
        max_displacement = np.max(displacements)
        
        # Variance in trajectory
        trajectory_variance = np.var(positions, axis=0).mean()
        
        return {
            'max_displacement': max_displacement,
            'trajectory_variance': trajectory_variance,
            'mean_displacement': np.mean(displacements)
        }
    
    def compare_simulations(self, classical_result: Dict, quantum_result: Dict) -> Dict:
        """Comprehensive comparison of classical and quantum results"""
        comparison = {
            'system_name': self.system_name,
            'classical': {},
            'quantum': {},
            'comparison': {}
        }
        
        # Speed comparison
        comparison['classical']['computation_time'] = classical_result.get('computation_time', 0)
        comparison['quantum']['computation_time'] = quantum_result.get('computation_time', 0)
        comparison['comparison']['speed_ratio'] = (
            quantum_result.get('computation_time', 1e-10) / 
            max(classical_result.get('computation_time', 1e-10), 1e-10)
        )
        
        # Energy conservation
        if 'energy' in classical_result:
            classical_energy_error = self.compute_energy_conservation_error(classical_result['energy'])
            comparison['classical']['energy_conservation'] = classical_energy_error
        
        if 'energy' in quantum_result:
            quantum_energy_error = self.compute_energy_conservation_error(quantum_result['energy'])
            comparison['quantum']['energy_conservation'] = quantum_energy_error
        
        # Trajectory analysis
        if 'positions' in classical_result:
            classical_traj = self.compute_trajectory_stability(classical_result['positions'])
            comparison['classical']['trajectory'] = classical_traj
        
        if 'positions' in quantum_result:
            quantum_traj = self.compute_trajectory_stability(quantum_result['positions'].reshape(-1, 1))
            comparison['quantum']['trajectory'] = quantum_traj
        
        # Number of time steps
        comparison['classical']['n_steps'] = classical_result.get('n_steps', 0)
        comparison['quantum']['n_steps'] = quantum_result.get('n_steps', 0)
        
        # Success status
        comparison['classical']['success'] = classical_result.get('success', False)
        comparison['quantum']['success'] = quantum_result.get('success', False)
        
        # Determine which is better
        comparison['comparison']['better_speed'] = (
            'classical' if comparison['comparison']['speed_ratio'] > 1.0 else 'quantum'
        )
        
        # Accuracy assessment (lower energy error is better)
        if 'energy_conservation' in comparison['classical'] and 'energy_conservation' in comparison['quantum']:
            classical_energy_err = comparison['classical']['energy_conservation']['mean_error']
            quantum_energy_err = comparison['quantum']['energy_conservation']['mean_error']
            
            if classical_energy_err < quantum_energy_err:
                comparison['comparison']['better_energy_conservation'] = 'classical'
            else:
                comparison['comparison']['better_energy_conservation'] = 'quantum'
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Print formatted comparison results"""
        print(f"\n{'='*80}")
        print(f"COMPARISON RESULTS: {comparison['system_name']}")
        print(f"{'='*80}\n")
        
        print("SPEED COMPARISON:")
        print(f"  Classical computation time: {comparison['classical']['computation_time']:.6f} seconds")
        print(f"  Quantum computation time: {comparison['quantum']['computation_time']:.6f} seconds")
        print(f"  Speed ratio (quantum/classical): {comparison['comparison']['speed_ratio']:.4f}")
        print(f"  → {'Classical' if comparison['comparison']['speed_ratio'] > 1.0 else 'Quantum'} is faster")
        
        print("\nENERGY CONSERVATION:")
        if 'energy_conservation' in comparison['classical']:
            ce = comparison['classical']['energy_conservation']
            print(f"  Classical mean relative error: {ce['mean_error']:.2e}")
            print(f"  Classical max relative error: {ce['max_error']:.2e}")
            print(f"  Classical energy drift: {ce['energy_drift']:.2e} J")
        
        if 'energy_conservation' in comparison['quantum']:
            qe = comparison['quantum']['energy_conservation']
            print(f"  Quantum mean relative error: {qe['mean_error']:.2e}")
            print(f"  Quantum max relative error: {qe['max_error']:.2e}")
            print(f"  Quantum energy drift: {qe['energy_drift']:.2e} J")
        
        if 'better_energy_conservation' in comparison['comparison']:
            print(f"  → {'Classical' if comparison['comparison']['better_energy_conservation'] == 'classical' else 'Quantum'} has better energy conservation")
        
        print("\nTRAJECTORY STABILITY:")
        if 'trajectory' in comparison['classical']:
            ct = comparison['classical']['trajectory']
            print(f"  Classical max displacement: {ct['max_displacement']:.2e} m")
            print(f"  Classical trajectory variance: {ct['trajectory_variance']:.2e} m²")
        
        if 'trajectory' in comparison['quantum']:
            qt = comparison['quantum']['trajectory']
            print(f"  Quantum max displacement: {qt['max_displacement']:.2e} m")
            print(f"  Quantum trajectory variance: {qt['trajectory_variance']:.2e} m²")
        
        print("\nNUMERICAL DETAILS:")
        print(f"  Classical steps: {comparison['classical']['n_steps']}")
        print(f"  Quantum steps: {comparison['quantum']['n_steps']}")
        print(f"  Classical success: {comparison['classical']['success']}")
        print(f"  Quantum success: {comparison['quantum']['success']}")
        
        print(f"\n{'='*80}\n")


def run_three_electron_comparison(t_span: Tuple[float, float] = (0.0, 1e-15),
                                  n_points: int = 100) -> Dict:
    """Run comparison for 3-electron system"""
    print("Running 3-electron system comparison...")
    
    # Create systems
    classical_system = create_three_electron_system()
    quantum_system = create_three_electron_quantum_system(grid_size=64, grid_range=2e-9)
    
    # Time points
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    # Run classical simulation
    print("  Running classical simulation...")
    classical_result = classical_system.simulate(t_span, t_eval)
    
    # Run quantum simulation
    print("  Running quantum simulation...")
    quantum_result = quantum_system.simulate(t_span, t_eval)
    
    # Compare
    comparator = SimulationComparison("3-Electron System")
    comparison = comparator.compare_simulations(classical_result, quantum_result)
    comparator.print_comparison(comparison)
    
    return {
        'comparison': comparison,
        'classical_result': classical_result,
        'quantum_result': quantum_result
    }


def run_electron_proton_neutron_comparison(t_span: Tuple[float, float] = (0.0, 1e-15),
                                            n_points: int = 100) -> Dict:
    """Run comparison for electron-proton-neutron system"""
    print("Running electron-proton-neutron system comparison...")
    
    # Create systems
    classical_system = create_electron_proton_neutron_system()
    quantum_system = create_electron_proton_neutron_quantum_system(grid_size=64, grid_range=2e-9)
    
    # Time points
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    # Run classical simulation
    print("  Running classical simulation...")
    classical_result = classical_system.simulate(t_span, t_eval)
    
    # Run quantum simulation
    print("  Running quantum simulation...")
    quantum_result = quantum_system.simulate(t_span, t_eval)
    
    # Compare
    comparator = SimulationComparison("Electron-Proton-Neutron System")
    comparison = comparator.compare_simulations(classical_result, quantum_result)
    comparator.print_comparison(comparison)
    
    return {
        'comparison': comparison,
        'classical_result': classical_result,
        'quantum_result': quantum_result
    }

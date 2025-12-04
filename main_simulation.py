"""
Main Simulation Runner
Runs classical and quantum simulations for both systems and compares results
"""

import numpy as np
import matplotlib.pyplot as plt
from comparison_framework import (
    run_three_electron_comparison,
    run_electron_proton_neutron_comparison
)
import sys


def plot_comparison_results(results_dict: dict, system_name: str, save_plots: bool = True):
    """Create visualization plots for comparison results"""
    comparison = results_dict['comparison']
    classical_result = results_dict['classical_result']
    quantum_result = results_dict['quantum_result']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{system_name} - Classical vs Quantum Comparison', fontsize=16)
    
    # Plot 1: Energy over time
    ax1 = axes[0, 0]
    if 'energy' in classical_result:
        ax1.plot(classical_result['time'], classical_result['energy'], 
                'b-', label='Classical', linewidth=2)
    if 'energy' in quantum_result:
        ax1.plot(quantum_result['time'], quantum_result['energy'], 
                'r--', label='Quantum', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Total Energy (J)')
    ax1.set_title('Energy Conservation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trajectories (position over time)
    ax2 = axes[0, 1]
    if 'positions' in classical_result:
        positions = classical_result['positions']
        # Plot center of mass or first particle
        if positions.ndim == 3:
            # Plot center of mass
            com = np.mean(positions, axis=1)
            ax2.plot(classical_result['time'], com[:, 0], 
                    'b-', label='Classical (COM)', linewidth=2)
    
    if 'positions' in quantum_result:
        positions_q = quantum_result['positions']
        ax2.plot(quantum_result['time'], positions_q, 
                'r--', label='Quantum (expectation)', linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Particle Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed comparison
    ax3 = axes[1, 0]
    methods = ['Classical', 'Quantum']
    times = [
        comparison['classical']['computation_time'],
        comparison['quantum']['computation_time']
    ]
    colors = ['blue', 'red']
    bars = ax3.bar(methods, times, color=colors, alpha=0.7)
    ax3.set_ylabel('Computation Time (s)')
    ax3.set_title('Speed Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom')
    
    # Plot 4: Energy conservation error
    ax4 = axes[1, 1]
    if 'energy_conservation' in comparison['classical'] and 'energy_conservation' in comparison['quantum']:
        methods = ['Classical', 'Quantum']
        errors = [
            comparison['classical']['energy_conservation']['mean_error'],
            comparison['quantum']['energy_conservation']['mean_error']
        ]
        colors = ['blue', 'red']
        bars = ax4.bar(methods, errors, color=colors, alpha=0.7)
        ax4.set_ylabel('Mean Relative Error')
        ax4.set_title('Energy Conservation Accuracy')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.2e}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{system_name.lower().replace(' ', '_')}_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {filename}")
    
    plt.close()


def print_summary(all_results: dict):
    """Print overall summary of all comparisons"""
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80 + "\n")
    
    for system_name, results in all_results.items():
        comp = results['comparison']
        print(f"{system_name}:")
        print(f"  Speed: {comp['comparison']['better_speed'].title()} is faster "
              f"(ratio: {comp['comparison']['speed_ratio']:.4f})")
        
        if 'better_energy_conservation' in comp['comparison']:
            print(f"  Energy Conservation: {comp['comparison']['better_energy_conservation'].title()} is more accurate")
        
        print()


def main():
    """Main simulation function"""
    print("="*80)
    print("CLASSICAL vs QUANTUM SIMULATION COMPARISON")
    print("="*80)
    print("\nThis simulation compares classical (Newtonian) and quantum (Schrödinger)")
    print("mechanics for two different systems:\n")
    print("1. 3-Electron System")
    print("2. Electron-Proton-Neutron System\n")
    print("="*80 + "\n")
    
    all_results = {}
    
    # Run 3-electron system comparison
    try:
        print("\n" + "="*80)
        print("SYSTEM 1: 3-ELECTRON SYSTEM")
        print("="*80)
        results_3e = run_three_electron_comparison(
            t_span=(0.0, 1e-15),  # 1 femtosecond
            n_points=50
        )
        all_results['3-Electron System'] = results_3e
        
        # Create plots
        try:
            plot_comparison_results(results_3e, "3-Electron System")
        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")
    
    except Exception as e:
        print(f"  Error in 3-electron simulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Run electron-proton-neutron system comparison
    try:
        print("\n" + "="*80)
        print("SYSTEM 2: ELECTRON-PROTON-NEUTRON SYSTEM")
        print("="*80)
        results_epn = run_electron_proton_neutron_comparison(
            t_span=(0.0, 1e-15),  # 1 femtosecond
            n_points=50
        )
        all_results['Electron-Proton-Neutron System'] = results_epn
        
        # Create plots
        try:
            plot_comparison_results(results_epn, "Electron-Proton-Neutron System")
        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")
    
    except Exception as e:
        print(f"  Error in electron-proton-neutron simulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Print overall summary
    if all_results:
        print_summary(all_results)
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()

"""
Main script to run classical vs quantum simulation comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from comparison import SimulationComparison
import json
import os


def plot_energy_comparison(results_3e, results_epn, save_dir='plots'):
    """Create visualization plots for energy comparisons."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: 3-electron system energy comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3-electron: Energy values
    ax = axes[0, 0]
    methods = ['Classical\n(Avg)', 'Quantum\n(HF)', 'Quantum\n(Var)']
    energies = [
        results_3e['classical']['average_kinetic'] + results_3e['classical']['average_potential'],
        results_3e['quantum']['hartree_fock']['ground_state_energy'],
        results_3e['quantum']['variational']['ground_state_energy']
    ]
    colors = ['blue', 'green', 'red']
    bars = ax.bar(methods, energies, color=colors, alpha=0.7)
    ax.set_ylabel('Energy (a.u.)')
    ax.set_title('3-Electron System: Energy Comparison')
    ax.grid(True, alpha=0.3)
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.4f}',
                ha='center', va='bottom')
    
    # 3-electron: Computation time
    ax = axes[0, 1]
    times = [
        results_3e['classical']['computation_time'],
        results_3e['quantum']['hartree_fock']['computation_time'],
        results_3e['quantum']['variational']['computation_time']
    ]
    bars = ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('3-Electron System: Speed Comparison')
    ax.grid(True, alpha=0.3)
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom')
    
    # Electron-proton-neutron: Energy values
    ax = axes[1, 0]
    methods_epn = ['Classical\n(Avg)', 'Quantum\n(H-like)', 'Analytical\n(H)']
    energies_epn = [
        results_epn['classical']['average_kinetic'] + results_epn['classical']['average_potential'],
        results_epn['quantum']['ground_state_energy'],
        results_epn['quantum']['analytical_energy']
    ]
    colors_epn = ['blue', 'green', 'orange']
    bars = ax.bar(methods_epn, energies_epn, color=colors_epn, alpha=0.7)
    ax.set_ylabel('Energy (a.u.)')
    ax.set_title('Electron-Proton-Neutron: Energy Comparison')
    ax.grid(True, alpha=0.3)
    for bar, energy in zip(bars, energies_epn):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.4f}',
                ha='center', va='bottom')
    
    # Electron-proton-neutron: Computation time
    ax = axes[1, 1]
    times_epn = [
        results_epn['classical']['computation_time'],
        results_epn['quantum']['computation_time'],
        results_epn['quantum']['full_3body']['computation_time']
    ]
    methods_time = ['Classical', 'Quantum\n(H-like)', 'Quantum\n(Full)']
    bars = ax.bar(methods_time, times_epn, color=colors_epn, alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Electron-Proton-Neutron: Speed Comparison')
    ax.grid(True, alpha=0.3)
    for bar, time_val in zip(bars, times_epn):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_dir}/comparison_plots.png")
    plt.close()


def save_results(results_3e, results_epn, filename='results.json'):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results = {
        'three_electron': convert_to_serializable(results_3e),
        'electron_proton_neutron': convert_to_serializable(results_epn)
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    """Main function to run all simulations and comparisons."""
    print("\n" + "=" * 80)
    print("CLASSICAL vs QUANTUM SIMULATION COMPARISON")
    print("=" * 80)
    print("\nThis script compares classical and quantum mechanical simulations")
    print("for two different systems:")
    print("  1. 3-electron system")
    print("  2. Electron-proton-neutron system")
    print("\nComparing: Speed and Accuracy")
    print("=" * 80)
    
    # Initialize comparison framework
    comparator = SimulationComparison()
    
    # Run comparisons
    print("\nStarting simulations...\n")
    
    # System 1: 3-electron
    results_3e = comparator.compare_three_electron_system(t_max=1.0, dt=0.01)
    
    # System 2: Electron-proton-neutron
    results_epn = comparator.compare_electron_proton_neutron_system(t_max=1.0, dt=0.01)
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)
    report = comparator.generate_report(results_3e, results_epn)
    print(report)
    
    # Save report to file
    with open('comparison_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to comparison_report.txt")
    
    # Save results to JSON
    save_results(results_3e, results_epn)
    
    # Generate plots
    try:
        plot_energy_comparison(results_3e, results_epn)
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")
        print("(This is okay - plots require matplotlib)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. SPEED:")
    print("   - Classical mechanics is typically faster for simple time evolution")
    print("   - Quantum mechanics requires iterative methods (slower but more accurate)")
    
    print("\n2. ACCURACY:")
    print("   - Quantum mechanics provides physically correct results")
    print("   - Classical mechanics violates quantum principles (Pauli exclusion, etc.)")
    print("   - Energy values differ significantly between methods")
    
    print("\n3. PHYSICAL CORRECTNESS:")
    print("   - For electrons and atomic systems: QUANTUM mechanics is essential")
    print("   - Classical mechanics fails to describe:")
    print("     * Stable atomic structure (predicts collapse)")
    print("     * Electron spin and antisymmetry")
    print("     * Discrete energy levels")
    print("     * Quantum tunneling and other quantum effects")
    
    print("\n4. RECOMMENDATION:")
    print("   - Use QUANTUM mechanics for atomic/molecular systems")
    print("   - Use CLASSICAL mechanics only when quantum effects are negligible")
    print("   - Speed vs accuracy trade-off: quantum is slower but physically correct")
    
    print("\n" + "=" * 80)
    print("Simulation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

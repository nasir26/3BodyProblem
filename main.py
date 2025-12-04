"""
Main script to run complete simulation comparison
Compares classical vs quantum simulations for two different systems
"""

import numpy as np
from comparison_framework import SimulationComparator
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


def main():
    """Run complete simulation comparison"""
    
    print("\n" + "="*80)
    print("CLASSICAL VS QUANTUM SIMULATION COMPARISON")
    print("="*80)
    print("\nThis simulation compares classical and quantum mechanics approaches")
    print("for two different atomic-scale systems:\n")
    print("1. 3-Electron System")
    print("2. 1 Electron + 1 Proton + 1 Neutron System\n")
    print("="*80 + "\n")
    
    # Initialize comparator
    comparator = SimulationComparator()
    
    # Simulation parameters
    # Time span: 1 femtosecond (1e-15 s) - appropriate for atomic dynamics
    t_span = (0.0, 1e-15)
    n_points = 100
    
    # Run comparison for 3-electron system
    print("\n" + "="*80)
    print("SYSTEM 1: 3-ELECTRON SYSTEM")
    print("="*80)
    comparator.run_comparison('3electron', t_span, n_points)
    
    # Run comparison for electron-proton-neutron system
    print("\n" + "="*80)
    print("SYSTEM 2: ELECTRON + PROTON + NEUTRON SYSTEM")
    print("="*80)
    comparator.run_comparison('electron_proton_neutron', t_span, n_points)
    
    # Generate and print report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80 + "\n")
    
    report = comparator.generate_report()
    print(report)
    
    # Save report to file
    with open('simulation_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to 'simulation_report.txt'")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot for 3-electron system
    comparator.plot_comparison('3electron', 'comparison_3electron.png')
    
    # Plot for electron-proton-neutron system
    comparator.plot_comparison('electron_proton_neutron', 
                               'comparison_electron_proton_neutron.png')
    
    # Create summary comparison plot
    create_summary_plot(comparator)
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print("\nAll results have been generated:")
    print("  - simulation_report.txt: Detailed text report")
    print("  - comparison_3electron.png: Visualization for 3-electron system")
    print("  - comparison_electron_proton_neutron.png: Visualization for electron-proton-neutron system")
    print("  - summary_comparison.png: Overall summary comparison")
    print("\n")


def create_summary_plot(comparator):
    """Create summary comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Summary: Classical vs Quantum Simulation Comparison', 
                 fontsize=16, fontweight='bold')
    
    systems = ['3electron', 'electron_proton_neutron']
    system_labels = ['3-Electron System', 'e⁻ + p⁺ + n System']
    
    # Speed comparison
    ax1 = axes[0, 0]
    classical_times = []
    quantum_times = []
    for sys in systems:
        comp = comparator.results[sys]['comparison']
        classical_times.append(comp['speed']['classical_time'])
        quantum_times.append(comp['speed']['quantum_time'])
    
    x = np.arange(len(system_labels))
    width = 0.35
    ax1.bar(x - width/2, classical_times, width, label='Classical', 
            color='blue', alpha=0.7)
    ax1.bar(x + width/2, quantum_times, width, label='Quantum', 
            color='red', alpha=0.7)
    ax1.set_ylabel('Computation Time (s)')
    ax1.set_title('Speed Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_labels, rotation=15, ha='right')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Energy conservation (classical)
    ax2 = axes[0, 1]
    energy_variations = []
    for sys in systems:
        comp = comparator.results[sys]['comparison']
        if 'energy_conservation_classical' in comp:
            energy_variations.append(
                comp['energy_conservation_classical']['relative_variation'] * 100
            )
        else:
            energy_variations.append(0)
    
    bars = ax2.bar(system_labels, energy_variations, color='blue', alpha=0.7)
    ax2.set_ylabel('Energy Variation (%)')
    ax2.set_title('Classical Energy Conservation')
    ax2.axhline(1.0, color='r', linestyle='--', label='1% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, var in zip(bars, energy_variations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.3f}%', ha='center', va='bottom')
    
    # Energy stability (quantum)
    ax3 = axes[1, 0]
    quantum_variations = []
    for sys in systems:
        comp = comparator.results[sys]['comparison']
        if 'energy_stability_quantum' in comp:
            quantum_variations.append(
                comp['energy_stability_quantum']['relative_variation'] * 100
            )
        else:
            quantum_variations.append(0)
    
    bars = ax3.bar(system_labels, quantum_variations, color='red', alpha=0.7)
    ax3.set_ylabel('Energy Variation (%)')
    ax3.set_title('Quantum Energy Stability')
    ax3.axhline(1.0, color='r', linestyle='--', label='1% threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, var in zip(bars, quantum_variations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.3f}%', ha='center', va='bottom')
    
    # Speedup ratio
    ax4 = axes[1, 1]
    speedups = []
    for sys in systems:
        comp = comparator.results[sys]['comparison']
        speedups.append(comp['speed']['speedup'])
    
    colors = ['green' if s > 1 else 'orange' for s in speedups]
    bars = ax4.bar(system_labels, speedups, color=colors, alpha=0.7)
    ax4.set_ylabel('Speedup Ratio (Classical/Quantum)')
    ax4.set_title('Relative Speed')
    ax4.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        label = f'{speedup:.2f}x'
        if speedup > 1:
            label += '\n(Classical faster)'
        else:
            label += '\n(Quantum faster)'
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('summary_comparison.png', dpi=300, bbox_inches='tight')
    print("Summary plot saved to 'summary_comparison.png'")
    plt.close()


if __name__ == '__main__':
    main()

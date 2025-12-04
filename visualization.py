"""
Visualization Module for Classical vs Quantum Simulations
==========================================================

This module provides visualization functions for comparing classical
and quantum simulations of atomic systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional
import os


def setup_style():
    """Configure matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_three_electron_comparison(comparison: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive visualization for three-electron system comparison.
    
    Args:
        comparison: Dictionary from compare_classical_quantum()
        save_path: Path to save figure (optional)
    """
    setup_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Classical trajectory (3D)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    traj = comparison['classical']['trajectory']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i in range(3):
        ax1.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], 
                color=colors[i], alpha=0.6, label=f'e⁻ {i+1}')
    ax1.set_xlabel('x (a₀)')
    ax1.set_ylabel('y (a₀)')
    ax1.set_zlabel('z (a₀)')
    ax1.set_title('Classical Electron Trajectories')
    ax1.legend(fontsize=8)
    
    # 2. Energy over time (classical)
    ax2 = fig.add_subplot(gs[0, 1])
    times = comparison['classical']['times']
    energies = comparison['classical']['energies']
    ax2.plot(times, energies, 'b-', alpha=0.8)
    ax2.axhline(y=np.mean(energies), color='r', linestyle='--', 
                label=f'Mean = {np.mean(energies):.3f} Ha')
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('Energy (Hartree)')
    ax2.set_title('Classical Energy Conservation')
    ax2.legend()
    
    # 3. Energy comparison bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    labels = ['Classical\nAverage', 'Hartree-Fock', 'VMC', 'Exact\n(Reference)']
    values = [
        comparison['energy_comparison']['classical_average'],
        comparison['energy_comparison']['quantum_hf'],
        comparison['energy_comparison']['quantum_vmc'],
        comparison['energy_comparison']['exact_reference']
    ]
    colors_bar = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
    bars = ax3.bar(labels, values, color=colors_bar, alpha=0.8)
    ax3.set_ylabel('Energy (Hartree)')
    ax3.set_title('Energy Comparison')
    ax3.axhline(y=values[-1], color='k', linestyle=':', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Computation time comparison
    ax4 = fig.add_subplot(gs[1, 0])
    methods = ['Classical MD', 'Quantum VMC']
    times_comp = [
        comparison['classical']['computation_time'],
        comparison['quantum']['computation_time']
    ]
    colors_time = ['#3498db', '#2ecc71']
    bars = ax4.bar(methods, times_comp, color=colors_time, alpha=0.8)
    ax4.set_ylabel('Computation Time (s)')
    ax4.set_title('Speed Comparison')
    for bar, t in zip(bars, times_comp):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{t:.3f}s', ha='center', va='bottom')
    
    # 5. Electron-electron distances (classical)
    ax5 = fig.add_subplot(gs[1, 1])
    traj = comparison['classical']['trajectory']
    r12 = np.linalg.norm(traj[:, 0] - traj[:, 1], axis=1)
    r13 = np.linalg.norm(traj[:, 0] - traj[:, 2], axis=1)
    r23 = np.linalg.norm(traj[:, 1] - traj[:, 2], axis=1)
    ax5.plot(times, r12, alpha=0.7, label='r₁₂')
    ax5.plot(times, r13, alpha=0.7, label='r₁₃')
    ax5.plot(times, r23, alpha=0.7, label='r₂₃')
    ax5.set_xlabel('Time (a.u.)')
    ax5.set_ylabel('Distance (a₀)')
    ax5.set_title('Electron-Electron Distances')
    ax5.legend()
    
    # 6. Error comparison
    ax6 = fig.add_subplot(gs[1, 2])
    exact = comparison['energy_comparison']['exact_reference']
    errors = [
        abs(comparison['energy_comparison']['classical_average'] - exact),
        abs(comparison['energy_comparison']['quantum_hf'] - exact),
        abs(comparison['energy_comparison']['quantum_vmc'] - exact)
    ]
    methods_err = ['Classical', 'Hartree-Fock', 'VMC']
    colors_err = ['#3498db', '#9b59b6', '#2ecc71']
    bars = ax6.bar(methods_err, errors, color=colors_err, alpha=0.8)
    ax6.set_ylabel('|Error| (Hartree)')
    ax6.set_title('Accuracy: Error vs Exact')
    for bar, e in zip(bars, errors):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{e:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 7-9. Summary text panels
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    THREE-ELECTRON SYSTEM (Li-like, Z=3) SIMULATION SUMMARY
    ═══════════════════════════════════════════════════════
    
    CLASSICAL MODEL (Point charges + Coulomb)                QUANTUM MODEL (Hartree-Fock + VMC)
    ─────────────────────────────────────────                ──────────────────────────────────────
    • Computation Time: {comparison['classical']['computation_time']:.4f} s                         • Computation Time: {comparison['quantum']['computation_time']:.4f} s
    • Energy: {comparison['energy_comparison']['classical_average']:.4f} Hartree                             • HF Energy: {comparison['energy_comparison']['quantum_hf']:.4f} Hartree
    • Energy Drift: {comparison['classical']['energy_drift']:.6f} Ha                        • VMC Energy: {comparison['energy_comparison']['quantum_vmc']:.4f} ± {comparison['quantum']['vmc_error']:.4f} Ha
    
    COMPARISON RESULTS
    ─────────────────────────────────────────────────────────────────────────────────────────────────
    • Speed: Classical is {comparison['speed_ratio']:.1f}x FASTER
    • Accuracy: Quantum (VMC) error = {abs(comparison['energy_comparison']['quantum_vmc'] - exact):.2f} Ha vs Classical error = {abs(comparison['energy_comparison']['classical_average'] - exact):.2f} Ha
    • Exact Reference Energy: {exact:.4f} Hartree (Experimental Li ground state)
    
    PHYSICS INSIGHTS
    ─────────────────────────────────────────────────────────────────────────────────────────────────
    • Classical model fails to capture exchange-correlation effects (Pauli exclusion)
    • Quantum antisymmetric wavefunctions (Slater determinants) essential for multi-electron systems
    • VMC includes electron correlation beyond mean-field Hartree-Fock
    """
    
    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, 
             fontsize=9, fontfamily='monospace', va='center', ha='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Classical vs Quantum: Three-Electron System', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_epn_comparison(comparison: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive visualization for electron-proton-neutron system.
    
    Args:
        comparison: Dictionary from compare_classical_quantum_epn()
        save_path: Path to save figure (optional)
    """
    setup_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Classical orbit trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    traj_e = comparison['classical']['trajectory']['electron']
    ax1.plot(traj_e[:, 0], traj_e[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    ax1.plot(0, 0, 'ro', markersize=10, label='Nucleus (p+n)')
    ax1.set_xlabel('x (a₀)')
    ax1.set_ylabel('y (a₀)')
    ax1.set_title('Classical Electron Orbit')
    ax1.axis('equal')
    ax1.legend()
    
    # 2. Classical energy conservation
    ax2 = fig.add_subplot(gs[0, 1])
    times = comparison['classical']['times']
    energies = comparison['classical']['energies']
    ax2.plot(times, energies, 'b-', alpha=0.8)
    ax2.axhline(y=comparison['classical']['bohr_predictions']['energy'], 
                color='r', linestyle='--', label='Bohr prediction')
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('Energy (Hartree)')
    ax2.set_title('Classical Energy (Bohr Model)')
    ax2.legend()
    
    # 3. Orbital radius over time
    ax3 = fig.add_subplot(gs[0, 2])
    radii = comparison['classical']['radii']
    ax3.plot(times, radii, 'b-', alpha=0.6)
    ax3.axhline(y=comparison['classical']['bohr_predictions']['radius'],
                color='r', linestyle='--', label='Bohr radius')
    ax3.set_xlabel('Time (a.u.)')
    ax3.set_ylabel('Radius (a₀)')
    ax3.set_title('Electron-Nucleus Distance')
    ax3.legend()
    
    # 4. Quantum energy levels
    ax4 = fig.add_subplot(gs[1, 0])
    n_levels = [1, 2, 3]
    energies_q = [comparison['quantum']['energy_levels'][n]['exact'] for n in n_levels]
    energies_eV = [comparison['quantum']['energy_levels'][n]['exact_eV'] for n in n_levels]
    
    for n, E, E_eV in zip(n_levels, energies_q, energies_eV):
        ax4.hlines(E, 0, 1, colors='b', linewidth=3)
        ax4.text(1.05, E, f'n={n}: {E:.3f} Ha\n({E_eV:.2f} eV)', va='center')
    
    ax4.set_xlim(-0.2, 1.8)
    ax4.set_ylim(min(energies_q) - 0.1, 0.1)
    ax4.set_ylabel('Energy (Hartree)')
    ax4.set_title('Quantum Energy Levels')
    ax4.set_xticks([])
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, label='Ionization')
    
    # 5. Radial probability density
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        from epn_system import QuantumEPNSystem
        quantum = QuantumEPNSystem()
        r = np.linspace(0.01, 10, 200)
        for n in [1, 2, 3]:
            P = quantum.probability_density_radial(r, n, 0)
            ax5.plot(r, P, label=f'n={n}')
    except Exception:
        # Fallback: use analytical formula directly
        r = np.linspace(0.01, 10, 200)
        for n, color in [(1, 'blue'), (2, 'orange'), (3, 'green')]:
            # Simplified radial probability for hydrogen
            if n == 1:
                P = 4 * r**2 * np.exp(-2*r)
            elif n == 2:
                P = r**2 * (2 - r)**2 * np.exp(-r) / 8
            else:
                P = r**2 * (1 - 2*r/3 + 2*r**2/27)**2 * np.exp(-2*r/3) * 4/729
            ax5.plot(r, P, label=f'n={n}', color=color)
    ax5.set_xlabel('r (a₀)')
    ax5.set_ylabel('P(r) = r²|R(r)|²')
    ax5.set_title('Radial Probability Density')
    ax5.legend()
    ax5.axvline(x=1.5, color='k', linestyle=':', alpha=0.5, label='⟨r⟩₁ₛ')
    
    # 6. Classical vs Quantum comparison
    ax6 = fig.add_subplot(gs[1, 2])
    categories = ['Energy\n(Ha)', 'Radius\n(a₀)', 'Time\n(ms)']
    classical_vals = [
        comparison['classical']['average_energy'],
        comparison['classical']['average_radius'],
        comparison['classical']['computation_time'] * 1000
    ]
    quantum_vals = [
        comparison['quantum']['energy_levels'][1]['exact'],
        comparison['quantum']['expectation_values'][1]['r_mean'],
        comparison['quantum']['computation_time'] * 1000
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, classical_vals, width, label='Classical', color='#3498db', alpha=0.8)
    ax6.bar(x + width/2, quantum_vals, width, label='Quantum', color='#2ecc71', alpha=0.8)
    ax6.set_ylabel('Value')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.set_title('Classical vs Quantum Comparison')
    ax6.legend()
    
    # 7. Uncertainty principle visualization
    ax7 = fig.add_subplot(gs[2, 0])
    unc = comparison['quantum']['uncertainty']
    bars = ax7.bar(['Δx·Δp', 'ℏ/2 (limit)'], 
                   [unc['uncertainty_product'], unc['heisenberg_limit']],
                   color=['#2ecc71', '#e74c3c'], alpha=0.8)
    ax7.set_ylabel('Uncertainty Product (a.u.)')
    ax7.set_title('Heisenberg Uncertainty Principle')
    ax7.text(0, unc['uncertainty_product'] + 0.05, 
             f"{unc['uncertainty_product']:.3f}", ha='center')
    ax7.text(1, unc['heisenberg_limit'] + 0.05,
             f"{unc['heisenberg_limit']:.3f}", ha='center')
    
    # 8. Time evolution
    ax8 = fig.add_subplot(gs[2, 1])
    t_evol = comparison['quantum']['time_evolution']
    ax8.plot(t_evol['times'], t_evol['r_expectation'], 'b-')
    ax8.set_xlabel('Time (a.u.)')
    ax8.set_ylabel('⟨r⟩ (a₀)')
    ax8.set_title(f"Wavepacket Oscillation\n(Period = {t_evol['oscillation_period']:.2f} a.u.)")
    
    # 9. Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    E_exact = comparison['energy_comparison']['quantum_exact']
    E_classical = comparison['energy_comparison']['classical']
    error_classical = comparison['energy_comparison']['error_classical']
    error_var = comparison['energy_comparison']['error_variational']
    
    summary = f"""
    ELECTRON-PROTON-NEUTRON SYSTEM
    (Hydrogen-like atom with deuteron nucleus)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CLASSICAL (Bohr Model):
    • Energy: {E_classical:.6f} Ha
    • Error: {error_classical:.6f} Ha
    
    QUANTUM (Exact):
    • E₁ = {E_exact:.6f} Ha
    • = {E_exact * 27.21:.4f} eV
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    KEY INSIGHT:
    Classical Bohr model gives
    EXACT energy for hydrogen!
    (But wrong physics: no
    uncertainty, no probability)
    """
    
    ax9.text(0.5, 0.5, summary, transform=ax9.transAxes,
             fontsize=10, fontfamily='monospace', va='center', ha='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Classical vs Quantum: Electron-Proton-Neutron System',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_combined_summary(comparison_3e: Dict, comparison_epn: Dict, 
                          save_path: Optional[str] = None):
    """
    Create combined summary figure comparing both systems.
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # System 1: Three electrons
    ax1, ax2, ax3 = axes[0]
    
    # 3e: Energy comparison
    labels = ['Classical', 'Quantum\n(VMC)', 'Exact']
    values = [
        comparison_3e['energy_comparison']['classical_average'],
        comparison_3e['energy_comparison']['quantum_vmc'],
        comparison_3e['energy_comparison']['exact_reference']
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax1.bar(labels, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title('3-Electron System: Energy')
    ax1.axhline(y=values[-1], color='k', linestyle=':', alpha=0.5)
    
    # 3e: Error comparison
    exact_3e = comparison_3e['energy_comparison']['exact_reference']
    errors_3e = [
        abs(comparison_3e['energy_comparison']['classical_average'] - exact_3e),
        abs(comparison_3e['energy_comparison']['quantum_vmc'] - exact_3e)
    ]
    ax2.bar(['Classical', 'Quantum'], errors_3e, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax2.set_ylabel('|Error| (Hartree)')
    ax2.set_title('3-Electron System: Accuracy')
    
    # 3e: Speed comparison
    times_3e = [
        comparison_3e['classical']['computation_time'],
        comparison_3e['quantum']['computation_time']
    ]
    ax3.bar(['Classical', 'Quantum'], times_3e, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('3-Electron System: Speed')
    
    # System 2: EPN
    ax4, ax5, ax6 = axes[1]
    
    # EPN: Energy comparison
    labels = ['Classical\n(Bohr)', 'Quantum\n(Exact)']
    values_epn = [
        comparison_epn['energy_comparison']['classical'],
        comparison_epn['energy_comparison']['quantum_exact']
    ]
    ax4.bar(labels, values_epn, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax4.set_ylabel('Energy (Hartree)')
    ax4.set_title('E-P-N System: Energy')
    
    # EPN: Error comparison
    exact_epn = comparison_epn['energy_comparison']['quantum_exact']
    errors_epn = [
        comparison_epn['energy_comparison']['error_classical'],
        comparison_epn['energy_comparison']['error_variational']
    ]
    ax5.bar(['Classical', 'Variational'], errors_epn, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax5.set_ylabel('|Error| (Hartree)')
    ax5.set_title('E-P-N System: Accuracy')
    
    # EPN: Speed comparison
    times_epn = [
        comparison_epn['classical']['computation_time'],
        comparison_epn['quantum']['computation_time']
    ]
    ax6.bar(['Classical', 'Quantum'], times_epn, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax6.set_ylabel('Time (seconds)')
    ax6.set_title('E-P-N System: Speed')
    
    plt.suptitle('Classical vs Quantum Simulation Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def generate_all_figures(comparison_3e: Dict, comparison_epn: Dict, 
                         output_dir: str = '.'):
    """Generate all visualization figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualization figures...")
    
    # Three-electron system
    fig1_path = os.path.join(output_dir, 'three_electron_comparison.png')
    plot_three_electron_comparison(comparison_3e, save_path=fig1_path)
    
    # EPN system
    fig2_path = os.path.join(output_dir, 'epn_comparison.png')
    plot_epn_comparison(comparison_epn, save_path=fig2_path)
    
    # Combined summary
    fig3_path = os.path.join(output_dir, 'combined_summary.png')
    plot_combined_summary(comparison_3e, comparison_epn, save_path=fig3_path)
    
    print(f"\nAll figures saved to {output_dir}/")
    
    return [fig1_path, fig2_path, fig3_path]

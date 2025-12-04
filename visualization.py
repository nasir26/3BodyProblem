"""
Visualization tools for simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict
import json


def plot_trajectories_classical(trajectory: List[Dict], title: str = "Classical Trajectories"):
    """Plot classical particle trajectories"""
    if not trajectory:
        print("No trajectory data to plot")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    n_particles = len(trajectory[0]['positions'])
    colors = plt.cm.tab10(np.linspace(0, 1, n_particles))
    
    for i in range(n_particles):
        positions = np.array([state['positions'][i] for state in trajectory])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=colors[i], label=f'Particle {i+1}', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                  color=colors[i], s=100, marker='o', label=f'Start {i+1}')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                  color=colors[i], s=100, marker='s', label=f'End {i+1}')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_energy_evolution(trajectory: List[Dict], title: str = "Energy Evolution"):
    """Plot energy evolution over time"""
    if not trajectory:
        print("No trajectory data to plot")
        return
    
    times = [state['time'] for state in trajectory]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Kinetic, Potential, Total
    if 'energies' in trajectory[0]:
        kinetic = [e['kinetic'] for e in [s['energies'] for s in trajectory]]
        potential = [e['potential'] for e in [s['energies'] for s in trajectory]]
        total = [e['total'] for e in [s['energies'] for s in trajectory]]
        
        axes[0].plot(times, kinetic, label='Kinetic', linewidth=2)
        axes[0].plot(times, potential, label='Potential', linewidth=2)
        axes[0].plot(times, total, label='Total', linewidth=2, linestyle='--')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Energy (J)')
        axes[0].set_title('Energy Components')
        axes[0].legend()
        axes[0].grid(True)
    
    # Energy conservation (total energy should be constant)
    if 'energies' in trajectory[0]:
        total_energy = [e['total'] for e in [s['energies'] for s in trajectory]]
        energy_variation = np.array(total_energy) - total_energy[0]
        axes[1].plot(times, energy_variation, linewidth=2, color='red')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Energy Variation (J)')
        axes[1].set_title('Energy Conservation')
        axes[1].grid(True)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_quantum_density(trajectory: List[Dict], title: str = "Quantum Probability Density"):
    """Plot quantum probability density"""
    if not trajectory or 'density' not in trajectory[0]:
        print("No quantum density data to plot")
        return
    
    # Plot final state
    final_state = trajectory[-1]
    density = final_state['density']
    
    # Take 2D slice through center
    n = density.shape[0]
    center = n // 2
    density_slice = density[center, :, :]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D contour plot
    im1 = axes[0].contourf(density_slice, levels=20, cmap='viridis')
    axes[0].set_title('Probability Density (2D Slice)')
    axes[0].set_xlabel('Y index')
    axes[0].set_ylabel('Z index')
    plt.colorbar(im1, ax=axes[0])
    
    # 1D radial profile
    # Compute radial average
    n = density.shape[0]
    center_idx = n // 2
    r_max = int(n * 0.7)
    radial_profile = []
    radii = []
    
    for r in range(r_max):
        count = 0
        total = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    dist = np.sqrt((i - center_idx)**2 + (j - center_idx)**2 + (k - center_idx)**2)
                    if abs(dist - r) < 0.5:
                        total += density[i, j, k]
                        count += 1
        if count > 0:
            radial_profile.append(total / count)
            radii.append(r)
    
    if radial_profile:
        axes[1].plot(radii, radial_profile, linewidth=2)
        axes[1].set_xlabel('Radial Distance (grid units)')
        axes[1].set_ylabel('Probability Density')
        axes[1].set_title('Radial Probability Distribution')
        axes[1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_comparison_benchmark(results_file: str = 'benchmark_results.json'):
    """Plot comparison of classical vs quantum benchmarks"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Results file {results_file} not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Speed comparison
    if '3electron' in results:
        r = results['3electron']
        classical_times = r['classical']['times']
        quantum_times = r['quantum']['times']
        
        if classical_times and quantum_times:
            axes[0, 0].bar(['Classical', 'Quantum'], 
                          [np.mean(classical_times), np.mean(quantum_times)],
                          yerr=[np.std(classical_times), np.std(quantum_times)],
                          capsize=5, color=['blue', 'red'], alpha=0.7)
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_title('3-Electron System: Speed Comparison')
            axes[0, 0].grid(True, alpha=0.3)
    
    if 'hydrogen_like' in results:
        r = results['hydrogen_like']
        classical_times = r['classical']['times']
        quantum_times = r['quantum']['times']
        
        if classical_times and quantum_times:
            axes[0, 1].bar(['Classical', 'Quantum'],
                          [np.mean(classical_times), np.mean(quantum_times)],
                          yerr=[np.std(classical_times), np.std(quantum_times)],
                          capsize=5, color=['blue', 'red'], alpha=0.7)
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].set_title('Hydrogen-like System: Speed Comparison')
            axes[0, 1].grid(True, alpha=0.3)
    
    # Energy comparison (for hydrogen-like system)
    if 'hydrogen_like' in results:
        r = results['hydrogen_like']
        if r['quantum']['energies'] and 'analytical' in r:
            quantum_energy = np.mean([e for e in r['quantum']['energies'] if e is not None])
            analytical_energy = r['analytical']['ground_state_energy']
            
            axes[1, 0].bar(['Quantum', 'Analytical'],
                          [quantum_energy, analytical_energy],
                          color=['red', 'green'], alpha=0.7)
            axes[1, 0].set_ylabel('Energy (Hartrees)')
            axes[1, 0].set_title('Hydrogen-like: Energy Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
            
            error = abs(quantum_energy - analytical_energy) / abs(analytical_energy) * 100
            axes[1, 0].text(0.5, 0.95, f'Error: {error:.2f}%',
                           transform=axes[1, 0].transAxes,
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Summary
    axes[1, 1].axis('off')
    summary_text = "Summary:\n\n"
    summary_text += "• Classical: Faster for simple systems\n"
    summary_text += "• Quantum: Required for atomic accuracy\n"
    summary_text += "• Speed depends on system complexity\n"
    summary_text += "• Quantum captures bound states correctly"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Classical vs Quantum Simulation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Example usage
    plot_comparison_benchmark()
    plt.show()

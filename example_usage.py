"""
Example usage script demonstrating how to use the simulation framework
"""

import numpy as np
from classical_simulation import (
    ClassicalSimulator, create_3electron_system,
    create_electron_proton_neutron_system
)
from quantum_simulation import QuantumSimulator
from benchmark_comparison import BenchmarkSuite

def example_classical_3electron():
    """Example: Run classical simulation for 3-electron system"""
    print("Example 1: Classical 3-Electron System")
    print("-" * 50)
    
    # Create system
    particles = create_3electron_system(initial_separation=1e-10)
    simulator = ClassicalSimulator(particles, dt=1e-18)
    
    # Run simulation
    print("Running simulation...")
    elapsed = simulator.simulate(total_time=1e-15, save_interval=100)
    
    print(f"Simulation completed in {elapsed:.4f} seconds")
    print(f"Number of time steps saved: {len(simulator.trajectory)}")
    
    # Display final energy
    if simulator.trajectory:
        final_energy = simulator.trajectory[-1]['energies']
        print(f"\nFinal Energies:")
        print(f"  Kinetic:   {final_energy['kinetic']:.6e} J")
        print(f"  Potential: {final_energy['potential']:.6e} J")
        print(f"  Total:     {final_energy['total']:.6e} J")
    
    return simulator


def example_quantum_hydrogen():
    """Example: Run quantum simulation for hydrogen-like system"""
    print("\nExample 2: Quantum Hydrogen-like System")
    print("-" * 50)
    
    # Create quantum simulator
    qsim = QuantumSimulator(grid_size=32, grid_range=10.0)
    
    # Solve for ground state
    print("Solving for ground state...")
    energies, wavefunctions = qsim.solve_stationary_hydrogen(n_states=3)
    
    print(f"\nEnergy Levels (Hartrees):")
    for i, E in enumerate(energies):
        print(f"  n={i+1}: {E:.6f}")
    
    # Time evolution
    if wavefunctions:
        print("\nRunning time evolution...")
        psi0 = wavefunctions[0]
        elapsed = qsim.simulate_time_evolution(
            psi0, total_time=1.0, dt=0.01, system_type='hydrogen'
        )
        
        print(f"Time evolution completed in {elapsed:.4f} seconds")
        
        if qsim.trajectory:
            final_state = qsim.trajectory[-1]
            print(f"\nFinal State:")
            print(f"  Energy: {final_state['energy']:.6f} Hartrees")
            print(f"  Position: {final_state['position']}")
            print(f"  Norm: {final_state['norm']:.6f}")
    
    return qsim


def example_benchmark():
    """Example: Run full benchmark comparison"""
    print("\nExample 3: Full Benchmark Comparison")
    print("-" * 50)
    
    benchmark = BenchmarkSuite()
    
    # Run benchmarks (with shorter times for quick demo)
    print("Running hydrogen-like benchmark...")
    benchmark.benchmark_hydrogen_like_system(
        simulation_time=1e-16,  # Shorter for demo
        n_runs=2
    )
    
    # Generate report
    report = benchmark.generate_comparison_report()
    print("\n" + report)
    
    return benchmark


if __name__ == '__main__':
    print("=" * 60)
    print("SIMULATION FRAMEWORK - EXAMPLE USAGE")
    print("=" * 60)
    
    try:
        # Run examples
        classical_sim = example_classical_3electron()
        quantum_sim = example_quantum_hydrogen()
        benchmark = example_benchmark()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

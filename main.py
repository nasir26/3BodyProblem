"""
Main execution script for comparing classical and quantum simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from benchmark_comparison import BenchmarkSuite
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Compare Classical vs Quantum Simulations')
    parser.add_argument('--system', type=str, choices=['both', '3electron', 'hydrogen'],
                       default='both', help='Which system to simulate')
    parser.add_argument('--time', type=float, default=1e-15,
                       help='Simulation time in seconds')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of benchmark runs')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CLASSICAL vs QUANTUM SIMULATION COMPARISON")
    print("=" * 80)
    print("\nThis simulation compares classical and quantum approaches for:")
    print("  1. Three-electron system")
    print("  2. One electron + one proton + one neutron system")
    print("\nMetrics: Speed and Accuracy")
    print("=" * 80 + "\n")
    
    benchmark = BenchmarkSuite()
    
    # Run benchmarks
    if args.system in ['both', '3electron']:
        try:
            benchmark.benchmark_3electron_system(
                simulation_time=args.time,
                n_runs=args.runs
            )
        except Exception as e:
            print(f"Error in 3-electron benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    if args.system in ['both', 'hydrogen']:
        try:
            benchmark.benchmark_hydrogen_like_system(
                simulation_time=args.time,
                n_runs=args.runs
            )
        except Exception as e:
            print(f"Error in hydrogen-like benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate and display report
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORT")
    print("=" * 80 + "\n")
    
    report = benchmark.generate_comparison_report()
    print(report)
    
    # Save report
    with open('comparison_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to comparison_report.txt")
    
    # Save results if requested
    if args.save:
        benchmark.save_results('benchmark_results.json')
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

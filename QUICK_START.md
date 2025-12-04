# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running Simulations

### Option 1: Full Comparison (Recommended)

Run the complete benchmark comparison:
```bash
python main.py
```

This will:
- Simulate both systems (3-electron and hydrogen-like)
- Compare classical vs quantum approaches
- Generate a detailed report in `comparison_report.txt`

### Option 2: Quick Demo

Run the example script for a quick demonstration:
```bash
python example_usage.py
```

### Option 3: Custom Parameters

```bash
# Run only hydrogen-like system with custom time
python main.py --system hydrogen --time 1e-16 --runs 5

# Save detailed results
python main.py --save
```

## Understanding the Output

### Speed Comparison
- Shows execution time for each method
- Classical is typically faster for simple systems
- Quantum requires more computational resources

### Accuracy Comparison
- For hydrogen-like system, compares with exact analytical solution
- Quantum should match analytical result closely
- Classical cannot capture bound states

### Key Metrics
- **Execution Time**: How long each simulation takes
- **Energy Accuracy**: How close to analytical solution
- **Energy Conservation**: How well energy is conserved (classical)

## Visualization

Generate plots:
```python
from visualization import plot_comparison_benchmark
import matplotlib.pyplot as plt

plot_comparison_benchmark('benchmark_results.json')
plt.show()
```

## Expected Results

### Hydrogen-like System
- **Quantum Energy**: ~-0.5 Hartrees (matches analytical)
- **Classical**: Cannot form stable bound state
- **Speed**: Classical ~3x faster, but inaccurate

### 3-Electron System
- **Complexity**: Requires many-body quantum treatment
- **Approximation**: Uses simplified mean-field method
- **Note**: Full treatment computationally expensive

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install numpy scipy matplotlib
```

### Memory Issues
Reduce grid size in quantum simulation:
```python
qsim = QuantumSimulator(grid_size=32, grid_range=5.0)  # Smaller grid
```

### Slow Performance
- Reduce simulation time: `--time 1e-16`
- Reduce number of runs: `--runs 1`
- Use smaller grid for quantum: `grid_size=32`

## Next Steps

1. Read `comparison_report.txt` for detailed analysis
2. Explore `visualization.py` for plotting options
3. Modify parameters in source files for custom studies
4. Extend to more complex systems

"""
Quick test script to verify simulations work correctly
"""

import numpy as np

def test_imports():
    """Test that all modules can be imported"""
    try:
        from classical_simulation import (
            ClassicalParticle, ClassicalSystem,
            create_three_electron_system,
            create_electron_proton_neutron_system
        )
        print("✓ Classical simulation imports successful")
        
        from quantum_simulation import (
            QuantumSystem,
            create_three_electron_quantum_system,
            create_electron_proton_neutron_quantum_system
        )
        print("✓ Quantum simulation imports successful")
        
        from comparison_framework import (
            SimulationComparison,
            run_three_electron_comparison,
            run_electron_proton_neutron_comparison
        )
        print("✓ Comparison framework imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_classical_system():
    """Test classical system creation and basic operations"""
    try:
        from classical_simulation import create_three_electron_system
        
        system = create_three_electron_system()
        assert len(system.particles) == 3
        assert all(p.charge == -1.0 for p in system.particles)
        print("✓ Classical 3-electron system creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Classical system test error: {e}")
        return False


def test_quantum_system():
    """Test quantum system creation"""
    try:
        from quantum_simulation import create_three_electron_quantum_system
        
        system = create_three_electron_quantum_system(grid_size=32)
        assert system.n_particles == 3
        assert len(system.masses) == 3
        print("✓ Quantum 3-electron system creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Quantum system test error: {e}")
        return False


if __name__ == "__main__":
    print("Running simulation tests...\n")
    
    all_passed = True
    all_passed &= test_imports()
    print()
    all_passed &= test_classical_system()
    all_passed &= test_quantum_system()
    
    print("\n" + "="*50)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed ✗")
    print("="*50)

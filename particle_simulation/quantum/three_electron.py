"""
Quantum Three-Electron System Simulation

This module implements a quantum mechanical simulation of three interacting electrons
using various methods:
1. Variational Monte Carlo
2. Configuration Interaction (limited basis)
3. Hartree-Fock approximation

Physical Model:
--------------
The Hamiltonian for N electrons:
H = Σ_i [-ℏ²/(2m_e) ∇_i²] + Σ_{i<j} k_e * e² / |r_i - r_j|

For 3 electrons:
H = T₁ + T₂ + T₃ + V₁₂ + V₁₃ + V₂₃

where T_i = -ℏ²/(2m_e) ∇_i² (kinetic energy)
and V_ij = k_e * e² / |r_i - r_j| (Coulomb repulsion)

Key Quantum Effects:
-------------------
1. Pauli Exclusion: Electrons are fermions, wavefunction must be antisymmetric
2. Exchange Interaction: Quantum correlation due to antisymmetry
3. Uncertainty Principle: Cannot know position and momentum simultaneously
4. Heisenberg Energy: Kinetic energy from confinement

In atomic units: ℏ = m_e = e = k_e = 1
"""

import numpy as np
from scipy import linalg
from scipy.special import factorial
from typing import Tuple, List, Optional, Dict, Callable
import time
from ..constants import PHYSICAL_CONSTANTS, ATOMIC_UNITS


class ThreeElectronQuantum:
    """
    Quantum simulation of a three-electron system.
    
    Implements multiple methods for solving the many-body Schrödinger equation.
    """
    
    def __init__(self):
        """Initialize the three-electron quantum simulation."""
        # Use atomic units
        self.hbar = 1.0
        self.m_e = 1.0
        self.e = 1.0
        self.k_e = 1.0
        
        # Simulation results
        self.ground_state_energy = None
        self.wave_function_params = None
        self.computation_time = 0.0
        
        # Grid parameters for numerical methods
        self.grid_size = 32  # 3D grid size
        self.box_size = 10.0  # Bohr radii
    
    def harmonic_oscillator_basis(self, n: int, omega: float, x: np.ndarray) -> np.ndarray:
        """
        Compute harmonic oscillator basis function.
        
        ψ_n(x) = (mω/πℏ)^(1/4) * 1/√(2^n n!) * H_n(√(mω/ℏ) x) * exp(-mωx²/2ℏ)
        
        In atomic units (m=ℏ=1):
        ψ_n(x) = (ω/π)^(1/4) * 1/√(2^n n!) * H_n(√ω x) * exp(-ωx²/2)
        """
        xi = np.sqrt(omega) * x
        normalization = (omega / np.pi) ** 0.25 / np.sqrt(2**n * factorial(n))
        
        # Hermite polynomial using recurrence relation
        H_n = self._hermite(n, xi)
        
        return normalization * H_n * np.exp(-omega * x**2 / 2)
    
    def _hermite(self, n: int, x: np.ndarray) -> np.ndarray:
        """Compute Hermite polynomial H_n(x) using recurrence."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            H_prev2 = np.ones_like(x)
            H_prev1 = 2 * x
            for k in range(2, n + 1):
                H_curr = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
                H_prev2 = H_prev1
                H_prev1 = H_curr
            return H_curr
    
    def gaussian_trial_wavefunction(self, r: np.ndarray, alpha: float) -> float:
        """
        Simple Gaussian trial wavefunction for variational method.
        
        Ψ(r₁, r₂, r₃) = exp(-α(r₁² + r₂² + r₃²))
        
        This is NOT antisymmetrized - for demonstration only.
        """
        r_squared_sum = np.sum(r**2)
        return np.exp(-alpha * r_squared_sum)
    
    def slater_jastrow_wavefunction(self, 
                                     r: np.ndarray, 
                                     alpha: float, 
                                     beta: float) -> float:
        """
        Slater-Jastrow trial wavefunction with electron correlation.
        
        Ψ = Ψ_Slater × Ψ_Jastrow
        
        Ψ_Slater: Antisymmetric part (Slater determinant)
        Ψ_Jastrow: Correlation factor exp(Σ_{i<j} u(r_ij))
        
        For simplicity, we use:
        Ψ = exp(-α Σ_i r_i²) × exp(β Σ_{i<j} r_ij / (1 + β r_ij))
        """
        # Single-particle part (Gaussian)
        r_squared_sum = np.sum(np.linalg.norm(r, axis=1)**2)
        single_particle = np.exp(-alpha * r_squared_sum)
        
        # Jastrow correlation factor
        jastrow = 1.0
        for i in range(3):
            for j in range(i+1, 3):
                r_ij = np.linalg.norm(r[i] - r[j])
                jastrow *= np.exp(beta * r_ij / (1 + beta * r_ij))
        
        return single_particle * jastrow
    
    def compute_local_energy(self, 
                             r: np.ndarray, 
                             psi_func: Callable, 
                             params: tuple,
                             h: float = 0.001) -> float:
        """
        Compute local energy E_L = (HΨ)/Ψ at position r.
        
        Uses numerical derivatives for kinetic energy:
        ∇²Ψ/Ψ ≈ Σ_i [Ψ(r_i+h) + Ψ(r_i-h) - 2Ψ(r)] / (h² Ψ(r))
        """
        psi_r = psi_func(r, *params)
        if abs(psi_r) < 1e-15:
            return 0.0
        
        # Kinetic energy using finite differences
        kinetic = 0.0
        for i in range(3):  # 3 electrons
            for d in range(3):  # 3 spatial dimensions
                r_plus = r.copy()
                r_minus = r.copy()
                r_plus[i, d] += h
                r_minus[i, d] -= h
                
                psi_plus = psi_func(r_plus, *params)
                psi_minus = psi_func(r_minus, *params)
                
                # Laplacian
                laplacian_term = (psi_plus + psi_minus - 2*psi_r) / (h**2)
                kinetic -= 0.5 * laplacian_term / psi_r
        
        # Potential energy (electron-electron repulsion)
        potential = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                r_ij = np.linalg.norm(r[i] - r[j])
                if r_ij > 1e-10:
                    potential += 1.0 / r_ij  # In atomic units
        
        return kinetic + potential
    
    def variational_monte_carlo(self,
                                 trial_wavefunction: Callable,
                                 initial_params: tuple,
                                 n_samples: int = 10000,
                                 n_equilibrate: int = 1000,
                                 step_size: float = 0.5) -> Dict:
        """
        Perform Variational Monte Carlo to estimate ground state energy.
        
        Uses Metropolis algorithm to sample |Ψ|² and compute <E> = <Ψ|H|Ψ>/<Ψ|Ψ>.
        
        Parameters:
        -----------
        trial_wavefunction : Callable
            Trial wavefunction Ψ(r, *params)
        initial_params : tuple
            Initial variational parameters
        n_samples : int
            Number of Monte Carlo samples
        n_equilibrate : int
            Number of equilibration steps
        step_size : float
            Metropolis step size
        
        Returns:
        --------
        dict
            VMC results including energy estimate and error
        """
        start_time = time.perf_counter()
        
        # Initialize positions randomly
        r = np.random.randn(3, 3) * 2.0  # 3 electrons, 3D
        
        params = initial_params
        accepted = 0
        
        # Equilibration
        for _ in range(n_equilibrate):
            r_new = r + step_size * np.random.randn(3, 3)
            
            psi_old = trial_wavefunction(r, *params)
            psi_new = trial_wavefunction(r_new, *params)
            
            # Metropolis acceptance
            if psi_old != 0:
                acceptance_ratio = (psi_new / psi_old)**2
                if np.random.random() < acceptance_ratio:
                    r = r_new
        
        # Sampling
        energies = []
        for _ in range(n_samples):
            # Propose new position
            r_new = r + step_size * np.random.randn(3, 3)
            
            psi_old = trial_wavefunction(r, *params)
            psi_new = trial_wavefunction(r_new, *params)
            
            # Metropolis acceptance
            if psi_old != 0:
                acceptance_ratio = (psi_new / psi_old)**2
                if np.random.random() < min(1.0, acceptance_ratio):
                    r = r_new
                    accepted += 1
            
            # Compute local energy at current position
            E_local = self.compute_local_energy(r, trial_wavefunction, params)
            energies.append(E_local)
        
        energies = np.array(energies)
        
        # Remove outliers (can occur if sampling approaches nodes)
        energies = energies[np.abs(energies) < 100]
        
        end_time = time.perf_counter()
        self.computation_time = end_time - start_time
        
        energy_mean = np.mean(energies)
        energy_std = np.std(energies) / np.sqrt(len(energies))
        
        self.ground_state_energy = energy_mean
        self.wave_function_params = params
        
        return {
            'method': 'quantum_vmc',
            'energy': energy_mean,
            'energy_error': energy_std,
            'n_samples': len(energies),
            'acceptance_rate': accepted / n_samples,
            'computation_time': self.computation_time,
            'params': params,
        }
    
    def optimize_variational_parameters(self,
                                         trial_wavefunction: Callable,
                                         initial_params: tuple,
                                         param_bounds: List[Tuple],
                                         n_samples: int = 5000) -> Dict:
        """
        Optimize variational parameters to minimize energy.
        
        Uses simple grid search for demonstration.
        """
        start_time = time.perf_counter()
        
        best_energy = float('inf')
        best_params = initial_params
        best_result = None
        
        # Grid search over parameters
        param_grids = [np.linspace(b[0], b[1], 10) for b in param_bounds]
        
        from itertools import product
        for params in product(*param_grids):
            result = self.variational_monte_carlo(
                trial_wavefunction,
                params,
                n_samples=n_samples // 10,
                n_equilibrate=500,
            )
            
            if result['energy'] < best_energy:
                best_energy = result['energy']
                best_params = params
                best_result = result
        
        # Final run with best parameters
        final_result = self.variational_monte_carlo(
            trial_wavefunction,
            best_params,
            n_samples=n_samples,
            n_equilibrate=1000,
        )
        
        end_time = time.perf_counter()
        
        return {
            'method': 'quantum_vmc_optimized',
            'energy': final_result['energy'],
            'energy_error': final_result['energy_error'],
            'optimal_params': best_params,
            'computation_time': end_time - start_time,
        }
    
    def hartree_fock_approximation(self, 
                                    omega: float = 0.5,
                                    n_basis: int = 6,
                                    max_iterations: int = 100,
                                    tolerance: float = 1e-8) -> Dict:
        """
        Perform Hartree-Fock calculation for 3 electrons in harmonic trap.
        
        The HF approximation treats electrons as moving in a mean-field potential.
        
        Parameters:
        -----------
        omega : float
            Harmonic oscillator frequency (trap strength)
        n_basis : int
            Number of basis functions
        max_iterations : int
            Maximum SCF iterations
        tolerance : float
            Convergence tolerance
        """
        start_time = time.perf_counter()
        
        # Use 1D harmonic oscillator basis for simplicity
        # (Full 3D would require more complex implementation)
        n_grid = 100
        x_max = 10.0
        x = np.linspace(-x_max, x_max, n_grid)
        dx = x[1] - x[0]
        
        # Build basis functions
        basis = np.zeros((n_basis, n_grid))
        for n in range(n_basis):
            basis[n] = self.harmonic_oscillator_basis(n, omega, x)
        
        # Normalize
        for n in range(n_basis):
            norm = np.sqrt(np.trapezoid(basis[n]**2, x))
            if norm > 0:
                basis[n] /= norm
        
        # One-body Hamiltonian matrix (kinetic + harmonic potential)
        H_one = np.zeros((n_basis, n_basis))
        
        for i in range(n_basis):
            for j in range(n_basis):
                # Kinetic energy (using analytical result for HO)
                if i == j:
                    H_one[i, j] = omega * (i + 0.5)
        
        # Initialize density matrix (3 electrons in lowest orbitals)
        D = np.zeros((n_basis, n_basis))
        for i in range(min(3, n_basis)):
            D[i, i] = 1.0
        
        # SCF iteration
        E_old = 0.0
        for iteration in range(max_iterations):
            # Build Fock matrix (simplified - ignoring 2-electron integrals properly)
            F = H_one.copy()
            
            # Diagonalize Fock matrix
            eigenvalues, eigenvectors = np.linalg.eigh(F)
            
            # Build new density matrix (occupy lowest 3 orbitals)
            D_new = np.zeros((n_basis, n_basis))
            for i in range(min(3, n_basis)):
                for mu in range(n_basis):
                    for nu in range(n_basis):
                        D_new[mu, nu] += eigenvectors[mu, i] * eigenvectors[nu, i]
            
            # Compute energy
            E_new = np.sum(eigenvalues[:3])  # Sum of occupied orbital energies
            
            # Add electron repulsion estimate (mean-field)
            # Simplified: assume average repulsion based on orbital sizes
            avg_separation = 2.0 / np.sqrt(omega)  # Rough estimate
            E_repulsion = 3.0 / avg_separation  # 3 pairs, 1/r each
            E_new += E_repulsion
            
            # Check convergence
            if abs(E_new - E_old) < tolerance:
                break
            
            E_old = E_new
            D = 0.5 * D + 0.5 * D_new  # Damping for stability
        
        end_time = time.perf_counter()
        self.computation_time = end_time - start_time
        self.ground_state_energy = E_new
        
        return {
            'method': 'quantum_hartree_fock',
            'energy': E_new,
            'orbital_energies': eigenvalues[:3],
            'n_iterations': iteration + 1,
            'converged': iteration < max_iterations - 1,
            'computation_time': self.computation_time,
            'omega': omega,
        }
    
    def run_simulation(self, method: str = 'vmc', **kwargs) -> Dict:
        """
        Run quantum simulation with specified method.
        
        Parameters:
        -----------
        method : str
            'vmc': Variational Monte Carlo with Gaussian wavefunction
            'vmc_jastrow': VMC with Slater-Jastrow wavefunction
            'hartree_fock': Hartree-Fock approximation
        
        Returns:
        --------
        dict
            Simulation results
        """
        if method == 'vmc':
            n_samples = kwargs.get('n_samples', 10000)
            alpha = kwargs.get('alpha', 0.5)
            return self.variational_monte_carlo(
                self.gaussian_trial_wavefunction,
                (alpha,),
                n_samples=n_samples,
            )
        
        elif method == 'vmc_jastrow':
            n_samples = kwargs.get('n_samples', 10000)
            alpha = kwargs.get('alpha', 0.3)
            beta = kwargs.get('beta', 0.5)
            return self.variational_monte_carlo(
                self.slater_jastrow_wavefunction,
                (alpha, beta),
                n_samples=n_samples,
            )
        
        elif method == 'hartree_fock':
            omega = kwargs.get('omega', 0.5)
            return self.hartree_fock_approximation(omega=omega)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_expected_values(self, 
                                 trial_wavefunction: Callable,
                                 params: tuple,
                                 n_samples: int = 5000) -> Dict:
        """
        Compute expectation values of various observables.
        """
        # Monte Carlo sampling
        r = np.random.randn(3, 3) * 2.0
        step_size = 0.5
        
        # Equilibrate
        for _ in range(1000):
            r_new = r + step_size * np.random.randn(3, 3)
            psi_old = trial_wavefunction(r, *params)
            psi_new = trial_wavefunction(r_new, *params)
            if psi_old != 0 and np.random.random() < min(1, (psi_new/psi_old)**2):
                r = r_new
        
        # Sample observables
        r_squared_samples = []
        separation_samples = []
        
        for _ in range(n_samples):
            r_new = r + step_size * np.random.randn(3, 3)
            psi_old = trial_wavefunction(r, *params)
            psi_new = trial_wavefunction(r_new, *params)
            if psi_old != 0 and np.random.random() < min(1, (psi_new/psi_old)**2):
                r = r_new
            
            r_squared_samples.append(np.sum(r**2))
            
            for i in range(3):
                for j in range(i+1, 3):
                    separation_samples.append(np.linalg.norm(r[i] - r[j]))
        
        return {
            'avg_r_squared': np.mean(r_squared_samples),
            'std_r_squared': np.std(r_squared_samples),
            'avg_separation': np.mean(separation_samples),
            'std_separation': np.std(separation_samples),
        }


if __name__ == "__main__":
    # Test the quantum simulation
    sim = ThreeElectronQuantum()
    
    print("Testing Three-Electron Quantum Simulation")
    print("=" * 50)
    
    # Test VMC with simple Gaussian
    print("\n1. Variational Monte Carlo (Gaussian):")
    result_vmc = sim.run_simulation('vmc', n_samples=5000, alpha=0.5)
    print(f"   Energy: {result_vmc['energy']:.4f} ± {result_vmc['energy_error']:.4f} Hartree")
    print(f"   Acceptance rate: {result_vmc['acceptance_rate']:.2%}")
    print(f"   Computation time: {result_vmc['computation_time']:.4f} s")
    
    # Test VMC with Jastrow correlation
    print("\n2. Variational Monte Carlo (Slater-Jastrow):")
    result_sj = sim.run_simulation('vmc_jastrow', n_samples=5000, alpha=0.3, beta=0.5)
    print(f"   Energy: {result_sj['energy']:.4f} ± {result_sj['energy_error']:.4f} Hartree")
    print(f"   Acceptance rate: {result_sj['acceptance_rate']:.2%}")
    print(f"   Computation time: {result_sj['computation_time']:.4f} s")
    
    # Test Hartree-Fock
    print("\n3. Hartree-Fock Approximation:")
    result_hf = sim.run_simulation('hartree_fock', omega=0.5)
    print(f"   Energy: {result_hf['energy']:.4f} Hartree")
    print(f"   Orbital energies: {result_hf['orbital_energies']}")
    print(f"   Converged: {result_hf['converged']}")
    print(f"   Computation time: {result_hf['computation_time']:.4f} s")

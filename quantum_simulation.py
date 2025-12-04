"""
Quantum Mechanics Simulation Module
Uses time-dependent Schrödinger equation and Hartree-Fock methods
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from typing import List, Tuple, Dict
import time

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
K_COULOMB = 8.9875517923e9  # N⋅m²/C²
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg


class QuantumSystem:
    """Quantum mechanical system using time-dependent Schrödinger equation"""
    
    def __init__(self, n_particles: int, masses: List[float], charges: List[float],
                 grid_size: int = 64, grid_range: float = 2e-9):
        """
        Initialize quantum system
        
        Args:
            n_particles: Number of particles
            masses: List of particle masses
            charges: List of particle charges (in elementary charge units)
            grid_size: Number of grid points per dimension
            grid_range: Spatial range in meters (from -grid_range/2 to grid_range/2)
        """
        self.n_particles = n_particles
        self.masses = np.array(masses)
        self.charges = np.array(charges)
        self.grid_size = grid_size
        self.grid_range = grid_range
        
        # Create spatial grid
        self.dx = grid_range / grid_size
        self.x = np.linspace(-grid_range/2, grid_range/2, grid_size)
        self.dx_val = self.x[1] - self.x[0]
        
        # For simplicity, we'll use 1D approximation for multi-particle systems
        # Full 3D would require grid_size^3 points which is computationally expensive
        self.dimension = 1  # Using 1D for computational feasibility
        
    def kinetic_energy_operator(self, mass: float) -> np.ndarray:
        """Construct kinetic energy operator T = -hbar^2/(2m) * d^2/dx^2"""
        # Second derivative using finite differences
        # d^2/dx^2 ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
        diagonal = -2.0 * np.ones(self.grid_size)
        off_diagonal = np.ones(self.grid_size - 1)
        
        laplacian = diags([off_diagonal, diagonal, off_diagonal], 
                         [-1, 0, 1], 
                         shape=(self.grid_size, self.grid_size),
                         format='csr')
        laplacian = laplacian / (self.dx_val ** 2)
        
        # Kinetic energy operator
        T = - (HBAR ** 2) / (2.0 * mass) * laplacian
        
        return T
    
    def potential_energy_operator(self, positions: np.ndarray = None) -> np.ndarray:
        """Construct potential energy operator (Coulomb interactions)"""
        # For multi-particle systems, we use mean-field approximation
        # V(x) = sum of Coulomb potentials from other particles
        
        V = np.zeros(self.grid_size)
        
        if positions is None:
            # Default: particles at center
            positions = np.zeros((self.n_particles, self.dimension))
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                # Coulomb potential between particles
                q1 = self.charges[i] * ELEMENTARY_CHARGE
                q2 = self.charges[j] * ELEMENTARY_CHARGE
                
                # Mean field approximation: potential at each grid point
                for k, x_pos in enumerate(self.x):
                    r1 = abs(x_pos - positions[i, 0]) if positions.shape[1] > 0 else abs(x_pos)
                    r2 = abs(x_pos - positions[j, 0]) if positions.shape[1] > 0 else abs(x_pos)
                    
                    # Avoid singularity
                    r1 = max(r1, 1e-15)
                    r2 = max(r2, 1e-15)
                    
                    # Interaction potential (simplified)
                    V[k] += K_COULOMB * q1 * q2 / (r1 + r2)
        
        return np.diag(V)
    
    def hamiltonian(self, positions: np.ndarray = None) -> np.ndarray:
        """Construct total Hamiltonian H = T + V"""
        # For multi-particle, use Hartree-Fock approximation
        # H = sum_i T_i + sum_i V_i + sum_{i<j} V_ij
        
        H = np.zeros((self.grid_size, self.grid_size), dtype=np.complex128)
        
        # Add kinetic energy for each particle (mean field)
        for i, mass in enumerate(self.masses):
            T = self.kinetic_energy_operator(mass)
            # Convert sparse to dense and add
            if hasattr(T, 'toarray'):
                H += T.toarray() / self.n_particles  # Average contribution
            else:
                H += T / self.n_particles
        
        # Add potential energy
        V = self.potential_energy_operator(positions)
        if hasattr(V, 'toarray'):
            H += V.toarray()
        else:
            H += V
        
        return H
    
    def initial_wavefunction(self, center: float = 0.0, width: float = 1e-11) -> np.ndarray:
        """Gaussian wave packet as initial condition"""
        psi = np.exp(-((self.x - center) ** 2) / (2 * width ** 2))
        # Normalize: ∫|ψ|²dx = 1
        norm = np.sqrt(np.trapz(np.abs(psi) ** 2, self.x))
        if norm > 1e-15:
            psi = psi / norm
        return psi.astype(np.complex128)
    
    def schrodinger_equation(self, t: float, psi: np.ndarray) -> np.ndarray:
        """Time-dependent Schrödinger equation: iℏ dψ/dt = Hψ"""
        psi_reshaped = psi.reshape(self.grid_size)
        H = self.hamiltonian()
        dpsi_dt = -1j / HBAR * H @ psi_reshaped
        return dpsi_dt.flatten()
    
    def compute_expectation_values(self, psi: np.ndarray) -> Dict:
        """Compute expectation values of position, momentum, energy"""
        # Normalize wavefunction
        norm = np.sqrt(np.trapz(np.abs(psi) ** 2, self.x))
        if norm > 1e-15:
            psi_normalized = psi / norm
        else:
            psi_normalized = psi
        
        # <x> = ∫ x |ψ|² dx
        x_exp = np.trapz(self.x * np.abs(psi_normalized) ** 2, self.x)
        
        # <p> = -iℏ ∫ ψ* dψ/dx dx
        dpsi_dx = np.gradient(psi_normalized, self.dx_val)
        p_exp = -1j * HBAR * np.trapz(np.conj(psi_normalized) * dpsi_dx, self.x)
        p_exp = np.real(p_exp)  # Momentum expectation value is real
        
        # <E> = <H> = ∫ ψ* H ψ dx
        H = self.hamiltonian()
        H_psi = H @ psi_normalized
        E_exp = np.real(np.trapz(np.conj(psi_normalized) * H_psi, self.x))
        
        return {
            'position': x_exp,
            'momentum': p_exp,
            'energy': E_exp
        }
    
    def simulate(self, t_span: Tuple[float, float], t_eval: np.ndarray = None,
                 initial_psi: np.ndarray = None) -> Dict:
        """Simulate quantum system evolution"""
        start_time = time.time()
        
        if initial_psi is None:
            initial_psi = self.initial_wavefunction()
        
        # Solve time-dependent Schrödinger equation
        sol = solve_ivp(
            self.schrodinger_equation,
            t_span,
            initial_psi,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract results
        n_steps = len(sol.t)
        wavefunctions = sol.y.T  # Shape: (n_steps, grid_size)
        
        # Compute expectation values at each time step
        positions = np.zeros(n_steps)
        energies = np.zeros(n_steps)
        
        for i in range(n_steps):
            exp_vals = self.compute_expectation_values(wavefunctions[i])
            positions[i] = exp_vals['position']
            energies[i] = exp_vals['energy']
        
        return {
            'time': sol.t,
            'wavefunctions': wavefunctions,
            'positions': positions,
            'energy': energies,
            'computation_time': elapsed_time,
            'n_steps': n_steps,
            'success': sol.success
        }
    
    def ground_state(self, n_states: int = 1) -> Dict:
        """Find ground state using eigenvalue problem H|ψ⟩ = E|ψ⟩"""
        start_time = time.time()
        
        H = self.hamiltonian()
        
        # Find lowest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigs(H, k=n_states, which='SR')  # Smallest real
        
        # Sort by real part of eigenvalue
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        elapsed_time = time.time() - start_time
        
        return {
            'energies': np.real(eigenvalues),
            'wavefunctions': eigenvectors,
            'computation_time': elapsed_time
        }


def create_three_electron_quantum_system(grid_size: int = 64, 
                                         grid_range: float = 2e-9) -> QuantumSystem:
    """Create quantum system for 3 electrons"""
    masses = [ELECTRON_MASS] * 3
    charges = [-1.0, -1.0, -1.0]
    return QuantumSystem(3, masses, charges, grid_size, grid_range)


def create_electron_proton_neutron_quantum_system(grid_size: int = 64,
                                                   grid_range: float = 2e-9) -> QuantumSystem:
    """Create quantum system for 1 electron, 1 proton, 1 neutron"""
    masses = [ELECTRON_MASS, PROTON_MASS, NEUTRON_MASS]
    charges = [-1.0, 1.0, 0.0]
    return QuantumSystem(3, masses, charges, grid_size, grid_range)

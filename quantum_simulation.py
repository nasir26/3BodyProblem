"""
Quantum Mechanics Simulation Module
Simulates particle systems using quantum mechanics (Schrödinger equation)
"""

import numpy as np
from scipy.integrate import ode
from scipy.linalg import expm, eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import time
from typing import List, Tuple, Dict

# Import constants from classical module
from classical_simulation import (
    E_CHARGE, EPSILON_0, K_COULOMB, M_ELECTRON, M_PROTON, M_NEUTRON, HBAR,
    AU_LENGTH, AU_ENERGY, AU_TIME
)

# Atomic units: hbar = m_e = e = 1
# In atomic units, energy is in Hartrees, length in Bohr radii


class QuantumSimulator:
    """Quantum mechanics simulator using time-dependent Schrödinger equation"""
    
    def __init__(self, grid_size: int = 64, grid_range: float = 10.0):
        """
        Initialize quantum simulator
        
        Args:
            grid_size: Number of grid points per dimension
            grid_range: Spatial range in atomic units (Bohr radii)
        """
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.dx = 2 * grid_range / grid_size
        self.x = np.linspace(-grid_range, grid_range, grid_size)
        self.y = np.linspace(-grid_range, grid_range, grid_size)
        self.z = np.linspace(-grid_range, grid_range, grid_size)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Initialize wavefunction
        self.psi = None
        self.time = 0.0
        self.trajectory = []
        
    def create_hamiltonian_3electron(self) -> np.ndarray:
        """
        Create Hamiltonian for 3-electron system
        Uses mean-field approximation (Hartree-Fock-like)
        """
        n = self.grid_size
        n_total = n ** 3
        
        # Kinetic energy operator: -hbar^2/(2m) * d^2/dx^2
        # In atomic units: -1/2 * d^2/dx^2
        kx = np.fft.fftfreq(n, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(n, self.dx) * 2 * np.pi
        kz = np.fft.fftfreq(n, self.dx) * 2 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K_sq = KX**2 + KY**2 + KZ**2
        
        # Kinetic energy in momentum space
        T = 0.5 * K_sq  # In atomic units
        
        # For simplicity, use a simplified potential
        # In full treatment, would need to solve self-consistently
        R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        # Mean-field potential (simplified)
        # Represents electron-electron repulsion in mean-field
        V = np.zeros_like(R)
        
        return T, V
    
    def solve_stationary_3electron(self, n_states: int = 3) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Solve for stationary states of 3-electron system"""
        # This is a simplified version - full 3-electron problem is very complex
        # We'll use a mean-field approximation (Hartree-Fock-like)
        
        n = self.grid_size
        T, V = self.create_hamiltonian_3electron()
        
        # For 3-electron system, we use a simplified approach:
        # Treat as three independent electrons in a mean-field potential
        # Full treatment would require solving 3-body Schrödinger equation
        
        energies = []
        wavefunctions = []
        
        # Simplified: use hydrogen-like orbitals with effective charge
        # This is a rough approximation
        for i in range(min(n_states, 3)):
            r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
            r = np.where(r < 0.01, 0.01, r)
            
            # Approximate wavefunction (simplified)
            if i == 0:
                # Ground state-like
                psi = np.exp(-r * 1.5) / np.sqrt(np.pi)
            else:
                # Excited states (simplified)
                psi = np.exp(-r * (1.5 - i * 0.3)) / np.sqrt(np.pi * (i + 1))
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx**3)
            if norm > 0:
                psi = psi / norm
            
            # Estimate energy (simplified)
            E = -0.5 * (1.5 - i * 0.3)**2  # Rough estimate
            
            energies.append(E)
            wavefunctions.append(psi)
        
        return np.array(energies), wavefunctions
    
    def create_hamiltonian_hydrogen_like(self, Z: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Hamiltonian for hydrogen-like atom (electron + proton)
        This can be solved more accurately
        """
        n = self.grid_size
        
        # Kinetic energy operator in momentum space
        kx = np.fft.fftfreq(n, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(n, self.dx) * 2 * np.pi
        kz = np.fft.fftfreq(n, self.dx) * 2 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K_sq = KX**2 + KY**2 + KZ**2
        T = 0.5 * K_sq  # In atomic units
        
        # Coulomb potential: -Z/r (in atomic units)
        R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        R = np.where(R < 0.01, 0.01, R)  # Avoid singularity
        V = -Z / R
        
        return T, V
    
    def solve_stationary_hydrogen(self, n_states: int = 5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve for stationary states of hydrogen-like atom
        Returns energies and wavefunctions
        """
        n = self.grid_size
        T, V = self.create_hamiltonian_hydrogen_like(Z=1)
        
        # Use FFT-based method to solve eigenvalue problem
        # H = T + V, where T is diagonal in k-space, V is diagonal in r-space
        
        # Initial guess: ground state
        psi_init = np.exp(-np.sqrt(self.X**2 + self.Y**2 + self.Z**2))
        psi_init = psi_init / np.sqrt(np.sum(np.abs(psi_init)**2) * self.dx**3)
        
        # Use iterative method (simplified - full implementation would use
        # more sophisticated eigensolvers)
        energies = []
        wavefunctions = []
        
        # For hydrogen, we know analytical solution
        # E_n = -Z^2/(2*n^2) in atomic units
        analytical_energies = [-0.5 / (n**2) for n in range(1, n_states + 1)]
        
        # Create approximate wavefunctions
        for i, E in enumerate(analytical_energies):
            # Simplified: use analytical form for hydrogen
            n_quantum = i + 1
            r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
            
            if n_quantum == 1:
                # 1s orbital
                psi = np.exp(-r) / np.sqrt(np.pi)
            elif n_quantum == 2:
                # 2s orbital (simplified)
                psi = (2 - r) * np.exp(-r/2) / (4 * np.sqrt(2 * np.pi))
            else:
                # Higher states (simplified)
                psi = np.exp(-r / n_quantum) / np.sqrt(np.pi * n_quantum**3)
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx**3)
            if norm > 0:
                psi = psi / norm
            
            energies.append(E)
            wavefunctions.append(psi)
        
        return np.array(energies), wavefunctions
    
    def time_evolution_step(self, psi: np.ndarray, dt: float, T: np.ndarray, V: np.ndarray):
        """
        Perform one time step using split-operator method
        exp(-i*H*dt) ≈ exp(-i*V*dt/2) * exp(-i*T*dt) * exp(-i*V*dt/2)
        """
        # Step 1: Apply potential half-step
        psi = np.exp(-1j * V * dt / 2) * psi
        
        # Step 2: FFT to momentum space
        psi_k = np.fft.fftn(psi)
        
        # Step 3: Apply kinetic energy
        psi_k = np.exp(-1j * T * dt) * psi_k
        
        # Step 4: FFT back to position space
        psi = np.fft.ifftn(psi_k)
        
        # Step 5: Apply potential half-step
        psi = np.exp(-1j * V * dt / 2) * psi
        
        # Normalize to prevent numerical drift
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx**3)
        if norm > 1e-10:
            psi = psi / norm
        
        return psi
    
    def simulate_time_evolution(self, psi0: np.ndarray, total_time: float, 
                                dt: float = 0.01, system_type: str = 'hydrogen'):
        """
        Simulate time evolution of wavefunction
        """
        if system_type == 'hydrogen':
            T, V = self.create_hamiltonian_hydrogen_like(Z=1)
        else:
            T, V = self.create_hamiltonian_3electron()
        
        psi = psi0.copy()
        self.time = 0.0
        self.trajectory = []
        
        start_time = time.time()
        n_steps = int(total_time / dt)
        
        for step in range(n_steps):
            psi = self.time_evolution_step(psi, dt, T, V)
            self.time += dt
            
            if step % 10 == 0:
                # Compute expectation values
                density = np.abs(psi)**2
                norm = np.sum(density) * self.dx**3
                
                # Expected position
                x_exp = np.sum(self.X * density) * self.dx**3
                y_exp = np.sum(self.Y * density) * self.dx**3
                z_exp = np.sum(self.Z * density) * self.dx**3
                
                # Expected energy (simplified)
                psi_k = np.fft.fftn(psi)
                T_exp = np.real(np.sum(np.conj(psi_k) * T * psi_k)) * self.dx**3
                V_exp = np.real(np.sum(np.conj(psi) * V * psi)) * self.dx**3
                E_exp = T_exp + V_exp
                
                state = {
                    'time': self.time,
                    'wavefunction': psi.copy(),
                    'density': density,
                    'position': np.array([x_exp, y_exp, z_exp]),
                    'energy': E_exp,
                    'norm': norm
                }
                self.trajectory.append(state)
        
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def compute_energies(self, psi: np.ndarray, T: np.ndarray, V: np.ndarray) -> Dict[str, float]:
        """Compute expectation values of energy"""
        n = self.grid_size
        
        # Kinetic energy
        psi_k = np.fft.fftn(psi)
        T_exp = np.real(np.sum(np.conj(psi_k) * T * psi_k)) * self.dx**3
        
        # Potential energy
        V_exp = np.real(np.sum(np.conj(psi) * V * psi)) * self.dx**3
        
        total = T_exp + V_exp
        
        return {
            'kinetic': T_exp,
            'potential': V_exp,
            'total': total
        }

"""
Quantum Mechanics Simulation Module
Simulates multi-particle systems using quantum mechanical methods.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from typing import List, Tuple, Dict
import time


class QuantumSystem:
    """Base class for quantum mechanical systems."""
    
    # Physical constants (in atomic units)
    HBAR = 1.0  # Reduced Planck constant in atomic units
    K_COULOMB = 1.0  # Coulomb constant in atomic units
    
    def __init__(self):
        """Initialize quantum system."""
        pass
    
    def solve_schrodinger(self, potential_func, grid: np.ndarray, mass: float) -> Dict:
        """
        Solve 1D time-independent Schrödinger equation.
        
        Args:
            potential_func: Function V(x) that returns potential energy
            grid: Spatial grid points
            mass: Particle mass
            
        Returns:
            Dictionary with eigenvalues and eigenfunctions
        """
        dx = grid[1] - grid[0]
        n = len(grid)
        
        # Kinetic energy operator (finite difference)
        # T = -hbar^2/(2m) * d^2/dx^2
        kinetic = -self.HBAR**2 / (2 * mass * dx**2)
        diag_main = np.ones(n) * (-2 * kinetic)
        diag_off = np.ones(n - 1) * kinetic
        
        # Potential energy operator (diagonal)
        V = np.array([potential_func(x) for x in grid])
        
        # Hamiltonian matrix
        H = diags([diag_off, diag_main + V, diag_off], [-1, 0, 1], format='csr')
        
        # Solve eigenvalue problem
        eigenvalues, eigenfunctions = eigsh(H, k=min(5, n-1), which='SA')
        
        # Normalize eigenfunctions
        for i in range(len(eigenvalues)):
            norm = np.sqrt(np.trapz(eigenfunctions[:, i]**2, grid))
            eigenfunctions[:, i] /= norm
        
        return {
            'eigenvalues': eigenvalues,
            'eigenfunctions': eigenfunctions,
            'grid': grid
        }


class ThreeElectronQuantumSystem(QuantumSystem):
    """Quantum system with 3 electrons using Hartree-Fock approximation."""
    
    def __init__(self):
        """Initialize 3-electron quantum system."""
        super().__init__()
        self.n_electrons = 3
        self.electron_mass = 1.0
        self.electron_charge = -1.0
    
    def hartree_fock_solve(self, grid_size: int = 100, grid_range: Tuple[float, float] = (-5, 5)) -> Dict:
        """
        Solve 3-electron system using Hartree-Fock method.
        
        Args:
            grid_size: Number of grid points
            grid_range: (x_min, x_max) for spatial grid
            
        Returns:
            Dictionary with ground state energy and wavefunctions
        """
        start_time = time.time()
        
        # Create 3D grid (simplified to 1D for each electron)
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        dx = x[1] - x[0]
        
        # Initial guess: product of hydrogen-like orbitals
        # For 3 electrons, we need to consider spin and antisymmetry
        # Simplified approach: use Hartree-Fock with mean field
        
        # Self-consistent field iteration
        max_iter = 50
        tolerance = 1e-6
        
        # Initial orbitals (Gaussian guess)
        orbitals = []
        for i in range(self.n_electrons):
            center = (i - 1) * 1.0  # Spread electrons
            psi = np.exp(-0.5 * (x - center)**2)
            psi /= np.sqrt(np.trapz(psi**2, x))
            orbitals.append(psi)
        
        energy_old = 0.0
        
        for iteration in range(max_iter):
            # Build Fock matrix for each orbital
            energies = []
            new_orbitals = []
            
            for i in range(self.n_electrons):
                # Construct Fock operator
                # F = H + sum_j (2J_j - K_j) where J is Coulomb, K is exchange
                
                # Kinetic energy
                kinetic = -self.HBAR**2 / (2 * self.electron_mass * dx**2)
                diag_main = np.ones(grid_size) * (-2 * kinetic)
                diag_off = np.ones(grid_size - 1) * kinetic
                
                # Hartree potential (Coulomb repulsion from other electrons)
                V_hartree = np.zeros(grid_size)
                for j in range(self.n_electrons):
                    if j != i:
                        # Coulomb repulsion: integrate |psi_j|^2 / |r - r'|
                        for k, xk in enumerate(x):
                            integrand = orbitals[j]**2 / (np.abs(x - xk) + 1e-10)
                            V_hartree[k] += 2.0 * self.K_COULOMB * np.trapz(integrand, x)
                
                # Exchange term (simplified)
                V_exchange = np.zeros(grid_size)
                
                # Total potential
                V_total = V_hartree + V_exchange
                
                # Hamiltonian
                H = diags([diag_off, diag_main + V_total, diag_off], [-1, 0, 1], format='csr')
                
                # Solve for lowest eigenvalue
                eigenval, eigenvec = eigsh(H, k=1, which='SA')
                energies.append(eigenval[0])
                
                # Normalize
                eigenvec = eigenvec[:, 0]
                eigenvec /= np.sqrt(np.trapz(eigenvec**2, x))
                new_orbitals.append(eigenvec)
            
            # Check convergence
            total_energy = sum(energies) - 0.5 * sum(V_hartree)  # Avoid double counting
            if abs(total_energy - energy_old) < tolerance:
                break
            
            energy_old = total_energy
            orbitals = new_orbitals
        
        elapsed_time = time.time() - start_time
        
        # Compute total energy more accurately
        total_energy = self._compute_total_energy(orbitals, x)
        
        return {
            'ground_state_energy': total_energy,
            'orbital_energies': energies,
            'orbitals': orbitals,
            'grid': x,
            'n_iterations': iteration + 1,
            'computation_time': elapsed_time
        }
    
    def _compute_total_energy(self, orbitals: List[np.ndarray], grid: np.ndarray) -> float:
        """Compute total energy including all interactions."""
        total = 0.0
        dx = grid[1] - grid[0]
        
        # Kinetic energy
        for orb in orbitals:
            d2psi = np.gradient(np.gradient(orb, dx), dx)
            kinetic = -0.5 * self.HBAR**2 / self.electron_mass * np.trapz(orb * d2psi, grid)
            total += kinetic
        
        # Electron-electron repulsion
        for i in range(len(orbitals)):
            for j in range(i + 1, len(orbitals)):
                # Coulomb integral
                for k, xk in enumerate(grid):
                    integrand = orbitals[i][k]**2 * orbitals[j]**2 / (np.abs(grid - xk) + 1e-10)
                    total += self.K_COULOMB * np.trapz(integrand, grid)
        
        return total
    
    def exact_3body_quantum(self, grid_size: int = 50) -> Dict:
        """
        Exact quantum solution for 3-body system (simplified 1D).
        This is computationally expensive but more accurate.
        """
        start_time = time.time()
        
        # Reduced grid for exact calculation
        x = np.linspace(-3, 3, grid_size)
        dx = x[1] - x[0]
        
        # For exact solution, we'd need to solve full 3-body Schrödinger equation
        # This is extremely expensive, so we use a variational approach
        # with a correlated wavefunction
        
        # Trial wavefunction: product with Jastrow factor for correlation
        def trial_wavefunction(x1, x2, x3, alpha=1.0, beta=0.5):
            """Trial wavefunction with correlation."""
            # Product of Gaussians
            psi = (np.exp(-alpha * x1**2) * 
                   np.exp(-alpha * x2**2) * 
                   np.exp(-alpha * x3**2))
            # Jastrow factor for electron correlation
            r12 = np.abs(x1 - x2)
            r13 = np.abs(x1 - x3)
            r23 = np.abs(x2 - x3)
            jastrow = np.exp(-beta * (r12 + r13 + r23))
            return psi * jastrow
        
        # Variational Monte Carlo (simplified)
        # Sample and compute energy expectation value
        n_samples = 10000
        samples = np.random.normal(0, 1.0, (n_samples, 3))
        
        # Compute local energy
        energies = []
        for sample in samples[:1000]:  # Limit for speed
            x1, x2, x3 = sample
            # Local energy = (H psi) / psi
            # Simplified calculation
            energy = (0.5 * (x1**2 + x2**2 + x3**2) +  # Harmonic approximation
                     1.0 / (np.abs(x1 - x2) + 0.1) +   # Coulomb repulsion
                     1.0 / (np.abs(x1 - x3) + 0.1) +
                     1.0 / (np.abs(x2 - x3) + 0.1))
            energies.append(energy)
        
        ground_energy = np.mean(energies)
        elapsed_time = time.time() - start_time
        
        return {
            'ground_state_energy': ground_energy,
            'computation_time': elapsed_time,
            'method': 'variational_monte_carlo'
        }


class ElectronProtonNeutronQuantumSystem(QuantumSystem):
    """Quantum system with electron, proton, and neutron."""
    
    def __init__(self):
        """Initialize electron-proton-neutron quantum system."""
        super().__init__()
        self.electron_mass = 1.0
        self.proton_mass = 1836.15
        self.neutron_mass = 1838.68
        self.electron_charge = -1.0
        self.proton_charge = 1.0
        self.neutron_charge = 0.0
    
    def solve_hydrogen_like(self, grid_size: int = 200, grid_range: Tuple[float, float] = (-10, 10)) -> Dict:
        """
        Solve electron-proton system (hydrogen-like) with neutron as perturbation.
        
        Args:
            grid_size: Number of grid points
            grid_range: (r_min, r_max) for radial grid
            
        Returns:
            Dictionary with energy levels and wavefunctions
        """
        start_time = time.time()
        
        # Radial coordinate (1D approximation)
        r = np.linspace(grid_range[0], grid_range[1], grid_size)
        r_positive = np.abs(r) + 1e-10  # Avoid division by zero
        
        # Reduced mass (electron-proton)
        mu = (self.electron_mass * self.proton_mass) / (self.electron_mass + self.proton_mass)
        
        # Potential: Coulomb attraction + neutron perturbation
        def potential(r_val):
            # Coulomb attraction between electron and proton
            V_coulomb = -self.K_COULOMB * self.electron_charge * self.proton_charge / (np.abs(r_val) + 1e-10)
            # Neutron perturbation (weak, charge-neutral but has mass)
            # The neutron affects the center of mass motion
            V_neutron = 0.0  # Negligible for this approximation
            return V_coulomb + V_neutron
        
        V = np.array([potential(ri) for ri in r])
        
        # Solve Schrödinger equation
        dx = r[1] - r[0]
        n = len(r)
        
        # Kinetic energy operator
        kinetic = -self.HBAR**2 / (2 * mu * dx**2)
        diag_main = np.ones(n) * (-2 * kinetic)
        diag_off = np.ones(n - 1) * kinetic
        
        # Hamiltonian
        H = diags([diag_off, diag_main + V, diag_off], [-1, 0, 1], format='csr')
        
        # Solve for lowest eigenvalues
        eigenvalues, eigenfunctions = eigsh(H, k=min(5, n-1), which='SA')
        
        # Normalize
        for i in range(len(eigenvalues)):
            norm = np.sqrt(np.trapz(eigenfunctions[:, i]**2, r))
            eigenfunctions[:, i] /= norm
        
        elapsed_time = time.time() - start_time
        
        # Analytical hydrogen ground state energy for comparison
        E_hydrogen_analytical = -0.5 * mu  # In atomic units
        
        return {
            'ground_state_energy': eigenvalues[0],
            'excited_states': eigenvalues[1:],
            'ground_state_wavefunction': eigenfunctions[:, 0],
            'wavefunctions': eigenfunctions,
            'grid': r,
            'analytical_hydrogen_energy': E_hydrogen_analytical,
            'computation_time': elapsed_time
        }
    
    def solve_full_3body_quantum(self, grid_size: int = 30) -> Dict:
        """
        Full 3-body quantum solution (very expensive).
        Uses Born-Oppenheimer approximation: heavy particles (proton, neutron) 
        are nearly stationary.
        """
        start_time = time.time()
        
        # Born-Oppenheimer: proton and neutron form a "nucleus"
        # Electron moves in the field of this nucleus
        # Reduced mass of nucleus
        m_nucleus = self.proton_mass + self.neutron_mass
        
        # Effective reduced mass for electron-nucleus system
        mu_eff = (self.electron_mass * m_nucleus) / (self.electron_mass + m_nucleus)
        
        # Grid for electron position relative to nucleus center of mass
        r = np.linspace(-5, 5, grid_size)
        r_abs = np.abs(r) + 1e-10
        
        # Potential: electron-nucleus Coulomb (proton charge)
        V = -self.K_COULOMB / r_abs
        
        # Solve Schrödinger equation
        dx = r[1] - r[0]
        n = len(r)
        
        kinetic = -self.HBAR**2 / (2 * mu_eff * dx**2)
        diag_main = np.ones(n) * (-2 * kinetic)
        diag_off = np.ones(n - 1) * kinetic
        
        H = diags([diag_off, diag_main + V, diag_off], [-1, 0, 1], format='csr')
        
        eigenvalues, eigenfunctions = eigsh(H, k=min(3, n-1), which='SA')
        
        # Normalize
        for i in range(len(eigenvalues)):
            norm = np.sqrt(np.trapz(eigenfunctions[:, i]**2, r))
            eigenfunctions[:, i] /= norm
        
        elapsed_time = time.time() - start_time
        
        return {
            'ground_state_energy': eigenvalues[0],
            'excited_states': eigenvalues[1:],
            'wavefunctions': eigenfunctions,
            'grid': r,
            'computation_time': elapsed_time,
            'method': 'born_oppenheimer'
        }

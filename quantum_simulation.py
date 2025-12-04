"""
Quantum Mechanics Simulation Module
Implements time-dependent Schrödinger equation and quantum dynamics
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.linalg import expm
from typing import Tuple, List, Dict
import time


class QuantumSimulator:
    """Quantum mechanics simulator using time-dependent Schrödinger equation"""
    
    # Physical constants
    H_BAR = 1.055e-34  # J⋅s (reduced Planck constant)
    K_COULOMB = 8.99e9  # N⋅m²/C²
    E_CHARGE = 1.602e-19  # C
    M_ELECTRON = 9.109e-31  # kg
    M_PROTON = 1.673e-27  # kg
    M_NEUTRON = 1.675e-27  # kg
    
    def __init__(self, particles: List[Dict], grid_size: int = 64, 
                 grid_range: float = 5e-10):
        """
        Initialize quantum simulator
        
        Args:
            particles: List of particle dicts with keys: 'charge', 'mass', 'position', 'velocity'
            grid_size: Number of grid points per dimension
            grid_range: Spatial extent of grid in meters (±grid_range)
        """
        self.particles = particles
        self.n_particles = len(particles)
        self.grid_size = grid_size
        self.grid_range = grid_range
        
        # Create spatial grid
        self.x = np.linspace(-grid_range, grid_range, grid_size)
        self.dx = self.x[1] - self.x[0]
        
        # For 3D, we'll use 1D approximation for computational efficiency
        # In full 3D, grid_size^3 would be needed
        
    def coulomb_potential_1d(self, x: np.ndarray, q1: float, q2: float, 
                            x1: float, x2: float) -> np.ndarray:
        """
        Calculate 1D Coulomb potential between two particles
        
        Args:
            x: Grid positions
            q1, q2: Charges
            x1, x2: Particle positions
            
        Returns:
            Potential energy array
        """
        # For 1D approximation, use V = k*q1*q2/|x-x1| for particle at x1
        # This is a simplified model
        r1 = np.abs(x - x1)
        r2 = np.abs(x - x2)
        
        # Avoid division by zero
        r1 = np.where(r1 < self.dx, self.dx, r1)
        r2 = np.where(r2 < self.dx, self.dx, r2)
        
        # 1D Coulomb potential (approximation)
        V = self.K_COULOMB * q1 * q2 / r1
        V += self.K_COULOMB * q1 * q2 / r2
        
        return V
    
    def build_hamiltonian_1d(self, mass: float, potential: np.ndarray) -> np.ndarray:
        """
        Build 1D Hamiltonian matrix: H = -ħ²/(2m) * d²/dx² + V(x)
        
        Args:
            mass: Particle mass
            potential: Potential energy array
            
        Returns:
            Hamiltonian matrix
        """
        n = len(self.x)
        
        # Kinetic energy operator: -ħ²/(2m) * d²/dx²
        # Using finite difference approximation
        kinetic_coeff = -self.H_BAR**2 / (2 * mass * self.dx**2)
        
        # Second derivative matrix (finite difference)
        diag_main = -2 * np.ones(n)
        diag_upper = np.ones(n - 1)
        diag_lower = np.ones(n - 1)
        
        laplacian = diags([diag_lower, diag_main, diag_upper], 
                         [-1, 0, 1], shape=(n, n)).toarray()
        
        kinetic = kinetic_coeff * laplacian
        
        # Potential energy operator (diagonal)
        potential_matrix = np.diag(potential)
        
        # Total Hamiltonian
        H = kinetic + potential_matrix
        
        return H
    
    def time_evolution_operator(self, H: np.ndarray, dt: float) -> np.ndarray:
        """
        Time evolution operator: U = exp(-i*H*dt/ħ)
        
        Args:
            H: Hamiltonian matrix
            dt: Time step
            
        Returns:
            Time evolution operator matrix
        """
        return expm(-1j * H * dt / self.H_BAR)
    
    def schrodinger_equation(self, t: float, psi: np.ndarray) -> np.ndarray:
        """
        Time-dependent Schrödinger equation: iħ dψ/dt = Hψ
        
        Args:
            t: Time
            psi: Wave function (flattened)
            
        Returns:
            Time derivative of wave function
        """
        # Reshape wave function
        psi_reshaped = psi.reshape(self.grid_size)
        
        # Get Hamiltonian for first particle (simplified)
        # In full treatment, would need many-body Hamiltonian
        mass = self.particles[0]['mass']
        
        # Build potential from all particles
        potential = np.zeros(self.grid_size)
        for i, p in enumerate(self.particles):
            q = p['charge']
            x_pos = p.get('position', [0.0])[0]  # Use x-component
            for j, p2 in enumerate(self.particles):
                if i != j:
                    q2 = p2['charge']
                    x_pos2 = p2.get('position', [0.0])[0]
                    potential += self.coulomb_potential_1d(
                        self.x, q, q2, x_pos, x_pos2
                    ) / len(self.particles)
        
        H = self.build_hamiltonian_1d(mass, potential)
        
        # Apply Hamiltonian: dψ/dt = -i/ħ * H * ψ
        dpsi_dt = -1j / self.H_BAR * H @ psi_reshaped
        
        return dpsi_dt.flatten()
    
    def calculate_expectation_values(self, psi: np.ndarray) -> Dict:
        """
        Calculate expectation values: <x>, <p>, <E>
        
        Args:
            psi: Wave function
            
        Returns:
            Dictionary with expectation values
        """
        psi_normalized = psi / np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        
        # <x> = ∫ x |ψ|² dx
        x_expectation = np.sum(self.x * np.abs(psi_normalized)**2) * self.dx
        
        # <p> = -iħ ∫ ψ* dψ/dx dx (momentum)
        dpsi_dx = np.gradient(psi_normalized, self.dx)
        p_expectation = -1j * self.H_BAR * np.sum(
            np.conj(psi_normalized) * dpsi_dx
        ) * self.dx
        
        # <E> = ∫ ψ* H ψ dx
        mass = self.particles[0]['mass']
        potential = np.zeros(self.grid_size)
        for i, p in enumerate(self.particles):
            q = p['charge']
            x_pos = p.get('position', [0.0])[0]
            for j, p2 in enumerate(self.particles):
                if i != j:
                    q2 = p2['charge']
                    x_pos2 = p2.get('position', [0.0])[0]
                    potential += self.coulomb_potential_1d(
                        self.x, q, q2, x_pos, x_pos2
                    ) / len(self.particles)
        
        H = self.build_hamiltonian_1d(mass, potential)
        H_psi = H @ psi_normalized
        E_expectation = np.sum(np.conj(psi_normalized) * H_psi) * self.dx
        
        return {
            'position': x_expectation,
            'momentum': p_expectation,
            'energy': E_expectation
        }
    
    def gaussian_wave_packet(self, x0: float, p0: float, sigma: float) -> np.ndarray:
        """
        Create Gaussian wave packet initial condition
        
        Args:
            x0: Center position
            p0: Initial momentum
            sigma: Width of wave packet
            
        Returns:
            Wave function array
        """
        mass = self.particles[0]['mass']
        psi = np.exp(-(self.x - x0)**2 / (2 * sigma**2)).astype(complex)
        psi *= np.exp(1j * p0 * (self.x - x0) / self.H_BAR)
        psi /= np.sqrt(np.sqrt(np.pi) * sigma)  # Normalization
        
        return psi
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: np.ndarray = None) -> Dict:
        """
        Run quantum simulation
        
        Args:
            t_span: (t_start, t_end) in seconds
            t_eval: Array of times to evaluate solution
            
        Returns:
            Dictionary with 'time', 'wavefunction', 'expectation_values', 
            'computation_time'
        """
        start_time = time.time()
        
        # Initial wave function (Gaussian wave packet)
        x0 = self.particles[0].get('position', [0.0])[0]
        v0 = self.particles[0].get('velocity', [0.0])[0]
        mass = self.particles[0]['mass']
        p0 = mass * v0
        sigma = self.grid_range / 10  # Wave packet width
        
        psi0 = self.gaussian_wave_packet(x0, p0, sigma)
        
        # Solve time-dependent Schrödinger equation
        solution = solve_ivp(
            self.schrodinger_equation,
            t_span,
            psi0.flatten(),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        computation_time = time.time() - start_time
        
        # Extract results
        wavefunctions = solution.y.T.reshape(-1, self.grid_size)
        
        # Calculate expectation values at each time step
        expectation_values = []
        for i in range(len(solution.t)):
            exp_vals = self.calculate_expectation_values(wavefunctions[i])
            expectation_values.append(exp_vals)
        
        # Extract energies
        energies = [ev['energy'].real for ev in expectation_values]
        positions = [ev['position'].real for ev in expectation_values]
        
        return {
            'time': solution.t,
            'wavefunction': wavefunctions,
            'expectation_values': expectation_values,
            'energy': np.array(energies),
            'position': np.array(positions),
            'computation_time': computation_time,
            'success': solution.success,
            'grid': self.x
        }


def create_3electron_quantum_system(initial_separation: float = 1e-10) -> List[Dict]:
    """Create initial conditions for 3-electron quantum system"""
    sim = QuantumSimulator([])
    
    particles = []
    for i in range(3):
        angle = 2 * np.pi * i / 3
        position = initial_separation * np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ])
        velocity = 1e5 * np.random.randn(3)
        
        particles.append({
            'charge': -sim.E_CHARGE,
            'mass': sim.M_ELECTRON,
            'position': position,
            'velocity': velocity
        })
    
    return particles


def create_electron_proton_neutron_quantum_system(
    initial_separation: float = 1e-10
) -> List[Dict]:
    """Create initial conditions for electron-proton-neutron quantum system"""
    sim = QuantumSimulator([])
    
    particles = [
        {
            'charge': -sim.E_CHARGE,
            'mass': sim.M_ELECTRON,
            'position': np.array([-initial_separation, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0])
        },
        {
            'charge': sim.E_CHARGE,
            'mass': sim.M_PROTON,
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0])
        },
        {
            'charge': 0.0,
            'mass': sim.M_NEUTRON,
            'position': np.array([initial_separation, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0])
        }
    ]
    
    return particles

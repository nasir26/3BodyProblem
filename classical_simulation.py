"""
Classical Mechanics Simulation Module
Implements Newtonian dynamics for multi-particle systems
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict
import time


class ClassicalSimulator:
    """Classical mechanics simulator using Newton's laws"""
    
    # Physical constants
    K_COULOMB = 8.99e9  # N⋅m²/C²
    E_CHARGE = 1.602e-19  # C
    M_ELECTRON = 9.109e-31  # kg
    M_PROTON = 1.673e-27  # kg
    M_NEUTRON = 1.675e-27  # kg
    
    def __init__(self, particles: List[Dict], dt: float = 1e-18):
        """
        Initialize classical simulator
        
        Args:
            particles: List of particle dicts with keys: 'charge', 'mass', 'position', 'velocity'
            dt: Time step for integration (seconds)
        """
        self.particles = particles
        self.n_particles = len(particles)
        self.dt = dt
        self.time_elapsed = 0.0
        
    def coulomb_force(self, r1: np.ndarray, r2: np.ndarray, q1: float, q2: float) -> np.ndarray:
        """
        Calculate Coulomb force between two charged particles
        
        Args:
            r1, r2: Position vectors (m)
            q1, q2: Charges (C)
            
        Returns:
            Force vector on particle 1 (N)
        """
        r_vec = r2 - r1
        r_mag = np.linalg.norm(r_vec)
        
        # Avoid division by zero
        if r_mag < 1e-15:
            return np.zeros(3)
        
        # Coulomb's law: F = k * q1 * q2 / r^2 * r_hat
        force_magnitude = self.K_COULOMB * q1 * q2 / (r_mag ** 2)
        force_direction = r_vec / r_mag
        
        return force_magnitude * force_direction
    
    def calculate_forces(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """
        Calculate net force on each particle
        
        Args:
            positions: Array of shape (n_particles, 3) with positions
            charges: Array of shape (n_particles,) with charges
            
        Returns:
            Array of shape (n_particles, 3) with net forces
        """
        forces = np.zeros((self.n_particles, 3))
        
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j:
                    forces[i] += self.coulomb_force(
                        positions[i], positions[j],
                        charges[i], charges[j]
                    )
        
        return forces
    
    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        System of ODEs: dy/dt = [velocities, accelerations]
        
        Args:
            t: Time
            y: State vector [positions, velocities] flattened
            
        Returns:
            Derivative vector
        """
        # Reshape state vector
        n = self.n_particles
        positions = y[:3*n].reshape(n, 3)
        velocities = y[3*n:].reshape(n, 3)
        
        # Extract charges and masses
        charges = np.array([p['charge'] for p in self.particles])
        masses = np.array([p['mass'] for p in self.particles])
        
        # Calculate forces
        forces = self.calculate_forces(positions, charges)
        
        # Calculate accelerations: a = F/m
        accelerations = forces / masses[:, np.newaxis]
        
        # Return derivatives: [d(positions)/dt, d(velocities)/dt]
        return np.concatenate([velocities.flatten(), accelerations.flatten()])
    
    def simulate(self, t_span: Tuple[float, float], t_eval: np.ndarray = None) -> Dict:
        """
        Run classical simulation
        
        Args:
            t_span: (t_start, t_end) in seconds
            t_eval: Array of times to evaluate solution
            
        Returns:
            Dictionary with 'time', 'positions', 'velocities', 'energy', 'computation_time'
        """
        start_time = time.time()
        
        # Initial conditions
        initial_positions = np.array([p['position'] for p in self.particles])
        initial_velocities = np.array([p['velocity'] for p in self.particles])
        y0 = np.concatenate([initial_positions.flatten(), initial_velocities.flatten()])
        
        # Solve ODE system
        solution = solve_ivp(
            self.equations_of_motion,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        computation_time = time.time() - start_time
        
        # Extract results
        n = self.n_particles
        positions = solution.y[:3*n].T.reshape(-1, n, 3)
        velocities = solution.y[3*n:].T.reshape(-1, n, 3)
        
        # Calculate energies
        charges = np.array([p['charge'] for p in self.particles])
        masses = np.array([p['mass'] for p in self.particles])
        
        kinetic_energy = []
        potential_energy = []
        
        for i in range(len(solution.t)):
            # Kinetic energy: KE = 0.5 * m * v^2
            ke = 0.5 * np.sum(masses * np.sum(velocities[i]**2, axis=1))
            
            # Potential energy: PE = k * q1 * q2 / r (sum over all pairs)
            pe = 0.0
            for j in range(n):
                for k in range(j+1, n):
                    r = np.linalg.norm(positions[i, j] - positions[i, k])
                    if r > 1e-15:
                        pe += self.K_COULOMB * charges[j] * charges[k] / r
            
            kinetic_energy.append(ke)
            potential_energy.append(pe)
        
        total_energy = np.array(kinetic_energy) + np.array(potential_energy)
        
        return {
            'time': solution.t,
            'positions': positions,
            'velocities': velocities,
            'kinetic_energy': np.array(kinetic_energy),
            'potential_energy': np.array(potential_energy),
            'total_energy': total_energy,
            'computation_time': computation_time,
            'success': solution.success
        }


def create_3electron_system(initial_separation: float = 1e-10) -> List[Dict]:
    """
    Create initial conditions for 3-electron system
    
    Args:
        initial_separation: Initial distance between electrons (m)
        
    Returns:
        List of particle dictionaries
    """
    sim = ClassicalSimulator([])
    
    # Place electrons in equilateral triangle
    particles = []
    for i in range(3):
        angle = 2 * np.pi * i / 3
        position = initial_separation * np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ])
        
        # Small random initial velocities to break symmetry
        velocity = 1e5 * np.random.randn(3)  # m/s
        
        particles.append({
            'charge': -sim.E_CHARGE,
            'mass': sim.M_ELECTRON,
            'position': position,
            'velocity': velocity
        })
    
    return particles


def create_electron_proton_neutron_system(initial_separation: float = 1e-10) -> List[Dict]:
    """
    Create initial conditions for 1 electron + 1 proton + 1 neutron system
    
    Args:
        initial_separation: Initial distance between particles (m)
        
    Returns:
        List of particle dictionaries
    """
    sim = ClassicalSimulator([])
    
    # Place particles in a line
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

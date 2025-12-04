"""
Classical Mechanics Simulation Module
Simulates multi-particle systems using Newtonian mechanics with Coulomb forces.
"""

import numpy as np
from scipy.integrate import odeint
from typing import List, Tuple, Dict
import time


class ClassicalParticle:
    """Represents a particle in classical mechanics."""
    
    def __init__(self, mass: float, charge: float, position: np.ndarray, velocity: np.ndarray):
        """
        Initialize a classical particle.
        
        Args:
            mass: Particle mass in atomic units
            charge: Particle charge in units of elementary charge
            position: Initial position vector (3D)
            velocity: Initial velocity vector (3D)
        """
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)


class ClassicalSystem:
    """Classical mechanics simulation system."""
    
    # Physical constants (in atomic units)
    K_COULOMB = 1.0  # Coulomb constant in atomic units
    EPSILON = 1e-10  # Small value to prevent division by zero
    
    def __init__(self, particles: List[ClassicalParticle]):
        """
        Initialize a classical system.
        
        Args:
            particles: List of ClassicalParticle objects
        """
        self.particles = particles
        self.n_particles = len(particles)
        
    def coulomb_force(self, r1: np.ndarray, r2: np.ndarray, q1: float, q2: float) -> np.ndarray:
        """
        Calculate Coulomb force between two particles.
        
        Args:
            r1: Position of particle 1
            r2: Position of particle 2
            q1: Charge of particle 1
            q2: Charge of particle 2
            
        Returns:
            Force vector on particle 1 due to particle 2
        """
        r_vec = r1 - r2
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < self.EPSILON:
            return np.zeros(3)
        
        force_magnitude = self.K_COULOMB * q1 * q2 / (r_mag ** 3)
        return force_magnitude * r_vec
    
    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute all forces on all particles.
        
        Args:
            positions: Array of shape (n_particles, 3) with all positions
            
        Returns:
            Array of shape (n_particles, 3) with all forces
        """
        forces = np.zeros((self.n_particles, 3))
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                force_ij = self.coulomb_force(
                    positions[i], 
                    positions[j],
                    self.particles[i].charge,
                    self.particles[j].charge
                )
                forces[i] += force_ij
                forces[j] -= force_ij  # Newton's third law
        
        return forces
    
    def equations_of_motion(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        System of differential equations for Newton's laws.
        
        Args:
            y: State vector [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]
            t: Time
            
        Returns:
            Derivative of state vector
        """
        # Reshape state vector
        positions = y[:3*self.n_particles].reshape(self.n_particles, 3)
        velocities = y[3*self.n_particles:].reshape(self.n_particles, 3)
        
        # Compute forces
        forces = self.compute_forces(positions)
        
        # Compute accelerations
        accelerations = np.zeros_like(forces)
        for i in range(self.n_particles):
            accelerations[i] = forces[i] / self.particles[i].mass
        
        # Return derivatives: [velocities, accelerations]
        dydt = np.concatenate([velocities.flatten(), accelerations.flatten()])
        return dydt
    
    def compute_energy(self, positions: np.ndarray, velocities: np.ndarray) -> Dict[str, float]:
        """
        Compute total energy (kinetic + potential).
        
        Args:
            positions: Array of shape (n_particles, 3)
            velocities: Array of shape (n_particles, 3)
            
        Returns:
            Dictionary with 'kinetic', 'potential', and 'total' energy
        """
        # Kinetic energy
        kinetic = 0.0
        for i in range(self.n_particles):
            v_sq = np.dot(velocities[i], velocities[i])
            kinetic += 0.5 * self.particles[i].mass * v_sq
        
        # Potential energy (Coulomb)
        potential = 0.0
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r_vec = positions[i] - positions[j]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > self.EPSILON:
                    potential += self.K_COULOMB * self.particles[i].charge * self.particles[j].charge / r_mag
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'total': kinetic + potential
        }
    
    def simulate(self, t_max: float, dt: float, method: str = 'odeint') -> Dict:
        """
        Run classical simulation.
        
        Args:
            t_max: Maximum simulation time
            dt: Time step
            method: Integration method ('odeint' or 'verlet')
            
        Returns:
            Dictionary with simulation results including timing
        """
        start_time = time.time()
        
        # Initial state vector
        initial_positions = np.array([p.position for p in self.particles])
        initial_velocities = np.array([p.velocity for p in self.particles])
        y0 = np.concatenate([initial_positions.flatten(), initial_velocities.flatten()])
        
        # Time array
        t = np.arange(0, t_max, dt)
        
        if method == 'odeint':
            # Solve ODE system
            solution = odeint(self.equations_of_motion, y0, t)
            
            # Extract positions and velocities
            positions = solution[:, :3*self.n_particles].reshape(len(t), self.n_particles, 3)
            velocities = solution[:, 3*self.n_particles:].reshape(len(t), self.n_particles, 3)
        else:
            # Verlet integration (alternative method)
            positions, velocities = self._verlet_integration(y0, t, dt)
        
        # Compute energies
        energies = []
        for i in range(len(t)):
            energy = self.compute_energy(positions[i], velocities[i])
            energies.append(energy)
        
        elapsed_time = time.time() - start_time
        
        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'energies': energies,
            'computation_time': elapsed_time,
            'n_steps': len(t)
        }
    
    def _verlet_integration(self, y0: np.ndarray, t: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Verlet integration method (alternative to odeint)."""
        n_steps = len(t)
        positions = np.zeros((n_steps, self.n_particles, 3))
        velocities = np.zeros((n_steps, self.n_particles, 3))
        
        # Initial conditions
        positions[0] = y0[:3*self.n_particles].reshape(self.n_particles, 3)
        velocities[0] = y0[3*self.n_particles:].reshape(self.n_particles, 3)
        
        # First step using Euler
        forces = self.compute_forces(positions[0])
        accelerations = np.array([forces[i] / self.particles[i].mass for i in range(self.n_particles)])
        positions[1] = positions[0] + velocities[0] * dt + 0.5 * accelerations * dt**2
        
        # Verlet steps
        for i in range(1, n_steps - 1):
            forces = self.compute_forces(positions[i])
            accelerations = np.array([forces[j] / self.particles[j].mass for j in range(self.n_particles)])
            positions[i+1] = 2 * positions[i] - positions[i-1] + accelerations * dt**2
            velocities[i] = (positions[i+1] - positions[i-1]) / (2 * dt)
        
        # Last velocity
        forces = self.compute_forces(positions[-1])
        accelerations = np.array([forces[i] / self.particles[i].mass for i in range(self.n_particles)])
        velocities[-1] = velocities[-2] + accelerations * dt
        
        return positions, velocities


def create_three_electron_system() -> ClassicalSystem:
    """Create a classical 3-electron system."""
    # Electrons have mass = 1, charge = -1 in atomic units
    # Place electrons in a triangle configuration
    particles = [
        ClassicalParticle(mass=1.0, charge=-1.0, position=[0.0, 0.0, 0.0], velocity=[0.1, 0.0, 0.0]),
        ClassicalParticle(mass=1.0, charge=-1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.1, 0.0]),
        ClassicalParticle(mass=1.0, charge=-1.0, position=[0.5, 0.866, 0.0], velocity=[-0.1, -0.1, 0.0])
    ]
    return ClassicalSystem(particles)


def create_electron_proton_neutron_system() -> ClassicalSystem:
    """Create a classical electron-proton-neutron system."""
    # Electron: mass = 1, charge = -1
    # Proton: mass ≈ 1836, charge = +1
    # Neutron: mass ≈ 1839, charge = 0
    particles = [
        ClassicalParticle(mass=1.0, charge=-1.0, position=[0.0, 0.0, 0.0], velocity=[0.1, 0.0, 0.0]),
        ClassicalParticle(mass=1836.15, charge=+1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0]),
        ClassicalParticle(mass=1838.68, charge=0.0, position=[1.0, 0.1, 0.0], velocity=[0.0, 0.0, 0.0])
    ]
    return ClassicalSystem(particles)

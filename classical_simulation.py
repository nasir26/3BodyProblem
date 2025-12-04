"""
Classical Mechanics Simulation Module
Simulates particle systems using Newtonian mechanics and Coulomb forces
"""

import numpy as np
from scipy.integrate import odeint
from typing import List, Tuple, Dict
import time

# Physical constants
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
K_COULOMB = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant
M_ELECTRON = 9.1093837015e-31  # Electron mass (kg)
M_PROTON = 1.67262192369e-27  # Proton mass (kg)
M_NEUTRON = 1.67492749804e-27  # Neutron mass (kg)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)

# Use atomic units for numerical stability (hbar = m_e = e = 1)
# In atomic units, distances are in Bohr radii, energies in Hartrees
AU_LENGTH = 5.29177210903e-11  # Bohr radius (m)
AU_ENERGY = 4.3597447222071e-18  # Hartree (J)
AU_TIME = 2.4188843265857e-17  # Atomic time unit (s)


class ClassicalParticle:
    """Represents a particle in classical mechanics"""
    
    def __init__(self, mass: float, charge: float, position: np.ndarray, velocity: np.ndarray):
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.force = np.zeros(3, dtype=np.float64)


class ClassicalSimulator:
    """Classical mechanics simulator using Newton's laws and Coulomb forces"""
    
    def __init__(self, particles: List[ClassicalParticle], dt: float = 1e-4):
        self.particles = particles
        self.dt = dt
        self.time = 0.0
        self.trajectory = []
        
    def compute_forces(self):
        """Compute Coulomb forces between all particles"""
        n = len(self.particles)
        
        # Reset forces
        for particle in self.particles:
            particle.force = np.zeros(3)
        
        # Compute pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = self.particles[i], self.particles[j]
                r_vec = p2.position - p1.position
                r = np.linalg.norm(r_vec)
                
                # Avoid division by zero
                if r < 1e-10:
                    r = 1e-10
                
                # Coulomb force: F = k * q1 * q2 / r^2 * r_hat
                force_magnitude = K_COULOMB * p1.charge * p2.charge / (r ** 2)
                force_vec = force_magnitude * r_vec / r
                
                p1.force += force_vec
                p2.force -= force_vec
    
    def step(self):
        """Perform one time step using Verlet integration"""
        self.compute_forces()
        
        # Update positions and velocities
        for particle in self.particles:
            # v = v + a*dt
            acceleration = particle.force / particle.mass
            particle.velocity += acceleration * self.dt
            # x = x + v*dt
            particle.position += particle.velocity * self.dt
        
        self.time += self.dt
    
    def simulate(self, total_time: float, save_interval: int = 10):
        """Run simulation for specified time"""
        n_steps = int(total_time / self.dt)
        self.trajectory = []
        
        start_time = time.time()
        
        for step in range(n_steps):
            self.step()
            
            if step % save_interval == 0:
                state = {
                    'time': self.time,
                    'positions': [p.position.copy() for p in self.particles],
                    'velocities': [p.velocity.copy() for p in self.particles],
                    'energies': self.compute_energies()
                }
                self.trajectory.append(state)
        
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def compute_energies(self) -> Dict[str, float]:
        """Compute kinetic, potential, and total energy"""
        kinetic = 0.0
        potential = 0.0
        
        # Kinetic energy
        for particle in self.particles:
            v_sq = np.dot(particle.velocity, particle.velocity)
            kinetic += 0.5 * particle.mass * v_sq
        
        # Potential energy (Coulomb)
        n = len(self.particles)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = self.particles[i], self.particles[j]
                r = np.linalg.norm(p2.position - p1.position)
                if r > 1e-10:
                    potential += K_COULOMB * p1.charge * p2.charge / r
        
        total = kinetic + potential
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'total': total
        }


def create_3electron_system(initial_separation: float = 1e-10) -> List[ClassicalParticle]:
    """Create a 3-electron system with initial conditions"""
    particles = []
    
    # Place electrons in a triangle configuration
    positions = [
        np.array([initial_separation, 0.0, 0.0]),
        np.array([-initial_separation/2, initial_separation * np.sqrt(3)/2, 0.0]),
        np.array([-initial_separation/2, -initial_separation * np.sqrt(3)/2, 0.0])
    ]
    
    # Small random velocities to break symmetry
    np.random.seed(42)
    for i, pos in enumerate(positions):
        vel = np.random.randn(3) * 1e5  # Small initial velocities
        particle = ClassicalParticle(
            mass=M_ELECTRON,
            charge=-E_CHARGE,
            position=pos,
            velocity=vel
        )
        particles.append(particle)
    
    return particles


def create_electron_proton_neutron_system(initial_separation: float = 5e-11) -> List[ClassicalParticle]:
    """Create a system with 1 electron, 1 proton, and 1 neutron"""
    particles = []
    
    # Proton at origin
    particles.append(ClassicalParticle(
        mass=M_PROTON,
        charge=E_CHARGE,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    ))
    
    # Electron at some distance
    particles.append(ClassicalParticle(
        mass=M_ELECTRON,
        charge=-E_CHARGE,
        position=np.array([initial_separation, 0.0, 0.0]),
        velocity=np.array([0.0, 1e6, 0.0])  # Orbital velocity
    ))
    
    # Neutron nearby (no charge, but has mass)
    particles.append(ClassicalParticle(
        mass=M_NEUTRON,
        charge=0.0,
        position=np.array([0.0, initial_separation * 0.5, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    ))
    
    return particles

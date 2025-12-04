"""
Classical Mechanics Simulation Module
Uses Newtonian mechanics with Coulomb forces for charged particles
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict
import time

# Physical constants
K_COULOMB = 8.9875517923e9  # N⋅m²/C²
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg


class ClassicalParticle:
    """Represents a particle in classical mechanics"""
    
    def __init__(self, mass: float, charge: float, position: np.ndarray, velocity: np.ndarray):
        self.mass = mass
        self.charge = charge  # in elementary charge units
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
    
    def copy(self):
        return ClassicalParticle(self.mass, self.charge, self.position.copy(), self.velocity.copy())


class ClassicalSystem:
    """Classical N-body system with Coulomb interactions"""
    
    def __init__(self, particles: List[ClassicalParticle]):
        self.particles = particles
        self.n_particles = len(particles)
        self.dim = 3  # 3D space
    
    def coulomb_force(self, r1: np.ndarray, r2: np.ndarray, q1: float, q2: float) -> np.ndarray:
        """Calculate Coulomb force on particle 1 due to particle 2"""
        r_vec = r1 - r2
        r_mag = np.linalg.norm(r_vec)
        
        # Avoid division by zero
        if r_mag < 1e-15:
            return np.zeros(3)
        
        # Coulomb force: F = k * q1 * q2 / r^2 * r_hat
        force_magnitude = K_COULOMB * q1 * q2 * (ELEMENTARY_CHARGE ** 2) / (r_mag ** 2)
        force = force_magnitude * r_vec / r_mag
        
        return force
    
    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute all forces on all particles"""
        forces = np.zeros((self.n_particles, self.dim))
        
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j:
                    q1 = self.particles[i].charge * ELEMENTARY_CHARGE
                    q2 = self.particles[j].charge * ELEMENTARY_CHARGE
                    r1 = positions[i * self.dim:(i + 1) * self.dim]
                    r2 = positions[j * self.dim:(j + 1) * self.dim]
                    
                    forces[i] += self.coulomb_force(r1, r2, q1, q2)
        
        return forces
    
    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """Differential equations for the system"""
        # y contains: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]
        n = len(y) // 2
        positions = y[:n]
        velocities = y[n:]
        
        # Reshape positions
        positions_reshaped = positions.reshape(self.n_particles, self.dim)
        
        # Compute forces
        forces = self.compute_forces(positions_reshaped)
        
        # Compute accelerations: a = F/m
        accelerations = np.zeros_like(forces)
        for i in range(self.n_particles):
            accelerations[i] = forces[i] / self.particles[i].mass
        
        # Return derivatives: [dx/dt, dv/dt] = [v, a]
        dydt = np.concatenate([velocities, accelerations.flatten()])
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float], t_eval: np.ndarray = None, 
                 rtol: float = 1e-8, atol: float = 1e-10) -> Dict:
        """Simulate the system over time"""
        start_time = time.time()
        
        # Initial conditions: [positions, velocities]
        y0 = np.concatenate([
            [p.position for p in self.particles],
            [p.velocity for p in self.particles]
        ]).flatten()
        
        # Solve ODE
        sol = solve_ivp(
            self.equations_of_motion,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract results
        n_steps = len(sol.t)
        positions = np.zeros((n_steps, self.n_particles, self.dim))
        velocities = np.zeros((n_steps, self.n_particles, self.dim))
        
        n = len(y0) // 2
        for i, t in enumerate(sol.t):
            y = sol.y[:, i]
            positions[i] = y[:n].reshape(self.n_particles, self.dim)
            velocities[i] = y[n:].reshape(self.n_particles, self.dim)
        
        # Compute energy
        total_energy = self.compute_energy(positions, velocities)
        
        return {
            'time': sol.t,
            'positions': positions,
            'velocities': velocities,
            'energy': total_energy,
            'computation_time': elapsed_time,
            'n_steps': n_steps,
            'success': sol.success
        }
    
    def compute_energy(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Compute total energy (kinetic + potential) at each time step"""
        n_steps = positions.shape[0]
        energy = np.zeros(n_steps)
        
        for t in range(n_steps):
            # Kinetic energy
            ke = 0.0
            for i in range(self.n_particles):
                v_sq = np.sum(velocities[t, i] ** 2)
                ke += 0.5 * self.particles[i].mass * v_sq
            
            # Potential energy (Coulomb)
            pe = 0.0
            for i in range(self.n_particles):
                for j in range(i + 1, self.n_particles):
                    r_vec = positions[t, i] - positions[t, j]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 1e-15:
                        q1 = self.particles[i].charge * ELEMENTARY_CHARGE
                        q2 = self.particles[j].charge * ELEMENTARY_CHARGE
                        pe += K_COULOMB * q1 * q2 / r_mag
            
            energy[t] = ke + pe
        
        return energy


def create_three_electron_system(initial_positions: List[np.ndarray] = None,
                                  initial_velocities: List[np.ndarray] = None) -> ClassicalSystem:
    """Create a 3-electron system"""
    if initial_positions is None:
        # Default: equilateral triangle configuration
        initial_positions = [
            np.array([1e-10, 0.0, 0.0]),  # 1 Angstrom separation
            np.array([-0.5e-10, 0.866e-10, 0.0]),
            np.array([-0.5e-10, -0.866e-10, 0.0])
        ]
    
    if initial_velocities is None:
        initial_velocities = [np.zeros(3) for _ in range(3)]
    
    particles = [
        ClassicalParticle(ELECTRON_MASS, -1.0, initial_positions[i], initial_velocities[i])
        for i in range(3)
    ]
    
    return ClassicalSystem(particles)


def create_electron_proton_neutron_system(initial_positions: List[np.ndarray] = None,
                                          initial_velocities: List[np.ndarray] = None) -> ClassicalSystem:
    """Create a system with 1 electron, 1 proton, and 1 neutron"""
    if initial_positions is None:
        # Default: electron and proton close, neutron nearby
        initial_positions = [
            np.array([0.0, 0.0, 0.0]),  # Electron at origin
            np.array([5.29e-11, 0.0, 0.0]),  # Proton at Bohr radius
            np.array([1e-10, 0.0, 0.0])  # Neutron nearby
        ]
    
    if initial_velocities is None:
        # Give electron orbital velocity
        v_electron = np.array([0.0, 2.19e6, 0.0])  # ~1% of c
        initial_velocities = [
            v_electron,
            np.zeros(3),  # Proton initially at rest
            np.zeros(3)   # Neutron initially at rest
        ]
    
    particles = [
        ClassicalParticle(ELECTRON_MASS, -1.0, initial_positions[0], initial_velocities[0]),
        ClassicalParticle(PROTON_MASS, 1.0, initial_positions[1], initial_velocities[1]),
        ClassicalParticle(NEUTRON_MASS, 0.0, initial_positions[2], initial_velocities[2])
    ]
    
    return ClassicalSystem(particles)

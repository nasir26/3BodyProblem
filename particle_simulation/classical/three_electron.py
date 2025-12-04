"""
Classical Three-Electron System Simulation

This module implements a classical simulation of three interacting electrons
using Newtonian mechanics and Coulomb's law.

Physical Model:
--------------
Each electron is treated as a classical point charge with:
- Mass: m_e = 9.109 × 10⁻³¹ kg
- Charge: q = -1.602 × 10⁻¹⁹ C

The electrons interact via Coulomb repulsion:
F_ij = k_e * q_i * q_j / r_ij² (repulsive for like charges)

Equations of Motion:
-------------------
m_i * d²r_i/dt² = Σ_j F_ij

This is solved numerically using the Velocity Verlet algorithm,
which provides O(Δt²) accuracy and conserves energy for Hamiltonian systems.
"""

import numpy as np
from typing import Tuple, List, Optional
import time
from ..constants import PHYSICAL_CONSTANTS, ATOMIC_UNITS


class ThreeElectronClassical:
    """
    Classical simulation of a three-electron system.
    
    Uses atomic units internally for numerical stability.
    """
    
    def __init__(self, use_atomic_units: bool = True):
        """
        Initialize the three-electron classical simulation.
        
        Parameters:
        -----------
        use_atomic_units : bool
            If True, use atomic units (recommended for numerical stability)
        """
        self.use_atomic_units = use_atomic_units
        
        if use_atomic_units:
            self.m_e = 1.0  # electron mass in atomic units
            self.k_e = 1.0  # Coulomb constant in atomic units
            self.q_e = -1.0  # electron charge in atomic units
        else:
            self.m_e = PHYSICAL_CONSTANTS['m_e']
            self.k_e = PHYSICAL_CONSTANTS['k_e']
            self.q_e = -PHYSICAL_CONSTANTS['e']
        
        # State variables: positions and velocities for 3 electrons in 3D
        self.positions = np.zeros((3, 3))  # 3 electrons, 3D coordinates
        self.velocities = np.zeros((3, 3))
        
        # Simulation tracking
        self.time = 0.0
        self.trajectory = []
        self.energies = []
        self.computation_time = 0.0
    
    def initialize_state(self, 
                         positions: Optional[np.ndarray] = None,
                         velocities: Optional[np.ndarray] = None,
                         configuration: str = 'equilateral'):
        """
        Initialize the positions and velocities of the electrons.
        
        Parameters:
        -----------
        positions : np.ndarray, optional
            Initial positions (3, 3) array
        velocities : np.ndarray, optional
            Initial velocities (3, 3) array
        configuration : str
            Preset configuration: 'equilateral', 'linear', 'random'
        """
        if positions is not None:
            self.positions = np.array(positions, dtype=np.float64)
        else:
            # Default configurations (in Bohr radii if using atomic units)
            if configuration == 'equilateral':
                # Equilateral triangle in xy-plane
                r = 3.0  # separation in Bohr radii
                self.positions = np.array([
                    [r, 0, 0],
                    [-r/2, r*np.sqrt(3)/2, 0],
                    [-r/2, -r*np.sqrt(3)/2, 0]
                ])
            elif configuration == 'linear':
                # Linear arrangement along x-axis
                self.positions = np.array([
                    [-3.0, 0, 0],
                    [0, 0, 0],
                    [3.0, 0, 0]
                ])
            elif configuration == 'random':
                # Random positions within a sphere
                self.positions = np.random.randn(3, 3) * 2.0
            else:
                raise ValueError(f"Unknown configuration: {configuration}")
        
        if velocities is not None:
            self.velocities = np.array(velocities, dtype=np.float64)
        else:
            # Small random initial velocities
            self.velocities = np.random.randn(3, 3) * 0.1
        
        self.time = 0.0
        self.trajectory = [self.positions.copy()]
        self.energies = [self.compute_total_energy()]
    
    def compute_coulomb_force(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Compute Coulomb force on particle 1 due to particle 2.
        
        F = k_e * q1 * q2 / |r|² * r̂
        
        For electrons (same charge), the force is repulsive.
        
        Parameters:
        -----------
        r1, r2 : np.ndarray
            Position vectors of particles 1 and 2
        
        Returns:
        --------
        np.ndarray
            Force vector on particle 1
        """
        r_vec = r1 - r2
        r_mag = np.linalg.norm(r_vec)
        
        # Avoid singularity at r=0 (classical model breaks down)
        if r_mag < 1e-10:
            return np.zeros(3)
        
        # Coulomb force: F = k * q1 * q2 / r² * r_hat
        # For electrons: q1 = q2 = -e, so q1*q2 = e² (positive, repulsive)
        force_magnitude = self.k_e * self.q_e**2 / r_mag**2
        force_direction = r_vec / r_mag
        
        return force_magnitude * force_direction
    
    def compute_total_force(self, i: int, positions: np.ndarray) -> np.ndarray:
        """
        Compute total force on electron i from all other electrons.
        
        Parameters:
        -----------
        i : int
            Index of the electron
        positions : np.ndarray
            Current positions of all electrons
        
        Returns:
        --------
        np.ndarray
            Total force vector on electron i
        """
        total_force = np.zeros(3)
        for j in range(3):
            if i != j:
                total_force += self.compute_coulomb_force(positions[i], positions[j])
        return total_force
    
    def compute_potential_energy(self, positions: Optional[np.ndarray] = None) -> float:
        """
        Compute total potential energy of the system.
        
        V = Σ_{i<j} k_e * q_i * q_j / r_ij
        
        For electrons: V = Σ_{i<j} k_e * e² / r_ij (positive, repulsive)
        """
        if positions is None:
            positions = self.positions
        
        V = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                if r_ij > 1e-10:
                    # q_e² is positive since q_e is negative
                    V += self.k_e * self.q_e**2 / r_ij
        return V
    
    def compute_kinetic_energy(self, velocities: Optional[np.ndarray] = None) -> float:
        """
        Compute total kinetic energy of the system.
        
        T = Σ_i (1/2) * m_i * |v_i|²
        """
        if velocities is None:
            velocities = self.velocities
        
        T = 0.0
        for i in range(3):
            T += 0.5 * self.m_e * np.dot(velocities[i], velocities[i])
        return T
    
    def compute_total_energy(self) -> float:
        """
        Compute total energy E = T + V.
        """
        return self.compute_kinetic_energy() + self.compute_potential_energy()
    
    def compute_angular_momentum(self) -> np.ndarray:
        """
        Compute total angular momentum L = Σ_i r_i × p_i.
        """
        L = np.zeros(3)
        for i in range(3):
            p_i = self.m_e * self.velocities[i]
            L += np.cross(self.positions[i], p_i)
        return L
    
    def step_velocity_verlet(self, dt: float):
        """
        Perform one step of the Velocity Verlet algorithm.
        
        The Velocity Verlet algorithm:
        1. v(t + dt/2) = v(t) + (dt/2) * a(t)
        2. r(t + dt) = r(t) + dt * v(t + dt/2)
        3. a(t + dt) = F(r(t + dt)) / m
        4. v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)
        
        This algorithm is symplectic (preserves phase space volume) and
        provides O(dt²) accuracy for positions.
        """
        # Compute current accelerations
        accelerations = np.zeros((3, 3))
        for i in range(3):
            force = self.compute_total_force(i, self.positions)
            accelerations[i] = force / self.m_e
        
        # Half-step velocity update
        self.velocities += 0.5 * dt * accelerations
        
        # Full-step position update
        self.positions += dt * self.velocities
        
        # Compute new accelerations at new positions
        new_accelerations = np.zeros((3, 3))
        for i in range(3):
            force = self.compute_total_force(i, self.positions)
            new_accelerations[i] = force / self.m_e
        
        # Complete velocity update
        self.velocities += 0.5 * dt * new_accelerations
        
        self.time += dt
    
    def step_rk4(self, dt: float):
        """
        Perform one step using 4th-order Runge-Kutta method.
        
        This provides O(dt⁴) accuracy but is more computationally expensive
        and not symplectic.
        """
        def derivatives(pos, vel):
            """Compute derivatives: dr/dt = v, dv/dt = F/m"""
            acc = np.zeros((3, 3))
            for i in range(3):
                force = self.compute_total_force(i, pos)
                acc[i] = force / self.m_e
            return vel.copy(), acc
        
        # k1
        k1_v, k1_a = derivatives(self.positions, self.velocities)
        
        # k2
        pos2 = self.positions + 0.5 * dt * k1_v
        vel2 = self.velocities + 0.5 * dt * k1_a
        k2_v, k2_a = derivatives(pos2, vel2)
        
        # k3
        pos3 = self.positions + 0.5 * dt * k2_v
        vel3 = self.velocities + 0.5 * dt * k2_a
        k3_v, k3_a = derivatives(pos3, vel3)
        
        # k4
        pos4 = self.positions + dt * k3_v
        vel4 = self.velocities + dt * k3_a
        k4_v, k4_a = derivatives(pos4, vel4)
        
        # Combine
        self.positions += (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.velocities += (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        
        self.time += dt
    
    def run_simulation(self, 
                       total_time: float, 
                       dt: float,
                       method: str = 'verlet',
                       save_interval: int = 1) -> dict:
        """
        Run the classical simulation.
        
        Parameters:
        -----------
        total_time : float
            Total simulation time (in atomic time units if use_atomic_units=True)
        dt : float
            Time step
        method : str
            Integration method: 'verlet' or 'rk4'
        save_interval : int
            Save state every N steps
        
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        start_time = time.perf_counter()
        
        n_steps = int(total_time / dt)
        step_func = self.step_velocity_verlet if method == 'verlet' else self.step_rk4
        
        self.trajectory = [self.positions.copy()]
        self.energies = [self.compute_total_energy()]
        times = [0.0]
        
        for step in range(n_steps):
            step_func(dt)
            
            if (step + 1) % save_interval == 0:
                self.trajectory.append(self.positions.copy())
                self.energies.append(self.compute_total_energy())
                times.append(self.time)
        
        end_time = time.perf_counter()
        self.computation_time = end_time - start_time
        
        # Convert to numpy arrays
        self.trajectory = np.array(self.trajectory)
        self.energies = np.array(self.energies)
        times = np.array(times)
        
        # Compute energy conservation error
        energy_error = np.abs(self.energies - self.energies[0]) / np.abs(self.energies[0])
        
        return {
            'method': 'classical',
            'integration': method,
            'trajectory': self.trajectory,
            'energies': self.energies,
            'times': times,
            'total_time': total_time,
            'dt': dt,
            'n_steps': n_steps,
            'computation_time': self.computation_time,
            'initial_energy': self.energies[0],
            'final_energy': self.energies[-1],
            'energy_conservation_error': energy_error[-1],
            'max_energy_error': np.max(energy_error),
            'angular_momentum': self.compute_angular_momentum(),
        }
    
    def get_average_separation(self) -> float:
        """Compute average separation between electrons."""
        separations = []
        for i in range(3):
            for j in range(i+1, 3):
                separations.append(np.linalg.norm(self.positions[i] - self.positions[j]))
        return np.mean(separations)


if __name__ == "__main__":
    # Test the simulation
    sim = ThreeElectronClassical(use_atomic_units=True)
    sim.initialize_state(configuration='equilateral')
    
    print("Initial state:")
    print(f"  Positions:\n{sim.positions}")
    print(f"  Total energy: {sim.compute_total_energy():.6f} Hartree")
    print(f"  Average separation: {sim.get_average_separation():.3f} Bohr")
    
    # Run simulation (1 atomic time unit ≈ 24 attoseconds)
    results = sim.run_simulation(total_time=10.0, dt=0.001, method='verlet')
    
    print(f"\nSimulation completed in {results['computation_time']:.4f} seconds")
    print(f"Energy conservation error: {results['energy_conservation_error']:.2e}")
    print(f"Final energy: {results['final_energy']:.6f} Hartree")

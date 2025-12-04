"""
Classical Electron-Proton-Neutron System Simulation

This module implements a classical simulation of a system containing:
- One electron
- One proton  
- One neutron

Physical Model:
--------------
This is essentially a hydrogen-like atom with an extra neutron.
The neutron does not participate in electromagnetic interactions
(charge = 0) but does have gravitational interaction (negligible
at atomic scales, but included for completeness).

Forces:
- Electron-Proton: Coulomb attraction (F = -k_e * e² / r²)
- Electron-Neutron: None (electromagnetic), negligible gravitational
- Proton-Neutron: None (electromagnetic), negligible gravitational
- Nuclear force: Not modeled classically (would require QCD)

This system is particularly interesting because:
1. Classically, the electron should spiral into the proton (radiation)
2. Quantum mechanics explains why atoms are stable
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import time
from ..constants import PHYSICAL_CONSTANTS, ATOMIC_UNITS, ELECTRON, PROTON, NEUTRON


class HydrogenLikeClassical:
    """
    Classical simulation of an electron-proton-neutron system.
    
    This simulates a hydrogen-like atom with a neutron.
    In reality, this would be part of a deuterium atom (hydrogen-2).
    """
    
    def __init__(self, use_atomic_units: bool = True, include_gravity: bool = False):
        """
        Initialize the hydrogen-like classical simulation.
        
        Parameters:
        -----------
        use_atomic_units : bool
            If True, use atomic units (recommended for numerical stability)
        include_gravity : bool
            If True, include gravitational interactions (negligible but educational)
        """
        self.use_atomic_units = use_atomic_units
        self.include_gravity = include_gravity
        
        if use_atomic_units:
            # Atomic units: ℏ = m_e = e = k_e = 1
            self.masses = np.array([
                1.0,                                    # electron
                PROTON.mass / ELECTRON.mass,            # proton (~1836.15)
                NEUTRON.mass / ELECTRON.mass,           # neutron (~1838.68)
            ])
            self.charges = np.array([-1.0, 1.0, 0.0])  # electron, proton, neutron
            self.k_e = 1.0
            self.G = 0.0  # Gravity negligible in atomic units
        else:
            self.masses = np.array([ELECTRON.mass, PROTON.mass, NEUTRON.mass])
            self.charges = np.array([ELECTRON.charge, PROTON.charge, NEUTRON.charge])
            self.k_e = PHYSICAL_CONSTANTS['k_e']
            self.G = 6.67430e-11  # Gravitational constant
        
        # State variables: [electron, proton, neutron]
        self.positions = np.zeros((3, 3))
        self.velocities = np.zeros((3, 3))
        
        # Simulation tracking
        self.time = 0.0
        self.trajectory = []
        self.energies = []
        self.computation_time = 0.0
        
        # Particle names for reference
        self.particle_names = ['electron', 'proton', 'neutron']
    
    def initialize_state(self,
                         positions: Optional[np.ndarray] = None,
                         velocities: Optional[np.ndarray] = None,
                         configuration: str = 'bohr_orbit'):
        """
        Initialize the positions and velocities.
        
        Parameters:
        -----------
        positions : np.ndarray, optional
            Initial positions (3, 3) array [electron, proton, neutron]
        velocities : np.ndarray, optional
            Initial velocities (3, 3) array
        configuration : str
            Preset: 'bohr_orbit', 'nucleus_at_origin', 'random'
        """
        if positions is not None:
            self.positions = np.array(positions, dtype=np.float64)
        else:
            if configuration == 'bohr_orbit':
                # Electron at Bohr radius, proton at origin, neutron nearby
                # In atomic units, Bohr radius = 1
                a0 = 1.0 if self.use_atomic_units else PHYSICAL_CONSTANTS['a_0']
                self.positions = np.array([
                    [a0, 0, 0],           # electron at Bohr radius
                    [0, 0, 0],            # proton at origin
                    [0.001, 0, 0],        # neutron very close to proton (nucleus)
                ])
            elif configuration == 'nucleus_at_origin':
                # Both proton and neutron at origin, electron at Bohr radius
                a0 = 1.0 if self.use_atomic_units else PHYSICAL_CONSTANTS['a_0']
                self.positions = np.array([
                    [a0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ])
            elif configuration == 'random':
                self.positions = np.random.randn(3, 3) * 2.0
                # Keep proton and neutron close together (nucleus)
                self.positions[2] = self.positions[1] + np.random.randn(3) * 0.01
            else:
                raise ValueError(f"Unknown configuration: {configuration}")
        
        if velocities is not None:
            self.velocities = np.array(velocities, dtype=np.float64)
        else:
            # For Bohr orbit, electron needs orbital velocity
            if configuration in ['bohr_orbit', 'nucleus_at_origin']:
                # Classical orbital velocity for circular orbit
                # v = sqrt(k_e * e² / (m_e * r)) in SI
                # In atomic units: v = 1/r for electron around proton
                r = np.linalg.norm(self.positions[0] - self.positions[1])
                if r > 1e-10:
                    v_orbital = 1.0 / np.sqrt(r) if self.use_atomic_units else \
                        np.sqrt(self.k_e * PHYSICAL_CONSTANTS['e']**2 / 
                                (ELECTRON.mass * r))
                    # Velocity perpendicular to position (circular orbit)
                    self.velocities[0] = [0, v_orbital, 0]
                # Proton and neutron stay relatively stationary
                self.velocities[1] = np.zeros(3)
                self.velocities[2] = np.zeros(3)
            else:
                self.velocities = np.random.randn(3, 3) * 0.1
        
        self.time = 0.0
        self.trajectory = [self.positions.copy()]
        self.energies = [self.compute_total_energy()]
    
    def compute_coulomb_force(self, i: int, j: int, positions: np.ndarray) -> np.ndarray:
        """
        Compute Coulomb force on particle i due to particle j.
        
        F = k_e * q_i * q_j / r² * r̂
        """
        r_vec = positions[i] - positions[j]
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e-10:
            return np.zeros(3)
        
        # Coulomb force
        force_magnitude = self.k_e * self.charges[i] * self.charges[j] / r_mag**2
        force_direction = r_vec / r_mag
        
        return force_magnitude * force_direction
    
    def compute_gravitational_force(self, i: int, j: int, positions: np.ndarray) -> np.ndarray:
        """
        Compute gravitational force on particle i due to particle j.
        
        F = -G * m_i * m_j / r² * r̂ (always attractive)
        """
        if not self.include_gravity or self.use_atomic_units:
            return np.zeros(3)
        
        r_vec = positions[i] - positions[j]
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e-10:
            return np.zeros(3)
        
        force_magnitude = -self.G * self.masses[i] * self.masses[j] / r_mag**2
        force_direction = r_vec / r_mag
        
        return force_magnitude * force_direction
    
    def compute_total_force(self, i: int, positions: np.ndarray) -> np.ndarray:
        """
        Compute total force on particle i.
        """
        total_force = np.zeros(3)
        for j in range(3):
            if i != j:
                total_force += self.compute_coulomb_force(i, j, positions)
                total_force += self.compute_gravitational_force(i, j, positions)
        return total_force
    
    def compute_potential_energy(self, positions: Optional[np.ndarray] = None) -> float:
        """
        Compute total potential energy.
        
        V = Σ_{i<j} [k_e * q_i * q_j / r_ij - G * m_i * m_j / r_ij]
        """
        if positions is None:
            positions = self.positions
        
        V = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                if r_ij > 1e-10:
                    # Coulomb potential
                    V += self.k_e * self.charges[i] * self.charges[j] / r_ij
                    # Gravitational potential (if included)
                    if self.include_gravity and not self.use_atomic_units:
                        V -= self.G * self.masses[i] * self.masses[j] / r_ij
        return V
    
    def compute_kinetic_energy(self, velocities: Optional[np.ndarray] = None) -> float:
        """
        Compute total kinetic energy.
        """
        if velocities is None:
            velocities = self.velocities
        
        T = 0.0
        for i in range(3):
            T += 0.5 * self.masses[i] * np.dot(velocities[i], velocities[i])
        return T
    
    def compute_total_energy(self) -> float:
        """Compute total energy E = T + V."""
        return self.compute_kinetic_energy() + self.compute_potential_energy()
    
    def compute_angular_momentum(self) -> np.ndarray:
        """Compute total angular momentum L = Σ_i r_i × p_i."""
        L = np.zeros(3)
        for i in range(3):
            p_i = self.masses[i] * self.velocities[i]
            L += np.cross(self.positions[i], p_i)
        return L
    
    def compute_electron_proton_distance(self) -> float:
        """Compute distance between electron and proton."""
        return np.linalg.norm(self.positions[0] - self.positions[1])
    
    def step_velocity_verlet(self, dt: float):
        """
        Perform one step using Velocity Verlet algorithm.
        """
        # Current accelerations
        accelerations = np.zeros((3, 3))
        for i in range(3):
            force = self.compute_total_force(i, self.positions)
            accelerations[i] = force / self.masses[i]
        
        # Half-step velocity
        self.velocities += 0.5 * dt * accelerations
        
        # Full-step position
        self.positions += dt * self.velocities
        
        # New accelerations
        new_accelerations = np.zeros((3, 3))
        for i in range(3):
            force = self.compute_total_force(i, self.positions)
            new_accelerations[i] = force / self.masses[i]
        
        # Complete velocity
        self.velocities += 0.5 * dt * new_accelerations
        
        self.time += dt
    
    def step_rk4(self, dt: float):
        """
        Perform one step using 4th-order Runge-Kutta.
        """
        def derivatives(pos, vel):
            acc = np.zeros((3, 3))
            for i in range(3):
                force = self.compute_total_force(i, pos)
                acc[i] = force / self.masses[i]
            return vel.copy(), acc
        
        k1_v, k1_a = derivatives(self.positions, self.velocities)
        
        pos2 = self.positions + 0.5 * dt * k1_v
        vel2 = self.velocities + 0.5 * dt * k1_a
        k2_v, k2_a = derivatives(pos2, vel2)
        
        pos3 = self.positions + 0.5 * dt * k2_v
        vel3 = self.velocities + 0.5 * dt * k2_a
        k3_v, k3_a = derivatives(pos3, vel3)
        
        pos4 = self.positions + dt * k3_v
        vel4 = self.velocities + dt * k3_a
        k4_v, k4_a = derivatives(pos4, vel4)
        
        self.positions += (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.velocities += (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        
        self.time += dt
    
    def run_simulation(self,
                       total_time: float,
                       dt: float,
                       method: str = 'verlet',
                       save_interval: int = 1) -> Dict:
        """
        Run the classical simulation.
        
        Parameters:
        -----------
        total_time : float
            Total simulation time
        dt : float
            Time step
        method : str
            'verlet' or 'rk4'
        save_interval : int
            Save every N steps
        
        Returns:
        --------
        dict
            Simulation results
        """
        start_time = time.perf_counter()
        
        n_steps = int(total_time / dt)
        step_func = self.step_velocity_verlet if method == 'verlet' else self.step_rk4
        
        self.trajectory = [self.positions.copy()]
        self.energies = [self.compute_total_energy()]
        times = [0.0]
        e_p_distances = [self.compute_electron_proton_distance()]
        
        for step in range(n_steps):
            step_func(dt)
            
            if (step + 1) % save_interval == 0:
                self.trajectory.append(self.positions.copy())
                self.energies.append(self.compute_total_energy())
                times.append(self.time)
                e_p_distances.append(self.compute_electron_proton_distance())
        
        end_time = time.perf_counter()
        self.computation_time = end_time - start_time
        
        self.trajectory = np.array(self.trajectory)
        self.energies = np.array(self.energies)
        times = np.array(times)
        e_p_distances = np.array(e_p_distances)
        
        energy_error = np.abs(self.energies - self.energies[0]) / (np.abs(self.energies[0]) + 1e-20)
        
        return {
            'method': 'classical',
            'integration': method,
            'trajectory': self.trajectory,
            'energies': self.energies,
            'times': times,
            'e_p_distances': e_p_distances,
            'total_time': total_time,
            'dt': dt,
            'n_steps': n_steps,
            'computation_time': self.computation_time,
            'initial_energy': self.energies[0],
            'final_energy': self.energies[-1],
            'energy_conservation_error': energy_error[-1],
            'max_energy_error': np.max(energy_error),
            'angular_momentum': self.compute_angular_momentum(),
            'final_e_p_distance': e_p_distances[-1],
            'avg_e_p_distance': np.mean(e_p_distances),
        }
    
    def analyze_orbit_stability(self) -> Dict:
        """
        Analyze the stability of the electron orbit.
        
        Classical prediction: Accelerating electron should radiate energy
        and spiral into nucleus. This is the "ultraviolet catastrophe"
        that led to quantum mechanics.
        """
        if len(self.energies) < 2:
            return {'stable': None, 'message': 'Not enough data'}
        
        energy_change = self.energies[-1] - self.energies[0]
        relative_change = energy_change / (np.abs(self.energies[0]) + 1e-20)
        
        # Check if electron is getting closer to proton
        distances = np.array([np.linalg.norm(pos[0] - pos[1]) 
                              for pos in self.trajectory])
        distance_trend = distances[-1] - distances[0]
        
        return {
            'initial_radius': distances[0],
            'final_radius': distances[-1],
            'radius_change': distance_trend,
            'energy_change': energy_change,
            'relative_energy_change': relative_change,
            'stable': np.abs(relative_change) < 0.01,
            'classical_issue': 'Classical model does not include radiation. '
                               'Real accelerating charges would radiate and spiral in.',
        }


if __name__ == "__main__":
    # Test the simulation
    sim = HydrogenLikeClassical(use_atomic_units=True)
    sim.initialize_state(configuration='bohr_orbit')
    
    print("Initial state:")
    print(f"  Electron position: {sim.positions[0]}")
    print(f"  Proton position: {sim.positions[1]}")
    print(f"  Neutron position: {sim.positions[2]}")
    print(f"  Electron-proton distance: {sim.compute_electron_proton_distance():.4f} Bohr")
    print(f"  Total energy: {sim.compute_total_energy():.6f} Hartree")
    print(f"  (Bohr model predicts E = -0.5 Hartree)")
    
    # Run simulation (1 orbital period ≈ 2π atomic time units)
    orbital_period = 2 * np.pi
    results = sim.run_simulation(total_time=10*orbital_period, dt=0.01, method='verlet')
    
    print(f"\nSimulation completed in {results['computation_time']:.4f} seconds")
    print(f"Energy conservation error: {results['energy_conservation_error']:.2e}")
    
    stability = sim.analyze_orbit_stability()
    print(f"\nOrbit stability analysis:")
    print(f"  Initial radius: {stability['initial_radius']:.4f} Bohr")
    print(f"  Final radius: {stability['final_radius']:.4f} Bohr")
    print(f"  Stable: {stability['stable']}")
    print(f"  Note: {stability['classical_issue']}")

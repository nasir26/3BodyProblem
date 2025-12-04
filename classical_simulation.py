"""
Classical Simulation Module

This module implements classical mechanics simulations for:
1. Three-electron system (with Coulomb repulsion)
2. Electron-proton-neutron system (with Coulomb and nuclear-like interactions)

Classical Model:
- Particles treated as point masses with definite positions and momenta
- Forces computed from potential energy gradients
- Motion evolved using Verlet integration (symplectic, energy-conserving)
- Coulomb interaction: V(r) = k_e * q1 * q2 / r
"""

import numpy as np
from scipy.integrate import odeint
import time
from constants import AtomicUnits as AU

class ClassicalParticle:
    """Represents a classical point particle."""
    
    def __init__(self, mass, charge, position, velocity):
        """
        Initialize a classical particle.
        
        Args:
            mass: Particle mass in atomic units
            charge: Particle charge in atomic units (e = 1)
            position: 3D position vector [x, y, z] in Bohr
            velocity: 3D velocity vector [vx, vy, vz] in a.u.
        """
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
    
    def kinetic_energy(self):
        """Calculate kinetic energy: T = 0.5 * m * v^2"""
        return 0.5 * self.mass * np.sum(self.velocity**2)
    
    def momentum(self):
        """Calculate momentum: p = m * v"""
        return self.mass * self.velocity


class ClassicalThreeElectronSystem:
    """
    Classical simulation of three electrons.
    
    Physical Model:
    - Three electrons with Coulomb repulsion
    - Confined by a harmonic potential (to prevent infinite dispersion)
    - V_total = Σᵢ 0.5*ω²*rᵢ² + Σᵢ<ⱼ 1/|rᵢ-rⱼ|
    
    Note: Pure classical electrons would accelerate infinitely apart.
    The harmonic confinement mimics an atom's nuclear attraction.
    """
    
    def __init__(self, omega=0.25):
        """
        Initialize the three-electron system.
        
        Args:
            omega: Harmonic confinement frequency (a.u.)
        """
        self.n_particles = 3
        self.masses = np.array([AU.m_e, AU.m_e, AU.m_e])
        self.charges = np.array([-1.0, -1.0, -1.0])  # Electrons have charge -e
        self.omega = omega
        
        # Initialize positions (triangular configuration)
        self.positions = np.array([
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3)/2, 0.0],
            [-0.5, -np.sqrt(3)/2, 0.0]
        ])
        
        # Initialize velocities (small random velocities)
        np.random.seed(42)
        self.velocities = 0.1 * (np.random.rand(3, 3) - 0.5)
        
        # Simulation results storage
        self.trajectory = []
        self.energies = []
        self.computation_time = 0.0
    
    def coulomb_potential(self, positions):
        """
        Calculate total Coulomb potential energy.
        
        V_coulomb = Σᵢ<ⱼ kₑ * qᵢ * qⱼ / |rᵢ - rⱼ|
        
        In atomic units with electrons: V = Σᵢ<ⱼ 1/rᵢⱼ
        (positive because electrons repel)
        """
        V = 0.0
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                if r_ij > 1e-10:  # Avoid singularity
                    # qi * qj = (-1)*(-1) = +1 for electron-electron
                    V += 1.0 / r_ij
        return V
    
    def harmonic_potential(self, positions):
        """
        Calculate harmonic confinement potential.
        
        V_harm = Σᵢ 0.5 * mᵢ * ω² * rᵢ²
        """
        V = 0.0
        for i in range(self.n_particles):
            r_squared = np.sum(positions[i]**2)
            V += 0.5 * self.masses[i] * self.omega**2 * r_squared
        return V
    
    def total_potential(self, positions):
        """Total potential energy."""
        return self.coulomb_potential(positions) + self.harmonic_potential(positions)
    
    def kinetic_energy(self, velocities):
        """
        Calculate total kinetic energy.
        
        T = Σᵢ 0.5 * mᵢ * vᵢ²
        """
        T = 0.0
        for i in range(self.n_particles):
            T += 0.5 * self.masses[i] * np.sum(velocities[i]**2)
        return T
    
    def total_energy(self, positions, velocities):
        """Total system energy (should be conserved)."""
        return self.kinetic_energy(velocities) + self.total_potential(positions)
    
    def compute_forces(self, positions):
        """
        Compute forces on all particles.
        
        F = -∇V
        
        Coulomb force: Fᵢⱼ = qᵢqⱼ * (rᵢ - rⱼ) / |rᵢ - rⱼ|³
        Harmonic force: Fᵢ = -mᵢω²rᵢ
        """
        forces = np.zeros((self.n_particles, 3))
        
        # Coulomb forces (electron-electron repulsion)
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j:
                    r_vec = positions[i] - positions[j]
                    r = np.linalg.norm(r_vec)
                    if r > 1e-10:
                        # Force on i due to j: F = qi*qj * r_vec / r³
                        # For electrons: qi*qj = 1, so repulsive
                        forces[i] += r_vec / r**3
        
        # Harmonic confinement forces
        for i in range(self.n_particles):
            forces[i] -= self.masses[i] * self.omega**2 * positions[i]
        
        return forces
    
    def velocity_verlet_step(self, dt):
        """
        Perform one step of velocity Verlet integration.
        
        This is a symplectic integrator that conserves energy well.
        
        Algorithm:
        1. v(t + dt/2) = v(t) + a(t) * dt/2
        2. r(t + dt) = r(t) + v(t + dt/2) * dt
        3. Compute a(t + dt) from r(t + dt)
        4. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        """
        # Current accelerations
        forces = self.compute_forces(self.positions)
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Half-step velocity update
        self.velocities += 0.5 * accelerations * dt
        
        # Full-step position update
        self.positions += self.velocities * dt
        
        # New accelerations
        forces = self.compute_forces(self.positions)
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Complete velocity update
        self.velocities += 0.5 * accelerations * dt
    
    def run_simulation(self, t_total=10.0, dt=0.01):
        """
        Run the classical simulation.
        
        Args:
            t_total: Total simulation time in atomic units
            dt: Time step
            
        Returns:
            dict with trajectory, energies, and timing information
        """
        n_steps = int(t_total / dt)
        
        self.trajectory = []
        self.energies = []
        
        start_time = time.time()
        
        for step in range(n_steps):
            # Store current state
            self.trajectory.append(self.positions.copy())
            self.energies.append(self.total_energy(self.positions, self.velocities))
            
            # Integrate
            self.velocity_verlet_step(dt)
        
        self.computation_time = time.time() - start_time
        
        self.trajectory = np.array(self.trajectory)
        self.energies = np.array(self.energies)
        
        return {
            'trajectory': self.trajectory,
            'energies': self.energies,
            'computation_time': self.computation_time,
            'n_steps': n_steps,
            'dt': dt
        }
    
    def get_average_properties(self):
        """Calculate time-averaged properties from the simulation."""
        if len(self.trajectory) == 0:
            return None
        
        # Average distances between electrons
        avg_distances = []
        for positions in self.trajectory:
            for i in range(self.n_particles):
                for j in range(i + 1, self.n_particles):
                    avg_distances.append(np.linalg.norm(positions[i] - positions[j]))
        
        return {
            'mean_energy': np.mean(self.energies),
            'energy_std': np.std(self.energies),
            'mean_inter_electron_distance': np.mean(avg_distances),
            'energy_conservation': np.abs(self.energies[-1] - self.energies[0]) / np.abs(self.energies[0])
        }


class ClassicalEPNSystem:
    """
    Classical simulation of electron-proton-neutron system.
    
    Physical Model:
    - Electron: mass = mₑ, charge = -e
    - Proton: mass = mₚ ≈ 1836 mₑ, charge = +e
    - Neutron: mass = mₙ ≈ 1839 mₑ, charge = 0
    
    Interactions:
    - Coulomb: electron-proton attraction
    - Nuclear: Short-range strong force between proton-neutron
      (modeled as Yukawa potential)
    
    Note: This is a simplified model. In reality, quantum mechanics
    is essential for nuclear binding.
    """
    
    def __init__(self):
        """Initialize the electron-proton-neutron system."""
        self.n_particles = 3
        self.masses = np.array([AU.m_e, AU.m_p, AU.m_n])
        self.charges = np.array([-1.0, 1.0, 0.0])  # electron, proton, neutron
        
        # Labels for clarity
        self.labels = ['electron', 'proton', 'neutron']
        
        # Nuclear force parameters (Yukawa potential)
        # V_nuclear = -V0 * exp(-r/a) / (r/a)
        # Scaled to atomic units
        self.V0_nuclear = 100.0  # Depth in Hartree (simplified)
        self.a_nuclear = 0.05   # Range in Bohr (≈ 2.6 fm, nuclear scale)
        
        # Initialize positions
        # Electron orbiting around proton-neutron nucleus
        self.positions = np.array([
            [2.0, 0.0, 0.0],      # Electron at ~2 Bohr
            [0.0, 0.01, 0.0],     # Proton at origin
            [0.0, -0.01, 0.0]     # Neutron very close to proton
        ])
        
        # Initialize velocities
        # Electron needs orbital velocity for stable orbit
        v_orbital = np.sqrt(1.0 / 2.0)  # v = sqrt(k*e²/(m*r))
        self.velocities = np.array([
            [0.0, v_orbital, 0.0],  # Electron orbital motion
            [0.0, 0.0, 0.0],        # Proton nearly stationary
            [0.0, 0.0, 0.0]         # Neutron nearly stationary
        ])
        
        # Results storage
        self.trajectory = []
        self.energies = []
        self.computation_time = 0.0
    
    def coulomb_potential(self, positions):
        """
        Calculate Coulomb potential energy.
        
        V = Σᵢ<ⱼ kₑ * qᵢ * qⱼ / rᵢⱼ
        
        Only electron-proton interaction is non-zero.
        """
        V = 0.0
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                if abs(self.charges[i] * self.charges[j]) > 1e-10:
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r > 1e-10:
                        V += self.charges[i] * self.charges[j] / r
        return V
    
    def nuclear_potential(self, positions):
        """
        Calculate nuclear potential (proton-neutron).
        
        Yukawa potential: V = -V₀ * exp(-r/a) / (r/a)
        
        This models the residual strong force.
        """
        # Proton is index 1, neutron is index 2
        r_pn = np.linalg.norm(positions[1] - positions[2])
        if r_pn < 1e-10:
            return -self.V0_nuclear * 20  # Cap at close approach
        
        x = r_pn / self.a_nuclear
        V = -self.V0_nuclear * np.exp(-x) / x
        return V
    
    def total_potential(self, positions):
        """Total potential energy."""
        return self.coulomb_potential(positions) + self.nuclear_potential(positions)
    
    def kinetic_energy(self, velocities):
        """Total kinetic energy."""
        T = 0.0
        for i in range(self.n_particles):
            T += 0.5 * self.masses[i] * np.sum(velocities[i]**2)
        return T
    
    def total_energy(self, positions, velocities):
        """Total system energy."""
        return self.kinetic_energy(velocities) + self.total_potential(positions)
    
    def compute_forces(self, positions):
        """
        Compute forces on all particles.
        
        Coulomb force: F = q₁q₂ * r̂ / r²
        Yukawa force: F = -∇V_yukawa
        """
        forces = np.zeros((self.n_particles, 3))
        
        # Coulomb forces
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j and abs(self.charges[i] * self.charges[j]) > 1e-10:
                    r_vec = positions[i] - positions[j]
                    r = np.linalg.norm(r_vec)
                    if r > 1e-10:
                        # F = qi*qj * r_vec / r³
                        forces[i] += self.charges[i] * self.charges[j] * r_vec / r**3
        
        # Nuclear force (Yukawa) between proton and neutron
        r_vec = positions[1] - positions[2]  # proton - neutron
        r = np.linalg.norm(r_vec)
        if r > 1e-10:
            x = r / self.a_nuclear
            # F = -dV/dr * r̂ = -V₀/a * (1/x + 1) * exp(-x) / x * r̂
            # F points toward other particle (attractive)
            f_magnitude = self.V0_nuclear / self.a_nuclear * (1/x + 1) * np.exp(-x) / x
            f_direction = r_vec / r
            forces[1] -= f_magnitude * f_direction  # On proton, toward neutron
            forces[2] += f_magnitude * f_direction  # On neutron, toward proton
        
        return forces
    
    def velocity_verlet_step(self, dt):
        """Velocity Verlet integration step."""
        forces = self.compute_forces(self.positions)
        accelerations = forces / self.masses[:, np.newaxis]
        
        self.velocities += 0.5 * accelerations * dt
        self.positions += self.velocities * dt
        
        forces = self.compute_forces(self.positions)
        accelerations = forces / self.masses[:, np.newaxis]
        
        self.velocities += 0.5 * accelerations * dt
    
    def run_simulation(self, t_total=50.0, dt=0.001):
        """
        Run the classical simulation.
        
        Note: Smaller dt needed due to nuclear force stiffness.
        """
        n_steps = int(t_total / dt)
        
        self.trajectory = []
        self.energies = []
        
        start_time = time.time()
        
        for step in range(n_steps):
            self.trajectory.append(self.positions.copy())
            self.energies.append(self.total_energy(self.positions, self.velocities))
            self.velocity_verlet_step(dt)
        
        self.computation_time = time.time() - start_time
        
        self.trajectory = np.array(self.trajectory)
        self.energies = np.array(self.energies)
        
        return {
            'trajectory': self.trajectory,
            'energies': self.energies,
            'computation_time': self.computation_time,
            'n_steps': n_steps,
            'dt': dt
        }
    
    def get_average_properties(self):
        """Calculate time-averaged properties."""
        if len(self.trajectory) == 0:
            return None
        
        # Electron-proton distance (should be ~1-2 Bohr for H-like)
        ep_distances = [np.linalg.norm(pos[0] - pos[1]) for pos in self.trajectory]
        
        # Proton-neutron distance (should be very small, ~fm scale)
        pn_distances = [np.linalg.norm(pos[1] - pos[2]) for pos in self.trajectory]
        
        return {
            'mean_energy': np.mean(self.energies),
            'energy_std': np.std(self.energies),
            'mean_electron_proton_distance': np.mean(ep_distances),
            'mean_proton_neutron_distance': np.mean(pn_distances),
            'energy_conservation': np.abs(self.energies[-1] - self.energies[0]) / (np.abs(self.energies[0]) + 1e-10)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("CLASSICAL SIMULATION - THREE ELECTRON SYSTEM")
    print("=" * 60)
    
    system_3e = ClassicalThreeElectronSystem(omega=0.25)
    results_3e = system_3e.run_simulation(t_total=10.0, dt=0.01)
    props_3e = system_3e.get_average_properties()
    
    print(f"Computation time: {results_3e['computation_time']:.4f} s")
    print(f"Number of steps: {results_3e['n_steps']}")
    print(f"Mean energy: {props_3e['mean_energy']:.6f} Hartree")
    print(f"Energy std: {props_3e['energy_std']:.6f} Hartree")
    print(f"Energy conservation: {props_3e['energy_conservation']:.2e}")
    print(f"Mean inter-electron distance: {props_3e['mean_inter_electron_distance']:.4f} Bohr")
    
    print("\n" + "=" * 60)
    print("CLASSICAL SIMULATION - ELECTRON-PROTON-NEUTRON SYSTEM")
    print("=" * 60)
    
    system_epn = ClassicalEPNSystem()
    results_epn = system_epn.run_simulation(t_total=20.0, dt=0.0005)
    props_epn = system_epn.get_average_properties()
    
    print(f"Computation time: {results_epn['computation_time']:.4f} s")
    print(f"Number of steps: {results_epn['n_steps']}")
    print(f"Mean energy: {props_epn['mean_energy']:.6f} Hartree")
    print(f"Energy std: {props_epn['energy_std']:.6f} Hartree")
    print(f"Energy conservation: {props_epn['energy_conservation']:.2e}")
    print(f"Mean e-p distance: {props_epn['mean_electron_proton_distance']:.4f} Bohr")
    print(f"Mean p-n distance: {props_epn['mean_proton_neutron_distance']:.4f} Bohr")

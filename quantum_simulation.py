"""
Quantum Simulation Module

This module implements quantum mechanical simulations for:
1. Three-electron system (using Variational Monte Carlo)
2. Electron-proton-neutron system (using Born-Oppenheimer approximation)

Quantum Model:
- Particles described by wave functions |Ψ⟩
- Governed by Schrödinger equation: Ĥ|Ψ⟩ = E|Ψ⟩
- Observables are expectation values: ⟨Ψ|Ô|Ψ⟩
- Fermions (electrons) require antisymmetric wave functions (Slater determinants)

Methods:
- Variational Monte Carlo (VMC) for ground state energy
- Hartree-Fock for mean-field solution
- Numerical integration for 1D reduced problems
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.special import factorial
import time
from constants import AtomicUnits as AU, HARTREE_TO_EV


class QuantumThreeElectronSystem:
    """
    Quantum simulation of three electrons using Variational Monte Carlo.
    
    Trial Wave Function:
    Ψ(r₁,r₂,r₃) = Φ(r₁,r₂,r₃) × J(r₁₂,r₁₃,r₂₃)
    
    where:
    - Φ is a Slater determinant of single-particle orbitals
    - J is a Jastrow correlation factor: J = exp(Σᵢ<ⱼ u(rᵢⱼ))
    
    For harmonically confined electrons (quantum dot model):
    - Single-particle orbitals: φₙ(r) ~ exp(-αr²) × Hₙ(√α r)
    - Jastrow: u(r) = ar/(1+br) (Padé form satisfying cusp condition)
    
    The cusp condition ensures correct behavior when electrons approach each other:
    ∂Ψ/∂rᵢⱼ|ᵣᵢⱼ→0 = Ψ/2 (for same-spin electrons)
    """
    
    def __init__(self, omega=0.25):
        """
        Initialize the quantum three-electron system.
        
        Args:
            omega: Harmonic confinement frequency (a.u.)
        """
        self.omega = omega
        self.n_electrons = 3
        
        # Variational parameters
        # alpha: orbital width, beta: Jastrow parameter
        self.params = {'alpha': omega / 2, 'beta': 0.5}
        
        # VMC parameters
        self.n_walkers = 1000
        self.n_steps = 5000
        self.n_equilibration = 1000
        
        # Results storage
        self.energies = []
        self.positions_history = []
        self.computation_time = 0.0
        
        # Analytical results for comparison
        self.non_interacting_energy = 3 * 1.5 * omega  # 3 electrons × (3/2)ℏω
    
    def single_particle_orbital(self, r, n, alpha):
        """
        Gaussian orbital: φ(r) = exp(-α|r|²)
        
        For ground state harmonic oscillator: n=0 for all three electrons
        (actually two with same spin, one different - but simplified here)
        """
        r_squared = np.sum(r**2)
        return np.exp(-alpha * r_squared)
    
    def slater_determinant(self, positions, alpha):
        """
        Calculate Slater determinant for antisymmetric wave function.
        
        For three electrons, we use three orbitals:
        - φ₀(r) = exp(-αr²) (s-orbital)
        - φ₁(r) = x·exp(-αr²) (p-orbital)
        - φ₂(r) = y·exp(-αr²) (p-orbital)
        
        Det|φᵢ(rⱼ)| ensures antisymmetry.
        """
        # Build the Slater matrix
        slater_matrix = np.zeros((3, 3))
        
        for j, r in enumerate(positions):
            r_sq = np.sum(r**2)
            exp_factor = np.exp(-alpha * r_sq)
            
            slater_matrix[0, j] = exp_factor                    # s-orbital
            slater_matrix[1, j] = r[0] * exp_factor             # px-orbital
            slater_matrix[2, j] = r[1] * exp_factor             # py-orbital
        
        return np.linalg.det(slater_matrix)
    
    def jastrow_factor(self, positions, beta):
        """
        Jastrow correlation factor to capture electron-electron correlation.
        
        J = exp(Σᵢ<ⱼ u(rᵢⱼ))
        u(r) = r / (1 + βr)  (Padé-Jastrow form)
        
        This satisfies the cusp condition and reduces electron-electron
        repulsion energy.
        """
        u_sum = 0.0
        for i in range(self.n_electrons):
            for j in range(i + 1, self.n_electrons):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                u_sum += r_ij / (1 + beta * r_ij)
        
        return np.exp(u_sum)
    
    def trial_wavefunction(self, positions, alpha, beta):
        """
        Full trial wave function: Ψ = Φ × J
        
        Returns the wave function value (not normalized).
        """
        slater = self.slater_determinant(positions, alpha)
        jastrow = self.jastrow_factor(positions, beta)
        return slater * jastrow
    
    def local_energy(self, positions, alpha, beta):
        """
        Calculate local energy: E_L = (ĤΨ)/Ψ
        
        Hamiltonian: Ĥ = Σᵢ[-½∇ᵢ² + ½ω²rᵢ²] + Σᵢ<ⱼ 1/rᵢⱼ
        
        Using finite differences for kinetic energy.
        """
        psi = self.trial_wavefunction(positions, alpha, beta)
        if abs(psi) < 1e-15:
            return 0.0
        
        # Kinetic energy via finite difference Laplacian
        h = 0.001  # Step size
        kinetic = 0.0
        
        for i in range(self.n_electrons):
            for d in range(3):  # x, y, z
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, d] += h
                pos_minus[i, d] -= h
                
                psi_plus = self.trial_wavefunction(pos_plus, alpha, beta)
                psi_minus = self.trial_wavefunction(pos_minus, alpha, beta)
                
                # Laplacian: ∇²Ψ ≈ (Ψ₊ - 2Ψ + Ψ₋) / h²
                laplacian = (psi_plus - 2*psi + psi_minus) / h**2
                kinetic += -0.5 * laplacian / psi
        
        # Harmonic potential
        harmonic = 0.0
        for i in range(self.n_electrons):
            r_sq = np.sum(positions[i]**2)
            harmonic += 0.5 * self.omega**2 * r_sq
        
        # Coulomb repulsion
        coulomb = 0.0
        for i in range(self.n_electrons):
            for j in range(i + 1, self.n_electrons):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                if r_ij > 1e-10:
                    coulomb += 1.0 / r_ij
        
        return kinetic + harmonic + coulomb
    
    def metropolis_step(self, positions, alpha, beta, step_size=0.5):
        """
        Metropolis-Hastings Monte Carlo step.
        
        Samples from |Ψ|² distribution.
        """
        new_positions = positions + step_size * (np.random.rand(3, 3) - 0.5)
        
        psi_old = self.trial_wavefunction(positions, alpha, beta)
        psi_new = self.trial_wavefunction(new_positions, alpha, beta)
        
        # Acceptance probability: min(1, |Ψ_new|²/|Ψ_old|²)
        ratio = (psi_new / psi_old)**2 if abs(psi_old) > 1e-15 else 0.0
        
        if np.random.rand() < ratio:
            return new_positions, True
        return positions, False
    
    def variational_monte_carlo(self, alpha, beta):
        """
        Run VMC simulation with given parameters.
        
        Returns mean energy and statistical error.
        """
        # Initialize walkers
        positions = np.random.randn(self.n_walkers, 3, 3)
        
        # Equilibration
        for _ in range(self.n_equilibration):
            for w in range(self.n_walkers):
                positions[w], _ = self.metropolis_step(positions[w], alpha, beta)
        
        # Production run
        local_energies = []
        
        for step in range(self.n_steps):
            for w in range(self.n_walkers):
                positions[w], _ = self.metropolis_step(positions[w], alpha, beta)
                E_L = self.local_energy(positions[w], alpha, beta)
                local_energies.append(E_L)
        
        energies = np.array(local_energies)
        mean_energy = np.mean(energies)
        std_error = np.std(energies) / np.sqrt(len(energies))
        
        return mean_energy, std_error
    
    def optimize_parameters(self):
        """
        Optimize variational parameters to minimize energy.
        """
        def objective(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10
            energy, _ = self.variational_monte_carlo(alpha, beta)
            return energy
        
        # Simple grid search for stability
        best_energy = float('inf')
        best_params = [self.omega / 2, 0.5]
        
        for alpha in np.linspace(0.1, 0.5, 5):
            for beta in np.linspace(0.2, 1.0, 5):
                try:
                    E, _ = self.variational_monte_carlo(alpha, beta)
                    if E < best_energy:
                        best_energy = E
                        best_params = [alpha, beta]
                except:
                    pass
        
        self.params['alpha'] = best_params[0]
        self.params['beta'] = best_params[1]
        
        return best_params
    
    def run_simulation(self, optimize=False):
        """
        Run the full quantum simulation.
        
        Args:
            optimize: Whether to optimize variational parameters
        """
        start_time = time.time()
        
        if optimize:
            self.optimize_parameters()
        
        alpha = self.params['alpha']
        beta = self.params['beta']
        
        mean_energy, std_error = self.variational_monte_carlo(alpha, beta)
        
        self.computation_time = time.time() - start_time
        
        return {
            'ground_state_energy': mean_energy,
            'energy_error': std_error,
            'variational_params': self.params.copy(),
            'computation_time': self.computation_time,
            'non_interacting_energy': self.non_interacting_energy,
            'correlation_energy': mean_energy - self.non_interacting_energy
        }


class QuantumEPNSystem:
    """
    Quantum simulation of electron-proton-neutron system.
    
    Physical Model (Born-Oppenheimer Approximation):
    - Separate nuclear and electronic degrees of freedom
    - Nucleons (p,n) treated quasi-classically due to large mass
    - Electron treated fully quantum mechanically
    
    For the electronic part:
    - Solve hydrogen-like problem: [-½∇² - Z_eff/r]ψ = Eψ
    - Ground state: ψ₁ₛ(r) = (Z³/π)^(1/2) exp(-Zr)
    - Energy: E = -Z²/2 Hartree
    
    For the nuclear part:
    - Deuteron-like binding via Yukawa potential
    - Solve 1D radial equation numerically
    """
    
    def __init__(self):
        """Initialize the quantum e-p-n system."""
        # Particle properties
        self.m_e = AU.m_e
        self.m_p = AU.m_p  # ~1836
        self.m_n = AU.m_n  # ~1839
        
        # Reduced masses
        self.mu_ep = self.m_e * self.m_p / (self.m_e + self.m_p)  # ≈ m_e
        self.mu_pn = self.m_p * self.m_n / (self.m_p + self.m_n)  # ≈ m_p/2
        
        # Nuclear potential parameters
        self.V0_nuclear = 100.0  # Hartree (adjusted for atomic units)
        self.a_nuclear = 0.05   # Bohr (nuclear range)
        
        # Results storage
        self.computation_time = 0.0
        
        # Known values for hydrogen
        self.hydrogen_ground_state = -0.5  # Hartree (exactly -13.6 eV)
    
    def hydrogen_wavefunction(self, r, Z=1):
        """
        Hydrogen 1s wave function.
        
        ψ₁ₛ(r) = (Z³/π)^(1/2) × exp(-Zr)
        
        Normalized such that ∫|ψ|²d³r = 1
        """
        return np.sqrt(Z**3 / np.pi) * np.exp(-Z * r)
    
    def hydrogen_energy(self, Z=1):
        """
        Hydrogen ground state energy.
        
        E = -Z²/2 Hartree = -Z² × 13.6 eV
        """
        return -Z**2 / 2
    
    def radial_hydrogen_probability(self, r, Z=1):
        """
        Radial probability density: P(r) = 4πr²|ψ|²
        """
        psi = self.hydrogen_wavefunction(r, Z)
        return 4 * np.pi * r**2 * psi**2
    
    def nuclear_potential(self, r):
        """
        Yukawa nuclear potential for proton-neutron.
        
        V(r) = -V₀ × exp(-r/a) / (r/a)
        """
        if r < 1e-10:
            return -self.V0_nuclear * 20
        x = r / self.a_nuclear
        return -self.V0_nuclear * np.exp(-x) / x
    
    def solve_nuclear_schrodinger(self, n_points=1000, r_max=1.0):
        """
        Solve the radial Schrödinger equation for p-n system.
        
        For s-wave (l=0):
        [-ℏ²/(2μ) × d²/dr² + V(r)] u(r) = E × u(r)
        
        where ψ(r) = u(r)/r
        
        Using finite difference method on a grid.
        """
        # Grid
        r = np.linspace(0.001, r_max, n_points)
        dr = r[1] - r[0]
        
        # Kinetic energy matrix (finite difference Laplacian)
        # -ℏ²/(2μ) × d²/dr² → -(1/2μ) × d²/dr²
        kinetic_coeff = 1 / (2 * self.mu_pn * dr**2)
        
        H = np.zeros((n_points, n_points))
        
        # Tridiagonal kinetic energy
        for i in range(n_points):
            H[i, i] = 2 * kinetic_coeff + self.nuclear_potential(r[i])
            if i > 0:
                H[i, i-1] = -kinetic_coeff
            if i < n_points - 1:
                H[i, i+1] = -kinetic_coeff
        
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = eigh(H)
        
        # Ground state
        E_ground = eigenvalues[0]
        psi_ground = eigenvectors[:, 0]
        
        # Normalize
        norm = np.sqrt(np.trapz(psi_ground**2, r))
        psi_ground /= norm
        
        return {
            'energy': E_ground,
            'wavefunction': psi_ground,
            'r_grid': r,
            'all_eigenvalues': eigenvalues[:5]
        }
    
    def electron_average_radius(self, Z=1):
        """
        Average radius ⟨r⟩ for hydrogen-like atom.
        
        ⟨r⟩ = (3/2) × a₀/Z = 3/(2Z) Bohr
        """
        return 3 / (2 * Z)
    
    def electron_momentum_uncertainty(self, Z=1):
        """
        Momentum uncertainty from ⟨p²⟩.
        
        ⟨p²⟩ = Z² (in atomic units)
        Δp ≈ Z
        """
        return Z
    
    def heisenberg_product(self, Z=1):
        """
        Verify Heisenberg uncertainty principle.
        
        ΔxΔp ≥ ℏ/2 = 0.5 (in atomic units)
        """
        delta_x = self.electron_average_radius(Z) / 2  # Rough estimate
        delta_p = self.electron_momentum_uncertainty(Z)
        return delta_x * delta_p
    
    def run_simulation(self):
        """
        Run the full quantum simulation.
        
        Returns comprehensive results for both electronic and nuclear parts.
        """
        start_time = time.time()
        
        # Electronic structure (hydrogen-like)
        Z_eff = 1.0  # Effective nuclear charge
        
        electron_energy = self.hydrogen_energy(Z_eff)
        avg_radius = self.electron_average_radius(Z_eff)
        
        # Verify with numerical integration
        r_grid = np.linspace(0.001, 20, 1000)
        psi_values = self.hydrogen_wavefunction(r_grid, Z_eff)
        
        # Calculate expectation values
        prob = 4 * np.pi * r_grid**2 * psi_values**2
        numerical_avg_r = np.trapz(r_grid * prob, r_grid)
        
        # Nuclear structure (p-n binding)
        nuclear_results = self.solve_nuclear_schrodinger()
        
        self.computation_time = time.time() - start_time
        
        # Total system energy
        total_energy = electron_energy + nuclear_results['energy']
        
        return {
            'electron_energy': electron_energy,
            'electron_energy_eV': electron_energy * HARTREE_TO_EV,
            'electron_avg_radius': avg_radius,
            'numerical_avg_radius': numerical_avg_r,
            'nuclear_binding_energy': nuclear_results['energy'],
            'nuclear_eigenvalues': nuclear_results['all_eigenvalues'],
            'total_energy': total_energy,
            'computation_time': self.computation_time,
            'heisenberg_product': self.heisenberg_product(Z_eff)
        }
    
    def get_probability_densities(self):
        """
        Return probability density functions for visualization.
        """
        r_electron = np.linspace(0.01, 10, 500)
        r_nuclear = np.linspace(0.001, 0.5, 500)
        
        # Electron radial probability
        psi_e = self.hydrogen_wavefunction(r_electron)
        P_electron = 4 * np.pi * r_electron**2 * psi_e**2
        
        # Nuclear radial probability
        nuclear_results = self.solve_nuclear_schrodinger()
        
        return {
            'r_electron': r_electron,
            'P_electron': P_electron,
            'r_nuclear': nuclear_results['r_grid'],
            'psi_nuclear': nuclear_results['wavefunction']
        }


class ExactDiagonalization:
    """
    Exact diagonalization for small systems in a finite basis.
    
    This provides a reference for accuracy comparison.
    For the harmonic quantum dot, we use the Fock-Darwin states.
    """
    
    def __init__(self, omega=0.25, n_basis=6):
        """
        Initialize exact diagonalization solver.
        
        Args:
            omega: Harmonic frequency
            n_basis: Number of single-particle basis states
        """
        self.omega = omega
        self.n_basis = n_basis
    
    def two_electron_quantum_dot(self):
        """
        Exact solution for 2 electrons in harmonic trap.
        
        Uses analytical result for comparison.
        Non-interacting: E = 2 × (3/2)ℏω = 3ω
        With interaction: E ≈ 3ω + 0.4ω (for ω = 0.25)
        """
        # Non-interacting part
        E_ni = 3 * self.omega
        
        # Approximate interaction correction (from perturbation theory)
        # First-order: ⟨Ψ|1/r₁₂|Ψ⟩ ≈ √(2ω/π) for ground state
        E_int = np.sqrt(2 * self.omega / np.pi)
        
        return {
            'non_interacting': E_ni,
            'interaction': E_int,
            'total': E_ni + E_int
        }


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM SIMULATION - THREE ELECTRON SYSTEM")
    print("=" * 60)
    
    system_3e = QuantumThreeElectronSystem(omega=0.25)
    
    # Use fewer samples for quick test
    system_3e.n_walkers = 200
    system_3e.n_steps = 1000
    system_3e.n_equilibration = 200
    
    results_3e = system_3e.run_simulation(optimize=False)
    
    print(f"Computation time: {results_3e['computation_time']:.4f} s")
    print(f"Ground state energy: {results_3e['ground_state_energy']:.6f} Hartree")
    print(f"Energy error: {results_3e['energy_error']:.6f} Hartree")
    print(f"Non-interacting energy: {results_3e['non_interacting_energy']:.6f} Hartree")
    print(f"Correlation energy: {results_3e['correlation_energy']:.6f} Hartree")
    print(f"Variational parameters: α={results_3e['variational_params']['alpha']:.4f}, "
          f"β={results_3e['variational_params']['beta']:.4f}")
    
    print("\n" + "=" * 60)
    print("QUANTUM SIMULATION - ELECTRON-PROTON-NEUTRON SYSTEM")
    print("=" * 60)
    
    system_epn = QuantumEPNSystem()
    results_epn = system_epn.run_simulation()
    
    print(f"Computation time: {results_epn['computation_time']:.4f} s")
    print(f"Electron energy: {results_epn['electron_energy']:.6f} Hartree "
          f"({results_epn['electron_energy_eV']:.4f} eV)")
    print(f"Expected H ground state: -0.5 Hartree (-13.6 eV)")
    print(f"Electron avg radius: {results_epn['electron_avg_radius']:.4f} Bohr "
          f"(numerical: {results_epn['numerical_avg_radius']:.4f})")
    print(f"Nuclear binding energy: {results_epn['nuclear_binding_energy']:.4f} Hartree")
    print(f"Total system energy: {results_epn['total_energy']:.4f} Hartree")
    print(f"Heisenberg product (ΔxΔp): {results_epn['heisenberg_product']:.4f} (should be ≥ 0.5)")

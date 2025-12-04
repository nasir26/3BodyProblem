"""Quantum baselines for the comparative study."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import time

import numpy as np

from . import constants, systems


@dataclass
class QuantumResult:
    name: str
    energy_hartree: float
    runtime: constants.RuntimeMetrics
    metadata: Dict[str, float | np.ndarray | Dict[str, float]]


def _second_derivative_matrix(points: int, dx: float) -> np.ndarray:
    diag = np.full(points, -2.0)
    off = np.ones(points - 1)
    lap = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
    return lap / (dx**2)


def _reduced_mass(m1: float, m2: float) -> float:
    return m1 * m2 / (m1 + m2)


def _hartree_potential(x: np.ndarray, density: np.ndarray, softening: float, dx: float) -> np.ndarray:
    result = np.zeros_like(density)
    for i, xi in enumerate(x):
        kernel = 1.0 / np.sqrt((xi - x) ** 2 + softening**2)
        result[i] = np.sum(kernel * density) * dx
    return result


def _run_harmonic_hartree(params: Dict[str, float | int | Tuple[int, ...]]) -> Tuple[float, Dict[str, float | np.ndarray]]:
    electrons = int(params.get("electrons", 1))
    omega = float(params.get("omega", 0.1))
    x_max = float(params.get("x_max", 10.0))
    grid_size = int(params.get("grid_size", 400))
    softening = float(params.get("softening", 0.2))
    mixing = float(params.get("mixing", 0.4))
    max_iter = int(params.get("max_iter", 100))
    tol = float(params.get("tol", 1e-7))
    occupations = tuple(int(v) for v in params.get("occupations", (electrons,)))

    x = np.linspace(-x_max, x_max, grid_size)
    dx = x[1] - x[0]
    lap = _second_derivative_matrix(grid_size, dx)
    kinetic = -(0.5 / constants.ELECTRON_MASS) * lap
    v_ext = 0.5 * constants.ELECTRON_MASS * (omega**2) * (x**2)

    hartree = np.zeros(grid_size)
    converged = False
    single_energies = None
    orbitals = None
    density = None

    for iteration in range(max_iter):
        h_matrix = kinetic + np.diag(v_ext + hartree)
        eigvals, eigvecs = np.linalg.eigh(h_matrix)
        orbitals = eigvecs[:, : len(occupations)]
        density = np.zeros(grid_size)
        single_energies = []
        for idx, occ in enumerate(occupations):
            vec = orbitals[:, idx]
            norm = np.sqrt(np.trapz(vec**2, x))
            psi = vec / max(norm, 1e-12)
            orbitals[:, idx] = psi
            density += occ * psi**2
            single_energies.append(eigvals[idx])
        new_hartree = _hartree_potential(x, density, softening, dx)
        if np.max(np.abs(new_hartree - hartree)) < tol:
            hartree = new_hartree
            converged = True
            break
        hartree = mixing * new_hartree + (1.0 - mixing) * hartree

    single_total = float(np.dot(single_energies, occupations))
    hartree_correction = 0.5 * float(np.trapz(density * hartree, x))
    total_energy = single_total - hartree_correction

    return total_energy, {
        "x": x,
        "density": density,
        "orbitals": orbitals,
        "single_particle_sum": single_total,
        "hartree_correction": hartree_correction,
        "converged": converged,
        "iterations": iteration + 1,
    }


def _solve_radial(
    mass: float,
    grid_points: int,
    r_max: float,
    potential: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    r = np.linspace(1e-6, r_max, grid_points)
    dr = r[1] - r[0]
    lap = _second_derivative_matrix(grid_points, dr)
    kinetic = -(0.5 / mass) * lap
    h_matrix = kinetic + np.diag(potential)
    eigvals, eigvecs = np.linalg.eigh(h_matrix)
    ground_energy = float(eigvals[0])
    wave = eigvecs[:, 0]
    norm = np.sqrt(np.trapz(wave**2, r))
    wave /= max(norm, 1e-12)
    return ground_energy, r, wave


def _run_epn_combo(params: Dict[str, float | int]) -> Tuple[float, Dict[str, float | np.ndarray]]:
    e_grid = int(params.get("electron_grid", 1200))
    e_r_max = float(params.get("electron_r_max", 60.0))
    e_soft = float(params.get("electron_softening", 1e-3))
    pn_grid = int(params.get("pn_grid", 1000))
    pn_r_max = float(params.get("pn_r_max", 5.0))
    pn_soft = float(params.get("pn_softening", 5e-4))
    yukawa_strength = float(params.get("yukawa_strength", 50.0))
    yukawa_range = float(params.get("yukawa_range", 0.25))

    r_e = np.linspace(1e-6, e_r_max, e_grid)
    v_e = -1.0 / np.sqrt(r_e**2 + e_soft**2)
    mu_ep = _reduced_mass(constants.ELECTRON_MASS, constants.PROTON_MASS)
    e_energy, r_e, psi_e = _solve_radial(mu_ep, e_grid, e_r_max, v_e)

    r_pn = np.linspace(1e-6, pn_r_max, pn_grid)
    v_pn = (
        -yukawa_strength
        * np.exp(-r_pn / yukawa_range)
        / np.sqrt(r_pn**2 + pn_soft**2)
    )
    mu_pn = _reduced_mass(constants.PROTON_MASS, constants.NEUTRON_MASS)
    pn_energy, r_pn, psi_pn = _solve_radial(mu_pn, pn_grid, pn_r_max, v_pn)

    total_energy = e_energy + pn_energy
    return total_energy, {
        "electron_energy": e_energy,
        "pn_energy": pn_energy,
        "r_e": r_e,
        "psi_e": psi_e,
        "r_pn": r_pn,
        "psi_pn": psi_pn,
    }


def simulate(config: systems.QuantumSystemConfig) -> QuantumResult:
    start = time.perf_counter()
    if config.solver == "harmonic_hartree":
        energy, metadata = _run_harmonic_hartree(config.params)
    elif config.solver == "epn_combo":
        energy, metadata = _run_epn_combo(config.params)
    else:
        raise ValueError(f"Unknown quantum solver '{config.solver}'.")
    runtime = constants.RuntimeMetrics(wall_time_s=float(time.perf_counter() - start))
    return QuantumResult(
        name=config.name,
        energy_hartree=energy,
        runtime=runtime,
        metadata=metadata,
    )

"""Classical molecular-style simulations for the target systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import time

import numpy as np

from . import constants, systems


@dataclass
class ClassicalResult:
    name: str
    average_energy_hartree: float
    std_energy_hartree: float
    runtime: constants.RuntimeMetrics
    metadata: Dict[str, np.ndarray | float | Dict[str, float]]


def _compute_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    masses: np.ndarray,
    config: systems.ClassicalSystemConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = len(positions)
    forces = np.zeros_like(positions)
    potentials = {"coulomb": 0.0, "trap": 0.0, "yukawa": 0.0}
    soft = config.coulomb_softening

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]
            distance = float(np.linalg.norm(r_vec))
            softened = np.sqrt(distance**2 + soft**2)
            if charges[i] == 0.0 and charges[j] == 0.0:
                continue
            strength = (
                constants.COULOMB_COEFFICIENT
                * charges[i]
                * charges[j]
                / (softened**3)
            )
            force = strength * r_vec
            forces[i] += force
            forces[j] -= force
            potentials["coulomb"] += (
                constants.COULOMB_COEFFICIENT
                * charges[i]
                * charges[j]
                / softened
            )

    if config.trap_omega is not None:
        omega2 = config.trap_omega**2
        trap_forces = -masses[:, None] * omega2 * positions
        forces += trap_forces
        trap_energy = 0.5 * masses * omega2 * np.sum(positions**2, axis=1)
        potentials["trap"] += float(np.sum(trap_energy))

    for term in config.yukawa_couplings:
        r_vec = positions[term.j] - positions[term.i]
        distance = float(np.linalg.norm(r_vec))
        softened = np.sqrt(distance**2 + soft**2)
        exp_part = np.exp(-distance / term.range_param)
        potentials["yukawa"] += -term.strength * exp_part / max(softened, 1e-12)
        prefactor = (
            -term.strength
            * exp_part
            * (1.0 / (term.range_param * max(softened, 1e-12)) + 1.0 / (softened**2))
        )
        if softened > 0:
            direction = r_vec / softened
        else:
            direction = np.zeros_like(r_vec)
        force_vec = prefactor * direction
        forces[term.i] += force_vec
        forces[term.j] -= force_vec

    return forces, potentials


def simulate(config: systems.ClassicalSystemConfig) -> ClassicalResult:
    dt = config.dt
    steps = int(config.total_time / dt)
    positions = np.array([p.position for p in config.particles], dtype=float)
    velocities = np.array([p.velocity for p in config.particles], dtype=float)
    masses = np.array([p.mass for p in config.particles], dtype=float)
    charges = np.array([p.charge for p in config.particles], dtype=float)

    kinetic = lambda vel: 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    energies = np.zeros(steps)
    potential_sums = {key: 0.0 for key in ["coulomb", "trap", "yukawa"]}

    start = time.perf_counter()
    forces, _ = _compute_forces(positions, charges, masses, config)
    accel = forces / masses[:, None]
    for step in range(steps):
        velocities += 0.5 * dt * accel
        positions += dt * velocities
        forces, potentials = _compute_forces(positions, charges, masses, config)
        accel = forces / masses[:, None]
        velocities += 0.5 * dt * accel

        total_potential = 0.0
        for key, value in potentials.items():
            potential_sums[key] += value
            total_potential += value
        energies[step] = kinetic(velocities) + total_potential

    runtime = constants.RuntimeMetrics(wall_time_s=float(time.perf_counter() - start))
    mean_energy = float(np.mean(energies[int(0.5 * steps) :]))
    std_energy = float(np.std(energies[int(0.5 * steps) :]))
    avg_potentials = {k: v / steps for k, v in potential_sums.items()}

    return ClassicalResult(
        name=config.name,
        average_energy_hartree=mean_energy,
        std_energy_hartree=std_energy,
        runtime=runtime,
        metadata={
            "energy_trace": energies,
            "potential_breakdown": avg_potentials,
            "dt": dt,
            "steps": steps,
        },
    )

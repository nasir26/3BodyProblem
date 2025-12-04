"""System definitions for the comparative study."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from . import constants


@dataclass(frozen=True)
class ParticleSpec:
    """Initial data for a classical particle."""

    name: str
    mass: float
    charge: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]


@dataclass(frozen=True)
class YukawaCoupling:
    """Strong-force inspired interaction between two classical particles."""

    i: int
    j: int
    strength: float
    range_param: float


@dataclass(frozen=True)
class ClassicalSystemConfig:
    name: str
    description: str
    total_time: float
    dt: float
    particles: Tuple[ParticleSpec, ...]
    coulomb_softening: float = 0.1
    trap_omega: float | None = None
    yukawa_couplings: Tuple[YukawaCoupling, ...] = ()


@dataclass(frozen=True)
class QuantumSystemConfig:
    name: str
    solver: str
    params: Dict[str, float | int | Sequence[float] | Dict[str, float | int | Sequence[float]]]


@dataclass(frozen=True)
class SimulationTarget:
    name: str
    description: str
    classical: ClassicalSystemConfig
    quantum: QuantumSystemConfig
    reference_quantum: QuantumSystemConfig


def _three_electron_particles() -> Tuple[ParticleSpec, ...]:
    radius = 1.5
    angles = np.deg2rad([0.0, 120.0, 240.0])
    positions = [
        (
            radius * float(np.cos(theta)),
            radius * float(np.sin(theta)),
            0.0,
        )
        for theta in angles
    ]
    return tuple(
        ParticleSpec(
            name=f"e{i+1}",
            mass=constants.ELECTRON_MASS,
            charge=-constants.ELEMENTARY_CHARGE,
            position=positions[i],
            velocity=(0.0, 0.0, 0.0),
        )
        for i in range(3)
    )


THREE_ELECTRON_TARGET = SimulationTarget(
    name="Penning-like three-electron trap",
    description=(
        "Three electrons confined by a weak harmonic trap. Classical dynamics "
        "are compared to a self-consistent Hartree treatment of the same trap."
    ),
    classical=ClassicalSystemConfig(
        name="three-electron-classical",
        description="Velocity-Verlet integration with harmonic confinement",
        total_time=800.0,
        dt=0.05,
        particles=_three_electron_particles(),
        coulomb_softening=0.2,
        trap_omega=0.05,
    ),
    quantum=QuantumSystemConfig(
        name="three-electron-hartree",
        solver="harmonic_hartree",
        params={
            "electrons": 3,
            "omega": 0.05,
            "x_max": 15.0,
            "grid_size": 512,
            "softening": 0.2,
            "mixing": 0.2,
            "max_iter": 160,
            "occupations": (2, 1),
        },
    ),
    reference_quantum=QuantumSystemConfig(
        name="three-electron-hartree-ref",
        solver="harmonic_hartree",
        params={
            "electrons": 3,
            "omega": 0.05,
            "x_max": 20.0,
            "grid_size": 1536,
            "softening": 0.15,
            "mixing": 0.2,
            "max_iter": 400,
            "occupations": (2, 1),
        },
    ),
)


def _epn_particles() -> Tuple[ParticleSpec, ...]:
    return (
        ParticleSpec(
            name="electron",
            mass=constants.ELECTRON_MASS,
            charge=-constants.ELEMENTARY_CHARGE,
            position=(-1.5, 0.0, 0.0),
            velocity=(0.0, 0.8, 0.0),
        ),
        ParticleSpec(
            name="proton",
            mass=constants.PROTON_MASS,
            charge=constants.ELEMENTARY_CHARGE,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
        ),
        ParticleSpec(
            name="neutron",
            mass=constants.NEUTRON_MASS,
            charge=0.0,
            position=(0.25, 0.0, 0.25),
            velocity=(0.0, -0.05, 0.05),
        ),
    )


EPN_TARGET = SimulationTarget(
    name="Hydrogen + spectator neutron",
    description=(
        "One electron, one proton, and one neutron. The classical model "
        "includes Coulomb and phenomenological Yukawa forces, while the "
        "quantum baseline solves separate radial Schrödinger problems "
        "for the electronic and nuclear subsystems."
    ),
    classical=ClassicalSystemConfig(
        name="epn-classical",
        description="Three-body integration with Coulomb + Yukawa forces",
        total_time=2.0,
        dt=0.0005,
        particles=_epn_particles(),
        coulomb_softening=0.05,
        trap_omega=None,
        yukawa_couplings=(
            YukawaCoupling(i=1, j=2, strength=45.0, range_param=0.2),
        ),
    ),
    quantum=QuantumSystemConfig(
        name="epn-radial",
        solver="epn_combo",
        params={
            "electron_grid": 1200,
            "electron_r_max": 60.0,
            "electron_softening": 1e-3,
            "pn_grid": 1000,
            "pn_r_max": 5.0,
            "pn_softening": 5e-4,
            "yukawa_strength": 55.0,
            "yukawa_range": 0.25,
        },
    ),
    reference_quantum=QuantumSystemConfig(
        name="epn-radial-ref",
        solver="epn_combo",
        params={
            "electron_grid": 2400,
            "electron_r_max": 80.0,
            "electron_softening": 5e-4,
            "pn_grid": 2000,
            "pn_r_max": 8.0,
            "pn_softening": 2.5e-4,
            "yukawa_strength": 55.0,
            "yukawa_range": 0.25,
        },
    ),
)


def all_targets() -> Tuple[SimulationTarget, ...]:
    return (THREE_ELECTRON_TARGET, EPN_TARGET)

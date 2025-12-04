"""Entry point for comparing classical and quantum simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import classical, constants, quantum, systems


@dataclass
class ComparisonRow:
    system_name: str
    description: str
    reference_energy_ev: float
    classical_energy_ev: float
    quantum_energy_ev: float
    classical_error_ev: float
    quantum_error_ev: float
    classical_runtime_s: float
    quantum_runtime_s: float

    @property
    def accuracy_winner(self) -> str:
        return "quantum" if self.quantum_error_ev < self.classical_error_ev else "classical"

    @property
    def speed_winner(self) -> str:
        return "quantum" if self.quantum_runtime_s < self.classical_runtime_s else "classical"


def _format_row(row: ComparisonRow) -> str:
    return (
        f"{row.system_name:<32}"
        f"{row.reference_energy_ev:>12.3f}"
        f"{row.classical_energy_ev:>12.3f}"
        f"{row.quantum_energy_ev:>12.3f}"
        f"{row.classical_error_ev:>12.3f}"
        f"{row.quantum_error_ev:>12.3f}"
        f"{row.classical_runtime_s:>10.3f}"
        f"{row.quantum_runtime_s:>10.3f}"
        f"{row.accuracy_winner:>12}"
        f"{row.speed_winner:>12}"
    )


def run() -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for target in systems.all_targets():
        reference = quantum.simulate(target.reference_quantum)
        quantum_result = quantum.simulate(target.quantum)
        classical_result = classical.simulate(target.classical)

        reference_ev = constants.hartree_to_ev(reference.energy_hartree)
        quantum_ev = constants.hartree_to_ev(quantum_result.energy_hartree)
        classical_ev = constants.hartree_to_ev(classical_result.average_energy_hartree)

        quantum_error = abs(quantum_result.energy_hartree - reference.energy_hartree)
        classical_error = abs(classical_result.average_energy_hartree - reference.energy_hartree)

        row = ComparisonRow(
            system_name=target.name,
            description=target.description,
            reference_energy_ev=reference_ev,
            classical_energy_ev=classical_ev,
            quantum_energy_ev=quantum_ev,
            classical_error_ev=constants.hartree_to_ev(classical_error),
            quantum_error_ev=constants.hartree_to_ev(quantum_error),
            classical_runtime_s=classical_result.runtime.wall_time_s,
            quantum_runtime_s=quantum_result.runtime.wall_time_s,
        )
        rows.append(row)

    header = (
        f"{'System':<32}"
        f"{'Ref (eV)':>12}"
        f"{'Class (eV)':>12}"
        f"{'Quant (eV)':>12}"
        f"{'Class err':>12}"
        f"{'Quant err':>12}"
        f"{'Class t(s)':>10}"
        f"{'Quant t':>10}"
        f"{'Acc win':>12}"
        f"{'Speed win':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(_format_row(row))

    for row in rows:
        print()
        print(f"System: {row.system_name}")
        print(f"  Description: {row.description}")
        print(f"  Accuracy winner: {row.accuracy_winner}")
        print(f"  Speed winner: {row.speed_winner}")

    return rows


def main() -> None:  # pragma: no cover - CLI helper
    run()


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()

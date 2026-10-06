"""Прямая плотная диагонализация XXX-цепочки без разбиения на сектора.

Все 2**N конфигураций битового базиса входят в одну матрицу. Матрица
строится обычными циклами, затем полностью диагонализуется.
Соглашения: J=1, S=σ/2, бит 1 — спин вверх.
"""

from __future__ import annotations

import numpy as np


def build_hamiltonian(n: int, periodic: bool) -> np.ndarray:
    """Построить полную матрицу H размером 2**N на 2**N."""
    if n < 2:
        raise ValueError("Требуется N >= 2")

    dimension = 2**n
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    bonds = n if periodic else n - 1

    for state in range(dimension):
        for site in range(bonds):
            next_site = (site + 1) % n
            spin = (state >> site) & 1
            next_spin = (state >> next_site) & 1

            if spin == next_spin:
                matrix[state, state] += 0.25
            else:
                matrix[state, state] -= 0.25
                flipped_state = state ^ (1 << site) ^ (1 << next_site)
                matrix[state, flipped_state] += 0.5

    return matrix


def ground_state(n: int, periodic: bool) -> dict[str, float | int | str]:
    """Вычислить полный спектр и вернуть состояние с минимальной энергией."""
    matrix = build_hamiltonian(n, periodic)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    energy = float(eigenvalues[0])
    vector = eigenvectors[:, 0]
    residual = float(np.linalg.norm(matrix @ vector - energy * vector))
    return {
        "N": n,
        "boundary": "PBC" if periodic else "OBC",
        "dimension": matrix.shape[0],
        "E0": energy,
        "e0": energy / n,
        "residual": residual,
    }

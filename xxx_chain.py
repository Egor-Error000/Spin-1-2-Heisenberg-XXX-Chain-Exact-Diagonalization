"""Sparse exact diagonalization of the spin-1/2 antiferromagnetic XXX chain.

Site i is bit i. Bit 1 denotes spin up; S = sigma/2 and J = 1.
"""

from __future__ import annotations

from math import comb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from tqdm.auto import tqdm


@njit(cache=True)
def _popcount(value: int) -> int:
    count = 0
    while value:
        count += value & 1
        value >>= 1
    return count


@njit(cache=True)
def _fill_sector(states: np.ndarray, n: int, n_up: int) -> None:
    index = 0
    for state in range(1 << n):
        if _popcount(state) == n_up:
            states[index] = state
            index += 1


def sector_basis(n: int, n_up: int) -> tuple[np.ndarray, np.ndarray]:
    """Return bit states and a full-state to sector-index lookup table."""
    if not (0 <= n_up <= n and 2 <= n <= 26):
        raise ValueError("Require 2 <= n <= 26 and 0 <= n_up <= n")
    states = np.empty(comb(n, n_up), dtype=np.int32)
    _fill_sector(states, n, n_up)
    lookup = np.full(1 << n, -1, dtype=np.int32)
    lookup[states] = np.arange(len(states), dtype=np.int32)
    return states, lookup


@njit(cache=True)
def _chunk_triplets(
    states: np.ndarray,
    lookup: np.ndarray,
    start: int,
    stop: int,
    n: int,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bonds = n if periodic else n - 1
    capacity = (stop - start) * (bonds + 1)
    rows = np.empty(capacity, dtype=np.int32)
    cols = np.empty(capacity, dtype=np.int32)
    data = np.empty(capacity, dtype=np.float64)
    used = 0
    for column in range(start, stop):
        state = states[column]
        diagonal = 0.0
        for i in range(bonds):
            j = (i + 1) % n
            if ((state >> i) & 1) == ((state >> j) & 1):
                diagonal += 0.25
            else:
                diagonal -= 0.25
                flipped = state ^ (1 << i) ^ (1 << j)
                rows[used] = lookup[flipped]
                cols[used] = column
                data[used] = 0.5
                used += 1
        rows[used] = column
        cols[used] = column
        data[used] = diagonal
        used += 1
    return rows[:used], cols[:used], data[:used]


def build_hamiltonian(
    n: int,
    periodic: bool,
    n_up: int | None = None,
    *,
    progress: bool = False,
    chunk_size: int = 1024,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build H in a fixed-Sz sector; returns H and its ordered bit states."""
    if n_up is None:
        if n % 2:
            raise ValueError("Half filling requires even n")
        n_up = n // 2
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    states, lookup = sector_basis(n, n_up)
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    starts = range(0, len(states), chunk_size)
    if progress and len(states) > chunk_size:
        starts = tqdm(starts, total=(len(states) + chunk_size - 1) // chunk_size,
                      desc=f"H N={n} {'PBC' if periodic else 'OBC'}", leave=False,
                      mininterval=0.5)
    for start in starts:
        rows, cols, data = _chunk_triplets(
            states, lookup, start, min(start + chunk_size, len(states)), n, periodic
        )
        rows_parts.append(rows)
        cols_parts.append(cols)
        data_parts.append(data)
    shape = (len(states), len(states))
    matrix = sparse.coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
        shape=shape,
        dtype=np.float64,
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix, states


def lowest_eigenpair(
    matrix: sparse.csr_matrix, *, seed: int = 2026, tolerance: float = 1e-11,
    progress: bool = False, label: str = "eigsh",
) -> tuple[float, np.ndarray, float]:
    """Return lowest eigenvalue, normalized vector, and absolute residual."""
    if matrix.shape[0] == 1:
        return float(matrix[0, 0]), np.ones(1), 0.0
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal(matrix.shape[0])
    if progress:
        with tqdm(desc=label, unit="matvec", mininterval=0.5, leave=False) as bar:
            def matvec(vector: np.ndarray) -> np.ndarray:
                bar.update(1)
                return matrix @ vector

            operator = LinearOperator(matrix.shape, matvec=matvec, dtype=np.float64)
            values, vectors = eigsh(operator, k=1, which="SA", v0=initial,
                                    tol=tolerance)
    else:
        values, vectors = eigsh(matrix, k=1, which="SA", v0=initial,
                                tol=tolerance)
    energy = float(values[0])
    vector = vectors[:, 0]
    residual = float(np.linalg.norm(matrix @ vector - energy * vector))
    return energy, vector, residual


def ground_state(
    n: int, periodic: bool, *, progress: bool = False, repeat: bool = False
) -> dict[str, float | int | str]:
    """Compute the half-filled ground energy and numerical diagnostics."""
    matrix, _ = build_hamiltonian(n, periodic, progress=progress)
    energy, vector, residual = lowest_eigenpair(
        matrix, progress=progress and n >= 20,
        label=f"eigsh N={n} {'PBC' if periodic else 'OBC'}",
    )
    agreement = np.nan
    if repeat:
        second, _, second_residual = lowest_eigenpair(matrix, seed=271828)
        agreement = abs(energy - second)
        residual = max(residual, second_residual)
    return {
        "N": n,
        "boundary": "PBC" if periodic else "OBC",
        "dimension": matrix.shape[0],
        "E0": energy,
        "e0": energy / n,
        "residual": residual,
        "norm_error": abs(float(np.linalg.norm(vector)) - 1.0),
        "repeat_delta": agreement,
    }


def compute_grid(
    sizes: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16),
    *,
    workers: int = 1,
    progress: bool = True,
    repeat_up_to: int = 16,
) -> list[dict[str, float | int | str]]:
    """Compute both boundaries; independent cases can use separate processes.

    Workers run with one BLAS/Numba thread each (set in compose.yaml), so a
    requested worker count of at most 14 never oversubscribes the CPU budget.
    """
    if not 1 <= workers <= 14:
        raise ValueError("workers must be between 1 and 14")
    cases = [(n, periodic) for n in sizes for periodic in (False, True)]
    if workers == 1:
        iterator = tqdm(cases, desc="Цепочки", mininterval=0.5) if progress else cases
        return [ground_state(n, periodic, progress=progress,
                             repeat=n <= repeat_up_to)
                for n, periodic in iterator]
    results: list[dict[str, float | int | str]] = []
    # Spawn works for a Jupyter kernel and for scripts on Windows and Linux.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(ground_state, n, periodic, progress=False,
                               repeat=n <= repeat_up_to)
                   for n, periodic in cases]
        iterator = tqdm(as_completed(futures), total=len(futures),
                        desc=f"Цепочки ({workers} процесса)",
                        mininterval=0.5) if progress else as_completed(futures)
        for future in iterator:
            results.append(future.result())
    return sorted(results, key=lambda row: (int(row["N"]), str(row["boundary"])))

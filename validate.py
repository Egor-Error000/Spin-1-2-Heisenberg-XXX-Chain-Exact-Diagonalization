"""Independent dense reference checks and numerical diagnostics.

Run with: docker compose run --rm notebook python validate.py
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from xxx_chain import build_hamiltonian, lowest_eigenpair


def pauli_reference(n: int, periodic: bool) -> np.ndarray:
    """Construct H from tensor products of Sx, Sy, Sz, independently of bits."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.diag([-0.5, 0.5]).astype(complex)
    identity = np.eye(2, dtype=complex)
    result = np.zeros((1 << n, 1 << n), dtype=complex)
    for i in range(n if periodic else n - 1):
        j = (i + 1) % n
        for spin in (sx, sy, sz):
            factors = [spin if site in (i, j) else identity
                       for site in range(n - 1, -1, -1)]
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            result += term
    return result


def check_dense_reference() -> None:
    for n in tqdm((4, 6, 8), desc="Независимый эталон Паули"):
        for periodic in (False, True):
            reference = pauli_reference(n, periodic)
            assert np.max(abs(reference.imag)) < 1e-13
            assert np.max(abs(reference - reference.conj().T)) < 1e-13
            magnetization = np.array([state.bit_count() - n / 2
                                      for state in range(1 << n)])
            commutator = (magnetization[:, None] - magnetization[None, :]) * reference
            assert np.max(abs(commutator)) < 1e-13

            block, states = build_hamiltonian(n, periodic, progress=False)
            projected = reference[np.ix_(states, states)]
            assert np.max(abs(projected - block.toarray())) < 1e-13
            block_spectrum = np.linalg.eigvalsh(projected)
            sparse_energy, _, residual = lowest_eigenpair(block)
            assert abs(sparse_energy - block_spectrum[0]) < 1e-10
            assert residual < 1e-9

            sector_minima = []
            for n_up in range(n + 1):
                sector = np.flatnonzero(magnetization == n_up - n / 2)
                sector_minima.append(float(np.linalg.eigvalsh(
                    reference[np.ix_(sector, sector)])[0]))
            assert abs(min(sector_minima) - block_spectrum[0]) < 1e-10
            assert abs(float(np.linalg.eigvalsh(reference)[0]) - block_spectrum[0]) < 1e-10

            bonds = n if periodic else n - 1
            all_up = (1 << n) - 1
            assert abs(reference[all_up, all_up].real - bonds / 4) < 1e-13
            assert np.count_nonzero(abs(reference[:, all_up]) > 1e-13) == 1
            if n == 4 and periodic:
                assert abs(sparse_energy + 2.0) < 1e-11


def check_working_sizes() -> None:
    cases = [(n, pbc) for n in (4, 6, 8, 10, 12, 14, 16)
             for pbc in (False, True)]
    for n, periodic in tqdm(cases, desc="Остатки и повторы eigsh"):
        matrix, _ = build_hamiltonian(n, periodic, progress=False)
        difference = matrix - matrix.T
        assert difference.nnz == 0 or np.max(abs(difference.data)) < 1e-13
        first, vector, residual = lowest_eigenpair(matrix, seed=2026)
        second, _, residual2 = lowest_eigenpair(matrix, seed=271828)
        assert abs(first - second) < 1e-9, (n, periodic, first, second)
        assert residual < 1e-8 and residual2 < 1e-8, (n, periodic, residual, residual2)
        assert abs(np.linalg.norm(vector) - 1) < 1e-12
        assert abs(float(vector @ (matrix @ vector)) - first) < 1e-9
        assert first < 0


def main() -> None:
    check_dense_reference()
    check_working_sizes()
    print("Проверки пройдены: независимый эталон, все сектора N<=8 и N<=16.")


if __name__ == "__main__":
    main()

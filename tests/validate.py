"""Independent dense reference checks and numerical diagnostics.

Run with: docker compose -f docker/compose.yaml run --rm notebook python -m unittest discover -s tests -v
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from xxx_chain import (
    METHOD_CSR,
    METHOD_SU2,
    METHOD_SZ,
    SingletHamiltonian,
    SzHamiltonian,
    build_hamiltonian,
    ground_state,
    lowest_eigenpair,
    singlet_dimension,
)


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
    for n in tqdm(range(3, 9), desc="Независимый эталон Паули"):
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
            if n == 3:
                assert abs(sparse_energy - (-0.75 if periodic else -1.0)) < 1e-11


def check_working_sizes() -> None:
    cases = [(n, pbc) for n in range(3, 17)
             for pbc in (False, True)]
    for n, periodic in tqdm(cases, desc="Все сектора, остатки и повторы eigsh"):
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
        sector_minima = []
        for n_up in range(n + 1):
            sector, _ = build_hamiltonian(n, periodic, n_up=n_up)
            energy, _, sector_residual = lowest_eigenpair(sector)
            assert sector_residual < 1e-8
            sector_minima.append(energy)
        assert abs(first - min(sector_minima)) < 1e-9
        if n % 2:
            assert abs(sector_minima[n // 2] - sector_minima[n // 2 + 1]) < 1e-9


def _operator_matrix(matvec, dimension: int) -> np.ndarray:
    columns = []
    basis = np.zeros(dimension)
    for index in range(dimension):
        basis[index] = 1.0
        columns.append(matvec(basis))
        basis[index] = 0.0
    return np.column_stack(columns)


def singlet_energies_from_pauli(n: int, periodic: bool) -> np.ndarray:
    """Eigenvalues of the Pauli Hamiltonian inside the Sz=0, S=0 subspace."""
    reference = pauli_reference(n, periodic).real
    n_up = n // 2
    states = [state for state in range(1 << n) if state.bit_count() == n_up]
    high_states = [state for state in range(1 << n) if state.bit_count() == n_up + 1]
    high_index = {state: index for index, state in enumerate(high_states)}
    raising = np.zeros((len(high_states), len(states)))
    for column, state in enumerate(states):
        for site in range(n):
            if (state >> site) & 1 == 0:
                raising[high_index[state | (1 << site)], column] += 1.0
    spin_squared = raising.T @ raising
    eigenvalues, eigenvectors = np.linalg.eigh(spin_squared)
    selected = eigenvalues < 1e-8
    if int(np.count_nonzero(selected)) != singlet_dimension(n):
        raise AssertionError((n, int(np.count_nonzero(selected)), singlet_dimension(n)))
    subspace = eigenvectors[:, selected]
    projected = reference[np.ix_(states, states)]
    return np.linalg.eigvalsh(subspace.T @ projected @ subspace)


def check_matrix_free_operators() -> None:
    """Match both new actions to the Pauli matrix and to CSR for N <= 10."""
    rng = np.random.default_rng(2026)
    for n in range(3, 11):
        for periodic in (False, True):
            matrix, states = build_hamiltonian(n, periodic)
            reference = pauli_reference(n, periodic).real
            projected = reference[np.ix_(states, states)]
            vector = rng.standard_normal(matrix.shape[0])
            sz_operator = SzHamiltonian(n, periodic)
            sz_image = sz_operator.matvec(vector)
            assert np.allclose(sz_image, matrix @ vector, atol=1e-12)
            assert np.allclose(sz_image, projected @ vector, atol=1e-10)
            if n % 2 == 0:
                singlet = SingletHamiltonian(n, periodic)
                assert singlet.dimension == singlet_dimension(n)
                young = _operator_matrix(singlet.matvec, singlet.dimension)
                assert np.allclose(young, young.T, atol=1e-10)
                young_spectrum = np.linalg.eigvalsh(young)
                pauli_spectrum = singlet_energies_from_pauli(n, periodic)
                assert np.allclose(young_spectrum, pauli_spectrum, atol=1e-8)


def check_singlet_permutations() -> None:
    """Coxeter relations and the closing transposition in the Young basis."""
    rng = np.random.default_rng(7)
    for n in (4, 6, 8):
        operator = SingletHamiltonian(n, False)
        vector = rng.standard_normal(operator.dimension)
        for generator in range(n - 1):
            twice = operator.apply_generator(operator.apply_generator(vector, generator), generator)
            assert np.allclose(twice, vector, atol=1e-10)
        for generator in range(n - 3):
            left = operator.apply_generator(operator.apply_generator(vector, generator), generator + 2)
            right = operator.apply_generator(operator.apply_generator(vector, generator + 2), generator)
            assert np.allclose(left, right, atol=1e-10)
        for generator in range(n - 2):
            def braid(order: tuple[int, int, int], current: np.ndarray) -> np.ndarray:
                for item in order:
                    current = operator.apply_generator(current, item)
                return current

            assert np.allclose(
                braid((generator, generator + 1, generator), vector),
                braid((generator + 1, generator, generator + 1), vector),
                atol=1e-10,
            )
        closing = operator.apply_periodic_permutation(vector)
        explicit = vector
        sequence = list(range(n - 1)) + list(range(n - 3, -1, -1))
        for generator in sequence:
            explicit = operator.apply_generator(explicit, generator)
        assert np.allclose(closing, explicit, atol=1e-10)
        assert np.allclose(operator.apply_periodic_permutation(closing), vector, atol=1e-10)
        for periodic in (False, True):
            hamiltonian = SingletHamiltonian(n, periodic)
            left = rng.standard_normal(hamiltonian.dimension)
            right = rng.standard_normal(hamiltonian.dimension)
            difference = left @ hamiltonian.matvec(right) - hamiltonian.matvec(left) @ right
            scale = np.linalg.norm(left) * np.linalg.norm(right)
            assert abs(difference) <= 1e-8 * max(scale, 1.0)


def check_reference_energies() -> None:
    """Exact small energies, norms, residuals, and a second starting vector."""
    expectations = (
        (3, False, -1.0, (METHOD_CSR, METHOD_SZ)),
        (3, True, -0.75, (METHOD_CSR, METHOD_SZ)),
        (4, True, -2.0, (METHOD_CSR, METHOD_SZ, METHOD_SU2)),
    )
    for n, periodic, expected, methods in expectations:
        for method in methods:
            row = ground_state(n, periodic, repeat=True, method=method)
            assert abs(float(row["E0"]) - expected) < 1e-10
            assert float(row["residual"]) < 1e-8
            assert float(row["norm_error"]) < 1e-10
            assert float(row["repeat_delta"]) < 1e-8
            if method == METHOD_CSR:
                assert "nnz" in row and "csr_gib" in row
            else:
                assert "nnz" not in row and "csr_gib" not in row


def main() -> None:
    check_dense_reference()
    check_working_sizes()
    check_matrix_free_operators()
    check_singlet_permutations()
    check_reference_energies()
    print("Проверки пройдены: независимый эталон N=3..8, все сектора N=3..16 "
          "и бесматричные операторы.")


if __name__ == "__main__":
    main()

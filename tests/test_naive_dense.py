"""Проверки прямой плотной реализации на полном битовом базисе."""

import unittest

import numpy as np

from xxx_chain_naive import build_hamiltonian, ground_state


def pauli_reference(n: int, periodic: bool) -> np.ndarray:
    """Эталон из тензорных произведений операторов спина."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.diag([-0.5, 0.5]).astype(complex)
    identity = np.eye(2, dtype=complex)
    result = np.zeros((2**n, 2**n), dtype=complex)
    for site in range(n if periodic else n - 1):
        next_site = (site + 1) % n
        for spin in (sx, sy, sz):
            factors = [spin if index in (site, next_site) else identity
                       for index in range(n - 1, -1, -1)]
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            result += term
    return result


class NaiveDenseTests(unittest.TestCase):
    def test_matrix_matches_independent_pauli_reference(self) -> None:
        for n in range(2, 7):
            for periodic in (False, True):
                with self.subTest(n=n, periodic=periodic):
                    matrix = build_hamiltonian(n, periodic)
                    np.testing.assert_allclose(
                        matrix, pauli_reference(n, periodic), atol=1e-13, rtol=0
                    )
                    np.testing.assert_allclose(matrix, matrix.T, atol=0, rtol=0)

    def test_known_ground_energies(self) -> None:
        self.assertAlmostEqual(ground_state(3, False)["E0"], -1.0, places=12)
        self.assertAlmostEqual(ground_state(3, True)["E0"], -0.75, places=12)
        self.assertAlmostEqual(ground_state(4, True)["E0"], -2.0, places=12)


if __name__ == "__main__":
    unittest.main()

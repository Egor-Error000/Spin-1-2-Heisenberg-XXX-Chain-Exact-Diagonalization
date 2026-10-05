"""Independent checks of the unrestricted dense baseline."""
import unittest

import numpy as np

from tests.validate import pauli_reference
from xxx_chain import build_dense_hamiltonian, full_dense_eigensystem
from scripts.compare_dense import dense_estimate_bytes


class DenseBaselineTests(unittest.TestCase):
    def test_full_matrix_and_spectrum_against_pauli_reference(self) -> None:
        for n in range(3, 9):
            for periodic in (False, True):
                with self.subTest(n=n, periodic=periodic):
                    reference = pauli_reference(n, periodic)
                    matrix = build_dense_hamiltonian(n, periodic)
                    np.testing.assert_allclose(matrix, reference, atol=1e-13, rtol=0)
                    self.assertEqual(matrix.shape, (1 << n, 1 << n))
                    values, vectors = full_dense_eigensystem(matrix)
                    self.assertEqual(len(values), 1 << n)
                    self.assertEqual(vectors.shape, matrix.shape)
                    np.testing.assert_allclose(values, np.linalg.eigvalsh(reference),
                                               atol=1e-12, rtol=0)
                    self.assertLess(np.linalg.norm(matrix @ vectors[:, 0]
                                                   - values[0] * vectors[:, 0]), 1e-11)
                    # eigh must preserve H for the residual calculation.
                    np.testing.assert_allclose(matrix, reference, atol=1e-13, rtol=0)

    def test_large_dense_allocation_is_refused_before_building(self) -> None:
        with self.assertRaises(MemoryError):
            build_dense_hamiltonian(26, True)
        self.assertGreater(dense_estimate_bytes(15), 32 * 1024**3)


if __name__ == "__main__":
    unittest.main()

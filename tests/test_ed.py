"""Run the independent reference checks with the standard unittest runner."""

import unittest

from tests.validate import (
    check_dense_reference,
    check_matrix_free_operators,
    check_reference_energies,
    check_singlet_permutations,
    check_working_sizes,
)


class ExactDiagonalizationTests(unittest.TestCase):
    def test_pauli_reference_and_all_small_sectors(self) -> None:
        check_dense_reference()

    def test_residuals_and_repeatability_through_n16(self) -> None:
        check_working_sizes()

    def test_matrix_free_actions_and_singlet_spectrum(self) -> None:
        check_matrix_free_operators()

    def test_young_permutation_relations(self) -> None:
        check_singlet_permutations()

    def test_exact_energies_for_every_method(self) -> None:
        check_reference_energies()


if __name__ == "__main__":
    unittest.main()

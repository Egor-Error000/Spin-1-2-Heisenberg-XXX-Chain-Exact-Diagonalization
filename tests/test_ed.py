"""Run the independent reference checks with the standard unittest runner."""

import unittest

from tests.validate import check_dense_reference, check_working_sizes


class ExactDiagonalizationTests(unittest.TestCase):
    def test_pauli_reference_and_all_small_sectors(self) -> None:
        check_dense_reference()

    def test_residuals_and_repeatability_through_n16(self) -> None:
        check_working_sizes()


if __name__ == "__main__":
    unittest.main()

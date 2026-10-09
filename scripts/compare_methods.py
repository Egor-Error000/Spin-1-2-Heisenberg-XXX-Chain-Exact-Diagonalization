"""Compare matrix-free ground energies with the CSR result for N <= 26.

matrix_free_su2 is compared only for even N. The tolerance is 1e-8.
Stored CSR cases are reused when result_store accepts them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from result_store import load_case
from xxx_chain import METHOD_CSR, METHOD_SU2, METHOD_SZ, ground_state


def reference_energy(n: int, periodic: bool, directory: Path) -> float:
    stored = load_case(directory, n, "PBC" if periodic else "OBC")
    if stored is not None and stored.get("method", METHOD_CSR) == METHOD_CSR:
        return float(stored["E0"])
    return float(ground_state(n, periodic, method=METHOD_CSR)["E0"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=26)
    parser.add_argument("--cases", type=Path, default=Path("data/cases"))
    args = parser.parse_args()
    if not 3 <= args.max_n <= 26:
        raise SystemExit("--max-n must lie between 3 and 26")
    largest = 0.0
    for n in range(3, args.max_n + 1):
        methods = [METHOD_SZ] if n % 2 else [METHOD_SZ, METHOD_SU2]
        for periodic in (False, True):
            boundary = "PBC" if periodic else "OBC"
            reference = reference_energy(n, periodic, args.cases)
            for method in methods:
                energy = float(ground_state(n, periodic, method=method)["E0"])
                delta = abs(energy - reference)
                largest = max(largest, delta)
                print(f"N={n:2d} {boundary} {method}: |ΔE0|={delta:.3e}", flush=True)
                if delta > 1e-8:
                    raise SystemExit(f"Energy mismatch for N={n} {boundary} {method}")
    print(f"Maximum |ΔE0| through N={args.max_n}: {largest:.3e}")


if __name__ == "__main__":
    sys.exit(main())

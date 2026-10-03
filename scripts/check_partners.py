"""Check spin-flip partner sectors for all odd working lengths."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tqdm.auto import tqdm

from result_store import SOURCE_HASH, load_case
from xxx_chain import build_hamiltonian, lowest_eigenpair


def main() -> None:
    directory = Path("data/cases")
    cases = [(n, boundary) for n in range(3, 27, 2)
             for boundary in ("OBC", "PBC")]
    max_delta = 0.0
    for n, boundary in tqdm(cases, desc="Партнёрские секторы Sz=+1/2"):
        main_row = load_case(directory, n, boundary)
        if main_row is None:
            raise RuntimeError(f"Missing validated result: N={n}, {boundary}")
        matrix, _ = build_hamiltonian(n, boundary == "PBC", n_up=(n + 1) // 2)
        energy, _, residual = lowest_eigenpair(matrix, seed=314159)
        delta = abs(energy - float(main_row["E0"]))
        if delta > 1e-8 or residual > 1e-8:
            raise AssertionError((n, boundary, delta, residual))
        max_delta = max(max_delta, delta)
        del matrix
    payload = {"source_sha256": SOURCE_HASH, "checked_cases": len(cases),
               "max_energy_delta": max_delta}
    path = Path("data/partner_checks.json")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

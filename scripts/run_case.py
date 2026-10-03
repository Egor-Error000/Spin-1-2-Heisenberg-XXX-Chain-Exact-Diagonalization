"""Solve one (N, boundary) case in an isolated Docker container."""

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path
from time import perf_counter, process_time

from result_store import write_case
from xxx_chain import build_hamiltonian, ground_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("boundary", choices=("OBC", "PBC"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cases"))
    args = parser.parse_args()
    build_hamiltonian(3, False)  # Compile Numba before the measurement.
    start_wall, start_cpu = perf_counter(), process_time()
    row = ground_state(args.n, args.boundary == "PBC", repeat=True)
    row["wall_seconds"] = perf_counter() - start_wall
    row["cpu_seconds"] = process_time() - start_cpu
    row["cpu_percent_one_core"] = 100 * row["cpu_seconds"] / row["wall_seconds"]
    row["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    row["docker_memory_gib"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    path = write_case(args.output_dir, row)
    print(json.dumps({"path": str(path), "result": row}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

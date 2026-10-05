"""Measure one XXX-chain solve in an isolated Docker container.

Example: docker compose -f docker/compose.yaml run --rm notebook python scripts/profile_resources.py 22 PBC
Linux's ru_maxrss reports the process peak in KiB. CPU time includes user and
system time; CPU usage is CPU time / wall time, where 100% means one full core.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
from time import perf_counter, process_time

from xxx_chain import build_hamiltonian, lowest_eigenpair


def gib(byte_count: int) -> float:
    return byte_count / 1024**3


def cgroup_value(name: str) -> int | None:
    try:
        value = open(f"/sys/fs/cgroup/{name}", encoding="ascii").read().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("boundary", choices=("OBC", "PBC"))
    args = parser.parse_args()

    # Warm the Numba kernels before starting the measurement.
    build_hamiltonian(4, False)
    periodic = args.boundary == "PBC"
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    start_wall = perf_counter()
    start_cpu = process_time()
    matrix, states = build_hamiltonian(args.n, periodic, progress=False)
    build_wall = perf_counter() - start_wall
    build_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    energy, vector, residual = lowest_eigenpair(matrix)
    solve_wall = perf_counter() - start_wall - build_wall
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cpu_seconds = process_time() - start_cpu
    wall_seconds = perf_counter() - start_wall
    cgroup_peak = cgroup_value("memory.peak")
    record = {
        "N": args.n,
        "boundary": args.boundary,
        "dimension": len(states),
        "nnz": matrix.nnz,
        "E0": energy,
        "residual": residual,
        "build_seconds": build_wall,
        "eigsh_seconds": solve_wall,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_percent_one_core": 100 * cpu_seconds / wall_seconds,
        "baseline_rss_gib": gib(baseline_rss),
        "build_peak_rss_gib": gib(build_peak_rss),
        "peak_rss_gib": gib(peak_rss),
        "cgroup_peak_gib": gib(cgroup_peak) if cgroup_peak is not None else None,
        "csr_gib": gib(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
        "docker_memory_gib": gib(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
    }
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    # Keep references alive through peak measurement.
    assert len(vector) == len(states)


if __name__ == "__main__":
    main()

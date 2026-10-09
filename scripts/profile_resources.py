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

from xxx_chain import (
    METHOD_CSR,
    METHOD_SU2,
    METHOD_SZ,
    SingletHamiltonian,
    SzHamiltonian,
    build_hamiltonian,
    lowest_eigenpair,
    lowest_eigenpair_operator,
    production_method,
    warmup,
)


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
    method = production_method(args.n)
    periodic = args.boundary == "PBC"
    warmup(method)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    start_wall = perf_counter()
    start_cpu = process_time()
    if method == METHOD_CSR:
        matrix, states = build_hamiltonian(args.n, periodic, progress=False)
        dimension = len(states)
        basis_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    elif method == METHOD_SZ:
        operator = SzHamiltonian(args.n, periodic)
        dimension = operator.dimension
        basis_bytes = operator.basis_bytes
    elif method == METHOD_SU2:
        operator = SingletHamiltonian(args.n, periodic)
        dimension = operator.dimension
        basis_bytes = operator.basis_bytes
    else:
        raise ValueError(method)
    build_wall = perf_counter() - start_wall
    build_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if method == METHOD_CSR:
        energy, vector, residual = lowest_eigenpair(matrix)
    else:
        energy, vector, residual = lowest_eigenpair_operator(
            operator.matvec, dimension, basis_bytes
        )
    solve_wall = perf_counter() - start_wall - build_wall
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cpu_seconds = process_time() - start_cpu
    wall_seconds = perf_counter() - start_wall
    cgroup_peak = cgroup_value("memory.peak")
    record = {
        "N": args.n,
        "boundary": args.boundary,
        "method": method,
        "dimension": dimension,
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
        "docker_memory_gib": gib(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
    }
    if method == METHOD_CSR:
        record["nnz"] = matrix.nnz
        record["csr_gib"] = gib(basis_bytes)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    assert len(vector) == dimension


if __name__ == "__main__":
    main()

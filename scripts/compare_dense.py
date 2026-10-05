"""Isolated dense-full-spectrum vs central-sector CSR/eigsh measurements.

Run in Docker (one BLAS thread, as configured by compose.yaml):
    python scripts/compare_dense.py --max-n 26 --case-timeout 180
The dense route stops after a memory refusal or timeout for each boundary.
Those cases are reported as skipped/timeout, never as completed measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = 1


def comparison_hash() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "src" / "xxx_chain.py", Path(__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dense_estimate_bytes(n: int) -> int:
    # Conservative forecast: H, eigensolver copy/eigenvectors, quadratic
    # divide-and-conquer workspace and margin, plus interpreter/JIT overhead.
    return 5 * 8 * (1 << n)**2 + 512 * 1024**2


def available_memory_bytes() -> int:
    if sys.platform != "linux":
        raise RuntimeError("Run this benchmark in Docker/Linux for comparable RSS measurements")
    available = next(int(line.split()[1]) * 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    # Honour a container limit as well as the VM's available memory.
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit = Path(limit_path).read_text().strip()
            if limit != "max":
                remaining = max(0, int(limit) - int(Path(usage_path).read_text()))
                available = min(available, remaining)
        except (OSError, ValueError):
            pass
    return available


def worker(n: int, boundary: str, method: str) -> dict:
    import resource
    import numpy as np
    from xxx_chain import (build_dense_hamiltonian, build_hamiltonian,
                           full_dense_eigensystem, lowest_eigenpair)

    periodic = boundary == "PBC"
    # Warm both builders in both processes; exclude compilation from timing.
    build_dense_hamiltonian(3, False)
    build_hamiltonian(3, False)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    start = perf_counter()
    start_cpu = process_time()
    if method == "dense":
        matrix = build_dense_hamiltonian(n, periodic, max_matrix_bytes=8 * (1 << n)**2)
        states = None
        matrix_bytes = matrix.nbytes
    else:
        matrix, states = build_hamiltonian(n, periodic)
        matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    build_seconds = perf_counter() - start
    solve_start = perf_counter()
    if method == "dense":
        energies, vectors = full_dense_eigensystem(matrix)
        energy, vector = float(energies[0]), vectors[:, 0]
        levels = len(energies)
    else:
        energy, vector, _ = lowest_eigenpair(matrix)
        levels = 1
    solve_seconds = perf_counter() - solve_start
    # Diagnostics are timed separately from matrix construction + solver.
    residual = float(np.linalg.norm(matrix @ vector - energy * vector))
    norm_error = abs(float(np.linalg.norm(vector)) - 1.0)
    if residual > 1e-8 or norm_error > 1e-10:
        raise RuntimeError(f"Failed numerical diagnostics: {residual=}, {norm_error=}")
    return {
        "N": n, "boundary": boundary, "method": method, "status": "ok",
        "dimension": matrix.shape[0], "computed_levels": levels,
        "matrix_gib": matrix_bytes / 1024**3, "E0": energy,
        "build_seconds": build_seconds, "solve_seconds": solve_seconds,
        "wall_seconds": build_seconds + solve_seconds,
        "diagnostics_seconds": perf_counter() - solve_start - solve_seconds,
        "cpu_seconds_including_diagnostics": process_time() - start_cpu,
        "baseline_rss_gib": baseline / 1024**3,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "residual": residual, "norm_error": norm_error,
        "nnz": None if method == "dense" else matrix.nnz,
    }


def load_comparison(directory: Path) -> dict | None:
    path = directory / "dense_comparison.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("source_sha256") != comparison_hash():
        return None
    seen = set()
    for row in payload["rows"]:
        key = (row["N"], row["boundary"], row["method"])
        if key in seen:
            raise ValueError("Duplicate comparison case")
        seen.add(key)
        if row["status"] == "ok":
            for field in ("E0", "wall_seconds", "peak_rss_gib", "residual", "norm_error"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Nonfinite comparison {field}")
            if row["residual"] > 1e-8 or row["norm_error"] > 1e-10:
                raise ValueError("Invalid comparison diagnostics")
    for row in payload["pairs"]:
        if row["energy_delta"] > 1e-8:
            raise ValueError("Dense/sparse energy mismatch")
    return payload


def write_results(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "dense_comparison.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(directory / "dense_comparison.json")
    for name, rows in (("dense_comparison.csv", payload["rows"]),
                       ("dense_comparison_pairs.csv", payload["pairs"])):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def run_comparison(max_n: int = 26, case_timeout: float = 180,
                   memory_fraction: float = 0.70, directory: Path | None = None) -> dict:
    if not 3 <= max_n <= 26 or case_timeout <= 0 or not 0 < memory_fraction <= 0.8:
        raise ValueError("Require 3 <= max_n <= 26, timeout > 0, 0 < memory_fraction <= 0.8")
    directory = ROOT / "data" if directory is None else directory
    import scipy
    import numpy as np
    payload = {
        "version": VERSION, "source_sha256": comparison_hash(),
        "settings": {"max_n": max_n, "case_timeout": case_timeout,
                     "memory_fraction": memory_fraction,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "threads": {key: os.environ.get(key) for key in
                                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                     "platform": sys.platform,
                     "timing": "warm builders; construction + one solver; diagnostics separate",
                     "memory": "isolated child-process peak RSS including imports/JIT/diagnostics"},
        "rows": [], "pairs": [],
    }
    stopped = {}
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[key] = "1"
    payload["settings"]["threads"] = {key: environment[key] for key in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    for n in range(3, max_n + 1):
        for boundary in ("OBC", "PBC"):
            estimate = dense_estimate_bytes(n)
            available = available_memory_bytes()
            budget = memory_fraction * available
            reason = stopped.get(boundary)
            if estimate > budget:
                reason = "memory_budget"
            if reason:
                stopped[boundary] = reason
                payload["rows"].append({"N": n, "boundary": boundary, "method": "dense",
                                        "status": "skipped", "reason": reason,
                                        "dimension": 1 << n, "matrix_gib": 8*(1 << n)**2 / 1024**3,
                                        "estimated_peak_gib": estimate / 1024**3,
                                        "memory_budget_gib": budget / 1024**3})
                write_results(directory, payload)
                continue
            measured = {}
            for method in ("dense", "sparse"):
                print(f"N={n:2d} {boundary} {method}: start", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--n", str(n), "--boundary", boundary, "--method", method]
                try:
                    process = subprocess.run(command, capture_output=True, text=True,
                                             timeout=case_timeout, env=environment)
                except subprocess.TimeoutExpired:
                    payload["rows"].append({"N": n, "boundary": boundary, "method": method,
                                            "status": "timeout", "reason": "case_timeout",
                                            "timeout_seconds": case_timeout,
                                            "estimated_peak_gib": estimate / 1024**3})
                    stopped[boundary] = "previous_timeout"
                    print(f"N={n} {boundary}: {method} timeout ({case_timeout:g} s)", flush=True)
                    write_results(directory, payload)
                    break
                if process.returncode:
                    payload["rows"].append({"N": n, "boundary": boundary, "method": method,
                                            "status": "error", "reason": process.stderr[-2000:]})
                    write_results(directory, payload)
                    raise RuntimeError(f"Comparison worker failed: {process.stderr[-2000:]}")
                row = json.loads(process.stdout)
                row["estimated_peak_gib"] = estimate / 1024**3 if method == "dense" else None
                row["memory_budget_gib"] = budget / 1024**3
                payload["rows"].append(row)
                measured[method] = row
                print(f"  {row['wall_seconds']:.3f} s, RSS {row['peak_rss_gib']:.3f} GiB", flush=True)
            if len(measured) == 2:
                dense, sparse = measured["dense"], measured["sparse"]
                delta = abs(dense["E0"] - sparse["E0"])
                if delta > 1e-8:
                    raise RuntimeError(f"Energy mismatch at N={n} {boundary}: {delta}")
                payload["pairs"].append({
                    "N": n, "boundary": boundary, "dense_E0": dense["E0"],
                    "sparse_E0": sparse["E0"], "energy_delta": delta,
                    "dense_seconds": dense["wall_seconds"], "sparse_seconds": sparse["wall_seconds"],
                    "time_ratio_dense_over_sparse": dense["wall_seconds"] / sparse["wall_seconds"],
                    "dense_peak_gib": dense["peak_rss_gib"], "sparse_peak_gib": sparse["peak_rss_gib"],
                    "dense_dimension": dense["dimension"], "sparse_dimension": sparse["dimension"],
                })
            write_results(directory, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=26)
    parser.add_argument("--case-timeout", type=float, default=180)
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--boundary", choices=("OBC", "PBC"), help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("dense", "sparse"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.n, args.boundary, args.method), allow_nan=False))
    else:
        run_comparison(args.max_n, args.case_timeout, args.memory_fraction)


if __name__ == "__main__":
    main()

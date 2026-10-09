"""Atomic, versioned results for independent Docker case workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from math import comb
from pathlib import Path


SOURCE_HASH = hashlib.sha256(Path(__file__).with_name("xxx_chain.py").read_bytes()).hexdigest()
CASE_VERSION = 2
# Results measured before the matrix-free solvers were added.
LEGACY_CASE_VERSION = 1
LEGACY_SOURCE_SHA256 = "8aeb60463a96851b420cad5ac241b12e6cbb1e603b55cf8447201152d3a2da5a"
CSR_METHODS = ("csr",)
SZ_METHODS = ("csr", "matrix_free_sz")


def case_path(directory: Path, n: int, boundary: str) -> Path:
    return directory / f"N{n:02d}_{boundary}.json"


def expected_dimension(n: int, method: str) -> int:
    """Block size of one production representation."""
    half = n // 2
    if method == "matrix_free_su2":
        if n % 2:
            raise ValueError("The singlet block exists only for even N")
        return comb(n, half) // (half + 1)
    if method in SZ_METHODS:
        return comb(n, half)
    raise ValueError(f"Unknown method {method}")


def write_case(directory: Path, row: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = case_path(directory, int(row["N"]), str(row["boundary"]))
    payload = {"case_version": CASE_VERSION, "source_sha256": SOURCE_HASH, "result": row}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _accepts(payload: dict, n: int, method: str) -> bool:
    version = payload.get("case_version")
    digest = payload.get("source_sha256")
    if version == CASE_VERSION and digest == SOURCE_HASH:
        return True
    return (
        version == LEGACY_CASE_VERSION
        and digest == LEGACY_SOURCE_SHA256
        and n <= 26
        and method == "csr"
    )


def load_case(directory: Path, n: int, boundary: str) -> dict | None:
    path = case_path(directory, n, boundary)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["result"]
    method = row.get("method", "csr")
    if not _accepts(payload, n, method):
        return None
    if "method" not in row:
        row = dict(row)
        row["method"] = "csr"
    if int(row["N"]) != n or row["boundary"] != boundary:
        raise ValueError(f"Wrong case in {path}")
    if int(row["n_up"]) != n // 2 or int(row["dimension"]) != expected_dimension(n, method):
        raise ValueError(f"Wrong sector in {path}")
    if method == "csr" and "nnz" not in row:
        raise ValueError(f"CSR result has no element count in {path}")
    if not all(math.isfinite(float(row[key])) for key in ("E0", "e0", "residual", "norm_error", "repeat_delta")):
        raise ValueError(f"Nonfinite result in {path}")
    if abs(float(row["e0"]) - float(row["E0"]) / n) > 1e-10:
        raise ValueError(f"Inconsistent energy in {path}")
    if float(row["residual"]) > 1e-8 or float(row["norm_error"]) > 1e-10 or float(row["repeat_delta"]) > 1e-8:
        raise ValueError(f"Failed eigensolver diagnostic in {path}")
    return row


def load_grid(directory: Path, sizes: range | tuple[int, ...]) -> tuple[list[dict], list[tuple[int, bool]]]:
    results, missing = [], []
    for n in sizes:
        for periodic in (False, True):
            row = load_case(directory, n, "PBC" if periodic else "OBC")
            if row is None:
                missing.append((n, periodic))
            else:
                results.append(row)
    return results, missing

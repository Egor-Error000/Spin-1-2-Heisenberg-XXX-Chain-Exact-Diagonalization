"""Atomic, versioned results for independent Docker case workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


SOURCE_HASH = hashlib.sha256(Path(__file__).with_name("xxx_chain.py").read_bytes()).hexdigest()
CASE_VERSION = 1


def case_path(directory: Path, n: int, boundary: str) -> Path:
    return directory / f"N{n:02d}_{boundary}.json"


def write_case(directory: Path, row: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = case_path(directory, int(row["N"]), str(row["boundary"]))
    payload = {"case_version": CASE_VERSION, "source_sha256": SOURCE_HASH, "result": row}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_case(directory: Path, n: int, boundary: str) -> dict | None:
    path = case_path(directory, n, boundary)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_version") != CASE_VERSION or payload.get("source_sha256") != SOURCE_HASH:
        return None
    row = payload["result"]
    if int(row["N"]) != n or row["boundary"] != boundary:
        raise ValueError(f"Wrong case in {path}")
    if int(row["n_up"]) != n // 2 or int(row["dimension"]) < 1:
        raise ValueError(f"Wrong sector in {path}")
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

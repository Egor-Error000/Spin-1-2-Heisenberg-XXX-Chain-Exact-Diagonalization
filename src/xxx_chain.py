"""Exact diagonalization of the spin-1/2 antiferromagnetic XXX chain.

Site i is bit i. Bit 1 denotes spin up; S = sigma/2 and J = 1.
The production solver is CSR for N <= 26, a matrix-free Sz sector for N = 27,
and the matrix-free total-spin singlet for N = 28.
"""

from __future__ import annotations

from math import comb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from numba import njit
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from tqdm.auto import tqdm

METHOD_CSR = "csr"
METHOD_SZ = "matrix_free_sz"
METHOD_SU2 = "matrix_free_su2"
LANCZOS_BYTE_LIMIT = 12 * 1024**3


@njit(cache=True)
def _popcount(value: int) -> int:
    count = 0
    while value:
        count += value & 1
        value >>= 1
    return count


@njit(cache=True)
def _fill_sector(states: np.ndarray, n: int, n_up: int) -> None:
    index = 0
    for state in range(1 << n):
        if _popcount(state) == n_up:
            states[index] = state
            index += 1


def sector_basis(n: int, n_up: int) -> tuple[np.ndarray, np.ndarray]:
    """Return bit states and a full-state to sector-index lookup table."""
    if not (0 <= n_up <= n and 2 <= n <= 26):
        raise ValueError("Require 2 <= n <= 26 and 0 <= n_up <= n")
    states = np.empty(comb(n, n_up), dtype=np.int32)
    _fill_sector(states, n, n_up)
    lookup = np.full(1 << n, -1, dtype=np.int32)
    lookup[states] = np.arange(len(states), dtype=np.int32)
    return states, lookup


@njit(cache=True)
def _chunk_triplets(
    states: np.ndarray,
    lookup: np.ndarray,
    start: int,
    stop: int,
    n: int,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bonds = n if periodic else n - 1
    capacity = (stop - start) * (bonds + 1)
    rows = np.empty(capacity, dtype=np.int32)
    cols = np.empty(capacity, dtype=np.int32)
    data = np.empty(capacity, dtype=np.float64)
    used = 0
    for column in range(start, stop):
        state = states[column]
        diagonal = 0.0
        for i in range(bonds):
            j = (i + 1) % n
            if ((state >> i) & 1) == ((state >> j) & 1):
                diagonal += 0.25
            else:
                diagonal -= 0.25
                flipped = state ^ (1 << i) ^ (1 << j)
                rows[used] = lookup[flipped]
                cols[used] = column
                data[used] = 0.5
                used += 1
        rows[used] = column
        cols[used] = column
        data[used] = diagonal
        used += 1
    return rows[:used], cols[:used], data[:used]


def build_hamiltonian(
    n: int,
    periodic: bool,
    n_up: int | None = None,
    *,
    progress: bool = False,
    chunk_size: int = 1024,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build H in a fixed-Sz sector; returns H and its ordered bit states."""
    if n_up is None:
        n_up = n // 2
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    states, lookup = sector_basis(n, n_up)
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    starts = range(0, len(states), chunk_size)
    if progress and len(states) > chunk_size:
        starts = tqdm(starts, total=(len(states) + chunk_size - 1) // chunk_size,
                      desc=f"H N={n} {'PBC' if periodic else 'OBC'}", leave=False,
                      mininterval=0.5)
    for start in starts:
        rows, cols, data = _chunk_triplets(
            states, lookup, start, min(start + chunk_size, len(states)), n, periodic
        )
        rows_parts.append(rows)
        cols_parts.append(cols)
        data_parts.append(data)
    shape = (len(states), len(states))
    matrix = sparse.coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
        shape=shape,
        dtype=np.float64,
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix, states


def production_method(n: int) -> str:
    """Return the solver used for a production ground-state case."""
    if 3 <= n <= 26:
        return METHOD_CSR
    if n == 27:
        return METHOD_SZ
    if n == 28:
        return METHOD_SU2
    raise ValueError("Production sizes are N=3..28")


def singlet_dimension(n: int) -> int:
    """Number of standard Young tableaux of shape (n/2, n/2)."""
    if n < 2 or n % 2:
        raise ValueError("The singlet basis requires even n >= 2")
    half = n // 2
    return comb(n, half) // (half + 1)


def sector_dimension(n: int) -> int:
    """Dimension of the central Sz sector."""
    return comb(n, n // 2)


class SzHamiltonian:
    """Matrix-free XXX Hamiltonian in one Sz sector.

    The state list and the full-configuration lookup replace the CSR matrix.
    N = 27 fits: every configuration is below 2^31, and the lookup is 0.5 GiB.
    """

    def __init__(self, n: int, periodic: bool, n_up: int | None = None):
        if n_up is None:
            n_up = n // 2
        if not (0 <= n_up <= n and 2 <= n <= 27):
            raise ValueError("matrix_free_sz requires 2 <= n <= 27 and a valid n_up")
        states = np.empty(comb(n, n_up), dtype=np.int32)
        _fill_sector(states, n, n_up)
        lookup = np.full(1 << n, -1, dtype=np.int32)
        lookup[states] = np.arange(len(states), dtype=np.int32)
        self.n = n
        self.periodic = periodic
        self.n_up = n_up
        self.states = states
        self.lookup = lookup
        self.dimension = len(states)
        self.shape = (self.dimension, self.dimension)
        self.basis_bytes = states.nbytes + lookup.nbytes

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError("Vector length does not match the Sz sector")
        out = np.empty(self.dimension, dtype=np.float64)
        _apply_sz(out, vector, self.states, self.lookup, self.n, self.periodic)
        return out


class SingletHamiltonian:
    """Matrix-free XXX Hamiltonian in the total-spin singlet.

    The basis is the orthonormal Young basis of shape (n/2, n/2). Adjacent
    transpositions use Young's orthogonal form. The periodic bond is the
    representation of (1 n), applied as a product of adjacent transpositions
    from the right, and is never expanded on individual basis vectors.
    """

    def __init__(self, n: int, periodic: bool):
        if n < 2 or n > 28 or n % 2:
            raise ValueError("matrix_free_su2 requires even 2 <= n <= 28")
        half = n // 2
        dimension = singlet_dimension(n)
        raw, found = _enumerate_ballot(half, dimension)
        if found != dimension:
            raise RuntimeError("Ballot enumeration does not have the Catalan cardinality")
        contents = _tableau_contents(raw, n)
        tops = _top_row_numbers(raw, half)
        lex_order = np.lexsort([tops[:, column] for column in range(half - 1, -1, -1)])
        masks = np.ascontiguousarray(raw[lex_order])
        contents = np.ascontiguousarray(contents[lex_order])
        by_mask = np.argsort(masks, kind="mergesort")
        sorted_masks = np.ascontiguousarray(masks[by_mask])
        partners, distances = _build_transitions(
            masks, sorted_masks, np.ascontiguousarray(by_mask), contents
        )
        del raw, masks, contents, sorted_masks, by_mask, tops
        _assert_transition_symmetry(partners, distances)
        diagonal = np.zeros(dimension, dtype=np.float64)
        for bond in range(n - 1):
            axial = distances[bond].astype(np.float64)
            diagonal += 0.5 / axial - 0.25
        self.n = n
        self.periodic = periodic
        self.dimension = dimension
        self.shape = (dimension, dimension)
        self.partners = partners
        self.distances = distances
        self.diagonal = diagonal
        self.scratch = np.empty(dimension, dtype=np.float64)
        self.permutation = np.empty(dimension, dtype=np.float64)
        self.basis_bytes = (
            partners.nbytes + distances.nbytes + diagonal.nbytes
            + self.scratch.nbytes + self.permutation.nbytes
        )

    def apply_generator(self, vector: np.ndarray, generator: int) -> np.ndarray:
        """Apply the adjacent transposition s_{generator+1}."""
        vector = np.ascontiguousarray(vector, dtype=np.float64)
        out = np.empty(self.dimension, dtype=np.float64)
        _apply_generator(out, vector, self.partners[generator], self.distances[generator])
        return out

    def apply_periodic_permutation(self, vector: np.ndarray) -> np.ndarray:
        """Apply rho((1 n)); the rightmost adjacent factor acts first."""
        vector = np.ascontiguousarray(vector, dtype=np.float64)
        out = np.empty(self.dimension, dtype=np.float64)
        _apply_long_cycle(out, vector, self.partners, self.distances, self.scratch)
        return out

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError("Vector length does not match the singlet basis")
        out = np.empty(self.dimension, dtype=np.float64)
        _apply_singlet(
            out, vector, self.partners, self.distances, self.diagonal,
            self.periodic, self.scratch, self.permutation,
        )
        return out


def warmup(method: str) -> None:
    """Compile the Numba kernels used by one production method."""
    if method == METHOD_CSR:
        build_hamiltonian(4, False)
    elif method == METHOD_SZ:
        operator = SzHamiltonian(4, False)
        operator.matvec(np.ones(operator.dimension))
    elif method == METHOD_SU2:
        operator = SingletHamiltonian(4, False)
        operator.matvec(np.ones(operator.dimension))
        operator.apply_periodic_permutation(np.ones(operator.dimension))
    else:
        raise ValueError(f"Unknown method {method}")


@njit(cache=True)
def _apply_sz(out, vector, states, lookup, n, periodic):
    bonds = n if periodic else n - 1
    for column in range(states.shape[0]):
        state = states[column]
        diagonal = 0.0
        acc = 0.0
        for site in range(bonds):
            other = (site + 1) % n
            if ((state >> site) & np.int32(1)) == ((state >> other) & np.int32(1)):
                diagonal += 0.25
            else:
                diagonal -= 0.25
                flipped = state ^ (np.int32(1) << site) ^ (np.int32(1) << other)
                acc += 0.5 * vector[lookup[flipped]]
        out[column] = acc + diagonal * vector[column]


@njit(cache=True)
def _enumerate_ballot(half, dimension):
    """Depth-first ballot sequences; bit k is set when number k+1 is on top."""
    masks = np.empty(dimension, dtype=np.uint32)
    limit = 2 * half
    stack_pos = np.empty(limit + 2, dtype=np.int32)
    stack_top = np.empty(limit + 2, dtype=np.int32)
    stack_bottom = np.empty(limit + 2, dtype=np.int32)
    stack_mask = np.empty(limit + 2, dtype=np.uint32)
    stack_stage = np.empty(limit + 2, dtype=np.int32)
    stack_pos[0] = 0
    stack_top[0] = 0
    stack_bottom[0] = 0
    stack_mask[0] = np.uint32(0)
    stack_stage[0] = 0
    head = 0
    found = 0
    while head >= 0:
        pos = stack_pos[head]
        if pos == limit:
            masks[found] = stack_mask[head]
            found += 1
            head -= 1
            continue
        stage = stack_stage[head]
        if stage == 0:
            stack_stage[head] = 1
            if stack_top[head] < half:
                head += 1
                stack_pos[head] = pos + 1
                stack_top[head] = stack_top[head - 1] + 1
                stack_bottom[head] = stack_bottom[head - 1]
                stack_mask[head] = stack_mask[head - 1] | (np.uint32(1) << np.uint32(pos))
                stack_stage[head] = 0
        elif stage == 1:
            stack_stage[head] = 2
            if stack_bottom[head] < stack_top[head]:
                head += 1
                stack_pos[head] = pos + 1
                stack_top[head] = stack_top[head - 1]
                stack_bottom[head] = stack_bottom[head - 1] + 1
                stack_mask[head] = stack_mask[head - 1]
                stack_stage[head] = 0
        else:
            head -= 1
    return masks, found


@njit(cache=True)
def _tableau_contents(masks, n):
    """Content c - r of every number, with both indices starting at zero."""
    contents = np.empty((masks.shape[0], n), dtype=np.int8)
    for index in range(masks.shape[0]):
        mask = masks[index]
        top_column = 0
        bottom_column = 0
        for bit in range(n):
            if (mask >> np.uint32(bit)) & np.uint32(1):
                contents[index, bit] = top_column
                top_column += 1
            else:
                contents[index, bit] = bottom_column - 1
                bottom_column += 1
    return contents


@njit(cache=True)
def _top_row_numbers(masks, half):
    numbers = np.empty((masks.shape[0], half), dtype=np.uint8)
    width = 2 * half
    for index in range(masks.shape[0]):
        mask = masks[index]
        column = 0
        for bit in range(width):
            if (mask >> np.uint32(bit)) & np.uint32(1):
                numbers[index, column] = bit + 1
                column += 1
    return numbers


@njit(cache=True)
def _rank_of(sorted_masks, lex_index, mask):
    low = 0
    high = sorted_masks.shape[0]
    while low < high:
        mid = (low + high) // 2
        if sorted_masks[mid] < mask:
            low = mid + 1
        else:
            high = mid
    if low == sorted_masks.shape[0] or sorted_masks[low] != mask:
        return np.int32(-1)
    return np.int32(lex_index[low])


@njit(cache=True)
def _build_transitions(masks, sorted_masks, lex_index, contents):
    dimension = masks.shape[0]
    n_generators = contents.shape[1] - 1
    partners = np.empty((n_generators, dimension), dtype=np.int32)
    distances = np.empty((n_generators, dimension), dtype=np.int8)
    for generator in range(n_generators):
        for index in range(dimension):
            distance = int(contents[index, generator + 1]) - int(contents[index, generator])
            distances[generator, index] = distance
            if distance == 1 or distance == -1:
                partners[generator, index] = -1
            else:
                swapped = masks[index] ^ (np.uint32(1) << np.uint32(generator))
                swapped ^= np.uint32(1) << np.uint32(generator + 1)
                partners[generator, index] = _rank_of(sorted_masks, lex_index, swapped)
    return partners, distances


def _assert_transition_symmetry(partners: np.ndarray, distances: np.ndarray) -> None:
    for generator in range(partners.shape[0]):
        partner = partners[generator]
        distance = distances[generator]
        linked = np.flatnonzero(partner >= 0)
        image = partner[linked]
        if np.any(image < 0) or np.any(partner[image] != linked):
            raise RuntimeError("Adjacent transposition is not an involution on tableaux")
        if np.any(distance[image].astype(np.int16) != -distance[linked].astype(np.int16)):
            raise RuntimeError("Axial distance did not change sign on the partner tableau")


@njit(cache=True)
def _apply_generator(out, vector, partner, distance):
    for index in range(vector.shape[0]):
        axial = np.float64(distance[index])
        acc = vector[index] / axial
        image = partner[index]
        if image >= 0:
            acc += np.sqrt((axial * axial - 1.0) / (axial * axial)) * vector[image]
        out[index] = acc


@njit(cache=True)
def _apply_long_cycle(out, vector, partners, distances, scratch):
    """Apply s1 s2 ... s_{n-1} ... s2 s1, with s1 acting first."""
    n_generators = partners.shape[0]
    steps = 2 * n_generators - 1
    _apply_generator(out, vector, partners[0], distances[0])
    source_is_out = True
    for step in range(1, steps):
        if step < n_generators:
            generator = step
        else:
            generator = 2 * n_generators - 2 - step
        if source_is_out:
            _apply_generator(scratch, out, partners[generator], distances[generator])
        else:
            _apply_generator(out, scratch, partners[generator], distances[generator])
        source_is_out = not source_is_out
    if not source_is_out:
        for index in range(vector.shape[0]):
            out[index] = scratch[index]


@njit(cache=True)
def _apply_singlet(out, vector, partners, distances, diagonal, periodic, scratch, permutation):
    n_generators = partners.shape[0]
    for index in range(vector.shape[0]):
        out[index] = diagonal[index] * vector[index]
    for generator in range(n_generators):
        partner = partners[generator]
        distance = distances[generator]
        for index in range(vector.shape[0]):
            image = partner[index]
            if image >= 0:
                axial = np.float64(distance[index])
                coefficient = 0.5 * np.sqrt((axial * axial - 1.0) / (axial * axial))
                out[index] += coefficient * vector[image]
    if periodic:
        _apply_long_cycle(permutation, vector, partners, distances, scratch)
        for index in range(vector.shape[0]):
            out[index] += 0.5 * permutation[index] - 0.25 * vector[index]


def _lanczos_bytes(dimension: int, ncv: int, basis_bytes: int) -> int:
    """Lanczos basis plus the stored operator basis plus two vectors."""
    return 8 * dimension * ncv + basis_bytes + 16 * dimension


def lowest_eigenpair_operator(
    matvec, dimension: int, basis_bytes: int, *, seed: int = 2026,
    tolerance: float = 1e-11, progress: bool = False, label: str = "eigsh",
) -> tuple[float, np.ndarray, float]:
    """Lowest eigenpair of a symmetric matrix-free operator.

    ncv starts at 20. It is increased only while the workspace estimate stays
    below 12 GiB and the residual is still above 1e-8.
    """
    if dimension == 1:
        vector = np.ones(1)
        energy = float(matvec(vector)[0])
        return energy, vector, 0.0
    ncv = min(20, dimension)
    energy = None
    vector = None
    residual = None
    while True:
        estimate = _lanczos_bytes(dimension, ncv, basis_bytes)
        if estimate >= LANCZOS_BYTE_LIMIT:
            if energy is None:
                raise MemoryError(
                    f"Lanczos estimate is {estimate / 1024**3:.2f} GiB, at the 12 GiB guard"
                )
            return energy, vector, residual
        rng = np.random.default_rng(seed)
        initial = rng.standard_normal(dimension)
        if progress:
            with tqdm(desc=label, unit="matvec", mininterval=0.5, leave=False) as bar:
                def counting(candidate, _bar=bar):
                    _bar.update(1)
                    return matvec(candidate)

                operator = LinearOperator((dimension, dimension), matvec=counting, dtype=np.float64)
                values, vectors = eigsh(
                    operator, k=1, which="SA", v0=initial, tol=tolerance, ncv=ncv
                )
        else:
            operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=np.float64)
            values, vectors = eigsh(
                operator, k=1, which="SA", v0=initial, tol=tolerance, ncv=ncv
            )
        energy = float(values[0])
        vector = np.array(vectors[:, 0], dtype=np.float64, copy=True)
        residual = float(np.linalg.norm(matvec(vector) - energy * vector))
        if residual <= 1e-8:
            return energy, vector, residual
        grown = min(dimension, ncv * 2)
        if grown <= ncv or _lanczos_bytes(dimension, grown, basis_bytes) >= LANCZOS_BYTE_LIMIT:
            return energy, vector, residual
        ncv = grown


def lowest_eigenpair(
    matrix: sparse.csr_matrix, *, seed: int = 2026, tolerance: float = 1e-11,
    progress: bool = False, label: str = "eigsh",
) -> tuple[float, np.ndarray, float]:
    """Return lowest eigenvalue, normalized vector, and absolute residual."""
    if matrix.shape[0] == 1:
        return float(matrix[0, 0]), np.ones(1), 0.0
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal(matrix.shape[0])
    if progress:
        with tqdm(desc=label, unit="matvec", mininterval=0.5, leave=False) as bar:
            def matvec(vector: np.ndarray) -> np.ndarray:
                bar.update(1)
                return matrix @ vector

            operator = LinearOperator(matrix.shape, matvec=matvec, dtype=np.float64)
            values, vectors = eigsh(operator, k=1, which="SA", v0=initial,
                                    tol=tolerance)
    else:
        values, vectors = eigsh(matrix, k=1, which="SA", v0=initial,
                                tol=tolerance)
    energy = float(values[0])
    vector = vectors[:, 0]
    residual = float(np.linalg.norm(matrix @ vector - energy * vector))
    return energy, vector, residual


def _result_row(
    n: int, periodic: bool, method: str, dimension: int, energy: float,
    vector: np.ndarray, residual: float, repeat_delta: float,
) -> dict[str, float | int | str]:
    return {
        "N": n,
        "boundary": "PBC" if periodic else "OBC",
        "method": method,
        "n_up": n // 2,
        "sz_total": (n // 2) - n / 2,
        "dimension": dimension,
        "E0": energy,
        "e0": energy / n,
        "residual": residual,
        "norm_error": abs(float(np.linalg.norm(vector)) - 1.0),
        "repeat_delta": repeat_delta,
    }


def ground_state(
    n: int, periodic: bool, *, progress: bool = False, repeat: bool = False,
    method: str | None = None,
) -> dict[str, float | int | str]:
    """Compute one ground energy with the requested or production method."""
    if method is None:
        method = production_method(n)
    label = f"eigsh N={n} {'PBC' if periodic else 'OBC'}"
    show_progress = progress and n >= 20
    if method == METHOD_CSR:
        matrix, _ = build_hamiltonian(n, periodic, progress=progress)
        energy, vector, residual = lowest_eigenpair(
            matrix, progress=show_progress, label=label,
        )
        agreement = np.nan
        if repeat:
            second, _, second_residual = lowest_eigenpair(matrix, seed=271828)
            agreement = abs(energy - second)
            residual = max(residual, second_residual)
        row = _result_row(n, periodic, method, matrix.shape[0], energy, vector, residual, agreement)
        row["nnz"] = matrix.nnz
        row["csr_gib"] = (
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        ) / 1024**3
        return row
    if method == METHOD_SZ:
        operator = SzHamiltonian(n, periodic)
    elif method == METHOD_SU2:
        operator = SingletHamiltonian(n, periodic)
    else:
        raise ValueError(f"Unknown method {method}")
    energy, vector, residual = lowest_eigenpair_operator(
        operator.matvec, operator.dimension, operator.basis_bytes,
        progress=show_progress, label=label,
    )
    agreement = np.nan
    if repeat:
        second, _, second_residual = lowest_eigenpair_operator(
            operator.matvec, operator.dimension, operator.basis_bytes, seed=271828,
        )
        agreement = abs(energy - second)
        residual = max(residual, second_residual)
    return _result_row(
        n, periodic, method, operator.dimension, energy, vector, residual, agreement,
    )


def compute_grid(
    sizes: tuple[int, ...] = tuple(range(3, 29)),
    *,
    workers: int = 1,
    progress: bool = True,
    repeat_up_to: int = 28,
) -> list[dict[str, float | int | str]]:
    """Compute both boundaries; independent cases can use separate processes.

    Workers run with one BLAS/Numba thread each (set in compose.yaml), so a
    requested worker count of at most 14 never oversubscribes the CPU budget.
    """
    if not 1 <= workers <= 14:
        raise ValueError("workers must be between 1 and 14")
    cases = [(n, periodic) for n in sizes for periodic in (False, True)]
    if workers == 1:
        iterator = tqdm(cases, desc="Цепочки", mininterval=0.5) if progress else cases
        return [ground_state(n, periodic, progress=progress,
                             repeat=n <= repeat_up_to)
                for n, periodic in iterator]
    results: list[dict[str, float | int | str]] = []
    # Spawn works for a Jupyter kernel and for scripts on Windows and Linux.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(ground_state, n, periodic, progress=False,
                               repeat=n <= repeat_up_to)
                   for n, periodic in cases]
        iterator = tqdm(as_completed(futures), total=len(futures),
                        desc=f"Цепочки ({workers} процесса)",
                        mininterval=0.5) if progress else as_completed(futures)
        for future in iterator:
            results.append(future.result())
    return sorted(results, key=lambda row: (int(row["N"]), str(row["boundary"])))


@njit(cache=True)
def _fill_dense(matrix: np.ndarray, n: int, periodic: bool) -> None:
    """Same local exchange as the CSR builder, on the entire bit basis."""
    bonds = n if periodic else n - 1
    for state in range(1 << n):
        diagonal = 0.0
        for i in range(bonds):
            j = (i + 1) % n
            if ((state >> i) & 1) == ((state >> j) & 1):
                diagonal += 0.25
            else:
                diagonal -= 0.25
                flipped = state ^ (1 << i) ^ (1 << j)
                matrix[flipped, state] += 0.5
        matrix[state, state] += diagonal


def build_dense_hamiltonian(
    n: int, periodic: bool, *, max_matrix_bytes: int = 2 * 1024**3,
) -> np.ndarray:
    """Full float64 matrix, without sector projection or sparse intermediates.

    The allocation limit covers only H. A benchmark must separately reserve
    LAPACK copies, all eigenvectors and workspace before calling this function.
    """
    if not 3 <= n <= 26:
        raise ValueError("Require 3 <= n <= 26")
    dimension = 1 << n
    matrix_bytes = 8 * dimension**2
    if matrix_bytes > max_matrix_bytes:
        raise MemoryError(f"Dense H requires {matrix_bytes} bytes; limit is {max_matrix_bytes}")
    matrix = np.zeros((dimension, dimension), dtype=np.float64, order="F")
    _fill_dense(matrix, n, periodic)
    return matrix


def full_dense_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ALL eigenvalues and eigenvectors with LAPACK divide-and-conquer.

    No spectral subset, Krylov method, or decomposition into spin sectors.
    Keep H intact so that the returned ground-state residual can be checked.
    """
    return eigh(matrix, driver="evd", overwrite_a=False, check_finite=False)

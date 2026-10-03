"""Optional comparison of process-level parallelism inside one container."""

from time import perf_counter

from xxx_chain import compute_grid


def main() -> None:
    sizes = tuple(range(3, 17))
    baseline = None
    measurements = []
    for workers in (1, 2, 4, 8, 14):
        start = perf_counter()
        rows = compute_grid(sizes, workers=workers, progress=True)
        elapsed = perf_counter() - start
        signatures = {(int(row["N"]), str(row["boundary"])): float(row["E0"])
                      for row in rows}
        if baseline is None:
            baseline = signatures
        else:
            assert all(abs(value - baseline[key]) < 1e-9
                       for key, value in signatures.items())
        measurements.append((workers, elapsed))
        print(f"workers={workers:2d}: {elapsed:.3f} s")
    winner = min(measurements, key=lambda pair: pair[1])
    print(f"Fastest on this host: workers={winner[0]} ({winner[1]:.3f} s)")


if __name__ == "__main__":
    main()

# XXX--: exact diagonalization of the spin chain

This project computes the ground-state energy of the antiferromagnetic spin-1/2 XXX chain with J=1 for every N=3,...,26, with open and periodic boundary conditions. The executed [Jupyter notebook](notebooks/XXX_chain_ED.ipynb) contains the derivation, results, tables, and plots. The derivation is documented in [docs/theory.md](docs/theory.md).

The numerical source is [src/xxx_chain.py](src/xxx_chain.py); the independent validation reference is [tests/validate.py](tests/validate.py). Local generated CSV files and plots are written under `data/`, which Git ignores. The executed notebook is tracked in `notebooks/`.

See [docker/README.md](docker/README.md) for building the image, starting Jupyter, running tests, and reproducing calculations. The latest measured comparison between direct dense diagonalization and the optimized sector method is in [reports/last-run-dense-comparison.md](reports/last-run-dense-comparison.md).

[GitHub repository](https://github.com/Egor-Error000/XXX--) · [Agent instructions](AGENTS.md)

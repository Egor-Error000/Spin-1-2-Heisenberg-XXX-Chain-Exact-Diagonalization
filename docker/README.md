# Docker: build and run

Run commands from the repository root in PowerShell. Install and start Docker Desktop first. All Docker-related configuration is grouped in this directory: `Dockerfile`, `compose.yaml`, and `Dockerfile.dockerignore`.

The Compose file is invoked with `-f docker/compose.yaml`; its build context and workspace mount intentionally point to the repository root, so project source, tests, notebooks, and local `data/` are available inside `/workspace`.

## Build and start Jupyter

```powershell
docker compose -f docker/compose.yaml build notebook
docker compose -f docker/compose.yaml up -d notebook
docker compose -f docker/compose.yaml logs -f notebook
```

Open `http://127.0.0.1:8888` in a browser. Jupyter prints the access token in its logs. The port is bound to loopback only. Stop the service with:

```powershell
docker compose -f docker/compose.yaml down
```

## Tests and calculations

Run the independent tests and odd/even sector cross-checks:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python -m unittest discover -s tests -v
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/check_partners.py
```

Compute the complete OBC/PBC grid for N=3,...,26 (PowerShell 7):

```powershell
pwsh -NoProfile -File scripts/run_grid.ps1
```

Windows PowerShell alternative:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_grid.ps1
```

The dispatcher builds the image, benchmarks worker caps unless `-MaxParallel` is specified, then schedules isolated per-case containers. It budgets 70% of Docker Engine memory and reserves 0.5 GiB per worker. Memory and concurrency are detected/chosen at runtime. For example, use `-MaxParallel 4` to set a fixed worker cap and skip the benchmark. Docker's memory total can be inspected with `docker info --format '{{.MemTotal}}'`; host-available RAM and Docker Engine memory are distinct figures.

Run the dense-vs-sector timing experiment:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/compare_dense.py --case-timeout 900
```

Execute all notebook cells in place:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/XXX_chain_ED.ipynb
```

For a single-case resource profile:

```powershell
docker compose -f docker/compose.yaml run --rm -T notebook python scripts/profile_resources.py 22 PBC
```

Generated measurements and plots are stored in `data/`, which is excluded from Git. Jupyter itself can be started in the background with `up -d`; use the `logs -f` command above to retrieve its URL and token.

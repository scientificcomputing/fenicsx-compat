# fenicsx-compat

Cross-version compatibility helpers for [DOLFINx](https://github.com/FEniCS/dolfinx).

Centralizes workarounds for breaking API changes across dolfinx versions
(nightly/main, stable, and back to release 0.10), previously duplicated
across `dolfinx-adjoint`, `io4dolfinx`, and `scifem`.

Pure Python, no compiled extensions. Supports dolfinx >= 0.10.0.

# Installation

`fenicsx-compat` requires Python 3.12 or newer and DOLFINx 0.10 or newer.

## conda-forge

```bash
conda install conda-forge::fenicsx-compat
```

This also installs DOLFINx from conda-forge if it is not already in the environment.

## pip

```bash
python3 -m pip install fenicsx-compat
```

Install DOLFINx first, for instance with conda, spack or one of the [FEniCS Docker images](https://github.com/FEniCS/dolfinx#docker-images). pip cannot build DOLFINx itself.

To install the development version from GitHub, run

```bash
python3 -m pip install git+https://github.com/scientificcomputing/fenicsx-compat.git
```

# Documentation

The [documentation](https://scientificcomputing.github.io/fenicsx-compat) lists every function in the package: what it does, what DOLFINx 0.10, 0.11 and main do natively, and how to call it.

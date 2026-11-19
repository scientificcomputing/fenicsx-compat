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

## From source

To build without build isolation, first install the build requirements from the `build` dependency group (requires pip >= 25.1), then install from a clone of the repository:

```bash
python3 -m pip install --group build
python3 -m pip install --check-build-dependencies --no-build-isolation .
```

# Documentation

The [documentation](https://scientificcomputing.github.io/fenicsx-compat) lists every function in the package: what it does, what DOLFINx 0.10, 0.11 and main do natively, and how to call it.

# Releasing

Versions are managed with [bump-my-version](https://callowayproject.github.io/bump-my-version/), configured in `pyproject.toml` and installed with the `dev` extra. Between releases the version is a development version, `X.Y.Z.devN`; a release drops the suffix. Each bump commits the version change, so start from a clean working tree.

To release the current development version, e.g. `0.6.0.dev0`, and start the next one:

```bash
bump-my-version bump release --no-tag  # 0.6.0.dev0 -> 0.6.0
bump-my-version bump minor --no-tag    # 0.6.0 -> 0.7.0.dev0
```

For a patch release of 0.6, bump `patch` instead of `minor` (`0.6.0 -> 0.6.1.dev0`), then `release` (`0.6.1.dev0 -> 0.6.1`). `bump-my-version show-bump` previews what each bump gives.

The bumps do not tag. A `v*` tag publishes to PyPI, so tag the release commit (`Bump version: 0.6.0.dev0 → 0.6.0`) as `v0.6.0`, for instance by creating a GitHub release from it.

Releasing also updates `.github/docs-versions.yml`, which lists the versions on the documentation website. A new minor version gets its own entry, built in the DOLFINx image of the previous one; edit `dolfinx_image` there afterwards if a newer DOLFINx release should be used. A patch release replaces its minor version's documentation.

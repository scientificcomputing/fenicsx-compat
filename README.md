# fenicsx-compat

Cross-version compatibility helpers for [DOLFINx](https://github.com/FEniCS/dolfinx).

Centralizes workarounds for breaking API changes across dolfinx versions
(nightly/main, stable, and back to release 0.10), previously duplicated
across `dolfinx-adjoint`, `io4dolfinx`, and `scifem`.

Pure Python, no compiled extensions. Supports dolfinx >= 0.10.0.

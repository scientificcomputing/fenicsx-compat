# UFL

DOLFINx 0.10 pins UFL 2025.2, 0.11 pins UFL 2026.1, and main uses UFL 2026.2.

(fn-domain-of)=
## `domain_of`

```python
fenicsx_compat.domain_of(expr) -> ufl.Mesh
```

The UFL domain of an expression. Raises `ValueError` unless there is exactly one.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11, main | `ufl.domain.extract_domains(expr)`, which returns all domains, or `ufl.domain.extract_unique_domain(expr)`, which returns `None` if there is none |

```python
domain = fenicsx_compat.domain_of(u)
```

It does not depend on the version. It is a shorthand that also checks that there is exactly one domain.

(fn-apply-pullback-inverse)=
## `apply_pullback_inverse`

```python
fenicsx_compat.apply_pullback_inverse(pullback, expr, domain) -> ufl.core.expr.Expr
```

Maps an expression on the physical cell to the reference cell, by inverting the element's pull back.

| DOLFINx (UFL) | Native |
|---|---|
| 0.10 (2025.2), 0.11 (2026.1) | None. Closed forms are used here for the identity (`u`), contravariant Piola (`det(J) K u`) and covariant Piola (`J^T u`) pull backs. Other pull backs raise `NotImplementedError`. |
| main (2026.2, UFL PR #511) | {py:meth}`pullback.apply_inverse(expr, domain)<ufl.AbstractPullback.apply_inverse>` |

```python
v = ufl.TestFunction(V)
v_ref = fenicsx_compat.apply_pullback_inverse(V.ufl_element().pullback, v, mesh.ufl_domain())
```

The Piola closed forms assume a vector-valued expression, which is what Piola-mapped elements have.

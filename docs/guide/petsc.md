# Forms and PETSc assembly

The PETSc functions need petsc4py. Without it, {py:func}`set_bc<fenicsx_compat.petsc.set_bc>`, {py:func}`apply_lifting_and_set_bc<fenicsx_compat.petsc.apply_lifting_and_set_bc>`, {py:func}`zero_petsc_vector<fenicsx_compat.petsc.zero_petsc_vector>` and {py:func}`ghost_update<fenicsx_compat.petsc.ghost_update>` raise `RuntimeError`; {py:func}`bcs_by_block<fenicsx_compat.petsc.bcs_by_block>` and {py:func}`form_map<fenicsx_compat.mesh.form_map>` work regardless.

(fn-form-map)=
## `form_map`

```python
fenicsx_compat.form_map(form) -> (IndexMap, int)
```

The index map and block size of the test space of a linear form.

| DOLFINx | Native |
|---|---|
| 0.10 | [`form.function_spaces[0].dofmaps(0).index_map`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.fem.html#dolfinx.fem.FunctionSpace.dofmaps), [`.index_map_bs`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.fem.html#dolfinx.fem.DofMap.index_map_bs); [`dofmaps`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.fem.html#dolfinx.fem.FunctionSpace.dofmaps) is a method |
| 0.11, main | {py:attr}`form.function_spaces[0].dofmaps[0].index_map<dolfinx.fem.FunctionSpace.dofmaps>`, {py:attr}`.index_map_bs<dolfinx.fem.DofMap.index_map_bs>`; {py:attr}`dofmaps<dolfinx.fem.FunctionSpace.dofmaps>` is a sequence |

```python
imap, bs = fenicsx_compat.form_map(L)
```

For a mesh with a single cell type, {py:attr}`form.function_spaces[0].dofmap.index_map<dolfinx.fem.FunctionSpace.dofmap>` and {py:attr}`.index_map_bs<dolfinx.fem.DofMap.index_map_bs>` work on every version, without this function.

(fn-bcs-by-block)=
## `bcs_by_block`

```python
fenicsx_compat.bcs_by_block(spaces, bcs) -> list[list[DirichletBC]]
```

Sorts boundary conditions by the space they constrain: one list per space in `spaces`, and an empty list where the space is `None`.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11, main | {py:func}`dolfinx.fem.bcs_by_block(spaces, bcs)<dolfinx.fem.bcs_by_block>` |

The native call works as long as `spaces` and {py:attr}`bc.function_space<dolfinx.fem.DirichletBC.function_space>` are both C++ objects (0.10, 0.11) or both Python objects (main), which is the case for {py:func}`dolfinx.fem.extract_function_spaces(a, i)<dolfinx.fem.extract_function_spaces>`. Which kind each one is changed in DOLFINx PRs #4342 and #4350, so the native call fails when the two are mixed, for instance with a list of spaces built in Python on 0.11. This function accepts either kind on every version:

```python
bcs1 = fenicsx_compat.bcs_by_block(dolfinx.fem.extract_function_spaces(a, 1), bcs)
bcs0 = fenicsx_compat.bcs_by_block([V, Q], bcs)
```

(fn-set-bc)=
## `set_bc`

```python
fenicsx_compat.set_bc(b, bcs, x0=None, alpha=1.0) -> None
```

Sets the Dirichlet values in a PETSc vector: `b[dofs] = alpha * (g - x0)`.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11, main | {py:func}`dolfinx.fem.petsc.set_bc(b, bcs, x0=x0, alpha=alpha)<dolfinx.fem.petsc.set_bc>`, with `bcs` flat for a plain vector, and one list per block for a blocked or nested vector |

The native call is the same everywhere. This function also accepts a single block given as `[[bc, ...]]` for a plain vector, which the native call rejects.

(fn-apply-lifting-and-set-bc)=
## `apply_lifting_and_set_bc`

```python
fenicsx_compat.apply_lifting_and_set_bc(b, a, bcs, x=None, alpha=1.0) -> None
```

Applies lifting to `b` and sets its Dirichlet values, with the ghost updates in between, for a single, blocked or nested system:

- Free rows: `b - alpha * A (g - x)`
- Constrained rows: `alpha * (g - x)`

`a` is a bilinear form, or a 2D list of them for a blocked or nested system. `bcs` is always a flat list; for a blocked system they are sorted by block with [`bcs_by_block`](fn-bcs-by-block).

| DOLFINx | Native, the same on every version |
|---|---|
| single form | {py:func}`apply_lifting(b, [a], bcs=[bcs], x0=[x], alpha=alpha)<dolfinx.fem.petsc.apply_lifting>`, reverse ghost update, {py:func}`set_bc(b, bcs, x0=x, alpha=alpha)<dolfinx.fem.petsc.set_bc>`, forward ghost update |
| blocked or nested | {py:func}`apply_lifting(b, a, bcs=bcs_by_block(extract_function_spaces(a, 1), bcs), x0=x, alpha=alpha)<dolfinx.fem.petsc.apply_lifting>`, reverse ghost update, {py:func}`set_bc(b, bcs_by_block(extract_function_spaces(a, 0), bcs), x0=x, alpha=alpha)<dolfinx.fem.petsc.set_bc>`, forward ghost update |

```python
b = dolfinx.fem.petsc.assemble_vector(L)
fenicsx_compat.apply_lifting_and_set_bc(b, a, bcs)

# Blocked, or nested with kind="nest"
b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form([L0, L1]), kind="mpi")
fenicsx_compat.apply_lifting_and_set_bc(b, dolfinx.fem.form([[a00, a01], [a10, a11]]), bcs)
```

```{versionchanged} 0.2.0
`x` works with a single form. In 0.1.0 that raises `AttributeError`; blocked and nested systems were not affected.
```

(fn-zero-petsc-vector)=
## `zero_petsc_vector`

```python
fenicsx_compat.zero_petsc_vector(b) -> None
```

Sets a PETSc vector to zero, ghosts included. Nested vectors work too.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11, main | `dolfinx.la.petsc._zero_vector(b)`, which is private |

It does not depend on the version. It exists so that code does not call the private DOLFINx function directly, and a rename only needs fixing here.

(fn-ghost-update)=
## `ghost_update`

```python
fenicsx_compat.ghost_update(x, insert_mode, scatter_mode) -> None
```

Updates the ghost values of a PETSc vector, sub-vector by sub-vector for a nested one.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11, main | `dolfinx.la.petsc._ghost_update(x, insert_mode, scatter_mode)`, which is private |

```python
from petsc4py import PETSc

fenicsx_compat.ghost_update(b, PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
```

Like {py:func}`zero_petsc_vector<fenicsx_compat.petsc.zero_petsc_vector>`, it does not depend on the version.

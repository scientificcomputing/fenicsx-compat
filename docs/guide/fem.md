# Finite elements and interpolation

(fn-real-functionspace)=
## `real_functionspace`

```python
fenicsx_compat.real_functionspace(mesh, value_shape=()) -> dolfinx.fem.FunctionSpace
```

Creates a real function space: one global value per component, the same on the whole mesh.

| DOLFINx | Native |
|---|---|
| 0.10 | None. `basix.ufl.real_element` exists, but a function space cannot be built from it. Raises `NotImplementedError`; `scifem.create_real_functionspace` has a compiled implementation. |
| 0.11, main | `dolfinx.fem.functionspace(mesh, basix.ufl.real_element(mesh.basix_cell(), value_shape, dtype=mesh.geometry.x.dtype))` |

```python
R = fenicsx_compat.real_functionspace(mesh)
R2 = fenicsx_compat.real_functionspace(mesh, (2,))
```

This is one of the few functions that checks the DOLFINx version rather than the API, since on 0.10 nothing about `real_element` shows that it will not work.

(fn-finite-element-ctor-kwargs)=
## `finite_element_ctor_kwargs`

```python
fenicsx_compat.finite_element_ctor_kwargs(constructor, basix_element, *, gdim, value_shape,
                                          symmetric=False)
```

Creates a C++ finite element (`dolfinx.cpp.fem.FiniteElement_float32` or `_float64`) from a C++ basix element.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | `constructor(basix_element, block_shape, symmetric)` |
| main (DOLFINx PR #4511) | `constructor(basix_element, gdim, block_shape, symmetric)`; `gdim` is the geometric dimension of the mesh |

```python
el = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
cpp_element = fenicsx_compat.finite_element_ctor_kwargs(
    dolfinx.cpp.fem.FiniteElement_float64,
    el.basix_element._e,
    gdim=mesh.geometry.dim,
    value_shape=el.reference_value_shape,
)
```

`basix_element` is the C++ basix element, `ufl_element.basix_element._e`. Pass `value_shape=None` for an element without blocks.

(fn-interpolate)=
## `interpolate`

```python
fenicsx_compat.interpolate(cpp_function, values, cells) -> None
```

Interpolates values already evaluated at the interpolation points into a function, on the given cells.

| DOLFINx | Native |
|---|---|
| 0.10 | `u._cpp_object.interpolate(values, cells)` |
| 0.11, main (DOLFINx PR #4049) | `u._cpp_object.interpolate_f(values, cells)` |

```python
cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
fenicsx_compat.interpolate(u._cpp_object, values, cells)
```

`values` is a C-contiguous array of shape `(value_size, num_points)`, with the points ordered cell by cell.

(fn-expression-eval)=
## `expression_eval`

```python
fenicsx_compat.expression_eval(expr, mesh, entities) -> numpy.ndarray
```

Evaluates a `dolfinx.fem.Expression` on cells or facets.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11, main | `expr.eval(mesh, entities)`: `entities` is a 1D array of cells, or an `(N, 2)` array of `(cell, local facet)` pairs for facets |

Not needed from 0.10 on: the 2D facet entities were introduced before 0.10 (DOLFINx PR #3625), and `expr.eval(mesh, entities)` works the same on every supported version. The function only adds a retry with the entities flattened, which is the shape older DOLFINx took. New code can call `expr.eval` directly.

(fn-interpolate-to-submesh-entity-maps)=
## `interpolate_to_submesh_entity_maps`

```python
fenicsx_compat.interpolate_to_submesh_entity_maps(expr, points, entity_maps=None)
    -> dolfinx.fem.Expression
```

Creates a `dolfinx.fem.Expression`, passing `entity_maps` where DOLFINx accepts it. Entity maps let the expression use coefficients that live on another mesh, typically a parent mesh of a submesh. Despite its name, the function interpolates nothing itself.

| DOLFINx | Native |
|---|---|
| 0.10 | `dolfinx.fem.Expression(expr, points)`. Entity maps are not supported. |
| 0.11, main (DOLFINx PR #4140) | `dolfinx.fem.Expression(expr, points, entity_maps=entity_maps)` |

```python
submesh, entity_map = dolfinx.mesh.create_submesh(mesh, tdim, cells)[:2]
e = fenicsx_compat.interpolate_to_submesh_entity_maps(u, V_sub.element.interpolation_points, [entity_map])
```

On 0.10, `entity_maps` is dropped without a warning, so an expression that needs it cannot work there.

(fn-permute-facet-quadrature)=
## `permute_facet_quadrature`

```python
fenicsx_compat.permute_facet_quadrature(facet_type, points) -> list[numpy.ndarray]
```

Returns one copy of facet reference points for each facet permutation, so that an expression evaluated on a facet is evaluated at `points` as seen from the cell, whatever the facet's orientation. Entry `p` is for facets with permutation `p` (see [`facet_permutations`](fn-facet-permutations)): 2 entries for an interval, 6 for a triangle and 8 for a quadrilateral.

| DOLFINx | What a facet `Expression` evaluates at |
|---|---|
| 0.10 | The points in the cell's own orientation of the facet. All entries returned are `points`. |
| 0.11, main (DOLFINx PR #4140, FFCx PR #836) | The points mapped through the facet's global orientation. Each entry is `points` with the inverse of that permutation applied. |

```python
perms = fenicsx_compat.facet_permutations(mesh.topology)
point_sets = fenicsx_compat.permute_facet_quadrature(basix.CellType.interval, points)
expressions = [dolfinx.fem.Expression(f, X) for X in point_sets]
# For a facet (cell, local_facet):
values = expressions[perms[cell, local_facet]].eval(mesh, np.array([[cell, local_facet]], dtype=np.int32))
```

Evaluate the facets with the same permutation together, one call per expression. On 0.11 and main this uses `ffcx.ir.elementtables`, which is not public FFCx API.

(fn-permute-interpolation-data)=
## `permute_interpolation_data`

```python
fenicsx_compat.permute_interpolation_data(data, integration_entities, cell_permutation_info,
                                          basix_element, facet_type) -> None
```

Reorders facet values, in place, from the cell's ordering of the facet's dofs to the facet's own. Needed when values evaluated on parent-mesh facets are interpolated into a space on a facet submesh.

| DOLFINx | Native |
|---|---|
| 0.10 | Permute each row of `data` with `basix_element.permute_subentity_closure_inv(perm, cell_info[cell], facet_type, local_facet)` |
| 0.11, main (DOLFINx PR #4140, FFCx PR #836) | Nothing to do: DOLFINx applies the permutation. This function does nothing. |

```python
fenicsx_compat.permute_interpolation_data(
    data, entities, fenicsx_compat.cell_permutation_info(mesh.topology),
    V.element.basix_element, basix.CellType.interval,
)
```

`data` has one row per entity in `entities`, an `(N, 2)` array of `(cell, local facet)`, and one column per dof on the facet's closure.

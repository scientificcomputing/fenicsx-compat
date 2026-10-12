# Geometry, topology and tags

(fn-cmap)=
## `cmap`

```python
fenicsx_compat.cmap(mesh) -> dolfinx.fem.CoordinateElement
```

The coordinate element of the mesh geometry.

| DOLFINx | Native |
|---|---|
| 0.10 | `mesh.geometry.cmap` |
| 0.11, main | `mesh.geometry.cmaps[0]`. `mesh.geometry.cmap` still works but emits a `DeprecationWarning`. |

```python
cell = 0
nodes = mesh.geometry.x[fenicsx_compat.dofmap(mesh)[cell]]
X = fenicsx_compat.cmap(mesh).pull_back(np.array([[0.2, 0.1, 0.0]]), nodes)
```

(fn-dofmap)=
## `dofmap`

```python
fenicsx_compat.dofmap(mesh) -> numpy.ndarray
```

The geometry dofmap: the node indices of each cell, shape `(num_cells, num_nodes_per_cell)`.

| DOLFINx | Native |
|---|---|
| 0.10 | `mesh.geometry.dofmap` |
| 0.11, main | `mesh.geometry.dofmaps[0]`. `mesh.geometry.dofmap` still works but emits a `DeprecationWarning`. |

```python
cell_nodes = mesh.geometry.x[fenicsx_compat.dofmap(mesh)]  # (num_cells, num_nodes_per_cell, 3)
```

(fn-meshtags-topology)=
## `meshtags_topology`

```{versionadded} 0.2.0
```

```python
fenicsx_compat.meshtags_topology(tags) -> dolfinx.mesh.Topology
```

The topology a `MeshTags` is defined on, always as the Python `dolfinx.mesh.Topology`.

| DOLFINx | `tags.topology` returns | Native Python topology |
|---|---|---|
| 0.10, 0.11 | the C++ topology | `dolfinx.mesh.Topology(tags.topology)` |
| main (DOLFINx PR #4350) | the Python topology | `tags.topology` |

```python
topology = fenicsx_compat.meshtags_topology(facet_tags)
cpp_topology = topology._cpp_object  # the C++ object, on every version
```

(fn-cell-permutation-info)=
## `cell_permutation_info`

```python
fenicsx_compat.cell_permutation_info(topology) -> numpy.ndarray  # uint32
```

Computes and returns the packed per-cell permutation info, ghost cells included, as used by dof transformations.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | `topology.create_entity_permutations()`, then `topology.get_cell_permutation_info()` |
| main (DOLFINx PR #3904) | `topology.create_cell_permutations()`, then `topology.get_cell_permutation_info()`. `create_entity_permutations` now takes a dimension. |

```python
cell_info = fenicsx_compat.cell_permutation_info(mesh.topology)
# e.g. for V.element.Tt_inv_apply(data, cell_info, n)
```

Calling it is also the portable way to make sure the permutations are computed before handing the topology to code that needs them.

(fn-facet-permutations)=
## `facet_permutations`

```python
fenicsx_compat.facet_permutations(topology) -> numpy.ndarray  # uint8
```

The permutation of every facet of every cell, shape `(num_cells, num_facets_per_cell)`, ghost cells included. These are the values FFCx kernels take as `quadrature_permutation`.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | `topology.create_entity_permutations()`, then `topology.get_facet_permutations()`, which is flat |
| main (DOLFINx PR #3904) | `topology.create_entity_permutations(tdim - 1)`, then `topology.get_entity_permutations(tdim - 1)`, which is flat |

```python
perms = fenicsx_compat.facet_permutations(mesh.topology)
perm = perms[cell, local_facet]
```

On main, an interval mesh gives an empty array, since points have no orientation.

(fn-transfer-meshtags-to-submesh)=
## `transfer_meshtags_to_submesh`

```python
fenicsx_compat.transfer_meshtags_to_submesh(entity_tag, submesh, cell_to_parent, vertex_to_parent)
    -> dolfinx.mesh.MeshTags
```

Transfers tags on a parent mesh to a submesh created with `dolfinx.mesh.create_submesh`.

| DOLFINx | Native |
|---|---|
| 0.10 | No equivalent. Raises `NotImplementedError`; `scifem.transfer_meshtags_to_submesh` has a compiled implementation. |
| 0.11 | `dolfinx.mesh.transfer_meshtags_to_submesh(entity_tag, submesh, vertex_to_parent, cell_to_parent)` |
| main | `dolfinx.mesh.transfer_meshtags_to_submesh(entity_tag, submesh, cell_to_parent, vertex_to_parent)` |

The two maps swapped places between 0.11 and main while keeping their names, so a positional call that is right on one version passes the maps the wrong way round on the other. That does not raise. The wrapper forwards by keyword, and takes the maps in the order `create_submesh` returns them:

```python
submesh, cell_map, vertex_map, _ = dolfinx.mesh.create_submesh(mesh, tdim, cells)
sub_tags = fenicsx_compat.transfer_meshtags_to_submesh(cell_tags, submesh, cell_map, vertex_map)
```

If you call DOLFINx directly, pass both maps by keyword: `cell_to_parent=cell_map, vertex_to_parent=vertex_map`.

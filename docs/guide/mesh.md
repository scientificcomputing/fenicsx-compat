# Geometry, topology and tags

(fn-cmap)=
## `cmap`

```python
fenicsx_compat.cmap(mesh) -> dolfinx.fem.CoordinateElement
```

The coordinate element of the mesh geometry.

| DOLFINx | Native |
|---|---|
| 0.10 | [`mesh.geometry.cmap`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Geometry.cmap) |
| 0.11, main | {py:attr}`mesh.geometry.cmaps[0]<dolfinx.mesh.Geometry.cmaps>`. {py:attr}`mesh.geometry.cmap<dolfinx.mesh.Geometry.cmap>` still works but emits a `DeprecationWarning`. |

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
| 0.10 | [`mesh.geometry.dofmap`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Geometry.dofmap) |
| 0.11, main | {py:attr}`mesh.geometry.dofmaps[0]<dolfinx.mesh.Geometry.dofmaps>`. {py:attr}`mesh.geometry.dofmap<dolfinx.mesh.Geometry.dofmap>` still works but emits a `DeprecationWarning`. |

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

The topology a {py:class}`MeshTags<dolfinx.mesh.MeshTags>` is defined on, always as the Python {py:class}`dolfinx.mesh.Topology`.

| DOLFINx | {py:attr}`tags.topology<dolfinx.mesh.MeshTags.topology>` returns | Native Python topology |
|---|---|---|
| 0.10, 0.11 | the C++ topology | [`dolfinx.mesh.Topology(tags.topology)`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Topology) |
| main (DOLFINx PR #4350) | the Python topology | {py:attr}`tags.topology<dolfinx.mesh.MeshTags.topology>` |

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
| 0.10, 0.11 | [`topology.create_entity_permutations()`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Topology.create_entity_permutations), then [`topology.get_cell_permutation_info()`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Topology.get_cell_permutation_info) |
| main (DOLFINx PR #3904) | {py:meth}`topology.create_cell_permutations()<dolfinx.mesh.Topology.create_cell_permutations>`, then {py:meth}`topology.get_cell_permutation_info()<dolfinx.mesh.Topology.get_cell_permutation_info>`. {py:meth}`create_entity_permutations<dolfinx.mesh.Topology.create_entity_permutations>` now takes a dimension. |

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
| 0.10, 0.11 | [`topology.create_entity_permutations()`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Topology.create_entity_permutations), then [`topology.get_facet_permutations()`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.mesh.html#dolfinx.mesh.Topology.get_facet_permutations), which is flat |
| main (DOLFINx PR #3904) | {py:meth}`topology.create_entity_permutations(tdim - 1)<dolfinx.mesh.Topology.create_entity_permutations>`, then {py:meth}`topology.get_entity_permutations(tdim - 1)<dolfinx.mesh.Topology.get_entity_permutations>`, which is flat |

```python
perms = fenicsx_compat.facet_permutations(mesh.topology)
perm = perms[cell, local_facet]
```

On main, an interval mesh gives an empty array, since points have no orientation.

(fn-create-facet-orientations)=
## `create_facet_orientations`

```{versionadded} 0.3.0
```

```python
fenicsx_compat.create_facet_orientations(topology) -> None
```

Computes the facet orientations that Raviart-Thomas and other contravariant Piola spaces need on a manifold. Call it on every rank, before creating the space.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | No equivalent, and none needed: such spaces are created without it. Does nothing. |
| main (DOLFINx PR #4625) | {py:meth}`topology.create_facet_orientations()<dolfinx.mesh.Topology.create_facet_orientations>`. Creating such a space on a manifold raises `RuntimeError` until it has been called. |

```python
submesh = dolfinx.mesh.create_submesh(mesh, 2, boundary_facets)[0]
fenicsx_compat.create_facet_orientations(submesh.topology)
V = dolfinx.fem.functionspace(submesh, ("RT", 1))
```

On main it raises `ValueError` for a topology whose dimension is not 2, as the native method does. Lagrange and Nédélec spaces do not need it.

(fn-submesh-to-parent)=
## `submesh_to_parent`

```{versionadded} 0.4.0
```

```python
fenicsx_compat.submesh_to_parent(entity_map, entities=None) -> numpy.ndarray
```

The parent entity of each of `entities` of a submesh, given one of the maps {py:func}`create_submesh<dolfinx.mesh.create_submesh>` returns (the entity map or the vertex map). Without `entities`, all local (owned and ghost) entities of the submesh are mapped.

| DOLFINx | {py:func}`create_submesh<dolfinx.mesh.create_submesh>` returns the maps as | Native |
|---|---|---|
| 0.10 | arrays, the parent entity of each submesh entity | `entity_map[entities]` |
| 0.11, main (DOLFINx PR #4140) | {py:class}`dolfinx.mesh.EntityMap` | {py:meth}`entity_map.sub_topology_to_topology(entities, inverse=False)<dolfinx.mesh.EntityMap.sub_topology_to_topology>` |

```python
submesh, cell_map, vertex_map, _ = dolfinx.mesh.create_submesh(mesh, tdim, cells)
parent_cells = fenicsx_compat.submesh_to_parent(cell_map)
parent_vertices = fenicsx_compat.submesh_to_parent(vertex_map)
```

(fn-transfer-meshtags-to-submesh)=
## `transfer_meshtags_to_submesh`

```python
fenicsx_compat.transfer_meshtags_to_submesh(entity_tag, submesh, cell_to_parent, vertex_to_parent)
    -> dolfinx.mesh.MeshTags
```

Transfers tags on a parent mesh to a submesh created with {py:func}`dolfinx.mesh.create_submesh`.

| DOLFINx | Native |
|---|---|
| 0.10 | No equivalent. Raises `NotImplementedError`; {py:func}`scifem.transfer_meshtags_to_submesh` has a compiled implementation. |
| 0.11 | [`dolfinx.mesh.transfer_meshtags_to_submesh(entity_tag, submesh, vertex_to_parent, cell_to_parent)`](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.mesh.html#dolfinx.mesh.transfer_meshtags_to_submesh) |
| main | {py:func}`dolfinx.mesh.transfer_meshtags_to_submesh(entity_tag, submesh, cell_to_parent, vertex_to_parent)<dolfinx.mesh.transfer_meshtags_to_submesh>` |

The two maps swapped places between 0.11 and main while keeping their names, so a positional call that is right on one version passes the maps the wrong way round on the other. That does not raise. The wrapper forwards by keyword, and takes the maps in the order {py:func}`create_submesh<dolfinx.mesh.create_submesh>` returns them:

```python
submesh, cell_map, vertex_map, _ = dolfinx.mesh.create_submesh(mesh, tdim, cells)
sub_tags = fenicsx_compat.transfer_meshtags_to_submesh(cell_tags, submesh, cell_map, vertex_map)
```

If you call DOLFINx directly, pass both maps by keyword: `cell_to_parent=cell_map, vertex_to_parent=vertex_map`.

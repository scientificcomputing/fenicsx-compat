# Creating meshes

Since DOLFINx PR #4403 (after 0.11), the ghost mode is an argument of `create_mesh` and `model_to_mesh`. In 0.10 and 0.11 it has to be built into the cell partitioner instead. The functions on this page take `ghost_mode` on every version and put it where the installed DOLFINx expects it.

```{important}
`fenicsx_compat.create_mesh` and `fenicsx_compat.model_to_mesh` default to `GhostMode.shared_facet`, while DOLFINx defaults to `GhostMode.none` on every version. Pass `ghost_mode` explicitly if you rely on either.
```

(fn-create-mesh)=
## `create_mesh`

```python
fenicsx_compat.create_mesh(comm, cells, e, x, *, partitioner=None,
                           ghost_mode=GhostMode.shared_facet,
                           max_facet_to_cell_links=2, num_threads=1)
```

Creates a distributed mesh from cell and node arrays with the given ghost mode.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | `dolfinx.mesh.create_mesh(comm, cells, e, x, partitioner=dolfinx.mesh.create_cell_partitioner(ghost_mode, max_facet_to_cell_links))` |
| main | `dolfinx.mesh.create_mesh(comm, cells, e, x, ghost_mode=ghost_mode, max_facet_to_cell_links=..., num_threads=...)` |

```python
from mpi4py import MPI
import basix.ufl
import dolfinx
import numpy as np
import ufl
import fenicsx_compat

comm = MPI.COMM_WORLD
e = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
# The input is each rank's share: here all of it on rank 0, none on the others
if comm.rank == 0:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
else:
    x = np.empty((0, 2))
    cells = np.empty((0, 3), dtype=np.int64)
mesh = fenicsx_compat.create_mesh(comm, cells, e, x, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
```

- `num_threads` only has an effect on main.
- An explicit `partitioner` is always used as given. On 0.10 and 0.11 that means `ghost_mode` only applies if the partitioner was built with it, see [`create_cell_partitioner`](fn-create-cell-partitioner).
- Passing the full arrays on every rank builds a mesh `comm.size` times too large.

(fn-create-cell-partitioner)=
## `create_cell_partitioner`

```python
fenicsx_compat.create_cell_partitioner(ghost_mode, max_facet_to_cell_links=2)
```

Builds a cell partitioner with the ghost mode built in, for 0.10 and 0.11. Raises `NotImplementedError` on main, where the partitioner no longer carries the ghost mode.

| DOLFINx | Native |
|---|---|
| 0.10 | `dolfinx.mesh.create_cell_partitioner(ghost_mode, max_facet_to_cell_links=2)` |
| 0.11 | `dolfinx.mesh.create_cell_partitioner(ghost_mode, max_facet_to_cell_links)`, with no default for the second argument |
| main | Removed. Pass `ghost_mode` to `create_mesh` or `model_to_mesh`. |

Prefer [`create_mesh`](fn-create-mesh) or [`model_to_mesh`](fn-model-to-mesh) with `ghost_mode`. Use this only for an API that accepts a partitioner but no ghost mode:

```python
try:  # DOLFINx 0.10 and 0.11
    kwargs = {"partitioner": fenicsx_compat.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)}
except NotImplementedError:  # main
    kwargs = {"ghost_mode": dolfinx.mesh.GhostMode.shared_facet}
```

The raise on main is deliberate: returning a plain partitioner there would silently give an unghosted mesh.

(fn-model-to-mesh)=
## `model_to_mesh`

```{versionadded} 0.2.0
```

```python
fenicsx_compat.model_to_mesh(model, comm, rank, gdim=3, *, partitioner=None,
                             ghost_mode=GhostMode.shared_facet,
                             max_facet_to_cell_links=2, dtype=default_real_type)
```

Creates a mesh from a gmsh model with the given ghost mode, and returns the `dolfinx.io.gmsh.MeshData` that DOLFINx builds, so the mesh is `.mesh`.

| DOLFINx | Native |
|---|---|
| 0.10 | `dolfinx.io.gmsh.model_to_mesh(model, comm, rank, gdim, partitioner=create_cell_partitioner(ghost_mode, max_facet_to_cell_links))`; no `max_facet_to_cell_links` argument |
| 0.11 | as 0.10, and `max_facet_to_cell_links` is also an argument of `model_to_mesh` |
| main | `dolfinx.io.gmsh.model_to_mesh(model, comm, rank, gdim, ghost_mode=..., max_facet_to_cell_links=...)` |

```python
import gmsh

gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.mesh.generate(2)
mesh_data = fenicsx_compat.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)
gmsh.finalize()
mesh = mesh_data.mesh
```

As for `create_mesh`, an explicit `partitioner` is used as given, and on 0.10 and 0.11 the ghost mode then only applies if it was built into it. On 0.10, `max_facet_to_cell_links` only reaches the mesh through the partitioner built here, so it has no effect in serial.

(fn-reconstruct-mesh)=
## `reconstruct_mesh`

```python
fenicsx_compat.reconstruct_mesh(mesh, coordinate_element_degree)
```

Returns a copy of `mesh` whose geometry uses a coordinate element of another degree. The topology is shared with `mesh`.

| DOLFINx | Native |
|---|---|
| 0.10 | No equivalent. The geometry is rebuilt by hand here. |
| 0.11, main | `dolfinx.fem.interpolate_geometry(mesh, dolfinx.fem.coordinate_element(mesh.topology.cell_type, degree, variant=mesh.geometry.cmaps[0].variant, dtype=mesh.geometry.x.dtype))` |

```python
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
quadratic = fenicsx_compat.reconstruct_mesh(mesh, 2)
assert fenicsx_compat.cmap(quadratic).degree == 2
```

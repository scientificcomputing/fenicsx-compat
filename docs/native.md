# Already the same on every version

From DOLFINx 0.10 on, these work the same way on 0.10, 0.11 and main, so fenicsx-compat has no wrapper for them. Call DOLFINx directly. Older code often still carries version branches for them, which can be deleted.

| To | Call this on every version | Notes |
|---|---|---|
| Get the interpolation points of an element | `V.element.interpolation_points` | A property on the Python `FiniteElement`. Only the C++ object has the method `V.element._cpp_object.interpolation_points()`. |
| Build a C++ function space | `dolfinx.cpp.fem.FunctionSpace_float64(mesh._cpp_object, element._cpp_object, dofmap._cpp_object)` | No `value_shape` argument. `_float32` likewise. |
| Count the dofs in an entity's closure | `len(layout.entity_closure_dofs(dim, i))` | `ElementDofLayout.num_entity_closure_dofs` and `num_entity_dofs` only exist in 0.10. `entity_dofs` and `entity_closure_dofs` exist everywhere. |
| Pack constants and coefficients | `dolfinx.fem.pack_constants(form)`, `dolfinx.fem.pack_coefficients(form)` | Public in `dolfinx.fem` everywhere. No need for `dolfinx.cpp.fem.pack_*(form._cpp_object)`. |
| Read gmsh models | `dolfinx.io.gmsh` | `dolfinx.io.gmshio` no longer exists. `model_to_mesh` returns a `MeshData` named tuple, so the mesh is `.mesh`. For the ghost mode, see [`model_to_mesh`](fn-model-to-mesh). |
| Find which process owns points | `dolfinx.geometry.determine_point_ownership(mesh, points, padding=tol)` | Returns an object with `src_owner`, `dest_owner`, `dest_points` and `dest_cells`. `padding` has no default. |
| Compute integration entities | `dolfinx.fem.compute_integration_domains(integral_type, mesh.topology, entities)` | Three arguments, no dimension. Takes the Python topology. Returns a flat array. |
| Build a C++ topology | `dolfinx.cpp.mesh.Topology(cell_type, vertex_map, cell_map, cells, original_index)` | The only constructor; it takes no communicator. The index maps and `cells` must be C++ objects, see [`unwrap_index_map`](fn-unwrap-index-map). |
| Create and assemble blocked or nested PETSc objects | `dolfinx.fem.petsc.create_vector(spaces, kind=...)`, `create_matrix(a, kind=...)`, `assemble_vector(L, kind=...)`, `assemble_matrix(a, bcs=...)` | The `*_block` and `*_nest` functions are gone. `kind="nest"` gives a nest object; for one blocked object, `kind="mpi"` is used by scifem on 0.11 and main. |

## Differences with no wrapper

These do differ between versions, and fenicsx-compat does not cover them.

| What | 0.10, 0.11 | main | Portable |
|---|---|---|---|
| Set Dirichlet values in an array | `dolfinx.fem.set_bc(b, bcs, x0, alpha)`, deprecated in 0.10 | removed | `for bc in bcs: bc.set(b, x0, alpha)` with `b` a NumPy array. For a PETSc vector, see [`set_bc`](fn-set-bc). |
| Newton solver problem class | `dolfinx.fem.petsc.NewtonSolverNonlinearProblem` | removed | `dolfinx.fem.petsc.NonlinearProblem`, which solves with PETSc SNES and has a different interface |
| Point ownership search | `determine_point_ownership(mesh, points, padding, cells=None)` | adds `find_closest_cell=True` | Do not pass `find_closest_cell` |

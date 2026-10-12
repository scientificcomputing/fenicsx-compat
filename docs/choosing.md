# Which function do I need?

Find what your code does in the first column. The last column says which DOLFINx versions behave differently, which is when the function matters. "None" means the native call is the same from 0.10 on, and the function is only a convenience. For things that need no function at all, see {doc}`native`.

## Meshes

| If your code needs to | Use | Differs between |
|---|---|---|
| Create a mesh from arrays with a ghost mode | [`create_mesh`](fn-create-mesh) | 0.10, 0.11 / main |
| Create a mesh from a gmsh model with a ghost mode | [`model_to_mesh`](fn-model-to-mesh) | 0.10, 0.11 / main |
| Pass a ghosted partitioner to an API that takes no ghost mode | [`create_cell_partitioner`](fn-create-cell-partitioner) | 0.10, 0.11 only; raises on main |
| Change the degree of the coordinate element | [`reconstruct_mesh`](fn-reconstruct-mesh) | 0.10 / 0.11, main |
| Get the coordinate element | [`cmap`](fn-cmap) | 0.10 / 0.11, main |
| Get the nodes of each cell | [`dofmap`](fn-dofmap) | 0.10 / 0.11, main |
| Get the topology of a `MeshTags` as the Python object | [`meshtags_topology`](fn-meshtags-topology) | 0.10, 0.11 / main |
| Get the cell permutation info | [`cell_permutation_info`](fn-cell-permutation-info) | 0.10, 0.11 / main |
| Get the orientation of each facet as seen from each cell | [`facet_permutations`](fn-facet-permutations) | 0.10, 0.11 / main |
| Move tags from a mesh to a submesh | [`transfer_meshtags_to_submesh`](fn-transfer-meshtags-to-submesh) | 0.10 / 0.11 / main |

## Index maps and vectors

| If your code needs to | Use | Differs between |
|---|---|---|
| Create an index map | [`create_index_map`](fn-create-index-map) | 0.10, 0.11 / main |
| Pass an index map to a `dolfinx.cpp` function | [`unwrap_index_map`](fn-unwrap-index-map) | 0.10, 0.11 / main |
| Create an index map for a subset of indices | [`create_sub_index_map`](fn-create-sub-index-map) | 0.10, 0.11 / main |
| Find the other ranks that hold each index | [`index_to_dest_ranks`](fn-index-to-dest-ranks) | 0.10, 0.11 / main |
| Split `range(N)` over ranks the way DOLFINx does | [`local_range`](fn-local-range) | 0.10, 0.11 / main |
| Create a `dolfinx.la.Vector` | [`vector`](fn-vector) | None, if `dtype` is a keyword |

## Function spaces, elements and interpolation

| If your code needs to | Use | Differs between |
|---|---|---|
| Create a real (global constant) function space | [`real_functionspace`](fn-real-functionspace) | 0.10 / 0.11, main |
| Create a C++ finite element | [`finite_element_ctor_kwargs`](fn-finite-element-ctor-kwargs) | 0.10, 0.11 / main |
| Interpolate values already evaluated at the interpolation points | [`interpolate`](fn-interpolate) | 0.10 / 0.11, main |
| Evaluate an expression on cells or facets | [`expression_eval`](fn-expression-eval) | None |
| Create an expression that uses coefficients from a parent mesh | [`interpolate_to_submesh_entity_maps`](fn-interpolate-to-submesh-entity-maps) | 0.10 / 0.11, main |
| Evaluate on facets at points seen from the cell | [`permute_facet_quadrature`](fn-permute-facet-quadrature) | 0.10 / 0.11, main |
| Reorder facet values to the facet's own orientation | [`permute_interpolation_data`](fn-permute-interpolation-data) | 0.10 / 0.11, main |

## Forms and PETSc

| If your code needs to | Use | Differs between |
|---|---|---|
| Get the index map and block size of a linear form | [`form_map`](fn-form-map) | 0.10 / 0.11, main |
| Sort boundary conditions by block | [`bcs_by_block`](fn-bcs-by-block) | 0.10, 0.11 / main, when spaces are mixed |
| Set Dirichlet values in a PETSc vector | [`set_bc`](fn-set-bc) | None |
| Apply lifting and set Dirichlet values in one call | [`apply_lifting_and_set_bc`](fn-apply-lifting-and-set-bc) | None |
| Zero a PETSc vector, or update its ghosts | [`zero_petsc_vector`](fn-zero-petsc-vector), [`ghost_update`](fn-ghost-update) | None |

## UFL

| If your code needs to | Use | Differs between |
|---|---|---|
| Get the domain of an expression | [`domain_of`](fn-domain-of) | None |
| Map an expression back to the reference cell | [`apply_pullback_inverse`](fn-apply-pullback-inverse) | 0.10, 0.11 / main |

## Output and other packages

| If your code needs to | Use | Differs between |
|---|---|---|
| Get ADIOS2's low-level bindings | [`resolve_adios_scope`](fn-resolve-adios-scope) | ADIOS2 2.9 / 2.10 and newer |
| Let PyVista use VTK's snake_case names | [`pyvista_allow_snake_case`](fn-pyvista-allow-snake-case) | PyVista 0.45, 0.46 / 0.47 and newer |
| Branch on the DOLFINx version | [`at_least`](fn-at-least), [`before`](fn-before) | None |
| Import an optional module with a helpful error | [`require_module`](fn-require-module) | None |

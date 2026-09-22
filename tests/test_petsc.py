from petsc4py import PETSc

import dolfinx.fem
import dolfinx.fem.petsc  # `import dolfinx.fem` alone does not expose `.petsc`
import dolfinx.mesh
import numpy as np
import ufl

from fenicsx_compat.petsc import (
    apply_lifting_and_set_bc,
    bcs_by_block,
    ghost_update,
    pack_coefficients,
    pack_constants,
    set_bc,
    zero_petsc_vector,
)


def test_pack_constants_and_coefficients_on_a_simple_form(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    c = dolfinx.fem.Constant(msh, 2.0)
    L = dolfinx.fem.form(c * v * ufl.dx)
    constants = pack_constants(L)
    coeffs = pack_coefficients(L)
    assert constants.shape[0] == 1
    assert constants[0] == 2.0
    assert isinstance(coeffs, dict)


def test_bcs_by_block_assigns_bc_to_matching_space(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V0 = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    V1 = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs0 = dolfinx.fem.locate_dofs_topological(V0, tdim - 1, facets)
    bc0 = dolfinx.fem.dirichletbc(0.0, dofs0, V0)

    result = bcs_by_block([V0, V1], [bc0])
    assert result[0] == [bc0]
    assert result[1] == []


def test_bcs_by_block_returns_empty_list_for_none_space(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V0 = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs0 = dolfinx.fem.locate_dofs_topological(V0, tdim - 1, facets)
    bc0 = dolfinx.fem.dirichletbc(0.0, dofs0, V0)

    result = bcs_by_block([None, V0], [bc0])
    assert result[0] == []
    assert result[1] == [bc0]


def test_zero_petsc_vector_zeroes_a_vector(comm):
    # NOTE: a plain `PETSc.Vec().createMPI((-1, 10), comm=comm)` (no ghost
    # padding) is NOT used here even though the brief specifies it verbatim:
    # `VecGhostGetLocalForm()` returns NULL for such a vector (documented
    # PETSc behaviour), and petsc4py's `Vec.localForm()` does not guard
    # against that NULL before wrapping it, so `zero_petsc_vector` (which
    # calls `dolfinx.la.petsc._zero_vector` -> `x.localForm()`) segfaults the
    # whole process (confirmed: SIGSEGV / MPI_ABORT, reproduces identically
    # under `mpirun -n 2`). `createGhost([], ...)` is a genuinely ghost-aware
    # vector (matching how dolfinx always constructs its own PETSc vectors,
    # even with zero ghosts in serial) and exercises the same code path
    # safely. See task-22-23-report.md for the full repro.
    vec = PETSc.Vec().createGhost([], (-1, 10), comm=comm)
    vec.array[:] = 5.0
    zero_petsc_vector(vec)
    assert np.allclose(vec.array, 0.0)
    vec.destroy()


def test_ghost_update_runs_without_error(comm):
    # See the note in test_zero_petsc_vector_zeroes_a_vector: a plain
    # `createMPI` vector is not ghosted, so `ghost_update` on it raises
    # `petsc4py.PETSc.Error: ... Vector is not ghosted` (confirmed) rather
    # than running "without error". `createGhost([], ...)` fixes this the
    # same way.
    vec = PETSc.Vec().createGhost([], (-1, 10), comm=comm)
    vec.array[:] = 3.0
    ghost_update(vec, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
    assert np.allclose(vec.array, 3.0)
    vec.destroy()


def test_ghost_update_accumulates_ghost_contributions(comm):
    # A genuinely ghosted vector: dofs shared with another rank each hold 1.0
    # locally, so a reverse ADD_VALUES update accumulates them on the owner.
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    vec = u.x.petsc_vec
    with vec.localForm() as loc:
        loc.set(1.0)

    ghost_update(vec, PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)

    owned = vec.array_r
    assert owned.min() == 1.0
    if comm.size > 1:
        # Verified under `mpirun -n 2`: shared dofs accumulate to 2.0.
        assert owned.max() > 1.0


def test_set_bc_applies_dirichlet_value(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets)
    bc = dolfinx.fem.dirichletbc(5.0, dofs, V)

    b = dolfinx.fem.petsc.create_vector(V)
    b.array[:] = 0.0
    set_bc(b, [bc])

    # `dofs` is local (owned + ghost); only owned entries land in b.array.
    owned_dofs = dofs[dofs < b.getLocalSize()]
    assert np.allclose(b.array[owned_dofs], 5.0)
    b.destroy()


def test_apply_lifting_and_set_bc_runs_on_a_simple_poisson_system(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 3, 3)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = dolfinx.fem.form(v * ufl.dx)

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets)
    # Ruling F5: a zero bc value can't distinguish "bc applied" from "vector
    # already zero there", so this uses a non-zero value the assertion below
    # can actually detect.
    bc = dolfinx.fem.dirichletbc(7.0, dofs, V)

    # NOTE: the brief specifies `b = dolfinx.fem.petsc.assemble_vector(L)`
    # (single form, no `kind`), which produces a plain vector with no
    # `_blocks` attribute. Under Ruling F6's `bcs_by_block` (no except),
    # `a=[[a]]` must be genuinely 2D for `extract_function_spaces`/
    # `bcs_by_block` to accept an explicit index (it raises `ValueError:
    # index must be None for a 1D array of forms` for a flat `a=[a]`,
    # confirmed) -- but `dolfinx.fem.petsc.apply_lifting` requires a FLAT
    # `a` when `b` has no `_blocks` attribute (confirmed:
    # `AttributeError: 'list' object has no attribute '_cpp_object'` when
    # a 2D `a` reaches a plain `b`). No shape of `a` satisfies both
    # `dolfinx.fem.petsc` functions simultaneously for a genuinely plain
    # `b`. `assemble_vector([L], kind="mpi")` builds `b` as a real
    # (single-block) blocked vector, which is what `a=[[a]]` (2D) actually
    # requires, and is the exact shape scifem's own tests use for the
    # non-blocked/"kind=None" analogue (see
    # third-party/scifem/tests/test_assembly.py:179-183). Verified working
    # end-to-end (owned bc dofs read back as 7.0) under 1/2/3 ranks; see
    # task-22-23-report.md for the full repro and reasoning.
    b = dolfinx.fem.petsc.assemble_vector([L], kind="mpi")
    apply_lifting_and_set_bc(b, [[a]], [bc])

    # `dofs` is local (owned + ghost); only owned entries land in b.array.
    owned_dofs = dofs[dofs < b.getLocalSize()]
    assert np.allclose(b.array[owned_dofs], 7.0)
    b.destroy()

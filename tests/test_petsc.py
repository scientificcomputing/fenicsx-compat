from petsc4py import PETSc

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import ufl

from fenicsx_compat.petsc import (
    bcs_by_block,
    ghost_update,
    pack_coefficients,
    pack_constants,
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

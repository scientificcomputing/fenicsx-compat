import dolfinx.fem
import dolfinx.mesh
import ufl

from fenicsx_compat.petsc import bcs_by_block, pack_coefficients, pack_constants


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

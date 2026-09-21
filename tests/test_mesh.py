import dolfinx.fem
import dolfinx.mesh
import ufl

from fenicsx_compat.mesh import cmap, dofmap, form_map, num_entity_closure_dofs


def test_cmap_returns_a_coordinate_element(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    result = cmap(msh)
    assert result is not None


def test_dofmap_returns_an_array_of_node_indices(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    result = dofmap(msh)
    assert result.ndim == 2
    assert result.shape[0] == msh.topology.index_map(msh.topology.dim).size_local + \
        msh.topology.index_map(msh.topology.dim).num_ghosts


def test_form_map_returns_index_map_and_block_size(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    L = dolfinx.fem.form(v * ufl.dx)
    index_map, bs = form_map(L)
    assert index_map.size_local == V.dofmap.index_map.size_local
    assert bs == V.dofmap.index_map_bs


def test_num_entity_closure_dofs_matches_entity_closure_dofs_length(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    dof_layout = cmap(msh).create_dof_layout()
    tdim = msh.topology.dim
    result = num_entity_closure_dofs(dof_layout, tdim)
    assert result == len(dof_layout.entity_closure_dofs(tdim, 0))

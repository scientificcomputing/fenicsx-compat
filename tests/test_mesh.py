import dolfinx.mesh

from fenicsx_compat.mesh import cmap, dofmap


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

import basix.ufl
import dolfinx
import dolfinx.fem
import dolfinx.graph
import dolfinx.mesh
import numpy as np
import pytest
import ufl

from fenicsx_compat.mesh import (
    cmap,
    create_cell_partitioner,
    create_mesh,
    dofmap,
    form_map,
    num_entity_closure_dofs,
)


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


def test_create_mesh_default_ghost_mode_none_has_no_ghosts(comm):
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64)
    msh = create_mesh(comm, cells, el, x, ghost_mode=dolfinx.mesh.GhostMode.none)
    assert msh.topology.index_map(2).num_ghosts == 0


def test_create_mesh_shared_facet_ghost_mode_has_ghosts_under_two_ranks(comm):
    if comm.size < 2:
        pytest.skip("requires at least 2 ranks")
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64)
    msh = create_mesh(comm, cells, el, x, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    assert msh.topology.index_map(2).num_ghosts > 0


def test_create_mesh_respects_explicit_partitioner(comm):
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64)
    calls = []
    base = dolfinx.graph.partitioner()

    def spy_partitioner(*args, **kwargs):
        calls.append(1)
        return base(*args, **kwargs)

    create_mesh(comm, cells, el, x, partitioner=spy_partitioner)
    if comm.size > 1:
        assert calls, "explicit partitioner must be used, not silently replaced"


def test_create_cell_partitioner_matches_the_installed_dolfinx_api():
    # Which branch runs is decided by the installed dolfinx, not by a mock:
    # the v0.10.0 and stable CI legs build a real partitioner here, the
    # nightly leg (where dolfinx removed create_cell_partitioner) raises.
    if hasattr(dolfinx.mesh, "create_cell_partitioner"):
        partitioner = create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
        assert partitioner is not None
    else:
        with pytest.raises(NotImplementedError, match="removed"):
            create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)

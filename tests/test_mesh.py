from mpi4py import MPI

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
    reconstruct_mesh,
    transfer_meshtags_to_submesh,
)


def test_cmap_returns_a_coordinate_element(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    result = cmap(msh)
    assert result is not None


def test_dofmap_returns_an_array_of_node_indices(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    result = dofmap(msh)
    assert result.ndim == 2
    assert (
        result.shape[0]
        == msh.topology.index_map(msh.topology.dim).size_local
        + msh.topology.index_map(msh.topology.dim).num_ghosts
    )


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


def _two_square_input(comm):
    """Rank-0-only mesh input.

    `create_mesh` treats each rank's arrays as that rank's share of the
    input, so passing the full arrays on every rank concatenates them and
    builds a doubled mesh (8 cells instead of 4 at 2 ranks). dolfinx
    nightly tolerates that silently; v0.10.0 aborts in C++. Supplying the
    description on rank 0 and empty arrays elsewhere is the standard
    idiom and yields the intended 4-cell mesh at any rank count.
    """
    if comm.rank == 0:
        x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=np.float64)
        cells = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64)
    else:
        x = np.empty((0, 2), dtype=np.float64)
        cells = np.empty((0, 3), dtype=np.int64)
    return cells, x


def test_create_mesh_default_ghost_mode_none_has_no_ghosts(comm):
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    cells, x = _two_square_input(comm)
    msh = create_mesh(comm, cells, el, x, ghost_mode=dolfinx.mesh.GhostMode.none)
    assert msh.topology.index_map(2).size_global == 4
    assert msh.topology.index_map(2).num_ghosts == 0


def test_create_mesh_shared_facet_ghost_mode_has_ghosts_under_two_ranks(comm):
    if comm.size < 2:
        pytest.skip("requires at least 2 ranks")
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    cells, x = _two_square_input(comm)
    msh = create_mesh(comm, cells, el, x, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    assert msh.topology.index_map(2).size_global == 4
    assert msh.topology.index_map(2).num_ghosts > 0


def test_create_mesh_respects_explicit_partitioner(comm):
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    cells, x = _two_square_input(comm)
    calls = []
    # Build the base partitioner the version-appropriate way: dolfinx 0.10/0.11
    # expect create_mesh's `partitioner` to be a *cell* partitioner (what
    # create_cell_partitioner builds), not the raw graph partitioner nightly
    # accepts directly - passing the wrong shape raises TypeError inside
    # dolfinx's C++ layer when it invokes the spy.
    if hasattr(dolfinx.mesh, "create_cell_partitioner"):
        base = create_cell_partitioner(dolfinx.mesh.GhostMode.none)
    else:
        base = dolfinx.graph.partitioner()

    def spy_partitioner(*args, **kwargs):
        calls.append(1)
        return base(*args, **kwargs)

    msh = create_mesh(comm, cells, el, x, partitioner=spy_partitioner)
    assert msh.topology.index_map(2).size_global == 4
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


def test_reconstruct_mesh_changes_coordinate_element_degree(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    new_msh = reconstruct_mesh(msh, coordinate_element_degree=2)
    assert new_msh.geometry.x.shape[0] > msh.geometry.x.shape[0]
    assert cmap(new_msh).degree == 2


def test_transfer_meshtags_to_submesh_matches_the_installed_dolfinx(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facet_indices = dolfinx.mesh.locate_entities(msh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    values = np.full(len(facet_indices), 7, dtype=np.int32)
    entity_tag = dolfinx.mesh.meshtags(msh, tdim - 1, facet_indices, values)

    num_cells = msh.topology.index_map(tdim).size_local
    submesh, cell_map, vertex_map, _ = dolfinx.mesh.create_submesh(
        msh, tdim, np.arange(num_cells, dtype=np.int32)
    )

    # dolfinx 0.11 made this native; on the v0.10.0 CI leg the compat
    # function has no pure-Python fallback and says so. Which branch runs
    # is decided by the installed dolfinx, not by a mock.
    if hasattr(dolfinx.mesh, "transfer_meshtags_to_submesh"):
        result = transfer_meshtags_to_submesh(entity_tag, submesh, cell_map, vertex_map)
        assert isinstance(result, dolfinx.mesh.MeshTags)
        # A merely-truthy isinstance check would still pass if the cell/vertex
        # maps were swapped internally (wrong entities tagged, or none at
        # all) -- that is exactly the failure this wrapper's keyword
        # forwarding was written to prevent. Check the transferred tags are
        # meaningful: correct dimension, and the tag value 7 actually made
        # it onto the submesh somewhere across all ranks collectively (a
        # single rank's local submesh may legitimately receive none of the
        # tagged facets, so this must be reduced across ranks -- a per-rank
        # skip/assert before a collective call would deadlock instead).
        assert result.dim == tdim - 1
        # The x==0 boundary of a 4x4 unit-square mesh always has exactly 4
        # facets, regardless of how many ranks the mesh is distributed
        # across -- verified empirically at 1, 2 and 3 ranks (LAND across a
        # per-rank "has tag 7" check is NOT safe here: at 3 ranks one rank's
        # local submesh legitimately holds none of the tagged facets, so
        # LAND would false-fail; SUM of local occurrences is the invariant
        # that actually holds at every rank count tried).
        local_tag_7_count = int(np.sum(result.values == 7))
        total_tag_7_count = comm.allreduce(local_tag_7_count, op=MPI.SUM)
        assert total_tag_7_count == 4
    else:
        with pytest.raises(NotImplementedError, match="0.11"):
            transfer_meshtags_to_submesh(entity_tag, submesh, cell_map, vertex_map)

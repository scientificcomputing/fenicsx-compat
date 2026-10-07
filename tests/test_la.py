import dolfinx.common
import dolfinx.la
import dolfinx.mesh
import numpy as np
import pytest

from fenicsx_compat.la import create_index_map, index_to_dest_ranks, unwrap_index_map, vector


def test_create_index_map_with_no_ghosts(comm):
    imap = create_index_map(
        comm, num_local=10, ghosts=np.array([], dtype=np.int64), owners=np.array([], dtype=np.int32)
    )
    assert imap.size_local == 10
    assert imap.num_ghosts == 0


def test_unwrap_index_map_returns_cpp_object(comm):
    imap = create_index_map(
        comm, num_local=10, ghosts=np.array([], dtype=np.int64), owners=np.array([], dtype=np.int32)
    )
    cpp_imap = unwrap_index_map(imap)
    assert isinstance(cpp_imap, dolfinx.cpp.common.IndexMap)
    # idempotent: unwrapping an already-cpp object returns it unchanged
    assert unwrap_index_map(cpp_imap) is cpp_imap


def test_vector_creates_distributed_vector_with_given_dtype(comm):
    imap = create_index_map(
        comm, num_local=10, ghosts=np.array([], dtype=np.int64), owners=np.array([], dtype=np.int32)
    )
    vec = vector(imap, bs=1, dtype=np.float64)
    assert isinstance(vec, dolfinx.la.Vector)
    assert vec.array.dtype == np.float64
    assert vec.array.shape[0] == 10


def test_vector_honours_non_default_dtype(comm):
    imap = create_index_map(
        comm, num_local=10, ghosts=np.array([], dtype=np.int64), owners=np.array([], dtype=np.int32)
    )
    vec = vector(imap, bs=1, dtype=np.float32)
    assert vec.array.dtype == np.float32


@pytest.mark.parametrize("unwrap", [False, True])
def test_index_to_dest_ranks_matches_gathered_ghosts(comm, unwrap):
    msh = dolfinx.mesh.create_unit_square(comm, 5, 5)
    imap = msh.topology.index_map(msh.topology.dim)
    ranks, offsets = index_to_dest_ranks(unwrap_index_map(imap) if unwrap else imap, tag=1202)

    # Independent oracle: every rank's ghosts, by global index.
    ghosts_by_rank = comm.allgather(set(imap.ghosts.tolist()))
    size_local = imap.size_local
    globals_ = imap.local_to_global(np.arange(size_local + imap.num_ghosts, dtype=np.int32))
    assert len(offsets) == size_local + imap.num_ghosts + 1
    for i, g in enumerate(globals_):
        ghosted_by = {r for r, ghosts in enumerate(ghosts_by_rank) if g in ghosts}
        if i >= size_local:
            ghosted_by.add(int(imap.owners[i - size_local]))
        ghosted_by.discard(comm.rank)
        assert set(ranks[offsets[i] : offsets[i + 1]].tolist()) == ghosted_by, f"index {i}"

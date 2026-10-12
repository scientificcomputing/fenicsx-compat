import dolfinx.common
import dolfinx.la
import dolfinx.mesh
import numpy as np
import pytest

from fenicsx_compat.la import (
    _local_range,
    create_index_map,
    create_sub_index_map,
    index_to_dest_ranks,
    local_range,
    unwrap_index_map,
    vector,
)


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


def _ring_index_map(comm):
    """Each rank owns one index (global index = rank) and ghosts the next rank's.

    Local index 0 is owned, local index 1 (with two or more ranks) is the ghost.
    """
    if comm.size == 1:
        ghosts = np.array([], dtype=np.int64)
        owners = np.array([], dtype=np.int32)
    else:
        neighbour = (comm.rank + 1) % comm.size
        ghosts = np.array([neighbour], dtype=np.int64)
        owners = np.array([neighbour], dtype=np.int32)
    return create_index_map(comm, 1, ghosts=ghosts, owners=owners, tag=1203)


@pytest.mark.parametrize("unwrap", [False, True])
def test_create_sub_index_map_keeps_owner_and_ghost_when_both_included(comm, unwrap):
    imap = _ring_index_map(comm)
    indices = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    sub_map, sub_to_parent = create_sub_index_map(
        unwrap_index_map(imap) if unwrap else imap, indices
    )
    assert sub_map.size_global == comm.size
    assert sub_map.size_local == 1
    assert sub_map.num_ghosts == imap.num_ghosts
    np.testing.assert_array_equal(sub_to_parent, indices)


def test_create_sub_index_map_drops_a_ghost_its_holder_leaves_out(comm):
    imap = _ring_index_map(comm)
    sub_map, sub_to_parent = create_sub_index_map(imap, np.array([0], dtype=np.int32))
    assert sub_map.size_global == comm.size
    assert sub_map.size_local == 1
    assert sub_map.num_ghosts == 0
    np.testing.assert_array_equal(sub_to_parent, [0])


def test_create_sub_index_map_refuses_an_owner_change_by_default(comm):
    if comm.size < 2:
        pytest.skip("an owner change needs a ghost, which needs at least 2 ranks")
    # Every rank keeps only its ghost, so every rank's owned index is wanted only by the
    # rank that ghosts it. Every rank then raises, on every dolfinx: a rank that did not
    # would carry on into the next collective alone.
    imap = _ring_index_map(comm)
    with pytest.raises(RuntimeError, match="owner change"):
        create_sub_index_map(imap, np.array([1], dtype=np.int32))


def test_create_sub_index_map_moves_the_owner_when_allowed(comm):
    if comm.size < 2:
        pytest.skip("an owner change needs a ghost, which needs at least 2 ranks")
    imap = _ring_index_map(comm)
    sub_map, sub_to_parent = create_sub_index_map(
        imap, np.array([1], dtype=np.int32), allow_owner_change=True
    )
    # The former ghost is now owned here, and it is the next rank's index that moved.
    assert sub_map.size_global == comm.size
    assert sub_map.size_local == 1
    assert sub_map.num_ghosts == 0
    np.testing.assert_array_equal(sub_to_parent, [1])
    np.testing.assert_array_equal(
        imap.local_to_global(sub_to_parent), [(comm.rank + 1) % comm.size]
    )


def _expected_local_ranges(N, size):
    """Independent oracle: the contiguous blocks `np.array_split` cuts `range(N)` into."""
    blocks = np.array_split(np.arange(N), size)
    starts = np.cumsum([0] + [len(b) for b in blocks])
    return [(int(starts[i]), int(starts[i + 1])) for i in range(size)]


# `_local_range` is the fallback for dolfinx 0.10 and 0.11; it is checked directly as well,
# so the formula is covered on every leg and not only where dolfinx lacks its own.
@pytest.mark.parametrize("function", [local_range, _local_range])
@pytest.mark.parametrize("N", [0, 1, 7, 10, 13])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_local_range_splits_into_almost_equal_contiguous_blocks(function, N, size):
    ranges = [function(index, N, size) for index in range(size)]
    assert ranges == _expected_local_ranges(N, size)
    assert all(type(v) is int for r in ranges for v in r)

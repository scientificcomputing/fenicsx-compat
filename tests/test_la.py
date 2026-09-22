import dolfinx.common
import dolfinx.la
import numpy as np

from fenicsx_compat.la import create_index_map, unwrap_index_map, vector


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

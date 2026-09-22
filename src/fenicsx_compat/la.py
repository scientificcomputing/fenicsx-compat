import dolfinx.common
import dolfinx.cpp.common
import numpy as np
import numpy.typing as npt


def create_index_map(
    comm,
    num_local: int,
    ghosts: npt.NDArray[np.int64],
    owners: npt.NDArray[np.int32],
    tag: int = 0,
) -> dolfinx.common.IndexMap:
    """Build an IndexMap, across dolfinx 0.11's class-constructor-to-factory-function rename.

    dolfinx 0.11 replaced the `dolfinx.common.IndexMap` class constructor
    with a `dolfinx.common.index_map` factory taking ghosts as a single
    `(ghosts, owners)` tuple kwarg. The `tag` argument (used internally
    for MPI communication) is caller-specific — pass a value unique
    across index maps that may communicate concurrently.
    """
    if hasattr(dolfinx.common, "index_map"):
        return dolfinx.common.index_map(comm, num_local, ghosts=(ghosts, owners), tag=tag)
    return dolfinx.common.IndexMap(comm, num_local, ghosts, owners, tag=tag)


def unwrap_index_map(index_map) -> dolfinx.cpp.common.IndexMap:
    """Return the underlying cpp IndexMap, whether given the Python wrapper or the cpp
    object itself.
    """
    if isinstance(index_map, dolfinx.cpp.common.IndexMap):
        return index_map
    return index_map._cpp_object

import dolfinx.common
import dolfinx.cpp.common
import dolfinx.la
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
    # dolfinx.common.IndexMap's constructor signature is the pre-0.11 shape
    # (positional ghosts/owners, no tag kwarg on some installs); typeshed here
    # reflects the newer factory-function shape, so this branch is only
    # reached -- and only type-checked against the wrong signature -- on
    # dolfinx versions where it is in fact the right call.
    return dolfinx.common.IndexMap(comm, num_local, ghosts, owners, tag=tag)  # type: ignore[call-arg]


def unwrap_index_map(index_map) -> dolfinx.cpp.common.IndexMap:
    """Return the underlying cpp IndexMap, whether given the Python wrapper or the cpp
    object itself.
    """
    if isinstance(index_map, dolfinx.cpp.common.IndexMap):
        return index_map
    return index_map._cpp_object


def vector(index_map, bs: int, dtype: npt.DTypeLike = np.float64) -> dolfinx.la.Vector:
    """Create a distributed vector compatible with an index map and block size.

    `dtype` must stay keyword — it became keyword-only from dolfinx 0.12.
    Whether `index_map` should be the raw cpp object (0.11) or the Python
    wrapper (0.12, dolfinx#4496) is handled internally by
    `dolfinx.la.vector` itself. Never pass `scatterer` — dolfinx 0.11 has
    no such parameter.
    """
    return dolfinx.la.vector(index_map, bs, dtype=dtype)

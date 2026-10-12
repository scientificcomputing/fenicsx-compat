from mpi4py import MPI

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


def index_to_dest_ranks(index_map, tag: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """The ranks sharing each local index of `index_map`, as `(ranks, offsets)`.

    Ranks sharing local index `i` are `ranks[offsets[i]:offsets[i + 1]]`:
    for an owned index, the ranks that ghost it; for a ghost, its owner
    and the other ranks that ghost it. The calling rank is excluded.
    Collective.

    dolfinx 0.10 and 0.11 take an MPI `tag` for the consensus exchange
    and return an `AdjacencyList`. dolfinx PR #4573 (post-0.11) dropped
    the tag and returns a `(ranks, offsets)` tuple. `tag` is only used on
    the older versions; pass a value no other exchange in flight uses.
    Accepts either the Python wrapper or the cpp `IndexMap`. Dispatches on
    the `TypeError` raised while binding the arguments, which happens on
    every rank before any communication, so it cannot deadlock.
    """
    try:
        dest = index_map.index_to_dest_ranks(tag)
    except TypeError:
        return index_map.index_to_dest_ranks()
    return dest.array, dest.offsets


def vector(index_map, bs: int, dtype: npt.DTypeLike = np.float64) -> dolfinx.la.Vector:
    """Create a distributed vector compatible with an index map and block size.

    `dtype` must stay keyword — it became keyword-only from dolfinx 0.12.
    Whether `index_map` should be the raw cpp object (0.11) or the Python
    wrapper (0.12, dolfinx#4496) is handled internally by
    `dolfinx.la.vector` itself. Never pass `scatterer` — dolfinx 0.11 has
    no such parameter.
    """
    return dolfinx.la.vector(index_map, bs, dtype=dtype)


def create_sub_index_map(
    index_map, indices: npt.NDArray[np.int32], *, allow_owner_change: bool = False
) -> tuple[dolfinx.common.IndexMap, npt.NDArray[np.int32]]:
    """Build an index map for a subset of the local indices of `index_map`.

    Returns `(sub_map, sub_to_parent)`: the sub-map, and the local index in
    `index_map` of each of its local indices. An index included by a rank
    that ghosts it but not by its owner has to change owner; unless
    `allow_owner_change`, that raises `RuntimeError`. Collective.

    dolfinx 0.10 and 0.11 have this only as
    `dolfinx.cpp.common.create_sub_index_map(map, indices,
    allow_owner_change)`, which throws on the owning rank alone. dolfinx
    PR #4479 (post-0.11) dropped the flag in favour of always allowing the
    change and returning a rank-local flag saying whether one happened, and
    #4496 added the Python `dolfinx.common.create_sub_index_map`. There the
    flag is reduced here before raising, so every rank raises together.
    Accepts either the Python wrapper or the cpp `IndexMap`; `sub_map` is
    whichever type the installed dolfinx returns. Dispatches on `hasattr`.
    """
    if not hasattr(dolfinx.common, "create_sub_index_map"):
        # The stubs describe the post-#4479 binding, which takes no flag and returns a
        # third value; this branch only runs on the 0.10/0.11 binding.
        return dolfinx.cpp.common.create_sub_index_map(  # type: ignore[call-arg, return-value]
            unwrap_index_map(index_map), indices, allow_owner_change
        )
    if isinstance(index_map, dolfinx.cpp.common.IndexMap):
        index_map = dolfinx.common.IndexMap(index_map)
    sub_map, sub_to_parent, owners_changed = dolfinx.common.create_sub_index_map(index_map, indices)
    if not allow_owner_change and index_map.comm.allreduce(owners_changed, op=MPI.LOR):
        raise RuntimeError(
            "Index owner change detected: an index is included by a rank that ghosts it but "
            "not by its owner. Pass allow_owner_change=True to let it change owner."
        )
    return sub_map, sub_to_parent


def local_range(index: int, N: int, size: int) -> tuple[int, int]:
    """The half-open range of `[0, N)` held by block `index` of `size` almost-equal blocks.

    The first `N % size` blocks hold one index more than the rest, the
    split dolfinx itself uses to distribute indices over ranks. dolfinx
    0.10 and 0.11 do not expose the free function
    `dolfinx.common.local_range` to Python (only the `IndexMap` property of
    the same name); it is reproduced here where it is missing. Dispatches
    on `hasattr`.
    """
    if hasattr(dolfinx.common, "local_range"):
        low, high = dolfinx.common.local_range(index, N, size)
        return int(low), int(high)
    return _local_range(index, N, size)


def _local_range(index: int, N: int, size: int) -> tuple[int, int]:
    """`dolfinx::common::local_range`, for dolfinx versions that do not expose it."""
    per_block, remainder = divmod(N, size)
    low = index * per_block + min(index, remainder)
    return low, low + per_block + (1 if index < remainder else 0)

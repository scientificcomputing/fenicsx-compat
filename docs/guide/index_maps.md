# Index maps and vectors

DOLFINx PR #4496 (after 0.11) wrapped {py:class}`IndexMap<dolfinx.cpp.common.IndexMap>` in a Python class. On 0.10 and 0.11, {py:class}`dolfinx.common.IndexMap` is the C++ class itself; on main it is a Python wrapper around it. Most functions on this page exist because of that change.

(fn-create-index-map)=
## `create_index_map`

```python
fenicsx_compat.create_index_map(comm, num_local, ghosts, owners, tag=0) -> IndexMap
```

Creates an index map with `num_local` owned indices, and ghosts given by their global indices and owning ranks.

| DOLFINx | Native | Returns |
|---|---|---|
| 0.10, 0.11 | [`dolfinx.common.IndexMap(comm, num_local, ghosts, owners, tag)`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.common.html#dolfinx.common.IndexMap) | the C++ {py:class}`IndexMap<dolfinx.cpp.common.IndexMap>` |
| main | {py:func}`dolfinx.common.index_map(comm, num_local, (ghosts, owners), tag=tag)<dolfinx.common.index_map>` | the Python {py:class}`IndexMap<dolfinx.common.IndexMap>` |

```python
comm = MPI.COMM_WORLD
neighbour = (comm.rank + 1) % comm.size
# Each rank owns 3 indices and ghosts the first index of the next rank
ghosts = np.array([3 * neighbour] if comm.size > 1 else [], dtype=np.int64)
owners = np.array([neighbour] if comm.size > 1 else [], dtype=np.int32)
imap = fenicsx_compat.create_index_map(comm, 3, ghosts, owners, tag=101)
```

`tag` is the MPI tag of the exchange that builds the map. Give maps created at the same time different tags.

(fn-unwrap-index-map)=
## `unwrap_index_map`

```python
fenicsx_compat.unwrap_index_map(index_map) -> dolfinx.cpp.common.IndexMap
```

The C++ index map, given either the Python or the C++ one.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | Nothing to do: every index map is the C++ one. |
| main | `index_map._cpp_object` |

Use it wherever an index map goes into a `dolfinx.cpp` function or constructor:

```python
dofmap = dolfinx.cpp.fem.DofMap(layout, fenicsx_compat.unwrap_index_map(imap), bs, cells, bs)
```

(fn-create-sub-index-map)=
## `create_sub_index_map`

```{versionadded} 0.2.0
```

```python
fenicsx_compat.create_sub_index_map(index_map, indices, *, allow_owner_change=False)
    -> (sub_map, sub_to_parent)
```

Creates an index map for a subset of the local indices of `index_map`. `sub_to_parent` holds, for each local index of the sub-map, its local index in `index_map`. An index that is kept by a rank that ghosts it but not by its owner has to change owner. Unless `allow_owner_change` is set, that is an error.

| DOLFINx | Native | Owner change |
|---|---|---|
| 0.10, 0.11 | `dolfinx.cpp.common.create_sub_index_map(cpp_map, indices, allow_owner_change)` returns `(sub_map, sub_to_parent)` | raises `RuntimeError`, only on the owning ranks, if not allowed |
| main (DOLFINx PRs #4479, #4496) | {py:func}`dolfinx.common.create_sub_index_map(index_map, indices)<dolfinx.common.create_sub_index_map>` returns `(sub_map, sub_to_parent, owners_changed)` | always allowed. `owners_changed` is local to each rank. |

```python
vertex_map = mesh.topology.index_map(0)
keep = np.arange(vertex_map.size_local + vertex_map.num_ghosts, dtype=np.int32)[::2]
sub_map, sub_to_parent = fenicsx_compat.create_sub_index_map(vertex_map, keep)
```

- `indices` must be sorted and unique: 0.10 and 0.11 require it.
- `sub_map` is the C++ map on 0.10 and 0.11 and the Python one on main, like {py:func}`create_index_map<fenicsx_compat.la.create_index_map>`.
- On main the owner-change error is raised on every rank together. On 0.10 and 0.11 it comes from DOLFINx, on the owning ranks only, and the other ranks carry on into their next collective call and hang. If an owner change is possible, choose `indices` so that it cannot happen, or pass `allow_owner_change=True`.

(fn-index-to-dest-ranks)=
## `index_to_dest_ranks`

```python
fenicsx_compat.index_to_dest_ranks(index_map, tag) -> (ranks, offsets)
```

For each local index, the other ranks that hold it: for an owned index, the ranks that ghost it; for a ghost, its owner and the other ranks that ghost it. The ranks of local index `i` are `ranks[offsets[i]:offsets[i + 1]]`. Collective.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | [`index_map.index_to_dest_ranks(tag)`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.common.html#dolfinx.common.IndexMap.index_to_dest_ranks) returns an [`AdjacencyList`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.cpp.graph.html#dolfinx.cpp.graph.AdjacencyList_int32); use [`.array`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.cpp.graph.html#dolfinx.cpp.graph.AdjacencyList_int32.array) and [`.offsets`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.cpp.graph.html#dolfinx.cpp.graph.AdjacencyList_int32.offsets) |
| main (DOLFINx PRs #4354, #4573) | {py:meth}`index_map.index_to_dest_ranks()<dolfinx.common.IndexMap.index_to_dest_ranks>` returns `(ranks, offsets)` |

```python
ranks, offsets = fenicsx_compat.index_to_dest_ranks(mesh.topology.index_map(0), tag=102)
```

`tag` is only used on 0.10 and 0.11. Pass one that no other exchange in progress uses. Both the Python and the C++ index map are accepted.

(fn-local-range)=
## `local_range`

```{versionadded} 0.2.0
```

```python
fenicsx_compat.local_range(index, N, size) -> (int, int)
```

The half-open range of `[0, N)` held by block `index` when `[0, N)` is split into `size` blocks of almost equal size, as DOLFINx splits indices over ranks. The first `N % size` blocks hold one index more than the others.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | Not available in Python. ([`IndexMap.local_range`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.common.html#dolfinx.common.IndexMap.local_range) is a different thing: the owned range of that map.) |
| main | `dolfinx.common.local_range(index, N, size)` |

```python
start, stop = fenicsx_compat.local_range(comm.rank, N, comm.size)
```

(fn-vector)=
## `vector`

```python
fenicsx_compat.vector(index_map, bs, dtype=np.float64) -> dolfinx.la.Vector
```

Creates a distributed vector over an index map with block size `bs`.

| DOLFINx | Native |
|---|---|
| 0.10, 0.11 | [`dolfinx.la.vector(index_map, bs, dtype)`](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.la.html#dolfinx.la.vector) |
| main (DOLFINx PRs #4489, #4496) | {py:func}`dolfinx.la.vector(index_map, bs, scatterer=None, *, dtype=...)<dolfinx.la.vector>`. `dtype` must be a keyword, and `index_map` must be the Python one. |

```python
x = fenicsx_compat.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.float64)
```

Pass the index map as DOLFINx or {py:func}`create_index_map<fenicsx_compat.la.create_index_map>` returns it; on main the C++ one raises `AttributeError`. A native call with `dtype=` as a keyword works on every version too.

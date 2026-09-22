import inspect
from collections.abc import Callable

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem
import dolfinx.graph
import numpy as np
import numpy.typing as npt
import ufl


def cmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.CoordinateElement:
    """Get the coordinate element for a mesh's geometry.

    Ported from scifem/io4dolfinx's `compat.cmap`, across three
    generations of `mesh.geometry`'s cmap API: plural `cmaps[]`, callable
    `cmap()`, and attribute `cmap`.
    """
    if hasattr(mesh.geometry, "cmaps"):
        return mesh.geometry.cmaps[0]
    if callable(mesh.geometry.cmap):
        return mesh.geometry.cmap()
    return mesh.geometry.cmap


def dofmap(mesh: dolfinx.mesh.Mesh) -> npt.NDArray[np.int32]:
    """Get the geometry dofmap for a mesh, across the same generations as `cmap`."""
    if hasattr(mesh.geometry, "dofmaps"):
        return mesh.geometry.dofmaps[0]
    if callable(mesh.geometry.dofmap):
        return mesh.geometry.dofmap()
    return mesh.geometry.dofmap


def form_map(form: dolfinx.fem.Form) -> tuple[dolfinx.common.IndexMap, int]:
    """Get the index map and block size for a linear form's test space.

    `FunctionSpace.dofmaps` was a callable method in older dolfinx and is a
    subscriptable sequence in newer dolfinx.
    """
    try:
        return (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
    except TypeError:
        return (
            form.function_spaces[0].dofmaps[0].index_map,
            form.function_spaces[0].dofmaps[0].index_map_bs,
        )


def num_entity_closure_dofs(dof_layout: dolfinx.cpp.fem.ElementDofLayout, dim: int) -> int:
    """Get the number of dofs in the closure of a `dim`-dimensional entity."""
    if hasattr(dof_layout, "num_entity_closure_dofs"):
        return dof_layout.num_entity_closure_dofs(dim)
    return len(dof_layout.entity_closure_dofs(dim, 0))


def create_cell_partitioner(
    ghost_mode: dolfinx.mesh.GhostMode,
    max_facet_to_cell_links: int = 2,
) -> Callable:
    """Build a cell partitioner with ghost mode baked in, across the
    0.10/0.11 `create_cell_partitioner` API.

    Not needed (and not called) on dolfinx generations where
    `create_cell_partitioner` was removed (post dolfinx PR #4403,
    unreleased as of this package's design) — `create_mesh` takes
    `ghost_mode` directly on those. Raises `NotImplementedError` if
    called anyway, so a caller doesn't silently get an un-ghosted mesh.
    """
    if not hasattr(dolfinx.mesh, "create_cell_partitioner"):
        raise NotImplementedError(
            "dolfinx.mesh.create_cell_partitioner was removed in this dolfinx version "
            "(see dolfinx PR #4403); pass ghost_mode directly to "
            "fenicsx_compat.mesh.create_mesh() instead of calling create_cell_partitioner()."
        )
    sig = inspect.signature(dolfinx.mesh.create_cell_partitioner)
    if "max_facet_to_cell_links" in sig.parameters:
        return dolfinx.mesh.create_cell_partitioner(ghost_mode, max_facet_to_cell_links)
    return dolfinx.mesh.create_cell_partitioner(ghost_mode)


def create_mesh(
    comm: MPI.Comm,
    cells: npt.NDArray[np.int64],
    e,
    x: npt.NDArray[np.floating],
    *,
    partitioner: Callable | None = None,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.shared_facet,
    max_facet_to_cell_links: int = 2,
    num_threads: int = 1,
) -> dolfinx.mesh.Mesh:
    """Create a mesh from topology and geometry arrays, across three `create_mesh` generations.

    See fenicsx-compat design spec §9 item 3 for the full generation
    history. If `partitioner` is given explicitly, it is always used
    as-is (never silently replaced) — `ghost_mode` is still threaded
    through to it on dolfinx generations that support that.
    """
    sig = inspect.signature(dolfinx.mesh.create_mesh)
    kwargs: dict = {}
    if "max_facet_to_cell_links" in sig.parameters:
        kwargs["max_facet_to_cell_links"] = max_facet_to_cell_links

    if "ghost_mode" in sig.parameters:
        # Post-#4403: ghost_mode is create_mesh's own kwarg, threaded through
        # to whichever partitioner is used, including the default.
        kwargs["ghost_mode"] = ghost_mode
        if "num_threads" in sig.parameters:
            kwargs["num_threads"] = num_threads
        if partitioner is None and comm.size > 1:
            partitioner = dolfinx.graph.partitioner()
    else:
        # 0.10 / 0.11: ghost mode must be baked into the partitioner itself.
        if partitioner is None and comm.size > 1:
            partitioner = create_cell_partitioner(ghost_mode, max_facet_to_cell_links)

    return dolfinx.mesh.create_mesh(comm, cells, e, x, partitioner=partitioner, **kwargs)


def reconstruct_mesh(
    mesh: dolfinx.mesh.Mesh, coordinate_element_degree: int
) -> dolfinx.mesh.Mesh:
    """Copy a mesh, changing its coordinate element degree.

    The topology is shared with the original mesh; the geometry is
    reconstructed. Delegates to the native `dolfinx.fem.interpolate_geometry`
    from dolfinx>=0.11; for 0.10, reconstructs the geometry manually (no
    native equivalent exists at that version).
    """
    if hasattr(dolfinx.fem, "interpolate_geometry"):
        new_cmap = dolfinx.fem.coordinate_element(
            mesh.topology.cell_type,
            coordinate_element_degree,
            dtype=mesh.geometry.x.dtype,
            variant=cmap(mesh).variant,
        )
        return dolfinx.fem.interpolate_geometry(mesh, new_cmap)

    ud = mesh.ufl_domain()
    assert ud is not None
    c_el = ud.ufl_coordinate_element()
    family = c_el.family_name
    lvar = c_el.lagrange_variant
    ct = c_el.cell_type

    new_c_el = basix.ufl.element(
        family,
        ct,
        coordinate_element_degree,
        shape=(mesh.geometry.dim,),
        lagrange_variant=lvar,
        dtype=mesh.geometry.x.dtype,
    )
    V_tmp = dolfinx.fem.functionspace(mesh, new_c_el)
    gdim = mesh.geometry.dim
    x = V_tmp.tabulate_dof_coordinates()[:, :gdim]

    geom_imap = V_tmp.dofmap.index_map
    geom_dofmap = V_tmp.dofmap.list
    num_nodes_local = geom_imap.size_local + geom_imap.num_ghosts
    original_input_indices = geom_imap.local_to_global(
        np.arange(num_nodes_local, dtype=np.int32)
    )
    coordinate_element = dolfinx.fem.coordinate_element(
        mesh.topology.cell_type, coordinate_element_degree, lvar, dtype=mesh.geometry.x.dtype
    )
    geom = dolfinx.mesh.Geometry(
        type(mesh.geometry._cpp_object)(
            geom_imap,
            geom_dofmap,
            coordinate_element._cpp_object,
            x,
            original_input_indices,
        )
    )
    new_top = mesh.topology
    cpp_mesh = type(mesh._cpp_object)(mesh.comm, new_top._cpp_object, geom._cpp_object)
    return dolfinx.mesh.Mesh(cpp_mesh, ufl.Mesh(new_c_el))

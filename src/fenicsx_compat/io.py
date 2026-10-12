import inspect

import dolfinx
import dolfinx.mesh

from .mesh import create_cell_partitioner


def resolve_adios_scope(adios2):
    """Resolve the ADIOS2 Python module scope, across the `adios2.bindings` submodule split."""
    scope = adios2.bindings if hasattr(adios2, "bindings") else adios2
    if not scope.is_built_with_mpi:
        raise ImportError("ADIOS2 must be built with MPI support")
    return scope


def pyvista_allow_snake_case(pyvista) -> None:
    """Allow snake_case VTK attribute access.

    Across pyvista's `_VTK_SNAKE_CASE_STATE` rename (pyvista 0.47+).
    """
    if hasattr(pyvista, "_VTK_SNAKE_CASE_STATE"):
        pyvista._VTK_SNAKE_CASE_STATE = "allow"
    else:
        pyvista.core.vtk_snake_case._state = "allow"


def model_to_mesh(
    model,
    comm,
    rank: int,
    gdim: int = 3,
    *,
    partitioner=None,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.shared_facet,
    max_facet_to_cell_links: int = 2,
    dtype=dolfinx.default_real_type,
):
    """Create a mesh from a gmsh model, with a given ghost mode.

    Returns the `dolfinx.io.gmsh.MeshData` that dolfinx builds. Across the
    same dolfinx PR #4403 split as `create_mesh`: from there,
    `model_to_mesh` takes `ghost_mode` and threads it into whichever
    partitioner is used. On 0.10 and 0.11 the ghost mode can only reach the
    mesh through a partitioner built with it, so one is built here when
    `partitioner` is not given. dolfinx 0.10 also has no
    `max_facet_to_cell_links` parameter on `model_to_mesh`, so there it
    only takes effect through that partitioner. Dispatches on
    `inspect.signature`.

    The default `ghost_mode` is `shared_facet`, as for `create_mesh`, while
    dolfinx's own default is `none` on every version.

    If `partitioner` is given explicitly it is always used as-is. On 0.10
    and 0.11, `ghost_mode` (and on 0.10 `max_facet_to_cell_links`) then
    only apply if that partitioner was built with them.
    """
    import dolfinx.io.gmsh

    params = inspect.signature(dolfinx.io.gmsh.model_to_mesh).parameters
    kwargs: dict = {}
    if "ghost_mode" in params:
        kwargs["ghost_mode"] = ghost_mode
    elif partitioner is None and comm.size > 1:
        partitioner = create_cell_partitioner(ghost_mode, max_facet_to_cell_links)
    if "max_facet_to_cell_links" in params:
        kwargs["max_facet_to_cell_links"] = max_facet_to_cell_links
    return dolfinx.io.gmsh.model_to_mesh(
        model, comm, rank, gdim=gdim, partitioner=partitioner, dtype=dtype, **kwargs
    )

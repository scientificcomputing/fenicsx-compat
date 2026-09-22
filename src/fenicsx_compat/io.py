def resolve_adios_scope(adios2):
    """Resolve the ADIOS2 Python module scope, across the `adios2.bindings` submodule split."""
    scope = adios2.bindings if hasattr(adios2, "bindings") else adios2
    if not scope.is_built_with_mpi:
        raise ImportError("ADIOS2 must be built with MPI support")
    return scope


def import_gmshio():
    """Import dolfinx's gmsh-mesh-reading module.

    Across its `dolfinx.io.gmshio` -> `dolfinx.io.gmsh` rename.
    """
    try:
        from dolfinx.io import gmsh as gmshio
    except ImportError:
        from dolfinx.io import gmshio
    return gmshio


def pyvista_allow_snake_case(pyvista) -> None:
    """Allow snake_case VTK attribute access.

    Across pyvista's `_VTK_SNAKE_CASE_STATE` rename (pyvista 0.47+).
    """
    if hasattr(pyvista, "_VTK_SNAKE_CASE_STATE"):
        pyvista._VTK_SNAKE_CASE_STATE = "allow"
    else:
        pyvista.core.vtk_snake_case._state = "allow"
